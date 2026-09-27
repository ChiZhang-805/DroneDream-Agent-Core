from __future__ import annotations

import base64
import io
import json
from email.message import Message
from http.client import IncompleteRead
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from clock_fixtures import isolate_monotonic
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from dronedream_agent_app import identity as module


# 功能：
#   将离线签名数据编码为不带补位符的 JOSE 字符串。
# 输入：
#   value：要编码的字节。
# 输出：
#   encoded：URL 安全的 Base64 文本。
def _b64(value: bytes) -> str:
    encoded = base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
    return encoded


# 功能：
#   构造带真实离线签名的测试身份，不读取用户凭证或访问 Supabase。
# 输入：
#   header_patch：覆盖测试令牌头的字段。
#   raw_claims：替代正常声明的原始 JSON 字节。
# 输出：
#   fixture：验证器、已签名令牌和缓存公钥。
def _fixture(header_patch=None, raw_claims=None):
    key = ed25519.Ed25519PrivateKey.generate()
    verifier = module.SupabaseJwtVerifier("https://example.invalid/auth/v1")
    jwk = {
        "kid": "offline-key",
        "alg": "EdDSA",
        "kty": "OKP",
        "crv": "Ed25519",
        "x": _b64(key.public_key().public_bytes_raw()),
    }
    verifier._keys = {jwk["kid"]: jwk}
    verifier._loaded_at = module.time.monotonic()
    now = int(module.time.time())
    claims = {
        "iss": verifier.issuer,
        "sub": "account-001",
        "aud": "authenticated",
        "role": "authenticated",
        "session_id": "session-0001",
        "iat": now - 10,
        "exp": now + 600,
    }
    header = {"alg": "EdDSA", "kid": jwk["kid"], **(header_patch or {})}
    encoded_header = _b64(json.dumps(header).encode())
    encoded_claims = _b64(raw_claims if raw_claims is not None else json.dumps(claims).encode())
    signed = f"{encoded_header}.{encoded_claims}".encode("ascii")
    token = f"{signed.decode('ascii')}.{_b64(key.sign(signed))}"
    fixture = (verifier, token, jwk)
    return fixture


# 功能：
#   在每个边界测试中封闭默认联网路径；需要 JWKS 的用例只能使用显式内存响应。
# 输入：
#   monkeypatch：测试作用域替换工具。
# 输出：
#   None：不返回业务数据。
@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    blocked = Mock(side_effect=AssertionError("Unexpected real identity network request"))
    monkeypatch.setattr(module.urllib.request, "urlopen", blocked)
    monkeypatch.setattr(module.urllib.request, "build_opener", blocked)


# 功能：
#   给旧直接打开及新专用打开路径提供同一有界内存响应，以比较修复前后的行为。
# 输入：
#   monkeypatch：测试作用域替换工具。
#   document：JWKS 对象或用于测试非法响应的原始字节。
# 输出：
#   opened：记录读取次数、参数并返回新响应流的替身。
def _response(monkeypatch, document):
    payload = document if isinstance(document, bytes) else json.dumps(document).encode()
    opened = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(payload))
    monkeypatch.setattr(module.urllib.request, "urlopen", opened)
    monkeypatch.setattr(module.urllib.request, "build_opener", Mock(return_value=Mock(open=opened)))
    return opened


# 功能：
#   确认非法字符、补位符和非规范尾位不被宽松 Base64 解码偷偷接受。
# 输入：
#   value：不符合本入口 JOSE 编码规则的文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["YQ=", "YQ==", "YQ!", "Y Q", "YQ\n", "YR", "YQ+/"])
def test_base64_requires_canonical_unpadded_encoding(value):
    with pytest.raises(ValueError, match="TOKEN_INVALID"):
        module._decode_base64url(value)


# 功能：
#   令牌和 JWKS 的 JSON 不接受重复键、非有限数或 UTF-16 编码。
# 输入：
#   raw：存在解析歧义或非标准数值的原始字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "raw",
    [b'{"sub":"one","sub":"two"}', b'{"n":NaN}', b'{"n":1e999}', '{"sub":"one"}'.encode("utf-16")],
)
def test_identity_json_is_unambiguous_utf8(raw):
    with pytest.raises(ValueError, match="TOKEN_INVALID"):
        module._json_object(raw)


# 功能：
#   验证非文本令牌及孤立 Unicode 代理字符被统一拒绝，不逸出为 500 型异常。
# 输入：
#   token：非法类型或编码的令牌。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("token", [1, b"abc.def.ghi", ["token"], "\ud800.x.y"])
def test_invalid_token_has_defined_rejection(token):
    verifier = module.SupabaseJwtVerifier()
    with pytest.raises(ValueError, match="TOKEN_INVALID"):
        verifier.verify(token)


# 功能：
#   即使签名有效，未实现的关键扩展和非法关键扩展列表也不能被忽略。
# 输入：
#   critical：令牌要求接收方处理的关键扩展声明。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("critical", [["future-policy"], [], "future-policy", None])
def test_unknown_critical_extension_is_not_accepted(critical):
    verifier, token, _ = _fixture({"crit": critical, "future-policy": True})
    with pytest.raises(ValueError, match="TOKEN_HEADER_INVALID"):
        verifier.verify(token)


# 功能：
#   公钥用途、算法和允许操作不一致时拒绝验签授权；畸形列表不能引发类型异常。
# 输入：
#   patch：写入离线公钥的错误约束。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "patch",
    [
        {"use": "enc"},
        {"use": None},
        {"key_ops": ["encrypt"]},
        {"key_ops": "verify"},
        {"key_ops": ["verify", "verify"]},
        {"key_ops": ["verify", {}]},
        {"use": "sig", "key_ops": ["verify", "encrypt"]},
        {"alg": ["EdDSA"]},
        {"alg": None},
    ],
)
def test_key_constraints_are_enforced(patch):
    verifier, token, jwk = _fixture()
    jwk.update(patch)
    with pytest.raises(ValueError, match="IDENTITY_TOKEN_(KEY|ALGORITHM)_INVALID"):
        verifier.verify(token)


# 功能：
#   显式签名用途与验证操作保留正常 Ed25519 账户验证链路。
# 输入：
#   operations：允许验证的合法操作列表。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("operations", [["verify"], ["sign", "verify"]])
def test_valid_key_constraints_preserve_identity(operations):
    verifier, token, jwk = _fixture()
    jwk.update(use="sig", key_ops=operations)
    result = verifier(token)
    assert result.owner_account_id == "account-001"
    assert result.tenant_id is None


# 功能：
#   一次冷启动未知公钥查询只获取一次 JWKS，避免同一次调用重复等待网络。
# 输入：
#   monkeypatch：隔离公钥响应的替换工具。
# 输出：
#   None：不返回业务数据。
def test_unknown_key_fetches_once_per_attempt(monkeypatch):
    verifier, _, jwk = _fixture()
    verifier._loaded_at = -1_000_000.0
    opened = _response(monkeypatch, {"keys": [jwk]})
    with pytest.raises(ValueError, match="KEY_UNKNOWN"):
        verifier._key("missing-key")
    assert opened.call_count == 1


# 功能：
#   连续不同未知公钥不能反复联网；短冷却后仍能正常接收轮换的新公钥。
# 输入：
#   monkeypatch：隔离网络与单调时钟的替换工具。
# 输出：
#   None：不返回业务数据。
def test_key_miss_cooldown_allows_later_rotation(monkeypatch):
    verifier, _, jwk = _fixture()
    clock = [100.0]
    isolate_monotonic(monkeypatch, module, lambda: clock[0])
    verifier._loaded_at = 100.0
    opened = _response(monkeypatch, {"keys": [jwk]})
    for key_id in ("missing-a", "missing-b", "missing-c"):
        with pytest.raises(ValueError, match="KEY_UNKNOWN"):
            verifier._key(key_id)
    assert opened.call_count == 1
    rotated = {**jwk, "kid": "new-key"}
    opened = _response(monkeypatch, {"keys": [jwk, rotated]})
    clock[0] += 6.0
    assert verifier._key("new-key") == rotated
    assert opened.call_count == 1


# 功能：
#   同一 kid 的歧义公钥集不能覆盖缓存，失败后原缓存及其时间戳保持不变。
# 输入：
#   monkeypatch：隔离 JWKS 响应的替换工具。
# 输出：
#   None：不返回业务数据。
def test_duplicate_key_identity_does_not_replace_cache(monkeypatch):
    verifier, _, jwk = _fixture()
    previous_keys, previous_time = verifier._keys, verifier._loaded_at
    _response(monkeypatch, {"keys": [jwk, {**jwk, "x": _b64(b"x" * 32)}]})
    with pytest.raises(ValueError, match="JWKS_INVALID"):
        verifier._load_keys()
    assert verifier._keys is previous_keys
    assert verifier._loaded_at == previous_time


# 功能：
#   公钥缓存时间必须是正整数，避免 NaN 永不过期、负数持续刷新或布尔伪配置。
# 输入：
#   seconds：非法缓存寿命。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seconds", [0, -1, True, float("nan"), float("inf"), "900"])
def test_cache_duration_rejects_invalid_configuration(seconds):
    with pytest.raises(ValueError, match="CACHE_SECONDS_INVALID"):
        module.SupabaseJwtVerifier(cache_seconds=seconds)


# 功能：
#   拒绝会改变公钥目标地址或包含身份信息的签发者配置，不容忍解析器自动丢弃空白。
# 输入：
#   issuer：非法固定签发者 URL。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "issuer",
    [
        "",
        "https://",
        "https://user:pass@example.invalid/auth/v1",
        "https://example.invalid/a?",
        "https://example.invalid/a#",
        "https://example.invalid:bad/a",
        "https://example.invalid:0/a",
        "https://example.invalid/a\n",
        "https://example.invalid\\elsewhere/a",
        123,
    ],
)
def test_issuer_configuration_is_exact_https_endpoint(issuer):
    with pytest.raises(ValueError, match="AUTH_ISSUER_INVALID"):
        module.SupabaseJwtVerifier(issuer)


# 功能：
#   下载失败后不能将过期公钥重新当作有效缓存；冷却后恢复可重新验证账户。
# 输入：
#   monkeypatch：隔离时钟和网络的替换工具。
# 输出：
#   None：不返回业务数据。
def test_failed_refresh_does_not_extend_expired_key_lifetime(monkeypatch):
    verifier, token, jwk = _fixture()
    clock = [100.0]
    isolate_monotonic(monkeypatch, module, lambda: clock[0])
    verifier._loaded_at = -1000.0
    previous = verifier._keys
    opened = _response(monkeypatch, {"keys": [jwk]})
    opened.side_effect = OSError("offline")
    for _ in range(2):
        with pytest.raises(ValueError, match="JWKS_UNAVAILABLE"):
            verifier.verify(token)
    assert opened.call_count == 1
    assert verifier._keys is previous
    assert verifier._loaded_at == -1000.0
    clock[0] += 6.0
    opened = _response(monkeypatch, {"keys": [jwk]})
    assert verifier.verify(token).owner_account_id == "account-001"
    assert opened.call_count == 1


# 功能：
#   JWKS 超限或结构损坏不替换旧缓存；响应无论解析是否成功都必须关闭。
# 输入：
#   monkeypatch：网络替换工具。
#   payload：超限、重复字段或错误顶层形状的响应字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "payload",
    [b" " * (module._MAX_JWKS_BYTES + 1), b'{"keys":[],"keys":[]}', b"[]", b'{"keys":{}}'],
    ids=["oversized", "duplicate-keys", "non-object", "keys-not-list"],
)
def test_invalid_jwks_is_closed_without_replacing_cache(monkeypatch, payload):
    verifier, _, _ = _fixture()
    previous = verifier._keys
    response = io.BytesIO(payload)
    opened = _response(monkeypatch, payload)
    opened.side_effect = None
    opened.return_value = response
    with pytest.raises(ValueError, match="JWKS_INVALID"):
        verifier._load_keys()
    assert response.closed
    assert verifier._keys is previous


# 功能：
#   重定向策略经标准库处理入口拒绝离开固定公钥地址，并关闭尚未交给调用方的连接。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_redirect_rejection_closes_source_response():
    handler = module._RejectJwksRedirect()
    request = module.urllib.request.Request("https://example.invalid/auth/v1/.well-known/jwks.json")
    response = io.BytesIO(b"unused response")
    headers = Message()
    headers["Location"] = "https://other.invalid/keys"
    with pytest.raises(ValueError, match="JWKS_REDIRECT_FORBIDDEN"):
        handler.http_error_302(request, response, 302, "Found", headers)
    assert response.closed


# 功能：
#   验证专用公钥下载确实安装拒绝重定向策略，并沿用固定端点与八秒网络超时。
# 输入：
#   monkeypatch：内存公钥响应替换工具。
# 输出：
#   None：不返回业务数据。
def test_jwks_download_installs_pinned_redirect_policy(monkeypatch):
    verifier, _, jwk = _fixture()
    opened = _response(monkeypatch, {"keys": [jwk]})
    verifier._load_keys()
    handler = module.urllib.request.build_opener.call_args.args[0]
    assert isinstance(handler, module._RejectJwksRedirect)
    request = opened.call_args.args[0]
    assert request.full_url == verifier.jwks_url
    assert opened.call_args.kwargs == {"timeout": 8.0}
    assert verifier._keys == {"offline-key": jwk}


# 功能：
#   三种算法中较易混淆的 ECDSA 路径必须以真实 JOSE r/s 签名验证，损坏后仍拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_es256_accepts_real_signature_and_rejects_wrong_coordinate_width():
    key = ec.generate_private_key(ec.SECP256R1())
    public = key.public_key().public_numbers()
    verifier, original, _ = _fixture()
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "alg": "ES256",
        "kid": "ec-key",
        "x": _b64(public.x.to_bytes(32, "big")),
        "y": _b64(public.y.to_bytes(32, "big")),
    }
    verifier._keys = {"ec-key": jwk}
    header = _b64(json.dumps({"alg": "ES256", "kid": "ec-key"}).encode())
    signed = f"{header}.{original.split('.')[1]}".encode("ascii")
    r, s = decode_dss_signature(key.sign(signed, ec.ECDSA(hashes.SHA256())))
    signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    token = f"{signed.decode('ascii')}.{_b64(signature)}"
    assert verifier.verify(token).owner_account_id == "account-001"
    # 数值相同的 33 字节坐标仍不符合 P-256 公钥格式，不能靠 int 转换偷偷消除它。
    jwk["x"] = _b64(b"\0" + public.x.to_bytes(32, "big"))
    with pytest.raises(ValueError, match="SIGNATURE_INVALID"):
        verifier.verify(token)
    jwk["x"] = _b64(public.x.to_bytes(32, "big"))
    damaged = bytes([signature[0] ^ 1]) + signature[1:]
    with pytest.raises(ValueError, match="SIGNATURE_INVALID"):
        verifier.verify(f"{signed.decode('ascii')}.{_b64(damaged)}")


# 功能：
#   RSA 签名有效也不能使用低于 JWA 最低强度的公钥。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rs256_rejects_weak_but_validly_signed_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    public = key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "alg": "RS256",
        "n": _b64(public.n.to_bytes(128, "big")),
        "e": _b64(public.e.to_bytes(3, "big")),
    }
    signed = b"header.claims"
    signature = key.sign(signed, padding.PKCS1v15(), hashes.SHA256())
    with pytest.raises(ValueError, match="KEY_INVALID"):
        module.SupabaseJwtVerifier._verify_signature("RS256", jwk, signed, signature)


# 功能：
#   不完整 HTTP 响应及携带连接的 HTTP 错误都转换为固定失败，错误响应必须关闭。
# 输入：
#   monkeypatch：替换下载路径的工具。
#   failure_kind：响应中断或 HTTP 状态失败。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure_kind", ["incomplete", "http-status"])
def test_transport_failures_are_closed_and_normalized(monkeypatch, failure_kind):
    verifier, _, jwk = _fixture()
    response = io.BytesIO(b"partial")
    opened = _response(monkeypatch, {"keys": [jwk]})
    if failure_kind == "incomplete":
        opened.side_effect = None
        opened.return_value = response
        monkeypatch.setattr(response, "read", Mock(side_effect=IncompleteRead(b"part", 100)))
    else:
        opened.side_effect = HTTPError(verifier.jwks_url, 503, "Unavailable", Message(), response)
    with pytest.raises(ValueError, match="JWKS_UNAVAILABLE"):
        verifier._load_keys()
    assert response.closed
