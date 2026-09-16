from __future__ import annotations

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from dronedream_agent_app.identity import SupabaseJwtVerifier


# 功能：
#   将离线测试字节编码为 JWT 使用的不补位 URL 安全 Base64。
# 输入：
#   value：待编码字节。
# 输出：
#   encoded：符合 JOSE 段格式的文本。
def _b64(value: bytes) -> str:
    encoded = base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
    return encoded


# 功能：
#   序列化离线测试头或声明，再生成要参与签名的编码段。
# 输入：
#   value：测试用 JSON 值。
# 输出：
#   encoded：紧凑 JSON 的 Base64URL 文本。
def _encoded(value: object) -> str:
    encoded = _b64(json.dumps(value, separators=(",", ":")).encode("utf-8"))
    return encoded


# 功能：
#   使用测试专用 RSA 私钥对原始编码段签名，生成不依赖真实用户账户的访问令牌。
# 输入：
#   private_key：本次测试生成的私钥。
#   claims：待签名的账户声明。
# 输出：
#   token：包含实际 RS256 签名的紧凑 JWT。
def _token(private_key, claims: dict[str, object]) -> str:
    header = _encoded({"alg": "RS256", "kid": "test-key", "typ": "JWT"})
    body = _encoded(claims)
    signed = f"{header}.{body}".encode("ascii")
    signature = private_key.sign(signed, padding.PKCS1v15(), hashes.SHA256())
    token = f"{header}.{body}.{_b64(signature)}"
    return token


# 功能：
#   创建 2048 位测试密钥并预装对应公钥缓存，正常验证不需要联网。
# 输入：
#   无。
# 输出：
#   fixture：验证器与仅限本次测试的私钥。
def _verifier():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private_key.public_key().public_numbers()
    verifier = SupabaseJwtVerifier("https://example.supabase.co/auth/v1")
    verifier._keys = {  # noqa: SLF001 - deterministic offline verifier fixture
        "test-key": {
            "kid": "test-key",
            "kty": "RSA",
            "alg": "RS256",
            "n": _b64(public.n.to_bytes((public.n.bit_length() + 7) // 8, "big")),
            "e": _b64(public.e.to_bytes((public.e.bit_length() + 7) // 8, "big")),
        }
    }
    verifier._loaded_at = time.monotonic()  # noqa: SLF001
    fixture = (verifier, private_key)
    return fixture


# 功能：
#   验证账户、租户和组织取自已签名的管理员声明，用户资料中的伪租户不能覆盖它们。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_supabase_identity_is_signature_verified_and_derived_from_admin_claims() -> None:
    verifier, private_key = _verifier()
    claims = {
        "iss": verifier.issuer,
        "sub": "account-001",
        "aud": "authenticated",
        "role": "authenticated",
        "session_id": "session-0001",
        "iat": int(time.time()) - 10,
        "exp": int(time.time()) + 600,
        "app_metadata": {"tenant_id": "tenant-001", "organization_id": "org-001"},
        "user_metadata": {"tenant_id": "attacker-controlled"},
    }

    identity = verifier.verify(_token(private_key, claims))

    assert identity.owner_account_id == "account-001"
    assert identity.tenant_id == "tenant-001"
    assert identity.organization_id == "org-001"


# 功能：
#   分别拒绝签名后篡改账户及重新签名但受众、签发者、会话、角色或时间不合法的声明。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_supabase_identity_rejects_tampering_and_invalid_claims() -> None:
    verifier, private_key = _verifier()
    now = int(time.time())
    claims = {
        "iss": verifier.issuer,
        "sub": "account-001",
        "aud": "authenticated",
        "role": "authenticated",
        "session_id": "session-0001",
        "iat": now - 10,
        "exp": now + 600,
        "app_metadata": {},
    }
    token = _token(private_key, claims)
    segments = token.split(".")
    tampered = f"{segments[0]}.{_encoded({**claims, 'sub': 'account-002'})}.{segments[2]}"
    with pytest.raises(ValueError, match="SIGNATURE_INVALID"):
        verifier.verify(tampered)
    invalid_claim_sets = [
        {**claims, "aud": "anonymous"},
        {**claims, "iss": "https://attacker.invalid/auth/v1"},
        {**claims, "exp": now - 1},
        {**claims, "role": "service_role"},
        {**claims, "session_id": ""},
        {key: value for key, value in claims.items() if key != "iat"},
        {**claims, "iat": "not-an-integer"},
        {**claims, "nbf": "not-an-integer"},
        {**claims, "nbf": now + 120},
    ]
    for invalid_claims in invalid_claim_sets:
        with pytest.raises(ValueError, match="CLAIMS_INVALID"):
            verifier.verify(_token(private_key, invalid_claims))


# 功能：
#   验证签名有效也不能让带空白的租户或空组织标识进入账户隔离边界。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_supabase_identity_rejects_malformed_admin_boundary_claims() -> None:
    verifier, private_key = _verifier()
    now = int(time.time())
    base = {
        "iss": verifier.issuer,
        "sub": "account-001",
        "aud": "authenticated",
        "role": "authenticated",
        "session_id": "session-0001",
        "iat": now - 10,
        "exp": now + 600,
    }

    with pytest.raises(ValueError, match="TENANT_CLAIM_INVALID"):
        verifier.verify(
            _token(private_key, {**base, "app_metadata": {"tenant_id": " bad tenant "}})
        )
    with pytest.raises(ValueError, match="ORGANIZATION_CLAIM_INVALID"):
        verifier.verify(_token(private_key, {**base, "app_metadata": {"organization_id": ""}}))


# 功能：
#   畸形算法字段应统一拒绝，不因列表、对象等不可哈希类型而崩溃。
# 输入：
#   algorithm：错误类型的算法声明。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("algorithm", [["RS256"], {"alg": "RS256"}, None, 1])
def test_malformed_algorithm_is_a_defined_identity_rejection(algorithm) -> None:
    verifier, _ = _verifier()
    token = f"{_encoded({'alg': algorithm, 'kid': 'test-key'})}.{_encoded({})}.{_b64(b'invalid')}"
    with pytest.raises(ValueError, match="ALGORITHM_INVALID"):
        verifier.verify(token)


# 功能：
#   验签前的声明解析必须限制深度，将恶意深层输入转换为明确拒绝而非递归崩溃。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_deeply_nested_unsigned_claims_cannot_crash_identity_parser() -> None:
    verifier, _ = _verifier()
    nested = b'{"x":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}"
    token = f"{_encoded({'alg': 'RS256', 'kid': 'test-key'})}.{_b64(nested)}.{_b64(b'invalid')}"
    with pytest.raises(ValueError, match="TOKEN_INVALID"):
        verifier.verify(token)
