"""Verified Supabase identity boundary for privileged local Core operations."""

from __future__ import annotations

import base64
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from dronedream_plugin_sdk.protocol import decode_json

DEFAULT_SUPABASE_AUTH_ISSUER = "https://yggabfynndpzymlqvnim.supabase.co/auth/v1"
_MAX_JWT_BYTES = 16 * 1024
_MAX_JWKS_BYTES = 256 * 1024
_MIN_JWKS_REFRESH_SECONDS = 5.0
_BOUNDARY_CLAIM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_BASE64URL = re.compile(r"[A-Za-z0-9_-]+")


# 功能：
#   1. 限量解码 JOSE 段，只允许 URL 安全字母表、无补位符和规范尾位。
#   2. 解码不代表身份可信；签名与账户声明仍须后续验证。
# 输入：
#   value：JWT 段或 JWK 数值的 Base64URL 文本。
# 输出：
#   decoded：对应的原始字节。
def _decode_base64url(value: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_JWT_BYTES
        or _BASE64URL.fullmatch(value) is None
    ):
        raise ValueError("IDENTITY_TOKEN_INVALID")
    try:
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("IDENTITY_TOKEN_INVALID") from error
    # 验签字节相同也不能接受另一种尾位写法，避免同一签名有多个文本身份。
    if base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != value:
        raise ValueError("IDENTITY_TOKEN_INVALID")
    return decoded


# 功能：
#   用共享有界解析拒绝重复键、非有限数、非 UTF-8 及过深内容，且只接受 JSON 对象。
# 输入：
#   value：解码后的令牌头、声明或公钥集字节。
# 输出：
#   parsed：通过结构检查的对象。
def _json_object(value: bytes) -> dict[str, Any]:
    try:
        parsed = decode_json(value, limit=_MAX_JWKS_BYTES, node_limit=65_536)
    except (ValueError, TypeError) as error:
        raise ValueError("IDENTITY_TOKEN_INVALID") from error
    if not isinstance(parsed, dict):
        raise ValueError("IDENTITY_TOKEN_INVALID")
    return parsed


class _RejectJwksRedirect(urllib.request.HTTPRedirectHandler):
    # 功能：
    #   禁止公钥下载跟随重定向，使信任源始终为本机配置的固定 HTTPS 端点。
    # 输入：
    #   self：JWKS 重定向策略。
    #   req：原始请求。
    #   fp：原始响应流。
    #   code：重定向状态码。
    #   msg：响应说明。
    #   headers：响应头。
    #   newurl：服务器提议的新地址。
    # 输出：
    #   None：不返回业务数据。
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 抛出自定义拒绝前关闭尚未交给调用者的响应，避免重定向错误泄漏连接。
        fp.close()
        raise ValueError("IDENTITY_JWKS_REDIRECT_FORBIDDEN")


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    owner_account_id: str
    tenant_id: str | None
    organization_id: str | None
    issuer: str
    expires_at: int


class SupabaseJwtVerifier:
    """Validate the access-token signature and derive immutable owner scope.

    Only the JWT subject and administrator-controlled ``app_metadata`` are used
    for the memory boundary.  Mutable ``user_metadata`` is deliberately ignored.
    """

    # 功能：
    #   1. 从本机配置固定签发者和公钥端点，不从令牌提供的 URL 发现公钥。
    #   2. 校验正整数缓存寿命，并为缓存替换和短期刷新限流建立独立状态。
    # 输入：
    #   self：待初始化的身份验证器。
    #   issuer：显式 HTTPS 签发者；缺省时使用环境配置或产品默认值。
    #   cache_seconds：公钥成功加载后的有效缓存秒数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, issuer: str | None = None, *, cache_seconds: int = 900) -> None:
        configured = (
            issuer
            if issuer is not None
            else os.environ.get("DRONEDREAM_SUPABASE_AUTH_ISSUER", DEFAULT_SUPABASE_AUTH_ISSUER)
        )
        if (
            not isinstance(configured, str)
            or not configured
            or any(character.isspace() or ord(character) < 32 for character in configured)
        ):
            raise ValueError("SUPABASE_AUTH_ISSUER_INVALID")
        try:
            parts = urlsplit(configured)
            if (
                parts.scheme != "https"
                or not parts.hostname
                or parts.username is not None
                or parts.password is not None
                or "?" in configured
                or "#" in configured
                or "\\" in configured
                or parts.port == 0
            ):
                raise ValueError("SUPABASE_AUTH_ISSUER_INVALID")
        except ValueError as error:
            raise ValueError("SUPABASE_AUTH_ISSUER_INVALID") from error
        if type(cache_seconds) is not int or cache_seconds <= 0:
            raise ValueError("IDENTITY_CACHE_SECONDS_INVALID")
        self.issuer = configured.rstrip("/")
        self.jwks_url = f"{self.issuer}/.well-known/jwks.json"
        self.cache_seconds = cache_seconds
        self._keys: dict[str, dict[str, Any]] = {}
        self._loaded_at = 0.0
        self._last_refresh_attempt: float | None = None
        self._lock = threading.Lock()

    # 功能：
    #   为依赖注入提供可调用入口，复用同一验签路径，不建立第二条信任规则。
    # 输入：
    #   self：身份验证器。
    #   token：待验证的访问令牌。
    # 输出：
    #   identity：经过签名和作用域验证的不可变账户身份。
    def __call__(self, token: str) -> VerifiedIdentity:
        identity = self.verify(token)
        return identity

    # 功能：
    #   1. 校验令牌格式、受支持算法、公钥用途与签名；未实现的关键扩展直接拒绝。
    #   2. 仅在验签成功后解析账户归属，错误不包含令牌或凭证原文。
    # 输入：
    #   self：固定签发者及缓存的验证器。
    #   token：紧凑 JWS 格式的 Supabase 访问令牌。
    # 输出：
    #   identity：可供本机特权操作绑定的账户身份。
    def verify(self, token: str) -> VerifiedIdentity:
        if (
            not isinstance(token, str)
            or not token
            or len(token) > _MAX_JWT_BYTES
            or not token.isascii()
        ):
            raise ValueError("IDENTITY_TOKEN_INVALID")
        segments = token.split(".")
        if len(segments) != 3:
            raise ValueError("IDENTITY_TOKEN_INVALID")
        header = _json_object(_decode_base64url(segments[0]))
        claims = _json_object(_decode_base64url(segments[1]))
        signature = _decode_base64url(segments[2])
        algorithm = header.get("alg")
        key_id = header.get("kid")
        if (
            not isinstance(algorithm, str)
            or algorithm not in {"RS256", "ES256", "EdDSA"}
            or not isinstance(key_id, str)
            or not key_id
        ):
            raise ValueError("IDENTITY_TOKEN_ALGORITHM_INVALID")
        # 本验证器没有实现 JOSE 关键扩展；不能忽略签发者要求必须理解的语义。
        if "crit" in header:
            raise ValueError("IDENTITY_TOKEN_HEADER_INVALID")
        jwk = self._key(key_id)
        if "alg" in jwk and jwk["alg"] != algorithm:
            raise ValueError("IDENTITY_TOKEN_ALGORITHM_INVALID")
        if "use" in jwk and jwk["use"] != "sig":
            raise ValueError("IDENTITY_TOKEN_KEY_INVALID")
        if "key_ops" in jwk:
            operations = jwk["key_ops"]
            if (
                not isinstance(operations, list)
                or not all(isinstance(operation, str) for operation in operations)
                or "verify" not in operations
                or len(set(operations)) != len(operations)
                or any(operation not in {"sign", "verify"} for operation in operations)
            ):
                raise ValueError("IDENTITY_TOKEN_KEY_INVALID")
        signed = f"{segments[0]}.{segments[1]}".encode("ascii")
        try:
            self._verify_signature(algorithm, jwk, signed, signature)
        except (InvalidSignature, ValueError, TypeError) as error:
            raise ValueError("IDENTITY_TOKEN_SIGNATURE_INVALID") from error
        identity = self._verified_claims(claims)
        return identity

    # 功能：
    #   1. 用锁序列化过期及轮换刷新，每次调用至多请求一次，失败不延长旧缓存寿命。
    #   2. 刷新尝试之间至少间隔五秒；未知 kid 洪泛及故障重试不连续占用网络。
    #   3. 冷却期间只使用未过期的已知公钥，新公钥最多等待剩余冷却时间后再尝试。
    # 输入：
    #   self：持有缓存及刷新时间的验证器。
    #   key_id：令牌中要查找的公钥标识。
    # 输出：
    #   key：本机固定签发者公钥集中的对应公钥。
    def _key(self, key_id: str) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            expired = not self._keys or now - self._loaded_at >= self.cache_seconds
            if expired or key_id not in self._keys:
                cooling = (
                    self._last_refresh_attempt is not None
                    and now - self._last_refresh_attempt < _MIN_JWKS_REFRESH_SECONDS
                )
                if cooling and expired:
                    raise ValueError("IDENTITY_JWKS_UNAVAILABLE")
                if not cooling:
                    # 成功或失败都记录尝试时刻；只有成功的 _load_keys 才更新有效期。
                    self._last_refresh_attempt = now
                    self._load_keys()
            key = self._keys.get(key_id)
        if key is None:
            raise ValueError("IDENTITY_TOKEN_KEY_UNKNOWN")
        return key

    # 功能：
    #   1. 限量读取固定 HTTPS 公钥集且不跟随重定向，拒绝重复 kid 的歧义索引。
    #   2. 全部结构检查成功后才替换缓存和有效期；失败保留旧内容但不授予过期使用权。
    # 输入：
    #   self：保存公钥端点及缓存的验证器。
    # 输出：
    #   None：不返回业务数据。
    def _load_keys(self) -> None:
        request = urllib.request.Request(  # noqa: S310
            self.jwks_url,
            headers={"Accept": "application/json", "User-Agent": "DroneDream-AGENT/1"},
        )
        opener = urllib.request.build_opener(_RejectJwksRedirect())
        try:
            with opener.open(request, timeout=8.0) as response:  # noqa: S310
                payload = response.read(_MAX_JWKS_BYTES + 1)
        except HTTPError as error:
            # HTTPError 自己持有响应流，但 open 未成功返回，外层 with 尚未取得它。
            try:
                error.close()
            finally:
                raise ValueError("IDENTITY_JWKS_UNAVAILABLE") from error
        except (OSError, HTTPException) as error:
            raise ValueError("IDENTITY_JWKS_UNAVAILABLE") from error
        if len(payload) > _MAX_JWKS_BYTES:
            raise ValueError("IDENTITY_JWKS_INVALID")
        try:
            document = _json_object(payload)
        except ValueError as error:
            raise ValueError("IDENTITY_JWKS_INVALID") from error
        keys = document.get("keys")
        if not isinstance(keys, list) or len(keys) > 32:
            raise ValueError("IDENTITY_JWKS_INVALID")
        parsed: dict[str, dict[str, Any]] = {}
        for raw in keys:
            # 无 kid 的公钥不匹配本产品访问令牌；同 kid 多钥却不能按最后一个静默覆盖。
            if not isinstance(raw, dict) or not isinstance(raw.get("kid"), str):
                continue
            key_id = raw["kid"]
            if not key_id or key_id in parsed:
                raise ValueError("IDENTITY_JWKS_INVALID")
            parsed[key_id] = raw
        if not parsed:
            raise ValueError("IDENTITY_JWKS_INVALID")
        self._keys = parsed
        self._loaded_at = time.monotonic()

    # 功能：
    #   1. 按算法核对公钥类型与曲线，使用密码库验签；ECDSA 的 JOSE r/s 转为库所需 DER。
    #   2. RSA 至少 2048 位，P-256 坐标各 32 字节，不能将不合规弱钥当作可信身份来源。
    # 输入：
    #   algorithm：入口白名单允许的签名算法。
    #   jwk：已经匹配 kid 及用途的公钥描述。
    #   signed：原始编码头和声明组成的签名消息。
    #   signature：解码后的签名字节。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def _verify_signature(
        algorithm: object,
        jwk: dict[str, Any],
        signed: bytes,
        signature: bytes,
    ) -> None:
        if algorithm == "RS256" and jwk.get("kty") == "RSA":
            modulus = int.from_bytes(_decode_base64url(jwk.get("n", "")), "big")
            exponent = int.from_bytes(_decode_base64url(jwk.get("e", "")), "big")
            if modulus.bit_length() < 2048:
                raise ValueError("IDENTITY_TOKEN_KEY_INVALID")
            rsa.RSAPublicNumbers(exponent, modulus).public_key().verify(
                signature, signed, padding.PKCS1v15(), hashes.SHA256()
            )
            return
        if algorithm == "ES256" and jwk.get("kty") == "EC" and jwk.get("crv") == "P-256":
            if len(signature) != 64:
                raise ValueError("IDENTITY_TOKEN_SIGNATURE_INVALID")
            x_bytes = _decode_base64url(jwk.get("x", ""))
            y_bytes = _decode_base64url(jwk.get("y", ""))
            if len(x_bytes) != 32 or len(y_bytes) != 32:
                raise ValueError("IDENTITY_TOKEN_KEY_INVALID")
            x = int.from_bytes(x_bytes, "big")
            y = int.from_bytes(y_bytes, "big")
            raw_r = int.from_bytes(signature[:32], "big")
            raw_s = int.from_bytes(signature[32:], "big")
            ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key().verify(
                encode_dss_signature(raw_r, raw_s),
                signed,
                ec.ECDSA(hashes.SHA256()),
            )
            return
        if algorithm == "EdDSA" and jwk.get("kty") == "OKP" and jwk.get("crv") == "Ed25519":
            public_bytes = _decode_base64url(jwk.get("x", ""))
            ed25519.Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, signed)
            return
        raise ValueError("IDENTITY_TOKEN_KEY_INVALID")

    # 功能：
    #   1. 检查签发者、受众、账户、会话和时间，拒绝过期及匿名或服务角色令牌。
    #   2. 只从签名内管理员控制的 app_metadata 派生租户与组织，不信任 user_metadata。
    #   3. 本地验签不实时检查云端撤销；令牌有效期与服务端撤销机制仍是独立约束。
    # 输入：
    #   self：固定签发者的验证器。
    #   claims：已经通过验签的声明对象。
    # 输出：
    #   identity：冻结的账户与可选租户、组织作用域。
    def _verified_claims(self, claims: dict[str, Any]) -> VerifiedIdentity:
        now = int(time.time())
        issuer = claims.get("iss")
        subject = claims.get("sub")
        expiry = claims.get("exp")
        role = claims.get("role")
        session_id = claims.get("session_id")
        audience = claims.get("aud")
        if isinstance(audience, str):
            audiences = {audience}
        elif isinstance(audience, list) and all(isinstance(value, str) for value in audience):
            audiences = set(audience)
        else:
            audiences = set()
        if (
            issuer != self.issuer
            or not isinstance(subject, str)
            or not 2 <= len(subject) <= 160
            or type(expiry) is not int
            or expiry <= now
            or role != "authenticated"
            or not isinstance(session_id, str)
            or not 8 <= len(session_id) <= 200
            or "authenticated" not in audiences
        ):
            raise ValueError("IDENTITY_TOKEN_CLAIMS_INVALID")
        not_before = claims.get("nbf")
        issued_at = claims.get("iat")
        if (
            type(issued_at) is not int
            or issued_at > now + 60
            or (not_before is not None and type(not_before) is not int)
            or (type(not_before) is int and not_before > now + 60)
        ):
            raise ValueError("IDENTITY_TOKEN_CLAIMS_INVALID")
        app_metadata = claims.get("app_metadata")
        # 用户可编辑的资料不能替代管理员账户边界；缺少管理员租户声明表示个人作用域。
        if not isinstance(app_metadata, dict):
            app_metadata = {}
        tenant = app_metadata.get("tenant_id")
        organization = app_metadata.get("organization_id", app_metadata.get("org_id"))
        if tenant is not None and (
            not isinstance(tenant, str) or _BOUNDARY_CLAIM.fullmatch(tenant) is None
        ):
            raise ValueError("IDENTITY_TENANT_CLAIM_INVALID")
        if organization is not None and (
            not isinstance(organization, str) or _BOUNDARY_CLAIM.fullmatch(organization) is None
        ):
            raise ValueError("IDENTITY_ORGANIZATION_CLAIM_INVALID")
        identity = VerifiedIdentity(
            owner_account_id=subject,
            tenant_id=tenant,
            organization_id=organization,
            issuer=self.issuer,
            expires_at=expiry,
        )
        return identity
