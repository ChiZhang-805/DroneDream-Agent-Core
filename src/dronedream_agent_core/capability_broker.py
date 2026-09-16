"""Core-owned capability broker for least-authority plugin I/O.

Plugins receive a scoped facade. They never receive a credential value and cannot
expand their own filesystem, network, or process authority at runtime.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

from dronedream_plugin_sdk.protocol import copy_json, decode_json

from .pinned_https import PinnedHTTPSHandler
from .plugin_contracts import CapabilityBrokerReceipt, PluginManifest
from .plugin_files import check_plain_plugin_path, portable_plugin_path, read_plugin_file
from .plugin_process import PluginProcessError
from .process_capture import capture_process

# A fixed stripe set avoids an unbounded path->lock cache. Distinct broker
# instances in this process share publication ordering for the same target.
_PUBLICATION_LOCKS = tuple(threading.RLock() for _ in range(64))


class CapabilityBrokerError(RuntimeError):
    """A denied or failed broker operation with a stable issue code."""

    def __init__(self, issue_code: str) -> None:
        """Expose a stable rejection reason without returning provider/file exception text."""
        super().__init__(issue_code)
        self.issue_code = issue_code


class CredentialResolver(Protocol):
    def resolve(self, reference: str, *, plugin_id: str) -> str: ...


class EnvironmentCredentialResolver:
    """Resolve named connector credentials without exposing the environment to plugins."""

    def __init__(self, *, prefix: str = "DRONEDREAM_CONNECTOR_") -> None:
        """Configure a development-only credential namespace, not account/plugin ACL storage."""
        self.prefix = prefix

    def resolve(self, reference: str, *, plugin_id: str) -> str:
        """Resolve an explicit reference; production callers use a plugin-scoped resolver."""
        if not isinstance(reference, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", reference):
            raise CapabilityBrokerError("BROKER_CREDENTIAL_REFERENCE_INVALID")
        variable = self.prefix + reference.upper().replace("-", "_")
        value = os.environ.get(variable, "")
        if not value:
            raise CapabilityBrokerError("BROKER_CREDENTIAL_UNAVAILABLE")
        return value


class _NoRedirect(HTTPRedirectHandler):
    """Do not forward injected credentials to a redirected location, even on the same host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        """Deny all redirects rather than validating only the initial URL."""
        raise CapabilityBrokerError("BROKER_NETWORK_REDIRECT_DENIED")


@dataclass(frozen=True)
class BrokerHttpResponse:
    """Bounded response bytes with filtered headers; HTTP errors remain failures."""

    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> object:
        """Decode finite, bounded, unambiguous JSON before plugin normalization."""
        return decode_json(self.body)


@dataclass(frozen=True)
class BrokerProcessResult:
    """Actual exit code and bounded decoded output from an explicitly allowed executable."""

    returncode: int
    stdout: str
    stderr: str


class CoreCapabilityBroker:
    """Authority root used by the app to mint plugin-specific broker facades."""

    def __init__(
        self,
        *,
        read_roots: Mapping[str, Path] | None = None,
        write_roots: Mapping[str, Path] | None = None,
        allowed_executables: Mapping[str, Path] | None = None,
        credential_resolver: CredentialResolver | None = None,
        receipt_sink: Callable[[CapabilityBrokerReceipt], None] | None = None,
    ) -> None:
        """Capture core-owned roots/executable allowlists; plugins never supply these mappings."""
        self._read_roots = self._normalize_roots(read_roots or {})
        self._write_roots = self._normalize_roots(write_roots or {}, create=True)
        self._allowed_executables = {
            name: path.resolve(strict=True) for name, path in (allowed_executables or {}).items()
        }
        self._credential_resolver = credential_resolver
        self._receipt_sink = receipt_sink

    @staticmethod
    def _normalize_roots(values: Mapping[str, Path], *, create: bool = False) -> dict[str, Path]:
        """Validate roots before creation/resolution, rejecting static directory redirection."""
        roots: dict[str, Path] = {}
        for name, value in values.items():
            if not name or not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError("BROKER_ROOT_NAME_INVALID")
            check_plain_plugin_path(value)
            if create:
                value.mkdir(parents=True, exist_ok=True)
            check_plain_plugin_path(value)
            if not value.is_dir():
                raise ValueError("BROKER_ROOT_DIRECTORY_REQUIRED")
            roots[name] = value.resolve(strict=True)
        return roots

    def scope(self, manifest: PluginManifest) -> ScopedCapabilityBroker:
        """Mint a new immutable permission snapshot for one validated plugin manifest."""
        return ScopedCapabilityBroker(
            manifest=manifest,
            read_roots=self._read_roots,
            write_roots=self._write_roots,
            allowed_executables=self._allowed_executables,
            credential_resolver=self._credential_resolver,
            receipt_sink=self._receipt_sink,
        )


class ScopedCapabilityBroker:
    """A non-escalatable I/O facade bound to one validated manifest."""

    def __init__(
        self,
        *,
        manifest: PluginManifest,
        read_roots: Mapping[str, Path],
        write_roots: Mapping[str, Path],
        allowed_executables: Mapping[str, Path],
        credential_resolver: CredentialResolver | None,
        receipt_sink: Callable[[CapabilityBrokerReceipt], None] | None,
    ) -> None:
        """Revalidate mutable model input before freezing permissions, hosts and resource bounds."""
        manifest = PluginManifest.model_validate(manifest.model_dump())
        self.plugin_id = manifest.plugin_id
        self._permissions = frozenset(manifest.permissions)
        self._hosts = frozenset(
            host.lower().rstrip(".") for host in manifest.resource_policy.allowed_network_hosts
        )
        self._read_roots = dict(read_roots)
        self._write_roots = dict(write_roots)
        self._allowed_executables = dict(allowed_executables)
        self._credential_resolver = credential_resolver
        self._receipt_sink = receipt_sink
        self._maximum_bytes = manifest.resource_policy.maximum_message_bytes
        self._timeout = manifest.runtime.call_timeout_seconds
        self._resource_policy = manifest.resource_policy.model_copy(deep=True)

    def _receipt(
        self,
        operation: str,
        outcome: Literal["accepted", "denied", "failed"],
        resource: str,
        *,
        byte_count: int = 0,
        issue_codes: list[str] | None = None,
    ) -> None:
        """Hash resource identifiers in receipts; never log raw URLs, paths or credential values."""
        if self._receipt_sink is not None:
            self._receipt_sink(
                CapabilityBrokerReceipt(
                    plugin_id=self.plugin_id,
                    operation=operation,
                    outcome=outcome,
                    resource_sha256=hashlib.sha256(resource.encode("utf-8")).hexdigest(),
                    byte_count=byte_count,
                    issue_codes=issue_codes or [],
                )
            )

    def _require(self, permission: str, *, operation: str, resource: str) -> None:
        """Enforce the captured grant on every operation, not only when the plugin is loaded."""
        if permission not in self._permissions:
            self._receipt(operation, "denied", resource, issue_codes=["BROKER_PERMISSION_DENIED"])
            raise CapabilityBrokerError("BROKER_PERMISSION_DENIED")

    @staticmethod
    def _resolve_beneath(root: Path, relative_path: str, *, must_exist: bool) -> Path:
        """Reject portable-path aliases and static links before resolution beneath the grant.

        This is not a filesystem sandbox against a hostile process racing parent
        directory changes. Isolated plugins still require the OS boundary.
        """
        if not isinstance(relative_path, str):
            raise CapabilityBrokerError("BROKER_PATH_INVALID")
        try:
            relative_path = portable_plugin_path(relative_path.replace("\\", "/"))
        except ValueError as error:
            raise CapabilityBrokerError("BROKER_PATH_INVALID") from error
        candidate_value = Path(relative_path)
        if (
            candidate_value.is_absolute()
            or ".." in candidate_value.parts
            or "\x00" in relative_path
        ):
            raise CapabilityBrokerError("BROKER_PATH_INVALID")
        try:
            check_plain_plugin_path(root / candidate_value)
        except ValueError as error:
            raise CapabilityBrokerError("BROKER_PATH_ESCAPE") from error
        candidate = (root / candidate_value).resolve(strict=must_exist)
        if candidate != root and root not in candidate.parents:
            raise CapabilityBrokerError("BROKER_PATH_ESCAPE")
        return candidate

    def read_bytes(self, root_name: str, relative_path: str) -> bytes:
        """Read bounded bytes and detect replacement; stat alone is not a read limit."""
        if not isinstance(root_name, str):
            raise CapabilityBrokerError("BROKER_ROOT_UNAVAILABLE")
        resource = f"{root_name}:{relative_path}"
        permission = "attachment.read" if root_name == "attachments" else "asset.read"
        self._require(permission, operation="filesystem.read", resource=resource)
        root = self._read_roots.get(root_name)
        if root is None:
            raise CapabilityBrokerError("BROKER_ROOT_UNAVAILABLE")
        try:
            path = self._resolve_beneath(root, relative_path, must_exist=True)
            if not path.is_file():
                raise CapabilityBrokerError("BROKER_FILE_REQUIRED")
            if path.stat().st_size > self._maximum_bytes:
                raise CapabilityBrokerError("BROKER_RESPONSE_TOO_LARGE")
            try:
                value = read_plugin_file(path, limit=self._maximum_bytes)
            except ValueError as error:
                raise CapabilityBrokerError("BROKER_FILE_CHANGED_OR_TOO_LARGE") from error
        except CapabilityBrokerError as error:
            self._receipt("filesystem.read", "denied", resource, issue_codes=[error.issue_code])
            raise
        except OSError as error:
            self._receipt(
                "filesystem.read", "failed", resource, issue_codes=["BROKER_FILE_READ_FAILED"]
            )
            raise CapabilityBrokerError("BROKER_FILE_READ_FAILED") from error
        self._receipt("filesystem.read", "accepted", resource, byte_count=len(value))
        return value

    def write_bytes(self, root_name: str, relative_path: str, value: bytes) -> None:
        """Use an exclusive same-directory temporary so concurrent writes never mix bytes."""
        if not isinstance(root_name, str):
            raise CapabilityBrokerError("BROKER_ROOT_UNAVAILABLE")
        resource = f"{root_name}:{relative_path}"
        permission = "asset.write-staging" if root_name == "staging" else "mission.write-output"
        self._require(permission, operation="filesystem.write", resource=resource)
        if type(value) is not bytes or len(value) > self._maximum_bytes:
            raise CapabilityBrokerError("BROKER_REQUEST_TOO_LARGE")
        root = self._write_roots.get(root_name)
        if root is None:
            raise CapabilityBrokerError("BROKER_ROOT_UNAVAILABLE")
        if not isinstance(relative_path, str):
            raise CapabilityBrokerError("BROKER_PATH_INVALID")
        identity = os.path.normcase(str(root / relative_path.replace("\\", "/")))
        with _PUBLICATION_LOCKS[hash(identity) % len(_PUBLICATION_LOCKS)]:
            self._write_at_root(root, relative_path, value, resource)

    def _write_at_root(self, root: Path, relative_path: str, value: bytes, resource: str) -> None:
        """Serialize in-process validation/publication; other-process conflicts fail closed."""
        temporary = None
        owns_temporary = False
        try:
            path = self._resolve_beneath(root, relative_path, must_exist=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            check_plain_plugin_path(path)
            temporary = path.with_name(path.name + f".broker-{uuid4().hex}.tmp")
            with temporary.open("xb") as stream:
                owns_temporary = True
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            check_plain_plugin_path(path)
            temporary.replace(path)
            temporary = None
        except CapabilityBrokerError as error:
            self._receipt("filesystem.write", "denied", resource, issue_codes=[error.issue_code])
            raise
        except (OSError, ValueError) as error:
            self._receipt(
                "filesystem.write", "failed", resource, issue_codes=["BROKER_FILE_WRITE_FAILED"]
            )
            raise CapabilityBrokerError("BROKER_FILE_WRITE_FAILED") from error
        finally:
            # Only this call's fresh temporary is disposable; never clear an
            # existing target, shared temporary name or staging directory.
            if temporary is not None and owns_temporary:
                temporary.unlink(missing_ok=True)
        self._receipt("filesystem.write", "accepted", resource, byte_count=len(value))

    @staticmethod
    def _validate_resolved_host(host: str, port: int, *, allow_private: bool) -> tuple[str, ...]:
        """Validate every DNS answer and return the exact bounded numeric destination set."""
        try:
            addresses = {
                item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            }
        except socket.gaierror as error:
            raise CapabilityBrokerError("BROKER_NETWORK_DNS_FAILED") from error
        if not addresses or len(addresses) > 16:
            raise CapabilityBrokerError("BROKER_NETWORK_DNS_FAILED")
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not allow_private and any(
                (
                    ip.is_private,
                    ip.is_loopback,
                    ip.is_link_local,
                    ip.is_multicast,
                    ip.is_reserved,
                    ip.is_unspecified,
                )
            ):
                raise CapabilityBrokerError("BROKER_NETWORK_PRIVATE_ADDRESS_DENIED")
        return tuple(sorted(addresses))

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        credential_reference: str | None = None,
        credential_header: str = "Authorization",
        credential_prefix: str = "Bearer ",
    ) -> BrokerHttpResponse:
        """Enforce URL/header/body boundaries and pin validated DNS addresses through TLS."""
        if not isinstance(url, str) or not url or len(url) > 8192 or any(ord(c) < 32 for c in url):
            raise CapabilityBrokerError("BROKER_NETWORK_TARGET_DENIED")
        if "network.external" not in self._permissions:
            self._require("network.local-device", operation="network.request", resource=url)
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or "").lower().rstrip(".")
            port = parsed.port
        except ValueError as error:
            raise CapabilityBrokerError("BROKER_NETWORK_TARGET_DENIED") from error
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username
            or parsed.password
            or parsed.fragment
            or port not in {None, 443}
            or host not in self._hosts
        ):
            self._receipt(
                "network.request", "denied", url, issue_codes=["BROKER_NETWORK_TARGET_DENIED"]
            )
            raise CapabilityBrokerError("BROKER_NETWORK_TARGET_DENIED")
        if not isinstance(method, str):
            raise CapabilityBrokerError("BROKER_NETWORK_METHOD_DENIED")
        verb = method.upper()
        if verb not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise CapabilityBrokerError("BROKER_NETWORK_METHOD_DENIED")
        payload = b"" if body is None else body
        if type(payload) is not bytes or len(payload) > self._maximum_bytes:
            raise CapabilityBrokerError("BROKER_REQUEST_TOO_LARGE")
        safe_headers: dict[str, str] = {}
        denied_headers = {
            "authorization",
            "cookie",
            "proxy-authorization",
            "host",
            "content-length",
            "transfer-encoding",
            "connection",
            "x-api-key",
            "api-key",
            "as-api-key",
        }
        header_values = {} if headers is None else headers
        if not isinstance(header_values, Mapping) or len(header_values) > 64:
            raise CapabilityBrokerError("BROKER_NETWORK_HEADER_DENIED")
        for name, value in header_values.items():
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
                or name.lower() in denied_headers
                or not isinstance(value, str)
                or len(name) > 128
                or len(value) > 8192
                or any(ord(c) < 32 or ord(c) == 127 for c in value)
            ):
                raise CapabilityBrokerError("BROKER_NETWORK_HEADER_DENIED")
            safe_headers[name] = value
        if credential_reference is not None:
            if (
                not isinstance(credential_reference, str)
                or not credential_reference
                or len(credential_reference) > 256
                or not isinstance(credential_header, str)
                or not isinstance(credential_prefix, str)
                or len(credential_prefix) > 256
                or any(ord(c) < 32 or ord(c) == 127 for c in credential_prefix)
            ):
                raise CapabilityBrokerError("BROKER_CREDENTIAL_HEADER_DENIED")
            self._require(
                "credential.reference", operation="credential.inject", resource=credential_reference
            )
            if self._credential_resolver is None:
                raise CapabilityBrokerError("BROKER_CREDENTIAL_RESOLVER_UNAVAILABLE")
            if credential_header.lower() not in {
                "authorization",
                "x-api-key",
                "api-key",
                "as-api-key",
            }:
                raise CapabilityBrokerError("BROKER_CREDENTIAL_HEADER_DENIED")
            secret = self._credential_resolver.resolve(
                credential_reference, plugin_id=self.plugin_id
            )
            if (
                not isinstance(secret, str)
                or not secret
                or len(secret) > 8192
                or any(ord(c) < 32 or ord(c) > 126 for c in secret)
            ):
                raise CapabilityBrokerError("BROKER_CREDENTIAL_VALUE_INVALID")
            safe_headers[credential_header] = credential_prefix + secret
            self._receipt("credential.inject", "accepted", credential_reference)
        deadline = time.monotonic() + self._timeout
        try:
            addresses = self._validate_resolved_host(
                host,
                443,
                allow_private="network.local-device" in self._permissions,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CapabilityBrokerError("BROKER_NETWORK_DEADLINE_EXCEEDED")
            # Never inherit system/environment proxies: they could resolve the
            # hostname again and bypass the vetted literal destination set.
            response = build_opener(
                ProxyHandler({}), _NoRedirect(), PinnedHTTPSHandler(host, addresses)
            ).open(
                Request(url, data=payload or None, headers=safe_headers, method=verb),
                timeout=remaining,
            )
            with response:
                response_body = bytearray()
                # read1 performs bounded underlying reads, unlike read(n),
                # which may accumulate a slow-drip body indefinitely. Socket
                # timeouts remain per-I/O; the checks also reject late delivery,
                # but do not claim OS-level cancellation of blocked DNS.
                read = getattr(response, "read1", response.read)
                while True:
                    if time.monotonic() >= deadline:
                        raise CapabilityBrokerError("BROKER_NETWORK_DEADLINE_EXCEEDED")
                    chunk = read(min(65536, self._maximum_bytes - len(response_body) + 1))
                    if time.monotonic() >= deadline:
                        raise CapabilityBrokerError("BROKER_NETWORK_DEADLINE_EXCEEDED")
                    if not isinstance(chunk, bytes):
                        raise CapabilityBrokerError("BROKER_NETWORK_RESPONSE_INVALID")
                    if not chunk:
                        break
                    response_body.extend(chunk)
                    if len(response_body) > self._maximum_bytes:
                        raise CapabilityBrokerError("BROKER_RESPONSE_TOO_LARGE")
                response_headers = {
                    name.lower(): value
                    for name, value in response.headers.items()
                    if name.lower() not in {"set-cookie", "authorization", "proxy-authenticate"}
                }
                result = BrokerHttpResponse(
                    status=int(response.status), headers=response_headers, body=bytes(response_body)
                )
        except CapabilityBrokerError as error:
            self._receipt("network.request", "denied", url, issue_codes=[error.issue_code])
            raise
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            self._receipt("network.request", "failed", url, issue_codes=["BROKER_NETWORK_FAILED"])
            raise CapabilityBrokerError("BROKER_NETWORK_FAILED") from error
        self._receipt("network.request", "accepted", url, byte_count=len(result.body))
        return result

    def spawn(
        self, executable_id: str, arguments: list[str], *, stdin: bytes = b""
    ) -> BrokerProcessResult:
        """Run a core-allowed executable, bounding output during capture rather than afterward."""
        if not isinstance(executable_id, str):
            raise CapabilityBrokerError("BROKER_EXECUTABLE_DENIED")
        resource = f"executable:{executable_id}"
        self._require("process.spawn", operation="process.spawn", resource=resource)
        executable = self._allowed_executables.get(executable_id)
        if executable is None:
            raise CapabilityBrokerError("BROKER_EXECUTABLE_DENIED")
        if (
            not isinstance(arguments, list)
            or len(arguments) > 64
            or any(
                not isinstance(value, str) or not value or "\x00" in value or len(value) > 1_024
                for value in arguments
            )
        ):
            raise CapabilityBrokerError("BROKER_ARGUMENTS_INVALID")
        if type(stdin) is not bytes or len(stdin) > self._maximum_bytes:
            raise CapabilityBrokerError("BROKER_REQUEST_TOO_LARGE")
        try:
            completed = capture_process(
                [str(executable), *arguments],
                stdin=stdin,
                maximum_bytes=self._maximum_bytes,
                timeout=self._timeout,
                environment={"PATH": "", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
                resource_policy=self._resource_policy,
            )
        except ValueError as error:
            self._receipt(
                "process.spawn", "denied", resource, issue_codes=["BROKER_RESPONSE_TOO_LARGE"]
            )
            raise CapabilityBrokerError("BROKER_RESPONSE_TOO_LARGE") from error
        except (OSError, subprocess.TimeoutExpired, PluginProcessError) as error:
            self._receipt(
                "process.spawn", "failed", resource, issue_codes=["BROKER_PROCESS_FAILED"]
            )
            raise CapabilityBrokerError("BROKER_PROCESS_FAILED") from error
        result = BrokerProcessResult(
            returncode=completed.returncode,
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
        )
        self._receipt(
            "process.spawn",
            "accepted" if completed.returncode == 0 else "failed",
            resource,
            byte_count=len(completed.stdout) + len(completed.stderr),
            issue_codes=[] if completed.returncode == 0 else ["BROKER_PROCESS_NONZERO"],
        )
        return result


class CapabilityBrokerHostServices:
    """Bounded JSON-RPC facade exposed to an isolated MCP child process."""

    def __init__(self, broker: ScopedCapabilityBroker) -> None:
        """Bind all reverse RPC operations to one plugin's existing core-owned grant."""
        self._broker = broker

    def _bytes(self, value: object, *, field: str) -> bytes:
        """Reject invalid/oversized Base64 before allocating decoded request bytes."""
        if not isinstance(value, str) or len(value) > 4 * ((self._broker._maximum_bytes + 2) // 3):
            raise CapabilityBrokerError(f"BROKER_{field.upper()}_INVALID")
        try:
            result = base64.b64decode(value, validate=True)
            if len(result) > self._broker._maximum_bytes:
                raise ValueError("decoded request too large")
            return result
        except (ValueError, TypeError) as error:
            raise CapabilityBrokerError(f"BROKER_{field.upper()}_INVALID") from error

    def __call__(self, method: str, params: dict[str, object]) -> object:
        """Decode the strict RPC envelope without coercing numbers/booleans to paths or methods."""
        shapes = {
            "dronedream/filesystem/read": ({"root", "path"}, set()),
            "dronedream/filesystem/write": ({"root", "path", "body_base64"}, set()),
            "dronedream/network/request": (
                {"url"},
                {
                    "http_method",
                    "headers",
                    "body_base64",
                    "credential_reference",
                    "credential_header",
                    "credential_prefix",
                },
            ),
            "dronedream/process/spawn": ({"executable_id"}, {"arguments", "stdin_base64"}),
        }
        if not isinstance(method, str) or method not in shapes:
            raise CapabilityBrokerError("BROKER_METHOD_DENIED")
        try:
            params = copy_json(params, limit=self._broker._maximum_bytes)
        except ValueError as error:
            raise CapabilityBrokerError("BROKER_PARAMETERS_INVALID") from error
        required, optional = shapes[method]
        if (
            not isinstance(params, dict)
            or not required <= params.keys()
            or params.keys() - required - optional
        ):
            raise CapabilityBrokerError("BROKER_PARAMETERS_INVALID")

        def text(name, default=""):
            """Keep explicit invalid values distinct from omitted optional fields."""
            value = params.get(name, default)
            if not isinstance(value, str):
                raise CapabilityBrokerError("BROKER_PARAMETERS_INVALID")
            return value

        if method == "dronedream/filesystem/read":
            value = self._broker.read_bytes(text("root"), text("path"))
            return {"body_base64": base64.b64encode(value).decode("ascii")}
        if method == "dronedream/filesystem/write":
            self._broker.write_bytes(
                text("root"),
                text("path"),
                self._bytes(params.get("body_base64"), field="body_base64"),
            )
            return {"accepted": True}
        if method == "dronedream/network/request":
            headers = params.get("headers", {})
            if not isinstance(headers, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
            ):
                raise CapabilityBrokerError("BROKER_NETWORK_HEADERS_INVALID")
            body_value = params.get("body_base64", "")
            response = self._broker.request(
                text("http_method", "GET"),
                text("url"),
                headers=headers,
                body=self._bytes(body_value, field="body_base64"),
                credential_reference=(
                    text("credential_reference")
                    if params.get("credential_reference") is not None
                    else None
                ),
                credential_header=text("credential_header", "Authorization"),
                credential_prefix=text("credential_prefix", "Bearer "),
            )
            return {
                "status": response.status,
                "headers": response.headers,
                "body_base64": base64.b64encode(response.body).decode("ascii"),
            }
        if method == "dronedream/process/spawn":
            arguments = params.get("arguments", [])
            if not isinstance(arguments, list) or not all(
                isinstance(value, str) for value in arguments
            ):
                raise CapabilityBrokerError("BROKER_ARGUMENTS_INVALID")
            stdin_value = params.get("stdin_base64", "")
            result = self._broker.spawn(
                text("executable_id"),
                arguments,
                stdin=self._bytes(stdin_value, field="stdin_base64"),
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        raise CapabilityBrokerError("BROKER_METHOD_DENIED")
