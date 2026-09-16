"""Secure acquisition of remote simulation-asset sources.

Remote inputs are copied into an inert temporary file and then enter the same
quarantine and normalization pipeline as local uploads.  No downloaded code is
executed and Git submodules, hooks, LFS smudging, local transports and private
network targets are rejected during URL validation. HTTPS connections pin checked
addresses while retaining TLS hostname checks; Git receives the same checked
resolution and no ambient proxy. These application checks are not an OS sandbox.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath, PureWindowsPath

from dronedream_agent_core.pinned_https import PinnedHTTPSHandler
from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
)
from dronedream_agent_core.plugin_process import PluginProcessError
from dronedream_agent_core.process_capture import capture_process

from .storage import AssetImportError

MAX_REMOTE_SOURCE_BYTES = 8 * 1024 * 1024 * 1024
MAX_GIT_SOURCE_FILES = 100_000
_GIT_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,159}$")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


# 功能：
#   校验 HTTPS 地址并保留本次检查通过的公网解析结果，供实际连接固定使用。
# 输入：
#   value：远程资产 URL。
# 输出：
#   target：解析后的 URL 和已验证数值地址元组。
def _validated_https_target(value: str) -> tuple[urllib.parse.SplitResult, tuple[str, ...]]:
    if (
        not isinstance(value, str)
        or len(value) > 4_096
        or any(ord(character) <= 32 or character.isspace() for character in value)
    ):
        raise AssetImportError("ASSET_REMOTE_URL_INVALID")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise AssetImportError("ASSET_REMOTE_URL_INVALID") from error
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port == 0
        or "\\" in parsed.netloc
    ):
        raise AssetImportError("ASSET_REMOTE_HTTPS_REQUIRED")
    if "%" in parsed.hostname:
        raise AssetImportError("ASSET_REMOTE_HOST_INVALID")
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(
                    parsed.hostname, port or 443, type=socket.SOCK_STREAM
                )
            }
        except socket.gaierror as error:
            raise AssetImportError("ASSET_REMOTE_HOST_UNRESOLVED") from error
    else:
        # 数值地址已经给定最终目标；不经 DNS，也不接受需要网卡作用域的地址。
        addresses = {str(literal)}
    if not addresses or len(addresses) > 64:
        raise AssetImportError("ASSET_REMOTE_HOST_UNRESOLVED")
    for address in addresses:
        try:
            if "%" in address:
                raise ValueError("scoped address")
            ip = ipaddress.ip_address(address)
        except ValueError as error:
            raise AssetImportError("ASSET_REMOTE_HOST_INVALID") from error
        if not ip.is_global:
            raise AssetImportError("ASSET_REMOTE_PRIVATE_NETWORK_FORBIDDEN")
    target = parsed, tuple(sorted(addresses))
    return target


# 功能：
#   验证远程地址并返回 URL 结构，用于跳转检查和来源名称解析。
# 输入：
#   value：待校验的 HTTPS 地址。
# 输出：
#   parsed：通过地址与公网解析检查的 URL 结构。
def _validated_https_url(value: str) -> urllib.parse.SplitResult:
    parsed, _addresses = _validated_https_target(value)
    return parsed


class _AssetHTTPSHandler(urllib.request.HTTPSHandler):
    """Resolve and pin each initial or redirected HTTPS request independently."""

    # 功能：
    #   将每一跳请求的已验证地址绑定到实际 TLS 连接，连接时不重新解析主机。
    # 输入：
    #   self：资产下载专用 HTTPS 处理器。
    #   request：初始请求或经验证的跳转请求。
    # 输出：
    #   response：固定网络目标返回的响应。
    def https_open(self, request):
        parsed, addresses = _validated_https_target(request.full_url)
        handler = PinnedHTTPSHandler(
            parsed.hostname.lower().rstrip("."), addresses, port=parsed.port or 443
        )
        response = handler.https_open(request)
        return response


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate each redirect before opening its destination connection."""

    # 功能：
    #   验证跳转目标；拒绝跳转时主动关闭未消费的响应体，避免连接泄漏。
    # 输入：
    #   self：当前跳转处理器。
    #   request：发起跳转的请求。
    #   fp：原响应体。
    #   code：重定向状态码。
    #   msg：响应状态说明。
    #   headers：原响应头。
    #   newurl：准备跳转的新地址。
    # 输出：
    #   redirected：urllib 构造的新请求或不支持跳转时的 None。
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        try:
            _validated_https_url(newurl)
        except AssetImportError:
            if fp is not None:
                fp.close()
            raise
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        return redirected


# 功能：
#   从远程路径提取有限长度的展示名称，不把远程路径作为本机目标路径使用。
# 输入：
#   value：URL 路径或仓库名称。
#   fallback：提取不到有效名称时采用的默认名称。
# 输出：
#   filename：最长 180 个字符的清理后名称。
def _safe_filename(value: str, fallback: str) -> str:
    candidate = Path(urllib.parse.unquote(value).replace("\\", "/")).name
    candidate = _SAFE_FILENAME.sub("-", candidate).strip(".-")
    if not candidate:
        candidate = fallback
    filename = candidate[:180]
    return filename


# 功能：
#   核对可选的预期内容摘要；摘要一致不代表发布者可信或拥有代码执行权限。
# 输入：
#   value：调用方指定的预期摘要，可为空。
#   actual：实际下载或封装内容的摘要。
# 输出：
#   None：不返回业务数据。
def _validate_expected_sha256(value: str | None, actual: str) -> None:
    if value is not None and value != actual:
        raise AssetImportError("ASSET_REMOTE_HASH_MISMATCH")


# 功能：
#   有界枚举自有工作树中的普通资产文件，跳过 Git 元数据并拒绝链接与特殊文件。
#   目录同样计入条目预算，排序前完成限额检查；路径检查不替代敌对文件系统沙箱。
# 输入：
#   root：用户选定的工作树子路径。
#   checkout：本次获取创建的工作树根目录。
# 输出：
#   ordered：按 UTF-8 相对路径排序的普通文件列表。
def _git_source_files(root: Path, checkout: Path) -> list[Path]:
    if any(part.casefold() == ".git" for part in root.relative_to(checkout).parts):
        raise AssetImportError("ASSET_REMOTE_GIT_SUBPATH_INVALID")
    pending = [root]
    files: list[Path] = []
    entries = 0
    while pending:
        path = pending.pop()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise AssetImportError("ASSET_REMOTE_GIT_SPECIAL_FILE_FORBIDDEN")
        if stat.S_ISREG(metadata.st_mode):
            files.append(path)
        elif stat.S_ISDIR(metadata.st_mode):
            with os.scandir(path) as children:
                for entry in children:
                    if entry.name.casefold() == ".git":
                        continue
                    entries += 1
                    if entries > MAX_GIT_SOURCE_FILES:
                        raise AssetImportError("ASSET_REMOTE_GIT_FILE_LIMIT_EXCEEDED")
                    pending.append(Path(entry.path))
        else:
            raise AssetImportError("ASSET_REMOTE_GIT_SPECIAL_FILE_FORBIDDEN")
    ordered = sorted(files, key=lambda path: path.relative_to(checkout).as_posix().encode("utf-8"))
    return ordered


# 功能：
#   将普通工作树文件有界复制到 ZIP64 成员，拒绝复制期间的替换、增长或修改。
# 输入：
#   candidate：工作树中的源文件。
#   bundle：本次创建的 ZIP 写入器。
#   member：已验证的可移植成员路径。
#   budget：剩余源内容字节预算。
# 输出：
#   copied：实际复制的字节数。
def _copy_git_member(candidate: Path, bundle: zipfile.ZipFile, member: str, budget: int) -> int:
    check_plain_plugin_path(candidate)
    before = candidate.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > budget:
        raise AssetImportError("ASSET_REMOTE_SOURCE_TOO_LARGE")
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.file_size = before.st_size
    copied = 0
    with candidate.open("rb") as source:
        opened = os.fstat(source.fileno())
        if (
            not os.path.samestat(before, opened)
            or opened.st_size != before.st_size
            or opened.st_mtime_ns != before.st_mtime_ns
        ):
            raise AssetImportError("ASSET_REMOTE_GIT_SOURCE_CHANGED")
        # ZIP64 支持允许预算内的大成员；固定元数据保证同一内容可重复封装。
        with bundle.open(info, "w", force_zip64=True) as output:
            while chunk := source.read(min(1024 * 1024, min(budget, opened.st_size) - copied + 1)):
                copied += len(chunk)
                if copied > budget:
                    raise AssetImportError("ASSET_REMOTE_SOURCE_TOO_LARGE")
                if copied > opened.st_size:
                    raise AssetImportError("ASSET_REMOTE_GIT_SOURCE_CHANGED")
                output.write(chunk)
        after = os.fstat(source.fileno())
    check_plain_plugin_path(candidate)
    if (
        copied != opened.st_size
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or not os.path.samestat(after, candidate.stat())
    ):
        raise AssetImportError("ASSET_REMOTE_GIT_SOURCE_CHANGED")
    return copied


class RemoteAssetSourceService:
    """Acquire remote sources without widening the trusted runtime boundary."""

    # 功能：
    #   配置本服务暂存根目录；生产应用传入自身数据目录，不在构造时联网或创建文件。
    # 输入：
    #   self：当前远程资产获取服务。
    #   staging_root：应用管理的暂存根目录，为空时使用系统临时目录。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, staging_root: Path | None = None) -> None:
        self._staging_root = staging_root

    # 功能：
    #   创建并核对本次获取使用的暂存根目录，拒绝指向其他目录的链接。
    # 输入：
    #   self：持有暂存位置配置的服务。
    # 输出：
    #   root：已准备好的暂存根目录，为空时交由 tempfile 选取系统位置。
    def _staging_directory(self) -> Path | None:
        root = self._staging_root
        if root is not None:
            check_plain_plugin_path(root)
            root.mkdir(parents=True, exist_ok=True)
        return root

    # 功能：
    #   获取惰性的远程数据来源，在导入期间保留暂存材料，退出上下文后清理自有文件。
    # 输入：
    #   self：当前远程获取服务。
    #   source_type：direct_url 或 git。
    #   location：HTTPS 下载地址或仓库地址。
    #   expected_sha256：可选的预期内容摘要。
    #   git_ref：可选的分支或标签。
    #   subpath：可选的仓库资产子路径。
    # 输出：
    #   result：上下文存续期间可读取的暂存路径与展示名称。
    @contextmanager
    def acquire(
        self,
        *,
        source_type: str,
        location: str,
        expected_sha256: str | None = None,
        git_ref: str | None = None,
        subpath: str | None = None,
    ) -> Iterator[tuple[Path, str]]:
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise AssetImportError("ASSET_REMOTE_EXPECTED_HASH_INVALID")
        if source_type == "direct_url":
            if git_ref is not None or subpath is not None:
                raise AssetImportError("ASSET_REMOTE_GIT_OPTIONS_REQUIRE_GIT")
            with self._download(location, expected_sha256=expected_sha256) as result:
                yield result
            return
        if source_type == "git":
            with self._clone_git(
                location,
                expected_sha256=expected_sha256,
                git_ref=git_ref,
                subpath=subpath,
            ) as result:
                yield result
            return
        raise AssetImportError("ASSET_REMOTE_SOURCE_TYPE_INVALID")

    # 功能：
    #   流式下载并核对实际字节数和摘要，关闭网络后才将内容交给导入器。
    #   总预算在每次有界读取边界检查，不把套接字超时宣称为 DNS 的硬取消。
    # 输入：
    #   self：当前远程获取服务。
    #   location：HTTPS 资产地址。
    #   expected_sha256：可选的预期内容摘要。
    # 输出：
    #   result：已关闭写入流的暂存文件路径及展示名称。
    @contextmanager
    def _download(
        self,
        location: str,
        *,
        expected_sha256: str | None,
    ) -> Iterator[tuple[Path, str]]:
        parsed = _validated_https_url(location)
        suffix = Path(parsed.path).suffix.casefold()
        if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) is None:
            suffix = ".download"
        # 请求参数构造失败时还没有创建文件，不留下未纳入清理的描述符或暂存材料。
        request = urllib.request.Request(
            location,
            headers={"Accept": "application/octet-stream", "User-Agent": "DroneDream-AGENT/1"},
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="dd-remote-asset-", suffix=suffix, dir=self._staging_directory()
        )
        owned = os.fstat(descriptor)
        output = os.fdopen(descriptor, "wb")
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        deadline = time.monotonic() + 300
        try:
            try:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({}), _SafeRedirectHandler(), _AssetHTTPSHandler()
                )
                # 直接使用独占创建时的描述符，不关闭后按路径重新打开可被替换的文件。
                with output, opener.open(request, timeout=60) as response:
                    final_parsed = _validated_https_url(response.geturl())
                    declared = response.headers.get("Content-Length")
                    if declared is not None:
                        if not declared.isascii() or not declared.isdecimal() or len(declared) > 16:
                            raise AssetImportError("ASSET_REMOTE_CONTENT_LENGTH_INVALID")
                        if not 0 < int(declared) <= MAX_REMOTE_SOURCE_BYTES:
                            raise AssetImportError("ASSET_REMOTE_SOURCE_TOO_LARGE")
                    while True:
                        if time.monotonic() >= deadline:
                            raise AssetImportError("ASSET_REMOTE_DOWNLOAD_TIMEOUT")
                        reader = getattr(response, "read1", response.read)
                        chunk = reader(min(1024 * 1024, MAX_REMOTE_SOURCE_BYTES + 1 - size))
                        if time.monotonic() >= deadline:
                            raise AssetImportError("ASSET_REMOTE_DOWNLOAD_TIMEOUT")
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_REMOTE_SOURCE_BYTES:
                            raise AssetImportError("ASSET_REMOTE_SOURCE_TOO_LARGE")
                        digest.update(chunk)
                        output.write(chunk)
                    if size <= 0:
                        raise AssetImportError("ASSET_REMOTE_SOURCE_EMPTY")
                    if declared is not None and size != int(declared):
                        raise AssetImportError("ASSET_REMOTE_CONTENT_LENGTH_MISMATCH")
                    _validate_expected_sha256(expected_sha256, digest.hexdigest())
                    output.flush()
                    os.fsync(output.fileno())
                    name = _safe_filename(final_parsed.path, "remote-asset" + suffix)
            except urllib.error.HTTPError as error:
                error.close()
                raise AssetImportError(f"ASSET_REMOTE_HTTP_STATUS:{error.code}") from error
            except urllib.error.URLError as error:
                raise AssetImportError("ASSET_REMOTE_DOWNLOAD_FAILED") from error
            check_plain_plugin_path(temporary)
            if not os.path.samestat(owned, temporary.stat()):
                raise AssetImportError("ASSET_REMOTE_STAGING_REPLACED")
            # 导入器自己的异常不能被网络异常映射覆盖。
            result = temporary, name
            yield result
        finally:
            try:
                output.close()
            finally:
                with suppress(OSError, ValueError):
                    check_plain_plugin_path(temporary)
                    if os.path.samestat(owned, temporary.stat()):
                        temporary.unlink()

    # 功能：
    #   隔离 Git 配置并固定网络目标，将普通工作树内容封装为数据 ZIP，不执行下载代码。
    #   源文件与最终归档受字节预算约束；克隆期间的总磁盘占用不等同于该源内容预算。
    # 输入：
    #   self：当前远程获取服务。
    #   location：经公网验证的 HTTPS 仓库地址。
    #   expected_sha256：可选的最终 ZIP 摘要，不是 Git 提交标识。
    #   git_ref：可选的分支或标签。
    #   subpath：可选的仓库资产子路径。
    # 输出：
    #   result：上下文存续期间的 ZIP 路径与展示名称。
    @contextmanager
    def _clone_git(
        self,
        location: str,
        *,
        expected_sha256: str | None,
        git_ref: str | None,
        subpath: str | None,
    ) -> Iterator[tuple[Path, str]]:
        if git_ref is not None and (
            not isinstance(git_ref, str)
            or _GIT_REF.fullmatch(git_ref) is None
            or ".." in git_ref
            or "//" in git_ref
            or git_ref.endswith("/")
        ):
            raise AssetImportError("ASSET_REMOTE_GIT_REF_INVALID")
        normalized_subpath: PurePosixPath | None = None
        if subpath is not None:
            if not isinstance(subpath, str) or not subpath or PureWindowsPath(subpath).drive:
                raise AssetImportError("ASSET_REMOTE_GIT_SUBPATH_INVALID")
            try:
                normalized_subpath = PurePosixPath(portable_plugin_path(subpath.replace("\\", "/")))
            except ValueError as error:
                raise AssetImportError("ASSET_REMOTE_GIT_SUBPATH_INVALID") from error
        parsed, addresses = _validated_https_target(location)
        git = shutil.which("git")
        if git is None:
            raise AssetImportError("ASSET_REMOTE_GIT_UNAVAILABLE")
        workspace = Path(tempfile.mkdtemp(prefix="dd-remote-git-", dir=self._staging_directory()))
        check_plain_plugin_path(workspace)
        owned = workspace.stat()
        checkout = workspace / "checkout"
        archive = workspace / "source.zip"
        resolved_addresses = ",".join(
            f"[{address}]" if ":" in address else address for address in addresses
        )
        network_options: list[str] = []
        try:
            ipaddress.ip_address(parsed.hostname)
        except ValueError:
            network_options = [
                "-c",
                f"http.curloptResolve={parsed.hostname}:{parsed.port or 443}:{resolved_addresses}",
            ]
        # 数值 URL 自身固定目标，尤其不能将 IPv6 主机冒号拼成 DNS 配置分隔符。
        command = [
            git,
            "-C",
            str(workspace),
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.https.allow=always",
            "-c",
            "http.followRedirects=false",
            "-c",
            "http.proxy=",
            "-c",
            "http.sslVerify=true",
            *network_options,
            "-c",
            "credential.helper=",
            "-c",
            f"core.hooksPath={workspace / 'empty-hooks'}",
            "-c",
            "protocol.file.allow=never",
            "-c",
            "submodule.recurse=false",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-tags",
            f"--template={workspace / 'empty-template'}",
        ]
        if git_ref is not None:
            command.extend(["--branch", git_ref, "--single-branch"])
        command.extend([location, str(checkout)])
        environment = {
            # 继承代理会替应用重新解析地址；Git 配置环境也不能覆盖本次隔离策略。
            **{
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith("GIT_")
                and key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}
            },
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "",
            "SSH_ASKPASS": "",
            "GIT_CEILING_DIRECTORIES": str(workspace.parent),
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        try:
            (workspace / "empty-hooks").mkdir()
            (workspace / "empty-template").mkdir()
            try:
                completed = capture_process(
                    command,
                    stdin=b"",
                    maximum_bytes=1024 * 1024,
                    environment=environment,
                    timeout=300,
                    resource_policy=PluginResourcePolicy(
                        memory_limit_mb=1024, cpu_time_limit_seconds=300, process_limit=16
                    ),
                )
            except subprocess.TimeoutExpired as error:
                raise AssetImportError("ASSET_REMOTE_GIT_TIMEOUT") from error
            except (OSError, ValueError, PluginProcessError) as error:
                raise AssetImportError("ASSET_REMOTE_GIT_PROCESS_FAILED") from error
            if completed.returncode != 0:
                raise AssetImportError("ASSET_REMOTE_GIT_CLONE_FAILED")
            root = (
                checkout
                if normalized_subpath is None
                else checkout.joinpath(*normalized_subpath.parts)
            )
            # Reject a selected subpath through a link before resolve() hides it.
            selected = checkout
            for part in root.relative_to(checkout).parts:
                selected = selected / part
                if selected.exists() or selected.is_symlink():
                    metadata = selected.lstat()
                    if stat.S_ISLNK(metadata.st_mode) or (
                        getattr(metadata, "st_file_attributes", 0) & 0x400
                    ):
                        raise AssetImportError("ASSET_REMOTE_GIT_SPECIAL_FILE_FORBIDDEN")
            root = root.resolve()
            if checkout.resolve() != root and checkout.resolve() not in root.parents:
                raise AssetImportError("ASSET_REMOTE_GIT_SUBPATH_INVALID")
            if not root.exists():
                raise AssetImportError("ASSET_REMOTE_GIT_SUBPATH_NOT_FOUND")
            total_size = 0
            file_count = 0
            names: set[str] = set()
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
                candidates = _git_source_files(root, checkout)
                for candidate in candidates:
                    if any(
                        part.casefold() == ".git" for part in candidate.relative_to(checkout).parts
                    ):
                        continue
                    if candidate.is_symlink() or not candidate.is_file():
                        raise AssetImportError("ASSET_REMOTE_GIT_SPECIAL_FILE_FORBIDDEN")
                    relative = candidate.relative_to(root.parent if root.is_file() else root)
                    try:
                        member = portable_plugin_path(PurePosixPath(*relative.parts).as_posix())
                    except ValueError as error:
                        raise AssetImportError("ASSET_REMOTE_GIT_MEMBER_INVALID") from error
                    if member.casefold() in names:
                        raise AssetImportError("ASSET_REMOTE_GIT_MEMBER_COLLISION")
                    names.add(member.casefold())
                    file_count += 1
                    if file_count > MAX_GIT_SOURCE_FILES:
                        raise AssetImportError("ASSET_REMOTE_GIT_FILE_LIMIT_EXCEEDED")
                    total_size += _copy_git_member(
                        candidate, bundle, member, MAX_REMOTE_SOURCE_BYTES - total_size
                    )
            if file_count == 0:
                raise AssetImportError("ASSET_REMOTE_SOURCE_EMPTY")
            try:
                digest = hash_plugin_file(archive, limit=MAX_REMOTE_SOURCE_BYTES)
            except ValueError as error:
                raise AssetImportError("ASSET_REMOTE_ARCHIVE_INVALID_OR_TOO_LARGE") from error
            _validate_expected_sha256(expected_sha256, digest)
            repository_name = Path(parsed.path.rstrip("/")).stem.removesuffix(".git")
            result = archive, _safe_filename(repository_name, "git-asset") + ".zip"
            yield result
        finally:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(workspace)
                if os.path.samestat(owned, workspace.stat()):
                    shutil.rmtree(workspace)
