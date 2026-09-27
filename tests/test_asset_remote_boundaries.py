"""Remote imports are tested with inert local streams/checkouts, never real network or Git."""

import io
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_app import asset_remote_sources as remote
from dronedream_agent_app.storage import AssetImportError


# 功能：
#   提供固定公网解析结果，保留真实 URL 校验但不访问 DNS 或外部服务。
# 输入：
#   monkeypatch：替换系统解析器的测试工具。
# 输出：
#   None：不返回业务数据。
@pytest.fixture
def public_url(monkeypatch):
    monkeypatch.setattr(
        remote.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )


class Body(io.BytesIO):
    headers = {}

    # 功能：
    #   返回内存响应对应的测试 URL。
    # 输入：
    #   self：响应体夹具。
    # 输出：
    #   url：测试来源地址。
    def geturl(self):
        url = "https://example.invalid/source.sdf"
        return url


# 功能：
#   验证交给导入器的文件已写完最后一块字节，且网络响应已关闭。
# 输入：
#   public_url：固定公网解析夹具。
#   monkeypatch：替换网络入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_download_handoff_contains_all_bytes_and_has_closed_network(public_url, monkeypatch):
    body = Body(b"small final buffered chunk")
    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *_: SimpleNamespace(open=lambda *_a, **_k: body)
    )
    with remote.RemoteAssetSourceService().acquire(
        source_type="direct_url", location=body.geturl()
    ) as (path, _name):
        assert path.read_bytes() == b"small final buffered chunk"
        assert body.closed
    assert not path.exists()


def test_download_preserves_reviewed_compound_map_suffix(monkeypatch):
    body = Body(b"name: clinic\nlevels: {}\n")
    body.geturl = lambda: "https://public.example/clinic.building.yaml"
    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *_: SimpleNamespace(open=lambda *_a, **_k: body)
    )
    monkeypatch.setattr(
        remote,
        "_validated_https_url",
        lambda _url: urllib.parse.urlsplit("https://public.example/clinic.building.yaml"),
    )
    with remote.RemoteAssetSourceService().acquire(
        source_type="direct_url",
        location="https://public.example/clinic.building.yaml",
    ) as (path, name):
        assert path.name.casefold().endswith(".building.yaml")
        assert name == "clinic.building.yaml"


# 功能：
#   验证拒绝不安全跳转时关闭原响应体，不遗留未消费连接。
# 输入：
#   monkeypatch：替换跳转验证结果的测试工具。
# 输出：
#   None：不返回业务数据。
def test_redirect_rejection_closes_unconsumed_body(monkeypatch):
    body = Body(b"redirect")
    monkeypatch.setattr(
        remote,
        "_validated_https_url",
        lambda _url: (_ for _ in ()).throw(AssetImportError("denied")),
    )
    with pytest.raises(AssetImportError):
        remote._SafeRedirectHandler().redirect_request(
            None, body, 302, "", {}, "http://bad.invalid"
        )
    assert body.closed


# 功能：
#   验证 Git 获取禁止继承配置、钩子、重定向及其他传输，并清理自有工作树。
# 输入：
#   public_url：固定公网解析夹具。
#   monkeypatch：替换 Git 捕获入口与进程环境的测试工具。
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_git_uses_isolated_configuration_and_no_redirects(public_url, monkeypatch, tmp_path):
    workspace = tmp_path / "git-stage"
    workspace.mkdir()
    monkeypatch.setattr(remote.tempfile, "mkdtemp", lambda **_kwargs: str(workspace))
    monkeypatch.setattr(remote.shutil, "which", lambda _name: "git.exe")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "untrusted")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "untrusted")
    monkeypatch.setenv("GIT_EXEC_PATH", "untrusted")
    observed = []

    # 功能：
    #   创建无害工作树，同时记录实际传给捕获器的命令与环境。
    # 输入：
    #   command：Git 命令行。
    #   kwargs：捕获器配置。
    # 输出：
    #   completed：模拟克隆成功的退出结果。
    def clone(command, **kwargs):
        observed.append((command, kwargs))
        checkout = workspace / "checkout"
        checkout.mkdir()
        (checkout / "map.sdf").write_bytes(b"map")
        completed = SimpleNamespace(returncode=0)
        return completed

    monkeypatch.setattr(remote, "capture_process", clone)
    with remote.RemoteAssetSourceService().acquire(
        source_type="git", location="https://example.invalid/repo.git"
    ) as (archive, _):
        assert archive.is_file()
        command, kwargs = observed[0]
        assert "http.followRedirects=false" in command
        assert "protocol.allow=never" in command
        assert "protocol.https.allow=always" in command
        assert "credential.helper=" in command
        assert "GIT_CONFIG_PARAMETERS" not in kwargs["environment"]
        assert "GIT_CONFIG_COUNT" not in kwargs["environment"]
        assert "GIT_CONFIG_KEY_0" not in kwargs["environment"]
        assert "GIT_CONFIG_VALUE_0" not in kwargs["environment"]
        assert "GIT_EXEC_PATH" not in kwargs["environment"]
    assert not workspace.exists()


def test_git_cleanup_removes_readonly_entries(tmp_path):
    workspace = tmp_path / "readonly-git-stage"
    workspace.mkdir()
    packed = workspace / "pack-file"
    packed.write_bytes(b"git object")
    packed.chmod(0o444)
    remote.shutil.rmtree(workspace, onerror=remote._remove_readonly_git_entry)
    assert not workspace.exists()


def test_git_commit_ref_is_fetched_and_checked_out_detached(public_url, monkeypatch, tmp_path):
    workspace = tmp_path / "git-commit-stage"
    workspace.mkdir()
    monkeypatch.setattr(remote.tempfile, "mkdtemp", lambda **_kwargs: str(workspace))
    monkeypatch.setattr(remote.shutil, "which", lambda _name: "git.exe")
    observed = []
    commit = "7851a5792d19a037833292a3e2a823b0f9e0c111"

    def run(command, **_kwargs):
        observed.append(command)
        checkout = workspace / "checkout"
        checkout.mkdir(exist_ok=True)
        (checkout / "clinic.building.yaml").write_text(
            "name: clinic\nlevels: {}\n", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote, "capture_process", run)
    with remote.RemoteAssetSourceService().acquire(
        source_type="git",
        location="https://example.invalid/repo.git",
        git_ref=commit,
        subpath="clinic.building.yaml",
    ) as (archive, _):
        assert archive.is_file()

    assert len(observed) == 3
    assert "--branch" not in observed[0]
    assert observed[1][-6:] == [
        "fetch",
        "--depth",
        "1",
        "--no-tags",
        "https://example.invalid/repo.git",
        commit,
    ]
    assert observed[2][-3:] == ["checkout", "--detach", commit]


# 功能：
#   验证控制字符和零端口在域名解析前被拒绝。
# 输入：
#   monkeypatch：禁止解析器被调用的测试工具。
#   url：非法来源地址。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "url", ["https://example.invalid/\npath", "https://example.invalid:0/path"]
)
def test_invalid_url_is_rejected_before_dns(monkeypatch, url):
    monkeypatch.setattr(
        remote.socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("Invalid URL reached DNS")
    )
    with pytest.raises(AssetImportError):
        remote._validated_https_url(url)


# 功能：
#   验证空目录也消耗枚举预算，不能等排序后才检查条目总量。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：降低枚举预算的测试工具。
# 输出：
#   None：不返回业务数据。
def test_git_tree_enumeration_caps_directories_before_sorting(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "MAX_GIT_SOURCE_FILES", 2)
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    with pytest.raises(AssetImportError, match="FILE_LIMIT_EXCEEDED"):
        remote._git_source_files(tmp_path, tmp_path)


# 功能：
#   验证源文件按规范顺序枚举且不包含 Git 自身配置。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_git_tree_is_deterministic_and_omits_repository_metadata(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("not an asset", encoding="utf-8")
    (tmp_path / "z").write_bytes(b"z")
    (tmp_path / "a").write_bytes(b"a")
    assert remote._git_source_files(tmp_path, tmp_path) == [tmp_path / "a", tmp_path / "z"]


# 功能：
#   验证重解析点在打开目录内容前被拒绝。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：注入重解析属性的测试工具。
# 输出：
#   None：不返回业务数据。
def test_git_tree_rejects_reparse_points_without_opening_them(tmp_path, monkeypatch):
    original = Path.lstat

    # 功能：
    #   仅为测试根目录补上 Windows 重解析属性，其余 stat 结果保持真实。
    # 输入：
    #   path：被查询的文件路径。
    #   args：其余位置参数。
    #   kwargs：其余具名参数。
    # 输出：
    #   result：真实属性或带重解析标记的替身属性。
    def reparse(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == tmp_path:
            result = SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result

    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(AssetImportError, match="SPECIAL_FILE_FORBIDDEN"):
        remote._git_source_files(tmp_path, tmp_path)


# 功能：
#   验证驱动器路径、备用数据流及歧义子路径不会创建克隆工作树。
# 输入：
#   public_url：固定公网解析夹具。
#   monkeypatch：拦截暂存目录创建的测试工具。
#   subpath：非法的仓库子路径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "subpath",
    ["C:/elsewhere", "C:relative", "folder/file:stream", "folder/.. ", "a//b", "./a", "."],
)
def test_windows_alias_subpaths_are_rejected_before_cloning(public_url, monkeypatch, subpath):
    monkeypatch.setattr(
        remote.tempfile, "mkdtemp", lambda **_kwargs: pytest.fail("Invalid path created a checkout")
    )
    with (
        pytest.raises(AssetImportError, match="SUBPATH_INVALID"),
        remote.RemoteAssetSourceService().acquire(
            source_type="git", location="https://example.invalid/repo.git", subpath=subpath
        ),
    ):
        pytest.fail("Invalid subpath yielded a source")


# 功能：
#   验证导入器抛出的异常保持原身份，不能被下载层误包装成网络错误。
# 输入：
#   public_url：固定公网解析夹具。
#   monkeypatch：替换网络入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_download_errors_from_the_importer_are_not_relabeled(public_url, monkeypatch):
    body = Body(b"asset")
    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *_: SimpleNamespace(open=lambda *_a, **_k: body)
    )
    error = remote.urllib.error.URLError("importer-specific-error")
    with (
        pytest.raises(remote.urllib.error.URLError) as caught,
        remote.RemoteAssetSourceService().acquire(
            source_type="direct_url", location=body.geturl()
        ) as (path, _),
    ):
        raise error
    assert caught.value is error
    assert not path.exists()
