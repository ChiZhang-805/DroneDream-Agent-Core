from fastapi.testclient import TestClient

from dronedream_agent_app.airspace_service import AirspaceService
from dronedream_agent_app.identity import VerifiedIdentity
from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore


# 功能：
#   验证空间接口同时需要本机令牌和真实身份，不因其只读而放松认证。
# 输入：
#   tmp_path、monkeypatch：隔离测试目录和服务替身安装器。
# 输出：
#   None：断言认证失败不触发空间读取，合法请求才返回。
def test_spatial_api_requires_local_and_account_identity(tmp_path, monkeypatch):
    calls = []
    def snapshot(self, request):
        calls.append(request)
        return {"authority": "preference-only"}
    monkeypatch.setattr(AirspaceService, "snapshot", snapshot)
    identity = VerifiedIdentity(owner_account_id="test-owner", tenant_id=None,
                                organization_id=None, issuer="https://example.supabase.co/auth/v1",
                                expires_at=2_000_000_000)
    store = AppStore(tmp_path)
    with TestClient(create_app(store=store, token="a" * 64,
                              identity_verifier=lambda token: identity)) as client:
        payload = dict(map_asset_id="map-test", map_content_sha256="b" * 64,
                       vehicle_asset_id="vehicle-test", vehicle_content_sha256="c" * 64)
        url = "/v1/assets/preferred-airspace"
        assert client.post(url, json=payload).status_code == 401
        headers = {"Authorization": "Bearer " + "a" * 64}
        assert client.post(url, json=payload, headers=headers).status_code == 401
        assert not calls
        headers["X-DroneDream-Identity-Token"] = "test-session"
        assert client.post(url, json=payload, headers=headers).status_code == 200
        assert len(calls) == 1
        assert client.post(url, json={**payload, "map_content_sha256": "old"},
                           headers=headers).status_code == 422
    store.close()
