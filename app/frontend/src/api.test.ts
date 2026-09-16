import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn().mockResolvedValue({
    base_url: "http://127.0.0.1:43123",
    token: "test-local-token",
  }),
}));

vi.mock("./auth", () => ({
  currentSession: vi.fn().mockResolvedValue({ access_token: "verified-user-jwt" }),
  currentSupabasePublishableKey: vi.fn().mockReturnValue("sb_publishable_test_key"),
}));

import { appApi, localRequest } from "./api";
import type { AssetImportJob } from "./types";

const fetchMock = vi.fn();

function response(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

describe("local AGENT Core API", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  it("loads a fresh external-asset repository with the sidecar bearer token", async () => {
    const payload = {
      models: [],
      threads: [],
      asset_import_jobs: [],
      asset_versions: [],
      asset_qualification_jobs: [],
      asset_source_adapters: [],
      plugins: [],
      connector_credentials: [],
      settings: { locale: "zh-CN", theme: "light", update_channel: "stable" },
    };
    fetchMock.mockResolvedValue(response(payload));

    const bootstrap = await appApi.bootstrap();

    expect(bootstrap.asset_versions).toEqual([]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:43123/v1/bootstrap",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer test-local-token",
          "Content-Type": "application/json",
          "X-DroneDream-Identity-Token": "verified-user-jwt",
          "X-DroneDream-Supabase-Publishable-Key": "sb_publishable_test_key",
        }),
      }),
    );
  });

  it("serializes the selected model when a user starts a new task", async () => {
    fetchMock.mockResolvedValue(response({ thread_id: "thread-1" }));

    await appApi.createThread("deepseek-v4-flash");

    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://127.0.0.1:43123/v1/threads",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          title: "新任务",
          selected_model: "deepseek-v4-flash",
          locale: "zh-CN",
        }),
      }),
    );
  });

  it("starts DDPKG imports through the resumable quarantine API", async () => {
    fetchMock.mockResolvedValue(response({ job_id: "asset-job-0123456789abcdef01234567" }, true, 202));
    const file = new File(["package"], "campus.ddpkg", { type: "application/zip" });

    await appApi.createAssetImportJob("map", file);

    const request = fetchMock.mock.calls.at(-1)?.[1] as RequestInit;
    expect(fetchMock.mock.calls.at(-1)?.[0]).toBe("http://127.0.0.1:43123/v1/asset-import-jobs");
    expect(request.method).toBe("POST");
    expect(request.body).toBeInstanceOf(FormData);
    const body = request.body as FormData;
    expect(body.get("source_format")).toBe("auto");
    expect(body.get("expected_kind")).toBe("map");
    expect((body.get("bundle") as File).name).toBe("campus.ddpkg");
  });

  it("binds companion results to the source hash and detected adapter", async () => {
    fetchMock.mockResolvedValue(response({ state: "needs_input" }));
    const result = new File(["ddpkg"], "campus.ddpkg", { type: "application/zip" });
    const job = {
      job_id: "asset-job-0123456789abcdef01234567",
      package_sha256: "a".repeat(64),
      source_adapter_id: "blender.phobos",
    } as AssetImportJob;

    await appApi.submitAssetImportCompanionResult(job, result);

    const request = fetchMock.mock.calls.at(-1)?.[1] as RequestInit;
    expect(fetchMock.mock.calls.at(-1)?.[0]).toBe(
      "http://127.0.0.1:43123/v1/asset-import-jobs/asset-job-0123456789abcdef01234567/companion-result",
    );
    const body = request.body as FormData;
    expect(body.get("source_package_sha256")).toBe("a".repeat(64));
    expect(body.get("adapter_id")).toBe("blender.phobos");
    expect((body.get("result") as File).name).toBe("campus.ddpkg");
  });

  it("surfaces the structured backend error instead of hiding it behind an HTTP code", async () => {
    fetchMock.mockResolvedValue(response({ detail: "ASSET_NOT_QUALIFIED" }, false, 422));

    await expect(localRequest("/v1/bootstrap")).rejects.toThrow("ASSET_NOT_QUALIFIED");
  });
});
