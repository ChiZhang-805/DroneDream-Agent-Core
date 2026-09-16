import { invoke } from "@tauri-apps/api/core";
import { currentSession, currentSupabasePublishableKey } from "./auth";
import type { AssetImportJob, AssetIssueReport, AssetQualificationEvidence, AssetQualificationJob, Bootstrap, CustomModelDiscovery, Message, ModelEntry, RuntimeSetupSnapshot, RuntimeStatus, TaskThread } from "./types";

interface BackendInfo {
  base_url: string;
  token: string;
}

let cachedInfo: BackendInfo | null = null;

async function backendInfo(): Promise<BackendInfo> {
  if (cachedInfo) return cachedInfo;
  try {
    cachedInfo = await invoke<BackendInfo>("backend_info");
  } catch {
    const base = import.meta.env.VITE_LOCAL_CORE_URL;
    const token = import.meta.env.VITE_LOCAL_CORE_TOKEN;
    if (!base || !token) throw new Error("LOCAL_CORE_UNAVAILABLE");
    cachedInfo = { base_url: base, token };
  }
  return cachedInfo;
}

export async function localRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const info = await backendInfo();
  const identity = await currentSession();
  const publishableKey = currentSupabasePublishableKey();
  const response = await fetch(`${info.base_url}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${info.token}`,
      ...(identity ? { "X-DroneDream-Identity-Token": identity.access_token } : {}),
      ...(identity && publishableKey
        ? { "X-DroneDream-Supabase-Publishable-Key": publishableKey }
        : {}),
      ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init.headers ?? {}),
    },
  });
  const parsed = await response.json().catch(() => ({})) as { detail?: string };
  if (!response.ok) throw new Error(parsed.detail ?? `HTTP_${response.status}`);
  return parsed as T;
}

export const appApi = {
  bootstrap: () => localRequest<Bootstrap>("/v1/bootstrap"),
  createThread: (selectedModel: string, title?: string, locale: "zh-CN" | "en-US" = "zh-CN") => localRequest<TaskThread>("/v1/threads", {
    method: "POST",
    body: JSON.stringify({
      title: title ?? (locale === "en-US" ? "New task" : "新任务"),
      selected_model: selectedModel,
      locale,
    }),
  }),
  getThread: (id: string) => localRequest<TaskThread>(`/v1/threads/${id}`),
  patchThread: (id: string, patch: Record<string, unknown>) => localRequest<TaskThread>(`/v1/threads/${id}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  }),
  appendMessage: (id: string, content: string) => localRequest<Message>(`/v1/threads/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({ content, role: "user", kind: "text", metadata: {} }),
  }),
  uploadAttachment: (id: string, file: File) => {
    const body = new FormData();
    body.append("attachment", file);
    return localRequest<{ attachment_id: string; display_name: string }>(`/v1/threads/${id}/attachments`, {
      method: "POST",
      body,
    });
  },
  createAssetImportJob: (kind: "map" | "vehicle", file: File) => {
    const body = new FormData();
    body.append("source_format", "auto");
    body.append("expected_kind", kind);
    body.append("bundle", file);
    return localRequest<AssetImportJob>("/v1/asset-import-jobs", { method: "POST", body });
  },
  createRemoteAssetImportJob: (kind: "map" | "vehicle", payload: {
    source_type: "direct_url" | "git";
    location: string;
    expected_sha256?: string;
    git_ref?: string;
    subpath?: string;
  }) => localRequest<AssetImportJob>("/v1/asset-import-jobs/remote", {
    method: "POST",
    body: JSON.stringify({ ...payload, source_format: "auto", expected_kind: kind }),
  }),
  processAssetImportJob: (jobId: string) => localRequest<AssetImportJob>(`/v1/asset-import-jobs/${jobId}/process`, { method: "POST" }),
  getAssetImportJobIssues: (jobId: string) => localRequest<AssetIssueReport>(`/v1/asset-import-jobs/${jobId}/issues`),
  submitAssetImportCompanionResult: (job: AssetImportJob, file: File) => {
    if (!job.package_sha256 || !job.source_adapter_id) throw new Error("ASSET_IMPORT_COMPANION_BINDING_MISSING");
    const body = new FormData();
    body.append("source_package_sha256", job.package_sha256);
    body.append("adapter_id", job.source_adapter_id);
    body.append("result", file);
    return localRequest<AssetImportJob>(`/v1/asset-import-jobs/${job.job_id}/companion-result`, { method: "POST", body });
  },
  cancelAssetImportJob: (jobId: string) => localRequest<AssetImportJob>(`/v1/asset-import-jobs/${jobId}/cancel`, { method: "POST" }),
  createAssetQualificationJob: (payload: { map_asset_id: string; map_content_sha256: string; vehicle_asset_id: string; vehicle_content_sha256: string }) => localRequest<AssetQualificationJob>("/v1/asset-qualification-jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  startAssetQualificationJob: (jobId: string) => localRequest<AssetQualificationJob>(`/v1/asset-qualification-jobs/${jobId}/start`, { method: "POST" }),
  pauseAssetQualificationJob: (jobId: string) => localRequest<AssetQualificationJob>(`/v1/asset-qualification-jobs/${jobId}/pause`, { method: "POST" }),
  cancelAssetQualificationJob: (jobId: string) => localRequest<AssetQualificationJob>(`/v1/asset-qualification-jobs/${jobId}/cancel`, { method: "POST" }),
  getAssetQualificationJobIssues: (jobId: string) => localRequest<AssetIssueReport>(`/v1/asset-qualification-jobs/${jobId}/issues`),
  getAssetQualificationEvidence: (jobId: string) => localRequest<AssetQualificationEvidence>(`/v1/asset-qualification-jobs/${jobId}/evidence`),
  importPlugin: (file: File) => {
    const body = new FormData();
    body.append("bundle", file);
    return localRequest("/v1/plugins/import", { method: "POST", body });
  },
  getPluginGovernance: () => localRequest<import("./types").PluginGovernanceOverview>("/v1/plugin-governance"),
  replacePluginGovernance: (policy: import("./types").PluginGovernancePolicy) => localRequest("/v1/plugin-governance", {
    method: "PUT",
    body: JSON.stringify(policy),
  }),
  getPluginMarketplace: () => localRequest<import("./types").PluginMarketplaceCatalog>("/v1/plugin-marketplace"),
  replacePluginMarketplaceSources: (sources: import("./types").PluginMarketplaceSource[]) => localRequest("/v1/plugin-marketplace/sources", {
    method: "PUT",
    body: JSON.stringify(sources),
  }),
  installMarketplacePlugin: (sourceId: string, pluginId: string, version: string) => localRequest("/v1/plugin-marketplace/install", {
    method: "POST",
    body: JSON.stringify({ source_id: sourceId, plugin_id: pluginId, version }),
  }),
  getPlugin: (id: string) => localRequest<import("./types").PluginDetail>(`/v1/plugins/${id}`),
  getPluginPanel: (id: string, threadId?: string | null) => localRequest<import("./types").PluginPanel>(`/v1/plugins/${id}/panel${threadId ? `?thread_id=${encodeURIComponent(threadId)}` : ""}`),
  setPlugin: (id: string, enabled: boolean) => localRequest(`/v1/plugins/${id}/${enabled ? "enable" : "disable"}`, { method: "POST" }),
  trustLocalPluginPackage: (id: string) => localRequest(`/v1/plugins/${id}/trust-local-package`, { method: "POST" }),
  trustLocalPluginVersion: (id: string, version: string) => localRequest(`/v1/plugins/${id}/versions/${encodeURIComponent(version)}/trust-local-package`, { method: "POST" }),
  revokePluginPackage: (id: string) => localRequest(`/v1/plugins/${id}/revoke-package`, { method: "POST" }),
  applyPluginProfile: (id: string) => localRequest(`/v1/plugins/${id}/apply-profile`, { method: "POST" }),
  checkPlugin: (id: string) => localRequest(`/v1/plugins/${id}/healthcheck`, { method: "POST" }),
  uninstallPlugin: (id: string) => localRequest(`/v1/plugins/${id}`, { method: "DELETE" }),
  rollbackPlugin: (id: string, version: string) => localRequest(`/v1/plugins/${id}/rollback`, {
    method: "POST",
    body: JSON.stringify({ version }),
  }),
  activatePlugin: (id: string, version: string) => localRequest(`/v1/plugins/${id}/activate`, {
    method: "POST",
    body: JSON.stringify({ version }),
  }),
  promotePlugin: (id: string, version: string) => localRequest(`/v1/plugins/${id}/promote`, {
    method: "POST",
    body: JSON.stringify({ version }),
  }),
  configurePlugin: (id: string, configuration: Record<string, unknown>) => localRequest(`/v1/plugins/${id}/configuration`, {
    method: "PATCH",
    body: JSON.stringify({ configuration }),
  }),
  createConnectorCredential: (payload: { display_name: string; secret: string; allowed_plugin_ids: string[] }) => localRequest<import("./types").ConnectorCredentialReference>("/v1/connector-credentials", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  deleteConnectorCredential: (reference: string) => localRequest(`/v1/connector-credentials/${encodeURIComponent(reference)}`, { method: "DELETE" }),
  patchSettings: (patch: Record<string, unknown>) => localRequest("/v1/settings", { method: "PATCH", body: JSON.stringify(patch) }),
  discoverCustomModels: (baseUrl: string, apiKey: string) => localRequest<CustomModelDiscovery>("/v1/custom-models/discover", {
    method: "POST",
    body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
  }),
  createCustomModel: (payload: Record<string, unknown>) => localRequest<Record<string, unknown>>("/v1/custom-models", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  testCustomModel: (profileId: string) => localRequest<Record<string, unknown>>(`/v1/custom-models/${profileId}/test`, { method: "POST" }),
  deleteCustomModel: (profileId: string) => localRequest<Record<string, unknown>>(`/v1/custom-models/${profileId}`, { method: "DELETE" }),
  issueCustomModelGrant: (model: ModelEntry, threadId: string) => {
    if (!model.profile_id) throw new Error("CUSTOM_MODEL_PROFILE_MISSING");
    return localRequest<{ grant: string; expires_at: string }>(`/v1/custom-models/${model.profile_id}/grants?thread_id=${encodeURIComponent(threadId)}`, { method: "POST" });
  },
  prepare: (id: string, payload: Record<string, unknown>) => localRequest<Record<string, unknown>>(`/v1/threads/${id}/prepare`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  runtimeStatus: () => localRequest<RuntimeStatus>("/v1/runtime/status"),
  provisionRuntime: () => localRequest<Record<string, unknown>>("/v1/runtime/provision", { method: "POST" }),
  runtimeSetup: () => localRequest<RuntimeSetupSnapshot>("/v1/runtime/setup"),
  startRuntimeSetup: () => localRequest<RuntimeSetupSnapshot>("/v1/runtime/setup", { method: "POST" }),
  execute: (id: string, payload: Record<string, unknown>) => localRequest<Record<string, unknown>>(`/v1/threads/${id}/execute`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  runtimeMessage: (id: string, text: string) => localRequest<Record<string, unknown>>(`/v1/threads/${id}/runtime-message`, {
    method: "POST",
    body: JSON.stringify({ text }),
  }),
  openPricing: async () => {
    try {
      await invoke("open_pricing_page");
    } catch {
      window.open("https://getdronedream.com/pricing/", "_blank", "noopener,noreferrer");
    }
  },
};
