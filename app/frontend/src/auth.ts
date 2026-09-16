import { createClient, type Session } from "@supabase/supabase-js";
import { invoke } from "@tauri-apps/api/core";
import type { AppLocale } from "./i18n";
import type { AccountOverview, DailyModelUsage, ManagedUsageSnapshot, ModelEntry } from "./types";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL?.replace(/\/+$/u, "") ?? "";
const publishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY ?? "";

export const authConfigured = Boolean(supabaseUrl && publishableKey);
export const supabase = createClient(supabaseUrl || "https://invalid.local", publishableKey || "missing", {
  auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
});

export function currentSupabasePublishableKey(): string {
  return authConfigured ? publishableKey : "";
}

export async function currentSession(): Promise<Session | null> {
  if (!authConfigured) return null;
  return (await supabase.auth.getSession()).data.session;
}

interface DesktopBrowserAuthSession {
  protocolVersion: "desktop-browser-auth-pkce-v1";
  editionId: "autonomy";
  authClientId: "dronedream-desktop-autonomy";
  accessToken: string;
  refreshToken: string;
}

export async function desktopBrowserAuthConfigured(): Promise<boolean> {
  try {
    return await invoke<boolean>("browser_auth_configured");
  } catch {
    return false;
  }
}

export async function signInWithDesktopBrowser(locale: AppLocale): Promise<Session> {
  if (!authConfigured) throw new Error("AUTH_NOT_CONFIGURED");
  const result = await invoke<DesktopBrowserAuthSession>("begin_browser_auth", {
    request: { locale },
  });
  if (
    result.protocolVersion !== "desktop-browser-auth-pkce-v1"
    || result.editionId !== "autonomy"
    || result.authClientId !== "dronedream-desktop-autonomy"
  ) {
    throw new Error("AUTONOMY_BROWSER_AUTH_IDENTITY_MISMATCH");
  }
  const adopted = await supabase.auth.setSession({
    access_token: result.accessToken,
    refresh_token: result.refreshToken,
  });
  if (adopted.error || !adopted.data.session) {
    throw new Error(adopted.error?.message || "AUTONOMY_BROWSER_AUTH_SESSION_REJECTED");
  }
  return adopted.data.session;
}

const ACCOUNT_AVATAR_BUCKET = "profile-avatars";
const MAX_ACCOUNT_AVATAR_BYTES = 1_048_576;

interface StorageFailureLike {
  message?: string;
  status?: number | string;
  statusCode?: number | string;
}

export function accountAvatarUploadError(reason: unknown): Error {
  const failure = (reason ?? {}) as StorageFailureLike;
  const message = typeof failure.message === "string" ? failure.message : "";
  const status = Number(failure.statusCode ?? failure.status ?? 0);
  if (status === 401) return new Error("AVATAR_UPLOAD_AUTH_REQUIRED");
  if (status === 403 || /row-level security|permission|forbidden/iu.test(message)) {
    return new Error("AVATAR_UPLOAD_FORBIDDEN");
  }
  if (status === 413 || /too large|maximum allowed size|payload/iu.test(message)) {
    return new Error("AVATAR_UPLOAD_TOO_LARGE");
  }
  if (/failed to fetch|network|load failed/iu.test(message)) {
    return new Error("AVATAR_UPLOAD_NETWORK");
  }
  return new Error(`AVATAR_UPLOAD_FAILED:${status || "UNKNOWN"}`);
}

export function jpegDataUrlToBlob(avatarDataUrl: string): Blob {
  const match = /^data:image\/jpeg;base64,([a-z0-9+/]+={0,2})$/iu.exec(avatarDataUrl);
  if (!match) throw new Error("AVATAR_FORMAT_INVALID");
  let decoded: string;
  try {
    decoded = globalThis.atob(match[1]);
  } catch {
    throw new Error("AVATAR_FORMAT_INVALID");
  }
  const bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index += 1) bytes[index] = decoded.charCodeAt(index);
  return new Blob([bytes], { type: "image/jpeg" });
}

export async function updateAccountAvatar(session: Session, avatarDataUrl: string): Promise<string> {
  const blob = jpegDataUrlToBlob(avatarDataUrl);
  if (!blob.size || blob.size > MAX_ACCOUNT_AVATAR_BYTES) {
    throw new Error("AVATAR_SIZE_INVALID");
  }
  const objectPath = `${session.user.id}/avatar.jpg`;
  try {
    const { error: uploadError } = await supabase.storage
      .from(ACCOUNT_AVATAR_BUCKET)
      .upload(objectPath, blob, {
        cacheControl: "3600",
        contentType: "image/jpeg",
        upsert: true,
      });
    if (uploadError) throw accountAvatarUploadError(uploadError);
  } catch (reason) {
    if (reason instanceof Error && reason.message.startsWith("AVATAR_UPLOAD_")) throw reason;
    throw accountAvatarUploadError(reason);
  }

  const publicUrl = supabase.storage.from(ACCOUNT_AVATAR_BUCKET).getPublicUrl(objectPath).data.publicUrl;
  const avatarUrl = `${publicUrl}?v=${Date.now()}`;
  const { error: metadataError } = await supabase.auth.updateUser({
    data: { avatar_url: avatarUrl },
  });
  if (metadataError) throw new Error(`AVATAR_METADATA_FAILED:${metadataError.message}`);
  return avatarUrl;
}

function modelGatewayEndpoint(): string {
  return import.meta.env.VITE_MODEL_GATEWAY_URL?.replace(/\/+$/u, "")
    ?? `${supabaseUrl}/functions/v1/model-gateway`;
}

function accountDisplayName(session: Session): string {
  const metadata = session.user.user_metadata;
  const candidate = metadata.display_name ?? metadata.full_name ?? metadata.name;
  return typeof candidate === "string" && candidate.trim()
    ? candidate.trim()
    : session.user.email?.split("@")[0] || "Pilot";
}

export function accountAvatar(session: Session): string | null {
  const metadata = session.user.user_metadata;
  const candidate = metadata.avatar_url ?? metadata.picture;
  return typeof candidate === "string"
    && (candidate.startsWith("https://") || candidate.startsWith("data:image/"))
    ? candidate
    : null;
}

async function gatewayRequest<T>(session: Session, path: string): Promise<T> {
  const response = await fetch(`${modelGatewayEndpoint()}${path}`, {
    headers: {
      Authorization: `Bearer ${session.access_token}`,
      Accept: "application/json",
    },
  });
  const envelope = await response.json().catch(() => ({})) as {
    data?: T;
    error?: { code?: string; message?: string };
  };
  if (!response.ok || envelope.data === undefined) {
    throw new Error(envelope.error?.message ?? envelope.error?.code ?? `HTTP_${response.status}`);
  }
  return envelope.data;
}

export async function loadAccountOverview(session: Session): Promise<AccountOverview> {
  const snapshot = await gatewayRequest<ManagedUsageSnapshot>(session, "/usage");
  const dailyUsage: DailyModelUsage[] = (snapshot.daily_usage ?? []).map((day) => ({
    date: day.date,
    aiCredits: day.consumed_ai_credits,
    totalTokens: day.total_tokens,
    requestCount: day.request_count,
  }));
  return {
    displayName: accountDisplayName(session),
    email: session.user.email ?? "",
    avatarUrl: accountAvatar(session),
    snapshot,
    dailyUsage,
    dailyUsageComplete: Array.isArray(snapshot.daily_usage),
  };
}

export async function issueModelGrant(session: Session, model: ModelEntry, threadId: string) {
  if (model.source !== "default") throw new Error("MANAGED_MODEL_REQUIRED");
  const endpoint = modelGatewayEndpoint();
  const response = await fetch(`${endpoint}/grants`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${session.access_token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      scope: "assistant",
      scope_reference: threadId,
      provider: model.provider,
      model: model.model,
    }),
  });
  const envelope = await response.json() as {
    data?: { grant: string; gateway_base_url: string; expires_at: string; max_calls: number };
    error?: { code?: string; message?: string };
  };
  if (!response.ok || !envelope.data) {
    throw new Error(envelope.error?.message ?? envelope.error?.code ?? "MODEL_GRANT_FAILED");
  }
  return envelope.data;
}
