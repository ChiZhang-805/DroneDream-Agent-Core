import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Session } from "@supabase/supabase-js";

const { from, upload, updateUser, getPublicUrl } = vi.hoisted(() => ({
  from: vi.fn(),
  upload: vi.fn(),
  updateUser: vi.fn(),
  getPublicUrl: vi.fn(),
}));

vi.mock("@supabase/supabase-js", () => ({
  createClient: vi.fn(() => ({
    auth: { updateUser },
    storage: {
      from,
    },
  })),
}));

import { accountAvatarUploadError, jpegDataUrlToBlob, loadAccountOverview, updateAccountAvatar } from "./auth";

describe("shared account avatar", () => {
  beforeEach(() => {
    upload.mockReset().mockResolvedValue({ error: null });
    updateUser.mockReset().mockResolvedValue({ error: null });
    getPublicUrl.mockReset().mockReturnValue({
      data: { publicUrl: "https://assets.example/profile-avatars/user-1/avatar.jpg" },
    });
    from.mockReset().mockReturnValue({ upload, getPublicUrl });
    vi.stubGlobal("fetch", vi.fn(() => { throw new Error("CSP_BLOCKED_DATA_FETCH"); }));
  });

  it("decodes the cropped JPEG locally without a data-URL fetch", () => {
    const blob = jpegDataUrlToBlob("data:image/jpeg;base64,YXZhdGFy");

    expect(blob.type).toBe("image/jpeg");
    expect(blob.size).toBe(6);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("uploads to the signed-in user's folder before publishing avatar_url", async () => {
    const session = { user: { id: "user-1" } } as Session;

    const avatarUrl = await updateAccountAvatar(session, "data:image/jpeg;base64,YXZhdGFy");

    expect(upload).toHaveBeenCalledWith(
      "user-1/avatar.jpg",
      expect.any(Blob),
      expect.objectContaining({ contentType: "image/jpeg", upsert: true }),
    );
    expect(updateUser).toHaveBeenCalledWith({
      data: { avatar_url: expect.stringContaining("/user-1/avatar.jpg?v=") },
    });
    expect(from).toHaveBeenCalledWith("profile-avatars");
    expect(avatarUrl).toContain("/user-1/avatar.jpg?v=");
    expect(fetch).not.toHaveBeenCalled();
  });

  it("keeps permission failures distinct from network failures", () => {
    expect(accountAvatarUploadError({ statusCode: "403", message: "new row violates row-level security" }).message)
      .toBe("AVATAR_UPLOAD_FORBIDDEN");
    expect(accountAvatarUploadError(new TypeError("Failed to fetch")).message)
      .toBe("AVATAR_UPLOAD_NETWORK");
  });

  it("uses the authoritative gateway daily history instead of querying protected usage rows", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({
        data: {
          plan: { id: "pro", name: "Pro", monthly_price_cny_fen: 19_900, included_ai_credits: 15_000_000, capability_set: "pro-v1" },
          period: { starts_at: "2026-08-01T08:12:00Z", ends_at: "2026-08-31T08:12:00Z" },
          usage: { reserved_ai_credits: 0, consumed_ai_credits: 120, remaining_ai_credits: 14_999_880, request_count: 1, input_tokens: 20, output_tokens: 10, total_tokens: 30, estimated_request_count: 0, credit_policy_version: 1 },
          daily_usage: [{ date: "2026-08-22", consumed_ai_credits: 120, request_count: 1, input_tokens: 20, output_tokens: 10, total_tokens: 30 }],
        },
      }),
    })));
    const session = {
      access_token: "session-token",
      user: { id: "user-1", email: "pilot@example.test", user_metadata: { display_name: "Pilot" } },
    } as unknown as Session;

    const overview = await loadAccountOverview(session);

    expect(overview.dailyUsage).toEqual([{ date: "2026-08-22", aiCredits: 120, totalTokens: 30, requestCount: 1 }]);
    expect(overview.dailyUsageComplete).toBe(true);
    expect(from).not.toHaveBeenCalledWith("model_usage_requests");
  });
});
