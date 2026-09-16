import { describe, expect, it } from "vitest";

import { runtimeBaseProgress, type RuntimeBaseInstallSnapshot } from "./runtimeBase";

function snapshot(
  phase: RuntimeBaseInstallSnapshot["phase"],
  patch: Partial<RuntimeBaseInstallSnapshot> = {},
): RuntimeBaseInstallSnapshot {
  return {
    operationId: "runtime-test",
    phase,
    bytesDownloaded: 0,
    bytesTotal: null,
    currentPart: null,
    totalParts: null,
    message: null,
    error: null,
    resumable: false,
    requiresRestart: false,
    targetRoot: "D:\\DroneDream",
    installedVersion: null,
    updatedAt: null,
    ...patch,
  };
}

describe("shared Runtime Base progress", () => {
  it("uses verified phase boundaries rather than elapsed time", () => {
    expect(runtimeBaseProgress(snapshot("verifyingManifest"))).toBe(5);
    expect(runtimeBaseProgress(snapshot("verifyingArchive"))).toBe(44);
    expect(runtimeBaseProgress(snapshot("completed"))).toBe(60);
  });

  it("credits only downloaded bytes inside the signed Runtime Base range", () => {
    expect(runtimeBaseProgress(snapshot("downloading", {
      bytesDownloaded: 25,
      bytesTotal: 100,
    }))).toBe(16.5);
    expect(runtimeBaseProgress(snapshot("downloading", {
      bytesDownloaded: 200,
      bytesTotal: 100,
    }))).toBe(42);
  });
});
