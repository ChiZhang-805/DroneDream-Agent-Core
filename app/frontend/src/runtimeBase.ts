import { invoke } from "@tauri-apps/api/core";

export type RuntimeBaseInstallPhase =
  | "idle"
  | "queued"
  | "verifyingManifest"
  | "downloading"
  | "verifyingArchive"
  | "backingUp"
  | "importing"
  | "starting"
  | "healthChecking"
  | "restoring"
  | "waitingForRestart"
  | "completed"
  | "failed"
  | "cancelled";

export interface RuntimeBaseStatus {
  runtimeName: string;
  installed: boolean;
  running: boolean;
  ready: boolean;
  version: string | null;
  dataRoot: string | null;
  diagnostics: string[];
}

export interface RuntimeBaseInstallPlan {
  runtimeName: string;
  targetRoot: string;
  estimatedDownloadBytes: number;
  estimatedInstalledBytes: number;
  requiresAdministrator: boolean;
  requiresRestart: boolean;
  canInstall: boolean;
  blockers: string[];
}

export interface RuntimeBaseInstallError {
  code: string;
  message: string;
  retryable: boolean;
  diagnosticsPath: string | null;
}

export interface RuntimeBaseInstallSnapshot {
  operationId: string | null;
  phase: RuntimeBaseInstallPhase;
  bytesDownloaded: number;
  bytesTotal: number | null;
  currentPart: number | null;
  totalParts: number | null;
  message: string | null;
  error: RuntimeBaseInstallError | null;
  resumable: boolean;
  requiresRestart: boolean;
  targetRoot: string | null;
  installedVersion: string | null;
  updatedAt: string | null;
}

export function probeRuntimeBase(): Promise<RuntimeBaseStatus> {
  return invoke<RuntimeBaseStatus>("probe_runtime_status");
}

export function planRuntimeBaseInstall(): Promise<RuntimeBaseInstallPlan> {
  return invoke<RuntimeBaseInstallPlan>("get_runtime_install_plan", { targetRoot: null });
}

export function beginRuntimeBaseInstall(targetRoot: string): Promise<RuntimeBaseInstallSnapshot> {
  return invoke<RuntimeBaseInstallSnapshot>("start_runtime_install", {
    request: { targetRoot, releaseManifestUrl: null },
  });
}

export function runtimeBaseInstallProgress(): Promise<RuntimeBaseInstallSnapshot> {
  return invoke<RuntimeBaseInstallSnapshot>("get_runtime_install_progress");
}

export function startRuntimeBase(): Promise<RuntimeBaseStatus> {
  return invoke<RuntimeBaseStatus>("start_runtime");
}

export function runtimeBaseProgress(snapshot: RuntimeBaseInstallSnapshot): number {
  if (snapshot.phase === "downloading") {
    const ratio = snapshot.bytesTotal && snapshot.bytesTotal > 0
      ? Math.min(1, snapshot.bytesDownloaded / snapshot.bytesTotal)
      : 0;
    return 8 + ratio * 34;
  }
  const checkpoints: Record<RuntimeBaseInstallPhase, number> = {
    idle: 0,
    queued: 1,
    verifyingManifest: 5,
    downloading: 8,
    verifyingArchive: 44,
    backingUp: 47,
    importing: 50,
    starting: 55,
    healthChecking: 58,
    restoring: 54,
    waitingForRestart: 58,
    completed: 60,
    failed: 0,
    cancelled: 0,
  };
  return checkpoints[snapshot.phase];
}
