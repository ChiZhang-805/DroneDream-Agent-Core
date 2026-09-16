import type { Bootstrap, RuntimeStatus } from "./types";

export type StartupCheckId =
  | "core"
  | "resources"
  | "runtime"
  | "models"
  | "assets"
  | "plugins";

export interface StartupCheck {
  id: StartupCheckId;
  progress: number;
  passed: boolean;
  issue: string | null;
}

export interface StartupReadiness {
  progress: number;
  ready: boolean;
  requiresSetup: boolean;
  issue: string | null;
  checks: StartupCheck[];
}

export function evaluateStartupReadiness(
  bootstrap: Bootstrap,
  runtime: RuntimeStatus,
): StartupReadiness {
  if (!runtime.resources_ready || !runtime.runtime_available || !runtime.provisioned) {
    return {
      progress: 0,
      ready: false,
      requiresSetup: true,
      issue: runtime.issue,
      checks: [],
    };
  }

  const checks: StartupCheck[] = [
    { id: "core", progress: 18, passed: true, issue: null },
    { id: "resources", progress: 34, passed: runtime.resources_ready, issue: "RUNTIME_RESOURCES_MISSING" },
    { id: "runtime", progress: 58, passed: runtime.provisioned, issue: runtime.issue ?? "RUNTIME_PROVISION_REQUIRED" },
    { id: "models", progress: 74, passed: bootstrap.models.length >= 7, issue: "MODEL_CATALOG_INCOMPLETE" },
    {
      id: "assets",
      progress: 90,
      passed: bootstrap.asset_versions.some((item) => (
        ["map", "world"].includes(item.kind) && item.maturity === "qualified"
      )) && bootstrap.asset_versions.some((item) => (
        item.kind === "vehicle" && item.maturity === "qualified"
      )),
      issue: "QUALIFIED_ASSETS_MISSING",
    },
    { id: "plugins", progress: 100, passed: bootstrap.plugins.length > 0, issue: "PLUGIN_CATALOG_EMPTY" },
  ];
  const failed = checks.find((check) => !check.passed);
  if (failed) {
    const index = checks.indexOf(failed);
    return {
      progress: index === 0 ? 0 : checks[index - 1].progress,
      ready: false,
      requiresSetup: false,
      issue: failed.issue,
      checks,
    };
  }
  return { progress: 100, ready: true, requiresSetup: false, issue: null, checks };
}
