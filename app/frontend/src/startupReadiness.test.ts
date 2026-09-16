import { describe, expect, it } from "vitest";

import { evaluateStartupReadiness } from "./startupReadiness";
import type { Bootstrap, RuntimeStatus } from "./types";

function fixture(): Bootstrap {
  const asset_versions: Bootstrap["asset_versions"] = [
    {
      asset_id: "map",
      content_sha256: "a".repeat(64),
      kind: "map",
      maturity: "qualified",
      bundle_root: "map-root",
      manifest: {},
      asset_ir: { name: "School Map" },
      imported_at: "",
    },
    {
      asset_id: "vehicle",
      content_sha256: "b".repeat(64),
      kind: "vehicle",
      maturity: "qualified",
      bundle_root: "vehicle-root",
      manifest: {},
      asset_ir: { name: "Drone" },
      imported_at: "",
    },
  ];
  return {
    models: Array.from({ length: 7 }, (_, index) => ({
      id: `model-${index}`,
      model: `model-${index}`,
      label: `Model ${index}`,
      provider: "test",
      icon: "test",
      source: "default",
    })),
    threads: [],
    asset_import_jobs: [],
    asset_versions,
    asset_qualification_jobs: [],
    asset_source_adapters: [],
    plugins: [{ plugin_id: "plugin" } as Bootstrap["plugins"][number]],
    connector_credentials: [],
    settings: {
      locale: "zh-CN", theme: "system", update_channel: "stable", default_model_id: "model-0",
      memory_enabled: true, remember_task_preferences: true, remember_asset_choices: true,
      last_map_id: null, last_map_content_sha256: null,
      last_vehicle_id: null, last_vehicle_content_sha256: null,
      plugin_update_ring: "stable",
      plugin_governance: {} as Bootstrap["settings"]["plugin_governance"], plugin_marketplace_sources: [],
    },
  };
}

const readyRuntime: RuntimeStatus = {
  distribution: "DroneDreamRuntime",
  runtime_available: true,
  resources_ready: true,
  provisioned: true,
  issue: null,
};

describe("startup readiness", () => {
  it("returns exactly zero when Runtime setup is required", () => {
    expect(evaluateStartupReadiness(fixture(), { ...readyRuntime, provisioned: false }).progress).toBe(0);
  });

  it("stops at the last completed evidence checkpoint", () => {
    const bootstrap = fixture();
    bootstrap.asset_versions = bootstrap.asset_versions.filter((item) => item.kind !== "map");
    const result = evaluateStartupReadiness(bootstrap, readyRuntime);
    expect(result.progress).toBe(74);
    expect(result.issue).toBe("QUALIFIED_ASSETS_MISSING");
  });

  it("reaches one hundred only when every contract passes", () => {
    expect(evaluateStartupReadiness(fixture(), readyRuntime)).toMatchObject({ progress: 100, ready: true });
  });
});
