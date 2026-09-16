import { describe, expect, it } from "vitest";

import {
  containsHan,
  humanizeIdentifier,
  localizedAssetName,
  localizedCategory,
  localizedDynamicDescription,
  localizedDynamicLabel,
  localizedPluginDescription,
  localizedPluginName,
  localizedSpeechRecognitionError,
  localizedSlot,
  localizedSystemTerm,
  localeSafeError,
  normalizeLocale,
} from "./i18n";
import type { PluginEntry } from "./types";

const plugin: PluginEntry = {
  plugin_id: "dronedream.route-risk-evaluator",
  name: "路线风险评估",
  version: "1.0.0",
  authority: "orchestrator",
  enabled: true,
  builtin: true,
  description: "从距离、净空与能耗角度评估路线。",
  publisher: "DroneDream",
  runtime_kind: "python",
  status: "active",
  health: "healthy",
  removable: false,
  disable_allowed: true,
  slot_required: false,
  package_sha256: "abc",
  last_error: null,
  trust_status: "verified",
  trust_decision: {},
  update_ring: "stable",
  capabilities: [{
    capability_id: "planning.route-risk",
    kind: "evaluator",
    name: "路线风险",
    description: "路线风险评估",
    authority: "orchestrator",
    metadata: {},
  }],
  permissions: [],
  dependencies: [],
  placement: {
    category_id: "planning",
    category_label: "规划",
    slot_id: "planning.route-evaluator",
    slot_label: "路线评估器",
    activation_mode: "multiple",
    scope: "mission",
    failure_mode: "advisory",
    swap_policy: "next-mission",
    category_order: 1,
    slot_order: 1,
    plugin_order: 1,
    pipeline_order: 1,
    runs_after: [],
    runs_before: [],
  },
  manifest: {},
};

describe("bilingual UI localization", () => {
  it("normalizes supported and system locale values", () => {
    expect(normalizeLocale("en-GB")).toBe("en-US");
    expect(normalizeLocale("zh-Hans")).toBe("zh-CN");
  });

  it("keeps qualified default assets native in each language", () => {
    const asset = { asset_id: "dronedream.school-map.v1", name: "School Map" } as Parameters<typeof localizedAssetName>[0];
    expect(localizedAssetName(asset, "zh-CN")).toBe("校园地图");
    expect(localizedAssetName(asset, "en-US")).toBe("School Map");
  });

  it("removes Chinese from dynamic plugin metadata in English mode", () => {
    const values = [
      localizedPluginName(plugin, "en-US"),
      localizedPluginDescription(plugin, "en-US"),
      localizedCategory(plugin.placement.category_id, plugin.placement.category_label, "en-US"),
      localizedSlot(plugin.placement.slot_id, plugin.placement.slot_label, "en-US"),
      localizedDynamicLabel("运行状态", "runtime-status", "en-US"),
    ];
    expect(values.every((value) => !containsHan(value))).toBe(true);
    expect(localizedDynamicDescription("中文说明", "en-US")).toBeUndefined();
  });

  it("preserves Chinese metadata in Chinese mode and formats technical identifiers", () => {
    expect(localizedPluginName(plugin, "zh-CN")).toBe("路线风险评估");
    expect(localizedPluginDescription(plugin, "zh-CN")).toContain("路线");
    expect(humanizeIdentifier("runtime.ros2-px4-bridge")).toBe("ROS 2 PX4 Bridge");
    expect(localizedDynamicDescription("English-only description", "zh-CN")).toBeUndefined();
  });

  it("keeps dynamic errors in the selected interface language", () => {
    const fallback = {
      zh: "环境检查失败",
      en: "Environment check failed",
    };
    expect(localeSafeError("后端暂时不可用", "en-US", fallback)).toBe(fallback.en);
    expect(localeSafeError("Backend is temporarily unavailable", "zh-CN", fallback)).toBe(fallback.zh);
    expect(localeSafeError("RUNTIME_SETUP_FAILED", "zh-CN", fallback)).toBe("环境检查失败（RUNTIME_SETUP_FAILED）");
    expect(localeSafeError("RUNTIME_SETUP_FAILED", "en-US", fallback)).toBe("Environment check failed (RUNTIME_SETUP_FAILED)");
  });

  it("localizes plugin policy terms and browser speech errors", () => {
    expect(localizedSystemTerm("fail-closed", "zh-CN")).toBe("失败时关闭");
    expect(localizedSystemTerm("next-mission", "en-US")).toBe("Next mission");
    expect(localizedSystemTerm("trust-local-package", "zh-CN")).toBe("批准本地包");
    expect(localizedSpeechRecognitionError("not-allowed", "zh-CN")).toBe("麦克风权限未开启");
    expect(localizedSpeechRecognitionError("network", "en-US")).toBe("The speech recognition network is unavailable");
  });
});
