import { createContext, useContext, useEffect, useMemo } from "react";

import type { PluginEntry } from "./types";

export type AppLocale = "zh-CN" | "en-US";

const LOCALE_STORAGE_KEY = "dronedream.autonomy.locale";
const HAN_PATTERN = /[\u3400-\u9fff\uf900-\ufaff]/u;

export function normalizeLocale(value: string | null | undefined): AppLocale {
  return value?.toLowerCase().startsWith("en") ? "en-US" : "zh-CN";
}

export function initialLocale(): AppLocale {
  try {
    const stored = globalThis.localStorage?.getItem(LOCALE_STORAGE_KEY);
    if (stored) return normalizeLocale(stored);
  } catch {
    // Storage can be unavailable in hardened WebViews. System language remains a safe fallback.
  }
  return normalizeLocale(globalThis.navigator?.language);
}

export function rememberLocale(locale: AppLocale): void {
  try {
    globalThis.localStorage?.setItem(LOCALE_STORAGE_KEY, locale);
  } catch {
    // The backend setting remains authoritative when local storage is unavailable.
  }
}

type I18nContextValue = {
  locale: AppLocale;
  isEnglish: boolean;
  tr: (chinese: string, english: string) => string;
  number: (value: number) => string;
  compactNumber: (value: number) => string;
  dateTime: (value: string | number | Date) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ locale, children }: { locale: AppLocale; children: React.ReactNode }) {
  useEffect(() => {
    document.documentElement.lang = locale;
    document.title = "DroneDream · AGENT";
  }, [locale]);

  const value = useMemo<I18nContextValue>(() => ({
    locale,
    isEnglish: locale === "en-US",
    tr: (chinese, english) => locale === "en-US" ? english : chinese,
    number: (input) => new Intl.NumberFormat(locale).format(Math.max(0, Math.round(input))),
    compactNumber: (input) => new Intl.NumberFormat(locale, {
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(Math.max(0, input)),
    dateTime: (input) => new Intl.DateTimeFormat(locale, {
      year: "numeric",
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(new Date(input)),
  }), [locale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const value = useContext(I18nContext);
  if (!value) throw new Error("I18N_PROVIDER_MISSING");
  return value;
}

const UPPERCASE_TOKENS = new Map([
  ["ai", "AI"], ["api", "API"], ["bim", "BIM"], ["csv", "CSV"], ["ekf", "EKF"],
  ["ekf2", "EKF2"], ["gis", "GIS"], ["gnss", "GNSS"], ["gps", "GPS"], ["json", "JSON"],
  ["lidar", "LiDAR"], ["llm", "LLM"], ["mavlink", "MAVLink"], ["mcp", "MCP"], ["mpc", "MPC"],
  ["px4", "PX4"], ["qr", "QR"], ["rfid", "RFID"], ["ros", "ROS"], ["ros2", "ROS 2"],
  ["rtk", "RTK"], ["sdk", "SDK"], ["slam", "SLAM"], ["sql", "SQL"], ["tls", "TLS"],
  ["uav", "UAV"], ["ui", "UI"], ["uwb", "UWB"], ["vio", "VIO"], ["webots", "Webots"],
  ["xml", "XML"], ["yaml", "YAML"],
]);

export function humanizeIdentifier(value: string): string {
  const source = value.split(/[.:/]/u).filter(Boolean).at(-1) ?? value;
  return source
    .split(/[-_\s]+/u)
    .filter(Boolean)
    .map((token) => UPPERCASE_TOKENS.get(token.toLowerCase())
      ?? `${token.charAt(0).toUpperCase()}${token.slice(1)}`)
    .join(" ");
}

const CATEGORY_ENGLISH: Record<string, string> = {
  "harness": "Harness",
  "models": "Models",
  "planning": "Planning",
  "perception": "Perception",
  "navigation": "Navigation",
  "control": "Flight Control",
  "safety": "Safety",
  "simulation": "Simulation",
  "assets": "Maps and Vehicles",
  "tools": "Tools",
  "memory": "Context and Memory",
  "observability": "Observability",
  "interaction": "Interaction",
  "connectors": "Connectors",
  "payload": "Payload",
  "workflow": "Workflow",
  "evaluation": "Evaluation",
  "security": "Security",
  "interface": "Interface",
};

const ASSET_NAMES: Record<string, { zh: string; en: string }> = {
  "dronedream.school-map.v1": { zh: "校园地图", en: "School Map" },
  "dronedream.my-drone.v1": { zh: "我的无人机", en: "My Drone" },
};

export function localizedAssetName(item: { asset_id: string; name: string }, locale: AppLocale): string {
  const known = ASSET_NAMES[item.asset_id];
  if (known) return locale === "en-US" ? known.en : known.zh;
  if (locale === "en-US" && HAN_PATTERN.test(item.name)) return humanizeIdentifier(item.asset_id);
  return item.name;
}

export function localizedPluginName(item: Pick<PluginEntry, "plugin_id" | "name">, locale: AppLocale): string {
  if (locale === "zh-CN" || !HAN_PATTERN.test(item.name)) return item.name;
  return humanizeIdentifier(item.plugin_id);
}

export function localizedPluginDescription(item: Pick<PluginEntry, "description" | "placement" | "capabilities">, locale: AppLocale): string {
  const hasHan = HAN_PATTERN.test(item.description);
  if ((locale === "zh-CN" && hasHan) || (locale === "en-US" && !hasHan)) return item.description;
  const capability = item.capabilities[0]?.capability_id;
  if (locale === "en-US") {
    const subject = capability ? humanizeIdentifier(capability) : humanizeIdentifier(item.placement.slot_id);
    return `${subject} capability for the ${localizedSlot(item.placement.slot_id, item.placement.slot_label, locale)} slot.`;
  }
  const slot = localizedSlot(item.placement.slot_id, item.placement.slot_label, locale);
  return HAN_PATTERN.test(slot) ? `用于“${slot}”插槽的插件能力。` : "可安装到当前插槽的插件能力。";
}

export function localizedCategory(categoryId: string, categoryLabel: string, locale: AppLocale): string {
  if (locale === "zh-CN") return categoryLabel;
  const key = categoryId.split(/[.:/]/u).filter(Boolean).at(-1) ?? categoryId;
  return CATEGORY_ENGLISH[key] ?? humanizeIdentifier(categoryId);
}

export function localizedSlot(slotId: string, slotLabel: string, locale: AppLocale): string {
  if (locale === "zh-CN") return slotLabel;
  return HAN_PATTERN.test(slotLabel) ? humanizeIdentifier(slotId) : slotLabel;
}

export function localizedDynamicLabel(label: string, fallbackId: string, locale: AppLocale): string {
  if (locale === "zh-CN" || !HAN_PATTERN.test(label)) return label;
  return humanizeIdentifier(fallbackId);
}

export function localizedDynamicDescription(description: string | undefined, locale: AppLocale): string | undefined {
  if (!description) return undefined;
  const hasHan = HAN_PATTERN.test(description);
  if ((locale === "en-US" && hasHan) || (locale === "zh-CN" && !hasHan)) return undefined;
  return description;
}

export function containsHan(value: string): boolean {
  return HAN_PATTERN.test(value);
}

export function localeSafeError(
  value: unknown,
  locale: AppLocale,
  fallback: { zh: string; en: string },
): string {
  const raw = value instanceof Error ? value.message : String(value ?? "");
  const normalized = raw.trim();
  const english = locale === "en-US";
  const localizedFallback = english ? fallback.en : fallback.zh;
  if (!normalized) return localizedFallback;

  const technicalCode = /^[A-Z0-9_.:-]+$/u.test(normalized);
  if (english) {
    if (containsHan(normalized)) return localizedFallback;
    return technicalCode ? `${localizedFallback} (${normalized})` : normalized;
  }
  if (containsHan(normalized)) return normalized;
  return technicalCode ? `${localizedFallback}（${normalized}）` : localizedFallback;
}

const SYSTEM_TERMS: Record<string, { zh: string; en: string }> = {
  "builtin-python": { zh: "内置 Python", en: "Built-in Python" },
  python: { zh: "Python", en: "Python" },
  process: { zh: "独立进程", en: "Process" },
  native: { zh: "原生组件", en: "Native" },
  "ui-declarative": { zh: "声明式界面", en: "Declarative UI" },
  "fail-closed": { zh: "失败时关闭", en: "Fail closed" },
  isolate: { zh: "故障隔离", en: "Isolate" },
  advisory: { zh: "仅供建议", en: "Advisory" },
  anytime: { zh: "随时切换", en: "Any time" },
  "next-mission": { zh: "下一任务生效", en: "Next mission" },
  "safe-hold": { zh: "安全悬停后切换", en: "After safe hold" },
  restart: { zh: "重启后生效", en: "After restart" },
  "certified-update": { zh: "认证更新", en: "Certified update" },
  quarantine: { zh: "隔离", en: "Quarantine" },
  activate: { zh: "激活", en: "Activate" },
  enable: { zh: "启用", en: "Enable" },
  import: { zh: "导入", en: "Import" },
  install: { zh: "安装", en: "Install" },
  update: { zh: "更新", en: "Update" },
  healthcheck: { zh: "健康检查", en: "Health check" },
  disable: { zh: "停用", en: "Disable" },
  rollback: { zh: "回滚", en: "Rollback" },
  promote: { zh: "晋升", en: "Promote" },
  uninstall: { zh: "卸载", en: "Uninstall" },
  "trust-local-package": { zh: "批准本地包", en: "Approve local package" },
  "revoke-package": { zh: "撤销包信任", en: "Revoke package trust" },
};

export function localizedSystemTerm(value: string, locale: AppLocale): string {
  const known = SYSTEM_TERMS[value.trim().toLowerCase()];
  if (known) return locale === "en-US" ? known.en : known.zh;
  return value;
}

const SPEECH_ERRORS: Record<string, { zh: string; en: string }> = {
  aborted: { zh: "语音输入已取消", en: "Voice input was cancelled" },
  "audio-capture": { zh: "无法读取麦克风", en: "Could not access the microphone" },
  "bad-grammar": { zh: "语音识别配置无效", en: "The speech recognition configuration is invalid" },
  "language-not-supported": { zh: "当前语言不支持语音识别", en: "Speech recognition does not support the selected language" },
  network: { zh: "语音识别网络暂时不可用", en: "The speech recognition network is unavailable" },
  "no-speech": { zh: "没有检测到语音", en: "No speech was detected" },
  "not-allowed": { zh: "麦克风权限未开启", en: "Microphone permission is not enabled" },
  "service-not-allowed": { zh: "系统未允许使用语音识别服务", en: "The system did not allow the speech recognition service" },
};

export function localizedSpeechRecognitionError(code: string, locale: AppLocale): string {
  const known = SPEECH_ERRORS[code.trim().toLowerCase()];
  if (known) return locale === "en-US" ? known.en : known.zh;
  return locale === "en-US" ? "Voice input failed" : "语音输入失败";
}
