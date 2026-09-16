import {
  Bot,
  BrainCircuit,
  Check,
  ExternalLink,
  GraduationCap,
  Languages,
  Monitor,
  Moon,
  Settings2,
  Sun,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";

import { appApi } from "./api";
import { ProviderLogo } from "./ProviderLogo";
import { localeSafeError, useI18n, type AppLocale } from "./i18n";
import type { Bootstrap } from "./types";

type TabId = "general" | "model" | "memory" | "course";

const COURSE_URL = "https://binhu7.github.io/courses/ECE498/Spring2025/ECE498home.html";

export function StartupSettingsDialog({
  data,
  onClose,
  onChanged,
  onLocaleChange,
}: {
  data: Bootstrap | null;
  onClose: () => void;
  onChanged: () => Promise<void>;
  onLocaleChange: (locale: AppLocale) => void;
}) {
  const { locale, tr } = useI18n();
  const [tab, setTab] = useState<TabId>("general");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const settings = data?.settings;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const patch = async (value: Record<string, unknown>) => {
    setBusy(true);
    setError("");
    try {
      await appApi.patchSettings(value);
      await onChanged();
    } catch (reason) {
      setError(localeSafeError(reason, locale, {
        zh: "设置保存失败，请稍后重试",
        en: "Could not save settings. Try again later.",
      }));
    } finally {
      setBusy(false);
    }
  };

  const setLocale = async (next: AppLocale) => {
    onLocaleChange(next);
    await patch({ locale: next });
  };

  const tabs: Array<{ id: TabId; label: string; icon: React.ReactNode }> = [
    { id: "general", label: tr("通用", "General"), icon: <Settings2 /> },
    { id: "model", label: tr("模型", "Models"), icon: <Bot /> },
    { id: "memory", label: tr("记忆", "Memory"), icon: <BrainCircuit /> },
    { id: "course", label: "ECE498BH", icon: <GraduationCap /> },
  ];

  return <div className="startup-modal-backdrop" role="presentation" onPointerDown={(event) => {
    if (event.target === event.currentTarget && !busy) onClose();
  }}>
    <section className="startup-settings-dialog" role="dialog" aria-modal="true" aria-labelledby="startup-settings-title">
      <header className="startup-settings-header">
        <h2 id="startup-settings-title">{tr("设置", "Settings")}</h2>
        <button className="startup-icon-button" onClick={onClose} disabled={busy} aria-label={tr("关闭", "Close")}><X /></button>
      </header>
      <div className="startup-settings-body">
        <nav className="startup-settings-tabs" aria-label={tr("设置模块", "Settings sections")}>
          {tabs.map((item) => <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)}>{item.icon}<span>{item.label}</span></button>)}
        </nav>
        <div className="startup-settings-panel">
          {tab === "general" && <>
            <SettingsSection icon={<Languages />} title={tr("语言", "Language")}>
              <div className="startup-choice-grid two">
                <Choice selected={locale === "zh-CN"} label={tr("简体中文", "Simplified Chinese")} onClick={() => void setLocale("zh-CN")} disabled={busy} />
                <Choice selected={locale === "en-US"} label="English" onClick={() => void setLocale("en-US")} disabled={busy} />
              </div>
            </SettingsSection>
            <SettingsSection icon={<Monitor />} title={tr("外观", "Appearance")}>
              <div className="startup-choice-grid three">
                <Choice selected={settings?.theme === "system"} icon={<Monitor />} label={tr("跟随系统", "System")} onClick={() => void patch({ theme: "system" })} disabled={busy} />
                <Choice selected={settings?.theme === "light"} icon={<Sun />} label={tr("浅色", "Light")} onClick={() => void patch({ theme: "light" })} disabled={busy} />
                <Choice selected={settings?.theme === "dark"} icon={<Moon />} label={tr("深色", "Dark")} onClick={() => void patch({ theme: "dark" })} disabled={busy} />
              </div>
            </SettingsSection>
            <SettingsSection title={tr("更新通道", "Update channel")}>
              <div className="startup-choice-grid two">
                <Choice selected={settings?.update_channel === "stable"} label={tr("稳定版", "Stable")} onClick={() => void patch({ update_channel: "stable" })} disabled={busy} />
                <Choice selected={settings?.update_channel === "preview"} label={tr("预览版", "Preview")} onClick={() => void patch({ update_channel: "preview" })} disabled={busy} />
              </div>
            </SettingsSection>
          </>}

          {tab === "model" && <SettingsSection icon={<Bot />} title={tr("默认模型", "Default model")}>
            <div className="startup-model-list">
              {(data?.models ?? []).map((model) => <button
                key={model.id}
                className={settings?.default_model_id === model.id ? "selected" : ""}
                onClick={() => void patch({ default_model_id: model.id })}
                disabled={busy}
              >
                <ProviderLogo provider={model.provider} icon={model.icon} />
                <span><strong>{model.label}</strong><small>{model.source === "default" ? tr("DroneDream 提供", "Provided by DroneDream") : tr("自定义", "Custom")}</small></span>
                {settings?.default_model_id === model.id && <Check />}
              </button>)}
              {!data?.models.length && <p className="startup-empty">{tr("模型目录尚未就绪", "The model catalog is not ready")}</p>}
            </div>
          </SettingsSection>}

          {tab === "memory" && <SettingsSection icon={<BrainCircuit />} title={tr("任务记忆", "Task memory")}>
            <Toggle checked={Boolean(settings?.memory_enabled)} label={tr("启用任务记忆", "Enable task memory")} onChange={(checked) => void patch({ memory_enabled: checked })} disabled={busy} />
            <Toggle checked={Boolean(settings?.remember_task_preferences)} label={tr("保留同一任务的上下文", "Retain context within the same task")} onChange={(checked) => void patch({ remember_task_preferences: checked })} disabled={busy || !settings?.memory_enabled} />
            <Toggle checked={Boolean(settings?.remember_asset_choices)} label={tr("记住地图与无人机选择", "Remember map and drone choices")} onChange={(checked) => void patch({ remember_asset_choices: checked })} disabled={busy || !settings?.memory_enabled} />
            <p className="startup-safety-note">{tr("记忆不会绕过安全边界、人工确认或紧急停止。", "Memory never bypasses safety boundaries, human confirmation, or emergency stop.")}</p>
          </SettingsSection>}

          {tab === "course" && <div className="startup-course-panel">
            <div className="startup-course-mark"><GraduationCap /></div>
            <h3>ECE498BH</h3>
            <p>{tr("DroneDream 面向伊利诺伊大学 ECE498BH 课程的学习与实验入口。", "DroneDream learning and lab access for the University of Illinois ECE498BH course.")}</p>
            <button onClick={() => window.open(COURSE_URL, "_blank", "noopener,noreferrer")}>{tr("打开课程网站", "Open course website")}<ExternalLink /></button>
          </div>}
          {error && <p className="startup-settings-error" role="alert">{error}</p>}
        </div>
      </div>
    </section>
  </div>;
}

function SettingsSection({ icon, title, children }: { icon?: React.ReactNode; title: string; children: React.ReactNode }) {
  return <section className="startup-settings-section"><h3>{icon}{title}</h3>{children}</section>;
}

function Choice({ selected, icon, label, onClick, disabled }: { selected: boolean; icon?: React.ReactNode; label: string; onClick: () => void; disabled: boolean }) {
  return <button className={`startup-choice ${selected ? "selected" : ""}`} onClick={onClick} disabled={disabled}>{icon}<span>{label}</span>{selected && <Check />}</button>;
}

function Toggle({ checked, label, onChange, disabled }: { checked: boolean; label: string; onChange: (value: boolean) => void; disabled: boolean }) {
  return <label className={`startup-toggle-row ${disabled ? "disabled" : ""}`}><span>{label}</span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} disabled={disabled} /><i aria-hidden="true" /></label>;
}
