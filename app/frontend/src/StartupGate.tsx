import { AlertTriangle, Check, Settings, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { Session } from "@supabase/supabase-js";

import { appApi } from "./api";
import { desktopBrowserAuthConfigured, signInWithDesktopBrowser } from "./auth";
import { localeSafeError, useI18n, type AppLocale } from "./i18n";
import { StartupDroneScene } from "./StartupDroneScene";
import { StartupSettingsDialog } from "./StartupSettingsDialog";
import { evaluateStartupReadiness } from "./startupReadiness";
import {
  beginRuntimeBaseInstall,
  planRuntimeBaseInstall,
  probeRuntimeBase,
  runtimeBaseInstallProgress,
  runtimeBaseProgress,
  startRuntimeBase,
  type RuntimeBaseInstallPhase,
  type RuntimeBaseInstallSnapshot,
} from "./runtimeBase";
import type { Bootstrap, RuntimeSetupSnapshot } from "./types";

type StartupFailure = { title: string; detail: string; phase?: string };

export function StartupGate({
  session,
  data,
  locale,
  onLocaleChange,
  onRefresh,
  onSignedIn,
  onEnter,
}: {
  session: Session | null | undefined;
  data: Bootstrap | null;
  locale: AppLocale;
  onLocaleChange: (locale: AppLocale) => void;
  onRefresh: () => Promise<Bootstrap | null>;
  onSignedIn: (session: Session) => void;
  onEnter: () => void;
}) {
  const { tr } = useI18n();
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState(tr("正在检查 AGENT Core", "Checking AGENT Core"));
  const [failure, setFailure] = useState<StartupFailure | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [runtimeNeeded, setRuntimeNeeded] = useState(false);
  const checkingRef = useRef(false);
  const pollingRef = useRef<number | null>(null);

  const fail = useCallback((title: string, detail: string, phase?: string) => {
    setFailure({ title, detail, phase });
  }, []);

  const runChecks = useCallback(async () => {
    if (checkingRef.current) return;
    checkingRef.current = true;
    setBusy(true);
    setFailure(null);
    setRuntimeNeeded(false);
    try {
      setStatus(tr("正在检查本地核心", "Checking local core"));
      const bootstrap = await onRefresh();
      if (!bootstrap) throw new Error("LOCAL_CORE_UNAVAILABLE");
      setProgress(18);

      setStatus(tr("正在检查运行资源", "Checking runtime resources"));
      const runtime = await appApi.runtimeStatus();
      const readiness = evaluateStartupReadiness(bootstrap, runtime);
      if (readiness.requiresSetup) {
        setProgress(0);
        setRuntimeNeeded(true);
        setStatus(tr("需要安装或更新 Runtime", "Runtime installation or update required"));
        return;
      }
      setProgress(34);

      setStatus(tr("正在验证 ROS、Gazebo 与 PX4", "Verifying ROS, Gazebo, and PX4"));
      setProgress(58);
      setStatus(tr("正在验证模型目录", "Verifying model catalog"));
      if (readiness.issue === "MODEL_CATALOG_INCOMPLETE") {
        throw new Error(readiness.issue);
      }
      setProgress(74);

      setStatus(tr("正在验证地图与无人机", "Verifying maps and drones"));
      if (readiness.issue === "QUALIFIED_ASSETS_MISSING") {
        throw new Error(readiness.issue);
      }
      setProgress(90);

      if (readiness.issue) {
        throw new Error(readiness.issue);
      }
      setStatus(tr("环境已就绪", "Environment ready"));
      setProgress(100);
    } catch (reason) {
      const detail = reason instanceof Error ? reason.message : "STARTUP_CHECK_FAILED";
      fail(tr("环境检查未通过", "Environment check failed"), readableIssue(detail, locale));
    } finally {
      checkingRef.current = false;
      setBusy(false);
    }
  }, [fail, locale, onRefresh, tr]);

  useEffect(() => {
    void runChecks();
    return () => {
      if (pollingRef.current !== null) window.clearTimeout(pollingRef.current);
    };
  }, [runChecks]);

  const pollSetup = useCallback(async () => {
    try {
      const snapshot = await appApi.runtimeSetup();
      setProgress(60 + Math.max(0, Math.min(100, snapshot.progress)) * 0.4);
      setStatus(phaseLabel(snapshot.phase, tr));
      if (snapshot.active) {
        pollingRef.current = window.setTimeout(() => void pollSetup(), 450);
        return;
      }
      setBusy(false);
      if (snapshot.phase === "completed") {
        await runChecks();
        return;
      }
      if (snapshot.phase === "failed") {
        fail(
          tr("Runtime 安装未完成", "Runtime setup did not complete"),
          readableIssue(snapshot.error ?? "RUNTIME_SETUP_FAILED", locale),
          phaseLabel(snapshot.failed_phase ?? snapshot.phase, tr),
        );
      }
    } catch (reason) {
      setBusy(false);
      fail(
        tr("Runtime 安装未完成", "Runtime setup did not complete"),
        readableIssue(reason instanceof Error ? reason.message : "RUNTIME_SETUP_FAILED", locale),
      );
    }
  }, [fail, locale, runChecks, tr]);

  const pollRuntimeBase = useCallback(async (): Promise<void> => {
    const snapshot = await runtimeBaseInstallProgress();
    setProgress(runtimeBaseProgress(snapshot));
    setStatus(runtimeBasePhaseLabel(snapshot, tr));
    if (runtimeBasePhaseActive(snapshot.phase)) {
      await new Promise((resolve) => window.setTimeout(resolve, 450));
      return pollRuntimeBase();
    }
    if (snapshot.phase === "completed") return;
    if (snapshot.phase === "waitingForRestart" || snapshot.requiresRestart) {
      throw new Error("RUNTIME_BASE_RESTART_REQUIRED");
    }
    if (snapshot.phase === "cancelled") throw new Error("RUNTIME_BASE_INSTALL_CANCELLED");
    throw new Error(snapshot.error?.message ?? snapshot.error?.code ?? "RUNTIME_BASE_INSTALL_FAILED");
  }, [tr]);

  const installRuntime = async () => {
    setBusy(true);
    setFailure(null);
    try {
      setStatus(tr("正在检查共享 Runtime Base", "Checking shared Runtime Base"));
      const baseStatus = await probeRuntimeBase();
      if (baseStatus.installed) {
        if (!baseStatus.ready) {
          setStatus(tr("正在启动共享 Runtime Base", "Starting shared Runtime Base"));
          const started = await startRuntimeBase();
          if (!started.ready) throw new Error("RUNTIME_BASE_NOT_READY");
        }
        setProgress(60);
      } else {
        const plan = await planRuntimeBaseInstall();
        if (!plan.canInstall) {
          throw new Error(plan.blockers[0] ?? "RUNTIME_BASE_INSTALL_BLOCKED");
        }
        const initial = await beginRuntimeBaseInstall(plan.targetRoot);
        setProgress(runtimeBaseProgress(initial));
        setStatus(runtimeBasePhaseLabel(initial, tr));
        await pollRuntimeBase();
      }
      const snapshot = await appApi.startRuntimeSetup();
      setProgress(60 + snapshot.progress * 0.4);
      setStatus(phaseLabel(snapshot.phase, tr));
      await pollSetup();
    } catch (reason) {
      setBusy(false);
      fail(
        tr("无法启动 Runtime 安装", "Could not start Runtime setup"),
        readableIssue(reason instanceof Error ? reason.message : "RUNTIME_SETUP_FAILED", locale),
      );
    }
  };

  const signInAndEnter = async () => {
    if (session) {
      onEnter();
      return;
    }
    setBusy(true);
    setFailure(null);
    try {
      if (!await desktopBrowserAuthConfigured()) throw new Error("AUTONOMY_BROWSER_AUTH_NOT_CONFIGURED");
      const next = await signInWithDesktopBrowser(locale);
      onSignedIn(next);
      onEnter();
    } catch (reason) {
      fail(
        tr("浏览器登录未完成", "Browser sign-in did not complete"),
        readableIssue(reason instanceof Error ? reason.message : "BROWSER_AUTH_FAILED", locale),
      );
    } finally {
      setBusy(false);
    }
  };

  return <main className="startup-gate">
    <header className="startup-gate-header">
      <img src="/brand/dronedream-agent-lockup.png" className="startup-lockup" alt="DroneDream · AGENT" />
      <button className="startup-settings-button" onClick={() => setSettingsOpen(true)} aria-label={tr("设置", "Settings")}><Settings /></button>
    </header>
    <section className="startup-hero">
      <div className="startup-visual"><StartupDroneScene progress={progress} /></div>
      <div className="startup-readiness">
        <div className="startup-progress-row"><span>{status}</span><strong>{Math.round(progress)}%</strong></div>
        <div className="startup-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}>
          <i style={{ width: `${progress}%` }} />
        </div>
        <div className="startup-action-slot">
          {progress === 0 && runtimeNeeded && <button className="startup-primary-action" disabled={busy} onClick={() => void installRuntime()}>{busy ? tr("正在准备 Runtime", "Preparing Runtime") : tr("安装或更新 Runtime", "Install or update Runtime")}</button>}
          {progress === 100 && <button className="startup-primary-action" disabled={busy} onClick={() => void signInAndEnter()}>{busy ? tr("正在打开浏览器", "Opening browser") : session ? tr("进入 AGENT", "Enter AGENT") : tr("登录并进入", "Sign in and enter")}</button>}
        </div>
      </div>
    </section>

    {settingsOpen && <StartupSettingsDialog data={data} onClose={() => setSettingsOpen(false)} onChanged={async () => { await onRefresh(); }} onLocaleChange={onLocaleChange} />}
    {failure && <div className="startup-modal-backdrop">
      <section className="startup-error-dialog" role="alertdialog" aria-modal="true" aria-labelledby="startup-error-title">
        <header><div><AlertTriangle /><h2 id="startup-error-title">{failure.title}</h2></div><button onClick={() => setFailure(null)} aria-label={tr("关闭", "Close")}><X /></button></header>
        <p>{failure.detail}</p>
        {failure.phase && <small>{tr("停止阶段", "Stopped at")} · {failure.phase}</small>}
        <footer><button onClick={() => setFailure(null)}>{tr("关闭", "Close")}</button><button className="primary" onClick={() => { setFailure(null); if (runtimeNeeded) void installRuntime(); else void runChecks(); }}>{tr("重试", "Retry")}</button></footer>
      </section>
    </div>}
  </main>;
}

function phaseLabel(phase: RuntimeSetupSnapshot["phase"], tr: (zh: string, en: string) => string): string {
  const labels: Record<RuntimeSetupSnapshot["phase"], [string, string]> = {
    idle: ["等待开始", "Waiting"],
    queued: ["正在准备安装", "Preparing setup"],
    validatingResources: ["正在验证安装资源", "Validating setup resources"],
    checkingBaseRuntime: ["正在检查基础 Runtime", "Checking base Runtime"],
    preparingEnvironment: ["正在准备环境", "Preparing environment"],
    installingCore: ["正在安装 AGENT Core", "Installing AGENT Core"],
    installingDependencies: ["正在安装依赖", "Installing dependencies"],
    buildingRosWorkspace: ["正在构建 ROS 工作区", "Building ROS workspace"],
    runningHealthChecks: ["正在执行健康检查", "Running health checks"],
    recordingReceipt: ["正在记录安装凭证", "Recording setup receipt"],
    completed: ["Runtime 已就绪", "Runtime ready"],
    failed: ["Runtime 安装已停止", "Runtime setup stopped"],
  };
  const label = labels[phase];
  return tr(label[0], label[1]);
}

function runtimeBasePhaseActive(phase: RuntimeBaseInstallPhase): boolean {
  return [
    "queued",
    "verifyingManifest",
    "downloading",
    "verifyingArchive",
    "backingUp",
    "importing",
    "starting",
    "healthChecking",
    "restoring",
  ].includes(phase);
}

function runtimeBasePhaseLabel(
  snapshot: RuntimeBaseInstallSnapshot,
  tr: (zh: string, en: string) => string,
): string {
  const labels: Record<RuntimeBaseInstallPhase, [string, string]> = {
    idle: ["等待安装 Runtime Base", "Waiting for Runtime Base setup"],
    queued: ["正在准备 Runtime Base", "Preparing Runtime Base"],
    verifyingManifest: ["正在验证签名清单", "Verifying signed manifest"],
    downloading: ["正在续传 Runtime Base", "Downloading Runtime Base"],
    verifyingArchive: ["正在校验 Runtime Base", "Verifying Runtime Base"],
    backingUp: ["正在备份当前 Runtime Base", "Backing up current Runtime Base"],
    importing: ["正在导入 Runtime Base", "Importing Runtime Base"],
    starting: ["正在启动 Runtime Base", "Starting Runtime Base"],
    healthChecking: ["正在执行 Runtime Base 健康检查", "Checking Runtime Base health"],
    restoring: ["正在恢复上一版本 Runtime Base", "Restoring the previous Runtime Base"],
    waitingForRestart: ["需要重启 Windows 后继续", "Restart Windows to continue"],
    completed: ["Runtime Base 已就绪", "Runtime Base is ready"],
    failed: ["Runtime Base 安装已停止", "Runtime Base setup stopped"],
    cancelled: ["Runtime Base 安装已取消", "Runtime Base setup cancelled"],
  };
  const label = labels[snapshot.phase];
  if (snapshot.phase === "downloading" && snapshot.totalParts && snapshot.currentPart) {
    return `${tr(label[0], label[1])} · ${snapshot.currentPart}/${snapshot.totalParts}`;
  }
  return tr(label[0], label[1]);
}

function readableIssue(code: string, locale: AppLocale): string {
  const english = locale === "en-US";
  const labels: Record<string, [string, string]> = {
    LOCAL_CORE_UNAVAILABLE: ["无法连接本地 AGENT Core。请重新启动软件后再试。", "Could not reach the local AGENT Core. Restart the app and try again."],
    MODEL_CATALOG_INCOMPLETE: ["模型目录不完整，七个默认模型尚未全部加载。", "The model catalog is incomplete. All seven default models must be available."],
    QUALIFIED_ASSETS_MISSING: ["缺少合格的地图或无人机模型。", "A qualified map or drone model is missing."],
    PLUGIN_CATALOG_EMPTY: ["插件目录为空，无法建立完整工作流。", "The plugin catalog is empty, so the workflow cannot be assembled."],
    AUTONOMY_BROWSER_AUTH_NOT_CONFIGURED: ["此构建尚未注册 AGENT 浏览器登录客户端。", "This build does not have a registered AGENT browser sign-in client."],
    BROWSER_AUTH_CANCELLED: ["浏览器登录已取消。", "Browser sign-in was cancelled."],
    BROWSER_AUTH_TIMEOUT: ["浏览器登录超时，请重试。", "Browser sign-in timed out. Try again."],
    RUNTIME_BASE_RESTART_REQUIRED: ["Windows 需要重启。重启后再次打开 AGENT，安装将从安全断点继续。", "Windows must restart. Open AGENT again afterward to resume from the verified checkpoint."],
    RUNTIME_BASE_INSTALL_CANCELLED: ["Runtime Base 安装已安全取消，可以稍后继续。", "Runtime Base setup was cancelled safely and can be resumed later."],
    RUNTIME_BASE_NOT_READY: ["Runtime Base 已安装但未能通过启动检查，请重试或查看诊断信息。", "Runtime Base is installed but did not pass its startup check. Retry or inspect diagnostics."],
  };
  const normalized = code.trim();
  const known = labels[normalized];
  if (known) return english ? known[1] : known[0];
  if (/missing base runtime|distribution.*not found|wsl/iu.test(normalized)) {
    return english
      ? "The shared DroneDream Runtime or WSL foundation is missing. Setup stopped before modifying the ROS workspace."
      : "缺少共享 DroneDream Runtime 或 WSL 基础环境，安装已在修改 ROS 工作区前停止。";
  }
  return localeSafeError(normalized, locale, {
    zh: "环境检查已停止，请查看诊断信息后重试。",
    en: "Environment setup stopped. Inspect diagnostics and try again.",
  });
}
