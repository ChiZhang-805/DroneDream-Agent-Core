import {
  Archive,
  Activity,
  Bot,
  BrainCircuit,
  Braces,
  Camera,
  Check,
  ChevronDown,
  Eye,
  EyeOff,
  FilePlus2,
  FlaskConical,
  Gauge,
  GitBranch,
  Globe2,
  ImagePlus,
  KeyRound,
  Link2,
  LogOut,
  Map,
  Menu,
  MessageCircle,
  Mic,
  MicOff,
  MoreHorizontal,
  Paperclip,
  Pause,
  Pin,
  Plane,
  Play,
  Plus,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  Wrench,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Session } from "@supabase/supabase-js";

import { AvatarCropDialog } from "./AvatarCropDialog";
import { JsonSchemaForm, type JsonSchema } from "./JsonSchemaForm";
import { ProviderLogo as ProviderBrandLogo } from "./ProviderLogo";
import { StartupGate } from "./StartupGate";
import { formatAllowanceRefillAt, remainingAllowancePercent } from "./allowance";
import { appApi } from "./api";
import { currentPlanApproval } from "./planApproval";
import { accountAvatar, currentSession, issueModelGrant, loadAccountOverview, supabase, updateAccountAvatar } from "./auth";
import {
  I18nProvider,
  containsHan,
  initialLocale,
  localizedAssetName,
  localizedCategory,
  localizedDynamicDescription,
  localizedDynamicLabel,
  localizedPluginDescription,
  localizedPluginName,
  localizedSlot,
  localizedSpeechRecognitionError,
  localizedSystemTerm,
  localeSafeError,
  normalizeLocale,
  rememberLocale,
  useI18n,
  type AppLocale,
} from "./i18n";
import type { AccountOverview, AssetImportJob, AssetImportState, AssetIssue, AssetQualificationEvidence, AssetQualificationJob, AssetQualificationState, AssetSourceAdapter, AssetVersionEntry, Bootstrap, ConnectorCredentialReference, DailyModelUsage, ManagedPlanId, ModelEntry, Page, PluginDetail, PluginEntry, PluginGovernancePolicy, PluginMarketplaceCatalog, PluginMarketplaceSource, PluginPanel, TaskThread } from "./types";

function BrandMark({ size = 34 }: { size?: number }) {
  return <img className="brand-mark" src="/brand/dronedream-agent-mark.png" width={size} height={size} alt="" />;
}

function BrandLockup() {
  return <img className="brand-lockup" src="/brand/dronedream-agent-lockup.png" alt="DroneDream · AGENT" />;
}

function MissionCloudIcon() {
  return <svg className="mission-cloud-artwork" viewBox="0 0 1536 1024" aria-hidden="true">
    <defs>
      <filter id="mission-cloud-black-to-alpha" x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
        <feColorMatrix values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  3 0 0 0 0" />
        <feComponentTransfer>
          <feFuncA type="linear" slope="1.05" intercept="-0.04" />
        </feComponentTransfer>
      </filter>
    </defs>
    <image href="/ui/autonomy-cloud-mission-source.png" width="1536" height="1024" filter="url(#mission-cloud-black-to-alpha)" />
  </svg>;
}

function App() {
  const [locale, setLocale] = useState<AppLocale>(() => initialLocale());
  const changeLocale = useCallback((next: AppLocale) => {
    rememberLocale(next);
    setLocale(next);
  }, []);
  return <I18nProvider locale={locale}><AppContent locale={locale} onLocaleChange={changeLocale} /></I18nProvider>;
}

function AppContent({ locale, onLocaleChange }: { locale: AppLocale; onLocaleChange: (locale: AppLocale) => void }) {
  const { tr } = useI18n();
  const [session, setSession] = useState<Session | null | undefined>(undefined);
  const [data, setData] = useState<Bootstrap | null>(null);
  const [launcherEntered, setLauncherEntered] = useState(() => import.meta.env.DEV
    && new URLSearchParams(window.location.search).has("workspace"));
  const [page, setPage] = useState<Page>("new");
  const [activeThread, setActiveThread] = useState<TaskThread | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [accountOpen, setAccountOpen] = useState(false);
  const [threadMenuOpen, setThreadMenuOpen] = useState<string | null>(null);
  const [account, setAccount] = useState<AccountOverview | null>(null);
  const [avatarEditorOpen, setAvatarEditorOpen] = useState(false);
  const [savedAvatarUrl, setSavedAvatarUrl] = useState<string | null>(null);
  const [accountError, setAccountError] = useState("");
  const [error, setError] = useState("");
  const localeBootstrapped = useRef(false);

  const refresh = useCallback(async (): Promise<Bootstrap | null> => {
    setError("");
    try {
      const next = await appApi.bootstrap();
      setData(next);
      if (!localeBootstrapped.current) {
        localeBootstrapped.current = true;
        const nextLocale = normalizeLocale(next.settings.locale);
        if (nextLocale !== locale) onLocaleChange(nextLocale);
      }
      return next;
    }
    catch (value) {
      setError(humanError(value, locale, tr("启动失败", "Startup failed")));
      return null;
    }
  }, [locale, onLocaleChange, tr]);

  useEffect(() => {
    currentSession().then(setSession);
    const { data: listener } = supabase.auth.onAuthStateChange((_event, next) => setSession(next));
    return () => listener.subscription.unsubscribe();
  }, []);
  useEffect(() => { localeBootstrapped.current = false; }, [session?.user.id]);
  useEffect(() => setSavedAvatarUrl(null), [session?.user.id]);
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    const active = data?.asset_qualification_jobs.some((job) => ["preparing", "running", "validating"].includes(job.state));
    if (!active) return undefined;
    const timer = window.setInterval(() => { void refresh(); }, 2_000);
    return () => window.clearInterval(timer);
  }, [data?.asset_qualification_jobs, refresh]);
  useEffect(() => {
    const preference = data?.settings.theme ?? "system";
    const dark = preference === "dark"
      || (preference === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }, [data?.settings.theme]);
  const refreshAccount = useCallback(async () => {
    if (!session) return;
    setAccountError("");
    try { setAccount(await loadAccountOverview(session)); }
    catch (value) {
      setAccount(null);
      setAccountError(humanError(value, locale, tr("账户数据加载失败", "Could not load account data")));
    }
  }, [locale, session, tr]);
  useEffect(() => { if (session) void refreshAccount(); }, [session, refreshAccount]);
  useEffect(() => {
    if (!session) return undefined;
    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") void refreshAccount();
    };
    const timer = window.setInterval(refreshWhenVisible, 15_000);
    window.addEventListener("focus", refreshWhenVisible);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", refreshWhenVisible);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [session, refreshAccount]);

  const openThread = async (thread: TaskThread) => {
    setActiveThread(await appApi.getThread(thread.thread_id));
    setPage("thread");
  };
  const openNew = () => { setActiveThread(null); setPage("new"); };
  const updateThread = async (thread: TaskThread, patch: Pick<TaskThread, "pinned"> | Pick<TaskThread, "archived">) => {
    await appApi.patchThread(thread.thread_id, patch);
    if ("archived" in patch && patch.archived && activeThread?.thread_id === thread.thread_id) openNew();
    setThreadMenuOpen(null);
    await refresh();
  };

  if (!launcherEntered || session === null) return <StartupGate
    session={session}
    data={data}
    locale={locale}
    onLocaleChange={onLocaleChange}
    onRefresh={refresh}
    onSignedIn={setSession}
    onEnter={() => setLauncherEntered(true)}
  />;
  if (session === undefined) return <main className="splash"><BrandMark size={64} /><span>DroneDream · AGENT</span></main>;
  if (!data) return <main className="splash"><BrandMark size={64} /><span>{error || tr("正在启动 AGENT Core", "Starting AGENT Core")}</span><button className="text-button" onClick={refresh}>{tr("重试", "Retry")}</button></main>;

  const displayName = account?.displayName
    ?? session.user.user_metadata.full_name
    ?? session.user.email?.split("@")[0]
    ?? "Pilot";
  const avatarUrl = savedAvatarUrl ?? account?.avatarUrl ?? accountAvatar(session);
  const planName = account?.snapshot.plan.name ?? "—";
  const billingScope = account?.snapshot.account?.billing_scope === "business" ? tr("商业", "Business") : tr("个人", "Individual");
  const remainingPercent = account
    ? Math.round(remainingAllowancePercent(account.snapshot.usage.remaining_ai_credits, account.snapshot.plan.included_ai_credits))
    : null;
  return <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
    <aside className="sidebar">
      <header className="sidebar-brand"><BrandLockup /><button className="icon-button sidebar-toggle" onClick={() => setSidebarOpen(false)} aria-label={tr("收起侧边栏", "Collapse sidebar")}><Menu /></button></header>
      <nav className="primary-nav">
        <NavButton icon={<MessageCircle />} label={tr("新对话", "New task")} active={page === "new"} onClick={openNew} />
        <NavButton icon={<Map />} label={tr("地图仓库", "Map library")} active={page === "maps"} onClick={() => setPage("maps")} />
        <NavButton icon={<Plane />} label={tr("无人机仓库", "Drone library")} active={page === "vehicles"} onClick={() => setPage("vehicles")} />
        <NavButton icon={<Sparkles />} label={tr("插件", "Plugins")} active={page === "plugins"} onClick={() => setPage("plugins")} />
      </nav>
      <div className="history-heading"><span>{tr("对话", "Tasks")}</span></div>
      <div className="thread-list">
        {data.threads.map((thread) => <div key={thread.thread_id} className={`thread-row ${activeThread?.thread_id === thread.thread_id ? "active" : ""}`}>
          <button className="thread-open" onClick={() => void openThread(thread)}><MessageCircle size={16} /><span>{thread.title}</span></button>
          <button className="thread-more" onClick={() => setThreadMenuOpen(threadMenuOpen === thread.thread_id ? null : thread.thread_id)} aria-label={`${thread.title} ${tr("菜单", "menu")}`}><MoreHorizontal /></button>
          {threadMenuOpen === thread.thread_id && <div className="thread-menu">
            <button onClick={() => void updateThread(thread, { pinned: !thread.pinned })}><Pin />{thread.pinned ? tr("取消置顶", "Unpin") : tr("置顶", "Pin")}</button>
            <button onClick={() => void updateThread(thread, { archived: true })}><Archive />{tr("归档", "Archive")}</button>
          </div>}
        </div>)}
        {!data.threads.length && <span className="empty-sidebar">{tr("还没有任务", "No tasks yet")}</span>}
      </div>
      <div className="account-wrap">
        {accountOpen && <div className="account-menu">
          <div className="account-identity"><Avatar name={displayName} src={avatarUrl} /><div><strong>{displayName}</strong><span>{billingScope} · {planName}</span></div></div>
          <button onClick={() => { setPage("settings"); setAccountOpen(false); void refreshAccount(); }}><Gauge />{tr("剩余额度", "Remaining allowance")} <span>{remainingPercent === null ? "—" : `${remainingPercent}%`}</span></button>
          <button onClick={() => { setPage("settings"); setAccountOpen(false); }}><Settings />{tr("设置", "Settings")}</button>
          <button onClick={() => void supabase.auth.signOut()}><LogOut />{tr("退出登录", "Sign out")}</button>
        </div>}
        <div className="account-button"><button className="account-avatar-trigger" aria-label={tr("更换头像", "Change profile photo")} onClick={() => setAvatarEditorOpen(true)}><Avatar name={displayName} src={avatarUrl} /></button><button className="account-menu-trigger" onClick={() => setAccountOpen(!accountOpen)}><span>{displayName}</span></button></div>
      </div>
    </aside>
    <section className="workspace">
      <header className="topbar">{!sidebarOpen && <button className="icon-button" onClick={() => setSidebarOpen(true)} aria-label={tr("展开侧边栏", "Expand sidebar")}><Menu /></button>}<h1>{pageTitle(page, activeThread, tr)}</h1></header>
      {error && <div className="global-error"><span>{error}</span><button onClick={() => setError("")}><X /></button></div>}
      {page === "new" && <ConversationPage session={session} data={data} displayName={displayName} avatarUrl={avatarUrl} onChanged={async (thread) => { await Promise.all([refresh(), refreshAccount()]); await openThread(thread); }} onError={setError} />}
      {page === "thread" && activeThread && <ConversationPage session={session} data={data} displayName={displayName} avatarUrl={avatarUrl} thread={activeThread} onChanged={async (thread) => { await Promise.all([refresh(), refreshAccount()]); await openThread(thread); }} onError={setError} />}
      {page === "maps" && <AssetLibrary kind="map" jobs={data.asset_import_jobs.filter((job) => job.asset_kind === "map" || job.asset_kind === "world")} versions={data.asset_versions} qualificationJobs={data.asset_qualification_jobs} adapters={data.asset_source_adapters} onOpenConnectors={() => setPage("plugins")} onChanged={async () => { await refresh(); }} onError={setError} />}
      {page === "vehicles" && <AssetLibrary kind="vehicle" jobs={data.asset_import_jobs.filter((job) => job.asset_kind === "vehicle")} versions={data.asset_versions} qualificationJobs={data.asset_qualification_jobs} adapters={data.asset_source_adapters} onOpenConnectors={() => setPage("plugins")} onChanged={async () => { await refresh(); }} onError={setError} />}
      {page === "plugins" && <PluginLibrary items={data.plugins} adapters={data.asset_source_adapters} credentials={data.connector_credentials} threadId={activeThread?.thread_id ?? null} onChanged={async () => { await refresh(); }} onError={setError} />}
      {page === "settings" && <SettingsPage session={session} data={data} account={account} accountError={accountError} onChanged={async () => { await refresh(); }} onRefreshAccount={refreshAccount} onEditAvatar={() => setAvatarEditorOpen(true)} onLocaleChange={onLocaleChange} />}
    </section>
    {avatarEditorOpen && <AvatarEditor session={session} displayName={displayName} avatarUrl={avatarUrl} onClose={() => setAvatarEditorOpen(false)} onSaved={(nextAvatarUrl) => {
      setSavedAvatarUrl(nextAvatarUrl);
      setAccount((current) => current ? { ...current, avatarUrl: nextAvatarUrl } : current);
    }} />}
  </div>;
}

function pageTitle(page: Page, thread: TaskThread | null, tr: (zh: string, en: string) => string) {
  return page === "thread" ? thread?.title : ({
    new: tr("新对话", "New task"),
    maps: tr("地图仓库", "Map library"),
    vehicles: tr("无人机仓库", "Drone library"),
    plugins: tr("插件", "Plugins"),
    settings: tr("设置", "Settings"),
  } as const)[page];
}

function NavButton({ icon, label, active, onClick }: { icon: React.ReactNode; label: string; active: boolean; onClick: () => void }) {
  return <button className={active ? "active" : ""} onClick={onClick}>{icon}<span>{label}</span></button>;
}

function Avatar({ name, src }: { name: string; src?: string | null }) {
  return <span className={`avatar ${src ? "has-image" : ""}`}>{src ? <img src={src} alt="" /> : name.slice(0, 1).toUpperCase()}</span>;
}

const MAX_AVATAR_SOURCE_BYTES = 8 * 1024 * 1024;

function cameraFrameBlob(video: HTMLVideoElement): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  if (!context) return Promise.reject(new Error("CAMERA_FRAME_UNAVAILABLE"));
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob(
    (blob) => blob ? resolve(blob) : reject(new Error("CAMERA_FRAME_UNAVAILABLE")),
    "image/jpeg",
    0.94,
  ));
}

function AvatarEditor({ session, displayName, avatarUrl, onClose, onSaved }: {
  session: Session;
  displayName: string;
  avatarUrl: string | null;
  onClose: () => void;
  onSaved: (avatarUrl: string) => void;
}) {
  const { locale, tr } = useI18n();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [cropSource, setCropSource] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);

  const stopCamera = useCallback(() => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    setCameraStream(null);
    setCameraReady(false);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !cameraStream) return;
    video.srcObject = cameraStream;
    void video.play().catch(() => undefined);
  }, [cameraStream]);

  useEffect(() => () => cameraStreamRef.current?.getTracks().forEach((track) => track.stop()), []);
  useEffect(() => {
    if (!cropSource) return undefined;
    return () => URL.revokeObjectURL(cropSource);
  }, [cropSource]);

  const choosePhoto = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) return setError(tr("请选择 JPEG、PNG 或 WebP 图片", "Choose a JPEG, PNG, or WebP image"));
    if (file.size > MAX_AVATAR_SOURCE_BYTES) return setError(tr("图片不能超过 8 MB", "The image must be no larger than 8 MB"));
    setError("");
    setCropSource(URL.createObjectURL(file));
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) return setError(tr("当前设备无法调用摄像头", "The camera is unavailable on this device"));
    setPending(true);
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      stopCamera();
      cameraStreamRef.current = stream;
      setCameraStream(stream);
    } catch (reason) {
      const name = reason instanceof DOMException ? reason.name : "";
      if (["NotAllowedError", "PermissionDeniedError", "SecurityError"].includes(name)) setError(tr("摄像头权限未开启", "Camera permission is not enabled"));
      else if (["NotFoundError", "DevicesNotFoundError"].includes(name)) setError(tr("没有找到摄像头", "No camera was found"));
      else setError(tr("摄像头暂时不可用", "The camera is temporarily unavailable"));
    } finally {
      setPending(false);
    }
  };

  const capture = async () => {
    const video = videoRef.current;
    if (!video || !cameraReady || video.videoWidth < 1) return;
    setPending(true);
    try {
      const blob = await cameraFrameBlob(video);
      stopCamera();
      setCropSource(URL.createObjectURL(blob));
    } catch (reason) {
      setError(humanError(reason, locale, tr("拍照失败", "Could not take the photo")));
    } finally {
      setPending(false);
    }
  };

  const save = async (avatarDataUrl: string) => {
    setPending(true);
    setError("");
    try {
      const nextAvatarUrl = await updateAccountAvatar(session, avatarDataUrl);
      onSaved(nextAvatarUrl);
      onClose();
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "AVATAR_SAVE_FAILED";
      const friendly = humanError(message, locale, tr("头像保存失败", "Could not save the profile photo"));
      setError(friendly);
      throw new Error(friendly);
    } finally {
      setPending(false);
    }
  };

  return <>
    <div className="profile-photo-backdrop" role="presentation" onPointerDown={(event) => {
      if (event.target === event.currentTarget && !pending) { stopCamera(); onClose(); }
    }}>
      <section className="profile-photo-dialog" role="dialog" aria-modal="true" aria-labelledby="profile-photo-title">
        <header><h2 id="profile-photo-title">{tr("更换头像", "Change profile photo")}</h2><button className="avatar-crop-close" aria-label={tr("关闭", "Close")} disabled={pending} onClick={() => { stopCamera(); onClose(); }}>×</button></header>
        <div className="profile-photo-current"><Avatar name={displayName} src={avatarUrl} /></div>
        <input ref={fileInput} className="avatar-file-input" type="file" accept="image/jpeg,image/png,image/webp" onChange={choosePhoto} />
        <div className="profile-photo-actions"><button className="secondary-button" disabled={pending} onClick={() => fileInput.current?.click()}><ImagePlus />{tr("选择照片", "Choose photo")}</button><button className="secondary-button" disabled={pending || Boolean(cameraStream)} onClick={() => void startCamera()}><Camera />{tr("拍照", "Use camera")}</button></div>
        {cameraStream && <div className="avatar-camera"><video ref={videoRef} autoPlay muted playsInline onCanPlay={() => setCameraReady(true)} /><div><button className="primary-button" disabled={!cameraReady || pending} onClick={() => void capture()}><Camera />{tr("拍摄", "Take photo")}</button><button className="secondary-button" disabled={pending} onClick={stopCamera}>{tr("取消", "Cancel")}</button></div></div>}
        {error && <p className="form-error" role="alert">{error}</p>}
      </section>
    </div>
    {cropSource && <AvatarCropDialog sourceUrl={cropSource} pending={pending} onCancel={() => setCropSource(null)} onConfirm={save} onSourceError={(message) => { setError(message); setCropSource(null); }} />}
  </>;
}

function ConversationPage({ session, data, displayName, avatarUrl, thread, onChanged, onError }: {
  session: Session; data: Bootstrap; displayName: string; avatarUrl: string | null; thread?: TaskThread;
  onChanged: (thread: TaskThread) => Promise<void>; onError: (value: string) => void;
}) {
  const { locale, tr } = useI18n();
  const configuredDefaultModel = data.models.some((item) => item.id === data.settings.default_model_id)
    ? data.settings.default_model_id
    : data.models[0]?.id ?? "gpt-5.4";
  const rememberedMapId = data.settings.memory_enabled
    && data.settings.remember_asset_choices
    && data.asset_versions.some((item) => (
      ["map", "world"].includes(item.kind) && item.asset_id === data.settings.last_map_id
    ))
    ? data.settings.last_map_id ?? ""
    : "";
  const rememberedVehicleId = data.settings.memory_enabled
    && data.settings.remember_asset_choices
    && data.asset_versions.some((item) => (
      item.kind === "vehicle" && item.asset_id === data.settings.last_vehicle_id
    ))
    ? data.settings.last_vehicle_id ?? ""
    : "";
  const mapChoices = buildAssetChoices(data.asset_versions, data.asset_qualification_jobs, "map");
  const vehicleChoices = buildAssetChoices(data.asset_versions, data.asset_qualification_jobs, "vehicle");
  const rememberedMapKey = data.settings.last_map_content_sha256
    ? `${rememberedMapId}@${data.settings.last_map_content_sha256}`
    : rememberedMapId;
  const rememberedVehicleKey = data.settings.last_vehicle_content_sha256
    ? `${rememberedVehicleId}@${data.settings.last_vehicle_content_sha256}`
    : rememberedVehicleId;
  const [message, setMessage] = useState("");
  const [modelId, setModelId] = useState(thread?.selected_model || configuredDefaultModel);
  const [mapKey, setMapKey] = useState(
    thread?.selected_map_content_sha256
      ? `${thread.selected_map_id}@${thread.selected_map_content_sha256}`
      : thread?.selected_map_id || rememberedMapKey,
  );
  const [vehicleKey, setVehicleKey] = useState(
    thread?.selected_vehicle_content_sha256
      ? `${thread.selected_vehicle_id}@${thread.selected_vehicle_content_sha256}`
      : thread?.selected_vehicle_id || rememberedVehicleKey,
  );
  const [plusOpen, setPlusOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const [inputChannel, setInputChannel] = useState<"text" | "voice" | "camera">("text");
  const [inputMetadata, setInputMetadata] = useState<Record<string, unknown>>({});
  const recognition = useRef<SpeechRecognitionLike | null>(null);
  const attachmentInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);
  const selectedModel = data.models.find((item) => item.id === modelId) ?? data.models[0];
  const selectedMap = mapChoices.find((item) => item.key === mapKey);
  const selectedVehicle = vehicleChoices.find((item) => item.key === vehicleKey);
  const selectedPairValid = Boolean(
    selectedMap?.qualificationId
    && selectedMap.qualificationId === selectedVehicle?.qualificationId,
  );
  const defaultModels = data.models.filter((item) => item.source === "default");
  const customModels = data.models.filter((item) => item.source === "custom");
  const voicePlugin = data.plugins.find((item) => item.enabled && item.placement.slot_id === "interaction.voice-input");

  const issueGrant = async (model: ModelEntry, threadId: string) => model.source === "custom"
    ? appApi.issueCustomModelGrant(model, threadId)
    : issueModelGrant(session, model, threadId);

  const toggleVoice = () => {
    if (listening) { recognition.current?.stop(); setListening(false); return; }
    const voiceEngine = voicePlugin?.capabilities[0]?.metadata.engine;
    if (voiceEngine === "audio-attachment") {
      setInputChannel("voice");
      setInputMetadata({ transcript_source: "audio-attachment" });
      attachmentInput.current?.click();
      return;
    }
    if (voiceEngine !== "web-speech") return onError(tr("请先在插件页面启用一种语音输入插件", "Enable a voice input plugin on the Plugins page first"));
    const Constructor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Constructor) return onError(tr("当前 WebView 不支持语音转写；可以继续使用文字或添加音频文件", "This WebView does not support speech transcription. You can type or attach an audio file instead."));
    const value = new Constructor();
    value.continuous = true; value.interimResults = true; value.lang = locale;
    value.onresult = (event) => {
      let transcript = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) transcript += event.results[index][0].transcript;
      if (transcript) setMessage((current) => `${current}${transcript}`);
    };
    value.onerror = (event) => { setListening(false); onError(localizedSpeechRecognitionError(event.error, locale)); };
    value.onend = () => setListening(false);
    recognition.current = value;
    setInputChannel("voice");
    setInputMetadata({ transcript_source: "web-speech", transcript_locale: locale });
    value.start(); setListening(true);
  };

  const send = async () => {
    if (!message.trim() || busy) return;
    setBusy(true); onError("");
    try {
      if (thread && ["executing", "holding", "landing"].includes(thread.state)) {
        await appApi.runtimeMessage(thread.thread_id, message.trim());
        setMessage("");
        await onChanged(thread);
        return;
      }
      const activeModelId = selectedModel.id;
      const current = thread ?? await appApi.createThread(activeModelId, tr("新任务", "New task"), locale);
      await appApi.patchThread(current.thread_id, {
        selected_model: activeModelId,
        selected_map_id: selectedMap?.assetId ?? null,
        selected_map_content_sha256: selectedMap?.contentSha256 ?? null,
        selected_vehicle_id: selectedVehicle?.assetId ?? null,
        selected_vehicle_content_sha256: selectedVehicle?.contentSha256 ?? null,
        locale,
        title: message.trim().slice(0, 32),
      });
      const uploaded = [];
      for (const attachment of attachments) {
        uploaded.push(await appApi.uploadAttachment(current.thread_id, attachment));
      }
      if (!selectedMap || !selectedVehicle || !selectedPairValid) {
        await appApi.appendMessage(current.thread_id, message.trim());
        setMessage("");
        onError(tr("已保存任务。请选择同一次真实仿真认证通过的地图与无人机组合", "Task saved. Select a map and drone qualified together in the same real-simulation run."));
        await onChanged(current);
        return;
      }
      const grant = await issueGrant(selectedModel, current.thread_id);
      await appApi.prepare(current.thread_id, {
        expected_owner_account_id: session.user.id,
        source_edition: "autonomy",
        message: message.trim(),
        map_id: selectedMap.assetId,
        map_content_sha256: selectedMap.contentSha256,
        vehicle_id: selectedVehicle.assetId,
        vehicle_content_sha256: selectedVehicle.contentSha256,
        model_id: activeModelId,
        model_grant: grant.grant, gateway_base_url: "gateway_base_url" in grant ? grant.gateway_base_url : undefined, locale,
        // Every imported world owns its own named launch entities. Let the Core
        // resolve a qualified launch node from the exact bound map contract;
        // never inject the School Map's office-specific alias here.
        start_entity: "__auto__", attachment_ids: uploaded.map((item) => item.attachment_id),
        input_channel: inputChannel, input_metadata: inputMetadata,
      });
      setMessage(""); setAttachments([]); setInputChannel("text"); setInputMetadata({});
      await onChanged(current);
    } catch (value) { onError(humanError(value, locale)); }
    finally { setBusy(false); }
  };

  const confirmablePlan = currentPlanApproval(thread);

  // 功能：
  //   先将用户点击绑定到当前可确认的具体卡片，再准备 Runtime 并请求服务端执行。
  // 输入：
  //   planMessageId：用户实际点击的计划消息身份，不以最新计划静默替换旧卡片。
  // 输出：
  //   None：更新任务视图或显示失败；没有对应计划时不触发安装或模型授权。
  const execute = async (planMessageId: string) => {
    if (!thread || busy) return;
    setBusy(true); onError("");
    try {
      const approval = currentPlanApproval(thread);
      if (!approval || approval.messageId !== planMessageId) {
        throw new Error("EXECUTION_PLAN_MESSAGE_NOT_CURRENT");
      }
      const status = await appApi.runtimeStatus();
      if (!status.provisioned) await appApi.provisionRuntime();
      const grant = await issueGrant(selectedModel, thread.thread_id);
      await appApi.execute(thread.thread_id, {
        expected_owner_account_id: session.user.id,
        source_edition: "autonomy",
        plan_revision_id: approval.revisionId,
        model_id: selectedModel.id,
        model_grant: grant.grant,
        gateway_base_url: "gateway_base_url" in grant ? grant.gateway_base_url : undefined,
      });
      await onChanged(thread);
    } catch (value) { onError(humanError(value, locale)); }
    finally { setBusy(false); }
  };

  return <main className={`conversation ${thread?.messages?.length ? "has-messages" : ""}`}>
    {!thread?.messages?.length ? <div className="conversation-welcome">
      <div className="welcome-icon"><MissionCloudIcon /></div>
      <h1>{tr(`你好，${displayName}。今天想让无人机完成什么？`, `Hello, ${displayName}. What should your drone accomplish today?`)}</h1>
    </div> : <div className="messages">
      {thread.messages.map((item) => <article key={item.message_id} className={`message ${item.role} ${item.kind}`}>
        <div className="message-role">{item.role === "user" ? <Avatar name={displayName} src={avatarUrl} /> : <BrandMark size={28} />}</div>
        <div className="message-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{item.content}</ReactMarkdown>{item.kind === "plan" && <PlanSummary metadata={item.metadata} busy={busy} confirmable={confirmablePlan?.messageId === item.message_id} onConfirm={() => execute(item.message_id)} />}</div>
      </article>)}
    </div>}
    <section className="composer-wrap">
      <div className="selection-chips">
        {selectedMap && <button onClick={() => setMapKey("")}><Map />{selectedMap.name}<X /></button>}
        {selectedVehicle && <button onClick={() => setVehicleKey("")}><Plane />{selectedVehicle.name}<X /></button>}
        {attachments.map((file, index) => <button key={`${file.name}-${file.lastModified}`} onClick={() => setAttachments((items) => items.filter((_item, itemIndex) => itemIndex !== index))}><FilePlus2 />{file.name}<X /></button>)}
      </div>
      <textarea value={message} onChange={(event) => setMessage(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }} placeholder={tr("描述任务或修改计划", "Describe a task or revise the plan")} rows={3} />
      <footer className="composer-actions">
        <div className="composer-left">
          <div className="popover-anchor">
            <button className="round-button" onClick={() => setPlusOpen(!plusOpen)} aria-label={tr("添加", "Add")}><Plus /></button>
            {plusOpen && <div className="popover plus-menu">
              <MenuSelect icon={<Map />} label={tr("选择地图", "Select map")} items={mapChoices} selected={mapKey} onSelect={(key) => {
                const nextMap = mapChoices.find((item) => item.key === key);
                const pairedVehicle = vehicleChoices.find((item) => item.qualificationId === nextMap?.qualificationId);
                setMapKey(key);
                setVehicleKey(pairedVehicle?.key ?? "");
                setPlusOpen(false);
              }} />
              <MenuSelect icon={<Plane />} label={tr("选择无人机", "Select drone")} items={vehicleChoices} selected={vehicleKey} onSelect={(key) => {
                const nextVehicle = vehicleChoices.find((item) => item.key === key);
                const pairedMap = mapChoices.find((item) => item.qualificationId === nextVehicle?.qualificationId);
                setVehicleKey(key);
                setMapKey(pairedMap?.key ?? "");
                setPlusOpen(false);
              }} />
              <button onClick={() => attachmentInput.current?.click()}><Paperclip />{tr("添加文件", "Attach files")}</button>
              <button onClick={() => cameraInput.current?.click()}><Camera />{tr("相机拍摄", "Use camera")}</button>
              <input ref={attachmentInput} type="file" multiple hidden onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                setAttachments(files);
                if (files.some((file) => file.type.startsWith("audio/"))) {
                  setInputChannel("voice");
                  setInputMetadata({ transcript_source: "audio-attachment" });
                }
                setPlusOpen(false);
              }} />
              <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) {
                  setAttachments((current) => [...current, file]);
                  setInputChannel("camera");
                  setInputMetadata({ capture_source: "device-camera", captured_at: new Date().toISOString() });
                }
                setPlusOpen(false);
              }} />
            </div>}
          </div>
        </div>
        <div className="composer-right">
          <div className="popover-anchor">
            <button className="model-button" onClick={() => setModelOpen(!modelOpen)}><ProviderBrandLogo provider={selectedModel.provider} icon={selectedModel.icon} /><span>{selectedModel.label}</span><ChevronDown /></button>
            {modelOpen && <div className="popover model-menu">
              <ModelGroup label={tr("默认", "Default")} models={defaultModels} selected={modelId} onSelect={(id) => { setModelId(id); setModelOpen(false); }} />
              <ModelGroup label={tr("自定义", "Custom")} models={customModels} selected={modelId} onSelect={(id) => { setModelId(id); setModelOpen(false); }} />
              {!customModels.length && <span className="model-empty">{tr("在设置中添加自定义模型", "Add a custom model in Settings")}</span>}
            </div>}
          </div>
          <button className={`round-button ${listening ? "recording" : ""}`} onClick={toggleVoice} aria-label={tr("语音输入", "Voice input")}>{listening ? <MicOff /> : <Mic />}</button>
          <button className="send-button" onClick={() => void send()} disabled={!message.trim() || busy} aria-label={tr("发送", "Send")}>{busy ? <span className="spinner" /> : <Send />}</button>
        </div>
      </footer>
    </section>
  </main>;
}

function ModelGroup({ label, models, selected, onSelect }: { label: string; models: ModelEntry[]; selected: string; onSelect: (id: string) => void }) {
  if (!models.length) return null;
  return <div className="model-group"><div className="model-group-label">{label}</div>{models.map((model) => <button key={model.id} className={model.id === selected ? "selected" : ""} onClick={() => onSelect(model.id)}><ProviderBrandLogo provider={model.provider} icon={model.icon} /><span>{model.label}</span>{model.id === selected && <Check />}</button>)}</div>;
}

interface AssetChoice {
  key: string;
  assetId: string;
  contentSha256: string | null;
  name: string;
  status: "qualified";
  versioned: boolean;
  qualificationId: string;
}

function buildAssetChoices(
  versions: AssetVersionEntry[],
  qualificationJobs: AssetQualificationJob[],
  kind: "map" | "vehicle",
): AssetChoice[] {
  const qualifiedVersions = versions.filter((item) => (
    item.maturity === "qualified"
    && (kind === "map" ? ["map", "world"].includes(item.kind) : item.kind === "vehicle")
  ));
  return qualifiedVersions.flatMap((item): AssetChoice[] => {
    const job = qualificationJobs.find((candidate) => (
      candidate.state === "qualified"
      && Boolean(candidate.qualification_id)
      && (kind === "map"
        ? candidate.map_asset_id === item.asset_id
          && candidate.result_map_content_sha256 === item.content_sha256
        : candidate.vehicle_asset_id === item.asset_id
          && candidate.result_vehicle_content_sha256 === item.content_sha256)
    ));
    if (!job?.qualification_id) return [];
    return [{
      key: `${item.asset_id}@${item.content_sha256}`,
      assetId: item.asset_id,
      contentSha256: item.content_sha256,
      name: `${typeof item.asset_ir.name === "string" ? item.asset_ir.name : item.asset_id} · ${item.content_sha256.slice(0, 8)}`,
      status: "qualified",
      versioned: true,
      qualificationId: job.qualification_id,
    }];
  });
}

function MenuSelect({ icon, label, items, selected, onSelect }: { icon: React.ReactNode; label: string; items: AssetChoice[]; selected: string; onSelect: (id: string) => void }) {
  const { tr } = useI18n();
  return <div className="menu-select"><div className="menu-label">{icon}{label}</div>{items.length ? items.map((item) => <button key={item.key} onClick={() => onSelect(item.key)}><span>{item.name}</span><small className={item.status}>{item.versioned ? tr("已认证版本", "Qualified version") : tr("合格", "Qualified")}</small>{selected === item.key && <Check />}</button>) : <span className="menu-empty">{tr("仓库为空", "Library is empty")}</span>}</div>;
}

// 功能：
//   展示计划概要；历史或已执行计划仅供查看，不显示可启动另一份计划的确认按钮。
// 输入：
//   metadata：本卡片的计划概要。
//   busy：当前正在等待操作完成。
//   confirmable：此卡片是否对应当前等待批准的计划。
//   onConfirm：已绑定此卡片消息身份的确认回调。
// 输出：
//   summary：计划概要及与其身份一致的操作控件。
function PlanSummary({ metadata, busy, confirmable, onConfirm }: { metadata: Record<string, unknown>; busy: boolean; confirmable: boolean; onConfirm: () => Promise<void> }) {
  const { tr } = useI18n();
  const route = Array.isArray(metadata.route_nodes) ? metadata.route_nodes.join(" → ") : "";
  const summary = <div className="plan-summary"><div><ShieldCheck /><strong>{confirmable ? tr("等待确认", "Awaiting confirmation") : tr("计划记录", "Plan record")}</strong></div>{route && <p>{route}</p>}<dl><div><dt>{tr("模型调用", "Model calls")}</dt><dd>{String(metadata.model_calls ?? "—")}</dd></div><div><dt>{tr("规划轮次", "Planning rounds")}</dt><dd>{String(metadata.planning_attempts ?? "—")}</dd></div><div><dt>{tr("最小净空", "Minimum clearance")}</dt><dd>{metadata.minimum_clearance_m ? `${metadata.minimum_clearance_m} m` : "—"}</dd></div></dl>{confirmable && <button className="primary-button" disabled={busy} onClick={() => void onConfirm()}>{busy ? tr("正在准备仿真环境", "Preparing the simulation") : tr("确认并开始仿真", "Confirm and start simulation")}</button>}</div>;
  return summary;
}

function acceptedAssetExtensions(adapters: AssetSourceAdapter[], kind: "map" | "vehicle"): string {
  const kinds = kind === "map" ? new Set(["map", "world"]) : new Set(["vehicle"]);
  return [...new Set(adapters.flatMap((adapter) => adapter.asset_kinds.some((value) => kinds.has(value)) ? adapter.file_extensions : []))].sort().join(",");
}

function RemoteAssetImportDialog({
  kind,
  onClose,
  onImport,
}: {
  kind: "map" | "vehicle";
  onClose: () => void;
  onImport: (payload: {
    source_type: "direct_url" | "git";
    location: string;
    expected_sha256?: string;
    git_ref?: string;
    subpath?: string;
  }) => Promise<void>;
}) {
  const { tr } = useI18n();
  const [sourceType, setSourceType] = useState<"direct_url" | "git">("direct_url");
  const [location, setLocation] = useState("");
  const [expectedSha256, setExpectedSha256] = useState("");
  const [gitRef, setGitRef] = useState("");
  const [subpath, setSubpath] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    try {
      await onImport({
        source_type: sourceType,
        location: location.trim(),
        ...(expectedSha256.trim() ? { expected_sha256: expectedSha256.trim().toLowerCase() } : {}),
        ...(sourceType === "git" && gitRef.trim() ? { git_ref: gitRef.trim() } : {}),
        ...(sourceType === "git" && subpath.trim() ? { subpath: subpath.trim() } : {}),
      });
      onClose();
    } finally {
      setBusy(false);
    }
  };
  const title = kind === "map"
    ? tr("从远程来源导入地图", "Import map from remote source")
    : tr("从远程来源导入无人机", "Import drone from remote source");
  return <div className="modal-backdrop"><section className="remote-asset-dialog" role="dialog" aria-modal="true" aria-label={title}>
    <header><h2>{title}</h2><button aria-label={tr("关闭", "Close")} onClick={onClose}><X /></button></header>
    <div className="remote-source-tabs">
      <button className={sourceType === "direct_url" ? "selected" : ""} onClick={() => setSourceType("direct_url")}><Globe2 />{tr("文件网址", "File URL")}</button>
      <button className={sourceType === "git" ? "selected" : ""} onClick={() => setSourceType("git")}><GitBranch />Git</button>
    </div>
    <label><span>{sourceType === "git" ? tr("Git 仓库 HTTPS 地址", "Git repository HTTPS URL") : tr("文件 HTTPS 地址", "File HTTPS URL")}</span><input type="url" value={location} onChange={(event) => setLocation(event.target.value)} placeholder="https://" /></label>
    {sourceType === "git" ? <div className="remote-git-fields"><label><span>{tr("分支或标签", "Branch or tag")}</span><input value={gitRef} onChange={(event) => setGitRef(event.target.value)} /></label><label><span>{tr("资产子目录", "Asset subdirectory")}</span><input value={subpath} onChange={(event) => setSubpath(event.target.value)} /></label></div> : null}
    <label><span>{tr("预期 SHA-256（可选）", "Expected SHA-256 (optional)")}</span><input value={expectedSha256} onChange={(event) => setExpectedSha256(event.target.value)} maxLength={64} /></label>
    <footer><button className="secondary-button" onClick={onClose}>{tr("取消", "Cancel")}</button><button className="primary-button" disabled={busy || !location.trim() || Boolean(expectedSha256 && !/^[0-9a-fA-F]{64}$/u.test(expectedSha256))} onClick={() => void submit()}>{busy ? tr("正在导入", "Importing") : tr("导入", "Import")}</button></footer>
  </section></div>;
}

function AssetLibrary({ kind, jobs, versions, qualificationJobs, adapters, onOpenConnectors, onChanged, onError }: { kind: "map" | "vehicle"; jobs: AssetImportJob[]; versions: AssetVersionEntry[]; qualificationJobs: AssetQualificationJob[]; adapters: AssetSourceAdapter[]; onOpenConnectors: () => void; onChanged: () => Promise<void>; onError: (value: string) => void }) {
  const { locale, tr } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  const [remoteOpen, setRemoteOpen] = useState(false);
  const libraryItems = versions.filter((item, index, all) => (
    (kind === "map" ? ["map", "world"].includes(item.kind) : item.kind === "vehicle")
    && all.findIndex((candidate) => candidate.asset_id === item.asset_id) === index
  ));
  const upload = async (file?: File) => {
    if (!file) return;
    try {
      const created = await appApi.createAssetImportJob(kind, file);
      await appApi.processAssetImportJob(created.job_id);
      await onChanged();
    }
    catch (value) { onError(humanError(value, locale)); }
  };
  const cancel = async (jobId: string) => {
    try { await appApi.cancelAssetImportJob(jobId); await onChanged(); }
    catch (value) { onError(humanError(value, locale)); }
  };
  const submitCompanion = async (job: AssetImportJob, file: File) => {
    try { await appApi.submitAssetImportCompanionResult(job, file); await onChanged(); }
    catch (value) { onError(humanError(value, locale)); }
  };
  const importRemote = async (payload: Parameters<typeof appApi.createRemoteAssetImportJob>[1]) => {
    try {
      const created = await appApi.createRemoteAssetImportJob(kind, payload);
      await appApi.processAssetImportJob(created.job_id);
      await onChanged();
    } catch (value) {
      onError(humanError(value, locale));
      throw value;
    }
  };
  return <main className="library-page"><div className="page-toolbar"><div className="asset-heading-actions"><button className="square-add" aria-label={tr("从远程来源导入", "Import from remote source")} onClick={() => setRemoteOpen(true)}><Globe2 /></button><button className="square-add" aria-label={tr("连接外部建模软件", "Connect external modeling software")} onClick={onOpenConnectors}><Link2 /></button><button className="square-add" aria-label={kind === "map" ? tr("导入地图", "Import map") : tr("导入无人机", "Import drone")} onClick={() => input.current?.click()}><Plus /></button></div><input ref={input} type="file" accept={acceptedAssetExtensions(adapters, kind)} hidden onChange={(event) => void upload(event.target.files?.[0])} /></div>
    {remoteOpen ? <RemoteAssetImportDialog kind={kind} onClose={() => setRemoteOpen(false)} onImport={importRemote} /> : null}
    {jobs.length > 0 && <section className="asset-import-jobs"><h2>{tr("导入与资格认证", "Import and qualification")}</h2>{jobs.map((job) => <AssetImportJobRow key={job.job_id} job={job} onCompanion={submitCompanion} onCancel={cancel} onOpenPlugins={onOpenConnectors} />)}</section>}
    <QualificationPanel versions={versions} jobs={qualificationJobs} onChanged={onChanged} onError={onError} />
    {libraryItems.length ? <div className="asset-grid">{libraryItems.map((item) => {
      const name = typeof item.asset_ir.name === "string" ? item.asset_ir.name : item.asset_id;
      const qualified = item.maturity === "qualified";
      return <article className="asset-card" key={`${item.asset_id}@${item.content_sha256}`}><div className={`asset-visual ${kind}`}>{kind === "map" ? <Map /> : <Plane />}</div><div className="asset-info"><div><strong>{localizedAssetName({ asset_id: item.asset_id, name }, locale)}</strong><small>{item.asset_id}</small></div><span className={`status ${qualified ? "qualified" : "draft"}`}>{qualified ? tr("合格", "Qualified") : tr("认证中", "Qualifying")}</span></div></article>;
    })}</div> : <EmptyLibrary kind={kind} onClick={() => input.current?.click()} />}
  </main>;
}

function AssetImportJobRow({ job, onCompanion, onCancel, onOpenPlugins }: { job: AssetImportJob; onCompanion: (job: AssetImportJob, file: File) => Promise<void>; onCancel: (jobId: string) => Promise<void>; onOpenPlugins: () => void }) {
  const { tr } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  const qualificationInputs = new Set(["qualification_evidence", "qualification_environment_versions", "local_qualification_run"]);
  const adapterPluginRequired = job.state === "needs_input" && job.required_inputs.some((value) => value.startsWith("plugin_adapter:"));
  const companionRequired = job.state === "needs_input" && job.required_inputs.some((value) => !qualificationInputs.has(value));
  const terminal = ["qualified", "failed", "cancelled"].includes(job.state);
  return <article><div><strong>{job.source_name}</strong><span>{assetImportStateLabel(job.state, tr)}</span></div><progress value={job.progress_percent} max={100} /><small>{requiredInputLabel(job.required_inputs[0], tr) ?? (job.issue_codes.length ? tr("存在需要处理的问题", "An issue needs attention") : `${job.progress_percent}%`)}</small><AssetIssueDetails jobId={job.job_id} kind="import" enabled={Boolean(job.issue_codes.length || job.required_inputs.length)} /><div className="asset-job-actions">{adapterPluginRequired && <button onClick={onOpenPlugins}>{tr("打开插件", "Open plugins")}</button>}{companionRequired && <><button onClick={() => input.current?.click()}>{tr("导入连接器结果", "Import connector result")}</button><input ref={input} type="file" accept=".ddpkg" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) void onCompanion(job, file); event.currentTarget.value = ""; }} /></>}{!terminal && <button onClick={() => void onCancel(job.job_id)}>{tr("取消", "Cancel")}</button>}</div></article>;
}

function AssetIssueDetails({ jobId, kind, enabled }: { jobId: string; kind: "import" | "qualification"; enabled: boolean }) {
  const { locale, tr } = useI18n();
  const [issues, setIssues] = useState<AssetIssue[]>([]);
  useEffect(() => {
    let active = true;
    if (!enabled) { setIssues([]); return () => { active = false; }; }
    const request = kind === "import"
      ? appApi.getAssetImportJobIssues(jobId)
      : appApi.getAssetQualificationJobIssues(jobId);
    void request.then((report) => { if (active) setIssues(report.issues); }).catch(() => { if (active) setIssues([]); });
    return () => { active = false; };
  }, [enabled, jobId, kind]);
  if (!issues.length) return null;
  const localeKey = locale === "en-US" ? "en-US" : "zh-CN";
  return <details className="asset-issue-details"><summary>{tr("查看问题与修复建议", "View issue and repair guidance")}</summary>{issues.map((issue) => <article key={`${issue.code}:${issue.location}`} data-severity={issue.severity}><div><strong>{issue.title[localeKey]}</strong><code>{issue.code}</code></div><small>{issue.location}</small><p>{issue.detail[localeKey]}</p><span>{issue.actions.map((action) => action[localeKey]).join(" · ")}</span></article>)}</details>;
}

function QualificationEvidenceDetails({ job }: { job: AssetQualificationJob }) {
  const { tr } = useI18n();
  const [evidence, setEvidence] = useState<AssetQualificationEvidence | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  if (job.state !== "qualified") return null;
  const load = async () => {
    setLoading(true);
    setFailed(false);
    try { setEvidence(await appApi.getAssetQualificationEvidence(job.job_id)); }
    catch { setFailed(true); }
    finally { setLoading(false); }
  };
  const runtime = evidence?.receipt.runtime_evidence;
  const gates = runtime?.gates ? Object.entries(runtime.gates) : [];
  const pluginChecks = evidence?.receipt.plugin_checks ?? [];
  const runtimeContracts = evidence?.runtime_contracts;
  const compatibleTargets = runtimeContracts
    ? runtimeContracts.map.simulation_targets.filter((mapTarget) => (
        runtimeContracts.vehicle.simulation_targets.some((vehicleTarget) => (
          vehicleTarget.simulator === mapTarget.simulator
          && vehicleTarget.simulator_version === mapTarget.simulator_version
          && vehicleTarget.ros_distribution === mapTarget.ros_distribution
          && vehicleTarget.autopilot === mapTarget.autopilot
        ))
      ))
    : [];
  return <details className="qualification-evidence-details" onToggle={(event) => { if (event.currentTarget.open && !evidence && !loading) void load(); }}>
    <summary>{loading ? tr("正在验证证据", "Verifying evidence") : tr("查看认证证据", "View qualification evidence")}</summary>
    {failed ? <p>{tr("证据校验失败，请重新运行认证。", "Evidence verification failed. Run qualification again.")}</p> : null}
    {evidence ? <>
      <dl>
        <div><dt>{tr("认证 ID", "Qualification ID")}</dt><dd>{evidence.qualification_id}</dd></div>
        <div><dt>{tr("地图哈希", "Map hash")}</dt><dd>{evidence.map_content_sha256.slice(0, 16)}</dd></div>
        <div><dt>{tr("无人机哈希", "Drone hash")}</dt><dd>{evidence.vehicle_content_sha256.slice(0, 16)}</dd></div>
        <div><dt>{tr("核心门禁", "Core gates")}</dt><dd>{gates.filter(([, passed]) => passed).length}/{gates.length}</dd></div>
        <div><dt>{tr("插件检查", "Plugin checks")}</dt><dd>{pluginChecks.filter((item) => item.accepted).length}/{pluginChecks.length}</dd></div>
        {runtimeContracts ? <>
          <div><dt>{tr("地图拓扑", "Map topology")}</dt><dd>{runtimeContracts.map.node_count} / {runtimeContracts.map.edge_count} / {runtimeContracts.map.named_entity_count}</dd></div>
          <div><dt>{tr("导航范围", "Navigation span")}</dt><dd>{Object.values(runtimeContracts.map.navigation_bounds_m.span).map((value) => value.toFixed(2)).join(" × ")} m</dd></div>
          <div><dt>{tr("无人机质量", "Drone mass")}</dt><dd>{runtimeContracts.vehicle.dry_mass_kg.toFixed(2)} / {runtimeContracts.vehicle.max_takeoff_mass_kg.toFixed(2)} kg</dd></div>
          <div><dt>{tr("碰撞半径", "Collision radius")}</dt><dd>{runtimeContracts.vehicle.body_radius_m.toFixed(3)} m</dd></div>
          <div><dt>{tr("合格航程", "Qualified range")}</dt><dd>{runtimeContracts.vehicle.qualified_range_m.toFixed(1)} m</dd></div>
          <div><dt>{tr("兼容运行目标", "Compatible runtime targets")}</dt><dd>{compatibleTargets.length}</dd></div>
        </> : null}
        {evidence.receipt.plugin_snapshot_sha256 ? <div><dt>{tr("插件快照", "Plugin snapshot")}</dt><dd>{evidence.receipt.plugin_snapshot_sha256.slice(0, 16)}</dd></div> : null}
      </dl>
      {compatibleTargets.length ? <ul className="qualification-plugin-checks">{compatibleTargets.map((target) => <li key={target.target_id}><span>{target.simulator} · {target.simulator_version} · {target.ros_distribution ?? tr("无 ROS", "No ROS")} · {target.autopilot}</span><strong>{tr("可运行", "Runnable")}</strong></li>)}</ul> : null}
      {pluginChecks.length ? <ul className="qualification-plugin-checks">{pluginChecks.map((check) => <li key={`${check.plugin_id}:${check.check_id}`}><span>{check.check_id}</span><strong>{check.accepted ? tr("通过", "Passed") : tr("拒绝", "Rejected")}</strong></li>)}</ul> : null}
    </> : null}
  </details>;
}

function assetImportStateLabel(state: AssetImportState, tr: (zh: string, en: string) => string): string {
  return ({
    created: tr("已创建", "Created"), quarantining: tr("正在隔离", "Quarantining"), parsing: tr("正在解析", "Parsing"), needs_input: tr("需要补充", "Needs input"), normalizing: tr("正在标准化", "Normalizing"), building: tr("正在构建", "Building"), validating: tr("正在验证", "Validating"), qualified: tr("已合格", "Qualified"), failed: tr("失败", "Failed"), cancelled: tr("已取消", "Cancelled"),
  })[state];
}

function QualificationPanel({ versions, jobs, onChanged, onError }: { versions: AssetVersionEntry[]; jobs: AssetQualificationJob[]; onChanged: () => Promise<void>; onError: (value: string) => void }) {
  const { locale, tr } = useI18n();
  const mapVersions = versions.filter((item) => item.kind === "map" || item.kind === "world");
  const vehicleVersions = versions.filter((item) => item.kind === "vehicle");
  const versionValue = (item: AssetVersionEntry) => `${item.asset_id}@${item.content_sha256}`;
  const [mapValue, setMapValue] = useState(() => mapVersions[0] ? versionValue(mapVersions[0]) : "");
  const [vehicleValue, setVehicleValue] = useState(() => vehicleVersions[0] ? versionValue(vehicleVersions[0]) : "");
  const [busy, setBusy] = useState<string | null>(null);
  useEffect(() => {
    if (!mapValue && mapVersions[0]) setMapValue(versionValue(mapVersions[0]));
    if (!vehicleValue && vehicleVersions[0]) setVehicleValue(versionValue(vehicleVersions[0]));
  }, [mapValue, mapVersions, vehicleValue, vehicleVersions]);
  const selected = (value: string, candidates: AssetVersionEntry[]) => candidates.find((item) => versionValue(item) === value);
  const act = async (key: string, action: () => Promise<unknown>) => {
    setBusy(key);
    try { await action(); await onChanged(); }
    catch (value) { onError(humanError(value, locale)); }
    finally { setBusy(null); }
  };
  const create = async () => {
    const map = selected(mapValue, mapVersions);
    const vehicle = selected(vehicleValue, vehicleVersions);
    if (!map || !vehicle) return;
    await act("create", async () => {
      const created = await appApi.createAssetQualificationJob({
        map_asset_id: map.asset_id,
        map_content_sha256: map.content_sha256,
        vehicle_asset_id: vehicle.asset_id,
        vehicle_content_sha256: vehicle.content_sha256,
      });
      await appApi.startAssetQualificationJob(created.job_id);
    });
  };
  if (!mapVersions.length || !vehicleVersions.length) return null;
  return <section className="qualification-panel">
    <div className="qualification-heading"><div><ShieldCheck /><h2>{tr("真实仿真资格认证", "Real simulation qualification")}</h2></div><button className="primary-button" disabled={busy !== null || !mapValue || !vehicleValue} onClick={() => void create()}><Play />{tr("开始认证", "Start qualification")}</button></div>
    <div className="qualification-pair">
      <label><span>{tr("地图包", "Map package")}</span><select value={mapValue} onChange={(event) => setMapValue(event.target.value)}>{mapVersions.map((item) => <option key={versionValue(item)} value={versionValue(item)}>{item.asset_ir.name ?? item.asset_id} · {assetMaturityLabel(item.maturity, tr)}</option>)}</select></label>
      <label><span>{tr("无人机包", "Drone package")}</span><select value={vehicleValue} onChange={(event) => setVehicleValue(event.target.value)}>{vehicleVersions.map((item) => <option key={versionValue(item)} value={versionValue(item)}>{item.asset_ir.name ?? item.asset_id} · {assetMaturityLabel(item.maturity, tr)}</option>)}</select></label>
    </div>
    {jobs.length > 0 && <div className="qualification-jobs">{jobs.slice(0, 5).map((job) => <article key={job.job_id}><div className="qualification-job-title"><strong>{job.map_asset_id} + {job.vehicle_asset_id}</strong><span>{assetQualificationStateLabel(job.state, tr)}</span></div><progress value={job.progress_percent} max={100} /><small>{job.qualification_id ?? (job.issue_codes.length ? tr("认证存在需要处理的问题", "Qualification needs attention") : `${job.progress_percent}%`)}</small><AssetIssueDetails jobId={job.job_id} kind="qualification" enabled={Boolean(job.issue_codes.length)} /><QualificationEvidenceDetails job={job} /><div className="qualification-job-actions">{["created", "paused"].includes(job.state) && <button disabled={busy !== null} onClick={() => void act(`${job.job_id}:start`, () => appApi.startAssetQualificationJob(job.job_id))}><Play />{job.state === "paused" ? tr("恢复", "Resume") : tr("开始", "Start")}</button>}{["preparing", "running", "validating"].includes(job.state) && <button disabled={busy !== null} onClick={() => void act(`${job.job_id}:pause`, () => appApi.pauseAssetQualificationJob(job.job_id))}><Pause />{tr("暂停", "Pause")}</button>}{!["qualified", "failed", "cancelled"].includes(job.state) && <button className="danger-text" disabled={busy !== null} onClick={() => void act(`${job.job_id}:cancel`, () => appApi.cancelAssetQualificationJob(job.job_id))}><X />{tr("取消", "Cancel")}</button>}</div></article>)}</div>}
  </section>;
}

function assetMaturityLabel(value: AssetVersionEntry["maturity"], tr: (zh: string, en: string) => string): string {
  return ({
    visual_only: tr("仅视觉", "Visual only"),
    physics_ready: tr("物理就绪", "Physics ready"),
    simulation_ready: tr("仿真就绪", "Simulation ready"),
    flight_ready: tr("可飞行", "Flight ready"),
    qualified: tr("已认证", "Qualified"),
  })[value];
}

function assetQualificationStateLabel(state: AssetQualificationState, tr: (zh: string, en: string) => string): string {
  return ({
    created: tr("已创建", "Created"), preparing: tr("正在准备", "Preparing"), running: tr("正在运行 Gazebo 与 PX4", "Running Gazebo and PX4"), validating: tr("正在绑定证据", "Binding evidence"), paused: tr("已暂停", "Paused"), qualified: tr("已认证", "Qualified"), failed: tr("失败", "Failed"), cancelled: tr("已取消", "Cancelled"),
  })[state];
}

function requiredInputLabel(value: string | undefined, tr: (zh: string, en: string) => string): string | null {
  if (value?.startsWith("plugin_adapter:")) return tr("需要安装并启用匹配此文件格式的转换插件", "Install and enable a conversion plugin for this file format");
  if (value === "qualification_evidence") return tr("需要在本机仿真环境中完成资格验证", "Qualification evidence is required from the local simulation runtime");
  if (value === "qualification_environment_versions") return tr("资格证据与本机 Runtime 版本不匹配", "Qualification evidence does not match the local Runtime versions");
  if (value === "local_qualification_run") return tr("等待本机运行几何、Gazebo 与 PX4 资格流水线", "Waiting for the local geometry, Gazebo and PX4 qualification pipeline");
  if (value === "blender_phobos_companion") return tr("需要安装并启用 Blender + Phobos 连接器", "Install and enable the Blender + Phobos connector");
  if (value === "isolated_xacro_expansion") return tr("需要在隔离的 ROS 2 Runtime 中展开 Xacro", "Expand Xacro inside the isolated ROS 2 Runtime");
  if (value === "solidworks_connector") return tr("需要安装 SolidWorks 连接器", "Install the SolidWorks connector");
  if (value === "fusion_connector") return tr("需要安装 Autodesk Fusion 连接器", "Install the Autodesk Fusion connector");
  if (value === "freecad_connector") return tr("需要安装 FreeCAD 连接器", "Install the FreeCAD connector");
  if (value === "cad_semantics_and_physics") return tr("需要补充 CAD 语义、质量、惯量与连接关系", "Add CAD semantics, mass, inertia and connection data");
  if (value === "gis_connector") return tr("需要安装 GIS / DEM 连接器", "Install the GIS / DEM connector");
  return value ?? null;
}

function EmptyLibrary({ kind, onClick }: { kind: "map" | "vehicle"; onClick: () => void }) {
  const { tr } = useI18n();
  return <div className="empty-library">{kind === "map" ? <Map /> : <Plane />}<h2>{kind === "map" ? tr("还没有地图", "No maps yet") : tr("还没有无人机", "No drones yet")}</h2><button className="primary-button" onClick={onClick}><Upload />{tr("导入资产包", "Import asset package")}</button></div>;
}

function PluginCategoryIcon({ id }: { id: string }) {
  if (id === "safety") return <ShieldCheck />;
  if (id === "planning" || id === "assets") return <Map />;
  if (id === "flight-control") return <Plane />;
  if (id === "interaction") return <Mic />;
  if (id === "harness" || id === "models") return <BrainCircuit />;
  if (id === "input" || id === "context") return <Braces />;
  if (id === "tools") return <Wrench />;
  if (id === "runtime") return <Activity />;
  if (id === "evidence" || id === "evaluation") return <Gauge />;
  if (id === "simulation") return <FlaskConical />;
  if (id === "interface") return <Settings />;
  if (id === "general") return <Bot />;
  return <Sparkles />;
}

function profileMembers(item: PluginEntry): string[] {
  const value = item.capabilities.find((capability) => capability.kind === "harness-profile")?.metadata.recommended_plugins;
  return Array.isArray(value) ? value.filter((pluginId): pluginId is string => typeof pluginId === "string") : [];
}

function PluginPanelWidget({ widget, configuration, onConfigurationChange }: { widget: PluginPanel["sections"][number]["widgets"][number]; configuration: Record<string, unknown>; onConfigurationChange: (value: Record<string, unknown>) => void }) {
  const { locale, tr } = useI18n();
  const label = localizedDynamicLabel(widget.label, widget.widget_id, locale);
  if (widget.kind === "configuration-form" && widget.schema) return <div className="panel-form"><JsonSchemaForm schema={widget.schema as JsonSchema} value={configuration} onChange={onConfigurationChange} /></div>;
  const value = widget.source === "static" ? widget.value : widget.resolved;
  if (widget.kind === "metric") return <div className="panel-metric"><span>{label}</span><strong>{typeof value === "number" || typeof value === "string" ? value : "—"}</strong>{widget.unit && <small>{widget.unit}</small>}</div>;
  if (widget.kind === "status") return <div className="panel-status"><span className={value === true || value === "healthy" || value === "ready" ? "ok" : ""} />{label}<strong>{String(value ?? tr("未知", "Unknown"))}</strong></div>;
  if (["log", "table", "replay", "telemetry"].includes(widget.kind)) return <div className={`panel-data panel-${widget.kind}`}><span>{label}</span><pre>{JSON.stringify(value ?? [], null, 2)}</pre></div>;
  return <p className="panel-text"><strong>{label}</strong>{String(value ?? "")}</p>;
}

function PluginLibrary({ items, adapters, credentials, threadId, onChanged, onError }: { items: PluginEntry[]; adapters: AssetSourceAdapter[]; credentials: ConnectorCredentialReference[]; threadId: string | null; onChanged: () => Promise<void>; onError: (value: string) => void }) {
  const { locale, tr, number, dateTime } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<PluginDetail | null>(null);
  const [configuration, setConfiguration] = useState<Record<string, unknown>>({});
  const [panel, setPanel] = useState<PluginPanel | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [credentialOpen, setCredentialOpen] = useState(false);
  const [credentialName, setCredentialName] = useState("");
  const [credentialSecret, setCredentialSecret] = useState("");
  const [showCredentialSecret, setShowCredentialSecret] = useState(false);
  const [credentialScopes, setCredentialScopes] = useState<string[]>([]);
  const [governanceOpen, setGovernanceOpen] = useState(false);
  const [governance, setGovernance] = useState<PluginGovernancePolicy | null>(null);
  const [marketplaceOpen, setMarketplaceOpen] = useState(false);
  const [marketplace, setMarketplace] = useState<PluginMarketplaceCatalog | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const credentialPlugins = useMemo(() => items.filter((item) => item.permissions.includes("credential.reference")), [items]);
  const categories = useMemo(() => {
    const categoryMap = new globalThis.Map<string, { id: string; label: string; order: number; slots: globalThis.Map<string, { id: string; label: string; order: number; mode: "single" | "multiple" | "pipeline"; items: PluginEntry[] }> }>();
    const normalizedQuery = query.trim().toLocaleLowerCase(locale);
    const visible = normalizedQuery ? items.filter((item) => [item.name, localizedPluginName(item, locale), item.plugin_id, item.description, item.placement.category_label, localizedCategory(item.placement.category_id, item.placement.category_label, locale), item.placement.slot_label, localizedSlot(item.placement.slot_id, item.placement.slot_label, locale)].some((value) => value.toLocaleLowerCase(locale).includes(normalizedQuery))) : items;
    for (const item of visible) {
      const placement = item.placement;
      const category = categoryMap.get(placement.category_id) ?? { id: placement.category_id, label: placement.category_label, order: placement.category_order, slots: new globalThis.Map<string, { id: string; label: string; order: number; mode: "single" | "multiple" | "pipeline"; items: PluginEntry[] }>() };
      const slot = category.slots.get(placement.slot_id) ?? { id: placement.slot_id, label: placement.slot_label, order: placement.slot_order, mode: placement.activation_mode, items: [] };
      slot.items.push(item); category.slots.set(slot.id, slot); categoryMap.set(category.id, category);
    }
    return [...categoryMap.values()].sort((a, b) => a.order - b.order || a.label.localeCompare(b.label, locale)).map((category) => ({
      ...category,
      slots: [...category.slots.values()].sort((a, b) => a.order - b.order || a.label.localeCompare(b.label, locale)).map((slot) => ({
        ...slot,
        items: slot.items.sort((a, b) => a.placement.plugin_order - b.placement.plugin_order || localizedPluginName(a, locale).localeCompare(localizedPluginName(b, locale), locale)),
      })),
    }));
  }, [items, locale, query]);
  const loadDetail = async (id: string) => {
    setBusy(`${id}:detail`);
    try {
      const value = await appApi.getPlugin(id);
      setDetail(value);
      setConfiguration(value.configuration);
      setPanel(value.runtime_kind === "ui-declarative" && value.enabled ? await appApi.getPluginPanel(id, threadId) : null);
    } catch (value) { onError(humanError(value, locale)); }
    finally { setBusy(null); }
  };
  const choose = async (id: string) => {
    if (selected === id) { setSelected(null); setDetail(null); setPanel(null); return; }
    setSelected(id);
    await loadDetail(id);
  };
  const act = async (key: string, action: () => Promise<unknown>, refreshId: string | null = selected) => {
    setBusy(key);
    try {
      await action();
      await onChanged();
      if (refreshId) await loadDetail(refreshId);
    }
    catch (value) { onError(humanError(value, locale)); }
    finally { setBusy(null); }
  };
  const upload = async (file?: File) => {
    if (!file) return;
    await act("import", () => appApi.importPlugin(file));
    if (input.current) input.current.value = "";
  };
  const toggle = async (item: PluginEntry) => {
    if (!item.disable_allowed || (item.enabled && item.slot_required)) return;
    if (!item.enabled && !item.builtin && !["verified", "local-approved"].includes(item.trust_status)) {
      setSelected(item.plugin_id);
      await loadDetail(item.plugin_id);
      return onError(tr("启用前请核对发布者、权限和包哈希，并批准这个精确版本", "Review the publisher, permissions, and package hash, then approve this exact version before enabling it."));
    }
    await act(item.plugin_id, () => appApi.setPlugin(item.plugin_id, !item.enabled));
  };
  const saveConfiguration = async (item: PluginEntry) => {
    try {
      await act(`${item.plugin_id}:configuration`, () => appApi.configurePlugin(item.plugin_id, configuration));
    } catch (value) { onError(humanError(value, locale)); }
  };
  const createCredential = async () => {
    if (!credentialName.trim() || !credentialSecret || credentialScopes.length === 0) return;
    await act("credential:create", () => appApi.createConnectorCredential({ display_name: credentialName.trim(), secret: credentialSecret, allowed_plugin_ids: credentialScopes }), null);
    setCredentialName(""); setCredentialSecret(""); setCredentialScopes([]); setShowCredentialSecret(false);
  };
  const openGovernance = async () => {
    const next = !governanceOpen;
    setGovernanceOpen(next); setMarketplaceOpen(false); setCredentialOpen(false);
    if (!next) return;
    try { setGovernance((await appApi.getPluginGovernance()).policy); }
    catch (value) { onError(humanError(value, locale)); }
  };
  const saveGovernance = async () => {
    if (!governance) return;
    await act("governance:save", () => appApi.replacePluginGovernance(governance), null);
  };
  const openMarketplace = async () => {
    const next = !marketplaceOpen;
    setMarketplaceOpen(next); setGovernanceOpen(false); setCredentialOpen(false);
    if (!next) return;
    try { setMarketplace(await appApi.getPluginMarketplace()); }
    catch (value) { onError(humanError(value, locale)); }
  };
  const saveMarketplaceSources = async (sources: PluginMarketplaceSource[]) => {
    await act("marketplace:sources", () => appApi.replacePluginMarketplaceSources(sources), null);
    setMarketplace(await appApi.getPluginMarketplace());
  };
  const addMarketplaceSource = async () => {
    if (!marketplace || !sourceName.trim() || !sourceUrl.trim()) return;
    const sourceId = `custom-${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`;
    await saveMarketplaceSources([...marketplace.sources, { schema_version: "dronedream.plugin-marketplace-source.v1", source_id: sourceId, name: sourceName.trim(), index_url: sourceUrl.trim(), enabled: true }]);
    setSourceName(""); setSourceUrl("");
  };
  const runPanelAction = async (item: PluginEntry, action: PluginPanel["sections"][number]["actions"][number]) => {
    const confirmation = localizedDynamicDescription(action.confirmation ?? undefined, locale)
      ?? tr("确认执行这个操作？", "Continue with this action?");
    if (action.confirmation && !window.confirm(confirmation)) return;
    if (action.action_id === "panel.refresh") return loadDetail(item.plugin_id);
    if (action.action_id === "plugin.healthcheck") return act(`${item.plugin_id}:panel-health`, () => appApi.checkPlugin(item.plugin_id));
    if (action.action_id === "plugin.disable") {
      await act(`${item.plugin_id}:panel-disable`, () => appApi.setPlugin(item.plugin_id, false), null);
      setSelected(null); setDetail(null); setPanel(null);
    }
  };
  return <main className="library-page plugin-library">
    <div className="plugin-page-inner"><div className="page-toolbar"><div className="plugin-heading-actions"><input className="plugin-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr("搜索插件", "Search plugins")} aria-label={tr("搜索插件", "Search plugins")} /><button className={`square-add ${marketplaceOpen ? "active" : ""}`} aria-label={tr("插件市场", "Plugin marketplace")} onClick={() => void openMarketplace()}><Sparkles /></button><button className={`square-add ${governanceOpen ? "active" : ""}`} aria-label={tr("插件治理", "Plugin governance")} onClick={() => void openGovernance()}><ShieldCheck /></button><button className={`square-add ${credentialOpen ? "active" : ""}`} aria-label={tr("管理连接器凭证", "Manage connector credentials")} onClick={() => { setCredentialOpen(!credentialOpen); setGovernanceOpen(false); setMarketplaceOpen(false); }}><KeyRound /></button><button className="square-add" aria-label={tr("导入插件", "Import plugin")} onClick={() => input.current?.click()} disabled={busy === "import"}><Plus /></button></div><input ref={input} type="file" accept=".zip" hidden onChange={(event) => void upload(event.target.files?.[0])} /></div>
    {governanceOpen && governance && <section className="plugin-governance"><div className="credential-manager-heading"><ShieldCheck /><h2>{tr("插件治理", "Plugin governance")}</h2></div><div className="governance-grid"><label><span>{tr("策略", "Policy")}</span><select value={governance.mode} onChange={(event) => setGovernance({ ...governance, mode: event.target.value as "personal" | "managed" })}><option value="personal">{tr("个人", "Personal")}</option><option value="managed">{tr("企业托管", "Managed")}</option></select></label><label><span>{tr("外部插件上限", "External plugin limit")}</span><input type="number" min={0} max={2000} value={governance.maximum_external_plugins} onChange={(event) => setGovernance({ ...governance, maximum_external_plugins: Number(event.target.value) })} /></label><label className="check-row"><input type="checkbox" checked={governance.require_verified_signatures} onChange={(event) => setGovernance({ ...governance, require_verified_signatures: event.target.checked })} />{tr("强制签名", "Require signatures")}</label><label className="check-row"><input type="checkbox" checked={governance.allow_local_approval} onChange={(event) => setGovernance({ ...governance, allow_local_approval: event.target.checked })} />{tr("允许本机批准", "Allow local approval")}</label><label className="wide"><span>{tr("允许的发布者", "Allowed publishers")}</span><input value={governance.allowed_publishers.join(", ")} placeholder={tr("留空表示不限制", "Leave blank for no restriction")} onChange={(event) => setGovernance({ ...governance, allowed_publishers: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) })} /></label><label className="wide"><span>{tr("允许的插件 ID", "Allowed plugin IDs")}</span><input value={governance.allowed_plugin_ids.join(", ")} placeholder={tr("留空表示不限制", "Leave blank for no restriction")} onChange={(event) => setGovernance({ ...governance, allowed_plugin_ids: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) })} /></label><label className="wide"><span>{tr("禁用权限", "Denied permissions")}</span><input value={governance.denied_permissions.join(", ")} placeholder={tr("例如 network.external, process.spawn", "For example: network.external, process.spawn")} onChange={(event) => setGovernance({ ...governance, denied_permissions: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) })} /></label></div><button className="primary-button" disabled={busy !== null} onClick={() => void saveGovernance()}>{tr("保存策略", "Save policy")}</button></section>}
    {marketplaceOpen && marketplace && <section className="plugin-marketplace"><div className="credential-manager-heading"><Sparkles /><h2>{tr("插件市场", "Plugin marketplace")}</h2></div><div className="marketplace-source-create"><input value={sourceName} placeholder={tr("源名称", "Source name")} onChange={(event) => setSourceName(event.target.value)} /><input value={sourceUrl} placeholder="https://…/index.json" onChange={(event) => setSourceUrl(event.target.value)} /><button disabled={!sourceName.trim() || !sourceUrl.trim()} onClick={() => void addMarketplaceSource()}>{tr("添加源", "Add source")}</button></div>{marketplace.sources.map((source) => <div className="marketplace-source" key={source.source_id}><label><input type="checkbox" checked={source.enabled} onChange={() => void saveMarketplaceSources(marketplace.sources.map((item) => item.source_id === source.source_id ? { ...item, enabled: !item.enabled } : item))} />{source.name}</label><span>{source.index_url}</span><button className="danger-text" onClick={() => void saveMarketplaceSources(marketplace.sources.filter((item) => item.source_id !== source.source_id))}>{tr("移除", "Remove")}</button></div>)}{marketplace.errors.map((error) => <div className="plugin-error" key={error.source_id}>{error.source_id} · {error.issue_code}</div>)}<div className="marketplace-list">{marketplace.entries.map((entry) => { const description = localizedDynamicDescription(entry.description, locale); return <article key={`${entry.source_id}:${entry.plugin_id}:${entry.version}`}><div><strong>{locale === "en-US" ? localizedDynamicLabel(entry.name, entry.plugin_id, locale) : entry.name}</strong><span>{entry.publisher}</span>{description ? <p>{description}</p> : null}</div><button disabled={busy !== null} onClick={() => void act(`market:${entry.plugin_id}`, () => appApi.installMarketplacePlugin(entry.source_id, entry.plugin_id, entry.version), null)}>{tr("安装", "Install")}</button></article>; })}</div></section>}
    {credentialOpen && <section className="credential-manager"><div className="credential-manager-heading"><KeyRound /><h2>{tr("连接器凭证", "Connector credentials")}</h2></div><div className="credential-create"><label><span>{tr("名称", "Name")}</span><input value={credentialName} maxLength={80} onChange={(event) => setCredentialName(event.target.value)} /></label><label><span>API Key</span><span className="secret-input"><input type={showCredentialSecret ? "text" : "password"} autoComplete="off" value={credentialSecret} onChange={(event) => setCredentialSecret(event.target.value)} /><button type="button" onClick={() => setShowCredentialSecret(!showCredentialSecret)} aria-label={showCredentialSecret ? tr("隐藏 API Key", "Hide API Key") : tr("显示 API Key", "Show API Key")}>{showCredentialSecret ? <EyeOff /> : <Eye />}</button></span></label><div className="credential-scopes"><span>{tr("授权插件", "Authorized plugins")}</span><div>{credentialPlugins.map((plugin) => <label key={plugin.plugin_id}><input type="checkbox" checked={credentialScopes.includes(plugin.plugin_id)} onChange={() => setCredentialScopes((current) => current.includes(plugin.plugin_id) ? current.filter((id) => id !== plugin.plugin_id) : [...current, plugin.plugin_id])} />{localizedPluginName(plugin, locale)}</label>)}</div></div><button className="primary-button" disabled={busy !== null || !credentialName.trim() || !credentialSecret || credentialScopes.length === 0} onClick={() => void createCredential()}>{tr("保存凭证", "Save credential")}</button></div>{credentials.length > 0 && <div className="credential-list">{credentials.map((credential) => <article key={credential.reference}><div><strong>{credential.display_name}</strong><span>{credential.reference} · {credential.allowed_plugin_ids.length} {tr("个插件", "plugins")}</span></div><button className="danger-icon" aria-label={`${tr("删除", "Delete")} ${credential.display_name}`} onClick={() => void act(`credential:${credential.reference}`, () => appApi.deleteConnectorCredential(credential.reference), null)}><Trash2 /></button></article>)}</div>}</section>}
    <section className="connector-adapters"><div className="credential-manager-heading"><Link2 /><h2>{tr("建模与仿真连接器", "Modeling and simulation connectors")}</h2></div><div className="connector-adapter-grid">{adapters.map((adapter) => <article key={adapter.adapter_id}><div><strong>{adapter.name}</strong><span>{adapter.source_formats.join(" · ")}</span></div><small className={adapter.enabled ? "ready" : "unavailable"}>{adapter.enabled ? tr("可用", "Available") : adapter.availability === "companion_required" ? tr("需要配套程序", "Companion required") : tr("需要插件", "Plugin required")}</small></article>)}</div></section>
    <div className="plugin-categories">{categories.map((category) => <section className="plugin-category" key={category.id}>
      <div className="plugin-category-heading"><span><PluginCategoryIcon id={category.id} /></span><h2>{localizedCategory(category.id, category.label, locale)}</h2></div>
      {category.slots.map((slot) => <div className="plugin-slot" key={slot.id}>
        <div className="plugin-slot-heading"><h3>{localizedSlot(slot.id, slot.label, locale)}</h3><span>{slot.mode === "single" ? tr("单选", "Single choice") : slot.mode === "pipeline" ? tr("有序管线", "Ordered pipeline") : tr("可多选", "Multiple choice")}</span></div>
        <div className="plugin-table">{slot.items.map((item) => <section key={item.plugin_id} className={`plugin-entry ${selected === item.plugin_id ? "expanded" : ""}`}>
      <article onClick={() => void choose(item.plugin_id)}>
        <div className="plugin-icon"><PluginCategoryIcon id={category.id} /></div>
        <div className="plugin-name"><strong>{localizedPluginName(item, locale)}</strong><span>{item.plugin_id}</span></div>
        <span className={`plugin-health ${item.health} trust-${item.trust_status}`}>{item.trust_status === "revoked" ? tr("已撤销", "Revoked") : item.trust_status === "unverified" ? tr("待信任", "Approval required") : item.health === "healthy" ? tr("正常", "Healthy") : item.status === "quarantined" ? tr("已隔离", "Quarantined") : item.enabled ? tr("检查中", "Checking") : tr("已停用", "Disabled")}</span>
        <button aria-label={item.disable_allowed && !(item.enabled && item.slot_required) ? (item.enabled ? tr("停用插件", "Disable plugin") : tr("启用插件", "Enable plugin")) : tr("系统必需插件", "Required system plugin")} disabled={!item.disable_allowed || (item.enabled && item.slot_required) || busy === item.plugin_id} className={`switch ${item.enabled ? "on" : ""} ${!item.disable_allowed || (item.enabled && item.slot_required) ? "locked" : ""}`} onClick={(event) => { event.stopPropagation(); void toggle(item); }}><span /></button>
      </article>
      {selected === item.plugin_id && <div className="plugin-detail">
        <p>{localizedPluginDescription(item, locale)}</p>
        <dl><div><dt>{tr("发布者", "Publisher")}</dt><dd>{item.publisher}</dd></div><div><dt>{tr("信任", "Trust")}</dt><dd>{item.trust_status === "verified" ? tr("签名已验证", "Signature verified") : item.trust_status === "local-approved" ? tr("本机已批准", "Approved locally") : item.trust_status === "revoked" ? tr("已撤销", "Revoked") : tr("未验证", "Unverified")}</dd></div><div><dt>{tr("运行方式", "Runtime")}</dt><dd>{localizedSystemTerm(item.runtime_kind, locale)}</dd></div><div><dt>{tr("故障策略", "Failure policy")}</dt><dd>{localizedSystemTerm(item.placement.failure_mode, locale)}</dd></div><div><dt>{tr("切换策略", "Swap policy")}</dt><dd>{localizedSystemTerm(item.placement.swap_policy, locale)}</dd></div>{item.placement.activation_mode === "pipeline" && <div><dt>{tr("管线顺序", "Pipeline order")}</dt><dd>{item.placement.pipeline_order}</dd></div>}<div><dt>{tr("权限", "Permissions")}</dt><dd>{item.permissions.length ? item.permissions.join(" · ") : tr("无", "None")}</dd></div><div><dt>{tr("能力", "Capabilities")}</dt><dd>{item.capabilities.map((capability) => localizedDynamicLabel(capability.name, capability.capability_id, locale)).join(" · ")}</dd></div>{profileMembers(item).length > 0 && <div><dt>{tr("组合", "Bundle")}</dt><dd title={profileMembers(item).join(" · ")}>{profileMembers(item).length} {tr("个插件", "plugins")}</dd></div>}<div><dt>{tr("包哈希", "Package hash")}</dt><dd title={item.package_sha256}>{item.package_sha256.slice(0, 16)}</dd></div></dl>
        {!item.builtin && item.trust_status === "unverified" && <div className="plugin-trust-warning">{tr("批准只绑定当前包哈希；更新后的版本需要重新核对。", "Approval is bound to the current package hash. Updated versions must be reviewed again.")}</div>}
        {item.last_error && <div className="plugin-error">{localeSafeError(item.last_error, locale, { zh: "插件运行异常，请查看诊断信息", en: "The plugin reported an error. Inspect diagnostics." })}</div>}
        {busy === `${item.plugin_id}:detail` && <span className="plugin-loading">{tr("正在读取插件状态", "Loading plugin status")}</span>}
        {detail?.plugin_id === item.plugin_id && <>
          {detail!.versions.length > 1 && <div className="plugin-version-list"><strong>{tr("已安装包", "Installed packages")}</strong>{detail!.versions.map((version) => <div key={version.version}><span>{version.package_sha256.slice(0, 12)}{version.version === detail!.version ? tr(" · 当前", " · Current") : ""}</span><span>{version.trust_status === "verified" ? tr("签名已验证", "Signature verified") : version.trust_status === "local-approved" ? tr("本机已批准", "Approved locally") : version.trust_status === "revoked" ? tr("已撤销", "Revoked") : tr("待批准", "Approval required")}</span>{!item.builtin && version.trust_status === "unverified" && <button disabled={busy !== null} onClick={() => void act(`${item.plugin_id}:${version.version}:trust`, () => appApi.trustLocalPluginVersion(item.plugin_id, version.version))}>{tr("批准", "Approve")}</button>}{version.version !== detail!.version && ["verified", "local-approved"].includes(version.trust_status) && <button disabled={busy !== null} onClick={() => void act(`${item.plugin_id}:${version.version}:promote`, () => appApi.promotePlugin(item.plugin_id, version.version))}>{tr("设为当前", "Set current")}</button>}</div>)}</div>}
          {Object.keys((detail!.manifest.configuration_schema as Record<string, unknown> | undefined) ?? {}).length > 0 && <div className="plugin-configuration"><JsonSchemaForm schema={detail!.manifest.configuration_schema as JsonSchema} value={configuration} onChange={setConfiguration} credentialReferences={credentials.filter((credential) => credential.allowed_plugin_ids.includes(item.plugin_id))} /><button onClick={() => void saveConfiguration(item)}>{tr("保存配置", "Save configuration")}</button></div>}
          {detail!.events.length > 0 && <small className="plugin-last-event">{tr("最近操作", "Latest action")}: {localizedSystemTerm(detail!.events[0].operation, locale)} · {dateTime(detail!.events[0].created_at)}</small>}
          {detail!.usage_summary.calls > 0 && <div className="plugin-usage-summary"><strong>{tr("运行审计", "Runtime audit")}</strong><span>{number(detail!.usage_summary.calls)} {tr("次调用", "calls")}</span><span>{number(detail!.usage_summary.errors)} {tr("次失败", "failures")}</span><span>{tr("平均", "Average")} {Math.round(detail!.usage_summary.average_duration_ms)} ms</span><span>{number(detail!.usage_summary.input_bytes + detail!.usage_summary.output_bytes)} bytes</span></div>}
          {detail!.governance_decisions.length > 0 && <small className="plugin-last-event">{tr("最近治理", "Latest governance decision")}: {localizedSystemTerm(detail!.governance_decisions[0].operation, locale)} · {detail!.governance_decisions[0].accepted ? tr("通过", "Accepted") : detail!.governance_decisions[0].issue_codes.join(" · ")}</small>}
          {panel && <div className="plugin-panel"><strong>{localizedDynamicLabel(panel.title, `${item.plugin_id}-panel`, locale)}</strong>{panel.sections.map((section) => <section key={section.section_id}><h4>{localizedDynamicLabel(section.title, section.section_id, locale)}</h4><div className="panel-widgets">{section.widgets.map((widget) => <PluginPanelWidget key={widget.widget_id} widget={widget} configuration={configuration} onConfigurationChange={setConfiguration} />)}</div>{section.actions.length > 0 && <div className="panel-actions">{section.actions.map((action) => <button key={action.action_id} className={action.style === "danger" ? "danger-text" : action.style === "primary" ? "primary-action" : ""} onClick={() => void runPanelAction(item, action)}>{localizedDynamicLabel(action.label, action.action_id, locale)}</button>)}</div>}</section>)}</div>}
        </>}
        <div className="plugin-actions">{item.placement.slot_id === "harness.profile" && <button className="primary-action" disabled={busy !== null} onClick={() => void act(`${item.plugin_id}:profile`, () => appApi.applyPluginProfile(item.plugin_id))}>{tr("应用方案", "Apply profile")}</button>}{!item.builtin && item.trust_status === "unverified" && <button disabled={busy !== null} onClick={() => void act(`${item.plugin_id}:trust`, () => appApi.trustLocalPluginPackage(item.plugin_id))}>{tr("批准此包", "Approve this package")}</button>}<button disabled={busy !== null} onClick={() => void act(`${item.plugin_id}:health`, () => appApi.checkPlugin(item.plugin_id))}>{tr("健康检查", "Health check")}</button>{!item.builtin && item.trust_status !== "revoked" && <button disabled={busy !== null} className="danger-text" onClick={() => void act(`${item.plugin_id}:revoke`, () => appApi.revokePluginPackage(item.plugin_id))}>{tr("撤销信任", "Revoke trust")}</button>}{!item.builtin && <button disabled={busy !== null} className="danger-text" onClick={() => { setSelected(null); setDetail(null); void act(`${item.plugin_id}:remove`, () => appApi.uninstallPlugin(item.plugin_id), null); }}>{tr("卸载", "Uninstall")}</button>}</div>
      </div>}
    </section>)}</div></div>)}</section>)}</div></div>
  </main>;
}

const PLAN_ALLOWANCES: Record<ManagedPlanId, number> = {
  free: 300_000,
  plus: 3_000_000,
  pro: 15_000_000,
};

function dateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function recentDays(count: number): string[] {
  const result: string[] = [];
  const current = new Date();
  current.setHours(0, 0, 0, 0);
  for (let offset = count - 1; offset >= 0; offset -= 1) {
    const value = new Date(current);
    value.setDate(current.getDate() - offset);
    result.push(dateKey(value));
  }
  return result;
}

function dailySeries(values: DailyModelUsage[], days: number): DailyModelUsage[] {
  const byDate = new globalThis.Map(values.map((item) => [item.date, item]));
  return recentDays(days).map((date) => byDate.get(date) ?? {
    date,
    aiCredits: 0,
    totalTokens: 0,
    requestCount: 0,
  });
}

function UsageVisualization({ account }: { account: AccountOverview }) {
  const { tr } = useI18n();
  const [range, setRange] = useState<"week" | "month" | "year">("week");
  const values = dailySeries(account.dailyUsage, range === "week" ? 7 : range === "month" ? 30 : 365);
  const maxCredits = Math.max(1, ...values.map((item) => item.aiCredits));
  return <section className="usage-section">
    <div className="usage-heading"><h2>{tr("用量", "Usage")}</h2><div className="range-tabs">
      <button className={range === "week" ? "active" : ""} onClick={() => setRange("week")}>{tr("7 天", "7 days")}</button>
      <button className={range === "month" ? "active" : ""} onClick={() => setRange("month")}>{tr("30 天", "30 days")}</button>
      <button className={range === "year" ? "active" : ""} onClick={() => setRange("year")}>{tr("一年", "1 year")}</button>
    </div></div>
    {range === "week"
      ? <WeeklyUsageChart values={values} maxCredits={maxCredits} />
      : <UsageHeatmap values={values} maxCredits={maxCredits} compact={range === "year"} />}
    {!account.dailyUsageComplete && <p className="usage-sync-note">{tr("每日明细尚未完整同步", "Daily usage details have not finished syncing")}</p>}
  </section>;
}

function WeeklyUsageChart({ values, maxCredits }: { values: DailyModelUsage[]; maxCredits: number }) {
  const { tr, number, compactNumber } = useI18n();
  const ticks = [maxCredits, maxCredits * .67, maxCredits * .33, 0];
  return <div className="weekly-chart">
    <div className="y-axis">{ticks.map((value, index) => <span key={index}>{compactNumber(value)}</span>)}</div>
    <div className="bar-plot">
      <div className="grid-lines"><i /><i /><i /><i /></div>
      {values.map((item) => <div className="bar-column" key={item.date}>
        <div className="bar-track"><div className="bar" style={{ height: `${item.aiCredits / maxCredits * 100}%` }}><span className="usage-tooltip">{item.date}<b>{number(item.aiCredits)} {tr("额度", "credits")}</b><small>{number(item.totalTokens)} Token · {number(item.requestCount)} {tr("次", "requests")}</small></span></div></div>
        <span>{item.date.slice(5).replace("-", "/")}</span>
      </div>)}
    </div>
  </div>;
}

function UsageHeatmap({ values, maxCredits, compact }: { values: DailyModelUsage[]; maxCredits: number; compact: boolean }) {
  const { locale, tr, number } = useI18n();
  const firstWeekday = new Date(`${values[0]?.date ?? dateKey(new Date())}T00:00:00`).getDay();
  return <div className={`usage-heatmap-wrap ${compact ? "year" : "month"}`}>
    <div className="weekday-labels">{(locale === "en-US" ? ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] : ["日", "一", "二", "三", "四", "五", "六"]).map((day) => <span key={day}>{day}</span>)}</div>
    <div className="usage-heatmap">
      {Array.from({ length: firstWeekday }, (_item, index) => <i key={`blank-${index}`} />)}
      {values.map((item) => {
        const level = item.aiCredits === 0 ? 0 : Math.max(1, Math.ceil(item.aiCredits / maxCredits * 4));
        return <button key={item.date} className={`heat-cell level-${level}`} aria-label={`${item.date}, ${item.aiCredits} ${tr("额度", "credits")}, ${item.totalTokens} Token`}>
          <span className="usage-tooltip">{item.date}<b>{number(item.aiCredits)} {tr("额度", "credits")}</b><small>{number(item.totalTokens)} Token · {number(item.requestCount)} {tr("次", "requests")}</small></span>
        </button>;
      })}
    </div>
  </div>;
}

function SettingsPage({ session, data, account, accountError, onChanged, onRefreshAccount, onEditAvatar, onLocaleChange }: {
  session: Session;
  data: Bootstrap;
  account: AccountOverview | null;
  accountError: string;
  onChanged: () => Promise<void>;
  onRefreshAccount: () => Promise<void>;
  onEditAvatar: () => void;
  onLocaleChange: (locale: AppLocale) => void;
}) {
  const { locale, tr, number } = useI18n();
  const [customOpen, setCustomOpen] = useState(false);
  const [customBusy, setCustomBusy] = useState(false);
  const [customError, setCustomError] = useState("");
  const [customNotice, setCustomNotice] = useState("");
  const [customBaseUrl, setCustomBaseUrl] = useState("https://api.openai.com/v1");
  const [customApiKey, setCustomApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [customModelId, setCustomModelId] = useState("");
  const [customName, setCustomName] = useState("");
  const [customProvider, setCustomProvider] = useState("");
  const [customModels, setCustomModels] = useState<string[]>([]);
  const [customApiStyle, setCustomApiStyle] = useState<"responses" | "chat-completions">("chat-completions");
  const configuredModels = data.models.filter((item) => item.source === "custom");
  const discoverModels = async () => {
    if (!customBaseUrl || !customApiKey) return;
    setCustomBusy(true); setCustomError(""); setCustomNotice("");
    try {
      const result = await appApi.discoverCustomModels(customBaseUrl, customApiKey);
      setCustomProvider(result.provider); setCustomModels(result.models);
      setCustomApiStyle(result.provider === "openai" ? "responses" : "chat-completions");
      if (!customModelId && result.models.length) setCustomModelId(result.models[0]);
      setCustomNotice(result.models.length
        ? tr(`已识别 ${result.provider}，发现 ${result.models.length} 个模型`, `${result.provider} identified. ${result.models.length} models found.`)
        : tr(`已识别 ${result.provider}，请填写模型 ID`, `${result.provider} identified. Enter a model ID.`));
    } catch (value) { setCustomError(humanError(value, locale)); }
    finally { setCustomBusy(false); }
  };
  const saveCustomModel = async () => {
    if (!customName || !customModelId || !customBaseUrl || !customApiKey) return;
    setCustomBusy(true); setCustomError(""); setCustomNotice("");
    try {
      await appApi.createCustomModel({
        display_name: customName, base_url: customBaseUrl, api_key: customApiKey,
        model_id: customModelId, provider: customProvider || undefined, api_style: customApiStyle,
      });
      setCustomApiKey(""); setCustomModelId(""); setCustomName(""); setCustomModels([]); setCustomProvider(""); setCustomOpen(false);
      await onChanged();
    } catch (value) { setCustomError(humanError(value, locale)); }
    finally { setCustomBusy(false); }
  };
  const currentPlan = account?.snapshot.plan.id;
  const usage = account?.snapshot.usage;
  const plan = account?.snapshot.plan;
  const scope = account?.snapshot.account?.billing_scope === "business" ? tr("商业", "Business") : tr("个人", "Individual");
  const percent = usage && plan
    ? remainingAllowancePercent(usage.remaining_ai_credits, plan.included_ai_credits)
    : 0;
  const displayName = account?.displayName ?? session.user.email?.split("@")[0] ?? "Pilot";
  return <main className="settings-page">
    <section><h2>{tr("账户", "Account")}</h2><div className="setting-row account-setting"><button className="settings-avatar-trigger" aria-label={tr("更换头像", "Change profile photo")} onClick={onEditAvatar}><Avatar name={displayName} src={account?.avatarUrl} /></button><div><strong>{account?.email ?? session.user.email}</strong><span>{scope} · {plan?.name ?? tr("正在读取", "Loading")}</span></div><button className="secondary-button" onClick={() => void appApi.openPricing()}>{tr("管理套餐", "Manage plan")}</button></div></section>
    <section><h2>{tr("通用", "General")}</h2><label className="setting-row"><span>{tr("语言", "Language")}</span><select value={locale} onChange={async (e) => { const next = normalizeLocale(e.target.value); onLocaleChange(next); try { await appApi.patchSettings({ locale: next }); await onChanged(); } catch (value) { setCustomError(humanError(value, next)); } }}><option value="zh-CN">{tr("简体中文", "Simplified Chinese")}</option><option value="en-US">{tr("英文", "English")}</option></select></label><label className="setting-row"><span>{tr("更新通道", "Update channel")}</span><select value={data.settings.update_channel} onChange={async (e) => { await appApi.patchSettings({ update_channel: e.target.value }); await onChanged(); }}><option value="stable">{tr("稳定版", "Stable")}</option><option value="preview">{tr("预览版", "Preview")}</option></select></label></section>
    <section className="custom-models"><div className="custom-model-heading"><h2>{tr("模型", "Models")}</h2><button className="secondary-button" onClick={() => setCustomOpen(!customOpen)}><KeyRound />{tr("添加模型", "Add model")}</button></div>
      {configuredModels.map((model) => <div className="custom-model-row" key={model.id}><ProviderBrandLogo provider={model.provider} icon={model.icon} /><div><strong>{model.label}</strong><span>{model.provider} · {model.model}</span></div><button onClick={async () => { if (!model.profile_id) return; setCustomBusy(true); setCustomError(""); setCustomNotice(""); try { await appApi.testCustomModel(model.profile_id); setCustomNotice(tr(`${model.label} 连接正常`, `${model.label} is connected`)); } catch (value) { setCustomError(humanError(value, locale)); } finally { setCustomBusy(false); } }}>{tr("测试", "Test")}</button><button className="danger-icon" aria-label={`${tr("删除", "Delete")} ${model.label}`} onClick={async () => { if (!model.profile_id) return; await appApi.deleteCustomModel(model.profile_id); await onChanged(); }}><Trash2 /></button></div>)}
      {!configuredModels.length && !customOpen && <div className="custom-model-empty">{tr("尚未添加自定义模型", "No custom models added")}</div>}
      {customOpen && <div className="custom-model-form">
        <label><span>{tr("API 地址", "API endpoint")}</span><input value={customBaseUrl} onChange={(event) => setCustomBaseUrl(event.target.value)} /></label>
        <label><span>API Key</span><span className="secret-input"><input type={showApiKey ? "text" : "password"} autoComplete="off" value={customApiKey} onChange={(event) => setCustomApiKey(event.target.value)} /><button type="button" aria-label={showApiKey ? tr("隐藏 API Key", "Hide API Key") : tr("显示 API Key", "Show API Key")} title={showApiKey ? tr("隐藏 API Key", "Hide API Key") : tr("显示 API Key", "Show API Key")} onClick={() => setShowApiKey(!showApiKey)}>{showApiKey ? <EyeOff /> : <Eye />}</button></span></label>
        <label><span>{tr("模型 ID", "Model ID")}</span><input list="custom-model-options" value={customModelId} onChange={(event) => setCustomModelId(event.target.value)} /><datalist id="custom-model-options">{customModels.map((id) => <option key={id} value={id} />)}</datalist></label>
        <label><span>{tr("显示名称", "Display name")}</span><input value={customName} onChange={(event) => setCustomName(event.target.value)} /></label>
        <label><span>{tr("接口协议", "API protocol")}</span><select value={customApiStyle} onChange={(event) => setCustomApiStyle(event.target.value as "responses" | "chat-completions")}><option value="chat-completions">Chat Completions</option><option value="responses">Responses</option></select></label>
        <div className="custom-model-actions"><div className="custom-provider-result">{customProvider && <><ProviderBrandLogo provider={customProvider} /><span>{customProvider}</span></>}</div><button className="secondary-button" disabled={customBusy || !customApiKey || !customBaseUrl} onClick={() => void discoverModels()}>{customBusy ? tr("正在识别", "Identifying") : tr("识别供应商与模型", "Identify provider and models")}</button><button className="primary-button" disabled={customBusy || !customName || !customModelId || !customApiKey} onClick={() => void saveCustomModel()}>{tr("保存", "Save")}</button></div>
      </div>}
      {customNotice && <div className="form-notice">{customNotice}</div>}
      {customError && <div className="form-error">{customError}</div>}
    </section>
    {accountError && <div className="account-error"><span>{accountError}</span><button onClick={() => void onRefreshAccount()}>{tr("重试", "Retry")}</button></div>}
    {account && usage && plan && <>
      <section><div className="allowance-heading"><h2>{tr("剩余额度", "Remaining allowance")}</h2><span>{tr("下次补满", "Next refill")}: {formatAllowanceRefillAt(account.snapshot.period.ends_at, locale)}</span></div><div className="allowance-panel">
        <div className="allowance-total"><strong>{number(usage.remaining_ai_credits)}</strong><span>{tr("剩余额度", "credits remaining")}</span></div>
        <div className="allowance-progress" role="progressbar" aria-label={tr("剩余额度", "Remaining allowance")} aria-valuemin={0} aria-valuemax={plan.included_ai_credits} aria-valuenow={usage.remaining_ai_credits} aria-valuetext={`${percent}%`}><div style={{ width: `${percent}%` }} /></div>
        <div className="allowance-metrics"><div><span>{tr("已用", "Used")}</span><strong>{number(usage.consumed_ai_credits)}</strong></div><div><span>{tr("总额", "Total")}</span><strong>{number(plan.included_ai_credits)}</strong></div><div><span>Token</span><strong>{number(usage.total_tokens)}</strong></div><div><span>{tr("调用", "Requests")}</span><strong>{number(usage.request_count)}</strong></div></div>
      </div></section>
      <UsageVisualization account={account} />
    </>}
    <section><h2>{tr("套餐", "Plans")}</h2><div className="plan-grid">{(["free", "plus", "pro"] as ManagedPlanId[]).map((id) => <Plan key={id} id={id} active={currentPlan === id} />)}</div></section>
  </main>;
}

function Plan({ id, active }: { id: ManagedPlanId; active: boolean }) {
  const { tr, number } = useI18n();
  const name = id[0].toUpperCase() + id.slice(1);
  return <article className={`plan ${active ? "active" : ""}`}><div><strong>{name}</strong><small>{number(PLAN_ALLOWANCES[id])} {tr("额度 / 30 天", "credits / 30 days")}</small></div>{active ? <span><Check />{tr("当前套餐", "Current plan")}</span> : <button onClick={() => void appApi.openPricing()}>{tr("管理", "Manage")}</button>}</article>;
}

function humanError(value: unknown, locale: AppLocale = "zh-CN", fallback?: string) {
  const code = value instanceof Error ? value.message : String(value);
  const en = locale === "en-US";
  if (code.startsWith("CUSTOM_MODEL_DISCOVERY_FAILED:AuthenticationError")) return en ? "The API Key is invalid or does not have access" : "API Key 无效或没有访问权限";
  if (code.startsWith("CUSTOM_MODEL_DISCOVERY_FAILED:")) return en ? "Could not connect to the model service. Check the API endpoint and network." : "无法连接到该模型服务，请检查 API 地址与网络";
  if (code.startsWith("CUSTOM_MODEL_TEST_FAILED:")) return en ? "The model connection test failed. Check the model ID and account access." : "模型连接测试失败，请检查模型 ID 与账户权限";
  if (code.startsWith("AVATAR_UPLOAD_FAILED:")) return en ? `The profile photo service returned an error (${code.split(":")[1]})` : `头像存储服务返回异常（${code.split(":")[1]}）`;
  if (code.startsWith("AVATAR_METADATA_FAILED:")) return en ? "The photo was uploaded, but the account profile could not be updated" : "头像已上传，但账户资料更新失败";
  const labels: Record<string, [string, string]> = {
    MAP_NOT_QUALIFIED: ["所选地图尚未达到执行资格", "The selected map is not qualified for execution"],
    VEHICLE_NOT_QUALIFIED: ["所选无人机尚未达到执行资格", "The selected drone is not qualified for execution"],
    ASSET_VERSION_NOT_QUALIFIED: ["所选资产版本尚未完成真实仿真认证", "The selected asset version has not completed real-simulation qualification"],
    ASSET_VERSION_QUALIFICATION_INVALID: ["所选资产的资格证据无效，请重新认证", "The selected asset has invalid qualification evidence. Qualify it again."],
    EXECUTION_ASSET_VERSION_PAIR_INCOMPLETE: ["地图与无人机必须同时选择精确的认证版本", "Select exact qualified versions for both the map and drone"],
    EXECUTION_ASSET_QUALIFICATION_STALE: ["运行环境已变化，请重新认证这组地图与无人机", "The Runtime has changed. Qualify this map and drone pair again."],
    EXECUTION_ASSET_PLAN_BINDING_MISMATCH: ["计划绑定的资产与当前选择不一致，已阻止执行", "Execution was blocked because the plan is bound to different assets"],
    EXECUTION_PLAN_MESSAGE_NOT_CURRENT: ["这份计划已不可确认，请查看当前任务的最新计划", "This plan cannot be confirmed. Review the latest plan for this task."],
    RUNTIME_PROVISION_REQUIRED: ["仿真运行环境需要先完成一次初始化", "The simulation runtime must be initialized first"],
    LOCAL_CORE_UNAVAILABLE: ["AGENT Core 尚未就绪", "AGENT Core is not ready"],
    ASSET_BUNDLE_MUST_BE_ZIP: ["请选择符合规范的 ZIP 资产包", "Choose a valid ZIP asset package"],
    CUSTOM_MODEL_BASE_URL_INVALID: ["外部模型必须使用 HTTPS；本机模型可以使用 localhost", "External models must use HTTPS. Local models may use localhost."],
    CUSTOM_MODEL_PROFILE_MISSING: ["自定义模型配置不完整", "The custom model configuration is incomplete"],
    CUSTOM_MODEL_GRANT_INVALID: ["自定义模型授权已失效，请重试", "The custom model grant has expired. Try again."],
    AVATAR_UPLOAD_AUTH_REQUIRED: ["登录状态已失效，请重新登录后再试", "Your session has expired. Sign in and try again."],
    AVATAR_UPLOAD_FORBIDDEN: ["头像存储权限未就绪，请更新软件后重试", "Profile photo storage is not ready. Update the app and try again."],
    AVATAR_UPLOAD_TOO_LARGE: ["头像文件过大，请缩小后重试", "The profile photo is too large. Reduce it and try again."],
    AVATAR_SIZE_INVALID: ["头像文件过大，请缩小后重试", "The profile photo is too large. Reduce it and try again."],
    AVATAR_UPLOAD_NETWORK: ["无法连接头像存储服务，请稍后重试", "Could not reach profile photo storage. Try again later."],
    AVATAR_SAVE_FAILED: ["头像保存失败", "Could not save the profile photo"],
    CAMERA_FRAME_UNAVAILABLE: ["无法读取摄像头画面", "Could not read the camera frame"],
    I18N_PROVIDER_MISSING: ["语言服务尚未就绪", "Language services are not ready"],
  };
  const label = labels[code];
  if (label) return en ? label[1] : label[0];
  if (fallback) return fallback;
  if (en && containsHan(code)) return "The operation failed. Try again.";
  if (!en && !containsHan(code)) {
    return /^[A-Z0-9_.:-]+$/u.test(code) ? `操作失败（${code}）` : "操作失败，请稍后重试";
  }
  return code;
}

export default App;
