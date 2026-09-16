import { spawn, spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(frontendRoot, "..", "..");
const evidenceRoot = path.join(repositoryRoot, "artifacts", "qa", "bilingual-ui");
const port = 4174;
const origin = `http://127.0.0.1:${port}`;
const supabaseUrl = process.env.VITE_SUPABASE_URL;

if (!supabaseUrl || !process.env.VITE_SUPABASE_PUBLISHABLE_KEY) {
  throw new Error("VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY are required");
}

let locale = "zh-CN";

const plugin = {
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
  package_sha256: "a".repeat(64),
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
    category_order: 20,
    slot_order: 10,
    plugin_order: 10,
    pipeline_order: 10,
    runs_after: [],
    runs_before: [],
  },
  manifest: {},
};

const governance = {
  schema_version: "dronedream.plugin-governance-policy.v1",
  policy_id: "personal-default",
  mode: "personal",
  allowed_plugin_ids: [],
  allowed_publishers: [],
  denied_permissions: [],
  allowed_update_rings: ["stable"],
  require_verified_signatures: true,
  allow_local_approval: true,
  maximum_external_plugins: 50,
};

const mapContent = "a".repeat(64);
const vehicleContent = "b".repeat(64);
const assetVersions = [
  {
    asset_id: "dronedream.school-map.v1",
    content_sha256: mapContent,
    kind: "map",
    maturity: "simulation_ready",
    bundle_root: "C:/qa/map",
    manifest: {},
    asset_ir: { name: "School Map", version: "0.1.0" },
    imported_at: "2026-08-20T00:00:00Z",
  },
  {
    asset_id: "dronedream.my-drone.v1",
    content_sha256: vehicleContent,
    kind: "vehicle",
    maturity: "flight_ready",
    bundle_root: "C:/qa/vehicle",
    manifest: {},
    asset_ir: { name: "My Drone", version: "0.1.0" },
    imported_at: "2026-08-20T00:00:00Z",
  },
];
const qualificationJobs = [{
  schema_version: "dronedream.asset-pair-qualification-job.v1",
  job_id: `asset-qualification-job-${"c".repeat(24)}`,
  map_asset_id: "dronedream.school-map.v1",
  map_content_sha256: mapContent,
  vehicle_asset_id: "dronedream.my-drone.v1",
  vehicle_content_sha256: vehicleContent,
  state: "running",
  progress_percent: 62,
  qualification_id: `asset-qualification-${"d".repeat(24)}`,
  result_map_content_sha256: null,
  result_vehicle_content_sha256: null,
  issue_codes: [],
  cancel_requested: false,
  created_at: "2026-08-20T00:00:00Z",
  updated_at: "2026-08-20T00:01:00Z",
}];
const assetSourceAdapters = [
  {
    adapter_id: "gazebo.sdf",
    name: "Gazebo SDF",
    version: "1.0.0",
    availability: "builtin",
    source_formats: ["gazebo-sdf", "gazebo-world"],
    asset_kinds: ["map", "world", "vehicle"],
    output_format: "ddpkg",
    execution_boundary: "declarative-parser",
    required_application: null,
    documentation_url: "https://gazebosim.org/docs/latest/sdf_worlds/",
    enabled: true,
    provider_plugin_ids: [],
  },
  {
    adapter_id: "blender.phobos",
    name: "Blender + Phobos",
    version: "1.0.0",
    availability: "companion_required",
    source_formats: ["blender-blend", "phobos-smurf"],
    asset_kinds: ["map", "world", "vehicle"],
    output_format: "ddpkg",
    execution_boundary: "isolated-local-companion",
    required_application: "Blender + Phobos",
    documentation_url: "https://github.com/dfki-ric/phobos",
    enabled: false,
    provider_plugin_ids: [],
  },
];

const bootstrap = () => ({
  models: [{
    id: "gpt-5.4",
    model: "gpt-5.4",
    label: "GPT 5.4",
    provider: "openai",
    icon: "openai",
    source: "default",
  }],
  threads: [],
  maps: [{
    asset_id: "dronedream.school-map.v1",
    kind: "map",
    name: "School Map",
    status: "qualified",
    manifest: {},
    created_at: "2026-08-20T00:00:00Z",
    updated_at: "2026-08-20T00:00:00Z",
  }],
  vehicles: [{
    asset_id: "dronedream.my-drone.v1",
    kind: "vehicle",
    name: "My Drone",
    status: "qualified",
    manifest: {},
    created_at: "2026-08-20T00:00:00Z",
    updated_at: "2026-08-20T00:00:00Z",
  }],
  asset_import_jobs: [],
  asset_versions: assetVersions,
  asset_qualification_jobs: qualificationJobs,
  asset_source_adapters: assetSourceAdapters,
  plugins: [plugin],
  connector_credentials: [],
  settings: {
    locale,
    theme: "light",
    update_channel: "stable",
    default_model_id: "gpt-5.4",
    memory_enabled: true,
    remember_task_preferences: true,
    remember_asset_choices: true,
    last_map_id: null,
    last_map_content_sha256: null,
    last_vehicle_id: null,
    last_vehicle_content_sha256: null,
    plugin_update_ring: "stable",
    plugin_governance: governance,
    plugin_marketplace_sources: [],
  },
});

const accountSnapshot = {
  plan: {
    id: "pro",
    name: "Pro",
    monthly_price_cny_fen: 0,
    included_ai_credits: 15_000_000,
    capability_set: "pro",
  },
  account: {
    billing_scope: "individual",
    organization_id: null,
    organization_name: null,
    organization_role: null,
  },
  period: {
    starts_at: "2026-08-01T08:00:00Z",
    ends_at: "2026-08-31T08:00:00Z",
  },
  usage: {
    reserved_ai_credits: 0,
    consumed_ai_credits: 2_500_000,
    remaining_ai_credits: 12_500_000,
    request_count: 18,
    input_tokens: 120_000,
    output_tokens: 30_000,
    total_tokens: 150_000,
    estimated_request_count: 0,
    credit_policy_version: 1,
  },
};

function base64Url(value) {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

const expiresAt = Math.floor(Date.now() / 1000) + 60 * 60;
const accessToken = `${base64Url({ alg: "HS256", typ: "JWT" })}.${base64Url({
  sub: "qa-user",
  aud: "authenticated",
  exp: expiresAt,
  email: "pilot@dronedream.test",
  role: "authenticated",
})}.qa`;
const user = {
  id: "qa-user",
  aud: "authenticated",
  role: "authenticated",
  email: "pilot@dronedream.test",
  email_confirmed_at: "2026-08-01T00:00:00Z",
  phone: "",
  confirmed_at: "2026-08-01T00:00:00Z",
  last_sign_in_at: "2026-08-20T00:00:00Z",
  app_metadata: { provider: "email", providers: ["email"] },
  user_metadata: { display_name: "Pilot" },
  identities: [],
  created_at: "2026-08-01T00:00:00Z",
  updated_at: "2026-08-20T00:00:00Z",
  is_anonymous: false,
};

const viteCommand = process.platform === "win32" ? process.env.ComSpec ?? "cmd.exe" : "npm";
const viteArguments = process.platform === "win32"
  ? ["/d", "/s", "/c", `npm run dev -- --port ${port}`]
  : ["run", "dev", "--", "--port", String(port)];
const vite = spawn(viteCommand, viteArguments, {
  cwd: frontendRoot,
  env: {
    ...process.env,
    VITE_LOCAL_CORE_URL: `${origin}/core`,
    VITE_LOCAL_CORE_TOKEN: "visual-qa-token",
  },
  stdio: ["ignore", "pipe", "pipe"],
});

async function waitForServer() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(origin);
      if (response.ok) return;
    } catch {
      // Vite is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Vite did not start in time");
}

let browser;
try {
  await waitForServer();
  await mkdir(evidenceRoot, { recursive: true });
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const projectRef = new URL(supabaseUrl).hostname.split(".")[0];
  await context.addInitScript(({ storageKey, session }) => {
    localStorage.setItem(storageKey, JSON.stringify(session));
  }, {
    storageKey: `sb-${projectRef}-auth-token`,
    session: {
      access_token: accessToken,
      refresh_token: "qa-refresh-token",
      expires_in: 3600,
      expires_at: expiresAt,
      token_type: "bearer",
      user,
    },
  });
  const page = await context.newPage();

  await page.route("**/core/v1/bootstrap", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(bootstrap()) });
  });
  await page.route("**/core/v1/settings", async (route) => {
    const body = route.request().postDataJSON();
    if (typeof body?.locale === "string") locale = body.locale;
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(bootstrap().settings) });
  });
  await page.route("**/core/v1/plugins/dronedream.route-risk-evaluator", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ...plugin,
        versions: [],
        events: [],
        configuration: {},
        governance_decisions: [],
        usage: [],
        usage_summary: {
          calls: 0,
          successes: 0,
          errors: 0,
          duration_ms: 0,
          average_duration_ms: 0,
          input_bytes: 0,
          output_bytes: 0,
          last_called_at: null,
        },
      }),
    });
  });
  await page.route("**/functions/v1/model-gateway/usage", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ data: accountSnapshot }) });
  });
  await page.route("**/rest/v1/model_usage_requests**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", headers: { "content-range": "*/0" }, body: "[]" });
  });
  await page.route("**/auth/v1/user", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(user) });
  });

  await page.goto(`${origin}/?workspace=1`, { waitUntil: "networkidle" });
  const chineseForbidden = [
    "Settings",
    "Plugins",
    "New task",
    "Map library",
    "Drone library",
    "Remaining allowance",
    "Usage",
    "Qualified",
    "Draft",
  ];
  const captureChinese = async (name) => {
    const body = await page.locator("body").innerText();
    for (const forbidden of chineseForbidden) {
      if (body.includes(forbidden)) throw new Error(`Chinese UI contains English interface label: ${forbidden}`);
    }
    await page.screenshot({ path: path.join(evidenceRoot, `${name}-zh-CN.png`), fullPage: true });
  };
  const captureEnglish = async (name) => {
    const body = await page.locator("body").innerText();
    if (/[㐀-鿿]/u.test(body)) throw new Error(`English ${name} UI still contains Chinese text`);
    await page.screenshot({ path: path.join(evidenceRoot, `${name}-en-US.png`), fullPage: true });
  };

  await captureChinese("new-task");
  await page.getByRole("button", { name: "地图仓库", exact: true }).click();
  await captureChinese("map-library");
  await page.getByRole("button", { name: "无人机仓库", exact: true }).click();
  await captureChinese("drone-library");
  await page.getByRole("button", { name: "插件", exact: true }).click();
  await page.getByRole("heading", { name: "建模与仿真连接器", exact: true }).waitFor();
  await page.getByText("Gazebo SDF", { exact: true }).waitFor();
  await page.getByText("需要配套程序", { exact: true }).waitFor();
  await page.getByText("路线风险评估", { exact: true }).click();
  await page.getByText("发布者", { exact: true }).waitFor();
  await captureChinese("plugins");

  await page.getByRole("button", { name: "Pilot", exact: true }).click();
  await page.getByRole("button", { name: "设置" }).click();
  await page.getByRole("heading", { name: "设置", exact: true }).waitFor();
  await captureChinese("settings");

  await page.locator("select").filter({ has: page.locator('option[value="en-US"]') }).selectOption("en-US");
  await page.getByRole("heading", { name: "Settings", exact: true }).waitFor();
  if (await page.locator("html").getAttribute("lang") !== "en-US") throw new Error("HTML language did not switch to en-US");
  await captureEnglish("settings");

  await page.getByRole("button", { name: "New task", exact: true }).click();
  await captureEnglish("new-task");
  await page.getByRole("button", { name: "Map library", exact: true }).click();
  await captureEnglish("map-library");
  await page.getByRole("button", { name: "Drone library", exact: true }).click();
  await captureEnglish("drone-library");
  await page.getByRole("button", { name: "Plugins", exact: true }).click();
  await page.getByRole("heading", { name: "Modeling and simulation connectors", exact: true }).waitFor();
  await page.getByText("Companion required", { exact: true }).waitFor();
  await page.getByText("Route Risk Evaluator", { exact: true }).click();
  await page.getByText("Publisher", { exact: true }).waitFor();
  await captureEnglish("plugins");

  process.stdout.write(`${evidenceRoot}\n`);
} finally {
  await browser?.close();
  if (process.platform === "win32" && vite.pid) {
    spawnSync("taskkill", ["/PID", String(vite.pid), "/T", "/F"], { stdio: "ignore" });
  } else {
    vite.kill();
  }
}
