import { spawn, spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(frontendRoot, "..", "..");
const evidenceRoot = path.join(repositoryRoot, "artifacts", "qa", "startup-ui");
const port = 4175;
const origin = `http://127.0.0.1:${port}`;
let locale = "zh-CN";
let runtimeReady = false;
let modelCatalogReady = true;

const models = Array.from({ length: 7 }, (_, index) => ({
  id: index === 0 ? "gpt-5.4" : `default-model-${index + 1}`,
  model: index === 0 ? "gpt-5.4" : `default-model-${index + 1}`,
  label: index === 0 ? "GPT 5.4" : `Model ${index + 1}`,
  provider: index === 0 ? "openai" : "deepseek",
  icon: index === 0 ? "openai" : "deepseek",
  source: "default",
}));
const governance = {
  schema_version: "dronedream.plugin-governance-policy.v1",
  policy_id: "personal-default",
  mode: "personal",
  allowed_plugin_ids: [], allowed_publishers: [], denied_permissions: [],
  allowed_update_rings: ["stable"], require_verified_signatures: true,
  allow_local_approval: true, maximum_external_plugins: 50,
};
const assetVersions = [
  {
    asset_id: "dronedream.school-map.v1",
    content_sha256: "a".repeat(64),
    kind: "map",
    maturity: "qualified",
    bundle_root: "school-map",
    manifest: {},
    asset_ir: { name: "School Map" },
    imported_at: "2026-09-06T00:00:00Z",
  },
  {
    asset_id: "dronedream.my-drone.v1",
    content_sha256: "b".repeat(64),
    kind: "vehicle",
    maturity: "qualified",
    bundle_root: "my-drone",
    manifest: {},
    asset_ir: { name: "My Drone" },
    imported_at: "2026-09-06T00:00:00Z",
  },
];
const bootstrap = () => ({
  models: modelCatalogReady ? models : models.slice(0, 6),
  threads: [],
  asset_import_jobs: [],
  asset_versions: assetVersions,
  asset_qualification_jobs: [],
  asset_source_adapters: [],
  plugins: [{ plugin_id: "dronedream.autonomy.harness", name: "Autonomy Harness", version: "1.0.0" }],
  connector_credentials: [],
  settings: {
    locale, theme: "system", update_channel: "stable", default_model_id: "gpt-5.4",
    memory_enabled: true, remember_task_preferences: true, remember_asset_choices: true,
    last_map_id: null, last_map_content_sha256: null,
    last_vehicle_id: null, last_vehicle_content_sha256: null,
    plugin_update_ring: "stable",
    plugin_governance: governance, plugin_marketplace_sources: [],
  },
});

const shell = process.platform === "win32" ? process.env.ComSpec ?? "cmd.exe" : "npm";
const args = process.platform === "win32"
  ? ["/d", "/s", "/c", `npm run dev -- --port ${port}`]
  : ["run", "dev", "--", "--port", String(port)];
const vite = spawn(shell, args, {
  cwd: frontendRoot,
  env: { ...process.env, VITE_LOCAL_CORE_URL: `${origin}/core`, VITE_LOCAL_CORE_TOKEN: "startup-visual-token", VITE_SUPABASE_URL: "", VITE_SUPABASE_PUBLISHABLE_KEY: "" },
  stdio: ["ignore", "pipe", "pipe"],
});

async function waitForServer() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try { if ((await fetch(origin)).ok) return; } catch { /* Vite is starting. */ }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Vite did not start in time");
}

let browser;
try {
  await waitForServer();
  await mkdir(evidenceRoot, { recursive: true });
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  await page.route("**/core/v1/bootstrap", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(bootstrap()) }));
  await page.route("**/core/v1/runtime/status", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ distribution: "DroneDreamRuntime", runtime_available: runtimeReady, resources_ready: true, provisioned: runtimeReady, issue: runtimeReady ? null : "DRONEDREAM_RUNTIME_UNAVAILABLE" }) }));
  await page.route("**/core/v1/settings", async (route) => {
    const body = route.request().postDataJSON();
    if (body.locale) locale = body.locale;
    Object.assign(bootstrap().settings, body);
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(bootstrap().settings) });
  });
  await page.goto(origin, { waitUntil: "networkidle" });
  await page.getByText("0%", { exact: true }).waitFor();
  await page.getByRole("button", { name: "安装或更新 Runtime", exact: true }).waitFor();
  if (await page.getByRole("button", { name: "登录并进入", exact: true }).count()) {
    throw new Error("The sign-in action must not appear at zero percent");
  }

  runtimeReady = true;
  modelCatalogReady = false;
  await page.reload({ waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "环境检查未通过", exact: true }).waitFor();
  await page.getByText("58%", { exact: true }).waitFor();
  if (await page.getByRole("button", { name: /安装或更新 Runtime|登录并进入/u }).count()) {
    throw new Error("The primary startup action must stay hidden between zero and one hundred percent");
  }
  await page.getByRole("button", { name: "关闭", exact: true }).last().click();

  modelCatalogReady = true;
  await page.reload({ waitUntil: "networkidle" });
  await page.getByText("100%", { exact: true }).waitFor();
  await page.getByRole("button", { name: "登录并进入", exact: true }).waitFor();
  await page.screenshot({ path: path.join(evidenceRoot, "startup-zh-CN.png") });

  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("heading", { name: "设置", exact: true }).waitFor();
  await page.screenshot({ path: path.join(evidenceRoot, "settings-general-zh-CN.png") });
  await page.getByRole("button", { name: /English/u }).click();
  await page.getByRole("heading", { name: "Settings", exact: true }).waitFor();
  const body = await page.locator("body").innerText();
  if (/[㐀-鿿]/u.test(body)) throw new Error("English startup settings contain Chinese text");
  await page.screenshot({ path: path.join(evidenceRoot, "settings-general-en-US.png") });
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page.screenshot({ path: path.join(evidenceRoot, "settings-models-en-US.png") });
  await page.getByRole("button", { name: "Memory", exact: true }).click();
  await page.screenshot({ path: path.join(evidenceRoot, "settings-memory-en-US.png") });
} finally {
  await browser?.close();
  if (process.platform === "win32" && vite.pid) {
    spawnSync("taskkill", ["/PID", String(vite.pid), "/T", "/F"], { stdio: "ignore" });
  } else {
    vite.kill("SIGTERM");
  }
}
