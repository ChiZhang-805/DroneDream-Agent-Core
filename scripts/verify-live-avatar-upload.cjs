const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(path.resolve(__dirname, "..", "app", "frontend", "node_modules", "playwright"));

const baseUrl = process.env.AUTONOMY_FRONTEND_URL || "http://127.0.0.1:4176";
const encodedSession = process.env.AUTONOMY_LIVE_SESSION_B64;
const photoPath = process.argv[2];
if (!encodedSession) throw new Error("AUTONOMY_LIVE_SESSION_B64 is required");
if (!photoPath || !fs.existsSync(photoPath)) throw new Error("A readable avatar photo path is required");
const session = JSON.parse(Buffer.from(encodedSession, "base64url").toString("utf8"));
const projectRef = new URL(process.env.VITE_SUPABASE_URL).hostname.split(".")[0];
const outputRoot = path.resolve(__dirname, "..", "artifacts", "ui");
fs.mkdirSync(outputRoot, { recursive: true });

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
  let uploadStatus = null;
  let uploadFailure = "";
  page.on("response", async (response) => {
    if (!response.url().includes("/storage/v1/object/profile-avatars/")) return;
    uploadStatus = response.status();
    if (!response.ok()) uploadFailure = (await response.text().catch(() => "")).slice(0, 300);
  });
  await page.addInitScript(({ storageKey, storedSession }) => {
    localStorage.setItem(storageKey, JSON.stringify(storedSession));
  }, { storageKey: `sb-${projectRef}-auth-token`, storedSession: session });
  await page.route("http://127.0.0.1:43123/**", async (route) => {
    const requestPath = new URL(route.request().url()).pathname;
    if (requestPath === "/v1/bootstrap") {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          models: [{ id: "gpt-5.4", model: "gpt-5.4", label: "GPT 5.4", provider: "openai", icon: "openai", source: "default" }],
          threads: [], maps: [], vehicles: [], plugins: [],
          settings: { locale: "zh-CN", theme: "light", update_channel: "stable" },
        }),
      });
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: "{}" });
  });

  try {
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.getByRole("button", { name: "更换头像" }).first().click();
    await page.locator("input.avatar-file-input").setInputFiles(photoPath);
    const cropDialog = page.getByRole("dialog", { name: "裁剪头像" });
    await cropDialog.waitFor();
    const zoom = cropDialog.locator('input[type="range"]');
    await zoom.fill("2.5");

    const cropPath = path.join(outputRoot, "avatar-live-crop-v1.png");
    await page.screenshot({ path: cropPath, fullPage: false });
    await cropDialog.getByRole("button", { name: "保存", exact: true }).click();
    await cropDialog.waitFor({ state: "detached", timeout: 30_000 }).catch(async (error) => {
      const visibleError = await page.locator(".avatar-crop-error").textContent().catch(() => "");
      throw new Error(`${error.message}; visible=${visibleError}; upload=${uploadStatus}; body=${uploadFailure}`);
    });
    await page.waitForFunction(() => document.querySelector(".account-avatar-trigger img")?.getAttribute("src")?.includes("profile-avatars"), undefined, { timeout: 30_000 });
    const avatarSource = await page.locator(".account-avatar-trigger img").getAttribute("src");
    const publicResponse = await page.request.get(avatarSource);
    if (!publicResponse.ok()) throw new Error(`Uploaded avatar is not publicly readable: ${publicResponse.status()}`);

    const resultPath = path.join(outputRoot, "avatar-live-account-v1.png");
    await page.screenshot({ path: resultPath, fullPage: false });
    console.log(`LIVE_AVATAR_UPLOAD_OK status=${uploadStatus} public=${publicResponse.status()}`);
    console.log(`CROP_SCREENSHOT=${cropPath}`);
    console.log(`ACCOUNT_SCREENSHOT=${resultPath}`);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
