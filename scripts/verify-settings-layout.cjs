const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(path.resolve(__dirname, "..", "app", "frontend", "node_modules", "playwright"));

const baseUrl = process.env.AUTONOMY_FRONTEND_URL || "http://127.0.0.1:4175";
const outputRoot = path.resolve(__dirname, "..", "artifacts", "ui");
fs.mkdirSync(outputRoot, { recursive: true });

const account = {
  id: "user-layout",
  aud: "authenticated",
  role: "authenticated",
  email: "layout-tester@example.test",
  user_metadata: { display_name: "Layout Tester" },
};
const session = {
  access_token: "layout-access-token",
  token_type: "bearer",
  expires_in: 31_536_000,
  expires_at: Math.floor(Date.now() / 1000) + 31_536_000,
  refresh_token: "layout-refresh-token",
  user: account,
};

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1200 }, deviceScaleFactor: 1 });
  const forbiddenDataFetches = [];
  page.on("request", (request) => {
    if (request.url().startsWith("data:image/")) forbiddenDataFetches.push(request.url().slice(0, 48));
  });
  await page.addInitScript((storedSession) => {
    localStorage.setItem("sb-autonomy-layout-auth-token", JSON.stringify(storedSession));
  }, session);
  await page.route("http://127.0.0.1:43123/**", async (route) => {
    const requestPath = new URL(route.request().url()).pathname;
    if (requestPath === "/v1/bootstrap") {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          models: [
            { id: "gpt-5.4", model: "gpt-5.4", label: "GPT 5.4", provider: "openai", icon: "openai", source: "default" },
            { id: "deepseek-v4-pro", model: "deepseek-v4-pro", label: "DeepSeek V4 Pro", provider: "deepseek", icon: "deepseek", source: "default" },
            { id: "kimi-k3", model: "kimi-k3", label: "Kimi K3", provider: "kimi", icon: "kimi", source: "default" },
          ],
          threads: [], maps: [], vehicles: [], plugins: [],
          settings: { locale: "zh-CN", theme: "light", update_channel: "stable" },
        }),
      });
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: "{}" });
  });
  await page.route("https://autonomy-layout.supabase.co/**", async (route) => {
    const url = route.request().url();
    if (url.includes("/functions/v1/model-gateway/usage")) {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ data: {
          plan: { id: "pro", name: "Pro", monthly_price_cny_fen: 0, included_ai_credits: 15_000_000, capability_set: "pro" },
          account: { billing_scope: "individual", organization_id: null, organization_name: null, organization_role: null },
          period: { starts_at: "2026-07-01T04:23:00Z", ends_at: "2026-07-31T04:23:00Z" },
          usage: { reserved_ai_credits: 0, consumed_ai_credits: 3_750_000, remaining_ai_credits: 11_250_000, request_count: 14, input_tokens: 625_000, output_tokens: 312_500, total_tokens: 937_500, estimated_request_count: 0, credit_policy_version: 1 },
        } }),
      });
      return;
    }
    if (url.includes("/rest/v1/model_usage_requests")) {
      await route.fulfill({ contentType: "application/json", body: "[]" });
      return;
    }
    if (url.includes("/auth/v1/user")) {
      await route.fulfill({ contentType: "application/json", body: JSON.stringify(account) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Layout Tester" }).click();
  const accountMenu = page.locator(".account-menu");
  await accountMenu.getByRole("button", { name: /剩余额度/u }).waitFor();
  await accountMenu.getByText("75%", { exact: true }).waitFor();
  const accountScreenshotPath = path.join(outputRoot, "account-remaining-allowance.png");
  await page.screenshot({ path: accountScreenshotPath, fullPage: false });
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("heading", { name: "设置", exact: true }).waitFor();
  await page.getByRole("button", { name: "添加模型" }).click();

  const layout = await page.evaluate(() => {
    const settings = document.querySelector(".settings-page");
    const form = document.querySelector(".custom-model-form");
    const labels = Array.from(document.querySelectorAll(".custom-model-form > label"));
    const actions = document.querySelector(".custom-model-actions");
    const progress = document.querySelector(".allowance-progress[role='progressbar']");
    const progressFill = progress?.firstElementChild;
    const refill = document.querySelector(".allowance-heading span");
    if (!settings || !form || labels.length < 5 || !actions || !progress || !progressFill || !refill) throw new Error("Settings form or allowance did not render");
    const settingsRect = settings.getBoundingClientRect();
    const formRect = form.getBoundingClientRect();
    const fieldRects = labels.slice(2, 5).map((node) => node.getBoundingClientRect());
    const actionRect = actions.getBoundingClientRect();
    return {
      scrollbarRightGap: window.innerWidth - settingsRect.right,
      fieldTopSpread: Math.max(...fieldRects.map((rect) => rect.top)) - Math.min(...fieldRects.map((rect) => rect.top)),
      actionsBelowFields: actionRect.top >= Math.max(...fieldRects.map((rect) => rect.bottom)),
      actionsRightGap: formRect.right - actionRect.right,
      remainingPercent: progress.getAttribute("aria-valuetext"),
      progressWidth: progressFill.getBoundingClientRect().width,
      progressTrackWidth: progress.getBoundingClientRect().width,
      refillText: refill.textContent,
    };
  });
  if (layout.scrollbarRightGap > 1) throw new Error(`Settings scrollbar is inset: ${JSON.stringify(layout)}`);
  if (layout.fieldTopSpread > 1 || !layout.actionsBelowFields || layout.actionsRightGap > 1) {
    throw new Error(`Custom model form rows are misaligned: ${JSON.stringify(layout)}`);
  }
  if (layout.remainingPercent !== "75%" || Math.abs(layout.progressWidth / layout.progressTrackWidth - 0.75) > 0.01) {
    throw new Error(`Remaining allowance progress is incorrect: ${JSON.stringify(layout)}`);
  }
  if (!/下次补满：.*7.*31.*12.*23/u.test(layout.refillText ?? "")) {
    throw new Error(`Exact 30-day refill timestamp is missing: ${JSON.stringify(layout)}`);
  }

    const screenshotPath = path.join(outputRoot, "settings-avatar-model.png");
  await page.screenshot({ path: screenshotPath, fullPage: false });
  await page.getByRole("button", { name: "更换头像" }).last().click();
  await page.getByRole("heading", { name: "更换头像" }).waitFor();
  const avatarPath = path.join(outputRoot, "avatar-editor.png");
  await page.screenshot({ path: avatarPath, fullPage: false });
  await page.locator("input.avatar-file-input").setInputFiles({
    name: "avatar-test.png",
    mimeType: "image/png",
    buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zp9sAAAAASUVORK5CYII=", "base64"),
  });
  await page.getByRole("heading", { name: "裁剪头像" }).waitFor();
  const cropPath = path.join(outputRoot, "avatar-crop.png");
  await page.screenshot({ path: cropPath, fullPage: false });
  await page.getByRole("dialog", { name: "裁剪头像" }).getByRole("button", { name: "保存", exact: true }).click();
  await page.getByRole("heading", { name: "裁剪头像" }).waitFor({ state: "detached" });
  if (forbiddenDataFetches.length) {
    throw new Error(`Avatar save attempted a CSP-blocked data URL fetch: ${JSON.stringify(forbiddenDataFetches)}`);
  }
  await browser.close();
  console.log(`SETTINGS_LAYOUT_OK ${JSON.stringify(layout)}`);
  console.log(`ACCOUNT_SCREENSHOT=${accountScreenshotPath}`);
  console.log(`SCREENSHOT=${screenshotPath}`);
  console.log(`AVATAR_SCREENSHOT=${avatarPath}`);
  console.log(`CROP_SCREENSHOT=${cropPath}`);
  console.log("AVATAR_SAVE_OK local-base64-decoder");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
