use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use reqwest::{blocking::Client, redirect::Policy, StatusCode, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Manager};
use tauri_plugin_opener::OpenerExt;
use uuid::Uuid;

const OAUTH_AUTHORIZE_URL: &str =
    "https://yggabfynndpzymlqvnim.supabase.co/auth/v1/oauth/authorize";
const OAUTH_TOKEN_URL: &str = "https://yggabfynndpzymlqvnim.supabase.co/auth/v1/oauth/token";
const EXPECTED_ISSUER: &str = "https://yggabfynndpzymlqvnim.supabase.co/auth/v1";
const FRIENDLY_CLIENT_ID: &str = "dronedream-desktop-autonomy";
const CALLBACK_PORT: u16 = 49214;
const CALLBACK_PATH: &str = "/desktop-auth/autonomy/callback";
const REDIRECT_URI: &str = "http://127.0.0.1:49214/desktop-auth/autonomy/callback";
const AUTH_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const HTTP_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_TOKEN_RESPONSE_BYTES: usize = 48 * 1024;

#[derive(Default)]
pub struct BrowserAuthCoordinator {
    activity: Mutex<Option<Arc<AtomicBool>>>,
}

impl BrowserAuthCoordinator {
    fn begin(&self) -> Result<Arc<AtomicBool>, String> {
        let mut activity = self
            .activity
            .lock()
            .map_err(|_| "Browser sign-in state is unavailable.".to_owned())?;
        if activity.is_some() {
            return Err("A browser sign-in is already in progress.".to_owned());
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        *activity = Some(cancelled.clone());
        Ok(cancelled)
    }

    fn finish(&self) {
        if let Ok(mut activity) = self.activity.lock() {
            *activity = None;
        }
    }

    fn cancel(&self) -> Result<bool, String> {
        let activity = self
            .activity
            .lock()
            .map_err(|_| "Browser sign-in state is unavailable.".to_owned())?;
        if let Some(cancelled) = activity.as_ref() {
            cancelled.store(true, Ordering::SeqCst);
            return Ok(true);
        }
        Ok(false)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAuthRequest {
    locale: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAuthSession {
    protocol_version: &'static str,
    edition_id: &'static str,
    auth_client_id: &'static str,
    access_token: String,
    refresh_token: String,
}

#[derive(Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    refresh_token: String,
    token_type: String,
    expires_in: u64,
    id_token: String,
}

enum AuthorizationCallback {
    Authorized { code: String, state: String },
    Denied { state: String },
}

impl AuthorizationCallback {
    fn state(&self) -> &str {
        match self {
            Self::Authorized { state, .. } | Self::Denied { state } => state,
        }
    }
}

#[tauri::command]
pub fn browser_auth_configured() -> bool {
    configured_client_id().is_ok()
}

#[tauri::command]
pub async fn begin_browser_auth(
    app: AppHandle,
    request: BrowserAuthRequest,
) -> Result<BrowserAuthSession, String> {
    if request.locale != "en-US" && request.locale != "zh-CN" {
        return Err("Browser sign-in locale must be en-US or zh-CN.".to_owned());
    }
    let cancelled = app.state::<BrowserAuthCoordinator>().begin()?;
    let worker_app = app.clone();
    let operation = tauri::async_runtime::spawn_blocking(move || {
        run_browser_auth(worker_app, request.locale, cancelled)
    })
    .await
    .map_err(|error| format!("Browser sign-in task failed: {error}"))
    .and_then(|result| result);
    app.state::<BrowserAuthCoordinator>().finish();
    operation
}

#[tauri::command]
pub fn cancel_browser_auth(
    coordinator: tauri::State<'_, BrowserAuthCoordinator>,
) -> Result<bool, String> {
    coordinator.cancel()
}

fn configured_client_id() -> Result<&'static str, String> {
    let client_id = option_env!("DRONEDREAM_OAUTH_CLIENT_ID_AGENT")
        .or(option_env!("DRONEDREAM_OAUTH_CLIENT_ID_AUTONOMY"))
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "AUTONOMY_BROWSER_AUTH_NOT_CONFIGURED".to_owned())?;
    Uuid::parse_str(client_id).map_err(|_| "AUTONOMY_BROWSER_AUTH_CLIENT_INVALID".to_owned())?;
    Ok(client_id)
}

fn run_browser_auth(
    app: AppHandle,
    locale: String,
    cancelled: Arc<AtomicBool>,
) -> Result<BrowserAuthSession, String> {
    let oauth_client_id = configured_client_id()?;
    let state = random_token();
    let nonce = random_token();
    let code_verifier = random_token();
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(code_verifier.as_bytes()));
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, CALLBACK_PORT))
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_CALLBACK_IN_USE".to_owned())?;
    listener
        .set_nonblocking(true)
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_CALLBACK_FAILED".to_owned())?;
    let authorize_url = authorize_url(oauth_client_id, &state, &nonce, &challenge)?;
    app.opener()
        .open_url(authorize_url.as_str(), None::<&str>)
        .map_err(|_| "AUTONOMY_BROWSER_OPEN_FAILED".to_owned())?;

    let deadline = Instant::now() + AUTH_TIMEOUT;
    loop {
        if cancelled.load(Ordering::SeqCst) {
            return Err("AUTONOMY_BROWSER_AUTH_CANCELLED".to_owned());
        }
        if Instant::now() >= deadline {
            return Err("AUTONOMY_BROWSER_AUTH_TIMEOUT".to_owned());
        }
        match listener.accept() {
            Ok((mut stream, peer)) => {
                if !peer.ip().is_loopback() {
                    continue;
                }
                let callback = match read_callback(&mut stream) {
                    Ok(value) => value,
                    Err(error) => {
                        write_plain(&mut stream, 400, "Bad Request")?;
                        if error == "AUTONOMY_BROWSER_AUTH_WRONG_PATH" {
                            continue;
                        }
                        return Err(error);
                    }
                };
                if !constant_time_equal(callback.state().as_bytes(), state.as_bytes()) {
                    write_plain(&mut stream, 403, "Forbidden")?;
                    return Err("AUTONOMY_BROWSER_AUTH_STATE_INVALID".to_owned());
                }
                let AuthorizationCallback::Authorized { code, .. } = callback else {
                    write_result(&mut stream, &locale, false)?;
                    return Err("AUTONOMY_BROWSER_AUTH_DENIED".to_owned());
                };
                let tokens = exchange_code(oauth_client_id, &code, &code_verifier)?;
                validate_tokens(&tokens, oauth_client_id, &nonce)?;
                write_result(&mut stream, &locale, true)?;
                return Ok(BrowserAuthSession {
                    protocol_version: "desktop-browser-auth-pkce-v1",
                    edition_id: "autonomy",
                    auth_client_id: FRIENDLY_CLIENT_ID,
                    access_token: tokens.access_token,
                    refresh_token: tokens.refresh_token,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(25));
            }
            Err(_) => return Err("AUTONOMY_BROWSER_AUTH_CALLBACK_FAILED".to_owned()),
        }
    }
}

fn random_token() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

fn authorize_url(
    client_id: &str,
    state: &str,
    nonce: &str,
    challenge: &str,
) -> Result<Url, String> {
    let mut url = Url::parse(OAUTH_AUTHORIZE_URL)
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_CONFIGURATION_INVALID".to_owned())?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("state", state)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("scope", "openid email profile")
        .append_pair("nonce", nonce);
    Ok(url)
}

fn read_callback(stream: &mut TcpStream) -> Result<AuthorizationCallback, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_CALLBACK_FAILED".to_owned())?;
    let mut buffer = Vec::with_capacity(2048);
    let mut chunk = [0_u8; 1024];
    while !buffer.windows(4).any(|window| window == b"\r\n\r\n") {
        let read = stream
            .read(&mut chunk)
            .map_err(|_| "AUTONOMY_BROWSER_AUTH_CALLBACK_FAILED".to_owned())?;
        if read == 0 {
            return Err("AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned());
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > MAX_HEADER_BYTES {
            return Err("AUTONOMY_BROWSER_AUTH_CALLBACK_TOO_LARGE".to_owned());
        }
    }
    let request = std::str::from_utf8(&buffer)
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned())?;
    let mut lines = request.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| "AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned())?;
    let parts = request_line.split_whitespace().collect::<Vec<_>>();
    if parts.len() != 3 || parts[0] != "GET" || parts[2] != "HTTP/1.1" {
        return Err("AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned());
    }
    let headers = lines
        .take_while(|line| !line.is_empty())
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect::<HashMap<_, _>>();
    if headers.get("host").map(String::as_str) != Some("127.0.0.1:49214") {
        return Err("AUTONOMY_BROWSER_AUTH_HOST_INVALID".to_owned());
    }
    parse_callback(parts[1])
}

fn parse_callback(target: &str) -> Result<AuthorizationCallback, String> {
    let url = Url::parse(&format!("http://127.0.0.1{target}"))
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned())?;
    if url.path() != CALLBACK_PATH || url.fragment().is_some() {
        return Err("AUTONOMY_BROWSER_AUTH_WRONG_PATH".to_owned());
    }
    let mut fields = HashMap::new();
    for (name, value) in url.query_pairs() {
        if fields
            .insert(name.into_owned(), value.into_owned())
            .is_some()
        {
            return Err("AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned());
        }
    }
    let state = fields
        .remove("state")
        .filter(|value| !value.is_empty() && value.len() <= 256)
        .ok_or_else(|| "AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned())?;
    if fields.contains_key("error") {
        return Ok(AuthorizationCallback::Denied { state });
    }
    if fields.len() != 1 {
        return Err("AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned());
    }
    let code = fields
        .remove("code")
        .filter(|value| !value.is_empty() && value.len() <= 8192)
        .ok_or_else(|| "AUTONOMY_BROWSER_AUTH_CALLBACK_INVALID".to_owned())?;
    Ok(AuthorizationCallback::Authorized { code, state })
}

fn exchange_code(
    client_id: &str,
    code: &str,
    code_verifier: &str,
) -> Result<OAuthTokenResponse, String> {
    let client = Client::builder()
        .connect_timeout(HTTP_TIMEOUT)
        .timeout(HTTP_TIMEOUT)
        .redirect(Policy::none())
        .user_agent("DroneDream-AGENT/1.0.0")
        .build()
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_TOKEN_CLIENT_FAILED".to_owned())?;
    let response = client
        .post(OAUTH_TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("client_id", client_id),
            ("redirect_uri", REDIRECT_URI),
            ("code_verifier", code_verifier),
        ])
        .send()
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_TOKEN_NETWORK".to_owned())?;
    if !response.status().is_success() {
        return if matches!(
            response.status(),
            StatusCode::BAD_REQUEST | StatusCode::UNAUTHORIZED
        ) {
            Err("AUTONOMY_BROWSER_AUTH_CODE_REJECTED".to_owned())
        } else {
            Err("AUTONOMY_BROWSER_AUTH_SERVICE_UNAVAILABLE".to_owned())
        };
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_TOKEN_RESPONSE_BYTES as u64)
    {
        return Err("AUTONOMY_BROWSER_AUTH_TOKEN_INVALID".to_owned());
    }
    response
        .json::<OAuthTokenResponse>()
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_TOKEN_INVALID".to_owned())
}

fn jwt_claims(token: &str) -> Result<serde_json::Value, String> {
    let segments = token.split('.').collect::<Vec<_>>();
    if segments.len() != 3 || token.len() > 16 * 1024 {
        return Err("AUTONOMY_BROWSER_AUTH_TOKEN_INVALID".to_owned());
    }
    let payload = URL_SAFE_NO_PAD
        .decode(segments[1])
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_TOKEN_INVALID".to_owned())?;
    serde_json::from_slice(&payload).map_err(|_| "AUTONOMY_BROWSER_AUTH_TOKEN_INVALID".to_owned())
}

fn validate_tokens(
    response: &OAuthTokenResponse,
    client_id: &str,
    nonce: &str,
) -> Result<(), String> {
    if !response.token_type.eq_ignore_ascii_case("bearer")
        || response.expires_in == 0
        || response.expires_in > 86_400
        || response.refresh_token.is_empty()
    {
        return Err("AUTONOMY_BROWSER_AUTH_TOKEN_INVALID".to_owned());
    }
    let access = jwt_claims(&response.access_token)?;
    let identity = jwt_claims(&response.id_token)?;
    let subject = access.get("sub").and_then(serde_json::Value::as_str);
    let identity_subject = identity.get("sub").and_then(serde_json::Value::as_str);
    let access_client = access.get("client_id").and_then(serde_json::Value::as_str);
    let audience_matches = identity.get("aud").is_some_and(|audience| {
        audience.as_str() == Some(client_id)
            || audience
                .as_array()
                .is_some_and(|values| values.iter().any(|value| value.as_str() == Some(client_id)))
    });
    if subject.is_none()
        || subject != identity_subject
        || access_client != Some(client_id)
        || access.get("iss").and_then(serde_json::Value::as_str) != Some(EXPECTED_ISSUER)
        || identity.get("iss").and_then(serde_json::Value::as_str) != Some(EXPECTED_ISSUER)
        || identity.get("nonce").and_then(serde_json::Value::as_str) != Some(nonce)
        || !audience_matches
    {
        return Err("AUTONOMY_BROWSER_AUTH_IDENTITY_MISMATCH".to_owned());
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "AUTONOMY_BROWSER_AUTH_TOKEN_INVALID".to_owned())?
        .as_secs() as i64;
    for claims in [&access, &identity] {
        if claims
            .get("exp")
            .and_then(serde_json::Value::as_i64)
            .is_none_or(|expiry| expiry <= now)
        {
            return Err("AUTONOMY_BROWSER_AUTH_TOKEN_EXPIRED".to_owned());
        }
    }
    Ok(())
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}

fn write_plain(stream: &mut TcpStream, status: u16, reason: &str) -> Result<(), String> {
    let body = reason.as_bytes();
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .and_then(|_| stream.write_all(body))
    .map_err(|_| "AUTONOMY_BROWSER_AUTH_CALLBACK_FAILED".to_owned())
}

fn write_result(stream: &mut TcpStream, locale: &str, success: bool) -> Result<(), String> {
    let (title, message) = if locale == "zh-CN" {
        if success {
            (
                "登录成功",
                "DroneDream · AGENT 已完成登录，可以关闭此页面。",
            )
        } else {
            ("登录未完成", "请返回 DroneDream · AGENT 后重试。")
        }
    } else if success {
        (
            "Sign-in complete",
            "DroneDream · AGENT is signed in. You may close this page.",
        )
    } else {
        (
            "Sign-in incomplete",
            "Return to DroneDream · AGENT and try again.",
        )
    };
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>{title}</title><style>body{{margin:0;display:grid;min-height:100vh;place-items:center;background:#fff7f9;color:#1d1720;font:16px system-ui}}main{{width:min(520px,calc(100% - 48px));border-top:3px solid #e3264f;padding:38px 6px}}h1{{margin:0 0 12px;font-size:28px}}p{{margin:0;color:#6f6570;line-height:1.7}}</style></head><body><main><h1>{title}</h1><p>{message}</p></main></body></html>"
    );
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nContent-Security-Policy: default-src 'none'; style-src 'unsafe-inline'\r\nConnection: close\r\n\r\n",
        html.len()
    )
    .and_then(|_| stream.write_all(html.as_bytes()))
    .map_err(|_| "AUTONOMY_BROWSER_AUTH_CALLBACK_FAILED".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_accepts_only_the_autonomy_path_and_expected_fields() {
        let accepted =
            parse_callback("/desktop-auth/autonomy/callback?code=abc&state=state-1").unwrap();
        assert!(matches!(accepted, AuthorizationCallback::Authorized { .. }));
        assert!(parse_callback("/desktop-auth/sim/callback?code=abc&state=state-1").is_err());
        assert!(
            parse_callback("/desktop-auth/autonomy/callback?code=a&code=b&state=state-1").is_err()
        );
    }

    #[test]
    fn pkce_authorize_url_is_bound_to_the_autonomy_callback() {
        let url = authorize_url(
            "00000000-0000-4000-8000-000000000000",
            "state",
            "nonce",
            "challenge",
        )
        .unwrap();
        let values = url.query_pairs().collect::<HashMap<_, _>>();
        assert_eq!(
            values.get("redirect_uri").map(|value| value.as_ref()),
            Some(REDIRECT_URI)
        );
        assert_eq!(
            values
                .get("code_challenge_method")
                .map(|value| value.as_ref()),
            Some("S256")
        );
    }
}
