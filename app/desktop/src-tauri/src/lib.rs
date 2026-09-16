mod browser_auth;
#[path = "../../../../shared/dronedream-runtime-source/desktop/src-tauri/src/installer_handoff.rs"]
mod installer_handoff;
#[path = "../../../../shared/dronedream-runtime-source/desktop/src-tauri/src/process.rs"]
mod process;
#[path = "../../../../shared/dronedream-runtime-source/desktop/src-tauri/src/runtime.rs"]
mod runtime;
#[path = "../../../../shared/dronedream-runtime-source/desktop/src-tauri/src/runtime_cache.rs"]
mod runtime_cache;
#[path = "../../../../shared/dronedream-runtime-source/desktop/src-tauri/src/runtime_installer.rs"]
mod runtime_installer;
#[path = "../../../../shared/dronedream-runtime-source/desktop/src-tauri/src/runtime_keepalive.rs"]
mod runtime_keepalive;

#[cfg(test)]
mod desktop_api_bridge {
    pub(crate) fn verify_live_anonymous_session_contract_for_test() -> Result<(), String> {
        Ok(())
    }
}

pub(crate) const MINIMUM_WINDOWS_BUILD: u32 = 19041;

use serde::Serialize;
use std::{
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream},
    process::Command as SystemCommand,
    sync::Mutex,
    time::Duration,
};
use tauri::{Manager, RunEvent};
use tauri_plugin_shell::{process::CommandChild, ShellExt};
use uuid::Uuid;

// MSVC can garbage-collect the private tauri-utils marker before the CLI's
// post-link patch. Exporting an equivalent live marker keeps NSIS/update bundle
// identity in the executable on Windows release builds.
#[cfg(windows)]
#[used]
#[no_mangle]
#[link_section = ".rdata"]
static mut __TAURI_BUNDLE_TYPE: [u8; 27] = *b"__TAURI_BUNDLE_TYPE_VAR_UNK";

#[derive(Clone, Serialize)]
struct BackendInfo {
    base_url: String,
    token: String,
}

struct BackendState {
    info: BackendInfo,
    child: Mutex<Option<CommandChild>>,
}

// The shared runtime updater shuts down the edition's AGENT Core before NSIS
// replaces packaged executables. The public editions provide this adapter from
// their own `agent_core` module; this private AGENT shell owns a loopback
// sidecar directly, so expose the same narrow shutdown contract here.
mod agent_core {
    use super::{request_backend_shutdown, BackendState};
    use std::time::Duration;
    use tauri::{AppHandle, Manager};

    pub(crate) fn stop(handle: &AppHandle) {
        let Some(state) = handle.try_state::<BackendState>() else {
            return;
        };
        request_backend_shutdown(&state.info);
        std::thread::sleep(Duration::from_millis(600));
        if let Ok(mut child) = state.child.lock() {
            if let Some(process) = child.take() {
                let _ = process.kill();
            }
        };
    }
}

#[tauri::command]
fn backend_info(state: tauri::State<'_, BackendState>) -> BackendInfo {
    state.info.clone()
}

#[tauri::command]
fn open_pricing_page() -> Result<(), String> {
    let url = "https://getdronedream.com/pricing/";
    #[cfg(target_os = "windows")]
    let child = SystemCommand::new("rundll32.exe")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn();
    #[cfg(target_os = "macos")]
    let child = SystemCommand::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let child = SystemCommand::new("xdg-open").arg(url).spawn();
    child
        .map(|_| ())
        .map_err(|error| format!("failed to open DroneDream pricing: {error}"))
}

fn reserve_loopback_port() -> Result<u16, String> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|error| format!("failed to reserve loopback port: {error}"))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| format!("failed to read loopback port: {error}"))
}

fn health_response_is_ready(response: &[u8]) -> bool {
    let Ok(response) = std::str::from_utf8(response) else {
        return false;
    };
    let Some((head, body)) = response.split_once("\r\n\r\n") else {
        return false;
    };
    let Some(status_line) = head.lines().next() else {
        return false;
    };
    if !status_line.starts_with("HTTP/1.1 200 ") && !status_line.starts_with("HTTP/1.0 200 ") {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("status")
                .and_then(|status| status.as_str())
                .map(str::to_owned)
        })
        .is_some_and(|status| status == "ready")
}

fn backend_health_ready(port: u16) -> bool {
    let address = SocketAddrV4::new(Ipv4Addr::LOCALHOST, port);
    let Ok(mut stream) = TcpStream::connect_timeout(&address.into(), Duration::from_millis(150))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let request = format!(
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::with_capacity(512);
    if stream.take(16 * 1024).read_to_end(&mut response).is_err() {
        return false;
    }
    health_response_is_ready(&response)
}

fn wait_until_ready(port: u16) -> Result<(), String> {
    for _ in 0..240 {
        if backend_health_ready(port) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err("AGENT Core did not become ready within 24 seconds".to_string())
}

#[cfg(test)]
mod core_readiness_tests {
    use super::health_response_is_ready;

    #[test]
    fn accepts_only_successful_structured_ready_health() {
        assert!(health_response_is_ready(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ready\",\"version\":\"1.0.0\"}"
        ));
        assert!(!health_response_is_ready(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"starting\"}"
        ));
        assert!(!health_response_is_ready(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ready\"}"
        ));
        assert!(!health_response_is_ready(b"not-http"));
    }
}

fn request_backend_shutdown(info: &BackendInfo) {
    let Some(port) = info
        .base_url
        .rsplit(':')
        .next()
        .and_then(|value| value.parse::<u16>().ok())
    else {
        return;
    };
    let address = SocketAddrV4::new(Ipv4Addr::LOCALHOST, port);
    let Ok(mut stream) = TcpStream::connect_timeout(&address.into(), Duration::from_millis(300))
    else {
        return;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let request = format!(
        "POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        info.token
    );
    if stream.write_all(request.as_bytes()).is_ok() {
        let mut response = [0_u8; 128];
        let _ = stream.read(&mut response);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    match installer_handoff::handle_early_command_line() {
        Ok(true) => return,
        Ok(false) => {}
        Err(error) => {
            eprintln!("Runtime installer handoff failed: {error}");
            std::process::exit(64);
        }
    }
    // Keep Tauri's bundle marker reachable in release binaries. Without this
    // reference MSVC can discard the marker before the CLI patches it, which
    // breaks reliable bundle/update identification even though packaging succeeds.
    std::hint::black_box(tauri::utils::platform::bundle_type());
    #[cfg(windows)]
    unsafe {
        std::hint::black_box(std::ptr::read_volatile(std::ptr::addr_of!(
            __TAURI_BUNDLE_TYPE
        )));
    }
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .manage(browser_auth::BrowserAuthCoordinator::default())
        .manage(runtime_installer::RuntimeInstaller::default())
        .manage(runtime_keepalive::RuntimeKeepalive::default())
        .setup(|app| {
            let port = reserve_loopback_port().map_err(std::io::Error::other)?;
            let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
            let data_root = app
                .path()
                .app_local_data_dir()
                .map_err(std::io::Error::other)?
                .join("core");
            let resource_root = app.path().resource_dir().map_err(std::io::Error::other)?;
            let plugin_isolator = resource_root.join("dronedream-plugin-isolator.exe");
            std::fs::create_dir_all(&data_root)?;
            let sidecar = app
                .shell()
                .sidecar("dronedream-autonomy-core")
                .map_err(std::io::Error::other)?
                .args([
                    "--host".to_string(),
                    "127.0.0.1".to_string(),
                    "--port".to_string(),
                    port.to_string(),
                    "--token".to_string(),
                    token.clone(),
                    "--data-root".to_string(),
                    data_root.to_string_lossy().into_owned(),
                    "--resource-root".to_string(),
                    resource_root.to_string_lossy().into_owned(),
                    "--plugin-isolator".to_string(),
                    plugin_isolator.to_string_lossy().into_owned(),
                ]);
            let (mut receiver, child) = sidecar.spawn().map_err(std::io::Error::other)?;
            tauri::async_runtime::spawn(async move { while receiver.recv().await.is_some() {} });
            wait_until_ready(port).map_err(std::io::Error::other)?;
            app.manage(BackendState {
                info: BackendInfo {
                    base_url: format!("http://127.0.0.1:{port}"),
                    token,
                },
                child: Mutex::new(Some(child)),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            backend_info,
            open_pricing_page,
            browser_auth::browser_auth_configured,
            browser_auth::begin_browser_auth,
            browser_auth::cancel_browser_auth,
            installer_handoff::get_installer_runtime_intent,
            installer_handoff::auto_start_installer_runtime,
            installer_handoff::discard_installer_runtime_intent,
            runtime::probe_runtime_status,
            runtime::get_runtime_install_plan,
            runtime_installer::start_runtime_install,
            runtime_installer::start_runtime_upgrade,
            runtime_installer::get_runtime_install_progress,
            runtime_installer::cancel_runtime_install,
            runtime_installer::start_runtime,
            runtime_installer::repair_runtime,
            runtime_keepalive::stop_runtime_for_exit,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build DroneDream AGENT desktop application");

    app.run(|handle, event| {
        if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
            agent_core::stop(handle);
        }
    });
}
