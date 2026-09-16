fn main() {
    println!("cargo:rerun-if-env-changed=DRONEDREAM_OAUTH_CLIENT_ID_AGENT");
    println!("cargo:rerun-if-env-changed=DRONEDREAM_OAUTH_CLIENT_ID_AUTONOMY");
    println!("cargo:rerun-if-env-changed=DRONEDREAM_RUNTIME_RELEASE_MANIFEST_URL");
    println!("cargo:rustc-env=DRONEDREAM_DESKTOP_EDITION_ID=autonomy");
    println!("cargo:rustc-env=DRONEDREAM_EDITION_PROFILE=autonomy-full");
    let runtime_manifest_url = std::env::var("DRONEDREAM_RUNTIME_RELEASE_MANIFEST_URL")
        .unwrap_or_else(|_| {
            "https://github.com/ChiZhang-805/DroneDream/releases/download/runtime-v0.1.0-beta.2/runtime-release.json".to_string()
        });
    assert!(
        runtime_manifest_url.starts_with("https://")
            && !runtime_manifest_url.contains(char::is_whitespace),
        "Runtime Base manifest URL must be an absolute HTTPS URL"
    );
    println!(
        "cargo:rustc-env=DRONEDREAM_PRODUCTION_RUNTIME_RELEASE_MANIFEST_URL={runtime_manifest_url}"
    );
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-arg-bin=dronedream-agent-desktop=/INCLUDE:__TAURI_BUNDLE_TYPE");
    }
    tauri_build::build()
}
