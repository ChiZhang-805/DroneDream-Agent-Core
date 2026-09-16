use std::{
    env, fs,
    net::{SocketAddr, TcpStream},
    process::ExitCode,
    time::Duration,
};

fn main() -> ExitCode {
    let outside = env::args()
        .nth(1)
        .unwrap_or_else(|| "C:\\Windows\\win.ini".to_string());
    let filesystem_denied = fs::read(&outside).is_err();
    let address: SocketAddr = "1.1.1.1:443".parse().expect("static address");
    let network_denied = TcpStream::connect_timeout(&address, Duration::from_millis(750)).is_err();
    println!("{{\"filesystem_denied\":{filesystem_denied},\"network_denied\":{network_denied}}}");
    if filesystem_denied && network_denied {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
