//! Ports for test subprocesses, with no window for another process to take one.
//!
//! The pattern this replaces — bind `:0`, read the port, release it, hand the number to a child to
//! bind — leaves a gap between the release and the child's bind in which any other process on the
//! host can take the port. Under parallel suites one did: a test's client reached another test's
//! server, or a server exited on `EADDRINUSE` while a bare connect succeeded against someone else's.
//! Here a child picks its own port and the test reads it back:
//!
//! - **The gateway** binds port 0 on both listeners ([`GATEWAY_LISTENERS`]), and [`gateway_ports`]
//!   reads them from the gateway's own `LISTEN` sockets ([`listening_on`]).
//! - **`nats-server`** is started with `-p -1 --ports_file_dir <dir>` and writes the port it got to a
//!   file ([`nats_port_from`]).
//! - **A port where nothing listens** is held bound, never listening ([`DeadPort`]).
//!
//! Reading a child's ports back needs Linux: its `LISTEN` sockets come from `/proc`, and the
//! gateway's metrics listener sits on `127.0.0.2`, which Linux routes to loopback with no setup.
//! Elsewhere the read-back fails at once and says so ([`require_proc`]), never a timeout.

use std::net::{SocketAddr, SocketAddrV4};
use std::path::Path;

/// The gateway's two listeners, as config lines: each on a port the kernel picks. Each at its own
/// loopback address, because Pingora keys a server's listeners by their address string — two
/// `127.0.0.1:0` collide (one listener comes up, and shutdown closes its descriptor twice) — and the
/// address is also how [`gateway_ports`] tells the client listener from the metrics one.
///
/// Linux only: `127.0.0.2` is loopback there without configuration; see [`require_proc`].
pub const GATEWAY_LISTENERS: &str = "listen = \"127.0.0.1:0\"\nmetrics_listen = \"127.0.0.2:0\"\n";

/// Where a gateway configured with [`GATEWAY_LISTENERS`] listens.
#[derive(Clone, Copy, Debug)]
pub struct GatewayPorts {
    /// Client traffic, on `127.0.0.1`.
    pub proxy: u16,
    /// `/metrics`, `/livez`, `/readyz`, on `127.0.0.2`.
    pub metrics: SocketAddr,
}

/// The ports gateway `pid` (configured with [`GATEWAY_LISTENERS`]) is listening on, once both of its
/// listeners are up; `None` until then. Poll it after spawning, checking the child has not exited.
pub fn gateway_ports(pid: u32) -> Option<GatewayPorts> {
    pick_gateway_ports(&listening_on(pid))
}

/// Which of a gateway's `LISTEN` sockets is which, by address.
fn pick_gateway_ports(bound: &[SocketAddrV4]) -> Option<GatewayPorts> {
    let at = |ip: [u8; 4]| bound.iter().find(|a| a.ip().octets() == ip).copied();
    Some(GatewayPorts {
        proxy: at([127, 0, 0, 1])?.port(),
        metrics: SocketAddr::V4(at([127, 0, 0, 2])?),
    })
}

/// The IPv4 addresses `pid` itself holds sockets in `LISTEN` on. `/proc/<pid>/net/tcp` lists every
/// socket in the network namespace, so it is filtered to the inodes among `pid`'s descriptors.
pub fn listening_on(pid: u32) -> Vec<SocketAddrV4> {
    require_proc();
    let Ok(fds) = std::fs::read_dir(format!("/proc/{pid}/fd")) else {
        return Vec::new();
    };
    let inodes: std::collections::HashSet<String> = fds
        .flatten()
        .filter_map(|fd| std::fs::read_link(fd.path()).ok())
        .filter_map(|target| {
            let target = target.to_string_lossy().into_owned();
            target
                .strip_prefix("socket:[")
                .and_then(|rest| rest.strip_suffix(']'))
                .map(str::to_owned)
        })
        .collect();
    let Ok(table) = std::fs::read_to_string(format!("/proc/{pid}/net/tcp")) else {
        return Vec::new();
    };
    // Columns: sl, local_address (hex addr:port, the address as the kernel's in-memory word), ...,
    // st (`0A` is LISTEN), ..., inode (10th).
    table
        .lines()
        .skip(1)
        .filter_map(|line| {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 10 || cols[3] != "0A" || !inodes.contains(cols[9]) {
                return None;
            }
            let (addr, port) = cols[1].split_once(':')?;
            let addr = u32::from_str_radix(addr, 16).ok()?;
            let port = u16::from_str_radix(port, 16).ok()?;
            Some(SocketAddrV4::new(addr.to_ne_bytes().into(), port))
        })
        .collect()
}

/// Fail now, naming the reason, where reading a child's ports cannot work — instead of every caller
/// polling an empty socket table until its startup deadline and reporting a "did not come up".
#[track_caller]
pub fn require_proc() {
    #[cfg(not(target_os = "linux"))]
    panic!(
        "unsupported platform: reading a child's ports needs Linux (/proc for its LISTEN sockets, \
         and 127.0.0.2 on loopback for the gateway's metrics listener)"
    );
    #[cfg(target_os = "linux")]
    require_proc_at(Path::new("/proc/self/net/tcp"));
}

#[cfg(target_os = "linux")]
#[track_caller]
fn require_proc_at(table: &Path) {
    assert!(
        std::fs::read_to_string(table).is_ok(),
        "unsupported environment: {} is not readable, and a child's ports are read from /proc \
         (mount procfs, or run these tests where it is)",
        table.display()
    );
}

/// The client port in the `<name>_<pid>.ports` file `nats-server --ports_file_dir <dir>` writes
/// (`{"nats":["nats://127.0.0.1:4222"], ...}`), once it is there and complete. The server writes it
/// after it is listening, so the port is then usable.
pub fn nats_port_from(dir: &Path) -> Option<u16> {
    let file = std::fs::read_dir(dir).ok()?.flatten().next()?.path();
    let ports: serde_json::Value = serde_json::from_slice(&std::fs::read(file).ok()?).ok()?;
    ports["nats"][0].as_str()?.rsplit(':').next()?.parse().ok()
}

/// A loopback port nothing listens on, held for as long as the value lives: bound but never
/// listening, so a connection is refused, and taken, so no other process can start listening there
/// in the meantime — which a port picked free and released cannot promise.
pub struct DeadPort {
    _socket: tokio::net::TcpSocket,
    port: u16,
}

impl DeadPort {
    pub fn bind() -> Self {
        let socket = tokio::net::TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .expect("bind a loopback port");
        let port = socket.local_addr().expect("its address").port();
        Self {
            _socket: socket,
            port,
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This process's own listener is found, at its address and port; a socket of another process is
    /// not.
    #[test]
    fn listening_on_finds_this_processes_own_listeners() {
        let a = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let b = std::net::TcpListener::bind("127.0.0.2:0").unwrap();
        let mine = listening_on(std::process::id());
        for l in [&a, &b] {
            let SocketAddr::V4(addr) = l.local_addr().unwrap() else {
                unreachable!()
            };
            assert!(mine.contains(&addr), "{addr} in {mine:?}");
        }
    }

    /// The listeners are told apart by address, and both must be up.
    #[test]
    fn gateway_ports_tells_the_listeners_apart_by_address() {
        let proxy = SocketAddrV4::new([127, 0, 0, 1].into(), 41000);
        let metrics = SocketAddrV4::new([127, 0, 0, 2].into(), 42000);
        let ports = pick_gateway_ports(&[metrics, proxy]).expect("both up");
        assert_eq!(ports.proxy, 41000);
        assert_eq!(ports.metrics, SocketAddr::V4(metrics));
        assert!(pick_gateway_ports(&[proxy]).is_none(), "metrics not up yet");
        assert!(pick_gateway_ports(&[metrics]).is_none(), "proxy not up yet");
    }

    /// Without a readable socket table the read-back fails at once, saying what is missing.
    #[cfg(target_os = "linux")]
    #[test]
    #[should_panic(expected = "unsupported environment")]
    fn a_missing_socket_table_fails_at_once_and_says_why() {
        require_proc_at(Path::new("/proc/no-such-table"));
    }

    /// Here it is readable, so the check passes.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_socket_table_is_readable_here() {
        require_proc();
    }

    /// A dead port refuses connections, and while it is held nobody can listen there.
    #[test]
    fn a_dead_port_refuses_and_cannot_be_taken() {
        let dead = DeadPort::bind();
        assert!(std::net::TcpStream::connect(("127.0.0.1", dead.port())).is_err());
        assert!(std::net::TcpListener::bind(("127.0.0.1", dead.port())).is_err());
    }
}
