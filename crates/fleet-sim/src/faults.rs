//! How a scenario breaks a replica, when the simulator does not own it.
//!
//! Locally the simulator owns everything: it spawned the replicas as child processes and it built
//! their network, so it kills a pid and toggles a veth. Attached to a real fleet it owns neither —
//! the replicas are ECS tasks and the network is a VPC — so every fault has to be somebody else's
//! to perform.
//!
//! That somebody is a caller-supplied command. It keeps the words `ECS`, `EFS` and `AWS` out of this
//! repo entirely, which is the boundary this crate is held to, and it means **the same matrix binary
//! runs in both places**: locally with a script that shells out to `ip link`, on AWS with one that
//! calls the EC2 API. A claim proved in one is proved by the same code in the other.

/// A fault injector: either the simulator's own mechanisms, or a command that owns them instead.
#[derive(Clone, Default)]
pub struct Faults {
    /// `<cmd> <action> <replica>`, when the fleet is attached rather than spawned.
    cmd: Option<String>,
}

/// What a scenario asks for. Spelled out rather than free-form strings at the call sites, so a typo
/// is a compile error and the set a fault script must implement is enumerable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Machine failure: no signal handler, no unwind, no chance to release a lock or seal a segment.
    /// This is the case the epoch fence exists for.
    Kill,
    /// The deploy case: `SIGTERM`, which a replica is supposed to survive gracefully.
    Term,
    /// Cut this replica off from its storage while its peers keep serving. Its lease stops being
    /// renewed, which is what makes a takeover lease-bound rather than instant.
    Partition,
    /// Give the storage back.
    Heal,
    /// Bring the replica back — a deploy replacing a task, not a process resurrecting.
    Restart,
}

impl Fault {
    /// The word passed to the fault command. Stable: a script matches on these.
    pub fn as_str(self) -> &'static str {
        match self {
            Fault::Kill => "kill",
            Fault::Term => "term",
            Fault::Partition => "partition",
            Fault::Heal => "heal",
            Fault::Restart => "restart",
        }
    }
}

impl Faults {
    /// Faults the simulator performs itself — the local substrates.
    pub fn local() -> Self {
        Self { cmd: None }
    }

    /// Faults a command performs on the simulator's behalf.
    pub fn external(cmd: impl Into<String>) -> Self {
        Self {
            cmd: Some(cmd.into()),
        }
    }

    /// Is a fault command configured? When it is, the local mechanisms are not used at all, even for
    /// a fault the simulator could technically perform — mixing the two would mean a scenario's
    /// faults came from two different places, which is the sort of thing that reads as a design
    /// failure when it is a harness one.
    pub fn is_external(&self) -> bool {
        self.cmd.is_some()
    }

    /// Ask for `fault` on `replica`, by name.
    ///
    /// Synchronous and blocking on purpose: a fault is a step in a scenario, and the next assertion
    /// is only meaningful once it has actually happened. The command's stderr comes back in the
    /// error, because a fault that silently did nothing produces a scenario that fails somewhere
    /// else entirely.
    pub fn run(&self, fault: Fault, replica: &str) -> Result<(), String> {
        let Some(cmd) = &self.cmd else {
            return Err(format!(
                "no fault command configured, so {} cannot be asked of {replica}",
                fault.as_str()
            ));
        };
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("{cmd} {} {replica}", fault.as_str()))
            .output()
            .map_err(|e| format!("fault command for {}: {e}", fault.as_str()))?;
        if out.status.success() {
            return Ok(());
        }
        Err(format!(
            "fault {} on {replica} failed ({}): {}",
            fault.as_str(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}
