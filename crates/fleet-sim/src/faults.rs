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
//!
//! The command answers one question as well as performing five actions: `<cmd> address <replica>`
//! prints where that replica answers now. A restarted replica is not obliged to come back where it
//! was — a replaced Fargate task draws a fresh address from its subnet — and the alternative to
//! asking is for this crate to know how the platform assigns addresses, which is the thing it is
//! built not to know.

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
    /// The deploy case: `SIGTERM`, and **return at once**. The window a drain scenario asserts on
    /// is the one where the replica is still up and still finishing what it owns, so a fault that
    /// waited for the process to be gone would close the window before anything could look through
    /// it — which is exactly what the first EFS run did: all four C7 assertions came back against a
    /// replica that had already exited.
    Term,
    /// `SIGTERM`, and wait for it to actually be gone. What chaos wants, where the next event must
    /// not begin until this one has finished.
    ///
    /// A separate word from [`Fault::Term`] rather than a flag on it, because the difference is not
    /// a nuance: one of them is defined by the replica still being there.
    Stop,
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
            Fault::Stop => "stop",
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

    /// Where `replica` answers **now**, by asking whoever runs it.
    ///
    /// A restarted replica does not have to come back at the address it left. Locally it does — the
    /// port is in the saved command line — but a replaced Fargate task draws a fresh private address
    /// from its subnet, and nothing in ECS will pin one. So the address is a question for the same
    /// party that performs the faults, asked through the same seam, rather than an assumption this
    /// crate makes about a platform it deliberately knows nothing about.
    pub fn address(&self, replica: &str) -> Result<String, String> {
        let Some(cmd) = &self.cmd else {
            return Err(format!(
                "no fault command configured, so {replica}'s address cannot be asked for"
            ));
        };
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("{cmd} address {replica}"))
            .output()
            .map_err(|e| format!("fault command for address: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "address of {replica} failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let addr = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        if addr.is_empty() {
            return Err(format!("address of {replica} came back empty"));
        }
        Ok(addr)
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
