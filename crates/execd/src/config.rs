//! The configuration: environment variables, read once at startup. There are no flags.
//!
//! Each variable is parsed here and nowhere else; a value that does not parse is an error naming
//! the variable and the value. An empty variable is unset. A boolean is `1`, `true`, `0` or
//! `false`. The binary takes no arguments (ARCHITECTURE.md, "Configuration"): a flag parser was
//! about 284 KiB of the resident memory of a process whose floor is under a megabyte.

use std::path::PathBuf;
use std::time::Duration;

use execd::log::Level;

/// What the environment asked for.
#[derive(Debug, PartialEq)]
pub struct Config {
    pub listen: String,
    pub root: Option<PathBuf>,
    pub home: Option<PathBuf>,
    pub bearer_file: Option<PathBuf>,
    pub insecure_no_auth: bool,
    pub run_as: Option<String>,
    pub max_response_bytes: usize,
    pub jobs_dir: Option<PathBuf>,
    pub state_dir: Option<PathBuf>,
    /// `EXECD_TERMINALS_MAX`: a cap on open terminals, for tests (see `Options::terminals_max`).
    pub terminals_max: Option<usize>,
    /// `EXECD_JOB_OUTPUT_MS`, `EXECD_JOBS_ENDED_BYTES`, `EXECD_JOBS_ENDED_MAX`, `EXECD_JOB_ENTRY_MS`:
    /// how long an ended job keeps its output and its entry (see `Options::retention`).
    pub retention: execd::Retention,
    /// `EXECD_MAX_STREAMS`: unset sizes the cap from the descriptor limit, `0` is no cap, `N` is N.
    pub max_streams: Option<usize>,
    /// `EXECD_RAISE_NOFILE`: raise the soft `RLIMIT_NOFILE` to the hard limit at start (default 1).
    pub raise_nofile: bool,
    /// `EXECD_SHED_WHEN_OUT_OF_FDS`: see `Options::shed_when_out_of_fds` (default 1).
    pub shed_when_out_of_fds: bool,
    pub log_level: Level,
    pub worker_threads: usize,
    pub idle_timeout: Option<Duration>,
}

/// The variables, printed when execd is given an argument. `EXECD_BEARER_FD` is internal.
pub const USAGE: &str = "\
execd takes no arguments; it is configured by environment variables:

  EXECD_LISTEN                    address to listen on (default 127.0.0.1:8080)
  EXECD_ROOT                      where a command runs when its request names no directory (default: the current directory)
  EXECD_HOME                      HOME for every command (default: execd's own, or with EXECD_RUN_AS the user's own)
  EXECD_BEARER_FILE               a file holding the bearer token every request must present (mode 0600 is best)
  EXECD_BEARER                    the bearer token itself, a weaker way than the file; execd refuses to start if it cannot move it out of the environment
  EXECD_INSECURE_NO_AUTH          1|true|0|false: serve with no bearer on a non-loopback address (default 0)
  EXECD_RUN_AS                    run every command as user, uid, user:group or uid:gid; execd must be root
  EXECD_MAX_RESPONSE_BYTES        the most output one answer carries (default 16777216)
  EXECD_JOBS_DIR                  where a kept job's last 8 MiB of output is spooled (default $TMPDIR, else /tmp)
  EXECD_STATE_DIR                 where the jobs ledger and each job's output files are kept, so a restart (SIGHUP) of execd leaves its jobs running and finds them; a stop (SIGTERM) always ends them (default none: a restart kills them too; use a tmpfs directory)
  EXECD_LOG                       off, error, warn, info, debug or trace (default info)
  EXECD_WORKER_THREADS            1 (default) one thread, 0 a worker per CPU, N workers
  EXECD_MAX_STREAMS               the most requests in flight, each over its limit answered 503: unset sizes it from the descriptor limit, 0 no cap
  EXECD_RAISE_NOFILE              1|true|0|false: raise the soft open file limit to the hard limit at start (default 1)
  EXECD_SHED_WHEN_OUT_OF_FDS      1|true|0|false: out of descriptors, answer waiting connections 503 instead of leaving them queued (default 1)
  EXECD_IDLE_TIMEOUT_MS           close a connection with no request in flight for this long; 0 (default) never
  EXECD_JOB_OUTPUT_MS             drop an ended job's output this long after its end, keeping its status (default 900000 = 15 min; 0 never)
  EXECD_JOBS_ENDED_BYTES          the most output retained over ended jobs; past it the oldest is dropped first (default 268435456 = 256 MiB; 0 unbounded)
  EXECD_JOBS_ENDED_MAX            the most ended jobs the server holds; past it the oldest entry is reaped (default 4096; 0 unbounded)
  EXECD_JOB_ENTRY_MS              reap an ended job's entry, status included, this long after its end (default 86400000 = 24 h; 0 never)
";

fn number<T: std::str::FromStr>(name: &str, v: Option<String>) -> Result<Option<T>, String> {
    v.map(|s| {
        s.parse::<T>()
            .map_err(|_| format!("{name}={s:?} is not a non-negative integer"))
    })
    .transpose()
}

fn boolean(name: &str, v: Option<String>) -> Result<bool, String> {
    match v.as_deref() {
        None | Some("0" | "false") => Ok(false),
        Some("1" | "true") => Ok(true),
        Some(s) => Err(format!("{name}={s:?} is not 1, true, 0 or false")),
    }
}

/// Read every setting through `env`.
pub fn parse<E: Fn(&str) -> Option<String>>(env: E) -> Result<Config, String> {
    // An empty variable is unset.
    let get = |k: &str| env(k).filter(|v| !v.is_empty());
    Ok(Config {
        listen: get("EXECD_LISTEN").unwrap_or_else(|| "127.0.0.1:8080".to_string()),
        root: get("EXECD_ROOT").map(PathBuf::from),
        home: get("EXECD_HOME").map(PathBuf::from),
        bearer_file: get("EXECD_BEARER_FILE").map(PathBuf::from),
        insecure_no_auth: boolean("EXECD_INSECURE_NO_AUTH", get("EXECD_INSECURE_NO_AUTH"))?,
        run_as: get("EXECD_RUN_AS"),
        max_response_bytes: number("EXECD_MAX_RESPONSE_BYTES", get("EXECD_MAX_RESPONSE_BYTES"))?
            .unwrap_or(execd::MAX_RESPONSE_BYTES),
        jobs_dir: get("EXECD_JOBS_DIR").map(PathBuf::from),
        state_dir: get("EXECD_STATE_DIR").map(PathBuf::from),
        terminals_max: get("EXECD_TERMINALS_MAX").and_then(|v| v.parse().ok()),
        retention: execd::Retention {
            output_ms: number::<u64>("EXECD_JOB_OUTPUT_MS", get("EXECD_JOB_OUTPUT_MS"))?
                .unwrap_or(execd::ENDED_OUTPUT_MS),
            bytes: number::<u64>("EXECD_JOBS_ENDED_BYTES", get("EXECD_JOBS_ENDED_BYTES"))?
                .unwrap_or(execd::ENDED_BYTES),
            max: number::<usize>("EXECD_JOBS_ENDED_MAX", get("EXECD_JOBS_ENDED_MAX"))?
                .unwrap_or(execd::ENDED_MAX),
            entry_ms: number::<u64>("EXECD_JOB_ENTRY_MS", get("EXECD_JOB_ENTRY_MS"))?
                .unwrap_or(execd::ENDED_ENTRY_MS),
        },
        max_streams: number("EXECD_MAX_STREAMS", get("EXECD_MAX_STREAMS"))?,
        raise_nofile: get("EXECD_RAISE_NOFILE").is_none()
            || boolean("EXECD_RAISE_NOFILE", get("EXECD_RAISE_NOFILE"))?,
        shed_when_out_of_fds: get("EXECD_SHED_WHEN_OUT_OF_FDS").is_none()
            || boolean(
                "EXECD_SHED_WHEN_OUT_OF_FDS",
                get("EXECD_SHED_WHEN_OUT_OF_FDS"),
            )?,
        log_level: match get("EXECD_LOG") {
            None => Level::Info,
            Some(s) => Level::parse(&s).ok_or_else(|| {
                format!("EXECD_LOG={s:?} is not off, error, warn, info, debug or trace")
            })?,
        },
        worker_threads: number("EXECD_WORKER_THREADS", get("EXECD_WORKER_THREADS"))?.unwrap_or(1),
        idle_timeout: number::<u64>("EXECD_IDLE_TIMEOUT_MS", get("EXECD_IDLE_TIMEOUT_MS"))?
            .filter(|ms| *ms > 0)
            .map(Duration::from_millis),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn cfg(env: &[(&str, &str)]) -> Result<Config, String> {
        parse(|k| {
            env.iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| (*v).to_string())
        })
    }

    #[test]
    fn the_defaults() {
        let c = cfg(&[]).unwrap();
        assert_eq!(c.listen, "127.0.0.1:8080");
        assert!(c.root.is_none() && c.home.is_none() && c.bearer_file.is_none());
        assert!(c.run_as.is_none() && c.jobs_dir.is_none());
        assert!(!c.insecure_no_auth);
        assert_eq!(c.max_response_bytes, execd::MAX_RESPONSE_BYTES);
        assert_eq!(c.log_level, Level::Info);
        assert_eq!(c.worker_threads, 1);
        assert_eq!(c.idle_timeout, None);
        assert_eq!(c.retention, execd::Retention::default());
        assert_eq!(
            c.retention,
            execd::Retention {
                output_ms: 900_000,
                bytes: 256 << 20,
                max: 4096,
                entry_ms: 86_400_000,
            }
        );
        assert_eq!(c.max_streams, None);
        assert!(c.raise_nofile && c.shed_when_out_of_fds);
    }

    /// Each retention knob is read, and `0` for all four is off (`Retention::enabled` is false):
    /// the sweeper is never started, which is today's behaviour.
    #[test]
    fn the_retention_knobs_are_read_and_zero_is_off() {
        let c = cfg(&[
            ("EXECD_JOB_OUTPUT_MS", "1"),
            ("EXECD_JOBS_ENDED_BYTES", "2"),
            ("EXECD_JOBS_ENDED_MAX", "3"),
            ("EXECD_JOB_ENTRY_MS", "4"),
        ])
        .unwrap();
        assert_eq!(
            c.retention,
            execd::Retention {
                output_ms: 1,
                bytes: 2,
                max: 3,
                entry_ms: 4,
            }
        );
        let off = cfg(&[
            ("EXECD_JOB_OUTPUT_MS", "0"),
            ("EXECD_JOBS_ENDED_BYTES", "0"),
            ("EXECD_JOBS_ENDED_MAX", "0"),
            ("EXECD_JOB_ENTRY_MS", "0"),
        ])
        .unwrap();
        assert!(!off.retention.enabled());
        assert!(c.retention.enabled() && cfg(&[]).unwrap().retention.enabled());
        for k in [
            "EXECD_JOB_OUTPUT_MS",
            "EXECD_JOBS_ENDED_BYTES",
            "EXECD_JOBS_ENDED_MAX",
            "EXECD_JOB_ENTRY_MS",
        ] {
            let e = cfg(&[(k, "-1")]).unwrap_err();
            assert!(e.contains(k), "{e}");
        }
    }

    /// The request cap is read and `0` turns it off.
    #[test]
    fn the_descriptor_knobs_turn_off_with_zero() {
        let c = cfg(&[
            ("EXECD_MAX_STREAMS", "0"),
            ("EXECD_RAISE_NOFILE", "0"),
            ("EXECD_SHED_WHEN_OUT_OF_FDS", "false"),
        ])
        .unwrap();
        assert_eq!(c.max_streams, Some(0));
        assert!(!c.raise_nofile && !c.shed_when_out_of_fds);
        assert!(cfg(&[("EXECD_RAISE_NOFILE", "maybe")]).is_err());
    }

    #[test]
    fn every_variable_is_read() {
        let c = cfg(&[
            ("EXECD_LISTEN", "0.0.0.0:9"),
            ("EXECD_ROOT", "/work"),
            ("EXECD_HOME", "/home/x"),
            ("EXECD_BEARER_FILE", "/run/token"),
            ("EXECD_INSECURE_NO_AUTH", "true"),
            ("EXECD_RUN_AS", "nobody:nogroup"),
            ("EXECD_MAX_RESPONSE_BYTES", "1024"),
            ("EXECD_JOBS_DIR", "/var/jobs"),
            ("EXECD_LOG", "debug"),
            ("EXECD_WORKER_THREADS", "0"),
            ("EXECD_IDLE_TIMEOUT_MS", "1500"),
        ])
        .unwrap();
        assert_eq!(c.listen, "0.0.0.0:9");
        assert_eq!(c.root, Some("/work".into()));
        assert_eq!(c.home, Some("/home/x".into()));
        assert_eq!(c.bearer_file, Some("/run/token".into()));
        assert!(c.insecure_no_auth);
        assert_eq!(c.run_as.as_deref(), Some("nobody:nogroup"));
        assert_eq!(c.max_response_bytes, 1024);
        assert_eq!(c.jobs_dir, Some("/var/jobs".into()));
        assert_eq!(c.log_level, Level::Debug);
        assert_eq!(c.worker_threads, 0);
        assert_eq!(c.idle_timeout, Some(Duration::from_millis(1500)));
        let zero = cfg(&[("EXECD_IDLE_TIMEOUT_MS", "0")]).unwrap();
        assert_eq!(zero.idle_timeout, None);
    }

    #[test]
    fn booleans_are_one_true_zero_false() {
        for (v, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            let c = cfg(&[("EXECD_INSECURE_NO_AUTH", v)]).unwrap();
            assert_eq!(c.insecure_no_auth, want, "{v}");
        }
        for bad in ["yes", "on", "TRUE", "2"] {
            let e = cfg(&[("EXECD_INSECURE_NO_AUTH", bad)]).unwrap_err();
            assert!(
                e.contains("EXECD_INSECURE_NO_AUTH") && e.contains(bad),
                "{e}"
            );
        }
    }

    #[test]
    fn an_empty_variable_is_unset() {
        let c = cfg(&[
            ("EXECD_ROOT", ""),
            ("EXECD_RUN_AS", ""),
            ("EXECD_LISTEN", ""),
            ("EXECD_INSECURE_NO_AUTH", ""),
        ])
        .unwrap();
        assert!(c.root.is_none() && c.run_as.is_none() && !c.insecure_no_auth);
        assert_eq!(c.listen, "127.0.0.1:8080");
    }

    #[test]
    fn errors_name_the_variable_and_the_value() {
        for (k, v) in [
            ("EXECD_MAX_RESPONSE_BYTES", "lots"),
            ("EXECD_WORKER_THREADS", "-1"),
            ("EXECD_IDLE_TIMEOUT_MS", "soon"),
            ("EXECD_LOG", "info,foo=debug"),
            ("EXECD_INSECURE_NO_AUTH", "maybe"),
        ] {
            let e = cfg(&[(k, v)]).unwrap_err();
            assert!(e.contains(k) && e.contains(v), "{e}");
        }
    }
}
