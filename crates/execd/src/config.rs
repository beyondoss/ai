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
  EXECD_LOG                       off, error, warn, info, debug or trace (default info)
  EXECD_WORKER_THREADS            1 (default) one thread, 0 a worker per CPU, N workers
  EXECD_IDLE_TIMEOUT_MS           close a connection with no request in flight for this long; 0 (default) never
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
