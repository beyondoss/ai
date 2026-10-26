//! The command line: a table of flags and a small parser over it.
//!
//! `--flag value` or `--flag=value`; a switch is `--flag` alone (or `--flag=true|false`). Each flag
//! also reads an environment variable, and the command line wins over it, which wins over the
//! default. An unknown flag, a repeated flag, a flag without its value, a value that does not parse
//! and a stray argument are errors that say which. `--help` and `--version` print and stop.
//!
//! This is a table and a loop instead of a general argument-parsing crate: the crate's parser was
//! about 284 KiB of the resident memory of a process whose floor is under a megabyte (ARCHITECTURE.md,
//! "Memory"), for eight flags.

use std::ffi::OsString;
use std::path::PathBuf;

use execd::log::Level;

/// What the command line asked for.
#[derive(Debug, PartialEq)]
pub struct Cli {
    pub listen: String,
    pub root: Option<PathBuf>,
    pub home: Option<PathBuf>,
    pub bearer_file: Option<PathBuf>,
    pub insecure_no_auth: bool,
    pub run_as: Option<String>,
    pub kept_jobs_survive_idle: bool,
    pub max_response_bytes: usize,
    pub jobs_dir: Option<PathBuf>,
    pub log_level: Level,
}

/// What `parse` found.
#[derive(Debug, PartialEq)]
pub enum Parsed {
    Run(Cli),
    /// `--help`: the text to print.
    Help(String),
    /// `--version`: the line to print.
    Version(String),
}

struct Flag {
    name: &'static str,
    env: &'static str,
    /// The value's name in the help, `None` for a switch.
    value: Option<&'static str>,
    help: &'static str,
}

const FLAGS: &[Flag] = &[
    Flag {
        name: "listen",
        env: "EXECD_LISTEN",
        value: Some("ADDR"),
        help: "address to listen on (default 127.0.0.1:8080)",
    },
    Flag {
        name: "root",
        env: "EXECD_ROOT",
        value: Some("DIR"),
        help: "where a command runs when its request names no directory (default: the current directory)",
    },
    Flag {
        name: "home",
        env: "EXECD_HOME",
        value: Some("DIR"),
        help: "HOME for every command (default: execd's own, or with --run-as the user's own)",
    },
    Flag {
        name: "bearer-file",
        env: "EXECD_BEARER_FILE",
        value: Some("FILE"),
        help: "a file holding the bearer token every request must present (mode 0600 is best); \
               EXECD_BEARER holds the token itself, a weaker way; never an argument, argv shows in ps",
    },
    Flag {
        name: "insecure-no-auth",
        env: "EXECD_INSECURE_NO_AUTH",
        value: None,
        help: "serve with no bearer on a non-loopback address; only for a network that is itself the access control",
    },
    Flag {
        name: "run-as",
        env: "EXECD_RUN_AS",
        value: Some("USER"),
        help: "run every command as user, uid, user:group or uid:gid, with the user's groups; execd must be root",
    },
    Flag {
        name: "kept-jobs-survive-idle",
        env: "EXECD_KEPT_JOBS_SURVIVE_IDLE",
        value: None,
        help: "advertise v1.4 job-end notify; say so only if this sandbox is not stopped when idle",
    },
    Flag {
        name: "max-response-bytes",
        env: "EXECD_MAX_RESPONSE_BYTES",
        value: Some("BYTES"),
        help: "the most output one v1.1 answer carries; match the agent's --exec-max-response-bytes (default 16777216)",
    },
    Flag {
        name: "jobs-dir",
        env: "EXECD_JOBS_DIR",
        value: Some("DIR"),
        help: "where a kept job's last 8 MiB of output is spooled (default $TMPDIR, else /tmp); use a disk-backed directory",
    },
    Flag {
        name: "log-level",
        env: "EXECD_LOG",
        value: Some("LEVEL"),
        help: "off, error, warn, info, debug or trace (default info)",
    },
];

/// The help text.
pub fn help() -> String {
    let mut out = String::from(
        "execd: an exec endpoint for a sandbox (exec protocol v1.1 to v1.4)\n\n\
         usage: execd [options]\n\noptions (each also reads the environment variable shown):\n",
    );
    for f in FLAGS {
        let arg = f.value.map_or(String::new(), |v| format!(" <{v}>"));
        out.push_str(&format!(
            "  --{}{}\n      {} [{}]\n",
            f.name, arg, f.help, f.env
        ));
    }
    out.push_str("  --help, --version\n\nEXECD_WORKER_THREADS: 1 (default) one thread, 0 a worker per CPU, N workers.\n");
    out
}

fn truthy(name: &str, v: &str) -> Result<bool, String> {
    match v {
        "1" | "true" | "yes" | "on" => Ok(true),
        "" | "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("{name}: {v:?} is not true or false")),
    }
}

/// Parse `args` (without the program name) against `env`.
pub fn parse<A, E>(args: A, env: E) -> Result<Parsed, String>
where
    A: IntoIterator<Item = OsString>,
    E: Fn(&str) -> Option<String>,
{
    // The command line, flag by flag.
    let mut given: Vec<Option<String>> = vec![None; FLAGS.len()];
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let arg = arg
            .into_string()
            .map_err(|a| format!("argument {a:?} is not UTF-8"))?;
        match arg.as_str() {
            "-h" | "--help" => return Ok(Parsed::Help(help())),
            "-V" | "--version" => {
                return Ok(Parsed::Version(format!(
                    "execd {}",
                    env!("CARGO_PKG_VERSION")
                )));
            }
            _ => {}
        }
        let Some(body) = arg.strip_prefix("--") else {
            return Err(format!("unexpected argument {arg:?} (see --help)"));
        };
        let (name, inline) = match body.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (body, None),
        };
        let Some(i) = FLAGS.iter().position(|f| f.name == name) else {
            return Err(format!("unknown option --{name} (see --help)"));
        };
        if given[i].is_some() {
            return Err(format!("--{name} given more than once"));
        }
        given[i] = Some(if FLAGS[i].value.is_none() {
            inline.unwrap_or_else(|| "true".into())
        } else if let Some(v) = inline {
            v
        } else {
            match it.next().map(|v| v.into_string()) {
                Some(Ok(v)) if !v.starts_with("--") => v,
                Some(Ok(_)) | None => return Err(format!("--{name} needs a value")),
                Some(Err(v)) => return Err(format!("the value of --{name} {v:?} is not UTF-8")),
            }
        });
    }

    // Command line, then environment.
    let mut value = |name: &str| -> Option<(String, bool)> {
        let i = FLAGS.iter().position(|f| f.name == name)?;
        match given[i].take() {
            Some(v) => Some((v, true)),
            None => env(FLAGS[i].env).map(|v| (v, false)),
        }
    };
    let text = |v: Option<(String, bool)>| v.map(|(v, _)| v);
    let on = |name: &str, v: Option<(String, bool)>| -> Result<bool, String> {
        match v {
            None => Ok(false),
            Some((v, _)) => truthy(&format!("--{name}"), &v),
        }
    };
    let path = |v: Option<(String, bool)>| text(v).filter(|s| !s.is_empty()).map(PathBuf::from);
    let cli = Cli {
        listen: text(value("listen")).unwrap_or_else(|| "127.0.0.1:8080".into()),
        root: path(value("root")),
        home: path(value("home")),
        bearer_file: path(value("bearer-file")),
        insecure_no_auth: on("insecure-no-auth", value("insecure-no-auth"))?,
        run_as: text(value("run-as")).filter(|s| !s.is_empty()),
        kept_jobs_survive_idle: on("kept-jobs-survive-idle", value("kept-jobs-survive-idle"))?,
        max_response_bytes: match text(value("max-response-bytes")) {
            None => execd::MAX_RESPONSE_BYTES,
            Some(v) => v
                .parse()
                .map_err(|_| format!("--max-response-bytes: {v:?} is not a number of bytes"))?,
        },
        jobs_dir: path(value("jobs-dir")),
        log_level: match text(value("log-level")) {
            None => Level::Info,
            Some(v) => Level::parse(&v).ok_or_else(|| {
                format!("--log-level: {v:?} is not off, error, warn, info, debug or trace")
            })?,
        },
    };
    Ok(Parsed::Run(cli))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn run(args: &[&str], env: &[(&str, &str)]) -> Result<Parsed, String> {
        parse(args.iter().map(OsString::from), |k| {
            env.iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        })
    }

    fn cli(args: &[&str], env: &[(&str, &str)]) -> Cli {
        match run(args, env).unwrap() {
            Parsed::Run(c) => c,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_defaults() {
        let c = cli(&[], &[]);
        assert_eq!(c.listen, "127.0.0.1:8080");
        assert_eq!(c.root, None);
        assert_eq!(c.home, None);
        assert_eq!(c.bearer_file, None);
        assert!(!c.insecure_no_auth && !c.kept_jobs_survive_idle);
        assert_eq!(c.run_as, None);
        assert_eq!(c.max_response_bytes, execd::MAX_RESPONSE_BYTES);
        assert_eq!(c.jobs_dir, None);
        assert_eq!(c.log_level, Level::Info);
    }

    #[test]
    fn every_flag_by_argument_in_both_spellings() {
        let c = cli(
            &[
                "--listen",
                "0.0.0.0:9",
                "--root",
                "/r",
                "--home=/h",
                "--bearer-file",
                "/b",
                "--insecure-no-auth",
                "--run-as=nobody:nogroup",
                "--kept-jobs-survive-idle",
                "--max-response-bytes",
                "123",
                "--jobs-dir=/j",
                "--log-level",
                "debug",
            ],
            &[],
        );
        assert_eq!(c.listen, "0.0.0.0:9");
        assert_eq!(c.root, Some("/r".into()));
        assert_eq!(c.home, Some("/h".into()));
        assert_eq!(c.bearer_file, Some("/b".into()));
        assert!(c.insecure_no_auth && c.kept_jobs_survive_idle);
        assert_eq!(c.run_as.as_deref(), Some("nobody:nogroup"));
        assert_eq!(c.max_response_bytes, 123);
        assert_eq!(c.jobs_dir, Some("/j".into()));
        assert_eq!(c.log_level, Level::Debug);
    }

    #[test]
    fn every_flag_by_environment() {
        let c = cli(
            &[],
            &[
                ("EXECD_LISTEN", "[::1]:7"),
                ("EXECD_ROOT", "/r"),
                ("EXECD_HOME", "/h"),
                ("EXECD_BEARER_FILE", "/b"),
                ("EXECD_INSECURE_NO_AUTH", "1"),
                ("EXECD_RUN_AS", "1000:1000"),
                ("EXECD_KEPT_JOBS_SURVIVE_IDLE", "true"),
                ("EXECD_MAX_RESPONSE_BYTES", "9"),
                ("EXECD_JOBS_DIR", "/j"),
                ("EXECD_LOG", "warn"),
            ],
        );
        assert_eq!(c.listen, "[::1]:7");
        assert_eq!(c.root, Some("/r".into()));
        assert_eq!(c.home, Some("/h".into()));
        assert_eq!(c.bearer_file, Some("/b".into()));
        assert!(c.insecure_no_auth && c.kept_jobs_survive_idle);
        assert_eq!(c.run_as.as_deref(), Some("1000:1000"));
        assert_eq!(c.max_response_bytes, 9);
        assert_eq!(c.jobs_dir, Some("/j".into()));
        assert_eq!(c.log_level, Level::Warn);
    }

    #[test]
    fn the_command_line_beats_the_environment_which_beats_the_default() {
        let env = [("EXECD_LISTEN", "env:1"), ("EXECD_ROOT", "/env")];
        let c = cli(&["--listen", "arg:2"], &env);
        assert_eq!(c.listen, "arg:2");
        assert_eq!(c.root, Some("/env".into()));
        // A switch given by argument stays on whatever the environment says.
        let c = cli(&["--insecure-no-auth"], &[("EXECD_INSECURE_NO_AUTH", "0")]);
        assert!(c.insecure_no_auth);
    }

    #[test]
    fn switches_take_the_usual_spellings() {
        for (v, want) in [
            ("1", true),
            ("true", true),
            ("yes", true),
            ("on", true),
            ("0", false),
            ("false", false),
            ("", false),
        ] {
            let c = cli(&[], &[("EXECD_INSECURE_NO_AUTH", v)]);
            assert_eq!(c.insecure_no_auth, want, "{v:?}");
        }
        assert!(cli(&["--insecure-no-auth=true"], &[]).insecure_no_auth);
        assert!(!cli(&["--insecure-no-auth=false"], &[]).insecure_no_auth);
        let e = run(&[], &[("EXECD_INSECURE_NO_AUTH", "maybe")]).unwrap_err();
        assert!(e.contains("not true or false"), "{e}");
    }

    #[test]
    fn an_empty_path_or_user_in_the_environment_is_unset() {
        let c = cli(
            &[],
            &[
                ("EXECD_ROOT", ""),
                ("EXECD_RUN_AS", ""),
                ("EXECD_JOBS_DIR", ""),
            ],
        );
        assert_eq!((c.root, c.run_as, c.jobs_dir), (None, None, None));
    }

    #[test]
    fn errors_say_what_is_wrong() {
        let err = |args: &[&str]| run(args, &[]).unwrap_err();
        assert!(err(&["--nope"]).contains("unknown option --nope"));
        assert!(err(&["--listen", "a", "--listen", "b"]).contains("--listen given more than once"));
        assert!(err(&["--listen=a", "--listen", "b"]).contains("more than once"));
        assert!(err(&["--listen"]).contains("--listen needs a value"));
        assert!(err(&["--root", "--home", "/h"]).contains("--root needs a value"));
        assert!(err(&["stray"]).contains("unexpected argument"));
        assert!(err(&["-x"]).contains("unexpected argument"));
        assert!(err(&["--max-response-bytes", "lots"]).contains("not a number"));
        assert!(err(&["--max-response-bytes", "-1"]).contains("not a number"));
        assert!(err(&["--log-level", "info,foo=debug"]).contains("not off, error"));
        let e = run(&[], &[("EXECD_LOG", "loud")]).unwrap_err();
        assert!(e.contains("not off, error"), "{e}");
    }

    #[test]
    fn help_and_version_stop_the_parse() {
        match run(&["--help"], &[]).unwrap() {
            Parsed::Help(t) => {
                for f in FLAGS {
                    assert!(
                        t.contains(&format!("--{}", f.name)) && t.contains(f.env),
                        "{}",
                        f.name
                    );
                }
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(run(&["-h"], &[]).unwrap(), Parsed::Help(_)));
        match run(&["--version"], &[]).unwrap() {
            Parsed::Version(v) => assert!(v.starts_with("execd 0."), "{v}"),
            other => panic!("{other:?}"),
        }
        // `--help` wins even after an unknown flag was not reached, and before a bad value.
        assert!(matches!(
            run(&["--listen", "x", "--help"], &[]).unwrap(),
            Parsed::Help(_)
        ));
    }

    #[test]
    fn a_non_utf8_argument_is_an_error() {
        use std::os::unix::ffi::OsStringExt as _;
        let bad = OsString::from_vec(vec![b'-', b'-', 0xff]);
        let e = parse([bad], |_| None).unwrap_err();
        assert!(e.contains("not UTF-8"), "{e}");
    }
}
