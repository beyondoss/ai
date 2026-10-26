//! Logging: one line to stderr per event, a fixed level, nothing else.
//!
//! `YYYY-MM-DDTHH:MM:SS.mmmZ LEVEL message key=value ...`, UTC, written with one `write` so lines
//! from different threads do not interleave. A disabled level costs one atomic load and builds
//! nothing. The library logs nothing until the host calls [`set_level`]; the binary sets it from
//! `EXECD_LOG`. Values that come from outside are written with `{:?}`, which escapes a newline,
//! so a request cannot forge a line.
//!
//! This replaces a general tracing subscriber and its per-target filter language: execd has one
//! knob, and the subscriber with an env filter was about half a megabyte of resident memory in a
//! process whose floor is under a megabyte (ARCHITECTURE.md, "Memory").

use std::fmt::{Arguments, Write as _};
use std::io::Write as _;
use std::sync::atomic::{AtomicU8, Ordering};

/// Most severe first; a level logs itself and everything more severe.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    /// `off`, `error`, `warn`, `info`, `debug` or `trace`.
    pub fn parse(s: &str) -> Option<Level> {
        Some(match s {
            "off" => Level::Off,
            "error" => Level::Error,
            "warn" => Level::Warn,
            "info" => Level::Info,
            "debug" => Level::Debug,
            "trace" => Level::Trace,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Level::Off => "OFF",
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }
}

static LEVEL: AtomicU8 = AtomicU8::new(Level::Off as u8);

/// The most verbose level that is written.
pub fn set_level(level: Level) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

#[doc(hidden)]
pub fn enabled(level: Level) -> bool {
    level as u8 <= LEVEL.load(Ordering::Relaxed)
}

#[doc(hidden)]
pub fn emit(level: Level, args: Arguments<'_>) {
    let mut line = String::with_capacity(128);
    timestamp(&mut line);
    let _ = writeln!(line, " {:<5} {args}", level.name());
    let _ = std::io::stderr().lock().write_all(line.as_bytes());
}

/// `2026-10-25T22:22:09.385Z`, from the wall clock.
fn timestamp(out: &mut String) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let (secs, ms) = (now.as_secs(), now.subsec_millis());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (y, m, d) = civil_from_days(i64::try_from(days).unwrap_or(0));
    let _ = write!(
        out,
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    );
}

/// The proleptic Gregorian date of a day count since 1970-01-01 (Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + i64::from(m <= 2), m, d)
}

#[doc(hidden)]
#[macro_export]
macro_rules! log_at {
    ($level:expr, $($arg:tt)*) => {
        if $crate::log::enabled($level) {
            $crate::log::emit($level, format_args!($($arg)*));
        }
    };
}

/// Log at error level: `log_error!("message key={}", value)`.
#[macro_export]
macro_rules! log_error { ($($arg:tt)*) => { $crate::log_at!($crate::log::Level::Error, $($arg)*) }; }
/// Log at warn level.
#[macro_export]
macro_rules! log_warn { ($($arg:tt)*) => { $crate::log_at!($crate::log::Level::Warn, $($arg)*) }; }
/// Log at info level.
#[macro_export]
macro_rules! log_info { ($($arg:tt)*) => { $crate::log_at!($crate::log::Level::Info, $($arg)*) }; }
/// Log at debug level.
#[macro_export]
macro_rules! log_debug { ($($arg:tt)*) => { $crate::log_at!($crate::log::Level::Debug, $($arg)*) }; }

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn levels_parse_and_order() {
        assert_eq!(Level::parse("warn"), Some(Level::Warn));
        assert_eq!(Level::parse("WARN"), None);
        assert_eq!(Level::parse("info,foo=debug"), None);
        assert!(Level::Error < Level::Info && Level::Info < Level::Trace);
    }

    #[test]
    fn dates_match_known_days() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(civil_from_days(20_751), (2026, 10, 25));
    }
}
