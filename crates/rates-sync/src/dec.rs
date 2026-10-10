//! Exact decimal rates. A vendor publishes a rate as decimal text (per million tokens, per token,
//! or in xAI's 1e-10 USD ticks per token); every conversion here is exact or an error, never a
//! rounding.

use crate::Result;

/// USD per million tokens, in units of 1e-12 USD. Twelve places hold a per-token price with up to
/// eighteen (OpenRouter publishes `"0.000000002994"`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Rate(pub u128);

const SCALE: u32 = 12;

impl Rate {
    pub const ZERO: Rate = Rate(0);

    /// A decimal USD amount per million tokens: `"1.25"`, `"0.075"`.
    pub fn per_million(text: &str) -> Result<Rate> {
        decimal(text, SCALE).map(Rate)
    }

    /// A decimal USD amount per token (OpenRouter): `"0.00000125"` is $1.25 per million.
    pub fn per_token(text: &str) -> Result<Rate> {
        decimal(text, SCALE + 6).map(Rate)
    }

    /// xAI's integer price in 1e-10 USD ticks per token: `12500` is $1.25 per million.
    pub fn xai_ticks(ticks: u64) -> Rate {
        // 1 tick/token = 1e-10 USD/token = 1e-4 USD/million = 1e8 units.
        Rate(u128::from(ticks) * 100_000_000)
    }

    /// `self × num / den`, exactly.
    pub fn ratio(self, num: u128, den: u128) -> Result<Rate> {
        let p = self
            .0
            .checked_mul(num)
            .ok_or_else(|| format!("{self} x {num}/{den} overflows"))?;
        if den == 0 || p % den != 0 {
            return Err(format!("{self} x {num}/{den} is not exact"));
        }
        Ok(Rate(p / den))
    }

    /// The rate as the decimal text the table holds: at most six places per million tokens (the
    /// pricer's precision), no trailing zeros. A rate finer than that is an error.
    pub fn table_text(self) -> Result<String> {
        let unit = 10u128.pow(SCALE);
        let micro = 10u128.pow(SCALE - 6);
        if !self.0.is_multiple_of(micro) {
            return Err(format!(
                "{self} per million has more than six decimal places; the pricer cannot hold it"
            ));
        }
        Ok(self.to_string_trimmed(unit))
    }

    fn to_string_trimmed(self, unit: u128) -> String {
        let whole = self.0 / unit;
        let mut frac = format!("{:012}", self.0 % unit);
        while frac.ends_with('0') {
            frac.pop();
        }
        if frac.is_empty() {
            whole.to_string()
        } else {
            format!("{whole}.{frac}")
        }
    }
}

impl std::fmt::Display for Rate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.to_string_trimmed(10u128.pow(SCALE)))
    }
}

/// A plain non-negative decimal (`"12.50"`, `"0.0000000001"`) as an integer scaled by
/// `10^shift`; more places than `shift` is an error.
pub fn decimal(text: &str, shift: u32) -> Result<u128> {
    let (whole, frac) = match text.split_once('.') {
        Some((w, f)) => (w, f),
        None => (text, ""),
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let dotted = text.contains('.');
    if !digits(whole) || (dotted && !digits(frac)) {
        return Err(format!("{text:?} is not a plain decimal"));
    }
    let frac = frac.trim_end_matches('0');
    let places = u32::try_from(frac.len()).map_err(|_| format!("{text:?}: too long"))?;
    if places > shift {
        return Err(format!("{text:?} has more than {shift} decimal places"));
    }
    let w: u128 = whole.parse().map_err(|_| format!("{text:?}: too large"))?;
    let f: u128 = if frac.is_empty() {
        0
    } else {
        frac.parse().map_err(|_| format!("{text:?}: too large"))?
    };
    w.checked_mul(10u128.pow(shift))
        .and_then(|w| w.checked_add(f * 10u128.pow(shift - places)))
        .ok_or_else(|| format!("{text:?}: too large"))
}

/// A JSON number Together publishes as a float (`1.0399999999999998`) as the decimal it stands
/// for: the nearest six-place value, which must be within 1e-9 of the float, or it is an error.
pub fn float_per_million(x: f64) -> Result<Rate> {
    if !x.is_finite() || x < 0.0 {
        return Err(format!("{x} is not a rate"));
    }
    let micros = (x * 1e6).round();
    if (micros / 1e6 - x).abs() > 1e-9 {
        return Err(format!(
            "{x} is not a decimal with at most six places per million"
        ));
    }
    // A rate is far below 2^53 micro-dollars, so the float holds the integer exactly.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let micros = micros as u128;
    Ok(Rate(micros * 1_000_000))
}

/// A multiplier written as decimal text (`"1.1"`, `"2"`, `"0.5"`) as basis points.
pub fn bps(text: &str) -> Result<u128> {
    decimal(text, 4)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn conversions_are_exact() {
        assert_eq!(
            Rate::per_million("12.50").unwrap().table_text().unwrap(),
            "12.5"
        );
        assert_eq!(
            Rate::per_token("0.000000002994")
                .unwrap()
                .table_text()
                .unwrap(),
            "0.002994"
        );
        assert_eq!(Rate::xai_ticks(12500).table_text().unwrap(), "1.25");
        assert!(
            Rate::per_token("0.0000000000001")
                .unwrap()
                .table_text()
                .is_err()
        );
        assert_eq!(
            float_per_million(1.0399999999999998)
                .unwrap()
                .table_text()
                .unwrap(),
            "1.04"
        );
        assert!(float_per_million(0.1234567).is_err());
        let r = Rate::per_million("5").unwrap();
        assert_eq!(
            r.ratio(11_000, 10_000).unwrap().table_text().unwrap(),
            "5.5"
        );
        assert!(Rate::per_million("x").is_err());
        assert!(Rate::per_million("1.").is_err());
        assert_eq!(bps("1.1").unwrap(), 11_000);
    }
}
