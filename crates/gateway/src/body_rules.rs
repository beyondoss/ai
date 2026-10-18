//! The rules a request body is refused by, in one order, checked wherever the gateway holds a body.
//!
//! A body is checked at three [`Point`]s, depending on how it arrives: before connecting (a small
//! catalog body read to choose its row), before a large catalog body's `FullBody` re-run, and once a
//! body is buffered whole in `request_body_filter` (a `/{provider}` relay, or that re-run). Each
//! [`Rule`] says where it applies, and [`Rule::ORDER`] is refusal precedence everywhere: a body that
//! breaks two rules is told the first. So the same body can be answered differently by point, but
//! only because a rule does not apply there, never because the order differs.
//!
//! A rule reads [`Facts`], never the bytes: each point extracts what applies there in one place
//! (`proxy`), from the scan it already made. Rewriting the body (the output-limit clamp, the
//! translation, the strips, the model re-spelling, `stream_options`) is a separate step after the
//! rules pass, on a buffered body only. Signed ids are checked apart too: verifying them also
//! rewrites the body, and a store relay checks them where it has the signer, not here. A foreign
//! id ([`Refusal::ForeignId`], [`Refusal::ForeignIdLate`] once the headers went upstream) is told
//! after every rule at [`Point::Connect`], and before every rule at [`Point::Relay`] and
//! [`Point::Buffered`]; `tests/refusal_precedence.rs` pins each.

use crate::refusal::Refusal;

/// Where a body is checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    /// In `request_filter`, before connecting: a small catalog-walk body read whole to choose its
    /// row. The request headers are still held, so a refusal names what is wrong.
    Connect,
    /// A large catalog-walk body read whole, before its `FullBody` re-run: the re-run checks the
    /// rest once it has buffered the body again.
    Relay,
    /// In `request_body_filter`, at the end of a body buffered whole: a managed `/{provider}` relay
    /// or a catalog walk. The request headers already went upstream.
    Buffered,
}

/// What the rules read about one body, extracted at its point. A fact is set only where its rule
/// applies and the request is one the rule is for (a catalog walk, a managed key, a Responses path).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Facts<'a> {
    /// More than one root `model` key.
    pub duplicate_model: bool,
    /// An output limit that is not a whole number, on a row with a published cap.
    pub inexact_limit: bool,
    /// What the body asks of the row that its card does not accept: the row and the kind.
    pub refused_input: Option<(&'static str, &'static str)>,
    /// A feature billed outside the usage a row can meter: [`crate::unpriced`]'s message.
    pub unpriced: Option<&'static str>,
    /// A managed `/{provider}` body whose model names no catalog row: the model it names.
    pub catalog_miss: Option<&'a str>,
    /// A managed Responses `background: true` request.
    pub background: bool,
}

/// One body rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Two root `model` keys: the row came from one, and the provider would serve whichever its
    /// parser keeps.
    DuplicateModel,
    /// A limit the clamp cannot hold to the row's maximum.
    OutputLimit,
    /// Input the row does not take (an image, tools).
    RefusedInput,
    /// Something no row could price.
    Unpriced,
    /// A managed key runs only what its billing row can price.
    CatalogMiss,
    /// A generation run after the request ends, which no usage tap can meter.
    Background,
}

impl Rule {
    /// Refusal precedence: the first rule a body breaks is the one it is told.
    pub const ORDER: [Rule; 6] = [
        Rule::DuplicateModel,
        Rule::OutputLimit,
        Rule::RefusedInput,
        Rule::Unpriced,
        Rule::CatalogMiss,
        Rule::Background,
    ];

    /// Whether the rule is checked at `point`. A large catalog body is checked for what its row
    /// accepts before the re-run, and for its own shape and price in the re-run's buffered body;
    /// what a row can price is only read once the body is buffered.
    pub const fn applies_at(self, point: Point) -> bool {
        match self {
            Rule::DuplicateModel | Rule::OutputLimit => {
                matches!(point, Point::Connect | Point::Buffered)
            }
            Rule::RefusedInput => matches!(point, Point::Connect | Point::Relay),
            Rule::Unpriced | Rule::CatalogMiss => matches!(point, Point::Buffered),
            Rule::Background => true,
        }
    }

    /// The refusal `facts` break this rule with, at `point`.
    fn check(self, point: Point, facts: &Facts<'_>) -> Option<Refusal> {
        match self {
            // Once the request headers left, the gateway's 400 there has no message of its own.
            Rule::DuplicateModel => facts.duplicate_model.then_some(match point {
                Point::Buffered => Refusal::DuplicateModelLate,
                Point::Connect | Point::Relay => Refusal::DuplicateModel,
            }),
            Rule::OutputLimit => facts.inexact_limit.then_some(Refusal::OutputLimit),
            Rule::RefusedInput => facts
                .refused_input
                .map(|(model, kind)| Refusal::RefusedInput { model, kind }),
            Rule::Unpriced => facts.unpriced.map(Refusal::Unpriced),
            Rule::CatalogMiss => facts.catalog_miss.map(|name| Refusal::CatalogMiss {
                name: Some(name.into()),
                header: false,
            }),
            Rule::Background => facts.background.then_some(Refusal::Background),
        }
    }
}

/// The refusal a body with `facts` is told at `point`, if any: the first rule in
/// [`Rule::ORDER`] that applies there and that it breaks.
///
/// `facts` by value, not by reference: a borrowed local in `request_filter` stays in its future
/// (boxed once per request) across every later await, where a copied one is dropped at its last use.
pub fn refusal(point: Point, facts: Facts<'_>) -> Option<Refusal> {
    Rule::ORDER
        .into_iter()
        .filter(|rule| rule.applies_at(point))
        .find_map(|rule| rule.check(point, &facts))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Facts breaking exactly `rule`.
    fn breaking(rule: Rule) -> Facts<'static> {
        let mut f = Facts::default();
        match rule {
            Rule::DuplicateModel => f.duplicate_model = true,
            Rule::OutputLimit => f.inexact_limit = true,
            Rule::RefusedInput => f.refused_input = Some(("deepseek-v4-pro", "image input")),
            Rule::Unpriced => f.unpriced = Some(crate::unpriced::CONTAINERS),
            Rule::CatalogMiss => f.catalog_miss = Some("not-a-model"),
            Rule::Background => f.background = true,
        }
        f
    }

    fn merge(a: Facts<'static>, b: Facts<'static>) -> Facts<'static> {
        Facts {
            duplicate_model: a.duplicate_model || b.duplicate_model,
            inexact_limit: a.inexact_limit || b.inexact_limit,
            refused_input: a.refused_input.or(b.refused_input),
            unpriced: a.unpriced.or(b.unpriced),
            catalog_miss: a.catalog_miss.or(b.catalog_miss),
            background: a.background || b.background,
        }
    }

    /// The rules each point checks, in precedence order, written apart from [`Rule::ORDER`] and
    /// [`Rule::applies_at`] so a change to either fails here. At `Connect` the request headers are
    /// still held, so a refusal there can name what is wrong; at `Buffered` they already went
    /// upstream, and the same broken rule answers differently: a duplicate `model` with the generic
    /// 400 the gateway gives an error with no message of its own there
    /// ([`Refusal::DuplicateModelLate`]), as a foreign signed id is answered without naming its
    /// field ([`Refusal::ForeignIdLate`]). That is why those two variants exist.
    const BEFORE: [(Point, &[Rule]); 3] = [
        (
            Point::Connect,
            &[
                Rule::DuplicateModel,
                Rule::OutputLimit,
                Rule::RefusedInput,
                Rule::Background,
            ],
        ),
        (Point::Relay, &[Rule::RefusedInput, Rule::Background]),
        (
            Point::Buffered,
            &[
                Rule::DuplicateModel,
                Rule::OutputLimit,
                Rule::Unpriced,
                Rule::CatalogMiss,
                Rule::Background,
            ],
        ),
    ];

    #[test]
    fn each_point_checks_its_rules_in_precedence_order() {
        for (point, before) in BEFORE {
            let now: Vec<Rule> = Rule::ORDER
                .into_iter()
                .filter(|r| r.applies_at(point))
                .collect();
            assert_eq!(now, before, "{point:?}");
        }
    }

    /// For every two rules a point checks, a body breaking both is told the one that point
    /// checked first; a rule that does not apply there is never told.
    #[test]
    fn two_broken_rules_are_answered_by_the_first_at_every_point() {
        for (point, before) in BEFORE {
            for (i, &first) in before.iter().enumerate() {
                let alone = refusal(point, breaking(first)).expect("a broken rule refuses");
                for &second in &before[i + 1..] {
                    let both = merge(breaking(first), breaking(second));
                    assert_eq!(
                        refusal(point, both).as_ref(),
                        Some(&alone),
                        "{point:?}: {first:?} + {second:?}"
                    );
                }
            }
            for rule in Rule::ORDER {
                if !before.contains(&rule) {
                    assert_eq!(refusal(point, breaking(rule)), None, "{point:?}: {rule:?}");
                }
            }
        }
    }

    /// The refusal each rule answers with, by point.
    #[test]
    fn each_rule_answers_with_its_refusal() {
        let at = |point, rule| refusal(point, breaking(rule));
        assert_eq!(
            at(Point::Connect, Rule::DuplicateModel),
            Some(Refusal::DuplicateModel)
        );
        assert_eq!(
            at(Point::Buffered, Rule::DuplicateModel),
            Some(Refusal::DuplicateModelLate)
        );
        for point in [Point::Connect, Point::Buffered] {
            assert_eq!(at(point, Rule::OutputLimit), Some(Refusal::OutputLimit));
        }
        for point in [Point::Connect, Point::Relay] {
            assert_eq!(
                at(point, Rule::RefusedInput),
                Some(Refusal::RefusedInput {
                    model: "deepseek-v4-pro",
                    kind: "image input"
                })
            );
        }
        assert_eq!(
            at(Point::Buffered, Rule::Unpriced),
            Some(Refusal::Unpriced(crate::unpriced::CONTAINERS))
        );
        assert_eq!(
            at(Point::Buffered, Rule::CatalogMiss),
            Some(Refusal::CatalogMiss {
                name: Some("not-a-model".into()),
                header: false
            })
        );
        for point in [Point::Connect, Point::Relay, Point::Buffered] {
            assert_eq!(at(point, Rule::Background), Some(Refusal::Background));
            assert_eq!(refusal(point, Facts::default()), None);
        }
    }
}
