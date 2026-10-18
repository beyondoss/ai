//! `request_filter`'s phases, in order, and the types each hands the next.

use super::*;

impl AiProxy {
    /// This tenant's concurrency slot. `Ok(None)` when no cap is configured; `Err` when the tenant
    /// is at its cap.
    pub(super) fn take_slot(
        &self,
        id: &Identity,
    ) -> std::result::Result<Option<TenantSlot<'static>>, ()> {
        match self.state.tenant_slots.as_ref() {
            None => Ok(None),
            Some(slots) => slots.take(id).map(Some).ok_or(()),
        }
    }

    /// `signed_id`: a large catalog body's ids, checked before its first attempt connects (its
    /// `FullBody` re-runs see the body only as it streams in). `Some` when the request is refused.
    pub(super) fn foreign_ids(
        &self,
        path: &str,
        route: &'static route::ModelRoute,
        body: &[u8],
        tenant_id: u64,
    ) -> Option<Refusal> {
        if !signed_id::catalog_applies(path, route) {
            return None;
        }
        match self.state.id_signer.as_ref() {
            None => Some(Refusal::IdSigningUnset),
            Some(signer) => signer
                .unsign_request(tenant_id, body, true)
                .err()
                .map(Refusal::ForeignId),
        }
    }
}

/// The `x-beyond-model` header, viewed as a catalog lookup. Present-but-unknown is distinct from
/// absent: the header wins, so an unknown header is a 404 rather than a fall-through to the body.
#[derive(Clone, Copy)]
pub(super) enum CatalogHeader {
    Absent,
    Known(&'static route::ModelRoute),
    Unknown,
}

pub(super) fn catalog_from_header(session: &Session) -> CatalogHeader {
    match session
        .req_header()
        .headers
        .get(route::MODEL_HEADER)
        .and_then(|v| v.to_str().ok())
    {
        None => CatalogHeader::Absent,
        Some(name) => match route::model_route(name) {
            Some(r) => CatalogHeader::Known(r),
            None => CatalogHeader::Unknown,
        },
    }
}

/// What the first path segment resolved to, before the key says whether the request may walk the
/// catalog.
///
/// Holds only routes that go somewhere: a path that resolves to nothing is a [`RouteMiss`], so no
/// later step has to wonder whether it is looking at one.
pub(super) enum Route {
    /// `/{provider}/…`. The provider is named by the request. Not a catalog walk — the escape hatch.
    Provider {
        provider: &'static Arc<Provider>,
        /// The inbound path with the `/{provider}` segment stripped, query kept.
        forward_path: String,
    },
    /// Bare `/v1` default. BYO keeps the dialect-picked provider; managed becomes a catalog walk.
    BareDefault { provider: &'static Arc<Provider> },
    /// `/auto/…`. `header` is the catalog row if `x-beyond-model` resolved; `None` means peek the
    /// body's root `model` after identity is known. An *unknown* header is [`RouteMiss::UnknownModel`],
    /// not this — the header wins, and a catalog miss is the allowlist.
    Auto {
        header: Option<&'static route::ModelRoute>,
    },
}

/// A path that resolves to nothing the gateway serves. Each shape gets its own rejection (404
/// unknown provider, 404 unknown model, 400 ambiguous credential), which is why it is a sum type and
/// not an `Option`.
pub(super) enum RouteMiss {
    /// A routing header (on `/auto`) naming a model we do not serve. Collapsed with a missing body
    /// model into one 404: a value we cannot match is a value we do not serve.
    UnknownModel,
    /// The first segment matches no provider, is not the bare default, and is not `/auto`.
    UnknownProvider,
    /// Bare `/v1` with BYO keys for two different providers (D82): a 400, never a guess.
    AmbiguousCredential,
}

/// What `request_filter` decided a request is, once its key is known: carried whole through
/// admission, so the route is never re-derived from which of several locals happen to be set.
pub(super) enum Plan {
    /// `/{provider}/…`, or a BYO bare `/v1` on its dialect-default provider.
    Provider {
        provider: &'static Arc<Provider>,
        /// [`ForwardPath::Stripped`] on `/{provider}/…`, [`ForwardPath::Inbound`] on the bare
        /// default.
        forward: ForwardPath,
    },
    /// A managed catalog walk (`/auto`, or a managed bare `/v1`) over a resolved row. Only a
    /// verified key gets one, so it carries the identity it was admitted on.
    Catalog(Walk),
}

/// A request on its way to a [`Plan`] once its key is known: ready, or a managed catalog walk that
/// still has to read its row from `x-beyond-model` or the body's root `model`.
pub(super) enum Admission {
    Ready(Plan),
    NeedsRow(Identity),
}

/// A catalog walk as `request_filter` admits it.
pub(super) struct Walk {
    pub(super) id: Identity,
    pub(super) row: &'static route::ModelRoute,
    /// The body, read in full, when it outgrew pingora's retry buffer: the request is re-run as a
    /// [`FullBody`] subrequest carrying it rather than proxied here.
    pub(super) relay: Option<Vec<u8>>,
}

impl Plan {
    /// The catalog row this request walks; `None` on a provider route.
    pub(super) fn row(&self) -> Option<&'static route::ModelRoute> {
        match self {
            Plan::Catalog(w) => Some(w.row),
            Plan::Provider { .. } => None,
        }
    }

    /// The forwarded `/{provider}` path (query included), when the client named the provider.
    pub(super) fn stripped(&self) -> Option<&str> {
        match self {
            Plan::Provider {
                forward: ForwardPath::Stripped(p),
                ..
            } => Some(p),
            _ => None,
        }
    }
}

/// What path an attempt sends upstream.
#[derive(Debug)]
pub(super) enum ForwardPath {
    /// The inbound path and query, as the client sent them (the bare `/v1` default). The common
    /// route builds nothing, so it allocates nothing. On a managed request its `key` params are the
    /// virtual key, stripped as the head goes out.
    Inbound,
    /// The client's path with the `/{provider}` segment stripped, its query kept (minus a managed
    /// request's `key` params). Forwarded **verbatim** — the gateway does no per-provider path
    /// rewriting. Built once, so held as a `Box<str>`: beside `Catalog`'s `String` that keeps the
    /// enum at a `String`'s size.
    Stripped(Box<str>),
    /// The serving candidate's own catalog path (plus a sub-resource's suffix), set as each
    /// attempt dials it; rewritten in place, so a failover allocates nothing after the first.
    /// Taken wholesale from the catalog rather than composed from a mount and the client's suffix:
    /// providers disagree on where an endpoint lives and the disagreement is not a prefix.
    Catalog(String),
}

impl ForwardPath {
    /// The path (and query) to set on the upstream request; `None` keeps the inbound one.
    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            ForwardPath::Inbound => None,
            ForwardPath::Stripped(p) => Some(p),
            ForwardPath::Catalog(p) => Some(p),
        }
    }
}
