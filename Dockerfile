# syntax=docker/dockerfile:1.7-labs
#
# Multi-stage build for the Beyond AI gateway (`gateway`).
#
# Built from the workspace root. The repo is a Cargo workspace; this image builds
# only the `gateway` member (`crates/gateway`). All deps come from crates.io:
#
#     docker build -t beyond-ai .
#
# We use cargo-chef to cache the (heavy: pingora + tokio) dependency build in a
# layer that's keyed only on the dependency graph, so source-only changes skip
# the slow dep compile.

# The toolchain CI tests with (mise.toml's `rust` pin; publish.yml passes it in).
# It must still lead the MSRV (1.85): cargo-chef's own build needs a newer rustc.
ARG RUST_VERSION=1.92

# ---------------------------------------------------------------------------
# Stage 1: chef — base with cargo-chef installed.
# ---------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-bookworm AS chef
# cmake builds the bundled zlib-ng (libz-ng-sys, pulled in by pingora); the base
# rust image ships a C/C++ toolchain but not cmake.
RUN apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-chef --locked --version ^0.1
WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: planner — compute a dependency-only recipe.
# ---------------------------------------------------------------------------
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ---------------------------------------------------------------------------
# Stage 3: builder — cook dependencies from the recipe, then build the binary.
# ---------------------------------------------------------------------------
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Cook only dependencies — cached until the recipe (the dependency graph)
# changes. Cache mounts keep the cargo registry + git db warm across builds.
# rustls uses ring, which builds C/asm with the toolchain already in the image;
# no extra apt packages are needed (pure-rustls, no OpenSSL/protobuf).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo chef cook --release --locked --recipe-path recipe.json

# Now copy the full source and build just the gateway member of the workspace.
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --locked -p beyond-gateway --bin gateway \
    && cp /app/target/release/gateway /usr/local/bin/gateway

# ---------------------------------------------------------------------------
# Stage 4: runtime — minimal Debian slim with just the binary and the CA
# certificates needed to verify outbound TLS to the LLM providers.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin beyond
USER beyond

COPY --from=builder /usr/local/bin/gateway /usr/local/bin/gateway

# 8080: data-plane listener (client requests). 9090: admin — /metrics
# (Prometheus), /livez, /readyz. Both default to 0.0.0.0; override via the
# mounted config or the AI_LISTEN / AI_METRICS_LISTEN env vars.
EXPOSE 8080/tcp 9090/tcp

ENTRYPOINT ["/usr/local/bin/gateway"]
