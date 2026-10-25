# Installing execd

`execd` is the in-sandbox exec endpoint the agent talks to (`--exec-url`). It ships as a single fully static musl binary (linux amd64 and arm64), so it runs in any sandbox image: Alpine, Debian, busybox, E2B templates.

Start it with `execd --listen 0.0.0.0:8080 --bearer-file /path/to/token`.

## Install

Three ways, all producing the same binary.

### 1. `COPY --from` (Dockerfile)

```dockerfile
COPY --from=ghcr.io/beyondoss/execd:latest /execd /usr/local/bin/execd
```

Pin a release with `:<tag>` (for example `:v1.2.3`) or a commit with `:<sha>`. The image is `FROM scratch` with the binary at `/execd`; it has no shell, so it cannot run commands itself. It exists to be copied from.

### 2. Installer script

```sh
curl -fsSL https://github.com/beyondoss/ai/releases/latest/download/install-execd.sh | sh
```

Works with `wget` instead of `curl` (busybox included). Environment:

| Variable            | Default                  | Meaning                                                                          |
| ------------------- | ------------------------ | -------------------------------------------------------------------------------- |
| `EXECD_VERSION`     | latest release           | Release tag to install                                                           |
| `EXECD_INSTALL_DIR` | `/usr/local/bin`         | Destination directory                                                            |
| `EXECD_BASE_URL`    | the tag's GitHub release | Directory holding the tarball and `.sha256` (a mirror); requires `EXECD_VERSION` |

The script checks the SHA-256, installs through a temp file and a rename, and refuses on any mismatch without touching an existing install. Running it again is safe.

### 3. Tarball

```sh
tag=v1.2.3 arch=amd64 # or arm64
base=https://github.com/beyondoss/ai/releases/download/$tag
curl -fsSLO "$base/execd-$tag-linux-$arch.tar.gz"
curl -fsSLO "$base/execd-$tag-linux-$arch.tar.gz.sha256"
sha256sum -c "execd-$tag-linux-$arch.tar.gz.sha256"
tar -xzf "execd-$tag-linux-$arch.tar.gz" execd
install -m 755 execd /usr/local/bin/execd
```

## Verify

```sh
file /usr/local/bin/execd # ... statically linked
ldd /usr/local/bin/execd  # not a dynamic executable (glibc); musl's ldd may print a loader line
```

Then start it and probe `/livez`:

```sh
execd --listen 127.0.0.1:8080 --bearer-file /path/to/token &
curl -fs http://127.0.0.1:8080/livez
```

## Tests

`test-install.sh` runs the installer in `alpine` (busybox `wget`) and `debian:bookworm-slim` (`curl`) against a local fake release served with `python3 -m http.server`: success, rerun, checksum mismatch (refused, no file left), mismatch over an existing install, unsupported architecture. It then builds `Dockerfile.execd`, copies it into a throwaway `FROM alpine` image and curls `/livez` from the host. It needs docker, python3 and a static musl build:

```sh
RUSTFLAGS="-C relocation-model=static" \
  cargo build --release --locked -p beyond-ai-execd --target x86_64-unknown-linux-musl
deploy/execd/test-install.sh target/x86_64-unknown-linux-musl/release/execd
```

## Release

A `v*` tag runs `.github/workflows/release-execd.yml` (tarballs, checksums, `install-execd.sh`) and `.github/workflows/publish.yml` (the `ghcr.io/beyondoss/execd` image).
