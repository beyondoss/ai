//! [`RedisBackend`] — agent memory in a Redis HASH. Selected by `--memory redis://…` / `rediss://`.
//!
//! One HASH per project (`beyond-ai-memory:<prefix>`), field = logical rel path, value = document
//! body. Reads are point lookups: `index` is `HGET MEMORY.md`, `view` of a document is one `HGET`,
//! a listing is `HKEYS` + `HSTRLEN` (sizes, not bodies). Mutations take a short `SET NX` lock
//! (same discipline as the file backend's [`super::file`] lock) and touch only the fields they
//! change. Search is the one full-body scan (`HGETALL`). Directories are implicit prefixes; see
//! [`crate::memory::docs`].
//!
//! The client is the `redis` crate — multiplexed [`redis::aio::ConnectionManager`], which is the
//! crate's own reconnect path. `rediss://` uses its `tokio-rustls-comp` feature (rustls with
//! `default-features = false`, so the workspace `ring` pin holds).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use redis::{AsyncCommands, Client};

use super::docs;
use super::{Hit, INDEX_FILE, MemPath, MemoryBackend, MemoryError, View};

const LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const LOCK_RETRY: Duration = Duration::from_millis(20);
const LOCK_TTL_SECS: u64 = 5;

/// A Redis-backed memory store. [`Self::connect`] opens a connection and `PING`s so a down server
/// fails at startup, not on the first `memory` call.
pub struct RedisBackend {
    conn: ConnectionManager,
    key: String,
    lock_key: String,
    root: &'static str,
}

impl RedisBackend {
    /// Connect, authenticate, select the DB, and `PING`. `prefix` namespaces the HASH so two
    /// projects sharing one Redis do not collide.
    pub async fn connect(dsn: &str, prefix: &str, root: &'static str) -> Result<Self, String> {
        if prefix.is_empty() {
            return Err("redis memory prefix must not be empty".into());
        }
        agent_core::ensure_provider();
        let client = Client::open(dsn).map_err(|e| format!("invalid redis DSN: {e}"))?;
        // ConnectionManager's default is 6 exponential retries with no cap — a closed port
        // would sit for minutes. One reconnect attempt and a 5s connect timeout is fail-fast
        // at startup and still enough to ride a brief blip later.
        let cfg = ConnectionManagerConfig::new()
            .set_number_of_retries(1)
            .set_connection_timeout(Duration::from_secs(5))
            .set_max_delay(200);
        let mut conn = tokio::time::timeout(
            Duration::from_secs(5),
            ConnectionManager::new_with_config(client, cfg),
        )
        .await
        .map_err(|_| "timed out connecting to redis".to_string())?
        .map_err(|e| format!("connecting to redis: {e}"))?;
        let _: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(|e| format!("redis PING failed: {e}"))?;
        let key = format!("beyond-ai-memory:{prefix}");
        Ok(Self {
            conn,
            lock_key: format!("{key}:lock"),
            key,
            root,
        })
    }

    async fn get(&self, rel: &str) -> Result<Option<String>, MemoryError> {
        let mut conn = self.conn.clone();
        conn.hget(&self.key, rel)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))
    }

    async fn keys(&self) -> Result<BTreeSet<String>, MemoryError> {
        let mut conn = self.conn.clone();
        let fields: Vec<String> = conn
            .hkeys(&self.key)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(fields.into_iter().collect())
    }

    async fn sizes_under(&self, under: &str) -> Result<BTreeMap<String, u64>, MemoryError> {
        let keys = self.keys().await?;
        let prefix = if under.is_empty() {
            String::new()
        } else {
            format!("{under}/")
        };
        let wanted: Vec<String> = keys
            .into_iter()
            .filter(|k| under.is_empty() || k.starts_with(&prefix))
            .collect();
        if wanted.is_empty() {
            return Ok(BTreeMap::new());
        }
        let mut conn = self.conn.clone();
        let mut pipe = redis::pipe();
        for k in &wanted {
            pipe.cmd("HSTRLEN").arg(&self.key).arg(k);
        }
        let lens: Vec<i64> = pipe
            .query_async(&mut conn)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(wanted
            .into_iter()
            .zip(lens)
            .map(|(k, n)| (k, n.max(0) as u64))
            .collect())
    }

    async fn with_lock<T, F, Fut>(&self, f: F) -> Result<T, MemoryError>
    where
        F: FnOnce(ConnectionManager) -> Fut,
        Fut: std::future::Future<Output = Result<T, MemoryError>>,
    {
        let token = lock_token()?;
        let mut conn = self.conn.clone();
        acquire_lock(&mut conn, &self.lock_key, &token).await?;
        let outcome = f(conn.clone()).await;
        release_lock(&mut conn, &self.lock_key, &token).await;
        outcome
    }
}

fn lock_token() -> Result<String, MemoryError> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|e| MemoryError::Backend(e.to_string()))?;
    Ok(hex::encode(bytes))
}

async fn acquire_lock(
    conn: &mut ConnectionManager,
    lock_key: &str,
    token: &str,
) -> Result<(), MemoryError> {
    let deadline = Instant::now() + LOCK_TIMEOUT;
    loop {
        let ok: bool = redis::cmd("SET")
            .arg(lock_key)
            .arg(token)
            .arg("NX")
            .arg("EX")
            .arg(LOCK_TTL_SECS)
            .query_async(conn)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        if ok {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(MemoryError::Backend(format!(
                "timed out waiting for redis memory lock at {lock_key}"
            )));
        }
        tokio::time::sleep(LOCK_RETRY).await;
    }
}

async fn release_lock(conn: &mut ConnectionManager, lock_key: &str, token: &str) {
    // Compare-and-delete so we never drop another holder's lock after our TTL expired.
    let script = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end";
    let _: Result<i32, _> = redis::cmd("EVAL")
        .arg(script)
        .arg(1)
        .arg(lock_key)
        .arg(token)
        .query_async(conn)
        .await;
}

#[async_trait]
impl MemoryBackend for RedisBackend {
    async fn index(&self) -> Result<String, MemoryError> {
        Ok(self
            .get(INDEX_FILE)
            .await?
            .map(|raw| docs::cap_index(&raw))
            .unwrap_or_default())
    }

    async fn view(
        &self,
        path: &MemPath,
        range: Option<(usize, usize)>,
    ) -> Result<View, MemoryError> {
        if path.is_root() {
            let sizes = self.sizes_under("").await?;
            return Ok(View::Listing(docs::listing_with_sizes(
                sizes.iter().map(|(rel, n)| (rel.as_str(), *n)),
                "",
                self.root,
            )));
        }
        if let Some(text) = self.get(path.rel()).await? {
            return Ok(View::Document(docs::slice_range(&text, range)));
        }
        let sizes = self.sizes_under(path.rel()).await?;
        if sizes.is_empty() {
            return Err(MemoryError::NotFound(path.display()));
        }
        Ok(View::Listing(docs::listing_with_sizes(
            sizes.iter().map(|(rel, n)| (rel.as_str(), *n)),
            path.rel(),
            self.root,
        )))
    }

    async fn create(&self, path: &MemPath, text: &str) -> Result<(), MemoryError> {
        if path.is_root() {
            return Err(MemoryError::InvalidPath(
                "cannot create the memory root itself".to_string(),
            ));
        }
        let path = path.clone();
        let text = text.to_string();
        let key = self.key.clone();
        self.with_lock(move |mut conn| async move {
            let fields: Vec<String> = conn
                .hkeys(&key)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            let keys: BTreeSet<String> = fields.into_iter().collect();
            docs::create_conflict_keys(&keys, &path)?;
            conn.hset::<_, _, _, ()>(&key, path.rel(), text)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            Ok(())
        })
        .await
    }

    async fn str_replace(&self, path: &MemPath, old: &str, new: &str) -> Result<(), MemoryError> {
        let path = path.clone();
        let old = old.to_string();
        let new = new.to_string();
        let key = self.key.clone();
        self.with_lock(move |mut conn| async move {
            let text: Option<String> = conn
                .hget(&key, path.rel())
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            let Some(text) = text else {
                return Err(not_a_document(&mut conn, &key, &path).await);
            };
            let replaced = docs::str_replace_once(&text, &old, &new, &path.display())?;
            conn.hset::<_, _, _, ()>(&key, path.rel(), replaced)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            Ok(())
        })
        .await
    }

    async fn insert(&self, path: &MemPath, line: usize, text: &str) -> Result<(), MemoryError> {
        let path = path.clone();
        let text = text.to_string();
        let key = self.key.clone();
        self.with_lock(move |mut conn| async move {
            let existing: Option<String> = conn
                .hget(&key, path.rel())
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            let Some(existing) = existing else {
                return Err(not_a_document(&mut conn, &key, &path).await);
            };
            conn.hset::<_, _, _, ()>(
                &key,
                path.rel(),
                docs::insert_at_line(&existing, line, &text),
            )
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
            Ok(())
        })
        .await
    }

    async fn delete(&self, path: &MemPath) -> Result<(), MemoryError> {
        if path.is_root() {
            return Err(MemoryError::InvalidPath(
                "cannot delete the memory root".to_string(),
            ));
        }
        let path = path.clone();
        let key = self.key.clone();
        self.with_lock(move |mut conn| async move {
            let removed: i32 = redis::cmd("HDEL")
                .arg(&key)
                .arg(path.rel())
                .query_async(&mut conn)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            if removed > 0 {
                return Ok(());
            }
            let fields: Vec<String> = conn
                .hkeys(&key)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            if docs::is_dir_keys(fields.iter(), path.rel()) {
                return Err(MemoryError::InvalidPath(format!(
                    "{} is a non-empty directory; delete its contents first",
                    path.display()
                )));
            }
            Err(MemoryError::NotFound(path.display()))
        })
        .await
    }

    async fn rename(&self, from: &MemPath, to: &MemPath) -> Result<(), MemoryError> {
        if from.is_root() || to.is_root() {
            return Err(MemoryError::InvalidPath(
                "cannot rename the memory root".to_string(),
            ));
        }
        if from.rel() == to.rel() {
            return Ok(());
        }
        let from = from.clone();
        let to = to.clone();
        let key = self.key.clone();
        self.with_lock(move |mut conn| async move {
            let fields: Vec<String> = conn
                .hkeys(&key)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            let keys: BTreeSet<String> = fields.into_iter().collect();
            let from_is_file = keys.contains(from.rel());
            let from_is_dir = docs::is_dir_keys(keys.iter(), from.rel());
            if !from_is_file && !from_is_dir {
                return Err(MemoryError::NotFound(from.display()));
            }
            if keys.contains(to.rel()) || docs::is_dir_keys(keys.iter(), to.rel()) {
                return Err(MemoryError::AlreadyExists(to.display()));
            }
            if from_is_dir
                && (to.rel() == from.rel() || to.rel().starts_with(&format!("{}/", from.rel())))
            {
                return Err(MemoryError::InvalidPath(format!(
                    "cannot rename {} into itself",
                    from.display()
                )));
            }
            docs::ancestor_is_file_keys(&keys, to.rel(), &to)?;

            if from_is_file {
                let body: Option<String> = conn
                    .hget(&key, from.rel())
                    .await
                    .map_err(|e| MemoryError::Backend(e.to_string()))?;
                let Some(body) = body else {
                    return Err(MemoryError::NotFound(from.display()));
                };
                let mut pipe = redis::pipe();
                pipe.cmd("HSET").arg(&key).arg(to.rel()).arg(body).ignore();
                pipe.cmd("HDEL").arg(&key).arg(from.rel()).ignore();
                pipe.query_async::<()>(&mut conn)
                    .await
                    .map_err(|e| MemoryError::Backend(e.to_string()))?;
                return Ok(());
            }

            let prefix = format!("{}/", from.rel());
            let moving: Vec<String> = keys
                .iter()
                .filter(|k| k.starts_with(&prefix))
                .cloned()
                .collect();
            if moving.is_empty() {
                return Err(MemoryError::NotFound(from.display()));
            }
            let bodies: Vec<Option<String>> = redis::cmd("HMGET")
                .arg(&key)
                .arg(&moving)
                .query_async(&mut conn)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            let mut pipe = redis::pipe();
            for (rel, body) in moving.iter().zip(bodies) {
                let Some(body) = body else {
                    continue;
                };
                let tail = &rel[prefix.len()..];
                pipe.cmd("HSET")
                    .arg(&key)
                    .arg(format!("{}/{tail}", to.rel()))
                    .arg(body)
                    .ignore();
                pipe.cmd("HDEL").arg(&key).arg(rel).ignore();
            }
            pipe.query_async::<()>(&mut conn)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            Ok(())
        })
        .await
    }

    async fn search(&self, query: &str) -> Result<Vec<Hit>, MemoryError> {
        let needle = query.to_lowercase();
        if needle.is_empty() {
            return Ok(Vec::new());
        }
        let mut conn = self.conn.clone();
        let map: HashMap<String, String> = conn
            .hgetall(&self.key)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(docs::search(&map.into_iter().collect(), query, self.root))
    }
}

/// A missing field is either an implicit directory or a real miss — `HKEYS` only, no bodies.
async fn not_a_document(conn: &mut ConnectionManager, key: &str, path: &MemPath) -> MemoryError {
    let fields: Result<Vec<String>, _> = conn.hkeys(key).await;
    match fields {
        Ok(fields) if docs::is_dir_keys(fields.iter(), path.rel()) => {
            MemoryError::InvalidPath(format!("{} is a directory, not a document", path.display()))
        }
        Ok(_) => MemoryError::NotFound(path.display()),
        Err(e) => MemoryError::Backend(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MEMORY_ROOT;

    #[tokio::test]
    async fn connect_to_a_closed_port_fails_fast() {
        let err = match RedisBackend::connect("redis://127.0.0.1:1", "t", MEMORY_ROOT).await {
            Ok(_) => panic!("closed port must not connect"),
            Err(e) => e,
        };
        assert!(
            err.contains("connecting to redis")
                || err.contains("PING")
                || err.contains("Connection"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn live_contract_when_redis_is_reachable() {
        let url = match std::env::var("AI_AGENT_TEST_REDIS_URL") {
            Ok(u) if !u.is_empty() => u,
            _ => return,
        };
        let prefix = format!("contract-{}", std::process::id());
        let b = RedisBackend::connect(&url, &prefix, MEMORY_ROOT)
            .await
            .expect("AI_AGENT_TEST_REDIS_URL must point at a live Redis");
        let p = |s: &str| MemPath::parse(s).unwrap();
        b.create(&p("/memories/notes.md"), "hello\n").await.unwrap();
        let View::Document(t) = b.view(&p("/memories/notes.md"), None).await.unwrap() else {
            panic!("expected a document");
        };
        assert_eq!(t, "hello\n");
        assert_eq!(b.index().await.unwrap(), "");
        b.create(&p("/memories/MEMORY.md"), "# notes.md\n")
            .await
            .unwrap();
        assert!(b.index().await.unwrap().contains("notes.md"));
        b.delete(&p("/memories/notes.md")).await.unwrap();
        b.delete(&p("/memories/MEMORY.md")).await.unwrap();
    }
}
