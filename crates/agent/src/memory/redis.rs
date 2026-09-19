//! [`RedisBackend`] — agent memory in a Redis HASH. Selected by `--memory redis://…` / `rediss://…`.
//!
//! One HASH per project (`beyond-ai-memory:<prefix>`), field = logical rel path, value = document
//! body. Memory is a small curated store (the injected index is capped at 25 KB), so `HGETALL` is
//! the whole snapshot; mutations take a short `SET NX` lock (same discipline as the file backend's
//! [`super::file`] lock) and write the diff. Directories are implicit prefixes; see
//! [`crate::memory::docs`].
//!
//! The client is the `redis` crate — multiplexed [`redis::aio::ConnectionManager`], which is the
//! crate's own reconnect path. `rediss://` uses its `tokio-rustls-comp` feature (rustls with
//! `default-features = false`, so the workspace `ring` pin holds).

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use redis::{AsyncCommands, Client};

use super::docs;
use super::{Hit, MemPath, MemoryBackend, MemoryError, View};

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

    async fn snapshot(&self) -> Result<BTreeMap<String, String>, MemoryError> {
        let mut conn = self.conn.clone();
        let map: HashMap<String, String> = conn
            .hgetall(&self.key)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(map.into_iter().collect())
    }

    async fn transact<T, F>(&self, f: F) -> Result<T, MemoryError>
    where
        F: FnOnce(&mut BTreeMap<String, String>) -> Result<T, MemoryError>,
    {
        let token = lock_token()?;
        let mut conn = self.conn.clone();
        acquire_lock(&mut conn, &self.lock_key, &token).await?;
        let outcome = async {
            let before: BTreeMap<String, String> = {
                let map: HashMap<String, String> = conn
                    .hgetall(&self.key)
                    .await
                    .map_err(|e| MemoryError::Backend(e.to_string()))?;
                map.into_iter().collect()
            };
            let mut after = before.clone();
            let result = f(&mut after)?;
            apply_diff(&mut conn, &self.key, &before, &after).await?;
            Ok(result)
        }
        .await;
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

async fn apply_diff(
    conn: &mut ConnectionManager,
    key: &str,
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
) -> Result<(), MemoryError> {
    let mut pipe = redis::pipe();
    let mut writes = false;
    for (field, value) in after {
        if before.get(field) != Some(value) {
            pipe.cmd("HSET").arg(key).arg(field).arg(value).ignore();
            writes = true;
        }
    }
    for field in before.keys() {
        if !after.contains_key(field) {
            pipe.cmd("HDEL").arg(key).arg(field).ignore();
            writes = true;
        }
    }
    if !writes {
        return Ok(());
    }
    pipe.query_async::<()>(conn)
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))
}

#[async_trait]
impl MemoryBackend for RedisBackend {
    async fn index(&self) -> Result<String, MemoryError> {
        Ok(docs::index(&self.snapshot().await?))
    }

    async fn view(
        &self,
        path: &MemPath,
        range: Option<(usize, usize)>,
    ) -> Result<View, MemoryError> {
        docs::view(&self.snapshot().await?, path, range, self.root)
    }

    async fn create(&self, path: &MemPath, text: &str) -> Result<(), MemoryError> {
        let path = path.clone();
        let text = text.to_string();
        self.transact(move |d| docs::create(d, &path, &text)).await
    }

    async fn str_replace(&self, path: &MemPath, old: &str, new: &str) -> Result<(), MemoryError> {
        let path = path.clone();
        let old = old.to_string();
        let new = new.to_string();
        self.transact(move |d| docs::str_replace(d, &path, &old, &new))
            .await
    }

    async fn insert(&self, path: &MemPath, line: usize, text: &str) -> Result<(), MemoryError> {
        let path = path.clone();
        let text = text.to_string();
        self.transact(move |d| docs::insert(d, &path, line, &text))
            .await
    }

    async fn delete(&self, path: &MemPath) -> Result<(), MemoryError> {
        let path = path.clone();
        self.transact(move |d| docs::delete(d, &path)).await
    }

    async fn rename(&self, from: &MemPath, to: &MemPath) -> Result<(), MemoryError> {
        let from = from.clone();
        let to = to.clone();
        self.transact(move |d| docs::rename(d, &from, &to)).await
    }

    async fn search(&self, query: &str) -> Result<Vec<Hit>, MemoryError> {
        Ok(docs::search(&self.snapshot().await?, query, self.root))
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
        b.delete(&p("/memories/notes.md")).await.unwrap();
    }
}
