//! [`PostgresBackend`] — agent memory in a Postgres table. Selected by `--memory postgres://…`.
//!
//! One row per document, keyed by `(project, rel)`. Mutations run in a single transaction with
//! `SELECT … FOR UPDATE` so two agents cannot clobber each other. Directories are implicit
//! prefixes; see [`crate::memory::docs`].
//!
//! The client is `tokio-postgres`, TLS via `tokio-postgres-rustls` (`ring` + native roots) so a
//! `sslmode=require` URL stays on the workspace rustls stack. The table is created idempotently
//! on connect (`CREATE TABLE IF NOT EXISTS`).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_postgres::config::SslMode;
use tokio_postgres::tls::NoTls;
use tokio_postgres::{Client, Config, IsolationLevel};

use super::docs;
use super::{Hit, MemPath, MemoryBackend, MemoryError, View};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_TABLE: &str = "agent_memory";

/// A Postgres-backed memory store. [`Self::connect`] opens a connection, creates the table if
/// needed, and `SELECT 1`s so a down server fails at startup.
pub struct PostgresBackend {
    config: Config,
    use_tls: bool,
    project: String,
    table: String,
    root: &'static str,
    client: Mutex<Client>,
}

impl PostgresBackend {
    /// Connect and ensure the table exists. `project` is the row-namespace (encoded cwd by
    /// default); `table` must be a plain SQL identifier (`[A-Za-z_][A-Za-z0-9_]*`).
    pub async fn connect(
        dsn: &str,
        project: &str,
        table: Option<&str>,
        root: &'static str,
    ) -> Result<Self, String> {
        if project.is_empty() {
            return Err("postgres memory project key must not be empty".into());
        }
        let table = match table {
            Some(t) => validate_ident(t)?.to_string(),
            None => DEFAULT_TABLE.to_string(),
        };
        let mut config: Config = dsn
            .parse()
            .map_err(|e| format!("invalid postgres DSN: {e}"))?;
        if config.get_connect_timeout().is_none() {
            config.connect_timeout(CONNECT_TIMEOUT);
        }
        let use_tls = config.get_ssl_mode() != SslMode::Disable;
        let client = dial(&config, use_tls).await.map_err(|e| e.to_string())?;
        bootstrap(&client, &table)
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self {
            config,
            use_tls,
            project: project.to_string(),
            table,
            root,
            client: Mutex::new(client),
        })
    }

    async fn client(&self) -> Result<tokio::sync::MutexGuard<'_, Client>, MemoryError> {
        let mut guard = self.client.lock().await;
        if guard.is_closed() {
            *guard = dial(&self.config, self.use_tls).await?;
        }
        Ok(guard)
    }

    async fn snapshot(&self) -> Result<BTreeMap<String, String>, MemoryError> {
        let client = self.client().await?;
        load_docs(&client, &self.table, &self.project).await
    }

    async fn transact<T>(
        &self,
        f: impl FnOnce(&mut BTreeMap<String, String>) -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        let mut client = self.client().await?;
        let tx = client
            .build_transaction()
            .isolation_level(IsolationLevel::ReadCommitted)
            .start()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        let select = format!(
            "SELECT rel, body FROM {} WHERE project = $1 FOR UPDATE",
            self.table
        );
        let rows = tx
            .query(&select, &[&self.project])
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        let mut before = BTreeMap::new();
        for row in rows {
            before.insert(row.get::<_, String>(0), row.get::<_, String>(1));
        }
        let mut after = before.clone();
        let result = match f(&mut after) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.rollback().await;
                return Err(e);
            }
        };
        let insert = format!(
            "INSERT INTO {} (project, rel, body) VALUES ($1, $2, $3)",
            self.table
        );
        let update = format!(
            "UPDATE {} SET body = $3 WHERE project = $1 AND rel = $2",
            self.table
        );
        let delete = format!("DELETE FROM {} WHERE project = $1 AND rel = $2", self.table);
        for (rel, body) in &after {
            match before.get(rel) {
                Some(old) if old == body => {}
                Some(_) => {
                    tx.execute(&update, &[&self.project, rel, body])
                        .await
                        .map_err(|e| MemoryError::Backend(e.to_string()))?;
                }
                None => {
                    tx.execute(&insert, &[&self.project, rel, body])
                        .await
                        .map_err(|e| MemoryError::Backend(e.to_string()))?;
                }
            }
        }
        for rel in before.keys() {
            if !after.contains_key(rel) {
                tx.execute(&delete, &[&self.project, rel])
                    .await
                    .map_err(|e| MemoryError::Backend(e.to_string()))?;
            }
        }
        tx.commit()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(result)
    }
}

async fn dial(config: &Config, use_tls: bool) -> Result<Client, MemoryError> {
    if use_tls {
        agent_core::ensure_provider();
        let tls = tokio_postgres_rustls::MakeRustlsConnect::new(Arc::unwrap_or_clone(
            super::tls::client_config().map_err(MemoryError::Backend)?,
        ));
        let (client, connection) = config
            .connect(tls)
            .await
            .map_err(|e| MemoryError::Backend(format!("connecting to postgres: {e}")))?;
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::warn!(error = %e, "postgres memory connection closed");
            }
        });
        Ok(client)
    } else {
        let (client, connection) = config
            .connect(NoTls)
            .await
            .map_err(|e| MemoryError::Backend(format!("connecting to postgres: {e}")))?;
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::warn!(error = %e, "postgres memory connection closed");
            }
        });
        Ok(client)
    }
}

async fn bootstrap(client: &Client, table: &str) -> Result<(), MemoryError> {
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {table} (\
             project TEXT NOT NULL, \
             rel TEXT NOT NULL, \
             body TEXT NOT NULL, \
             PRIMARY KEY (project, rel)\
         )"
    );
    client
        .simple_query(&sql)
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    client
        .simple_query("SELECT 1")
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    Ok(())
}

async fn load_docs(
    client: &Client,
    table: &str,
    project: &str,
) -> Result<BTreeMap<String, String>, MemoryError> {
    let sql = format!("SELECT rel, body FROM {table} WHERE project = $1");
    let rows = client
        .query(&sql, &[&project])
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    let mut out = BTreeMap::new();
    for row in rows {
        out.insert(row.get::<_, String>(0), row.get::<_, String>(1));
    }
    Ok(out)
}

/// A SQL identifier used as a table name — reject anything that would need quoting so the name
/// can be interpolated into DDL/DML without becoming an injection surface.
fn validate_ident(name: &str) -> Result<&str, String> {
    let ok = !name.is_empty()
        && name.len() <= 63
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if ok {
        Ok(name)
    } else {
        Err(format!(
            "postgres memory table `{name}` must be a plain identifier ([A-Za-z_][A-Za-z0-9_]*)"
        ))
    }
}

#[async_trait]
impl MemoryBackend for PostgresBackend {
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
        self.transact(|d| docs::create(d, path, text)).await
    }

    async fn str_replace(&self, path: &MemPath, old: &str, new: &str) -> Result<(), MemoryError> {
        self.transact(|d| docs::str_replace(d, path, old, new))
            .await
    }

    async fn insert(&self, path: &MemPath, line: usize, text: &str) -> Result<(), MemoryError> {
        self.transact(|d| docs::insert(d, path, line, text)).await
    }

    async fn delete(&self, path: &MemPath) -> Result<(), MemoryError> {
        self.transact(|d| docs::delete(d, path)).await
    }

    async fn rename(&self, from: &MemPath, to: &MemPath) -> Result<(), MemoryError> {
        self.transact(|d| docs::rename(d, from, to)).await
    }

    async fn search(&self, query: &str) -> Result<Vec<Hit>, MemoryError> {
        Ok(docs::search(&self.snapshot().await?, query, self.root))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MEMORY_ROOT;

    #[test]
    fn table_ident_rejects_injection() {
        assert!(validate_ident("agent_memory").is_ok());
        assert!(validate_ident("memories").is_ok());
        assert!(validate_ident("agent-memory").is_err());
        assert!(validate_ident("mem;drop").is_err());
        assert!(validate_ident("").is_err());
    }

    #[tokio::test]
    async fn connect_to_a_closed_port_fails_fast() {
        let err = match PostgresBackend::connect(
            "postgres://postgres@127.0.0.1:1/postgres?sslmode=disable",
            "t",
            None,
            MEMORY_ROOT,
        )
        .await
        {
            Ok(_) => panic!("closed port must not connect"),
            Err(e) => e,
        };
        assert!(
            err.contains("connecting to postgres") || err.contains("Connection"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn live_contract_when_postgres_is_reachable() {
        let url = match std::env::var("AI_AGENT_TEST_POSTGRES_URL") {
            Ok(u) if !u.is_empty() => u,
            _ => return,
        };
        let project = format!("contract_{}", std::process::id());
        let b = PostgresBackend::connect(&url, &project, None, MEMORY_ROOT)
            .await
            .expect("AI_AGENT_TEST_POSTGRES_URL must point at a live Postgres");
        let p = |s: &str| MemPath::parse(s).unwrap();
        b.create(&p("/memories/notes.md"), "hello\n").await.unwrap();
        let View::Document(t) = b.view(&p("/memories/notes.md"), None).await.unwrap() else {
            panic!("expected a document");
        };
        assert_eq!(t, "hello\n");
        b.delete(&p("/memories/notes.md")).await.unwrap();
    }
}
