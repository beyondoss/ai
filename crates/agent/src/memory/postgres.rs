//! [`PostgresBackend`] — agent memory in a Postgres table. Selected by `--memory postgres://…`.
//!
//! One row per document, keyed by `(project, rel)`. Reads are point lookups: `index` is
//! `MEMORY.md` only, `view` of a document is one `SELECT`, a listing is `rel` + `length(body)`.
//! Single-document edits lock that row (`FOR UPDATE`). Prefix `create`/`delete`/`rename` take a
//! per-project advisory lock and consult the key set — they do not pull every body. Search
//! filters in SQL (`position(lower(needle) in lower(body))`) so non-matching documents stay put.
//! Directories are implicit prefixes; see [`crate::memory::docs`].
//!
//! The client is `tokio-postgres`, TLS via `tokio-postgres-rustls` (`ring` + native roots) so a
//! `sslmode=require` URL stays on the workspace rustls stack. The table is the agent's: created
//! idempotently on connect, then probed so an existing table with the wrong columns fails fast.
//! Payload stays `TEXT` (markdown; `str_replace` can span lines). Paths use `COLLATE "C"` so
//! order matches the in-process map, plus `CHECK`s that encode [`MemPath`], and `updated_at`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_postgres::config::SslMode;
use tokio_postgres::tls::NoTls;
use tokio_postgres::{Client, Config, Transaction};

use super::docs;
use super::{Hit, INDEX_FILE, MemPath, MemoryBackend, MemoryError, View};

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

    async fn get_body(&self, rel: &str) -> Result<Option<String>, MemoryError> {
        let client = self.client().await?;
        get_body(&client, &self.table, &self.project, rel).await
    }

    async fn sizes_under(&self, under: &str) -> Result<BTreeMap<String, u64>, MemoryError> {
        let client = self.client().await?;
        sizes_under(&client, &self.table, &self.project, under).await
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

/// DDL for a new table. `TEXT` for the document (not `BYTEA`/`JSONB`/`TEXT[]` — see the module
/// doc). `COLLATE "C"` so `rel` order is locale-independent. `CHECK`s match [`MemPath`].
fn create_table_sql(table: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {table} (\
             project TEXT NOT NULL COLLATE \"C\", \
             rel TEXT NOT NULL COLLATE \"C\", \
             body TEXT NOT NULL, \
             updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), \
             PRIMARY KEY (project, rel), \
             CONSTRAINT {table}_rel_nonempty CHECK (rel <> ''), \
             CONSTRAINT {table}_rel_components CHECK (\
                 rel !~ '(^|/)(\\.|\\.\\.)(/|$)' AND rel NOT LIKE '%//%'\
             ), \
             CONSTRAINT {table}_rel_nul CHECK (position(chr(0) in rel) = 0)\
         )"
    )
}

async fn bootstrap(client: &Client, table: &str) -> Result<(), MemoryError> {
    client
        .simple_query(&create_table_sql(table))
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    // A table that already existed under this name may predate `updated_at`. Add it if missing,
    // then probe the contract so a same-name table with the wrong columns fails at open.
    let add_updated = format!(
        "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS \
         updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
    );
    client
        .simple_query(&add_updated)
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    let probe = format!("SELECT project, rel, body, updated_at FROM {table} WHERE false");
    client.query(&probe, &[]).await.map_err(|e| {
        MemoryError::Backend(format!(
            "postgres table `{table}` does not match the memory schema \
                 (need project, rel, body, updated_at): {e}"
        ))
    })?;
    Ok(())
}

async fn get_body(
    client: &Client,
    table: &str,
    project: &str,
    rel: &str,
) -> Result<Option<String>, MemoryError> {
    let sql = format!("SELECT body FROM {table} WHERE project = $1 AND rel = $2");
    let row = client
        .query_opt(&sql, &[&project, &rel])
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    Ok(row.map(|r| r.get(0)))
}

async fn sizes_under(
    client: &Client,
    table: &str,
    project: &str,
    under: &str,
) -> Result<BTreeMap<String, u64>, MemoryError> {
    let rows = if under.is_empty() {
        let sql = format!("SELECT rel, length(body)::bigint FROM {table} WHERE project = $1");
        client
            .query(&sql, &[&project])
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?
    } else {
        let prefix = format!("{under}/");
        let sql = format!(
            "SELECT rel, length(body)::bigint FROM {table} \
             WHERE project = $1 AND strpos(rel, $2) = 1"
        );
        client
            .query(&sql, &[&project, &prefix])
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?
    };
    let mut out = BTreeMap::new();
    for row in rows {
        let rel: String = row.get(0);
        let n: i64 = row.get(1);
        out.insert(rel, n.max(0) as u64);
    }
    Ok(out)
}

async fn lock_project(tx: &Transaction<'_>, project: &str) -> Result<(), MemoryError> {
    tx.execute(
        "SELECT pg_advisory_xact_lock(hashtext('beyond-ai-memory'), hashtext($1))",
        &[&project],
    )
    .await
    .map_err(|e| MemoryError::Backend(e.to_string()))?;
    Ok(())
}

async fn list_keys(
    tx: &Transaction<'_>,
    table: &str,
    project: &str,
) -> Result<BTreeSet<String>, MemoryError> {
    let sql = format!("SELECT rel FROM {table} WHERE project = $1");
    let rows = tx
        .query(&sql, &[&project])
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    Ok(rows.into_iter().map(|r| r.get(0)).collect())
}

async fn has_children(
    tx: &Transaction<'_>,
    table: &str,
    project: &str,
    rel: &str,
) -> Result<bool, MemoryError> {
    let prefix = format!("{rel}/");
    let sql = format!("SELECT 1 FROM {table} WHERE project = $1 AND strpos(rel, $2) = 1 LIMIT 1");
    let row = tx
        .query_opt(&sql, &[&project, &prefix])
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    Ok(row.is_some())
}

async fn not_a_document(
    tx: &Transaction<'_>,
    table: &str,
    project: &str,
    path: &MemPath,
) -> Result<MemoryError, MemoryError> {
    if has_children(tx, table, project, path.rel()).await? {
        Ok(MemoryError::InvalidPath(format!(
            "{} is a directory, not a document",
            path.display()
        )))
    } else {
        Ok(MemoryError::NotFound(path.display()))
    }
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
        Ok(self
            .get_body(INDEX_FILE)
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
        if let Some(text) = self.get_body(path.rel()).await? {
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
        let mut client = self.client().await?;
        let tx = client
            .transaction()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        lock_project(&tx, &self.project).await?;
        let keys = list_keys(&tx, &self.table, &self.project).await?;
        if let Err(e) = docs::create_conflict_keys(&keys, path) {
            let _ = tx.rollback().await;
            return Err(e);
        }
        let insert = format!(
            "INSERT INTO {} (project, rel, body) VALUES ($1, $2, $3)",
            self.table
        );
        if let Err(e) = tx
            .execute(&insert, &[&self.project, &path.rel(), &text])
            .await
        {
            let _ = tx.rollback().await;
            return Err(MemoryError::Backend(e.to_string()));
        }
        tx.commit()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn str_replace(&self, path: &MemPath, old: &str, new: &str) -> Result<(), MemoryError> {
        let mut client = self.client().await?;
        let tx = client
            .transaction()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        let select = format!(
            "SELECT body FROM {} WHERE project = $1 AND rel = $2 FOR UPDATE",
            self.table
        );
        let row = match tx.query_opt(&select, &[&self.project, &path.rel()]).await {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.rollback().await;
                return Err(MemoryError::Backend(e.to_string()));
            }
        };
        let Some(row) = row else {
            let err = not_a_document(&tx, &self.table, &self.project, path).await;
            let _ = tx.rollback().await;
            return Err(err?);
        };
        let text: String = row.get(0);
        let replaced = match docs::str_replace_once(&text, old, new, &path.display()) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.rollback().await;
                return Err(e);
            }
        };
        let update = format!(
            "UPDATE {} SET body = $3, updated_at = now() WHERE project = $1 AND rel = $2",
            self.table
        );
        if let Err(e) = tx
            .execute(&update, &[&self.project, &path.rel(), &replaced])
            .await
        {
            let _ = tx.rollback().await;
            return Err(MemoryError::Backend(e.to_string()));
        }
        tx.commit()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn insert(&self, path: &MemPath, line: usize, text: &str) -> Result<(), MemoryError> {
        let mut client = self.client().await?;
        let tx = client
            .transaction()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        let select = format!(
            "SELECT body FROM {} WHERE project = $1 AND rel = $2 FOR UPDATE",
            self.table
        );
        let row = match tx.query_opt(&select, &[&self.project, &path.rel()]).await {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.rollback().await;
                return Err(MemoryError::Backend(e.to_string()));
            }
        };
        let Some(row) = row else {
            let err = not_a_document(&tx, &self.table, &self.project, path).await;
            let _ = tx.rollback().await;
            return Err(err?);
        };
        let existing: String = row.get(0);
        let update = format!(
            "UPDATE {} SET body = $3, updated_at = now() WHERE project = $1 AND rel = $2",
            self.table
        );
        if let Err(e) = tx
            .execute(
                &update,
                &[
                    &self.project,
                    &path.rel(),
                    &docs::insert_at_line(&existing, line, text),
                ],
            )
            .await
        {
            let _ = tx.rollback().await;
            return Err(MemoryError::Backend(e.to_string()));
        }
        tx.commit()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, path: &MemPath) -> Result<(), MemoryError> {
        if path.is_root() {
            return Err(MemoryError::InvalidPath(
                "cannot delete the memory root".to_string(),
            ));
        }
        let mut client = self.client().await?;
        let tx = client
            .transaction()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        lock_project(&tx, &self.project).await?;
        let delete = format!("DELETE FROM {} WHERE project = $1 AND rel = $2", self.table);
        let n = match tx.execute(&delete, &[&self.project, &path.rel()]).await {
            Ok(n) => n,
            Err(e) => {
                let _ = tx.rollback().await;
                return Err(MemoryError::Backend(e.to_string()));
            }
        };
        if n > 0 {
            tx.commit()
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
            return Ok(());
        }
        let children = match has_children(&tx, &self.table, &self.project, path.rel()).await {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.rollback().await;
                return Err(e);
            }
        };
        let _ = tx.rollback().await;
        if children {
            Err(MemoryError::InvalidPath(format!(
                "{} is a non-empty directory; delete its contents first",
                path.display()
            )))
        } else {
            Err(MemoryError::NotFound(path.display()))
        }
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
        let mut client = self.client().await?;
        let tx = client
            .transaction()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        lock_project(&tx, &self.project).await?;
        let keys = match list_keys(&tx, &self.table, &self.project).await {
            Ok(k) => k,
            Err(e) => {
                let _ = tx.rollback().await;
                return Err(e);
            }
        };
        let from_is_file = keys.contains(from.rel());
        let from_is_dir = docs::is_dir_keys(keys.iter(), from.rel());
        if !from_is_file && !from_is_dir {
            let _ = tx.rollback().await;
            return Err(MemoryError::NotFound(from.display()));
        }
        if keys.contains(to.rel()) || docs::is_dir_keys(keys.iter(), to.rel()) {
            let _ = tx.rollback().await;
            return Err(MemoryError::AlreadyExists(to.display()));
        }
        if from_is_dir
            && (to.rel() == from.rel() || to.rel().starts_with(&format!("{}/", from.rel())))
        {
            let _ = tx.rollback().await;
            return Err(MemoryError::InvalidPath(format!(
                "cannot rename {} into itself",
                from.display()
            )));
        }
        if let Err(e) = docs::ancestor_is_file_keys(&keys, to.rel(), to) {
            let _ = tx.rollback().await;
            return Err(e);
        }

        if from_is_file {
            let update = format!(
                "UPDATE {} SET rel = $3, updated_at = now() WHERE project = $1 AND rel = $2",
                self.table
            );
            if let Err(e) = tx
                .execute(&update, &[&self.project, &from.rel(), &to.rel()])
                .await
            {
                let _ = tx.rollback().await;
                return Err(MemoryError::Backend(e.to_string()));
            }
            tx.commit()
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
        let update = format!(
            "UPDATE {} SET rel = $3, updated_at = now() WHERE project = $1 AND rel = $2",
            self.table
        );
        for rel in moving {
            let tail = &rel[prefix.len()..];
            let dest = format!("{}/{tail}", to.rel());
            if let Err(e) = tx.execute(&update, &[&self.project, &rel, &dest]).await {
                let _ = tx.rollback().await;
                return Err(MemoryError::Backend(e.to_string()));
            }
        }
        tx.commit()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn search(&self, query: &str) -> Result<Vec<Hit>, MemoryError> {
        let needle = query.to_lowercase();
        if needle.is_empty() {
            return Ok(Vec::new());
        }
        let client = self.client().await?;
        let sql = format!(
            "SELECT rel, body FROM {} \
             WHERE project = $1 AND position(lower($2) in lower(body)) > 0 \
             ORDER BY rel",
            self.table
        );
        let rows = client
            .query(&sql, &[&self.project, &needle])
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        let mut hits = Vec::new();
        for row in rows {
            let rel: String = row.get(0);
            let body: String = row.get(1);
            let path = format!("{}/{rel}", self.root);
            hits.extend(docs::search_in(&path, &body, &needle));
        }
        Ok(hits)
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

    #[test]
    fn create_table_sql_is_the_owned_row_type() {
        let sql = create_table_sql("agent_memory");
        assert!(sql.contains("COLLATE \"C\""));
        assert!(sql.contains("updated_at TIMESTAMPTZ"));
        assert!(sql.contains("PRIMARY KEY (project, rel)"));
        assert!(sql.contains("agent_memory_rel_nonempty"));
        assert!(sql.contains("agent_memory_rel_components"));
        assert!(sql.contains("chr(0)"));
        assert!(!sql.contains("BYTEA"));
        assert!(!sql.contains("JSONB"));
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
        assert_eq!(b.index().await.unwrap(), "");
        b.create(&p("/memories/MEMORY.md"), "# notes.md\n")
            .await
            .unwrap();
        assert!(b.index().await.unwrap().contains("notes.md"));
        b.delete(&p("/memories/notes.md")).await.unwrap();
        b.delete(&p("/memories/MEMORY.md")).await.unwrap();
    }
}
