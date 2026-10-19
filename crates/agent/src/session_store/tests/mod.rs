use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_core::compaction::CompactionProvenance;
use agent_core::{ContentBlock, Message, Role, Session};
use serde_json::Value;

use super::*;
use serde_json::json;

fn tmpdir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

mod branches;
mod compaction;
mod custom;
mod format;
mod journal_key;
mod labels;
mod layout;
mod listing_search;
mod lock;
mod persist;
mod repo;
mod switch;
mod title_trash;
