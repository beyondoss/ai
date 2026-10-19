//! `settings`: update the stored defaults, then print every field's current value.

use crate::{Command, parse_reasoning_effort};

/// `settings`: apply the requested updates, then print.
pub(crate) fn settings(cmd: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::Settings {
        model,
        clear_model,
        gateway_url,
        clear_gateway_url,
        session_dir,
        clear_session_dir,
        default_project_trust,
        clear_default_project_trust,
        default_reasoning_effort,
        clear_default_reasoning_effort,
        block_images,
        clear_block_images,
        image_auto_resize,
        clear_image_auto_resize,
        thinking_budget,
        clear_thinking_budget,
        default_bash_shell_path,
        clear_default_bash_shell_path,
        default_bash_command_prefix,
        clear_default_bash_command_prefix,
        default_compaction_reserve_tokens,
        clear_default_compaction_reserve_tokens,
        default_compaction_keep_recent_tokens,
        clear_default_compaction_keep_recent_tokens,
        default_retry_max_retries,
        clear_default_retry_max_retries,
        default_retry_base_delay_ms,
        clear_default_retry_base_delay_ms,
        default_provider_timeout_ms,
        clear_default_provider_timeout_ms,
        default_retry_max_backoff_ms,
        clear_default_retry_max_backoff_ms,
        default_models,
        clear_default_models,
        default_skill_paths,
        clear_default_skill_paths,
        default_prompt_template_paths,
        clear_default_prompt_template_paths,
        default_branch_summary_reserve_tokens,
        clear_default_branch_summary_reserve_tokens,
    } = cmd
    else {
        unreachable!("dispatched on Command::Settings")
    };
    let mut store = beyond_ai_agent::settings::SettingsStore::open_default();
    let any_write = model.is_some()
        || clear_model
        || gateway_url.is_some()
        || clear_gateway_url
        || session_dir.is_some()
        || clear_session_dir
        || default_project_trust.is_some()
        || clear_default_project_trust
        || default_reasoning_effort.is_some()
        || clear_default_reasoning_effort
        || block_images.is_some()
        || clear_block_images
        || image_auto_resize.is_some()
        || clear_image_auto_resize
        || !thinking_budget.is_empty()
        || !clear_thinking_budget.is_empty()
        || default_bash_shell_path.is_some()
        || clear_default_bash_shell_path
        || default_bash_command_prefix.is_some()
        || clear_default_bash_command_prefix
        || default_compaction_reserve_tokens.is_some()
        || clear_default_compaction_reserve_tokens
        || default_compaction_keep_recent_tokens.is_some()
        || clear_default_compaction_keep_recent_tokens
        || default_retry_max_retries.is_some()
        || clear_default_retry_max_retries
        || default_retry_base_delay_ms.is_some()
        || clear_default_retry_base_delay_ms
        || default_provider_timeout_ms.is_some()
        || clear_default_provider_timeout_ms
        || default_retry_max_backoff_ms.is_some()
        || clear_default_retry_max_backoff_ms
        || default_models.is_some()
        || clear_default_models
        || default_skill_paths.is_some()
        || clear_default_skill_paths
        || default_prompt_template_paths.is_some()
        || clear_default_prompt_template_paths
        || default_branch_summary_reserve_tokens.is_some()
        || clear_default_branch_summary_reserve_tokens;
    if model.is_some() || clear_model {
        store.set_default_model(model)?;
    }
    if gateway_url.is_some() || clear_gateway_url {
        store.set_default_gateway_url(gateway_url)?;
    }
    if session_dir.is_some() || clear_session_dir {
        store.set_default_session_dir(session_dir)?;
    }
    if default_project_trust.is_some() || clear_default_project_trust {
        store.set_default_project_trust(default_project_trust)?;
    }
    if default_reasoning_effort.is_some() || clear_default_reasoning_effort {
        store.set_default_reasoning_effort(
            default_reasoning_effort.map(|e| e.as_str().to_string()),
        )?;
    }
    if block_images.is_some() || clear_block_images {
        store.set_block_images(if clear_block_images {
            None
        } else {
            block_images
        })?;
    }
    if image_auto_resize.is_some() || clear_image_auto_resize {
        store.set_image_auto_resize(if clear_image_auto_resize {
            None
        } else {
            image_auto_resize
        })?;
    }
    for kv in &thinking_budget {
        let (effort, tokens) = kv
            .split_once('=')
            .ok_or_else(|| format!("--thinking-budget {kv:?} must be EFFORT=TOKENS"))?;
        parse_reasoning_effort(effort).map_err(|e| format!("--thinking-budget {kv:?}: {e}"))?;
        let tokens: u32 = tokens.parse().map_err(|_| {
            format!("--thinking-budget {kv:?}: TOKENS must be a non-negative integer")
        })?;
        store.set_thinking_budget_override(effort.to_string(), Some(tokens))?;
    }
    for effort in &clear_thinking_budget {
        parse_reasoning_effort(effort)
            .map_err(|e| format!("--clear-thinking-budget {effort:?}: {e}"))?;
        store.set_thinking_budget_override(effort.to_string(), None)?;
    }
    if default_bash_shell_path.is_some() || clear_default_bash_shell_path {
        store.set_default_bash_shell_path(default_bash_shell_path)?;
    }
    if default_bash_command_prefix.is_some() || clear_default_bash_command_prefix {
        store.set_default_bash_command_prefix(default_bash_command_prefix)?;
    }
    if default_compaction_reserve_tokens.is_some() || clear_default_compaction_reserve_tokens {
        store.set_default_compaction_reserve_tokens(default_compaction_reserve_tokens)?;
    }
    if default_compaction_keep_recent_tokens.is_some()
        || clear_default_compaction_keep_recent_tokens
    {
        store.set_default_compaction_keep_recent_tokens(default_compaction_keep_recent_tokens)?;
    }
    if default_retry_max_retries.is_some() || clear_default_retry_max_retries {
        store.set_default_retry_max_retries(default_retry_max_retries)?;
    }
    if default_retry_base_delay_ms.is_some() || clear_default_retry_base_delay_ms {
        store.set_default_retry_base_delay_ms(default_retry_base_delay_ms)?;
    }
    if default_provider_timeout_ms.is_some() || clear_default_provider_timeout_ms {
        store.set_default_provider_timeout_ms(default_provider_timeout_ms)?;
    }
    if default_retry_max_backoff_ms.is_some() || clear_default_retry_max_backoff_ms {
        store.set_default_retry_max_backoff_ms(default_retry_max_backoff_ms)?;
    }
    if default_models.is_some() || clear_default_models {
        store.set_default_models_list(default_models)?;
    }
    if default_skill_paths.is_some() || clear_default_skill_paths {
        store.set_default_skill_paths(default_skill_paths)?;
    }
    if default_prompt_template_paths.is_some() || clear_default_prompt_template_paths {
        store.set_default_prompt_template_paths(default_prompt_template_paths)?;
    }
    if default_branch_summary_reserve_tokens.is_some()
        || clear_default_branch_summary_reserve_tokens
    {
        store.set_default_branch_summary_reserve_tokens(default_branch_summary_reserve_tokens)?;
    }
    print_settings(&store, any_write);
    Ok(())
}

/// Every field of the global store, plus where a project tier or `models.json` changes what `run`/`serve` will use.
fn print_settings(store: &beyond_ai_agent::settings::SettingsStore, any_write: bool) {
    if any_write {
        println!("updated settings:");
    }
    let s = store.get();
    println!(
        "default_model: {}",
        s.default_model.as_deref().unwrap_or("(not set)")
    );
    println!(
        "default_gateway_url: {}",
        s.default_gateway_url.as_deref().unwrap_or("(not set)")
    );
    println!(
        "default_session_dir: {}",
        s.default_session_dir.as_deref().unwrap_or("(not set)")
    );
    println!(
        "default_project_trust: {}",
        match s.default_project_trust {
            Some(beyond_ai_agent::settings::TrustPolicy::Always) => "always",
            Some(beyond_ai_agent::settings::TrustPolicy::Never) => "never",
            Some(beyond_ai_agent::settings::TrustPolicy::Ask) => "ask",
            None => "(not set)",
        }
    );
    println!(
        "default_reasoning_effort: {}",
        s.default_reasoning_effort.as_deref().unwrap_or("(not set)")
    );
    println!(
        "block_images: {}",
        match s.block_images {
            Some(true) => "true",
            Some(false) => "false",
            None => "(not set)",
        }
    );
    println!(
        "image_auto_resize: {}",
        match s.image_auto_resize {
            Some(true) => "true",
            Some(false) => "false",
            None => "(not set)",
        }
    );
    match &s.thinking_budget_overrides {
        Some(overrides) if !overrides.is_empty() => {
            let rendered = overrides
                .iter()
                .map(|(effort, tokens)| format!("{effort}={tokens}"))
                .collect::<Vec<_>>()
                .join(", ");
            println!("thinking_budget_overrides: {rendered}");
        }
        _ => println!("thinking_budget_overrides: (not set)"),
    }
    println!(
        "default_bash_shell_path: {}",
        s.default_bash_shell_path.as_deref().unwrap_or("(not set)")
    );
    println!(
        "default_bash_command_prefix: {}",
        s.default_bash_command_prefix
            .as_deref()
            .unwrap_or("(not set)")
    );
    println!(
        "default_compaction_reserve_tokens: {}",
        s.default_compaction_reserve_tokens
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_compaction_keep_recent_tokens: {}",
        s.default_compaction_keep_recent_tokens
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_retry_max_retries: {}",
        s.default_retry_max_retries
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_retry_base_delay_ms: {}",
        s.default_retry_base_delay_ms
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_provider_timeout_ms: {}",
        s.default_provider_timeout_ms
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_retry_max_backoff_ms: {}",
        s.default_retry_max_backoff_ms
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_models_list: {}",
        s.default_models_list
            .as_ref()
            .map(|v| v.join(","))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_skill_paths: {}",
        s.default_skill_paths
            .as_ref()
            .map(|v| v.join(","))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_prompt_template_paths: {}",
        s.default_prompt_template_paths
            .as_ref()
            .map(|v| v.join(","))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "default_branch_summary_reserve_tokens: {}",
        s.default_branch_summary_reserve_tokens
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    // The values above are the global store alone; `run`/`serve` also merge a trusted
    // project's settings files over it, so say where that changes what they will use.
    if let Ok(cwd) = std::env::current_dir() {
        for note in beyond_ai_agent::settings::project_override_notes(&cwd) {
            println!("{note}");
        }
    }
    // Fix 9's "CLI-visible" requirement: this file is entirely hand-edited (like pi's own
    // `models.json`), with no `--set`/`--clear` flags of its own here — just enough surface so
    // an operator debugging "why is this model still hitting the gateway" can confirm the file
    // is actually being read, and how many overrides it currently holds, without a dedicated
    // dump-the-whole-file command.
    let overrides = beyond_ai_agent::settings::ModelOverrides::open_default();
    let overrides_path = beyond_ai_agent::settings::model_overrides_path();
    if overrides.is_empty() {
        println!(
            "model_overrides: {} (not present or empty)",
            overrides_path.display()
        );
    } else {
        println!(
            "model_overrides: {} ({} model id(s) overridden)",
            overrides_path.display(),
            overrides.len()
        );
    }
}
