"""Harbor installed-agent adapter for `beyond-ai-agent`.

Copies a host-built static musl binary into the task container and runs it
headless. LLM calls go out during `agent.run()` only — pass Harbor
`--allow-agent-host openrouter.ai` (or the provider you actually hit) because
Frontier Terminal-Bench tasks default to no internet.

This is the eval harness (verifier pass/fail, turns, tokens, $). It is not Code
Mode: Terminal-Bench has no MCP catalog, and the density-default binary does
not link QuickJS.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import override

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_DONE_RE = re.compile(
    r"\[done in (\d+) step\(s\); (\d+) in / (\d+) out tokens\]"
)
_REMOTE_BIN = "/usr/local/bin/beyond-ai-agent"
_OUTPUT_JSONL = "beyond.jsonl"
_OUTPUT_STDERR = "beyond.stderr"

# OpenRouter list price for Kimi K3 at FrontierHarness freeze time
# (skills/frontierharness-eval/reference.md). Used only to fill cost_usd.
_KIMI_K3_INPUT_PER_M = 3.00
_KIMI_K3_CACHE_PER_M = 0.30
_KIMI_K3_OUTPUT_PER_M = 15.00


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_agent_binary() -> Path:
    """Host path of the binary uploaded into each task container."""
    override = os.environ.get("BEYOND_AI_AGENT_BIN")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"BEYOND_AI_AGENT_BIN is not a file: {path}")
        return path

    workspace = _workspace_root()
    candidates: list[Path] = []
    which = shutil.which("beyond-ai-agent")
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            workspace / "target/x86_64-unknown-linux-musl/release/beyond-ai-agent",
            workspace / "target/release/beyond-ai-agent",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "beyond-ai-agent binary not found. Build a static musl release "
        "(so it runs in any Terminal-Bench image):\n"
        "  cargo build -p beyond-ai-agent --release --target x86_64-unknown-linux-musl\n"
        "or set BEYOND_AI_AGENT_BIN to an existing binary."
    )


def _cli_model_id(model_name: str | None) -> str:
    """Harbor `-m openrouter/moonshotai/kimi-k3` → the id OpenRouter expects."""
    if not model_name:
        return "moonshotai/kimi-k3"
    name = model_name.strip()
    for prefix in ("openrouter/", "openai/"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


class BeyondAiAgent(BaseInstalledAgent):
    CLI_FLAGS = [
        CliFlag(
            "max_steps",
            cli="--max-steps",
            type="int",
            default=120,
            env_fallback="AI_AGENT_MAX_STEPS",
        ),
    ]

    @staticmethod
    @override
    def name() -> str:
        return "beyond-ai-agent"

    @override
    def get_version_command(self) -> str | None:
        return f"{_REMOTE_BIN} --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("ca_certificates",))
        binary = resolve_agent_binary()
        await environment.upload_file(binary, _REMOTE_BIN)
        quoted = shlex.quote(_REMOTE_BIN)
        await self.exec_as_root(
            environment,
            command=f"chmod 755 {quoted} && {quoted} --version",
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        logs = self.environment_logs_dir.as_posix()
        jsonl = f"{logs}/{_OUTPUT_JSONL}"
        stderr_path = f"{logs}/{_OUTPUT_STDERR}"
        instruction_path = f"{logs}/instruction.md"

        await self.exec_as_agent(
            environment, command=f"mkdir -p {shlex.quote(logs)}"
        )
        with tempfile.TemporaryDirectory(prefix="beyond-instruction-") as tmp:
            local = Path(tmp) / "instruction.md"
            local.write_text(instruction)
            await environment.upload_file(local, instruction_path)
        if environment.default_user is not None:
            owner = shlex.quote(str(environment.default_user))
            await self.exec_as_root(
                environment,
                command=(
                    f"chown {owner} {shlex.quote(instruction_path)} "
                    f"{shlex.quote(logs)}"
                ),
            )

        model = _cli_model_id(self.model_name)
        flags = self.build_cli_flags()
        if flags:
            flags = f" {flags}"

        provider = self._get_env("AI_PROVIDER") or "openrouter"
        env = {
            "AI_PROVIDER": provider,
            "AI_DIRECT": self._get_env("AI_DIRECT") or "1",
            "AI_AGENT_MODEL": model,
        }
        api_key = self._get_env("OPENROUTER_API_KEY")
        if api_key:
            env["OPENROUTER_API_KEY"] = api_key
        for extra in ("AI_API_KEY", "AI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            value = self._get_env(extra)
            if value:
                env[extra] = value
        if provider == "openrouter" and "OPENROUTER_API_KEY" not in env:
            raise ValueError(
                "OPENROUTER_API_KEY is required (host env or --ae OPENROUTER_API_KEY=…)"
            )

        # Stdin is the instruction: `@file` would wrap it in a <file> block.
        cmd = (
            f"mkdir -p {shlex.quote(logs)}; "
            f"{shlex.quote(_REMOTE_BIN)} run --json --no-session-persistence "
            f"--model {shlex.quote(model)}{flags} "
            f"< {shlex.quote(instruction_path)} "
            f"> {shlex.quote(jsonl)} 2> {shlex.quote(stderr_path)}"
        )
        try:
            await self.exec_as_agent(environment, command=cmd, env=env)
        except NonZeroAgentExitCodeError as exc:
            # Leave the workspace for the verifier; a non-zero CLI exit is a
            # scored miss (or a transport error), not a Harbor infra exception.
            self.logger.warning("beyond-ai-agent exited non-zero: %s", exc)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        input_tokens = 0
        output_tokens = 0
        cache_read = 0
        steps = 0

        jsonl = self.logs_dir / _OUTPUT_JSONL
        if jsonl.is_file():
            for line in jsonl.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("kind")
                if kind == "turn_end":
                    steps += 1
                if kind == "stream" and event.get("type") == "usage":
                    input_tokens += int(event.get("input_tokens") or 0)
                    output_tokens += int(event.get("output_tokens") or 0)
                    cache_read += int(event.get("cache_read_tokens") or 0)

        stderr = self.logs_dir / _OUTPUT_STDERR
        if stderr.is_file():
            text = stderr.read_text(errors="replace")
            match = None
            for match in _DONE_RE.finditer(text):
                pass
            if match:
                steps = max(steps, int(match.group(1)))
                if input_tokens == 0:
                    input_tokens = int(match.group(2))
                if output_tokens == 0:
                    output_tokens = int(match.group(3))

        if steps:
            context.metadata = {**(context.metadata or {}), "steps": steps}
        if input_tokens or output_tokens or cache_read:
            # Harbor: n_input_tokens includes cache.
            context.n_input_tokens = input_tokens + cache_read
            context.n_output_tokens = output_tokens
            context.n_cache_tokens = cache_read
            fresh = max(input_tokens, 0)
            context.cost_usd = (
                fresh / 1_000_000 * _KIMI_K3_INPUT_PER_M
                + cache_read / 1_000_000 * _KIMI_K3_CACHE_PER_M
                + output_tokens / 1_000_000 * _KIMI_K3_OUTPUT_PER_M
            )
