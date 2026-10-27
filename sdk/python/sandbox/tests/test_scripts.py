"""The shell scripts are constants sent byte for byte: they must be the reference's, not a copy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / "crates/sandbox/src/fs/shell.rs"


def test_scripts_match_the_reference_sources() -> None:
    if not REFERENCE.is_file():
        pytest.skip("the Rust reference is not available outside the repository")
    gen = ROOT / "scripts" / "gen_scripts.py"
    done = subprocess.run([sys.executable, str(gen), "--check"], capture_output=True, text=True)
    assert done.returncode == 0, "run scripts/gen_scripts.py: _scripts.py is out of date"
