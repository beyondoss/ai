"""The shell scripts are constants sent byte for byte: they must be the reference's, not a copy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
REFERENCE = REPO / "crates/sandbox/src/fs/shell.rs"


def test_scripts_match_the_reference_sources() -> None:
    if not REFERENCE.is_file():
        # Inside a checkout a missing reference is a broken tree, not a skip.
        if (REPO / "Cargo.toml").is_file():
            pytest.fail(f"{REFERENCE} is missing: expected the Rust reference scripts there")
        pytest.skip("the Rust reference is not available outside the repository")
    gen = ROOT / "scripts" / "gen_scripts.py"
    done = subprocess.run([sys.executable, str(gen), "--check"], capture_output=True, text=True)
    assert done.returncode == 0, "run scripts/gen_scripts.py: _scripts.py is out of date"


def test_bulk_scripts_are_the_specs_appendix() -> None:
    """Upload, download and stage (appendix G, H, I) are sent byte for byte as the spec prints them."""
    import re

    from beyond_sandbox import _scripts

    spec = (REPO / "sdk/conformance/FILES-OVER-EXEC.md").read_text()
    appendix = spec[spec.index("## 7. Appendix: the scripts") :]
    for letter, name, const in (
        ("G", "UPLOAD_SCRIPT", _scripts.UPLOAD),
        ("H", "DOWNLOAD_SCRIPT", _scripts.DOWNLOAD),
        ("I", "STAGE_SCRIPT", _scripts.STAGE),
    ):
        m = re.search(rf"### {letter}\. {name}[^\n]*\n\n```sh\n(.*?)\n```", appendix, re.S)
        assert m, f"appendix {letter} is missing"
        assert m.group(1) == const, f"{name} differs from appendix {letter}"
