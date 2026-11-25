"""Builds the job client's kernel into the wheel.

``crates/job-client-ffi`` is the proven kernel behind a C ABI; the package loads it with ``ctypes``
(``beyond_sandbox/_kernel.py``). A wheel therefore carries one native library and is tagged for the platform
it was built on (``py3-none-<platform>``): the library needs no CPython ABI, so one wheel serves every Python
of its platform. There is no source distribution and no pure-Python fallback: a platform with no wheel fails
to install, which is the honest answer for code whose decisions are proven only as the library.

An editable install (``pip install -e``) builds nothing: a checkout's library is built by
``mise run build:py-kernel`` (``crates/job-client-ffi/build-native.sh``), the same script this hook runs.
"""

from __future__ import annotations

import subprocess
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class NativeKernel(BuildHookInterface):  # type: ignore[misc]
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        root = Path(self.root).resolve().parents[2]
        script = root / "crates" / "job-client-ffi" / "build-native.sh"
        if not script.is_file():
            raise RuntimeError(
                f"{script} is missing: the wheel is built from a checkout of the repository "
                "(the native kernel has no other source), never from a copy of this directory"
            )
        # The library is staged outside the source tree, so a wheel build leaves the checkout clean.
        stage = Path(tempfile.mkdtemp(prefix="beyond-kernel-"))
        subprocess.run(["bash", str(script), str(stage)], check=True, cwd=root)
        build_data["pure_python"] = False
        # The platform the way setuptools names it: cibuildwheel sets it per architecture, and repairs a
        # Linux wheel's `linux_<arch>` into the manylinux or musllinux tag its library is compatible with.
        platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["tag"] = f"py3-none-{platform}"
        for f in stage.iterdir():
            build_data["force_include"][str(f)] = f"beyond_sandbox/_native/{f.name}"
