"""The real repositories the long-context tasks are built from, pinned by tag.

Each is a pure-Python project whose own test suite runs in seconds with nothing but pytest
(plus the listed runtime deps), so a task's grade is that suite and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Repo:
    name: str
    slug: str  # GitHub owner/repo
    tag: str
    src: str  # the package directory mutations are drawn from, relative to the repo root
    tests: tuple[str, ...]  # test paths, relative to the repo root; restored before grading
    pythonpath: str = "."  # what PYTHONPATH must hold for `import <package>` to find `src`
    deps: tuple[str, ...] = ()  # runtime deps installed into the shared venv (never the package)
    pytest_args: tuple[str, ...] = ()
    # Files outside `tests` the grade also restores (pytest config, conftest).
    restore: tuple[str, ...] = field(default=("pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", "conftest.py"))


REPOS: tuple[Repo, ...] = (
    Repo("click", "pallets/click", "8.1.8", "src/click", ("tests",), pythonpath="src"),
    Repo("jinja", "pallets/jinja", "3.1.6", "src/jinja2", ("tests",), pythonpath="src", deps=("markupsafe==3.0.2", "trio==0.27.0")),
    Repo("sqlparse", "andialbrecht/sqlparse", "0.5.3", "sqlparse", ("tests",)),
    Repo("more-itertools", "more-itertools/more-itertools", "v10.6.0", "more_itertools", ("tests",)),
    Repo("toolz", "pytoolz/toolz", "1.0.0", "toolz", ("toolz/tests",)),
    Repo("boltons", "mahmoud/boltons", "24.1.0", "boltons", ("tests",)),
    Repo("mistune", "lepture/mistune", "v3.1.1", "src/mistune", ("tests",), pythonpath="src"),
    Repo("tabulate", "astanin/python-tabulate", "v0.9.0", "tabulate", ("test",)),
    Repo("parso", "davidhalter/parso", "v0.8.4", "parso", ("test",)),
    # The railroad-diagram tests need jinja2, which would put a pristine copy of the jinja task's
    # package in the venv.
    Repo("pyparsing", "pyparsing/pyparsing", "3.2.1", "pyparsing", ("tests",), pytest_args=("--ignore=tests/test_diagram.py",)),
    Repo("marshmallow", "marshmallow-code/marshmallow", "3.26.1", "src/marshmallow", ("tests",), pythonpath="src", deps=("simplejson==3.19.3", "packaging")),
)

BY_NAME = {r.name: r for r in REPOS}

# A task is one or more stubbed repos side by side in one workspace, under one prompt. Two per task
# puts a run's context past 200K tokens, the trigger range the sweep has to straddle.
TASKS: dict[str, tuple[str, ...]] = {
    "pallets": ("click", "jinja"),  # 4881 stubbed lines
    "parsers": ("parso", "pyparsing", "sqlparse"),  # 4666
    "utils": ("boltons", "toolz", "marshmallow"),  # 4287
    "text": ("mistune", "more-itertools", "tabulate"),  # 3988
}

# Test-only tooling in the shared venv. Never the packages under test: an installed copy would be
# a pristine reference the agent could diff against.
VENV_DEPS = ("pytest==8.3.4",)
