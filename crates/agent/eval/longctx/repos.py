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
    # Tests that fail on the pristine tree in this environment for a reason no reimplementation can
    # fix (a missing optional dependency, the interpreter's own messages). Deselected for the agent and
    # the grader alike (`PYTEST_ADDOPTS`), so the pristine tree passes 100% and every test the agent
    # sees is passable. `validate.py` fails if the pristine tree fails anything else.
    deselect: tuple[str, ...] = ()
    # A distribution whose metadata the tests read (`importlib.metadata.version(...)`); upstream's test
    # env installs the package. The harness puts a metadata-only `<dist>-<version>.dist-info` (name
    # and version, no files) on PYTHONPATH instead of an installed copy.
    dist: str = ""

    @property
    def version(self) -> str:
        return self.tag.removeprefix("v")


REPOS: tuple[Repo, ...] = (
    Repo("click", "pallets/click", "8.1.8", "src/click", ("tests",), pythonpath="src"),
    Repo("jinja", "pallets/jinja", "3.1.6", "src/jinja2", ("tests",), pythonpath="src", deps=("markupsafe==3.0.2", "trio==0.27.0")),
    Repo("sqlparse", "andialbrecht/sqlparse", "0.5.3", "sqlparse", ("tests",)),
    Repo("more-itertools", "more-itertools/more-itertools", "v10.6.0", "more_itertools", ("tests",)),
    Repo("toolz", "pytoolz/toolz", "1.0.0", "toolz", ("toolz/tests",)),
    Repo("boltons", "mahmoud/boltons", "24.1.0", "boltons", ("tests",)),
    Repo("mistune", "lepture/mistune", "v3.1.1", "src/mistune", ("tests",), pythonpath="src"),
    Repo("tabulate", "astanin/python-tabulate", "v0.9.0", "tabulate", ("test",)),
    # parso checks its syntax errors against the running interpreter's; these 14 messages changed
    # after parso 0.8.4 (CPython 3.12 wording).
    Repo("parso", "davidhalter/parso", "v0.8.4", "parso", ("test",), deselect=tuple(
        f"test/test_python_errors.py::test_python_exception_matches[{case}]" for case in (
            "def f(x=3, y): pass", "lambda x=3, y: x", "yield from []", 'f"{*x}"', 'f"{}"', 'f"{#}"',
            "f'{1!b}'", "f'{1:{5:{3}}}'", "f'{continue}'", "f'{1;1}'", "f'{a;}'", "f'{b\"\" \"\"}'",
            r'''f"{'\\n'}"''', "f'{1=!b}'"))),
    # The railroad-diagram tests (and `test_range_check`, which draws one) need jinja2, which would put
    # a pristine copy of the jinja task's package in the venv.
    Repo("pyparsing", "pyparsing/pyparsing", "3.2.1", "pyparsing", ("tests",), pytest_args=("--ignore=tests/test_diagram.py",),
         deselect=("tests/test_examples.py::TestExamples::test_range_check",)),
    Repo("marshmallow", "marshmallow-code/marshmallow", "3.26.1", "src/marshmallow", ("tests",), pythonpath="src",
         deps=("simplejson==3.19.3", "packaging"), dist="marshmallow"),
)

BY_NAME = {r.name: r for r in REPOS}

# A task is one or more stubbed repos side by side in one workspace, under one prompt. Two per task
# puts a run's context past 200K tokens, the trigger range the sweep has to straddle.
TASKS: dict[str, tuple[str, ...]] = {
    "pallets": ("click", "jinja"),  # 4881 stubbed lines
    "parsers": ("parso", "pyparsing", "sqlparse"),  # 4666
    "utils": ("boltons", "toolz", "marshmallow"),  # 4287
    "text": ("mistune", "more-itertools", "tabulate"),  # 3988
    # Five or six repos: an uncompacted Sonnet run passes 500K, so a 500K trigger actually fires.
    "pallets-parsers": ("click", "jinja", "parso", "pyparsing", "sqlparse"),  # 9547
    "utils-text": ("boltons", "toolz", "marshmallow", "mistune", "more-itertools", "tabulate"),  # 8275
    # Single small repos: cheap-model gate tasks whose context mostly stays under 100K.
    "toolz": ("toolz",),  # 681
    "sqlparse": ("sqlparse",),  # 840
    "tabulate": ("tabulate",),  # 425
    "marshmallow": ("marshmallow",),  # 1101
}

# Test-only tooling in the shared venv. Never the packages under test: an installed copy would be
# a pristine reference the agent could diff against.
VENV_DEPS = ("pytest==8.3.4",)
