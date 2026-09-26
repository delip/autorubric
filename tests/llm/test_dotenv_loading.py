"""Importing AutoRubric loads the ``.env`` found from the working directory upward.

Each test runs a script in a fresh interpreter, because the load happens once, when
``autorubric.llm`` is imported. The probe is a script file rather than ``python -c``:
python-dotenv treats a ``__main__`` without ``__file__`` as interactive, which is the case
the working-directory search always covered.
"""

import os
import subprocess
import sys
from pathlib import Path

PROBE = "AUTORUBRIC_DOTENV_PROBE"


def _probe_value(cwd: Path, *, environ: dict[str, str] | None = None) -> str:
    """Run a script that imports autorubric from ``cwd`` and return the probe's value."""
    script = cwd / "probe_dotenv.py"
    script.write_text(
        "import os\n"
        "import autorubric  # noqa: F401\n"
        f"print(os.environ.get({PROBE!r}, '<unset>'))\n",
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if k != PROBE}
    env.update(environ or {})
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip().splitlines()[-1]


def test_script_loads_the_dotenv_in_its_working_directory(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{PROBE}=from-working-directory\n", encoding="utf-8")

    assert _probe_value(tmp_path) == "from-working-directory"


def test_script_loads_the_nearest_dotenv_above_its_working_directory(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{PROBE}=from-parent\n", encoding="utf-8")
    working_directory = tmp_path / "project" / "scripts"
    working_directory.mkdir(parents=True)

    assert _probe_value(working_directory) == "from-parent"


def test_variables_already_set_take_precedence_over_the_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{PROBE}=from-dotenv\n", encoding="utf-8")

    assert _probe_value(tmp_path, environ={PROBE: "from-environment"}) == "from-environment"
