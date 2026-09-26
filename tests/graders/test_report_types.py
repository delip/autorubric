"""``Rubric.grade`` and ``Grader.grade`` are typed by the grader's report type.

``Grader`` is generic in the report type its ``aggregate`` returns: ``CriterionGrader``
returns an ``EnsembleEvaluationReport``, and a custom grader declares its own. The check
runs the project's type checker (``ty``) on a small module, because the annotations are
the feature; nothing changes at runtime.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# The ty executable installed beside this interpreter (ty.exe on Windows), if any.
TY = shutil.which("ty", path=str(Path(sys.executable).parent))

MODULE = """
from typing import Any, assert_type

from autorubric import EnsembleEvaluationReport, EvaluationReport, LLMConfig, Rubric
from autorubric.graders import CriterionGrader, Grader
from autorubric.types import Criterion


class PlainGrader(Grader[EvaluationReport]):
    async def judge(
        self,
        to_grade: str,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> Any:
        return None

    async def aggregate(self, judge_results: Any, *, normalize: bool = True) -> EvaluationReport:
        return EvaluationReport(score=1.0)


async def check(rubric: Rubric) -> None:
    grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"))
    assert_type(await rubric.grade(to_grade="x", grader=grader), EnsembleEvaluationReport)
    assert_type(await grader.grade("x", rubric.rubric), EnsembleEvaluationReport)
    assert_type(await rubric.grade(to_grade="x", grader=PlainGrader()), EvaluationReport)
"""


def test_grade_returns_the_graders_report_type(tmp_path: Path) -> None:
    if TY is None:
        pytest.skip("the ty type checker is not installed in this environment")
    module = tmp_path / "report_types.py"
    module.write_text(MODULE, encoding="utf-8")

    completed = subprocess.run(
        [TY, "check", "--python", sys.executable, "--output-format", "concise", str(module)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
