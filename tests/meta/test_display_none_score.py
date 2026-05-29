"""Issue #7: meta-rubric display/render must be None-safe for a score-less report.

These render the meta-rubric eval report (normally a computed report), but must not
crash if a grade-FAILURE (score=None) is passed in.
"""

from pathlib import Path

from autorubric.meta._display import display_to_stdout, render_to_html
from autorubric.types import EnsembleEvaluationReport

_META_RUBRIC_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autorubric"
    / "meta"
    / "data"
    / "meta_rubric_standalone.json"
)


def _none_score_report() -> EnsembleEvaluationReport:
    return EnsembleEvaluationReport(
        score=None,
        raw_score=None,
        report=[],
        error="No judge results to aggregate",
    )


def test_render_to_html_none_score_does_not_raise():
    html = render_to_html(
        _none_score_report(),
        meta_rubric_path=_META_RUBRIC_PATH,
        title="Test",
    )
    # Renders an n/a score instead of crashing on a None comparison/format.
    assert isinstance(html, str)
    assert "n/a" in html


def test_display_to_stdout_none_score_does_not_raise():
    # display_to_stdout prints to a Console; the assertion is simply "no exception".
    display_to_stdout(
        _none_score_report(),
        meta_rubric_path=_META_RUBRIC_PATH,
        title="Test",
    )
