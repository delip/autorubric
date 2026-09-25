"""Full-panel metrics are those of the library before cascade-aware metrics.

``tests/golden/metrics/full_panel_metrics.json`` holds ``compute_metrics`` outputs captured
from that library (see ``capture_metrics_fixtures.py`` next to it) for hand-built ensemble
panels, a single-judge run and a panel graded by ``CriterionGrader`` with mocked judges,
each under several settings. A full panel (every judge votes on every criterion of every
item) has no missing vote and no superseded vote, so every ``MetricsResult`` field
(``model_dump``), both ``summary()`` texts and the ``to_dataframe()`` frame must be
unchanged, as must the error of a setting the library refuses.

On the interpreter and platform the outputs were captured on, the current library
reproduces them bit for bit. Elsewhere a float's last bits may differ from the capture
whichever library computes it, so floats are compared up to ``FLOAT_TOLERANCE`` and
everything else exactly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "metrics"

FLOAT_TOLERANCE = 1e-12
"""How far a float may be from the captured one: relative, or absolute near zero.

A float's last bits depend on where it is computed. The builtin ``sum()`` of floats
rounds differently before Python 3.12 (compensated summation since), and numpy's dot
products (behind the Pearson coefficient and its norms) run through the platform's BLAS,
whose accumulation order and fused multiply-adds depend on the build and the CPU; libm
differences reach the p-values. These unit-scale metrics move by a few units in the last
place (around 1e-16) that way, while any change to what a metric measures moves it by
orders of magnitude more than this tolerance.
"""


def _load_capture_module() -> ModuleType:
    name = "capture_metrics_fixtures"
    spec = importlib.util.spec_from_file_location(name, GOLDEN_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_capture_module()
GOLDEN = json.loads((GOLDEN_DIR / "full_panel_metrics.json").read_text(encoding="utf-8"))
CASES = {
    case.name: case
    for case in [*CAPTURE.build_cases(), CAPTURE.graded_case(GOLDEN["graded_records"])]
}


def assert_same(actual: Any, expected: Any, path: str = "") -> None:
    """Assert that ``actual`` is ``expected``, floats up to ``FLOAT_TOLERANCE``.

    Types must match (an ``int`` is not a ``float``, ``None`` is not ``0.0``), as must dict
    keys, list lengths and every value that is not a float. NaN matches only NaN.
    """
    assert type(actual) is type(expected), f"{path}: {actual!r} != {expected!r}"
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: {sorted(actual)} != {sorted(expected)}"
        for key, value in expected.items():
            assert_same(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: {len(actual)} != {len(expected)} items"
        for idx, (item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            assert_same(item, expected_item, f"{path}[{idx}]")
    elif isinstance(expected, float):
        same = (
            math.isnan(actual)
            if math.isnan(expected)
            else math.isclose(actual, expected, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)
        )
        assert same, f"{path}: {actual!r} != {expected!r}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _frame_rows(rows: list[list[str | None]]) -> list[list[Any]]:
    """The frame's cells, with the float cells (``"float:<repr>"``) decoded to floats."""
    return [
        [
            float(cell.removeprefix("float:"))
            if cell is not None and cell.startswith("float:")
            else cell
            for cell in row
        ]
        for row in rows
    ]


def test_every_captured_case_is_rebuilt():
    assert sorted(CASES) == sorted(GOLDEN["outcomes"])


@pytest.mark.parametrize(
    ("case_name", "setting_idx"),
    [(name, idx) for name in sorted(GOLDEN["outcomes"]) for idx in range(len(GOLDEN["settings"]))],
)
def test_full_panel_metrics_match_the_captured_library(case_name: str, setting_idx: int):
    pytest.importorskip("pandas")
    expected = GOLDEN["outcomes"][case_name][setting_idx]
    actual = CAPTURE.outcome(CASES[case_name], GOLDEN["settings"][setting_idx])
    assert actual.keys() == expected.keys()
    assert actual.get("error") == expected.get("error")
    if "error" in expected:
        return
    # The texts round every float to a few decimals, so they are compared exactly.
    assert actual["summary"] == expected["summary"]
    assert actual["summary_verbose"] == expected["summary_verbose"]
    assert_same(json.loads(actual["model_dump"]), json.loads(expected["model_dump"]))
    assert actual["frame"]["columns"] == expected["frame"]["columns"]
    assert actual["frame"]["dtypes"] == expected["frame"]["dtypes"]
    assert_same(_frame_rows(actual["frame"]["rows"]), _frame_rows(expected["frame"]["rows"]))


def test_the_comparison_allows_only_the_last_bits_of_a_float_to_differ():
    captured = {"kappa": 0.607936507936508, "n": 3, "p": None, "label": "moderate", "ok": [True]}
    # The same kappas averaged by Python 3.11's sum(): one unit in the last place lower.
    assert_same({**captured, "kappa": 0.6079365079365079}, captured)
    assert_same([math.nan], [math.nan])
    changed = [
        {**captured, "kappa": 0.6079366},
        {**captured, "kappa": None},
        {**captured, "n": 3.0},
        {**captured, "p": 0.0},
        {**captured, "label": "fair"},
        {**captured, "ok": [1]},
        {**captured, "ok": []},
        {key: value for key, value in captured.items() if key != "n"},
        {**captured, "extra": 1},
    ]
    for value in changed:
        with pytest.raises(AssertionError):
            assert_same(value, captured)
    for actual, expected in [(0.5, math.nan), (math.nan, 0.5)]:
        with pytest.raises(AssertionError):
            assert_same(actual, expected)
