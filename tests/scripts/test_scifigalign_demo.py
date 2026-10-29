"""Tests for ``examples/scifigalign_demo.py``'s reference rows.

The paper's Table 4 reports two kinds of SRCC: the baselines' is the macro average of the
per-dimension Spearman coefficients, and SciFigAlign's is on the overall score. The demo
prints each reference row beside the same kind of SRCC from the run. The demo is not part of
the package, so it is loaded from its file; nothing is graded or fetched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

from autorubric.metrics import OrdinalCriterionMetrics

DEMO = Path(__file__).resolve().parents[2] / "examples" / "scifigalign_demo.py"


def _load_demo() -> ModuleType:
    name = "scifigalign_demo"
    spec = importlib.util.spec_from_file_location(name, DEMO)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DEMO_MODULE = _load_demo()


def dimension(name: str, srcc: float) -> MagicMock:
    metrics = MagicMock(spec=OrdinalCriterionMetrics)
    metrics.configure_mock(
        name=name,
        n_samples=30,
        mae=0.2,
        rmse=0.25,
        exact_accuracy=0.5,
        adjacent_accuracy=0.9,
        weighted_kappa=0.3,
        spearman=SimpleNamespace(coefficient=srcc),
    )
    return metrics


def test_each_reference_row_names_its_kind_of_srcc():
    kinds = {name: kind for name, _mae, _srcc, kind, _pa, _dims in DEMO_MODULE.REFERENCE_ROWS}
    assert kinds == {
        "LLM-as-judge (v9), paper": "macro",
        "SciFigAlign (fine-tuned), paper": "overall",
    }


def test_each_reference_row_is_printed_beside_the_runs_comparable_srcc(monkeypatch, capsys):
    monkeypatch.setattr(DEMO_MODULE, "same_paper_pairs", lambda dataset: 0)
    per_dimension = [
        dimension(name, srcc)
        for name, srcc in zip(
            ("clarity", "relevance", "informativeness", "structure"), (0.1, 0.2, 0.3, 0.4)
        )
    ]
    metrics = SimpleNamespace(
        per_criterion=per_dimension,
        score_spearman=SimpleNamespace(coefficient=0.31),
        bootstrap=None,
        mean_kappa=0.3,
    )
    DEMO_MODULE.print_metrics(metrics, None)
    lines = capsys.readouterr().out.splitlines()
    (judge,) = [line for line in lines if line.strip().startswith("LLM-as-judge")]
    (fine_tuned,) = [line for line in lines if line.strip().startswith("SciFigAlign (fine")]
    # The macro SRCC is the mean of the four per-dimension coefficients.
    assert judge.split()[4:7] == ["0.198", "macro", "0.250"]
    assert fine_tuned.split()[4:7] == ["0.3088", "overall", "0.310"]
