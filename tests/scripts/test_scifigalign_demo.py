"""Tests for ``examples/scifigalign_demo.py``: its options and its reference rows.

The LLM judge is called once per score by default (``--llm-calls per_criterion``, the mode
the docs recommend); ``per_item`` stays available. A dry run sends nothing.

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
from typing import Any
from unittest.mock import MagicMock

import litellm
import pytest

from autorubric import Image, RubricDataset
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
CONVERTER = Path(__file__).resolve().parents[2] / "scripts" / "convert_scifigalign.py"

# A 2x1 RGB PNG (colour type 2, bit depth 8), as Pillow writes it.
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"{@\xe8\xdd\x00\x00\x00\x0fIDATx\x9cc\xfc\xff\xff?\x03\x03\x03\x00\x0e\xff\x02\xff\xb8"
    b"\xc5\x9e\x99\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _converter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_scifigalign_for_demo", CONVERTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_split(n_items: int = 3) -> RubricDataset:
    """A SciFigAlign-shaped split built in memory: the converter's rubric, figures that are
    a tiny PNG and a caption, and human scores. The real splits are not in the repository."""
    convert = _converter()
    data = RubricDataset(prompt="Rate this figure.", rubric=convert.build_rubric())
    for i in range(n_items):
        labels = [
            convert.option_label(d, 1 + (i + j) % 5) for j, d in enumerate(convert.DIMENSIONS)
        ]
        data.add_item(
            submission=[Image(TINY_PNG), f"Figure {i}: a caption."],
            description=f"ICLR_2024_{i:05d} fig_01",
            ground_truth=labels,
        )
    return data


class TestHowTheLLMJudgeIsCalled:
    def test_per_criterion_by_default(self):
        assert DEMO_MODULE.parse_args([]).llm_calls == "per_criterion"

    def test_per_item_stays_available(self):
        assert DEMO_MODULE.parse_args(["--llm-calls", "per_item"]).llm_calls == "per_item"

    def test_the_help_names_the_default(self, capsys):
        with pytest.raises(SystemExit):
            DEMO_MODULE.parse_args(["--help"])
        help_text = " ".join(capsys.readouterr().out.split())
        assert "one call per score (default)" in help_text

    def test_a_decision_model_takes_no_mode(self):
        assert DEMO_MODULE.parse_args(["--judge", "jev"]).llm_calls is None

    @pytest.mark.asyncio
    async def test_a_dry_run_sends_nothing(self, monkeypatch, refuse_network, capsys):
        """The default dry run estimates one call per figure and score, and sends nothing:
        every LiteLLM call and every connection raises. Async, so the network is refused
        once the test's event loop exists (``refuse_network``)."""
        split = _small_split()
        monkeypatch.setattr(DEMO_MODULE, "load_dataset", lambda name: split)

        def no_call(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the dry run called the LLM")

        monkeypatch.setattr(litellm, "completion", no_call)
        monkeypatch.setattr(litellm, "acompletion", no_call)
        refuse_network("the dry run reached the network")

        DEMO_MODULE.main([])

        out = capsys.readouterr().out
        assert "(per_criterion)" in out
        assert "per_criterion (12 calls)" in out  # 3 figures x 4 scores
        assert "Nothing was sent." in out


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
