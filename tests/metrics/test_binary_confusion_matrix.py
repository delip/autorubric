"""Tests for the ConfusionMatrix type's derived cells and rates.

These pin the binary positive/negative layout (``["MET", "UNMET"]``), the derived
tp/fp/fn/tn cells, the undefined->None rate convention, and the guard that refuses to
expose a single positive class on a larger (e.g. 3x3) matrix.
"""

import pytest

from autorubric.metrics import ConfusionMatrix


class TestConfusionMatrixShape:
    """Basic shape helpers (matrix/labels, n_classes, total)."""

    def test_n_classes_and_total(self):
        cm = ConfusionMatrix(matrix=[[1, 2], [3, 4]], labels=["MET", "UNMET"])
        assert cm.n_classes == 2
        assert cm.total == 10

    def test_three_class_n_classes_and_total(self):
        cm = ConfusionMatrix(
            matrix=[[1, 0, 0], [0, 2, 1], [0, 0, 3]],
            labels=["A", "B", "NA"],
        )
        assert cm.n_classes == 3
        assert cm.total == 7


class TestBinaryDerivedCells:
    """tp/fp/fn/tn on a binary MET/UNMET matrix (rows=true, cols=pred)."""

    def test_tp_fp_fn_tn_positions(self):
        # rows=true, cols=pred; labels[0]="MET" (positive), labels[1]="UNMET" (negative)
        #   true MET, pred MET  -> tp = matrix[0][0]
        #   true MET, pred UNMET-> fn = matrix[0][1]
        #   true UNMET, pred MET-> fp = matrix[1][0]
        #   true UNMET, pred UNMET-> tn = matrix[1][1]
        cm = ConfusionMatrix(matrix=[[5, 2], [3, 7]], labels=["MET", "UNMET"])
        assert cm.tp == 5
        assert cm.fn == 2
        assert cm.fp == 3
        assert cm.tn == 7

    def test_precision_recall(self):
        cm = ConfusionMatrix(matrix=[[5, 2], [3, 7]], labels=["MET", "UNMET"])
        # precision = tp / (tp + fp) = 5 / 8
        assert cm.precision == pytest.approx(5 / 8)
        # recall = tp / (tp + fn) = 5 / 7
        assert cm.recall == pytest.approx(5 / 7)

    def test_fpr_fnr(self):
        cm = ConfusionMatrix(matrix=[[5, 2], [3, 7]], labels=["MET", "UNMET"])
        # fpr = fp / (fp + tn) = 3 / 10
        assert cm.fpr == pytest.approx(3 / 10)
        # fnr = fn / (fn + tp) = 2 / 7
        assert cm.fnr == pytest.approx(2 / 7)


class TestRatesUndefinedToNone:
    """Each rate is None (never a fabricated 0.0) when its denominator is zero."""

    def test_fpr_none_when_no_true_negatives(self):
        # No UNMET ground truth at all -> fp + tn == 0 -> fpr undefined -> None.
        cm = ConfusionMatrix(matrix=[[4, 1], [0, 0]], labels=["MET", "UNMET"])
        assert cm.fpr is None

    def test_fnr_none_when_no_true_positives(self):
        # No MET ground truth at all -> fn + tp == 0 -> fnr undefined -> None.
        cm = ConfusionMatrix(matrix=[[0, 0], [2, 3]], labels=["MET", "UNMET"])
        assert cm.fnr is None

    def test_precision_none_when_nothing_predicted_met(self):
        # No MET predictions -> tp + fp == 0 -> precision undefined -> None.
        cm = ConfusionMatrix(matrix=[[0, 4], [0, 5]], labels=["MET", "UNMET"])
        assert cm.precision is None

    def test_recall_none_when_no_true_met(self):
        # No MET ground truth -> tp + fn == 0 -> recall undefined -> None.
        cm = ConfusionMatrix(matrix=[[0, 0], [2, 3]], labels=["MET", "UNMET"])
        assert cm.recall is None

    def test_all_rates_none_on_empty_matrix(self):
        cm = ConfusionMatrix(matrix=[[0, 0], [0, 0]], labels=["MET", "UNMET"])
        assert cm.fpr is None
        assert cm.fnr is None
        assert cm.precision is None
        assert cm.recall is None
        assert cm.total == 0


class TestBinaryGuard:
    """The binary-only cells raise ValueError on a non-binary or non-MET-positive matrix."""

    def test_three_by_three_tp_raises(self):
        cm = ConfusionMatrix(
            matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
            labels=["A", "B", "C"],
        )
        with pytest.raises(ValueError):
            _ = cm.tp

    def test_three_by_three_fpr_raises(self):
        cm = ConfusionMatrix(
            matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
            labels=["MET", "UNMET", "CANNOT_ASSESS"],
        )
        with pytest.raises(ValueError):
            _ = cm.fpr

    def test_two_by_two_wrong_positive_label_raises(self):
        # 2x2 but labels[0] != "MET" -> no positive=MET layout -> raise.
        cm = ConfusionMatrix(matrix=[[1, 2], [3, 4]], labels=["UNMET", "MET"])
        with pytest.raises(ValueError):
            _ = cm.tp


class TestFrozen:
    """ConfusionMatrix is frozen (immutable)."""

    def test_cannot_mutate(self):
        from pydantic import ValidationError

        cm = ConfusionMatrix(matrix=[[1, 0], [0, 1]], labels=["MET", "UNMET"])
        with pytest.raises(ValidationError):
            cm.labels = ["X", "Y"]  # type: ignore[misc]
