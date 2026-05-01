"""Tests for meta-rubric evaluation entrypoints and judgment schema."""

from __future__ import annotations

from autorubric.dataset import DataItem, RubricDataset
from autorubric.graders.criterion_grader import FewShotConfig
from autorubric.meta import get_in_context_meta_rubric, get_standalone_meta_rubric
from autorubric.meta._evaluate import MetaCriterionJudgment, _resolve_few_shot
from autorubric.types import CriterionVerdict


class TestMetaCriterionJudgmentSchema:
    """Schema-level tests for the LLM-facing judgment model."""

    def test_evidence_quote_defaults_to_none(self) -> None:
        judgment = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="ok",
        )
        assert judgment.evidence_quote is None

    def test_evidence_quote_accepts_string(self) -> None:
        judgment = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="anti-pattern detected",
            evidence_quote="The summary is clear and sufficiently detailed.",
        )
        assert judgment.evidence_quote == "The summary is clear and sufficiently detailed."

    def test_affected_criteria_and_evidence_quote_coexist(self) -> None:
        judgment = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="bundles three checks",
            affected_criteria=[2, 4],
            evidence_quote="clear AND concise AND accurate",
        )
        assert judgment.affected_criteria == [2, 4]
        assert judgment.evidence_quote == "clear AND concise AND accurate"

    def test_field_appears_in_pydantic_schema(self) -> None:
        """Structured-output description is the LLM contract; pin its presence."""
        schema = MetaCriterionJudgment.model_json_schema()
        assert "evidence_quote" in schema["properties"]
        prop = schema["properties"]["evidence_quote"]
        assert prop.get("description"), "evidence_quote must carry a description for the LLM"


# ============================================================================
# Meta-rubric structure and content
# ============================================================================


_DROPPED_CRITERIA = {
    "boundary_clarity",
    "well_defined_options",
    "consistent_granularity",
    "circular_tautological",
    "verbosity_rewarding",
    "vague_wording",
    "boundary_ambiguity",
}


class TestMetaRubricStructure:
    """Pin the post-cleanup shape of the shipped meta-rubrics."""

    def test_standalone_count(self) -> None:
        rubric = get_standalone_meta_rubric()
        assert len(rubric.rubric) == 22

    def test_in_context_count(self) -> None:
        rubric = get_in_context_meta_rubric()
        assert len(rubric.rubric) == 31

    def test_standalone_drops(self) -> None:
        rubric = get_standalone_meta_rubric()
        names = {c.name for c in rubric.rubric}
        assert _DROPPED_CRITERIA.isdisjoint(names), (
            f"dropped criteria still present: {_DROPPED_CRITERIA & names}"
        )

    def test_in_context_drops(self) -> None:
        rubric = get_in_context_meta_rubric()
        names = {c.name for c in rubric.rubric}
        assert _DROPPED_CRITERIA.isdisjoint(names), (
            f"dropped criteria still present: {_DROPPED_CRITERIA & names}"
        )

    def test_standalone_has_unanchored_subjectivity(self) -> None:
        rubric = get_standalone_meta_rubric()
        names = {c.name for c in rubric.rubric}
        assert "unanchored_subjectivity" in names

    def test_in_context_has_unanchored_subjectivity(self) -> None:
        rubric = get_in_context_meta_rubric()
        names = {c.name for c in rubric.rubric}
        assert "unanchored_subjectivity" in names

    def test_unanchored_subjectivity_is_anti_pattern(self) -> None:
        """The merger preserves the source criteria's negative weight."""
        rubric = get_standalone_meta_rubric()
        crit = next(c for c in rubric.rubric if c.name == "unanchored_subjectivity")
        assert crit.weight == -8


# ============================================================================
# New anti-pattern criteria (Commit D)
# ============================================================================


class TestNewAntiPatternCriteria:
    """The three failure-pattern criteria added in Commit D."""

    def test_unanchored_grounding_in_both(self) -> None:
        for rubric in (get_standalone_meta_rubric(), get_in_context_meta_rubric()):
            names = {c.name for c in rubric.rubric}
            assert "unanchored_grounding" in names
            crit = next(c for c in rubric.rubric if c.name == "unanchored_grounding")
            assert crit.weight == -8

    def test_proxy_gameable_in_both(self) -> None:
        for rubric in (get_standalone_meta_rubric(), get_in_context_meta_rubric()):
            names = {c.name for c in rubric.rubric}
            assert "proxy_gameable" in names
            crit = next(c for c in rubric.rubric if c.name == "proxy_gameable")
            assert crit.weight == -8

    def test_over_constrained_only_in_context(self) -> None:
        standalone_names = {c.name for c in get_standalone_meta_rubric().rubric}
        in_context_names = {c.name for c in get_in_context_meta_rubric().rubric}
        assert "over_constrained" not in standalone_names, (
            "over_constrained requires task context; should not be in standalone"
        )
        assert "over_constrained" in in_context_names

    def test_new_criteria_carry_detection_procedure_inline(self) -> None:
        """Detection procedures live inline in the requirement text (not a schema field)."""
        rubric = get_in_context_meta_rubric()
        for name in ("unanchored_grounding", "proxy_gameable", "over_constrained"):
            crit = next(c for c in rubric.rubric if c.name == name)
            assert "manifestations" in crit.requirement.lower() or any(
                marker in crit.requirement
                for marker in ("(A)", "(B)", "Manifestations:")
            ), f"{name} requirement should enumerate sub-cases"


# ============================================================================
# Inline disambiguation appendices (Commit D)
# ============================================================================


class TestDisambiguationAppendices:
    """Cross-references between criteria are appended to requirement text."""

    def test_irrelevant_criteria_points_to_over_constrained(self) -> None:
        rubric = get_in_context_meta_rubric()
        crit = next(c for c in rubric.rubric if c.name == "irrelevant_criteria")
        assert "over_constrained" in crit.requirement

    def test_deterministic_assessability_points_to_unanchored_grounding(self) -> None:
        for rubric in (get_standalone_meta_rubric(), get_in_context_meta_rubric()):
            crit = next(c for c in rubric.rubric if c.name == "deterministic_assessability")
            assert "unanchored_grounding" in crit.requirement

    def test_distinguishes_quality_points_to_discrimination_evaluator(self) -> None:
        rubric = get_in_context_meta_rubric()
        crit = next(c for c in rubric.rubric if c.name == "distinguishes_quality")
        assert "evaluate_rubric_discrimination" in crit.requirement

    def test_missing_critical_points_to_unanchored_grounding(self) -> None:
        rubric = get_in_context_meta_rubric()
        crit = next(c for c in rubric.rubric if c.name == "missing_critical")
        assert "unanchored_grounding" in crit.requirement


# ============================================================================
# reasonable_count text update (Open Decision §1)
# ============================================================================


class TestReasonableCountUpdate:
    """The criterion text now acknowledges meta-instruments may exceed 15 criteria."""

    def test_standalone_text_mentions_meta_rubric(self) -> None:
        rubric = get_standalone_meta_rubric()
        crit = next(c for c in rubric.rubric if c.name == "reasonable_count")
        assert "meta-rubric" in crit.requirement.lower()

    def test_in_context_text_mentions_meta_rubric(self) -> None:
        rubric = get_in_context_meta_rubric()
        crit = next(c for c in rubric.rubric if c.name == "reasonable_count")
        assert "meta-rubric" in crit.requirement.lower()


# ============================================================================
# Few-shot wiring (Commit E)
# ============================================================================


class TestResolveFewShot:
    """Unit tests for the _resolve_few_shot helper."""

    def test_both_none_returns_both_none(self, tmp_path) -> None:
        bundled = tmp_path / "missing.json"  # not created
        examples, fsc = _resolve_few_shot(None, None, bundled)
        assert examples is None
        assert fsc is None

    def test_user_supplied_examples_get_default_config(self, tmp_path) -> None:
        meta = get_standalone_meta_rubric()
        custom = RubricDataset(
            prompt="evaluate this rubric",
            rubric=meta,
            items=[
                DataItem(
                    submission="x",
                    description="dummy",
                    ground_truth=[CriterionVerdict.MET] * len(meta.rubric),
                )
            ],
        )
        examples, fsc = _resolve_few_shot(custom, None, tmp_path / "missing.json")
        assert examples is custom
        assert isinstance(fsc, FewShotConfig)
        assert fsc.n_examples == 3
        assert fsc.balance_verdicts is True
        assert fsc.include_reason is True

    def test_config_without_examples_loads_bundled_when_present(self, tmp_path) -> None:
        meta = get_standalone_meta_rubric()
        bundled = tmp_path / "examples.json"
        ds = RubricDataset(
            prompt="evaluate this rubric",
            rubric=meta,
            items=[
                DataItem(
                    submission="x",
                    description="dummy",
                    ground_truth=[CriterionVerdict.MET] * len(meta.rubric),
                )
            ],
        )
        ds.to_file(str(bundled))

        custom_config = FewShotConfig(n_examples=2, balance_verdicts=False)
        examples, fsc = _resolve_few_shot(None, custom_config, bundled)
        assert examples is not None
        assert len(examples.items) == 1
        assert fsc is custom_config  # caller's config preserved

    def test_config_without_examples_disables_when_bundle_missing(self, tmp_path) -> None:
        bundled = tmp_path / "missing.json"  # not created
        examples, fsc = _resolve_few_shot(None, FewShotConfig(), bundled)
        # Graceful: no error, just no few-shot.
        assert examples is None
        assert fsc is None

    def test_user_examples_override_bundled(self, tmp_path) -> None:
        """Custom dataset wins even if bundled file also exists."""
        meta = get_standalone_meta_rubric()
        bundled = tmp_path / "examples.json"
        bundled_ds = RubricDataset(
            prompt="evaluate this rubric",
            rubric=meta,
            items=[
                DataItem(
                    submission="from_bundle",
                    description="bundled item",
                    ground_truth=[CriterionVerdict.MET] * len(meta.rubric),
                )
            ],
        )
        bundled_ds.to_file(str(bundled))

        custom = RubricDataset(
            prompt="evaluate this rubric",
            rubric=meta,
            items=[
                DataItem(
                    submission="from_caller",
                    description="caller item",
                    ground_truth=[CriterionVerdict.UNMET] * len(meta.rubric),
                )
            ],
        )
        examples, _fsc = _resolve_few_shot(custom, None, bundled)
        assert examples is custom
        assert examples.items[0].submission == "from_caller"
