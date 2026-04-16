"""Unified criterion-based grader with compositional few-shot and ensemble support."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from autorubric.graders.base import Grader
from autorubric.llm import GenerateResult, LLMClient, LLMConfig
from autorubric.prompts import (
    FEW_SHOT_SYSTEM_PROMPT_ADDITION,
    GRADER_SYSTEM_PROMPT_DEFAULT,
    MULTI_CHOICE_FEW_SHOT_ADDITION,
    MULTI_CHOICE_SYSTEM_PROMPT,
    build_few_shot_user_prompt,
    build_multi_choice_few_shot_user_prompt,
    build_multi_choice_user_prompt,
    build_user_prompt,
)
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    AggregationStrategy,
    CannotAssessConfig,
    CannotAssessStrategy,
    Criterion,
    CriterionJudgment,
    CriterionReport,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    FewShotConfig,
    FewShotExample,
    JudgeVote,
    LengthPenalty,
    MultiChoiceJudgeVote,
    MultiChoiceJudgment,
    MultiChoiceVerdict,
    NominalAggregation,
    OrdinalAggregation,
    TokenUsage,
)

if TYPE_CHECKING:
    from autorubric.dataset import RubricDataset

logger = logging.getLogger(__name__)


def _derive_shuffle_rng(
    master_seed: int,
    item_key: str,
    criterion_idx: int,
    judge_id: str,
) -> random.Random:
    """Derive a deterministic, concurrency-safe RNG for option shuffling.

    Produces a unique random.Random instance for each combination of
    master seed, item content, criterion, and judge.
    """
    key = f"{master_seed}:{item_key}:{criterion_idx}:{judge_id}"
    derived = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**31)
    return random.Random(derived)


@dataclass
class JudgeSpec:
    """Specification for a single judge in an ensemble.

    Attributes:
        llm_config: Configuration for this judge's LLM.
        judge_id: Unique identifier for this judge (e.g., "gpt-4", "claude-sonnet").
        weight: Voting weight for weighted aggregation (default 1.0).
    """

    llm_config: LLMConfig
    judge_id: str
    weight: float = 1.0


@dataclass
class CriterionResult:
    """Result from evaluating a single criterion by a single judge."""

    report: CriterionReport
    usage: TokenUsage | None = None
    cost: float | None = None


@dataclass
class JudgeCriterionResults:
    """All criterion results from a single judge."""

    judge_id: str
    weight: float
    criterion_results: list[CriterionResult] = field(default_factory=list)

    @property
    def reports(self) -> list[CriterionReport]:
        return [r.report for r in self.criterion_results]

    @property
    def total_usage(self) -> TokenUsage | None:
        usages = [r.usage for r in self.criterion_results if r.usage is not None]
        return sum(usages, TokenUsage()) if usages else None

    @property
    def total_cost(self) -> float | None:
        costs = [r.cost for r in self.criterion_results if r.cost is not None]
        return sum(costs) if costs else None


class CriterionGrader(Grader):
    """Unified criterion-based grader with compositional few-shot and ensemble support.

    This grader evaluates each criterion independently and supports:
    - Single LLM mode (via llm_config)
    - Ensemble mode with multiple judges (via judges)
    - Few-shot prompting (via training_data + few_shot_config)

    All combinations work: single LLM, single + few-shot, ensemble, ensemble + few-shot.

    Parameters are orthogonal:
    - llm_config OR judges: Choose single-LLM or ensemble mode
    - training_data + few_shot_config: Enable few-shot prompting (applies to all judges)

    Example:
        >>> from autorubric import LLMConfig, FewShotConfig, RubricDataset
        >>> from autorubric.graders import CriterionGrader, JudgeSpec
        >>>
        >>> # Single LLM
        >>> grader = CriterionGrader(llm_config=LLMConfig(model="gemini/gemini-3-flash-preview"))
        >>>
        >>> # Single LLM + few-shot
        >>> train, test = dataset.split_train_test(n_train=100)
        >>> grader = CriterionGrader(
        ...     llm_config=LLMConfig(model="gemini/gemini-3-flash-preview"),
        ...     training_data=train,
        ...     few_shot_config=FewShotConfig(n_examples=3),
        ... )
        >>>
        >>> # Ensemble
        >>> grader = CriterionGrader(
        ...     judges=[
        ...         JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini"),
        ...         JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
        ...     ],
        ...     aggregation="majority",
        ... )
        >>>
        >>> # Ensemble + few-shot
        >>> grader = CriterionGrader(
        ...     judges=[JudgeSpec(...), JudgeSpec(...)],
        ...     aggregation="majority",
        ...     training_data=train,
        ...     few_shot_config=FewShotConfig(n_examples=3),
        ... )
    """

    def __init__(
        self,
        *,
        # Single LLM mode
        llm_config: LLMConfig | None = None,
        # Ensemble mode (overrides llm_config)
        judges: list[JudgeSpec] | None = None,
        aggregation: AggregationStrategy = "majority",
        # Multi-choice aggregation strategies
        ordinal_aggregation: OrdinalAggregation = "mean",
        nominal_aggregation: NominalAggregation = "mode",
        # Few-shot mode (orthogonal - applies to all judges)
        training_data: RubricDataset | None = None,
        few_shot_config: FewShotConfig | None = None,
        # Common parameters
        system_prompt: str | None = None,
        multi_choice_system_prompt: str | None = None,
        length_penalty: LengthPenalty | None = None,
        normalize: bool = True,
        cannot_assess_config: CannotAssessConfig | None = None,
        # Position bias mitigation
        shuffle_options: bool = True,
        # Reproducibility
        seed: int | None = None,
        # Structured output override for binary criteria
        binary_response_format: type[BaseModel] | None = None,
    ):
        """Initialize the criterion grader.

        Args:
            llm_config: Configuration for single-LLM mode. Mutually exclusive with judges.
            judges: List of JudgeSpec for ensemble mode. Mutually exclusive with llm_config.
            aggregation: Strategy for aggregating votes in ensemble mode (binary criteria).
            ordinal_aggregation: Strategy for aggregating ordinal multi-choice votes.
                Options: "mean", "median", "weighted_mean", "mode".
            nominal_aggregation: Strategy for aggregating nominal multi-choice votes.
                Options: "mode", "weighted_mode", "unanimous".
            training_data: Dataset for few-shot examples. If provided, enables few-shot prompting.
            few_shot_config: Configuration for few-shot example selection.
            system_prompt: Custom system prompt for binary criteria.
            multi_choice_system_prompt: Custom system prompt for multi-choice criteria.
            length_penalty: Optional length penalty configuration.
            normalize: If True, normalize score to [0, 1]. If False, return raw sum.
            cannot_assess_config: Configuration for handling CANNOT_ASSESS verdicts.
            shuffle_options: If True (default), randomize the order of multi-choice options
                presented to the LLM to mitigate position bias. Each judge/call sees a
                different random order, and responses are mapped back to original indices.
                Disable for deterministic behavior in tests.
            seed: Master seed for all non-LLM randomness (option shuffling, few-shot
                example selection). Auto-generated when None so that randomness is always
                pinned and reproducible. Inspect via the ``seed`` property after construction.
            binary_response_format: Pydantic model to use as the structured output schema
                for binary criterion judgments. Must be a subclass of (or compatible with)
                CriterionJudgment. If the model includes an ``affected_criteria`` field
                (list[int]), matching indices are injected as an ``[Affects: ...]`` tag
                into the reason string. Defaults to CriterionJudgment.

        Raises:
            ValueError: If neither llm_config nor judges is provided, or both are provided.
        """
        super().__init__(length_penalty=length_penalty, normalize=normalize)

        # Validate: must have either llm_config or judges, not both, not neither
        if llm_config is None and judges is None:
            raise ValueError("Must provide either llm_config or judges")
        if llm_config is not None and judges is not None:
            raise ValueError("Cannot provide both llm_config and judges")

        # Normalize to ensemble representation (single LLM = ensemble of 1)
        if llm_config is not None:
            self._judges = [JudgeSpec(llm_config=llm_config, judge_id="default", weight=1.0)]
        else:
            self._judges = judges  # type: ignore

        self._aggregation = aggregation
        self._ordinal_aggregation = ordinal_aggregation
        self._nominal_aggregation = nominal_aggregation
        self._training_data = training_data
        self._cannot_assess_config = cannot_assess_config or CannotAssessConfig()
        self._shuffle_options = shuffle_options
        self._seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        # Coordinate few-shot seed with master seed when unset
        fsc = few_shot_config or FewShotConfig()
        if fsc.seed is None and training_data is not None:
            fsc = dataclasses.replace(fsc, seed=self._seed)
        self._few_shot_config = fsc
        self._binary_response_format = binary_response_format or CriterionJudgment

        # Build system prompts (separate for binary and multi-choice)
        if system_prompt is None:
            self._system_prompt = GRADER_SYSTEM_PROMPT_DEFAULT
            if training_data is not None:
                self._system_prompt += FEW_SHOT_SYSTEM_PROMPT_ADDITION
        else:
            self._system_prompt = system_prompt

        if multi_choice_system_prompt is None:
            self._multi_choice_system_prompt = MULTI_CHOICE_SYSTEM_PROMPT
            if training_data is not None:
                self._multi_choice_system_prompt += MULTI_CHOICE_FEW_SHOT_ADDITION
        else:
            self._multi_choice_system_prompt = multi_choice_system_prompt

        # Create LLM clients for each judge
        self._clients = {
            judge.judge_id: LLMClient(judge.llm_config)
            for judge in self._judges
        }

        # Pre-compute few-shot examples if training data provided
        # Note: For multi-choice, examples are stored as (submission, selected_index, reason)
        self._criterion_examples: dict[int, list[FewShotExample]] = {}
        self._multi_choice_examples: dict[int, list[tuple[str, int, str | None]]] = {}
        if training_data is not None:
            self._prepare_examples()

    @property
    def is_ensemble(self) -> bool:
        """Whether this grader uses multiple judges."""
        return len(self._judges) > 1

    @property
    def has_few_shot(self) -> bool:
        """Whether this grader uses few-shot prompting."""
        return self._training_data is not None

    @property
    def seed(self) -> int:
        """The master seed governing all non-LLM randomness in this grader."""
        return self._seed

    # =========================================================================
    # Few-Shot Example Preparation
    # =========================================================================

    def _prepare_examples(self) -> None:
        """Pre-compute few-shot examples for each criterion."""
        if self._training_data is None:
            return

        n_criteria = self._training_data.num_criteria
        rubric_criteria = self._training_data.rubric.rubric if self._training_data.rubric else []

        for criterion_idx in range(n_criteria):
            criterion = rubric_criteria[criterion_idx] if criterion_idx < len(rubric_criteria) else None
            if criterion is not None and criterion.is_multi_choice:
                examples = self._select_multi_choice_examples(criterion_idx)
                self._multi_choice_examples[criterion_idx] = examples
            else:
                examples = self._select_examples_for_criterion(criterion_idx)
                self._criterion_examples[criterion_idx] = examples

    def _select_examples_for_criterion(self, criterion_idx: int) -> list[FewShotExample]:
        """Select stratified examples for a specific criterion."""
        if self._training_data is None:
            return []

        config = self._few_shot_config
        rng = random.Random(config.seed)

        # Group items by verdict for this criterion
        verdict_groups: dict[CriterionVerdict, list] = {
            CriterionVerdict.MET: [],
            CriterionVerdict.UNMET: [],
            CriterionVerdict.CANNOT_ASSESS: [],
        }

        for item in self._training_data:
            if item.ground_truth is None:
                continue
            verdict = item.ground_truth[criterion_idx]
            verdict_groups[verdict].append(item)

        available_verdicts = [v for v, items in verdict_groups.items() if items]
        if not available_verdicts:
            return []

        n_examples = config.n_examples

        if config.balance_verdicts and len(available_verdicts) > 1:
            return self._balanced_selection(verdict_groups, n_examples, criterion_idx, rng)
        else:
            all_items = [item for items in verdict_groups.values() for item in items]
            rng.shuffle(all_items)
            return [
                FewShotExample(
                    submission=item.submission,
                    verdict=item.ground_truth[criterion_idx],  # type: ignore
                    reason=None,
                )
                for item in all_items[:n_examples]
            ]

    def _balanced_pick(
        self,
        groups: dict,
        n_examples: int,
        rng: random.Random,
        transform,
        identity,
    ) -> list:
        """Select items with balanced distribution across non-empty groups.

        Divides slots evenly, assigns remainder to the first groups, then
        fills any remaining slots from unused items across all groups.

        Args:
            groups: Mapping from group key to list of items.
            n_examples: Total number of items to select.
            rng: Random instance for shuffling.
            transform: Convert an item to the output type.
            identity: Extract a hashable dedup key from an item.
        """
        results = []
        available = [(k, v) for k, v in groups.items() if v]
        if not available:
            return results

        base_per_group = n_examples // len(available)
        remainder = n_examples % len(available)

        used_ids: set = set()
        for i, (_key, items) in enumerate(available):
            pool = items.copy()
            rng.shuffle(pool)
            count = min(base_per_group + (1 if i < remainder else 0), len(pool))
            for item in pool[:count]:
                results.append(transform(item))
                used_ids.add(identity(item))

        if len(results) < n_examples:
            remaining = [
                item
                for _k, items in groups.items()
                for item in items
                if identity(item) not in used_ids
            ]
            rng.shuffle(remaining)
            for item in remaining[: n_examples - len(results)]:
                results.append(transform(item))

        return results

    def _balanced_selection(
        self,
        verdict_groups: dict[CriterionVerdict, list],
        n_examples: int,
        criterion_idx: int,
        rng: random.Random,
    ) -> list[FewShotExample]:
        """Select examples with balanced verdict distribution."""
        return self._balanced_pick(
            groups=verdict_groups,
            n_examples=n_examples,
            rng=rng,
            transform=lambda item: FewShotExample(
                submission=item.submission,
                verdict=item.ground_truth[criterion_idx],  # type: ignore
                reason=None,
            ),
            identity=lambda item: item.submission,
        )

    def _label_to_option_index(self, criterion_idx: int, label: str) -> int | None:
        """Map a ground truth label string to a 0-based option index."""
        if self._training_data is None or self._training_data.rubric is None:
            return None
        criteria = self._training_data.rubric.rubric
        if criterion_idx >= len(criteria):
            return None
        try:
            return criteria[criterion_idx].find_option_by_label(label)
        except ValueError:
            return None

    def _select_multi_choice_examples(
        self, criterion_idx: int
    ) -> list[tuple[str, int, str | None]]:
        """Select few-shot examples for a multi-choice criterion.

        Groups training items by their selected option index and balances
        across options when configured. Ground truth labels are converted
        to 0-based option indices.
        """
        if self._training_data is None:
            return []

        config = self._few_shot_config
        rng = random.Random(config.seed)

        # Group items by option index (converting label to index)
        option_groups: dict[int, list] = {}
        for item in self._training_data:
            if item.ground_truth is None:
                continue
            label = item.ground_truth[criterion_idx]
            idx = self._label_to_option_index(criterion_idx, label) if isinstance(label, str) else label
            if idx is None:
                continue
            option_groups.setdefault(idx, []).append((item, idx))

        available_options = [k for k, v in option_groups.items() if v]
        if not available_options:
            return []

        n_examples = config.n_examples

        if config.balance_verdicts and len(available_options) > 1:
            sorted_groups = {k: option_groups[k] for k in sorted(available_options)}
            return self._balanced_pick(
                groups=sorted_groups,
                n_examples=n_examples,
                rng=rng,
                transform=lambda pair: (pair[0].submission, pair[1], None),
                identity=lambda pair: pair[0].submission,
            )
        else:
            all_pairs = [(item, resolved_idx) for pairs in option_groups.values() for item, resolved_idx in pairs]
            rng.shuffle(all_pairs)
            return [
                (item.submission, resolved_idx, None)
                for item, resolved_idx in all_pairs[:n_examples]
            ]

    # =========================================================================
    # Single Criterion Evaluation
    # =========================================================================

    async def _judge_single_criterion(
        self,
        judge: JudgeSpec,
        criterion: Criterion,
        criterion_idx: int,
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> CriterionResult:
        """Judge a single criterion with a single judge.

        Handles both binary (MET/UNMET) and multi-choice criteria.
        """
        client = self._clients[judge.judge_id]

        # Dispatch to appropriate handler based on criterion type
        if criterion.is_multi_choice:
            return await self._judge_multi_choice_criterion(
                client, judge, criterion, criterion_idx, to_grade, query, reference_submission
            )
        else:
            return await self._judge_binary_criterion(
                client, judge, criterion, criterion_idx, to_grade, query, reference_submission
            )

    async def _judge_binary_criterion(
        self,
        client: LLMClient,
        judge: JudgeSpec,
        criterion: Criterion,
        criterion_idx: int,
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> CriterionResult:
        """Judge a binary (MET/UNMET) criterion."""
        examples = self._criterion_examples.get(criterion_idx, [])

        # Build prompt (with or without few-shot examples)
        if examples:
            user_prompt = build_few_shot_user_prompt(
                criterion=criterion,
                to_grade=to_grade,
                examples=examples,
                query=query,
                include_reason=self._few_shot_config.include_reason,
                reference_submission=reference_submission,
            )
        else:
            user_prompt = build_user_prompt(criterion, to_grade, query, reference_submission)

        try:
            result: GenerateResult = await client.generate(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                response_format=self._binary_response_format,
                return_result=True,
            )

            judgment = result.parsed
            reason = judgment.explanation

            # Inject [Affects: ...] tag if the response model includes affected_criteria
            if hasattr(judgment, "affected_criteria") and judgment.affected_criteria:
                tag = ", ".join(f"#{i}" for i in judgment.affected_criteria)
                reason = f"{reason} [Affects: {tag}]"

            # Inject [Cited: ...] tag if the response model includes evidence_cited
            if hasattr(judgment, "evidence_cited") and judgment.evidence_cited:
                cited_tag = ", ".join(judgment.evidence_cited)
                reason = f"{reason} [Cited: {cited_tag}]"

            report = CriterionReport(
                requirement=criterion.requirement,
                verdict=judgment.criterion_status,
                reason=reason,
                weight=criterion.weight,
                name=criterion.name,
                options=criterion.options,
                scale_type=criterion.scale_type,
                aggregation=criterion.aggregation,
            )
            return CriterionResult(report=report, usage=result.usage, cost=result.cost)

        except (ValidationError, Exception) as e:
            # Conservative default: worst case for each criterion type
            default_verdict = (
                CriterionVerdict.MET if criterion.weight < 0 else CriterionVerdict.UNMET
            )
            logger.warning(
                f"Error evaluating criterion '{criterion.requirement[:50]}...' "
                f"with judge '{judge.judge_id}': {e}"
            )
            report = CriterionReport(
                requirement=criterion.requirement,
                verdict=default_verdict,
                reason=f"Error parsing judge response: {str(e)}",
                weight=criterion.weight,
                name=criterion.name,
                options=criterion.options,
                scale_type=criterion.scale_type,
                aggregation=criterion.aggregation,
            )
            return CriterionResult(report=report, usage=None, cost=None)

    async def _judge_multi_choice_criterion(
        self,
        client: LLMClient,
        judge: JudgeSpec,
        criterion: Criterion,
        criterion_idx: int,
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> CriterionResult:
        """Judge a multi-choice criterion.

        If shuffle_options is enabled, options are presented to the LLM in a
        randomized order to mitigate position bias. The response is mapped back
        to the original option indices.
        """
        if criterion.options is None:
            raise ValueError("Multi-choice criterion must have options")

        # Shuffle options to mitigate position bias
        # shuffled_indices[shuffled_pos] = original_pos
        if self._shuffle_options:
            original_indices = list(range(len(criterion.options)))
            shuffled_indices = original_indices.copy()
            item_key = hashlib.sha256(to_grade.encode()).hexdigest()[:16]
            rng = _derive_shuffle_rng(self._seed, item_key, criterion_idx, judge.judge_id)
            rng.shuffle(shuffled_indices)

            # Create shuffled options list
            shuffled_options = [criterion.options[i] for i in shuffled_indices]

            # Create criterion with shuffled options for prompt building
            from autorubric.types import CriterionOption

            prompt_criterion = Criterion(
                weight=criterion.weight,
                requirement=criterion.requirement,
                name=criterion.name,
                options=shuffled_options,
                scale_type=criterion.scale_type,
                aggregation=criterion.aggregation,
            )
        else:
            shuffled_indices = list(range(len(criterion.options)))
            prompt_criterion = criterion

        examples = self._multi_choice_examples.get(criterion_idx, [])

        # Build prompt (with or without few-shot examples)
        if examples:
            # Note: If shuffling is enabled and few-shot examples are used,
            # we need to transform example indices to match shuffled order.
            # Create inverse mapping: original_to_shuffled[original_pos] = shuffled_pos
            original_to_shuffled = {orig: shuf for shuf, orig in enumerate(shuffled_indices)}
            transformed_examples = [
                (submission, original_to_shuffled[orig_idx], reason)
                for submission, orig_idx, reason in examples
            ]
            user_prompt = build_multi_choice_few_shot_user_prompt(
                criterion=prompt_criterion,
                to_grade=to_grade,
                examples=transformed_examples,
                query=query,
                include_reason=self._few_shot_config.include_reason,
                reference_submission=reference_submission,
            )
        else:
            user_prompt = build_multi_choice_user_prompt(
                prompt_criterion, to_grade, query, reference_submission
            )

        try:
            result: GenerateResult = await client.generate(
                system_prompt=self._multi_choice_system_prompt,
                user_prompt=user_prompt,
                response_format=MultiChoiceJudgment,
                return_result=True,
            )

            judgment: MultiChoiceJudgment = result.parsed

            # Convert 1-indexed response to 0-indexed (in shuffled space)
            shuffled_idx = judgment.selected_option - 1

            # Validate index is in range
            if shuffled_idx < 0 or shuffled_idx >= len(criterion.options):
                raise ValueError(
                    f"Selected option {judgment.selected_option} out of range "
                    f"[1, {len(criterion.options)}]"
                )

            # Map back from shuffled position to original index
            original_idx = shuffled_indices[shuffled_idx]

            selected_option = criterion.options[original_idx]
            multi_choice_verdict = MultiChoiceVerdict(
                selected_index=original_idx,
                selected_label=selected_option.label,
                value=selected_option.value,
                na=selected_option.na,
            )

            report = CriterionReport(
                requirement=criterion.requirement,
                verdict=None,  # Binary verdict is None for multi-choice
                multi_choice_verdict=multi_choice_verdict,
                reason=judgment.explanation,
                weight=criterion.weight,
                name=criterion.name,
                options=criterion.options,
                scale_type=criterion.scale_type,
                aggregation=criterion.aggregation,
                shuffle_order=shuffled_indices if self._shuffle_options else None,
            )
            return CriterionResult(report=report, usage=result.usage, cost=result.cost)

        except (ValidationError, Exception) as e:
            # Conservative default for multi-choice: select lowest value option
            # (or NA option if available)
            logger.warning(
                f"Error evaluating multi-choice criterion '{criterion.requirement[:50]}...' "
                f"with judge '{judge.judge_id}': {e}"
            )

            # Find worst-case option (lowest value, or NA)
            worst_idx = 0
            worst_value = criterion.options[0].value
            for i, opt in enumerate(criterion.options):
                if opt.na:
                    worst_idx = i
                    break
                if opt.value < worst_value:
                    worst_idx = i
                    worst_value = opt.value

            worst_option = criterion.options[worst_idx]
            multi_choice_verdict = MultiChoiceVerdict(
                selected_index=worst_idx,
                selected_label=worst_option.label,
                value=worst_option.value,
                na=worst_option.na,
            )

            report = CriterionReport(
                requirement=criterion.requirement,
                verdict=None,
                multi_choice_verdict=multi_choice_verdict,
                reason=f"Error parsing judge response: {str(e)}",
                weight=criterion.weight,
                name=criterion.name,
                options=criterion.options,
                scale_type=criterion.scale_type,
                aggregation=criterion.aggregation,
                shuffle_order=shuffled_indices if self._shuffle_options else None,
            )
            return CriterionResult(report=report, usage=None, cost=None)

    async def _judge_all_criteria_for_judge(
        self,
        judge: JudgeSpec,
        rubric: list[Criterion],
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> JudgeCriterionResults:
        """Evaluate all criteria for a single judge (parallel per criterion)."""
        tasks = [
            self._judge_single_criterion(judge, criterion, idx, to_grade, query, reference_submission)
            for idx, criterion in enumerate(rubric)
        ]
        results = await asyncio.gather(*tasks)
        return JudgeCriterionResults(
            judge_id=judge.judge_id,
            weight=judge.weight,
            criterion_results=list(results),
        )

    # =========================================================================
    # Judge and Aggregate (Grader Interface)
    # =========================================================================

    async def judge(
        self,
        to_grade: str,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> list[JudgeCriterionResults]:
        """Judge all criteria with all judges (parallel across judges)."""
        tasks = [
            self._judge_all_criteria_for_judge(judge, rubric, to_grade, query, reference_submission)
            for judge in self._judges
        ]
        return list(await asyncio.gather(*tasks))

    async def aggregate(
        self, judge_results: list[JudgeCriterionResults], *, normalize: bool = True
    ) -> EnsembleEvaluationReport:
        """Aggregate results from all judges into final report.

        Handles both binary and multi-choice criteria:
        - Binary: Uses JudgeVote and _aggregate_votes()
        - Multi-choice: Uses MultiChoiceJudgeVote and _aggregate_multi_choice_votes()
        """
        if not judge_results:
            return EnsembleEvaluationReport(
                score=0.0,
                raw_score=0.0,
                llm_raw_score=0.0,
                error="No judge results to aggregate",
            )

        n_criteria = len(judge_results[0].criterion_results)

        # Build ensemble criterion reports
        ensemble_reports: list[EnsembleCriterionReport] = []
        for criterion_idx in range(n_criteria):
            # Get criterion from first judge's result
            first_cr = judge_results[0].criterion_results[criterion_idx]
            criterion_report = first_cr.report

            if criterion_report.is_multi_choice:
                # Multi-choice: build MultiChoiceJudgeVote list
                mc_votes: list[MultiChoiceJudgeVote] = []
                for judge_result in judge_results:
                    cr = judge_result.criterion_results[criterion_idx]
                    mcv = cr.report.multi_choice_verdict
                    if mcv is not None:
                        mc_votes.append(
                            MultiChoiceJudgeVote(
                                judge_id=judge_result.judge_id,
                                selected_index=mcv.selected_index,
                                selected_label=mcv.selected_label,
                                value=mcv.value,
                                reason=cr.report.reason,
                                weight=judge_result.weight,
                                na=mcv.na,
                                shuffle_order=cr.report.shuffle_order,
                            )
                        )

                # Aggregate multi-choice votes
                final_mc_verdict, final_reason = self._aggregate_multi_choice_votes(
                    mc_votes, criterion_report
                )

                ensemble_reports.append(
                    EnsembleCriterionReport(
                        criterion=Criterion(
                            weight=criterion_report.weight,
                            requirement=criterion_report.requirement,
                            name=criterion_report.name,
                            options=criterion_report.options,
                            scale_type=criterion_report.scale_type,
                            aggregation=criterion_report.aggregation,
                        ),
                        final_verdict=None,  # Binary verdict is None for multi-choice
                        final_reason=final_reason,
                        votes=[],  # Binary votes empty for multi-choice
                        final_multi_choice_verdict=final_mc_verdict,
                        multi_choice_votes=mc_votes,
                    )
                )
            else:
                # Binary: build JudgeVote list
                votes: list[JudgeVote] = []
                for judge_result in judge_results:
                    cr = judge_result.criterion_results[criterion_idx]
                    votes.append(
                        JudgeVote(
                            judge_id=judge_result.judge_id,
                            verdict=cr.report.verdict,
                            reason=cr.report.reason,
                            weight=judge_result.weight,
                        )
                    )

                final_verdict, final_reason = self._aggregate_votes(votes)

                ensemble_reports.append(
                    EnsembleCriterionReport(
                        criterion=Criterion(
                            weight=criterion_report.weight,
                            requirement=criterion_report.requirement,
                            name=criterion_report.name,
                        ),
                        final_verdict=final_verdict,
                        final_reason=final_reason,
                        votes=votes,
                    )
                )

        # Calculate per-judge scores
        judge_scores = {}
        for judge_result in judge_results:
            score = self._calculate_score_from_reports(judge_result.reports, normalize)
            judge_scores[judge_result.judge_id] = score

        # Calculate final score from aggregated verdicts
        final_reports = []
        for er in ensemble_reports:
            if er.final_multi_choice_verdict is not None:
                # Multi-choice criterion
                final_reports.append(
                    CriterionReport(
                        weight=er.criterion.weight,
                        requirement=er.criterion.requirement,
                        name=er.criterion.name,
                        options=er.criterion.options,
                        scale_type=er.criterion.scale_type,
                        aggregation=er.criterion.aggregation,
                        verdict=None,  # Binary verdict is None
                        multi_choice_verdict=er.final_multi_choice_verdict,
                        reason=er.final_reason,
                    )
                )
            else:
                # Binary criterion
                final_reports.append(
                    CriterionReport(
                        weight=er.criterion.weight,
                        requirement=er.criterion.requirement,
                        name=er.criterion.name,
                        verdict=er.final_verdict,
                        reason=er.final_reason,
                    )
                )
        final_score = self._calculate_score_from_reports(final_reports, normalize)
        raw_score = self._calculate_score_from_reports(final_reports, normalize=False)

        # Calculate agreement
        mean_agreement = (
            sum(er.agreement for er in ensemble_reports) / len(ensemble_reports)
            if ensemble_reports
            else 1.0
        )

        # Count CANNOT_ASSESS (binary) and NA (multi-choice)
        cannot_assess_count = sum(
            1
            for er in ensemble_reports
            if (er.final_verdict == CriterionVerdict.CANNOT_ASSESS)
            or (er.final_multi_choice_verdict is not None and er.final_multi_choice_verdict.na)
        )

        # Aggregate token usage and cost
        total_usage = TokenUsage()
        total_cost = 0.0
        for jr in judge_results:
            if jr.total_usage:
                total_usage = total_usage + jr.total_usage
            if jr.total_cost:
                total_cost += jr.total_cost

        return EnsembleEvaluationReport(
            score=final_score,
            raw_score=raw_score,
            llm_raw_score=raw_score,
            report=ensemble_reports,
            judge_scores=judge_scores,
            mean_agreement=mean_agreement,
            cannot_assess_count=cannot_assess_count,
            token_usage=total_usage if total_usage.total_tokens > 0 else None,
            completion_cost=total_cost if total_cost > 0 else None,
        )

    def _aggregate_votes(
        self, votes: list[JudgeVote]
    ) -> tuple[CriterionVerdict, str]:
        """Aggregate votes from multiple judges into a single verdict."""
        if not votes:
            return CriterionVerdict.CANNOT_ASSESS, "No votes"

        # Filter out CANNOT_ASSESS for aggregation (unless all are CANNOT_ASSESS)
        assessable_votes = [v for v in votes if v.verdict != CriterionVerdict.CANNOT_ASSESS]
        if not assessable_votes:
            return CriterionVerdict.CANNOT_ASSESS, "All judges could not assess"

        met_weight = sum(v.weight for v in assessable_votes if v.verdict == CriterionVerdict.MET)
        unmet_weight = sum(v.weight for v in assessable_votes if v.verdict == CriterionVerdict.UNMET)
        total_weight = met_weight + unmet_weight

        if self._aggregation == "majority":
            verdict = CriterionVerdict.MET if met_weight > unmet_weight else CriterionVerdict.UNMET
        elif self._aggregation == "weighted":
            verdict = CriterionVerdict.MET if met_weight > unmet_weight else CriterionVerdict.UNMET
        elif self._aggregation == "unanimous":
            verdict = CriterionVerdict.MET if unmet_weight == 0 else CriterionVerdict.UNMET
        elif self._aggregation == "any":
            verdict = CriterionVerdict.MET if met_weight > 0 else CriterionVerdict.UNMET
        else:
            verdict = CriterionVerdict.MET if met_weight > unmet_weight else CriterionVerdict.UNMET

        # Combine reasons
        reasons = [f"{v.judge_id}: {v.reason}" for v in votes]
        combined_reason = " | ".join(reasons)

        return verdict, combined_reason

    def _aggregate_multi_choice_votes(
        self,
        votes: list[MultiChoiceJudgeVote],
        criterion: CriterionReport,
    ) -> tuple[AggregatedMultiChoiceVerdict, str]:
        """Aggregate multi-choice votes from multiple judges.

        Uses ordinal_aggregation for ordinal scale criteria and
        nominal_aggregation for nominal scale criteria.

        Args:
            votes: List of multi-choice votes from all judges.
            criterion: The criterion report (to access options and scale_type).

        Returns:
            Tuple of (aggregated verdict, combined reason).
        """
        if not votes:
            # Return NA verdict if no votes
            if criterion.options:
                na_opt = next((o for o in criterion.options if o.na), None)
                if na_opt:
                    idx = criterion.options.index(na_opt)
                    return (
                        AggregatedMultiChoiceVerdict(
                            selected_index=idx,
                            selected_label=na_opt.label,
                            value=na_opt.value,
                            na=True,
                            aggregated_value=0.0,
                        ),
                        "No votes",
                    )
            # Fallback: return first option as worst case
            return (
                AggregatedMultiChoiceVerdict(
                    selected_index=0,
                    selected_label=criterion.options[0].label if criterion.options else "",
                    value=criterion.options[0].value if criterion.options else 0.0,
                    na=criterion.options[0].na if criterion.options else False,
                    aggregated_value=0.0,
                ),
                "No votes",
            )

        # Filter out NA votes for aggregation (unless all are NA)
        assessable_votes = [v for v in votes if not v.na]
        if not assessable_votes:
            # All votes are NA
            na_vote = votes[0]
            reasons = [f"{v.judge_id}: {v.reason}" for v in votes]
            return (
                AggregatedMultiChoiceVerdict(
                    selected_index=na_vote.selected_index,
                    selected_label=na_vote.selected_label,
                    value=na_vote.value,
                    na=True,
                    aggregated_value=na_vote.value,
                ),
                " | ".join(reasons),
            )

        # Check for per-criterion aggregation override
        agg_strategy = criterion.aggregation
        scale_type = criterion.scale_type

        # Determine which aggregation to use
        if scale_type == "ordinal":
            agg = agg_strategy or self._ordinal_aggregation
            result = self._aggregate_ordinal_votes(assessable_votes, criterion.options or [], agg)
        else:  # nominal
            agg = agg_strategy or self._nominal_aggregation
            result = self._aggregate_nominal_votes(assessable_votes, criterion.options or [], agg)

        # Combine reasons
        reasons = [f"{v.judge_id}: {v.reason}" for v in votes]
        combined_reason = " | ".join(reasons)

        return result, combined_reason

    def _aggregate_ordinal_votes(
        self,
        votes: list[MultiChoiceJudgeVote],
        options: list,  # list[CriterionOption]
        strategy: str,
    ) -> AggregatedMultiChoiceVerdict:
        """Aggregate ordinal multi-choice votes.

        Strategies:
        - mean: Average of score values, snap to nearest option
        - median: Median of score values, snap to nearest option
        - weighted_mean: Weighted average by judge weight
        - mode: Most common selection
        """
        from collections import Counter

        values = [v.value for v in votes]
        weights = [v.weight for v in votes]

        if strategy == "mean":
            aggregated_value = sum(values) / len(values)
        elif strategy == "median":
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                aggregated_value = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
            else:
                aggregated_value = sorted_values[n // 2]
        elif strategy == "weighted_mean":
            total_weight = sum(weights)
            if total_weight > 0:
                aggregated_value = sum(v * w for v, w in zip(values, weights)) / total_weight
            else:
                aggregated_value = sum(values) / len(values)
        elif strategy == "mode":
            # For mode on ordinal, we use the index
            indices = [v.selected_index for v in votes]
            most_common_idx = Counter(indices).most_common(1)[0][0]
            selected_option = options[most_common_idx]
            return AggregatedMultiChoiceVerdict(
                selected_index=most_common_idx,
                selected_label=selected_option.label,
                value=selected_option.value,
                na=selected_option.na,
                aggregated_value=selected_option.value,  # For mode, no continuous value
            )
        else:
            # Default to mean
            aggregated_value = sum(values) / len(values)

        # Snap to nearest option by value
        closest_idx = 0
        closest_diff = float("inf")
        for i, opt in enumerate(options):
            if not opt.na:
                diff = abs(opt.value - aggregated_value)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_idx = i

        selected_option = options[closest_idx]
        return AggregatedMultiChoiceVerdict(
            selected_index=closest_idx,
            selected_label=selected_option.label,
            value=selected_option.value,
            na=selected_option.na,
            aggregated_value=aggregated_value,  # Store continuous value before snap
        )

    def _aggregate_nominal_votes(
        self,
        votes: list[MultiChoiceJudgeVote],
        options: list,  # list[CriterionOption]
        strategy: str,
    ) -> AggregatedMultiChoiceVerdict:
        """Aggregate nominal multi-choice votes.

        Strategies:
        - mode: Most common selection (majority)
        - weighted_mode: Weight votes by judge weight
        - unanimous: All judges must agree (else pick most common)
        """
        from collections import Counter

        indices = [v.selected_index for v in votes]

        if strategy == "mode":
            most_common_idx = Counter(indices).most_common(1)[0][0]
        elif strategy == "weighted_mode":
            # Accumulate weights per index
            weight_per_idx: dict[int, float] = {}
            for v in votes:
                weight_per_idx[v.selected_index] = weight_per_idx.get(v.selected_index, 0.0) + v.weight
            most_common_idx = max(weight_per_idx, key=weight_per_idx.get)  # type: ignore
        elif strategy == "unanimous":
            unique_indices = set(indices)
            if len(unique_indices) == 1:
                most_common_idx = indices[0]
            else:
                # Not unanimous, fall back to mode
                most_common_idx = Counter(indices).most_common(1)[0][0]
        else:
            # Default to mode
            most_common_idx = Counter(indices).most_common(1)[0][0]

        selected_option = options[most_common_idx]
        return AggregatedMultiChoiceVerdict(
            selected_index=most_common_idx,
            selected_label=selected_option.label,
            value=selected_option.value,
            na=selected_option.na,
            aggregated_value=selected_option.value,  # For nominal, discrete = continuous
        )

    def _calculate_score_from_reports(
        self, reports: list[CriterionReport], normalize: bool
    ) -> float:
        """Calculate score from criterion reports using CANNOT_ASSESS config.

        Supports both binary and multi-choice criteria using the score_value property.
        """
        config = self._cannot_assess_config

        # Separate assessable from cannot_assess (handles both binary CANNOT_ASSESS
        # and multi-choice NA options via the is_na property)
        assessable = [r for r in reports if not r.is_na]
        cannot_assess = [r for r in reports if r.is_na]

        # Apply strategy
        if config.strategy == CannotAssessStrategy.SKIP:
            working_reports = assessable
        elif config.strategy == CannotAssessStrategy.FAIL:
            # For binary: UNMET for positive, MET for negative
            # For multi-choice NA: use score_value of 0.0 (worst case for positive weight)
            fail_reports = []
            for r in cannot_assess:
                if r.is_multi_choice:
                    # For multi-choice, we don't modify - the NA value is already 0
                    fail_reports.append(r)
                else:
                    # For binary, convert to worst case verdict
                    fail_reports.append(
                        CriterionReport(
                            requirement=r.requirement,
                            verdict=CriterionVerdict.UNMET if r.weight > 0 else CriterionVerdict.MET,
                            reason=r.reason,
                            weight=r.weight,
                            name=r.name,
                            options=r.options,
                            scale_type=r.scale_type,
                            aggregation=r.aggregation,
                        )
                    )
            working_reports = assessable + fail_reports
        elif config.strategy == CannotAssessStrategy.ZERO:
            # Treat as UNMET (0 contribution) for both binary and multi-choice
            zero_reports = []
            for r in cannot_assess:
                if r.is_multi_choice:
                    # Already has value from NA option
                    zero_reports.append(r)
                else:
                    zero_reports.append(
                        CriterionReport(
                            requirement=r.requirement,
                            verdict=CriterionVerdict.UNMET,
                            reason=r.reason,
                            weight=r.weight,
                            name=r.name,
                            options=r.options,
                            scale_type=r.scale_type,
                            aggregation=r.aggregation,
                        )
                    )
            working_reports = assessable + zero_reports
        else:  # PARTIAL
            working_reports = assessable

        # Calculate weights
        if config.strategy == CannotAssessStrategy.SKIP:
            total_positive_weight = sum(max(0.0, r.weight) for r in working_reports)
            total_negative_weight = sum(abs(r.weight) for r in working_reports if r.weight < 0)
        else:
            total_positive_weight = sum(max(0.0, r.weight) for r in reports)
            total_negative_weight = sum(abs(r.weight) for r in reports if r.weight < 0)

        # Calculate weighted sum using score_value (handles both binary and multi-choice)
        weighted_sum = sum(r.score_value * r.weight for r in working_reports)

        # Add partial credit for PARTIAL strategy
        if config.strategy == CannotAssessStrategy.PARTIAL:
            for r in cannot_assess:
                if r.weight > 0:
                    weighted_sum += config.partial_credit * r.weight

        if not normalize:
            return weighted_sum

        if total_positive_weight > 0:
            return max(0.0, min(1.0, weighted_sum / total_positive_weight))
        elif total_negative_weight > 0:
            return max(0.0, min(1.0, 1.0 + weighted_sum / total_negative_weight))
        else:
            return 0.0
