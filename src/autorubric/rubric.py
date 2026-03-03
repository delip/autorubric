"""Core Rubric class for evaluating text outputs against a set of weighted criteria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from autorubric.graders import Grader
from autorubric.types import (
    CannotAssessStrategy,
    Criterion,
    CriterionVerdict,
    EvaluationReport,
    ToGradeInput,
)


class Rubric:
    """A rubric is a list of criteria used to evaluate text outputs.

    Each criterion has a weight and requirement. Use the grade() method
    to evaluate text against this rubric using a grader.
    """

    def __init__(self, rubric: list[Criterion]):
        self.rubric = rubric

    async def grade(
        self,
        to_grade: ToGradeInput,
        grader: Grader,
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> EvaluationReport:
        """Grade text against this rubric using a grader.

        Args:
            to_grade: The text to evaluate. Can be either:
                - A string (optionally with <thinking>/<output> markers)
                - A dict with 'thinking' and 'output' keys
            grader: The grader to use. REQUIRED - must be provided.
                Configure length_penalty and normalize on the grader if needed.
            query: Optional input/query that prompted the response.
            reference_submission: Optional exemplar response for grading context.
                When present, provides calibration for the grader.

        Raises:
            TypeError: If grader is not provided.
        """
        return await grader.grade(
            to_grade=to_grade,
            rubric=self.rubric,
            query=query,
            reference_submission=reference_submission,
        )

    @staticmethod
    def validate_and_create_criteria(
        data: list[dict[str, Any]] | dict[str, Any],
    ) -> list[Criterion]:
        """Validate and create Criterion objects from raw data.

        Supports multiple formats:
        - Flat list of criteria
        - List of sections with criteria
        - Dict with 'sections' key containing list of sections
        - Dict with 'rubric' key containing sections
        """
        if isinstance(data, dict):
            if "rubric" in data:
                data = data["rubric"]

            if isinstance(data, dict):
                if "sections" in data:
                    sections = data["sections"]
                    if not isinstance(sections, list):
                        raise ValueError(
                            f"Invalid rubric format. Expected 'sections' to be a list, "
                            f"got {type(sections).__name__}"
                        )
                    data = sections
                else:
                    raise ValueError(
                        "Invalid rubric format. Dict must contain either 'sections' or 'rubric' key"
                    )

        if not isinstance(data, list):
            raise ValueError(
                f"Invalid rubric format. Expected a list, got {type(data).__name__}"
            )

        if not data:
            raise ValueError("No criteria found")

        flattened_criteria_data = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid item at index {idx}: expected a dictionary, got {type(item).__name__}"
                )

            if "criteria" in item:
                section_criteria = item["criteria"]
                if not isinstance(section_criteria, list):
                    raise ValueError(
                        f"Invalid section at index {idx}: 'criteria' must be a list, "
                        f"got {type(section_criteria).__name__}"
                    )
                flattened_criteria_data.extend(section_criteria)
            else:
                flattened_criteria_data.append(item)

        if not flattened_criteria_data:
            raise ValueError("No criteria found")

        criteria = []
        for idx, criterion_data in enumerate(flattened_criteria_data):
            if not isinstance(criterion_data, dict):
                raise ValueError(
                    f"Invalid criterion at index {idx}: expected a dictionary, "
                    f"got {type(criterion_data).__name__}"
                )

            try:
                criteria.append(Criterion(**criterion_data))  # type: ignore[arg-type]
            except ValidationError as e:
                error_details = []
                for error in e.errors():
                    field = ".".join(str(loc) for loc in error["loc"])
                    error_details.append(f"{field}: {error['msg']}")

                error_msg = f"Invalid criterion at index {idx}:\n  " + "\n  ".join(
                    error_details
                )
                raise ValueError(error_msg) from e
            except Exception as e:
                raise ValueError(
                    f"Failed to create criterion at index {idx}: {e}"
                ) from e

        return criteria

    @classmethod
    def from_yaml(cls, yaml_string: str) -> "Rubric":
        """Parse rubric from a YAML string."""
        try:
            data = yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML string: {e}") from e

        criteria = cls.validate_and_create_criteria(data)
        return cls(criteria)

    @classmethod
    def from_json(cls, json_string: str) -> "Rubric":
        """Parse rubric from a JSON string."""
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON string: {e}") from e

        criteria = cls.validate_and_create_criteria(data)
        return cls(criteria)

    @classmethod
    def from_file(cls, source: str | Any) -> "Rubric":
        """Load rubric from a file path or file-like object, auto-detecting format."""
        if hasattr(source, "read"):
            file_name = getattr(source, "name", "")  # type: ignore[arg-type]
            extension = Path(file_name).suffix.lower() if file_name else ""

            if not extension:
                raise ValueError(
                    "Cannot determine file format from file object. "
                    "File object must have a 'name' attribute with a file extension."
                )

            try:
                content = source.read()  # type: ignore[misc]
            except Exception as e:
                raise ValueError(f"Failed to read from file object: {e}") from e

            if extension in [".yaml", ".yml"]:
                try:
                    data = yaml.safe_load(content)
                except yaml.YAMLError as e:
                    raise ValueError(
                        f"Failed to parse YAML from file object: {e}"
                    ) from e
                criteria = cls.validate_and_create_criteria(data)
                return cls(criteria)
            elif extension == ".json":
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse JSON from file object: {e}"
                    ) from e
                criteria = cls.validate_and_create_criteria(data)
                return cls(criteria)
            else:
                raise ValueError(
                    f"Unsupported file format '{extension}' for file object: {file_name}\n"
                    f"Supported formats: .yaml, .yml, .json"
                )

        elif isinstance(source, str):
            path = Path(source)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {source}")

            extension = path.suffix.lower()

            if extension in [".yaml", ".yml"]:
                with open(source, encoding="utf-8") as f:
                    try:
                        data = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        raise ValueError(f"Failed to parse YAML file: {e}") from e
                criteria = cls.validate_and_create_criteria(data)
                return cls(criteria)
            elif extension == ".json":
                with open(source, encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse JSON file: {e}") from e
                criteria = cls.validate_and_create_criteria(data)
                return cls(criteria)
            else:
                raise ValueError(
                    f"Unsupported file format '{extension}' for file: {source}\n"
                    f"Supported formats: .yaml, .yml, .json"
                )
        else:
            raise ValueError(
                f"Invalid source type: expected str (file path) or file-like object, "
                f"got {type(source).__name__}"
            )

    @staticmethod
    def _apply_cannot_assess_strategy(
        strategy: CannotAssessStrategy,
        weight: float,
        partial_credit: float,
    ) -> tuple[float, float, bool]:
        """Apply CANNOT_ASSESS/NA strategy for a single criterion.

        Returns:
            (score_contribution, skip_positive_weight_adjustment, should_skip_criterion)
        """
        if strategy == CannotAssessStrategy.SKIP:
            return 0.0, max(0.0, weight), True
        elif strategy == CannotAssessStrategy.FAIL:
            # Worst case: 0 for positive weight, MET (subtract) for negative
            score = weight if weight < 0 else 0.0
            return score, 0.0, False
        elif strategy == CannotAssessStrategy.PARTIAL:
            score = partial_credit * weight if weight > 0 else 0.0
            return score, 0.0, False
        else:  # ZERO
            return 0.0, 0.0, False

    def compute_score(
        self,
        verdicts: list[CriterionVerdict | str],
        normalize: bool = True,
        cannot_assess_strategy: CannotAssessStrategy = CannotAssessStrategy.SKIP,
        partial_credit: float = 0.5,
    ) -> float:
        """Compute a weighted score from raw verdicts against this rubric.

        Single source of truth for scoring from verdict lists (e.g. ground truth
        labels). Handles binary (MET/UNMET/CANNOT_ASSESS) and multi-choice
        (option label strings) criteria.

        Args:
            verdicts: One value per criterion. Binary criteria accept
                CriterionVerdict or its string form; multi-choice criteria
                accept an option label string.
            normalize: If True, normalise to [0, 1]. If False, return the raw
                weighted sum.
            cannot_assess_strategy: How to handle CANNOT_ASSESS / NA verdicts.
            partial_credit: Credit fraction when strategy is PARTIAL.

        Returns:
            The computed score.
        """
        if len(verdicts) != len(self.rubric):
            raise ValueError(
                f"Expected {len(self.rubric)} verdicts, got {len(verdicts)}"
            )

        weighted_sum = 0.0
        total_positive_weight = 0.0
        total_negative_weight = 0.0
        skip_positive_weight = 0.0

        for criterion, verdict in zip(self.rubric, verdicts):
            weight = criterion.weight

            if criterion.is_multi_choice:
                if not isinstance(verdict, str):
                    raise ValueError(
                        f"Multi-choice criterion '{criterion.name}' requires a "
                        f"label string, got {type(verdict).__name__}"
                    )
                idx = criterion.find_option_by_label(verdict)
                opt = criterion.options[idx]  # type: ignore[index]
                if opt.na:
                    score, skip_adj, should_skip = self._apply_cannot_assess_strategy(
                        cannot_assess_strategy, weight, partial_credit
                    )
                    weighted_sum += score
                    skip_positive_weight += skip_adj
                    if should_skip:
                        continue
                else:
                    weighted_sum += opt.value * weight
                if weight > 0:
                    total_positive_weight += weight
                else:
                    total_negative_weight += abs(weight)
            else:
                # Binary criterion
                if isinstance(verdict, str):
                    try:
                        verdict = CriterionVerdict(verdict)
                    except ValueError:
                        raise ValueError(
                            f"Invalid binary verdict '{verdict}'. "
                            f"Must be 'MET', 'UNMET', or 'CANNOT_ASSESS'."
                        ) from None

                if verdict == CriterionVerdict.CANNOT_ASSESS:
                    score, skip_adj, should_skip = self._apply_cannot_assess_strategy(
                        cannot_assess_strategy, weight, partial_credit
                    )
                    weighted_sum += score
                    skip_positive_weight += skip_adj
                    if should_skip:
                        continue
                elif verdict == CriterionVerdict.MET:
                    weighted_sum += weight

                if weight > 0:
                    total_positive_weight += weight
                else:
                    total_negative_weight += abs(weight)

        if not normalize:
            return weighted_sum

        # Adjust denominator for SKIP strategy
        effective_positive = total_positive_weight - skip_positive_weight
        if effective_positive > 0:
            return max(0.0, min(1.0, weighted_sum / effective_positive))
        elif total_negative_weight > 0:
            return max(0.0, min(1.0, 1.0 + weighted_sum / total_negative_weight))
        else:
            return 0.0

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]] | dict[str, Any]) -> "Rubric":
        """Create rubric from a list of dictionaries or a dict with sections."""
        criteria = cls.validate_and_create_criteria(data)
        return cls(criteria)
