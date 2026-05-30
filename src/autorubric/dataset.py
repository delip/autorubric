"""Dataset classes for rubric-based evaluation.

This module provides data structures for organizing evaluation datasets with
ground truth labels, supporting both training and evaluation workflows.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from autorubric.rubric import Rubric
from autorubric.types import CriterionVerdict

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class DataItem:
    """A single item to be graded, optionally with ground truth verdicts.

    Attributes:
        submission: The content to be evaluated. Can be plain text or a JSON-serialized
            string for structured data (e.g., dialogues, multi-part responses).
        description: A brief description of this item (e.g., "High quality response").
        ground_truth: Optional list of ground truth values, one per criterion.
            - For binary criteria: CriterionVerdict (MET, UNMET, CANNOT_ASSESS)
            - For multi-choice criteria: str (option label)
            Used for computing evaluation metrics against LLM predictions.
        rubric: Optional per-item rubric. If provided, this rubric is used for grading
            instead of the dataset-level rubric. Useful for datasets where each item
            has unique evaluation criteria (e.g., ResearcherBench).
        reference_submission: Optional exemplar response for grading context. When
            present, helps calibrate the grader's expectations. Item-level takes
            precedence over dataset-level reference.
        prompt: Optional per-item prompt. If provided, overrides the dataset-level
            prompt for this item.

    Example:
        >>> # Binary criteria only
        >>> item = DataItem(
        ...     submission="The Industrial Revolution began in Britain around 1760...",
        ...     description="Excellent essay covering all criteria",
        ...     ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.UNMET]
        ... )
        >>> # Mixed binary and multi-choice
        >>> item = DataItem(
        ...     submission="The assistant responded helpfully...",
        ...     description="Good dialogue",
        ...     ground_truth=[CriterionVerdict.MET, "Very satisfied", "Yes - reasonable"]
        ... )
        >>> # With per-item rubric
        >>> from autorubric import Rubric, Criterion
        >>> item = DataItem(
        ...     submission="Response to a specific question...",
        ...     description="Question-specific grading",
        ...     rubric=Rubric([Criterion(name="Relevance", weight=1.0, requirement="...")])
        ... )
    """

    submission: str
    description: str
    ground_truth: list[CriterionVerdict | str] | None = None
    rubric: Rubric | None = None
    reference_submission: str | None = None
    prompt: str | None = None

    def __post_init__(self) -> None:
        """Validate ground truth values and rubric consistency."""
        if self.ground_truth is not None:
            for v in self.ground_truth:
                if not isinstance(v, (CriterionVerdict, str)):
                    raise ValueError(
                        f"Ground truth values must be CriterionVerdict or str, "
                        f"got {type(v).__name__}"
                    )
            # Validate ground_truth length against per-item rubric if present
            if self.rubric is not None and len(self.ground_truth) != len(self.rubric.rubric):
                raise ValueError(
                    f"Ground truth has {len(self.ground_truth)} values, "
                    f"but item rubric has {len(self.rubric.rubric)} criteria"
                )


@dataclass
class RubricDataset:
    """A collection of DataItems tied to a specific prompt and rubric.

    The RubricDataset encapsulates:
    - The prompt that generated the responses
    - The rubric used for evaluation (global or per-item)
    - A collection of DataItems with optional ground truth labels

    This is useful for:
    - Evaluating LLM grader accuracy against human judgments
    - Training reward models with labeled data
    - Benchmarking different grading strategies

    Attributes:
        prompt: Optional global prompt/question that items are responses to. Can be
            None if all items have their own prompts.
        rubric: Optional global Rubric used to evaluate items. Can be None if all
            items have their own rubrics.
        items: List of DataItem instances to evaluate.
        name: Optional name for the dataset (e.g., "essay-grading-v1").
        reference_submission: Optional global exemplar response for grading context.
            When present, provides calibration for the grader. Item-level reference
            takes precedence over this dataset-level reference.

    Example:
        >>> from autorubric import Rubric, Criterion, CriterionVerdict
        >>> rubric = Rubric([
        ...     Criterion(name="Accuracy", weight=10.0, requirement="Factually correct"),
        ...     Criterion(name="Clarity", weight=5.0, requirement="Clear and concise"),
        ... ])
        >>> dataset = RubricDataset(
        ...     prompt="Explain photosynthesis",
        ...     rubric=rubric,
        ... )
        >>> dataset.add_item(
        ...     submission="Photosynthesis is the process...",
        ...     description="Good response",
        ...     ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET]
        ... )
    """

    prompt: str | None = None
    rubric: Rubric | None = None
    items: list[DataItem] = field(default_factory=list)
    name: str | None = None
    reference_submission: str | None = None

    def __post_init__(self) -> None:
        """Validate dataset consistency."""
        for i, item in enumerate(self.items):
            # Validate prompt availability
            if item.prompt is None and self.prompt is None:
                raise ValueError(f"Item {i} has no prompt and dataset has no global prompt")
            effective_rubric = self.get_item_rubric(i)
            if item.ground_truth is not None and len(item.ground_truth) != len(
                effective_rubric.rubric
            ):
                raise ValueError(
                    f"Item {i} has {len(item.ground_truth)} ground truth values, "
                    f"but rubric has {len(effective_rubric.rubric)} criteria"
                )

    def get_item_rubric(self, idx: int) -> Rubric:
        """Get the effective rubric for an item (per-item or global fallback).

        Args:
            idx: Index of the item.

        Returns:
            The item's rubric if set, otherwise the dataset's global rubric.

        Raises:
            ValueError: If neither item nor dataset has a rubric.
        """
        item = self.items[idx]
        if item.rubric is not None:
            return item.rubric
        if self.rubric is not None:
            return self.rubric
        raise ValueError(f"Item {idx} has no rubric and dataset has no global rubric")

    def get_item_reference_submission(self, idx: int) -> str | None:
        """Get the effective reference submission for an item.

        Item-level reference takes precedence over dataset-level reference.

        Args:
            idx: Index of the item.

        Returns:
            The item's reference_submission if set, otherwise the dataset's
            global reference_submission. May be None if neither is set.
        """
        item = self.items[idx]
        if item.reference_submission is not None:
            return item.reference_submission
        return self.reference_submission

    def get_item_prompt(self, idx: int) -> str:
        """Get the effective prompt for an item (per-item or global fallback).

        Args:
            idx: Index of the item.

        Returns:
            The item's prompt if set, otherwise the dataset's global prompt.

        Raises:
            ValueError: If neither item nor dataset has a prompt.
        """
        item = self.items[idx]
        if item.prompt is not None:
            return item.prompt
        if self.prompt is not None:
            return self.prompt
        raise ValueError(f"Item {idx} has no prompt and dataset has no global prompt")

    @property
    def criterion_names(self) -> list[str]:
        """Get criterion names from global rubric.

        Raises:
            ValueError: If no global rubric is set.
        """
        if self.rubric is None:
            raise ValueError(
                "Cannot access criterion_names: no global rubric set. "
                "Use get_item_rubric(idx) for per-item rubrics."
            )
        return [c.name or f"C{i + 1}" for i, c in enumerate(self.rubric.rubric)]

    @property
    def num_criteria(self) -> int:
        """Number of criteria in the global rubric.

        Raises:
            ValueError: If no global rubric is set.
        """
        if self.rubric is None:
            raise ValueError(
                "Cannot access num_criteria: no global rubric set. "
                "Use get_item_rubric(idx) for per-item rubrics."
            )
        return len(self.rubric.rubric)

    @property
    def total_positive_weight(self) -> float:
        """Sum of all positive criterion weights in global rubric.

        Raises:
            ValueError: If no global rubric is set.
        """
        if self.rubric is None:
            raise ValueError(
                "Cannot access total_positive_weight: no global rubric set. "
                "Use get_item_rubric(idx) for per-item rubrics."
            )
        return sum(c.weight for c in self.rubric.rubric if c.weight > 0)

    def compute_weighted_score(
        self,
        verdicts: list[CriterionVerdict | str],
        normalize: bool = True,
        rubric: Rubric | None = None,
    ) -> float:
        """Compute weighted score from verdicts (binary or multi-choice).

        Args:
            verdicts: List of verdict values, one per criterion.
                - For binary criteria: CriterionVerdict (MET=1.0, UNMET=0.0)
                - For multi-choice criteria: str (option label, resolved to value)
            normalize: If True, normalize score to [0, 1]. If False, return raw sum.
            rubric: Optional rubric to use for scoring. If None, uses global rubric.

        Returns:
            Weighted score based on criterion weights and verdicts.

        Raises:
            ValueError: If a multi-choice label doesn't match any option, or if
                rubric is None and no global rubric is set.
        """
        effective_rubric = rubric if rubric is not None else self.rubric
        if effective_rubric is None:
            raise ValueError("Cannot compute score: no rubric provided and no global rubric set")
        # Delegate to the rubric's unified scorer. The default
        # cannot_assess_strategy=SKIP matches this method's historical behavior
        # (CANNOT_ASSESS / NA excluded from numerator and denominator), and routes
        # through the shared score_reports core so all scoring paths agree.
        return effective_rubric.compute_score(verdicts, normalize=normalize)

    def add_item(
        self,
        submission: str,
        description: str,
        ground_truth: list[CriterionVerdict | str] | None = None,
        rubric: Rubric | None = None,
        reference_submission: str | None = None,
        prompt: str | None = None,
    ) -> None:
        """Add a new item to the dataset.

        Args:
            submission: The content to be evaluated.
            description: A brief description of this item.
            ground_truth: Optional list of ground truth values.
                - For binary criteria: CriterionVerdict (MET, UNMET, CANNOT_ASSESS)
                - For multi-choice criteria: str (option label)
            rubric: Optional per-item rubric. If None, uses global rubric.
            reference_submission: Optional exemplar response for grading context.
            prompt: Optional per-item prompt. If None, uses global prompt.

        Raises:
            ValueError: If ground_truth length doesn't match effective rubric criteria count,
                or if neither per-item nor global rubric is available, or if neither
                per-item nor global prompt is available.
        """
        if prompt is None and self.prompt is None:
            raise ValueError(
                "Cannot add item: no per-item prompt provided and no global prompt set"
            )
        item = DataItem(
            submission=submission,
            description=description,
            ground_truth=ground_truth,
            rubric=rubric,
            reference_submission=reference_submission,
            prompt=prompt,
        )
        effective_rubric = item.rubric if item.rubric is not None else self.rubric
        if effective_rubric is None:
            raise ValueError(
                "Cannot add item: no per-item rubric provided and no global rubric set"
            )
        if item.ground_truth is not None and len(item.ground_truth) != len(effective_rubric.rubric):
            raise ValueError(
                f"Ground truth has {len(item.ground_truth)} values, "
                f"but rubric has {len(effective_rubric.rubric)} criteria"
            )
        self.items.append(item)

    def __len__(self) -> int:
        """Return number of items in the dataset."""
        return len(self.items)

    def __iter__(self) -> Iterator[DataItem]:
        """Iterate over items in the dataset."""
        return iter(self.items)

    def __getitem__(self, idx: int) -> DataItem:
        """Get item by index."""
        return self.items[idx]

    # =========================================================================
    # Serialization
    # =========================================================================

    def _serialize_rubric(self, rubric: Rubric) -> list[dict[str, Any]]:
        """Serialize a Rubric to a list of criterion dicts."""
        rubric_data = []
        for c in rubric.rubric:
            criterion_data: dict[str, Any] = {
                "name": c.name,
                "weight": c.weight,
                "requirement": c.requirement,
            }
            if c.options is not None:
                criterion_data["scale_type"] = c.scale_type
                criterion_data["options"] = [
                    {"label": opt.label, "value": opt.value, "na": opt.na} for opt in c.options
                ]
                if c.aggregation is not None:
                    criterion_data["aggregation"] = c.aggregation
            rubric_data.append(criterion_data)
        return rubric_data

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize the dataset to a JSON string.

        Args:
            indent: Number of spaces for indentation. None for compact output.

        Returns:
            JSON string representation of the dataset.
        """
        data: dict[str, Any] = {}
        if self.name is not None:
            data["name"] = self.name
        if self.prompt is not None:
            data["prompt"] = self.prompt

        # Serialize global rubric (can be None)
        if self.rubric is not None:
            data["rubric"] = self._serialize_rubric(self.rubric)
        else:
            data["rubric"] = None

        # Serialize global reference_submission if present
        if self.reference_submission is not None:
            data["reference_submission"] = self.reference_submission

        # Serialize items with ground truth and per-item rubrics
        items_data = []
        for item in self.items:
            item_data: dict[str, Any] = {
                "submission": item.submission,
                "description": item.description,
            }
            if item.ground_truth is not None:
                # Serialize ground truth: CriterionVerdict -> str, str stays str
                gt_values = []
                for v in item.ground_truth:
                    if isinstance(v, CriterionVerdict):
                        gt_values.append(v.value)
                    else:
                        gt_values.append(v)  # Already a string (option label)
                item_data["ground_truth"] = gt_values
            else:
                item_data["ground_truth"] = None
            # Serialize per-item rubric if present
            if item.rubric is not None:
                item_data["rubric"] = self._serialize_rubric(item.rubric)
            # Serialize per-item reference_submission if present
            if item.reference_submission is not None:
                item_data["reference_submission"] = item.reference_submission
            # Serialize per-item prompt if present
            if item.prompt is not None:
                item_data["prompt"] = item.prompt
            items_data.append(item_data)
        data["items"] = items_data

        return json.dumps(data, indent=indent)

    def to_file(self, path: str | Path) -> None:
        """Save dataset to a JSON file.

        Args:
            path: Path to write the JSON file.
        """
        from pathlib import Path

        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, json_string: str) -> RubricDataset:
        """Deserialize a dataset from a JSON string.

        Args:
            json_string: JSON string representation of the dataset.

        Returns:
            RubricDataset instance.

        Raises:
            ValueError: If the JSON is invalid, missing required fields, or if
                an item has no rubric when no global rubric is set.
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        # Validate required fields
        if "rubric" not in data:
            raise ValueError("Missing required field: 'rubric'")

        # Parse global rubric (can be None/null)
        rubric_data = data["rubric"]
        rubric: Rubric | None = None
        if rubric_data is not None:
            rubric = Rubric.from_dict(rubric_data)

        # Parse items
        items: list[DataItem] = []
        for i, item_data in enumerate(data.get("items", [])):
            if not isinstance(item_data, dict):
                raise ValueError(f"Item {i} must be a dict, got {type(item_data).__name__}")

            submission = item_data.get("submission")
            description = item_data.get("description")

            if submission is None:
                raise ValueError(f"Item {i} missing required field: 'submission'")
            if description is None:
                raise ValueError(f"Item {i} missing required field: 'description'")

            # Parse per-item rubric if present
            item_rubric_data = item_data.get("rubric")
            item_rubric: Rubric | None = None
            if item_rubric_data is not None:
                item_rubric = Rubric.from_dict(item_rubric_data)

            # Validate that item has access to a rubric
            effective_rubric = item_rubric if item_rubric is not None else rubric
            if effective_rubric is None:
                raise ValueError(f"Item {i} has no rubric and dataset has no global rubric")

            # Parse ground truth against the effective rubric
            ground_truth_raw = item_data.get("ground_truth")
            ground_truth: list[CriterionVerdict | str] | None = None
            if ground_truth_raw is not None:
                ground_truth = []
                for j, v in enumerate(ground_truth_raw):
                    criterion = (
                        effective_rubric.rubric[j] if j < len(effective_rubric.rubric) else None
                    )

                    if criterion is not None and criterion.is_multi_choice:
                        # Multi-choice: keep as string (option label)
                        if not isinstance(v, str):
                            raise ValueError(
                                f"Item {i}, ground_truth[{j}]: multi-choice criterion "
                                f"expects option label string, got {type(v).__name__}"
                            )
                        # Validate that the label exists
                        try:
                            criterion.find_option_by_label(v)
                        except ValueError as e:
                            raise ValueError(f"Item {i}, ground_truth[{j}]: {e}") from None
                        ground_truth.append(v)
                    else:
                        # Binary: parse as CriterionVerdict
                        try:
                            ground_truth.append(CriterionVerdict(v))
                        except ValueError:
                            raise ValueError(
                                f"Item {i}, ground_truth[{j}]: invalid verdict '{v}'. "
                                f"Must be 'MET', 'UNMET', or 'CANNOT_ASSESS'."
                            ) from None

            # Parse per-item reference_submission if present
            item_reference = item_data.get("reference_submission")
            # Parse per-item prompt if present
            item_prompt = item_data.get("prompt")

            items.append(
                DataItem(
                    submission=submission,
                    description=description,
                    ground_truth=ground_truth,
                    rubric=item_rubric,
                    reference_submission=item_reference,
                    prompt=item_prompt,
                )
            )

        return cls(
            prompt=data.get("prompt"),
            rubric=rubric,
            items=items,
            name=data.get("name"),
            reference_submission=data.get("reference_submission"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> RubricDataset:
        """Load dataset from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            RubricDataset instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the JSON is invalid.
        """
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        return cls.from_json(path.read_text(encoding="utf-8"))

    # =========================================================================
    # Dataset Splitting
    # =========================================================================

    def split_train_test(
        self,
        n_train: int,
        *,
        stratify: bool = True,
        seed: int | None = None,
    ) -> tuple[RubricDataset, RubricDataset]:
        """Split dataset into training and test sets.

        The training set can be used to provide few-shot examples for grading,
        while the test set is used for evaluation.

        Args:
            n_train: Exact number of items for training set.
            stratify: If True, stratify by per-criterion verdict distribution.
                This ensures each split has similar proportion of MET/UNMET/CANNOT_ASSESS
                for each criterion position. Requires all items to have ground_truth.
            seed: Random seed for reproducible splits.

        Returns:
            Tuple of (train_dataset, test_dataset).

        Raises:
            ValueError: If n_train is invalid or stratify=True but items lack ground_truth.

        Example:
            >>> dataset = RubricDataset.from_file("data.json")
            >>> train, test = dataset.split_train_test(n_train=100, stratify=True, seed=42)
            >>> print(f"Train: {len(train)}, Test: {len(test)}")
        """
        if n_train < 0:
            raise ValueError(f"n_train must be non-negative, got {n_train}")
        if n_train > len(self.items):
            raise ValueError(f"n_train ({n_train}) exceeds dataset size ({len(self.items)})")

        rng = random.Random(seed)

        if stratify:
            train_items, test_items = self._stratified_split(n_train, rng)
        else:
            indices = list(range(len(self.items)))
            rng.shuffle(indices)
            train_items = [self.items[i] for i in indices[:n_train]]
            test_items = [self.items[i] for i in indices[n_train:]]

        train_dataset = RubricDataset(
            prompt=self.prompt,
            rubric=self.rubric,
            items=train_items,
            name=self.name,
            reference_submission=self.reference_submission,
        )
        test_dataset = RubricDataset(
            prompt=self.prompt,
            rubric=self.rubric,
            items=test_items,
            reference_submission=self.reference_submission,
            name=self.name,
        )

        return train_dataset, test_dataset

    def _stratified_split(
        self,
        n_train: int,
        rng: random.Random,
    ) -> tuple[list[DataItem], list[DataItem]]:
        """Perform stratified split based on verdict signature patterns.

        Groups items by their full verdict signature (e.g., "MET-UNMET-MET" for
        binary criteria, or "MET-Satisfied-Just right" for mixed criteria).

        Args:
            n_train: Number of items for training set.
            rng: Random number generator for reproducibility.

        Returns:
            Tuple of (train_items, test_items).
        """
        from collections import defaultdict

        # Group by verdict signature
        groups: dict[str, list[DataItem]] = defaultdict(list)
        for item in self.items:
            if item.ground_truth is None:
                raise ValueError(
                    "Stratified split requires ground_truth on all items. "
                    "Use stratify=False for items without ground truth."
                )
            # Build signature: CriterionVerdict.value for binary, str for multi-choice
            sig_parts = []
            for v in item.ground_truth:
                if isinstance(v, CriterionVerdict):
                    sig_parts.append(v.value)
                else:
                    sig_parts.append(v)  # Already a string (option label)
            signature = "-".join(sig_parts)
            groups[signature].append(item)

        train_items: list[DataItem] = []
        test_items: list[DataItem] = []
        total_items = len(self.items)

        # Shuffle each group
        for items in groups.values():
            rng.shuffle(items)

        # Allocate proportionally from each group
        remaining_train = n_train
        remaining_total = total_items

        for signature, group_items in groups.items():
            group_size = len(group_items)

            # Calculate how many to take from this group
            # Use proportional allocation with rounding
            if remaining_total > 0:
                n_from_group = round(group_size * remaining_train / remaining_total)
                n_from_group = max(0, min(n_from_group, group_size, remaining_train))
            else:
                n_from_group = 0

            train_items.extend(group_items[:n_from_group])
            test_items.extend(group_items[n_from_group:])

            remaining_train -= n_from_group
            remaining_total -= group_size

        # Final shuffle to mix up the signatures
        rng.shuffle(train_items)
        rng.shuffle(test_items)

        return train_items, test_items
