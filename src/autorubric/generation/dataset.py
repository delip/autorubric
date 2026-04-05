"""Data model for AutorubricLM training and evaluation datasets.

Defines the schema for rubric generation examples: (input, output) pairs
where the input is a grading problem description (with optional reference
response and sample submissions) and the output is a list[Criterion].

Example usage:
    from autorubric.generation import RubricGenerationExample, RubricGenerationDataset
    from autorubric.types import Criterion

    example = RubricGenerationExample(
        grading_problem_description="Evaluate a student essay on the causes of WWI.",
        criteria=[
            Criterion(name="thesis", weight=30.0, requirement="Has a clear thesis statement"),
            Criterion(name="evidence", weight=40.0, requirement="Supports claims with evidence"),
            Criterion(name="organization", weight=30.0, requirement="Logically organized argument"),
        ],
        domain="writing",
        source_dataset="manual",
        is_human_authored=True,
    )

    dataset = RubricGenerationDataset(name="v0.1", examples=[example])
    dataset.to_json("training_data.json")
    loaded = RubricGenerationDataset.from_json("training_data.json")
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from autorubric.types import Criterion

CriterionMix = Literal["binary", "ordinal", "nominal", "heterogeneous"]
HumanAuthored = Literal[True, False, "unknown"]


class RubricGenerationExample(BaseModel):
    """A single (input, output) training example for AutorubricLM SFT.

    Input fields mirror the AutorubricLM spec's input contract. The output
    is a list[Criterion] in the existing autorubric schema. Metadata fields
    are kept minimal — quality scores and other derived signals belong in
    pipeline-specific sidecar data, not the core schema.

    Attributes:
        grading_problem_description: The task specification — what was asked,
            what context matters.
        reference_response: Optional gold-standard response establishing what
            "correct" or "ideal" looks like.
        responses: Optional set of actual submissions showing the quality range
            the rubric must discriminate between. Never truncated or summarized.
        target_criteria_count: Soft target for number of criteria to generate.
            When None, the model decides based on problem complexity.
        criterion_mix: What types of criteria to generate. When None during
            training, the prompt omits this instruction.
        criteria: The rubric the model should produce. Must conform to the
            existing autorubric Criterion schema.
        id: Unique identifier. Auto-generated as a deterministic content hash
            if not provided, making deduplication trivial.
        source_dataset: Origin dataset name for traceability.
        is_human_authored: Whether the rubric was authored or validated by a
            human (True), purely LLM-generated (False), or not known ("unknown").
        domain: Primary domain tag (free-form string).
        language: ISO 639-1 language code.
    """

    # ── Input (what the model sees) ─────────────────────────────────

    grading_problem_description: str
    reference_response: str | None = None
    responses: list[str] | None = None
    target_criteria_count: int | None = Field(default=None, ge=1, le=30)
    criterion_mix: CriterionMix | None = None

    # ── Output (what the model should produce) ──────────────────────

    criteria: list[Criterion]

    # ── Metadata ────────────────────────────────────────────────────

    id: str = ""
    source_dataset: str | None = None
    is_human_authored: HumanAuthored = "unknown"
    domain: str | None = None
    language: str = "en"

    # ── Validators ──────────────────────────────────────────────────

    @model_validator(mode="after")
    def _assign_id_if_empty(self) -> RubricGenerationExample:
        """Generate a deterministic content-based ID if none was provided."""
        if not self.id:
            content = self.grading_problem_description + json.dumps(
                [c.model_dump() for c in self.criteria], sort_keys=True
            )
            object.__setattr__(
                self, "id", hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            )
        return self

    # ── Computed properties ──────────────────────────────────────────

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_criteria(self) -> int:
        """Number of criteria in the rubric."""
        return len(self.criteria)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_negative_criteria(self) -> bool:
        """Whether any criterion has a negative weight (penalty)."""
        return any(c.weight < 0 for c in self.criteria)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def positive_weight_sum(self) -> float:
        """Sum of all positive criterion weights (should be 100 for well-formed rubrics)."""
        return sum(c.weight for c in self.criteria if c.weight > 0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def criterion_type_counts(self) -> dict[str, int]:
        """Count of binary, ordinal, and nominal criteria."""
        counts: dict[str, int] = {"binary": 0, "ordinal": 0, "nominal": 0}
        for c in self.criteria:
            if c.is_binary:
                counts["binary"] += 1
            elif c.scale_type == "nominal":
                counts["nominal"] += 1
            else:
                counts["ordinal"] += 1
        return counts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_criterion_mix(self) -> CriterionMix:
        """What criterion mix the output actually contains."""
        counts = self.criterion_type_counts
        present = {k for k, v in counts.items() if v > 0}
        if len(present) > 1:
            return "heterogeneous"
        return present.pop() if present else "binary"


class RubricGenerationDataset(BaseModel):
    """A collection of rubric generation training examples.

    Attributes:
        name: Dataset name or version identifier.
        version: Semantic version for the dataset.
        created_at: When this dataset was assembled.
        split: Split type (train, val, test, non-iid-test, etc.).
        examples: The training examples.
    """

    name: str
    version: str = "0.1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    examples: list[RubricGenerationExample] = Field(default_factory=list)
    split: str | None = None  # train, val, test, non-iid-test, etc.

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):  # type: ignore[override]
        return iter(self.examples)

    def __getitem__(self, idx: int) -> RubricGenerationExample:
        return self.examples[idx]

    def merge(
        self,
        other: RubricGenerationDataset,
        *,
        name: str | None = None,
        version: str | None = None,
        split: str | None = None,
    ) -> RubricGenerationDataset:
        """Merge another dataset into this one, returning a new dataset.

        Examples from both datasets are concatenated. Duplicates (by id)
        are kept — deduplication is the caller's responsibility.

        Args:
            other: Dataset to merge with this one.
            name: Name for the merged dataset. Defaults to this dataset's name.
            version: Version for the merged dataset. Defaults to this dataset's version.
            split: Split label for the merged dataset. Defaults to this dataset's split.
        """
        return RubricGenerationDataset(
            name=name or self.name,
            version=version or self.version,
            split=split or self.split,
            examples=list(self.examples) + list(other.examples),
        )

    def to_json(self, path: str) -> None:
        """Write the full dataset (metadata + examples) as a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: str) -> RubricGenerationDataset:
        """Load a dataset from a JSON file written by ``to_json``."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
