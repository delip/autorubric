"""Core Rubric class for evaluating text outputs against a set of weighted criteria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from autorubric.graders import Grader
from autorubric.graders.base import (
    ReportT,
    _accepts_guidelines,
    _takes_guidelines,
    _warn_ignores_guidelines,
)
from autorubric.scoring import score_reports
from autorubric.types import (
    CannotAssessConfig,
    CannotAssessStrategy,
    Criterion,
    CriterionReport,
    CriterionVerdict,
    MultiChoiceVerdict,
    ToGradeInput,
)
from autorubric.utils import _normalize_guidelines


def _guidelines_field(data: dict[str, Any]) -> str | None:
    """Read the optional ``"guidelines"`` key of a rubric dict, in its stored form.

    A missing key and ``null`` both mean no guidelines; blank text is normalized as in
    ``_normalize_guidelines``.

    Raises:
        ValueError: If the value is neither a string nor ``null``.
    """
    guidelines = data.get("guidelines")
    if guidelines is not None and not isinstance(guidelines, str):
        raise ValueError(
            f"Invalid rubric format. Expected 'guidelines' to be a string, "
            f"got {type(guidelines).__name__}"
        )
    return _normalize_guidelines(guidelines)


def _rubric_guidelines(data: Any) -> str | None:
    """Read the guidelines of raw rubric data: beside the criteria or inside ``"rubric"``.

    Only the ``"guidelines"`` key of a dict and, when its ``"rubric"`` value is a dict,
    that dict's own ``"guidelines"`` key are read, whatever else the data holds: a format of
    a subclass's own (parsed by its ``validate_and_create_criteria``) carries guidelines
    the same way, and a list carries none.

    Raises:
        ValueError: If a ``"guidelines"`` value is neither a string nor ``null``, or
            guidelines are given both beside and inside ``"rubric"``.
    """
    if not isinstance(data, dict):
        return None
    guidelines = _guidelines_field(data)
    inner = data.get("rubric")
    if isinstance(inner, dict):
        inner_guidelines = _guidelines_field(inner)
        if inner_guidelines is not None:
            if guidelines is not None:
                raise ValueError(
                    "Invalid rubric format. 'guidelines' is given both beside "
                    "and inside 'rubric'; give it once"
                )
            guidelines = inner_guidelines
    return guidelines


def _criteria_entries(data: dict[str, Any], missing_message: str) -> list[Any]:
    """Return the criteria list of a rubric dict: its ``"sections"``, else ``"criteria"``.

    Raises:
        ValueError: With ``missing_message`` if the dict has neither key, or if the value
            is not a list.
    """
    for key in ("sections", "criteria"):
        if key in data:
            entries = data[key]
            if not isinstance(entries, list):
                raise ValueError(
                    f"Invalid rubric format. Expected '{key}' to be a list, "
                    f"got {type(entries).__name__}"
                )
            return entries
    raise ValueError(missing_message)


class Rubric:
    """A rubric is a list of criteria used to evaluate text outputs.

    Each criterion has a weight and requirement. Use the grade() method
    to evaluate text against this rubric using a grader.

    Attributes:
        rubric: The criteria, in order.
        guidelines: Optional free text that applies to every criterion of the rubric
            (grading conventions, definitions, audience, scale anchors), or ``None``.
            Blank text (empty or whitespace only) means no guidelines and is stored as
            ``None``; any other text is kept verbatim. Assignment applies the same rules.
            On disk a rubric with guidelines is written as
            ``{"guidelines": "...", "criteria": [...]}``; one without them keeps the list
            form.
    """

    # Class-level default: a Rubric pickled before guidelines existed restores without
    # the instance attribute and reads as having no guidelines.
    _guidelines: str | None = None

    def __init__(self, rubric: list[Criterion], *, guidelines: str | None = None):
        """Create a rubric.

        Args:
            rubric: The criteria, in order.
            guidelines: Optional free text that applies to every criterion. Blank text
                means no guidelines.

        Raises:
            TypeError: If ``guidelines`` is neither a ``str`` nor ``None``.
        """
        self.rubric = rubric
        self.guidelines = guidelines

    @property
    def guidelines(self) -> str | None:
        """Free text that applies to every criterion, or ``None`` when there is none."""
        return self._guidelines

    @guidelines.setter
    def guidelines(self, value: str | None) -> None:
        self._guidelines = _normalize_guidelines(value)

    async def grade(
        self,
        to_grade: ToGradeInput,
        grader: Grader[ReportT],
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> ReportT:
        """Grade text against this rubric using a grader.

        The rubric's guidelines, when it has them, go to the grader as
        ``grader.grade(..., guidelines=self.guidelines)``; a rubric without guidelines calls
        ``grader.grade`` exactly as before guidelines existed. A grader whose ``grade`` does
        not accept ``guidelines=`` (an override with the old signature) is called without
        them and warns once per instance that it ignores them, like a ``Grader`` whose
        ``judge`` does not accept them (see ``Grader.grade``).

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

        Warns:
            UserWarning: "<GraderClass> ignores rubric guidelines", once per grader
                instance, when the rubric has guidelines and the grader cannot take them.
        """
        guidelines = self.guidelines
        if guidelines is not None and not _takes_guidelines(grader, "grade"):
            _warn_ignores_guidelines(grader)
            guidelines = None
        if guidelines is None:
            return await grader.grade(
                to_grade=to_grade,
                rubric=self.rubric,
                query=query,
                reference_submission=reference_submission,
            )
        return await grader.grade(
            to_grade=to_grade,
            rubric=self.rubric,
            query=query,
            reference_submission=reference_submission,
            guidelines=guidelines,
        )

    @staticmethod
    def validate_and_create_criteria(
        data: list[dict[str, Any]] | dict[str, Any],
    ) -> list[Criterion]:
        """Validate and create Criterion objects from raw data.

        Accepts every rubric format ``from_dict`` accepts and returns the criteria alone;
        guidelines in ``data`` are validated as ``from_dict`` validates them, but not
        returned. Every loader (``from_dict``, ``from_json``, ``from_yaml``, ``from_file``)
        takes its criteria from ``cls.validate_and_create_criteria``, so a subclass that
        overrides this method changes how all of them parse.

        Raises:
            ValueError: If the format or a criterion is invalid, ``"guidelines"`` is not a
                string, or guidelines are given both beside and inside ``"rubric"``.
        """
        _rubric_guidelines(data)  # Validated here; the loaders read them.
        if isinstance(data, dict):
            if "rubric" in data:
                data = data["rubric"]
                if isinstance(data, dict):
                    data = _criteria_entries(
                        data,
                        "Invalid rubric format. 'rubric' must be a list, or a dict with a "
                        "'sections' or 'criteria' key",
                    )
            else:
                data = _criteria_entries(
                    data,
                    "Invalid rubric format. Dict must contain either 'sections' or 'rubric' "
                    "key, or a 'criteria' key (each may be combined with 'guidelines')",
                )

        if not isinstance(data, list):
            raise ValueError(f"Invalid rubric format. Expected a list, got {type(data).__name__}")

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

                error_msg = f"Invalid criterion at index {idx}:\n  " + "\n  ".join(error_details)
                raise ValueError(error_msg) from e
            except Exception as e:
                raise ValueError(f"Failed to create criterion at index {idx}: {e}") from e

        return criteria

    @classmethod
    def _from_data(cls, data: Any) -> Rubric:
        """Build a rubric of this class from raw rubric data: the last step of every loader.

        The criteria come from ``cls.validate_and_create_criteria`` and the guidelines from
        ``_rubric_guidelines``. Without guidelines, ``cls(criteria)`` is called exactly as
        before guidelines existed. With them, ``cls`` gets ``guidelines=`` only when its
        ``__init__`` declares a ``guidelines`` parameter that can be passed by keyword, and
        then decides what they become. Any other subclass, including one whose ``__init__``
        takes ``**kwargs`` (which may predate guidelines, and then either reject the keyword
        or keep it as an option of its own), is built with ``cls(criteria)`` as before and
        then given the guidelines through the ``guidelines`` property. So a subclass never
        receives a keyword it does not declare, and every rubric file loads with its
        guidelines honoured.
        """
        criteria = cls.validate_and_create_criteria(data)
        guidelines = _rubric_guidelines(data)
        if guidelines is None:
            return cls(criteria)
        if _accepts_guidelines(cls, through_kwargs=False):
            return cls(criteria, guidelines=guidelines)
        rubric = cls(criteria)
        rubric.guidelines = guidelines
        return rubric

    @classmethod
    def from_yaml(cls, yaml_string: str) -> Rubric:
        """Parse a rubric from a YAML string holding any format ``from_dict`` accepts."""
        try:
            data = yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML string: {e}") from e

        return cls._from_data(data)

    @classmethod
    def from_json(cls, json_string: str) -> Rubric:
        """Parse a rubric from a JSON string holding any format ``from_dict`` accepts."""
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON string: {e}") from e

        return cls._from_data(data)

    @classmethod
    def from_file(cls, source: str | Any) -> Rubric:
        """Load a rubric from a file path or file-like object, auto-detecting format.

        The file (``.json``, ``.yaml`` or ``.yml``) may hold any format ``from_dict``
        accepts.
        """
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
                    raise ValueError(f"Failed to parse YAML from file object: {e}") from e
                return cls._from_data(data)
            elif extension == ".json":
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON from file object: {e}") from e
                return cls._from_data(data)
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
                return cls._from_data(data)
            elif extension == ".json":
                with open(source, encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse JSON file: {e}") from e
                return cls._from_data(data)
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

        Parses and validates each verdict into a ``CriterionReport`` and delegates
        to the shared ``score_reports`` core, so this path agrees exactly with the
        live grader and ``RubricDataset.compute_weighted_score`` across every
        ``CannotAssessStrategy`` x {binary, multi-choice} x {+/- weight}.

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
            raise ValueError(f"Expected {len(self.rubric)} verdicts, got {len(verdicts)}")

        reports: list[CriterionReport] = []
        for criterion, verdict in zip(self.rubric, verdicts):
            if criterion.is_multi_choice:
                if not isinstance(verdict, str):
                    raise ValueError(
                        f"Multi-choice criterion '{criterion.name}' requires a "
                        f"label string, got {type(verdict).__name__}"
                    )
                idx = criterion.find_option_by_label(verdict)
                opt = criterion.options[idx]  # type: ignore[index]
                reports.append(
                    CriterionReport(
                        requirement=criterion.requirement,
                        name=criterion.name,
                        weight=criterion.weight,
                        options=criterion.options,
                        scale_type=criterion.scale_type,
                        aggregation=criterion.aggregation,
                        multi_choice_verdict=MultiChoiceVerdict(
                            selected_index=idx,
                            selected_label=opt.label,
                            value=opt.value,
                            na=opt.na,
                        ),
                        reason="",
                    )
                )
            else:
                if isinstance(verdict, str):
                    try:
                        verdict = CriterionVerdict(verdict)
                    except ValueError:
                        raise ValueError(
                            f"Invalid binary verdict '{verdict}'. "
                            f"Must be 'MET', 'UNMET', or 'CANNOT_ASSESS'."
                        ) from None
                reports.append(
                    CriterionReport(
                        requirement=criterion.requirement,
                        name=criterion.name,
                        weight=criterion.weight,
                        verdict=verdict,
                        reason="",
                    )
                )

        config = CannotAssessConfig(strategy=cannot_assess_strategy, partial_credit=partial_credit)
        return score_reports(reports, config, normalize)

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]] | dict[str, Any]) -> Rubric:
        """Create a rubric from parsed data: a list of criteria or a rubric dict.

        Supported formats:

        - A list of criteria, or of sections (a dict with a ``"criteria"`` list), or both.
        - A dict with a ``"sections"`` list or a ``"criteria"`` list (such a list may
          itself hold sections).
        - A dict with a ``"rubric"`` key holding a list, or a dict with a ``"sections"`` or
          ``"criteria"`` list.

        Any dict form may carry ``"guidelines"`` (a string, or ``null`` for none), beside
        or inside ``"rubric"`` but not both; they become ``Rubric.guidelines``, e.g.
        ``{"guidelines": "...", "criteria": [...]}``. A list carries no guidelines. When a
        dict has several of the keys, ``"rubric"`` wins over ``"sections"``, which wins over
        ``"criteria"``; other keys are ignored.

        Raises:
            ValueError: If the data is not a valid rubric, ``"guidelines"`` is not a string,
                or guidelines are given both beside and inside ``"rubric"``.
        """
        return cls._from_data(data)
