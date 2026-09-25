"""Rubric-level guidelines: the ``Rubric.guidelines`` model and the rubric file formats.

Guidelines are free text that applies to every criterion of a rubric. A rubric without
guidelines behaves, loads and serializes exactly as before they existed; the formats that
carry them (``{"guidelines": ..., "criteria": [...]}``, and ``"guidelines"`` beside
``"sections"`` or ``"rubric"``) are additive.
"""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import pickle
from typing import Any

import pytest
import yaml

from autorubric import Criterion, Rubric

GUIDELINES = (
    "Writers are English-language learners in grades 8-12; judge against that level.\n"
    "'Cited' means any attribution, not formal citation style."
)
CRITERIA_DATA = [
    {"name": "thesis", "weight": 3.0, "requirement": "States a clear, arguable thesis"},
    {"name": "evidence", "weight": 2.0, "requirement": "Supports claims with cited evidence"},
]
SECTIONS_DATA = [
    {"name": "Argument", "criteria": [CRITERIA_DATA[0]]},
    {"name": "Support", "criteria": [CRITERIA_DATA[1]]},
]


def _criteria() -> list[Criterion]:
    return [Criterion(**c) for c in CRITERIA_DATA]


def _requirements(rubric: Rubric) -> list[str]:
    return [c.requirement for c in rubric.rubric]


EXPECTED_REQUIREMENTS = [c["requirement"] for c in CRITERIA_DATA]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestGuidelinesModel:
    def test_default_is_none(self) -> None:
        rubric = Rubric(_criteria())
        assert rubric.guidelines is None

    def test_keyword_sets_attribute(self) -> None:
        rubric = Rubric(_criteria(), guidelines=GUIDELINES)
        assert rubric.guidelines == GUIDELINES

    def test_first_parameter_keeps_its_name(self) -> None:
        rubric = Rubric(rubric=_criteria(), guidelines=GUIDELINES)
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS
        assert rubric.guidelines == GUIDELINES

    def test_guidelines_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            Rubric(_criteria(), GUIDELINES)  # type: ignore[misc]

    def test_text_is_kept_verbatim(self) -> None:
        text = "  Leading and trailing whitespace is kept.\n\n"
        assert Rubric(_criteria(), guidelines=text).guidelines == text

    @pytest.mark.parametrize("blank", ["", " ", "\n", " \t\r\n "])
    def test_blank_text_means_no_guidelines(self, blank: str) -> None:
        assert Rubric(_criteria(), guidelines=blank).guidelines is None

    @pytest.mark.parametrize("bad", [42, ["a"], {"text": "a"}, b"bytes"])
    def test_non_string_raises_type_error(self, bad: object) -> None:
        with pytest.raises(TypeError, match="guidelines must be a str or None, got"):
            Rubric(_criteria(), guidelines=bad)  # type: ignore[arg-type]

    def test_assignment_applies_the_same_rules(self) -> None:
        rubric = Rubric(_criteria())
        rubric.guidelines = GUIDELINES
        assert rubric.guidelines == GUIDELINES
        rubric.guidelines = "   "
        assert rubric.guidelines is None
        with pytest.raises(TypeError):
            rubric.guidelines = 3  # type: ignore[assignment]

    def test_copy_deepcopy_and_pickle_keep_guidelines(self) -> None:
        rubric = Rubric(_criteria(), guidelines=GUIDELINES)
        for clone in (
            copy.copy(rubric),
            copy.deepcopy(rubric),
            pickle.loads(pickle.dumps(rubric)),
        ):
            assert clone.guidelines == GUIDELINES
            assert _requirements(clone) == EXPECTED_REQUIREMENTS

    def test_rubric_pickled_before_guidelines_existed_reads_as_none(self) -> None:
        """An instance state without the guidelines attribute (an older pickle) has none."""
        restored = Rubric.__new__(Rubric)
        restored.__dict__.update({"rubric": _criteria()})
        assert restored.guidelines is None


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------


class TestGuidelinesFormats:
    def test_list_form_has_no_guidelines(self) -> None:
        rubric = Rubric.from_dict(CRITERIA_DATA)
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS
        assert rubric.guidelines is None

    @pytest.mark.parametrize(
        "data",
        [
            {"sections": SECTIONS_DATA},
            {"rubric": CRITERIA_DATA},
            {"rubric": SECTIONS_DATA},
            {"rubric": {"sections": SECTIONS_DATA}},
        ],
        ids=["sections", "rubric-list", "rubric-sections-list", "rubric-sections-dict"],
    )
    def test_legacy_dict_forms_have_no_guidelines(self, data: dict) -> None:
        rubric = Rubric.from_dict(data)
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS
        assert rubric.guidelines is None

    def test_criteria_form_alone(self) -> None:
        rubric = Rubric.from_dict({"criteria": CRITERIA_DATA})
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS
        assert rubric.guidelines is None

    @pytest.mark.parametrize(
        "data",
        [
            {"guidelines": GUIDELINES, "criteria": CRITERIA_DATA},
            {"criteria": CRITERIA_DATA, "guidelines": GUIDELINES},
            {"guidelines": GUIDELINES, "criteria": SECTIONS_DATA},
            {"guidelines": GUIDELINES, "sections": SECTIONS_DATA},
            {"guidelines": GUIDELINES, "rubric": CRITERIA_DATA},
            {"guidelines": GUIDELINES, "rubric": {"sections": SECTIONS_DATA}},
            {"rubric": {"guidelines": GUIDELINES, "sections": SECTIONS_DATA}},
            {"rubric": {"guidelines": GUIDELINES, "criteria": CRITERIA_DATA}},
            {"id": "r1", "guidelines": GUIDELINES, "criteria": CRITERIA_DATA, "title": "t"},
        ],
        ids=[
            "guidelines+criteria",
            "criteria+guidelines-key-order",
            "guidelines+criteria-of-sections",
            "guidelines+sections",
            "guidelines+rubric-list",
            "guidelines+rubric-sections-dict",
            "rubric-dict-with-guidelines+sections",
            "rubric-dict-with-guidelines+criteria",
            "extra-keys-ignored",
        ],
    )
    def test_forms_with_guidelines(self, data: dict) -> None:
        rubric = Rubric.from_dict(data)
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS
        assert rubric.guidelines == GUIDELINES

    def test_null_guidelines_means_none(self) -> None:
        rubric = Rubric.from_dict({"guidelines": None, "criteria": CRITERIA_DATA})
        assert rubric.guidelines is None

    def test_blank_guidelines_in_a_file_means_none(self) -> None:
        rubric = Rubric.from_dict({"guidelines": "  \n", "criteria": CRITERIA_DATA})
        assert rubric.guidelines is None

    @pytest.mark.parametrize("bad", [42, 1.5, True, ["a"], {"text": "a"}])
    def test_non_string_guidelines_raises_value_error(self, bad: object) -> None:
        with pytest.raises(ValueError, match="Expected 'guidelines' to be a string, got"):
            Rubric.from_dict({"guidelines": bad, "criteria": CRITERIA_DATA})

    def test_non_string_guidelines_inside_rubric_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Expected 'guidelines' to be a string, got int"):
            Rubric.from_dict({"rubric": {"guidelines": 7, "sections": SECTIONS_DATA}})

    def test_guidelines_beside_and_inside_rubric_raises(self) -> None:
        data = {
            "guidelines": GUIDELINES,
            "rubric": {"guidelines": GUIDELINES, "sections": SECTIONS_DATA},
        }
        with pytest.raises(ValueError, match="both beside and inside 'rubric'"):
            Rubric.from_dict(data)

    def test_blank_guidelines_at_one_level_do_not_conflict(self) -> None:
        data = {"guidelines": "", "rubric": {"guidelines": GUIDELINES, "criteria": CRITERIA_DATA}}
        assert Rubric.from_dict(data).guidelines == GUIDELINES

    def test_criteria_must_be_a_list(self) -> None:
        with pytest.raises(ValueError, match="Expected 'criteria' to be a list, got dict"):
            Rubric.from_dict({"guidelines": GUIDELINES, "criteria": {"a": 1}})

    def test_empty_criteria_raises(self) -> None:
        with pytest.raises(ValueError, match="No criteria found"):
            Rubric.from_dict({"guidelines": GUIDELINES, "criteria": []})

    def test_guidelines_without_criteria_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Rubric.from_dict({"guidelines": GUIDELINES})
        message = str(exc_info.value)
        assert "Dict must contain either 'sections' or 'rubric' key" in message
        assert "'criteria'" in message

    def test_rubric_dict_without_criteria_raises(self) -> None:
        with pytest.raises(ValueError, match="'rubric' must be a list, or a dict with"):
            Rubric.from_dict({"rubric": {"guidelines": GUIDELINES}})

    def test_existing_key_precedence_is_unchanged(self) -> None:
        """'rubric' wins over 'sections', which wins over 'criteria', as before."""
        other = [{"weight": 1.0, "requirement": "Ignored"}]
        assert (
            _requirements(
                Rubric.from_dict({"rubric": CRITERIA_DATA, "sections": other, "criteria": other})
            )
            == EXPECTED_REQUIREMENTS
        )
        assert (
            _requirements(Rubric.from_dict({"sections": SECTIONS_DATA, "criteria": other}))
            == EXPECTED_REQUIREMENTS
        )

    def test_validate_and_create_criteria_accepts_every_form(self) -> None:
        for data in (
            CRITERIA_DATA,
            {"guidelines": GUIDELINES, "criteria": CRITERIA_DATA},
            {"guidelines": GUIDELINES, "sections": SECTIONS_DATA},
            {"rubric": {"guidelines": GUIDELINES, "sections": SECTIONS_DATA}},
        ):
            criteria = Rubric.validate_and_create_criteria(data)
            assert [c.requirement for c in criteria] == EXPECTED_REQUIREMENTS

    def test_validate_and_create_criteria_validates_guidelines(self) -> None:
        with pytest.raises(ValueError, match="Expected 'guidelines' to be a string"):
            Rubric.validate_and_create_criteria({"guidelines": 1, "criteria": CRITERIA_DATA})


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

DICT_FORM = {"guidelines": GUIDELINES, "criteria": CRITERIA_DATA}


class TestGuidelinesLoaders:
    def test_from_json(self) -> None:
        rubric = Rubric.from_json(json.dumps(DICT_FORM))
        assert rubric.guidelines == GUIDELINES
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS

    def test_from_yaml(self) -> None:
        rubric = Rubric.from_yaml(yaml.safe_dump(DICT_FORM))
        assert rubric.guidelines == GUIDELINES
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS

    def test_from_yaml_block_scalar_keeps_text(self) -> None:
        text = """\
guidelines: |
  Judge against a grade 8-12 level.
  Any attribution counts as a citation.
sections:
  - name: Argument
    criteria:
      - name: thesis
        weight: 3
        requirement: States a clear, arguable thesis
"""
        rubric = Rubric.from_yaml(text)
        assert rubric.guidelines == (
            "Judge against a grade 8-12 level.\nAny attribution counts as a citation.\n"
        )
        assert _requirements(rubric) == ["States a clear, arguable thesis"]

    @pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
    def test_from_file_path(self, tmp_path, suffix: str) -> None:
        path = tmp_path / f"rubric{suffix}"
        dump = json.dumps if suffix == ".json" else yaml.safe_dump
        path.write_text(dump(DICT_FORM), encoding="utf-8")
        rubric = Rubric.from_file(str(path))
        assert rubric.guidelines == GUIDELINES
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS

    @pytest.mark.parametrize("suffix", [".json", ".yaml"])
    def test_from_file_object(self, suffix: str) -> None:
        dump = json.dumps if suffix == ".json" else yaml.safe_dump
        handle = io.StringIO(dump(DICT_FORM))
        handle.name = f"rubric{suffix}"
        rubric = Rubric.from_file(handle)
        assert rubric.guidelines == GUIDELINES
        assert _requirements(rubric) == EXPECTED_REQUIREMENTS

    def test_from_file_list_form_has_no_guidelines(self, tmp_path) -> None:
        path = tmp_path / "rubric.json"
        path.write_text(json.dumps(CRITERIA_DATA), encoding="utf-8")
        assert Rubric.from_file(str(path)).guidelines is None

    def test_invalid_guidelines_in_file_raises_value_error(self, tmp_path) -> None:
        path = tmp_path / "rubric.yaml"
        path.write_text(yaml.safe_dump({"guidelines": [1, 2], "criteria": CRITERIA_DATA}))
        with pytest.raises(ValueError, match="Expected 'guidelines' to be a string, got list"):
            Rubric.from_file(str(path))


# ---------------------------------------------------------------------------
# Subclasses: the loaders' extension points are unchanged
# ---------------------------------------------------------------------------


class LegacyInitRubric(Rubric):
    """A subclass written before guidelines existed: its ``__init__`` takes the criteria only."""

    def __init__(self, rubric: list[Criterion]) -> None:
        super().__init__(rubric)
        self.tag = "legacy"


class GuidelinesInitRubric(Rubric):
    """A subclass that takes guidelines in its own ``__init__`` and records them."""

    def __init__(self, rubric: list[Criterion], *, guidelines: str | None = None) -> None:
        self.init_guidelines = guidelines
        super().__init__(rubric, guidelines=guidelines)


@dataclasses.dataclass
class _Options:
    owner: str = "me"


class ForwardingKwargsRubric(Rubric):
    """Written before guidelines existed: its ``**options`` go to a strict consumer, which
    rejects any keyword it does not know."""

    def __init__(self, rubric: list[Criterion], **options: Any) -> None:
        super().__init__(rubric)
        self.options = _Options(**options)


class SwallowingKwargsRubric(Rubric):
    """Written before guidelines existed: it keeps whatever ``**options`` it is given."""

    def __init__(self, rubric: list[Criterion], **options: Any) -> None:
        super().__init__(rubric)
        self.options = options


class NoSuperInitKwargsRubric(Rubric):
    """Takes ``**options`` and never calls ``Rubric.__init__``: it sets the criteria itself."""

    def __init__(self, rubric: list[Criterion], **options: Any) -> None:
        self.rubric = rubric
        self.options = options


class DefaultGuidelinesRubric(Rubric):
    """Supplies its own guidelines through ``**kwargs`` when none are given."""

    def __init__(self, rubric: list[Criterion], **kwargs: Any) -> None:
        kwargs.setdefault("guidelines", "House style.")
        super().__init__(rubric, **kwargs)


class TransformingGuidelinesRubric(Rubric):
    """Declares ``guidelines`` and passes them on transformed."""

    def __init__(self, rubric: list[Criterion], *, guidelines: str | None = None) -> None:
        super().__init__(rubric, guidelines=None if guidelines is None else guidelines.upper())


# The state each ``**options`` subclass's ``__init__`` builds when given no options.
KWARGS_SUBCLASSES = {
    ForwardingKwargsRubric: _Options(),
    SwallowingKwargsRubric: {},
    NoSuperInitKwargsRubric: {},
}


class ItemsFormatRubric(Rubric):
    """A subclass that parses its own file shape, ``{"items": [...]}``, through the hook."""

    @staticmethod
    def validate_and_create_criteria(data: Any) -> list[Criterion]:
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        return Rubric.validate_and_create_criteria(data)


class RecordingFromDictRubric(Rubric):
    """A subclass whose ``from_dict`` override records each call."""

    from_dict_calls: list[Any] = []

    @classmethod
    def from_dict(cls, data: Any) -> Rubric:
        cls.from_dict_calls.append(data)
        return super().from_dict(data)


def _load_every_way(cls: type[Rubric], data: Any, tmp_path) -> dict[str, Rubric]:
    """Load ``data`` through every loader of ``cls``: dict, JSON, YAML, file path, file object."""
    loaded = {
        "from_dict": cls.from_dict(data),
        "from_json": cls.from_json(json.dumps(data)),
        "from_yaml": cls.from_yaml(yaml.safe_dump(data)),
    }
    for suffix in (".json", ".yaml"):
        dump = json.dumps if suffix == ".json" else yaml.safe_dump
        path = tmp_path / f"rubric{suffix}"
        path.write_text(dump(data), encoding="utf-8")
        loaded[f"from_file-path{suffix}"] = cls.from_file(str(path))
        handle = io.StringIO(dump(data))
        handle.name = f"rubric{suffix}"
        loaded[f"from_file-object{suffix}"] = cls.from_file(handle)
    return loaded


# The file forms without and with guidelines that every subclass must load.
FORMS_WITHOUT_GUIDELINES = pytest.mark.parametrize(
    "data",
    [CRITERIA_DATA, {"sections": SECTIONS_DATA}, {"rubric": {"sections": SECTIONS_DATA}}],
    ids=["list", "sections", "rubric-sections-dict"],
)
FORMS_WITH_GUIDELINES = pytest.mark.parametrize(
    "data",
    [
        {"guidelines": GUIDELINES, "sections": SECTIONS_DATA},
        {"guidelines": GUIDELINES, "rubric": CRITERIA_DATA},
        {"guidelines": GUIDELINES, "rubric": {"sections": SECTIONS_DATA}},
        {"rubric": {"guidelines": GUIDELINES, "sections": SECTIONS_DATA}},
        DICT_FORM,
    ],
    ids=["sections", "rubric-list", "rubric-sections-dict", "inside-rubric", "criteria"],
)


class TestSubclassLoading:
    @FORMS_WITHOUT_GUIDELINES
    def test_subclass_with_legacy_init_loads_rubrics_without_guidelines(
        self, data: Any, tmp_path
    ) -> None:
        """``guidelines=`` reaches ``cls`` only when there are guidelines."""
        for how, rubric in _load_every_way(LegacyInitRubric, data, tmp_path).items():
            assert type(rubric) is LegacyInitRubric, how
            assert _requirements(rubric) == EXPECTED_REQUIREMENTS, how
            assert rubric.guidelines is None, how

    @FORMS_WITH_GUIDELINES
    def test_subclass_with_legacy_init_loads_rubrics_with_guidelines(
        self, data: Any, tmp_path
    ) -> None:
        """A subclass whose ``__init__`` cannot take ``guidelines=`` is built by its own
        ``__init__`` as before, and the guidelines are then set on the rubric, so a file with
        guidelines loads with them honoured instead of raising ``TypeError``."""
        for how, rubric in _load_every_way(LegacyInitRubric, data, tmp_path).items():
            assert type(rubric) is LegacyInitRubric, how
            assert rubric.tag == "legacy", how
            assert _requirements(rubric) == EXPECTED_REQUIREMENTS, how
            assert rubric.guidelines == GUIDELINES, how

    def test_subclass_init_receives_guidelines(self, tmp_path) -> None:
        for how, rubric in _load_every_way(GuidelinesInitRubric, DICT_FORM, tmp_path).items():
            assert type(rubric) is GuidelinesInitRubric, how
            assert rubric.init_guidelines == GUIDELINES, how
            assert rubric.guidelines == GUIDELINES, how

    @pytest.mark.parametrize("cls", list(KWARGS_SUBCLASSES), ids=lambda cls: cls.__name__)
    @FORMS_WITHOUT_GUIDELINES
    def test_subclass_with_kwargs_init_loads_rubrics_without_guidelines(
        self, cls: type[Rubric], data: Any, tmp_path
    ) -> None:
        for how, rubric in _load_every_way(cls, data, tmp_path).items():
            assert type(rubric) is cls, how
            assert rubric.options == KWARGS_SUBCLASSES[cls], how
            assert _requirements(rubric) == EXPECTED_REQUIREMENTS, how
            assert rubric.guidelines is None, how

    @pytest.mark.parametrize("cls", list(KWARGS_SUBCLASSES), ids=lambda cls: cls.__name__)
    @FORMS_WITH_GUIDELINES
    def test_subclass_with_kwargs_init_loads_rubrics_with_guidelines(
        self, cls: type[Rubric], data: Any, tmp_path
    ) -> None:
        """An ``__init__`` that takes ``**kwargs`` but declares no ``guidelines`` parameter
        may predate guidelines, so it is never handed a ``guidelines`` keyword it does not
        expect (which could raise, or be kept as an option of its own). It is built with
        ``cls(criteria)`` as before, and the guidelines are then set on the rubric."""
        for how, rubric in _load_every_way(cls, data, tmp_path).items():
            assert type(rubric) is cls, how
            assert rubric.options == KWARGS_SUBCLASSES[cls], how
            assert _requirements(rubric) == EXPECTED_REQUIREMENTS, how
            assert rubric.guidelines == GUIDELINES, how

    def test_file_guidelines_override_guidelines_an_init_supplies_itself(self, tmp_path) -> None:
        """Set on the rubric after ``cls(criteria)``, a file's guidelines take the place of
        any its ``__init__`` supplies when given none, as ``guidelines=`` passed to that
        ``__init__`` would; a file without guidelines keeps them."""
        loaded = _load_every_way(DefaultGuidelinesRubric, DICT_FORM, tmp_path)
        for how, rubric in loaded.items():
            assert rubric.guidelines == GUIDELINES, how
        loaded = _load_every_way(DefaultGuidelinesRubric, CRITERIA_DATA, tmp_path)
        for how, rubric in loaded.items():
            assert rubric.guidelines == "House style.", how

    def test_subclass_init_that_declares_guidelines_decides_what_they_become(
        self, tmp_path
    ) -> None:
        loaded = _load_every_way(TransformingGuidelinesRubric, DICT_FORM, tmp_path)
        for how, rubric in loaded.items():
            assert rubric.guidelines == GUIDELINES.upper(), how

    def test_every_loader_parses_criteria_through_the_overridable_hook(self, tmp_path) -> None:
        data = {"items": CRITERIA_DATA}
        for how, rubric in _load_every_way(ItemsFormatRubric, data, tmp_path).items():
            assert type(rubric) is ItemsFormatRubric, how
            assert _requirements(rubric) == EXPECTED_REQUIREMENTS, how
            assert rubric.guidelines is None, how

    def test_guidelines_are_read_beside_a_custom_criteria_shape(self, tmp_path) -> None:
        data = {"guidelines": GUIDELINES, "items": CRITERIA_DATA}
        for how, rubric in _load_every_way(ItemsFormatRubric, data, tmp_path).items():
            assert _requirements(rubric) == EXPECTED_REQUIREMENTS, how
            assert rubric.guidelines == GUIDELINES, how

    def test_file_loaders_do_not_dispatch_through_from_dict(self, tmp_path) -> None:
        RecordingFromDictRubric.from_dict_calls = []
        path = tmp_path / "rubric.json"
        path.write_text(json.dumps(DICT_FORM), encoding="utf-8")
        handle = io.StringIO(yaml.safe_dump(DICT_FORM))
        handle.name = "rubric.yaml"

        loaded = [
            RecordingFromDictRubric.from_json(json.dumps(DICT_FORM)),
            RecordingFromDictRubric.from_yaml(yaml.safe_dump(DICT_FORM)),
            RecordingFromDictRubric.from_file(str(path)),
            RecordingFromDictRubric.from_file(handle),
        ]

        assert RecordingFromDictRubric.from_dict_calls == []
        for rubric in loaded:
            assert type(rubric) is RecordingFromDictRubric
            assert rubric.guidelines == GUIDELINES
