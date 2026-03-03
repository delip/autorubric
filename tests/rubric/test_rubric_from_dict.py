import pytest

from autorubric import Rubric

VALID_CRITERIA = [
    {"weight": 1.0, "requirement": "First requirement"},
    {"weight": 2.0, "requirement": "Second requirement"},
]


def test_from_dict_valid():
    rubric = Rubric.from_dict(VALID_CRITERIA)
    assert len(rubric.rubric) == 2
    assert rubric.rubric[0].weight == 1.0
    assert rubric.rubric[0].requirement == "First requirement"
    assert rubric.rubric[1].weight == 2.0
    assert rubric.rubric[1].requirement == "Second requirement"


def test_from_dict_empty_list():
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_dict([])
    assert "No criteria found" in str(exc_info.value)


def test_from_dict_not_list():
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_dict({"weight": 1.0, "requirement": "test"})  # type: ignore
    assert "Dict must contain either 'sections' or 'rubric' key" in str(exc_info.value)


def test_from_dict_missing_weight_uses_default():
    """Weight is optional and defaults to 10.0."""
    data = [{"requirement": "Missing weight, should use default"}]
    rubric = Rubric.from_dict(data)
    assert len(rubric.rubric) == 1
    assert rubric.rubric[0].weight == 10.0
    assert rubric.rubric[0].requirement == "Missing weight, should use default"


def test_from_dict_uniform_weighting():
    """Test that omitting weight from all criteria gives uniform weighting."""
    data = [
        {"requirement": "First requirement"},
        {"requirement": "Second requirement"},
        {"requirement": "Third requirement"},
    ]
    rubric = Rubric.from_dict(data)
    assert len(rubric.rubric) == 3
    # All should have default weight of 10.0
    assert all(c.weight == 10.0 for c in rubric.rubric)


def test_from_dict_missing_requirement():
    invalid_data = [{"weight": 1.0}]
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_dict(invalid_data)
    assert "Invalid criterion at index 0" in str(exc_info.value)
    assert "requirement" in str(exc_info.value).lower()


def test_from_dict_invalid_weight_type():
    invalid_data = [{"weight": "invalid", "requirement": "test"}]
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_dict(invalid_data)
    assert "Invalid criterion at index 0" in str(exc_info.value)


def test_from_dict_item_not_dict():
    invalid_data = ["not a dict"]
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_dict(invalid_data)
    assert "Invalid item at index 0" in str(exc_info.value)
    assert "expected a dictionary" in str(exc_info.value)


def test_from_dict_with_sections():
    sectioned_data = {
        "rubric": {
            "id": "rubric-id",
            "sections": [
                {
                    "id": "section-id-1",
                    "title": "Section Title 1",
                    "criteria": [
                        {
                            "id": "criterion-id-1",
                            "weight": 15,
                            "requirement": "Explicitly references...",
                        },
                        {
                            "id": "criterion-id-2",
                            "weight": 8,
                            "requirement": "States volume as...",
                        },
                    ],
                },
                {
                    "id": "section-id-2",
                    "title": "Section Title 2",
                    "criteria": [
                        {
                            "id": "criterion-id-3",
                            "weight": 10,
                            "requirement": "Correctly identifies...",
                        }
                    ],
                },
            ],
        }
    }
    rubric = Rubric.from_dict(sectioned_data)
    assert len(rubric.rubric) == 3
    assert rubric.rubric[0].weight == 15
    assert rubric.rubric[0].requirement == "Explicitly references..."
    assert rubric.rubric[1].weight == 8
    assert rubric.rubric[1].requirement == "States volume as..."
    assert rubric.rubric[2].weight == 10
    assert rubric.rubric[2].requirement == "Correctly identifies..."


def test_from_dict_with_rubric_containing_sections_list():
    data = {
        "rubric": [
            {
                "id": "section-id-1",
                "criteria": [
                    {"weight": 5, "requirement": "First criterion"},
                    {"weight": 10, "requirement": "Second criterion"},
                ],
            },
            {
                "id": "section-id-2",
                "criteria": [
                    {"weight": 15, "requirement": "Third criterion"},
                ],
            },
        ]
    }
    rubric = Rubric.from_dict(data)
    assert len(rubric.rubric) == 3
    assert rubric.rubric[0].weight == 5
    assert rubric.rubric[1].weight == 10
    assert rubric.rubric[2].weight == 15


def test_from_dict_with_rubric_containing_criteria_list():
    data = {
        "rubric": [
            {"weight": 5, "requirement": "First criterion"},
            {"weight": 10, "requirement": "Second criterion"},
        ]
    }
    rubric = Rubric.from_dict(data)
    assert len(rubric.rubric) == 2
    assert rubric.rubric[0].weight == 5
    assert rubric.rubric[0].requirement == "First criterion"
    assert rubric.rubric[1].weight == 10
    assert rubric.rubric[1].requirement == "Second criterion"


def test_from_dict_with_criterion_names():
    """Test that Criterion.name field is loaded from dict."""
    data = [
        {"name": "accuracy", "weight": 2.0, "requirement": "Is accurate"},
        {"name": "clarity", "weight": 1.0, "requirement": "Is clear"},
        {"weight": 1.0, "requirement": "No name"},  # name=None
    ]
    rubric = Rubric.from_dict(data)
    assert len(rubric.rubric) == 3
    assert rubric.rubric[0].name == "accuracy"
    assert rubric.rubric[1].name == "clarity"
    assert rubric.rubric[2].name is None
