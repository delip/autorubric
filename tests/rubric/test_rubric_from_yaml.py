import pytest
import yaml

from autorubric import Rubric

VALID_CRITERIA = [
    {"weight": 1.0, "requirement": "First requirement"},
    {"weight": 2.0, "requirement": "Second requirement"},
]


def test_from_yaml_string():
    yaml_string = """
- weight: 1.0
  requirement: First requirement
- weight: 2.0
  requirement: Second requirement
"""
    rubric = Rubric.from_yaml(yaml_string)
    assert len(rubric.rubric) == 2
    assert rubric.rubric[0].weight == 1.0
    assert rubric.rubric[1].weight == 2.0


def test_from_yaml_invalid_yaml():
    invalid_yaml = "{ invalid: yaml: content"
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_yaml(invalid_yaml)
    assert "Failed to parse YAML" in str(exc_info.value)


def test_from_yaml_invalid_criteria():
    yaml_string = """
- weight: 1.0
  requirement: Valid criterion
- weight: invalid_weight
  requirement: Invalid criterion
"""
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_yaml(yaml_string)
    assert "Invalid criterion at index 1" in str(exc_info.value)


def test_from_yaml_empty_list():
    yaml_string = "[]"
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_yaml(yaml_string)
    assert "No criteria found" in str(exc_info.value)


def test_from_yaml_not_list():
    yaml_string = "weight: 1.0\nrequirement: test"
    with pytest.raises(ValueError) as exc_info:
        Rubric.from_yaml(yaml_string)
    assert "Dict must contain either 'sections' or 'rubric' key" in str(exc_info.value)


def test_from_yaml_with_extra_fields():
    yaml_string = """
- weight: 1.0
  requirement: Test requirement
  extra_field: This should be ignored
"""
    rubric = Rubric.from_yaml(yaml_string)
    assert len(rubric.rubric) == 1
    assert rubric.rubric[0].weight == 1.0


def test_from_yaml_multiple_criteria():
    criteria = [{"weight": float(i), "requirement": f"Requirement {i}"} for i in range(1, 11)]
    yaml_string = yaml.dump(criteria)
    rubric = Rubric.from_yaml(yaml_string)
    assert len(rubric.rubric) == 10
    for i, criterion in enumerate(rubric.rubric, 1):
        assert criterion.weight == float(i)
        assert criterion.requirement == f"Requirement {i}"


def test_from_yaml_with_criterion_names():
    """Test that Criterion.name field is loaded from YAML."""
    yaml_string = """
- name: accuracy
  weight: 2.0
  requirement: Is accurate
- name: clarity
  weight: 1.0
  requirement: Is clear
- weight: 1.0
  requirement: No name criterion
"""
    rubric = Rubric.from_yaml(yaml_string)
    assert len(rubric.rubric) == 3
    assert rubric.rubric[0].name == "accuracy"
    assert rubric.rubric[0].weight == 2.0
    assert rubric.rubric[1].name == "clarity"
    assert rubric.rubric[1].weight == 1.0
    assert rubric.rubric[2].name is None  # No name provided


def test_from_yaml_without_weight_uses_default():
    """Weight is optional and defaults to 10.0."""
    yaml_string = """
- requirement: First requirement without weight
- requirement: Second requirement without weight
"""
    rubric = Rubric.from_yaml(yaml_string)
    assert len(rubric.rubric) == 2
    assert rubric.rubric[0].weight == 10.0
    assert rubric.rubric[0].requirement == "First requirement without weight"
    assert rubric.rubric[1].weight == 10.0
    assert rubric.rubric[1].requirement == "Second requirement without weight"


def test_from_yaml_mixed_weights():
    """Test that some criteria can have explicit weight while others use default."""
    yaml_string = """
- weight: 5.0
  requirement: Has explicit weight
- requirement: Uses default weight
- weight: -10.0
  requirement: Negative weight criterion
"""
    rubric = Rubric.from_yaml(yaml_string)
    assert len(rubric.rubric) == 3
    assert rubric.rubric[0].weight == 5.0
    assert rubric.rubric[1].weight == 10.0  # Default
    assert rubric.rubric[2].weight == -10.0
