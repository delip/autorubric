import pytest

from autorubric import Rubric


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
