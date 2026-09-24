"""Tests for LLMConfig class."""

import tempfile
from dataclasses import MISSING, asdict, fields
from pathlib import Path

import pytest
import yaml

from autorubric.llm import LLMConfig, ThinkingConfig, ThinkingLevel


class TestLLMConfigCreation:
    """Tests for LLMConfig initialization."""

    def test_create_with_required_model(self):
        """LLMConfig can be created with only the required model parameter."""
        config = LLMConfig(model="openai/gpt-5.2")
        assert config.model == "openai/gpt-5.2"

    def test_create_with_all_parameters(self):
        """LLMConfig can be created with all parameters."""
        config = LLMConfig(
            model="anthropic/claude-sonnet-4-5-20250929",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            timeout=120.0,
            max_retries=5,
            retry_min_wait=2.0,
            retry_max_wait=120.0,
            max_parallel_requests=8,
            cache_enabled=True,
            cache_dir="/tmp/test_cache",
            cache_ttl=7200,
            api_key="test-key",
            api_base="https://api.example.com",
            thinking=20000,
            prompt_caching=True,
            seed=42,
            extra_headers={"X-Custom": "header"},
            extra_params={"custom_param": "value"},
        )
        assert config.model == "anthropic/claude-sonnet-4-5-20250929"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.top_p == 0.9
        assert config.timeout == 120.0
        assert config.max_retries == 5
        assert config.retry_min_wait == 2.0
        assert config.retry_max_wait == 120.0
        assert config.max_parallel_requests == 8
        assert config.cache_enabled is True
        assert config.cache_dir == "/tmp/test_cache"
        assert config.cache_ttl == 7200
        assert config.api_key == "test-key"
        assert config.api_base == "https://api.example.com"
        assert config.thinking == 20000
        assert config.prompt_caching is True
        assert config.seed == 42
        assert config.extra_headers == {"X-Custom": "header"}
        assert config.extra_params == {"custom_param": "value"}


class TestLLMConfigDefaults:
    """Tests for LLMConfig default values."""

    def test_all_default_values(self):
        """All optional parameters have correct default values."""
        config = LLMConfig(model="test-model")

        assert config.temperature is None
        assert config.max_tokens is None
        assert config.top_p is None
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_min_wait == 1.0
        assert config.retry_max_wait == 60.0
        assert config.max_parallel_requests is None
        assert config.cache_enabled is False
        assert config.cache_dir == ".autorubric_cache"
        assert config.cache_ttl is None
        assert config.api_key is None
        assert config.api_base is None
        assert config.thinking is None
        assert config.prompt_caching is True  # Enabled by default for supported models
        assert config.seed is None
        assert config.extra_headers == {}
        assert config.extra_params == {}


class TestLLMConfigFromYaml:
    """Tests for LLMConfig.from_yaml() method."""

    def test_from_yaml_loads_valid_file(self):
        """LLMConfig.from_yaml() loads configuration from a valid YAML file."""
        config_data = {
            "model": "openai/gpt-5.2",
            "temperature": 0.5,
            "max_tokens": 2048,
            "cache_enabled": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = f.name

        try:
            config = LLMConfig.from_yaml(temp_path)
            assert config.model == "openai/gpt-5.2"
            assert config.temperature == 0.5
            assert config.max_tokens == 2048
            assert config.cache_enabled is True
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_raises_file_not_found_for_missing_file(self):
        """LLMConfig.from_yaml() raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="LLM config file not found"):
            LLMConfig.from_yaml("/non/existent/path/config.yaml")

    def test_from_yaml_raises_value_error_for_missing_model(self):
        """LLMConfig.from_yaml() raises ValueError when model field is missing."""
        config_data = {
            "temperature": 0.5,
            "max_tokens": 2048,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must specify 'model' field"):
                LLMConfig.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_raises_value_error_for_invalid_yaml(self):
        """LLMConfig.from_yaml() raises ValueError for non-dict YAML content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")  # List instead of dict
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="expected dict"):
                LLMConfig.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_unknown_keys_go_to_extra_params(self):
        """Unknown YAML keys are added to extra_params."""
        config_data = {
            "model": "openai/gpt-5.2",
            "unknown_key": "some_value",
            "another_unknown": 42,
            "custom_setting": {"nested": "value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = f.name

        try:
            config = LLMConfig.from_yaml(temp_path)
            assert config.model == "openai/gpt-5.2"
            assert config.extra_params == {
                "unknown_key": "some_value",
                "another_unknown": 42,
                "custom_setting": {"nested": "value"},
            }
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_preserves_existing_extra_params(self):
        """Extra params from YAML are merged with unknown keys."""
        config_data = {
            "model": "openai/gpt-5.2",
            "extra_params": {"existing": "param"},
            "unknown_key": "value",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = f.name

        try:
            config = LLMConfig.from_yaml(temp_path)
            assert config.extra_params == {
                "existing": "param",
                "unknown_key": "value",
            }
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_loads_max_parallel_requests(self):
        """max_parallel_requests is a config field, not a provider param for extra_params."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"model": "m", "max_parallel_requests": 4}, f)
            temp_path = f.name

        try:
            config = LLMConfig.from_yaml(temp_path)
            assert config.max_parallel_requests == 4
            assert config.extra_params == {}
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_accepts_path_object(self):
        """LLMConfig.from_yaml() accepts Path objects."""
        config_data = {"model": "test-model"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = LLMConfig.from_yaml(temp_path)
            assert config.model == "test-model"
        finally:
            temp_path.unlink()


class TestLLMConfigToYaml:
    """Tests for LLMConfig.to_yaml() method."""

    def test_to_yaml_saves_configuration(self):
        """LLMConfig.to_yaml() saves configuration to a YAML file."""
        config = LLMConfig(
            model="openai/gpt-5.2",
            temperature=0.7,
            max_tokens=1024,
            cache_enabled=True,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.to_yaml(temp_path)

            with open(temp_path, encoding="utf-8") as f:
                saved_data = yaml.safe_load(f)

            assert saved_data["model"] == "openai/gpt-5.2"
            assert saved_data["temperature"] == 0.7
            assert saved_data["max_tokens"] == 1024
            assert saved_data["cache_enabled"] is True
        finally:
            Path(temp_path).unlink()

    def test_to_yaml_omits_none_values(self):
        """LLMConfig.to_yaml() omits None values from the output."""
        config = LLMConfig(model="openai/gpt-5.2")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.to_yaml(temp_path)

            with open(temp_path, encoding="utf-8") as f:
                saved_data = yaml.safe_load(f)

            # These have None defaults and should be omitted
            assert "max_tokens" not in saved_data
            assert "top_p" not in saved_data
            assert "cache_ttl" not in saved_data
            assert "api_key" not in saved_data
        finally:
            Path(temp_path).unlink()

    def test_to_yaml_omits_empty_dicts(self):
        """LLMConfig.to_yaml() omits empty dict values from the output."""
        config = LLMConfig(model="openai/gpt-5.2")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.to_yaml(temp_path)

            with open(temp_path, encoding="utf-8") as f:
                saved_data = yaml.safe_load(f)

            # Empty dicts should be omitted
            assert "extra_headers" not in saved_data
            assert "extra_params" not in saved_data
        finally:
            Path(temp_path).unlink()

    def test_to_yaml_accepts_path_object(self):
        """LLMConfig.to_yaml() accepts Path objects."""
        config = LLMConfig(model="test-model")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            config.to_yaml(temp_path)

            with open(temp_path, encoding="utf-8") as f:
                saved_data = yaml.safe_load(f)

            assert saved_data["model"] == "test-model"
        finally:
            temp_path.unlink()

    def test_roundtrip_yaml(self):
        """Configuration can be saved and loaded back."""
        original = LLMConfig(
            model="anthropic/claude-sonnet-4-5-20250929",
            temperature=0.5,
            max_tokens=2048,
            cache_enabled=True,
            thinking="high",
            extra_params={"custom": "value"},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            original.to_yaml(temp_path)
            loaded = LLMConfig.from_yaml(temp_path)

            assert loaded.model == original.model
            assert loaded.temperature == original.temperature
            assert loaded.max_tokens == original.max_tokens
            assert loaded.cache_enabled == original.cache_enabled
            assert loaded.thinking == original.thinking
            assert loaded.extra_params == original.extra_params
        finally:
            Path(temp_path).unlink()

    def test_roundtrip_yaml_every_field_non_default(self):
        """Every LLMConfig field survives to_yaml/from_yaml when set to a non-default value."""
        original = LLMConfig(
            model="anthropic/claude-sonnet-4-5-20250929",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            timeout=120.0,
            max_retries=5,
            retry_min_wait=2.0,
            retry_max_wait=120.0,
            max_parallel_requests=4,
            cache_enabled=True,
            cache_dir="/tmp/test_cache",
            cache_ttl=7200,
            api_key="test-key",
            api_base="https://api.example.com",
            thinking="high",
            prompt_caching=False,
            seed=42,
            extra_headers={"X-Custom": "header"},
            extra_params={"custom_param": "value"},
        )

        # Guard: a field added to LLMConfig later must be set here too, or this
        # test would silently stop covering it.
        for config_field in fields(LLMConfig):
            if config_field.default is not MISSING:
                default = config_field.default
            elif config_field.default_factory is not MISSING:
                default = config_field.default_factory()
            else:
                continue
            name = config_field.name
            assert getattr(original, name) != default, f"{name} is left at its default"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            original.to_yaml(temp_path)
            loaded = LLMConfig.from_yaml(temp_path)
            assert loaded == original
        finally:
            Path(temp_path).unlink()


class TestLLMConfigTemperatureSerialization:
    """Serialization of the optional temperature (None = provider default)."""

    def test_from_yaml_loads_explicit_zero_temperature(self, tmp_path):
        """A YAML file with temperature: 0.0 loads as 0.0, not as the None default."""
        path = tmp_path / "llm_config.yaml"
        path.write_text("model: openai/gpt-5.2\ntemperature: 0.0\n", encoding="utf-8")

        config = LLMConfig.from_yaml(path)

        assert config.temperature == 0.0
        assert config.temperature is not None

    def test_default_temperature_roundtrips_through_yaml(self, tmp_path):
        """to_yaml omits a None temperature, and from_yaml restores the None default."""
        path = tmp_path / "llm_config.yaml"
        LLMConfig(model="openai/gpt-5.2").to_yaml(path)

        saved_data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "temperature" not in saved_data

        assert LLMConfig.from_yaml(path).temperature is None

    def test_asdict_with_explicit_zero_temperature_reloads_as_zero(self):
        """Previously serialized configs carrying temperature 0.0 keep it."""
        data = asdict(LLMConfig(model="m", temperature=0.0))

        assert LLMConfig(**data).temperature == 0.0


class TestLLMConfigThinkingSerialization:
    """YAML serialization of every ThinkingParam form."""

    @pytest.mark.parametrize(
        ("thinking", "yaml_value", "loaded_type"),
        [
            pytest.param("high", "high", str, id="level-string"),
            pytest.param(32000, 32000, int, id="int-budget"),
            # A ThinkingLevel is written as its string value and loads back as that plain
            # string: equal to the member, same get_thinking_config().
            pytest.param(ThinkingLevel.HIGH, "high", str, id="level-enum"),
            pytest.param(ThinkingLevel.NONE, "none", str, id="level-enum-none"),
            pytest.param(
                ThinkingConfig(level=ThinkingLevel.HIGH),
                {"level": "high", "budget_tokens": None},
                ThinkingConfig,
                id="config-level",
            ),
            pytest.param(
                ThinkingConfig(budget_tokens=9000),
                {"level": "medium", "budget_tokens": 9000},
                ThinkingConfig,
                id="config-budget",
            ),
            pytest.param(
                ThinkingConfig(level="low", budget_tokens=500),
                {"level": "low", "budget_tokens": 500},
                ThinkingConfig,
                id="config-level-and-budget",
            ),
        ],
    )
    def test_thinking_roundtrips_through_yaml(self, tmp_path, thinking, yaml_value, loaded_type):
        """Each thinking form is written as plain YAML and loads back equivalent."""
        path = tmp_path / "llm_config.yaml"
        original = LLMConfig(model="m", thinking=thinking)

        original.to_yaml(path)

        assert yaml.safe_load(path.read_text(encoding="utf-8"))["thinking"] == yaml_value
        loaded = LLMConfig.from_yaml(path)
        assert type(loaded.thinking) is loaded_type
        assert loaded.get_thinking_config() == original.get_thinking_config()
        assert loaded == original

    def test_from_yaml_loads_hand_written_thinking_mapping(self, tmp_path):
        """A thinking mapping may omit fields; omitted ones take ThinkingConfig defaults."""
        path = tmp_path / "llm_config.yaml"
        path.write_text("model: m\nthinking:\n  budget_tokens: 32000\n", encoding="utf-8")

        config = LLMConfig.from_yaml(path)

        assert config.thinking == ThinkingConfig(level=ThinkingLevel.MEDIUM, budget_tokens=32000)

    def test_from_yaml_rejects_unknown_thinking_mapping_key(self, tmp_path):
        """An unknown key in the thinking mapping is a config error, not a later TypeError."""
        path = tmp_path / "llm_config.yaml"
        path.write_text("model: m\nthinking:\n  effort: high\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid 'thinking' mapping"):
            LLMConfig.from_yaml(path)

    def test_to_yaml_output_unchanged_for_plain_values(self, tmp_path):
        """Configs that serialized before enum support still produce the same bytes."""
        path = tmp_path / "llm_config.yaml"
        config = LLMConfig(model="m", thinking="high", cache_dir=Path("/tmp/autorubric_cache"))

        config.to_yaml(path)

        assert path.read_text(encoding="utf-8") == (
            "model: m\n"
            "timeout: 60.0\n"
            "max_retries: 3\n"
            "retry_min_wait: 1.0\n"
            "retry_max_wait: 60.0\n"
            "cache_enabled: false\n"
            "cache_dir: /tmp/autorubric_cache\n"
            "thinking: high\n"
            "prompt_caching: true\n"
        )
