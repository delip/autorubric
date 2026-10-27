"""Tests for LLMConfig class."""

import tempfile
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path
from typing import cast

import numpy as np
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


class TestLLMConfigValidation:
    """``max_tokens`` and ``top_p`` are refused at construction when they mean nothing."""

    @pytest.mark.parametrize(
        "max_tokens",
        [0, -1, True, False, np.bool_(True), np.int64(0), 1.5, 1024.0, "1024", float("nan")],
    )
    def test_max_tokens_that_is_not_a_positive_integer_is_refused(self, max_tokens):
        with pytest.raises(ValueError, match=r"LLMConfig\.max_tokens must be None or an integer"):
            LLMConfig(model="openai/gpt-5.2", max_tokens=max_tokens)

    @pytest.mark.parametrize(
        "top_p",
        [
            -0.1,
            1.1,
            2,
            float("nan"),
            float("inf"),
            True,
            False,
            np.bool_(True),
            "0.9",
            complex(0.5, 0),
        ],
    )
    def test_top_p_outside_zero_to_one_is_refused(self, top_p):
        with pytest.raises(ValueError, match=r"LLMConfig\.top_p must be None or a number in"):
            LLMConfig(model="openai/gpt-5.2", top_p=top_p)

    @pytest.mark.parametrize("max_tokens", [None, 1, 1024])
    def test_valid_max_tokens_is_kept_unchanged(self, max_tokens):
        assert LLMConfig(model="openai/gpt-5.2", max_tokens=max_tokens).max_tokens == max_tokens

    @pytest.mark.parametrize("top_p", [None, 0, 0.0, 0.5, 1, 1.0])
    def test_valid_top_p_is_kept_unchanged(self, top_p):
        config = LLMConfig(model="openai/gpt-5.2", top_p=top_p)
        assert config.top_p == top_p
        assert type(config.top_p) is type(top_p)

    def test_a_numpy_integer_max_tokens_is_stored_as_int(self):
        max_tokens = cast(int, np.int64(256))  # accepted, though the annotation names only int
        config = LLMConfig(model="openai/gpt-5.2", max_tokens=max_tokens)
        assert config.max_tokens == 256
        assert type(config.max_tokens) is int

    @pytest.mark.parametrize("top_p", [np.float64(0.5), np.float32(0.5)])
    def test_a_numpy_float_top_p_is_stored_as_float(self, top_p):
        config = LLMConfig(model="openai/gpt-5.2", top_p=top_p)
        assert config.top_p == 0.5
        assert type(config.top_p) is float

    def test_replace_validates_again(self):
        config = LLMConfig(model="openai/gpt-5.2", top_p=0.5)
        with pytest.raises(ValueError, match="top_p"):
            replace(config, top_p=1.5)

    def test_from_yaml_refuses_a_zero_max_tokens(self, tmp_path):
        path = tmp_path / "llm.yaml"
        path.write_text("model: openai/gpt-5.2\nmax_tokens: 0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig.from_yaml(path)


class TestThinkingValidation:
    """``thinking`` is refused at construction when no provider can mean it.

    It used to be normalized only when a request was built, so a value such as
    ``"extreme"`` raised there, and every judge call failed as a ``parse`` abstention.
    """

    @pytest.mark.parametrize(
        "thinking",
        [
            "extreme",
            "HIGH",
            "",
            True,
            False,
            np.bool_(True),
            -2,
            -1024,
            np.int64(-2),
            32000.0,
            1.5,
            "32000",
            {"level": "high"},
            [1024],
            object(),
        ],
    )
    def test_a_thinking_value_no_provider_can_mean_is_refused(self, thinking):
        with pytest.raises(ValueError, match=r"LLMConfig\.thinking must be None"):
            LLMConfig(model="m", thinking=thinking)

    @pytest.mark.parametrize(
        ("thinking", "expected"),
        [
            pytest.param(None, None, id="none"),
            pytest.param("none", ThinkingConfig(level=ThinkingLevel.NONE), id="level-none"),
            pytest.param("low", ThinkingConfig(level=ThinkingLevel.LOW), id="level-string"),
            pytest.param(ThinkingLevel.HIGH, ThinkingConfig(level=ThinkingLevel.HIGH), id="enum"),
            pytest.param(32000, ThinkingConfig(budget_tokens=32000), id="budget"),
            pytest.param(1, ThinkingConfig(budget_tokens=1), id="budget-one"),
            # Gemini 2.5 reads a zero budget as thinking off and -1 as dynamic thinking.
            pytest.param(0, ThinkingConfig(budget_tokens=0), id="budget-zero"),
            pytest.param(-1, ThinkingConfig(budget_tokens=-1), id="budget-dynamic"),
            pytest.param(
                ThinkingConfig(level="low", budget_tokens=500),
                ThinkingConfig(level=ThinkingLevel.LOW, budget_tokens=500),
                id="config",
            ),
        ],
    )
    def test_every_valid_form_is_kept_unchanged(self, thinking, expected):
        config = LLMConfig(model="m", thinking=thinking)
        assert config.thinking is thinking
        assert config.get_thinking_config() == expected

    def test_a_numpy_integer_budget_is_stored_as_int(self):
        thinking = cast(int, np.int64(8000))  # accepted, though the annotation names only int
        config = LLMConfig(model="m", thinking=thinking)
        assert config.thinking == 8000
        assert type(config.thinking) is int
        budget = config.get_thinking_config()
        assert budget is not None
        assert type(budget.budget_tokens) is int

    def test_the_error_chains_from_the_thinking_config_error(self):
        thinking = cast(ThinkingLevel, "extreme")  # a str the annotation does not allow
        with pytest.raises(ValueError, match="'extreme'") as excinfo:
            LLMConfig(model="m", thinking=thinking)
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_the_forms_name_every_level(self):
        with pytest.raises(ValueError) as excinfo:
            LLMConfig(model="m", thinking=cast(ThinkingLevel, "extreme"))
        for level in ThinkingLevel:
            assert repr(level.value) in str(excinfo.value)

    def test_replace_validates_again(self):
        config = LLMConfig(model="m", thinking="high")
        with pytest.raises(ValueError, match="thinking"):
            replace(config, thinking="extreme")

    def test_from_yaml_refuses_an_unknown_level(self, tmp_path):
        path = tmp_path / "llm.yaml"
        path.write_text("model: m\nthinking: extreme\n", encoding="utf-8")
        with pytest.raises(ValueError, match="thinking"):
            LLMConfig.from_yaml(path)

    @pytest.mark.parametrize(
        "mapping", ["{level: extreme}", "{budget_tokens: true}", "{budget_tokens: -5}"]
    )
    def test_from_yaml_refuses_an_invalid_thinking_mapping(self, tmp_path, mapping):
        path = tmp_path / "llm.yaml"
        path.write_text(f"model: m\nthinking: {mapping}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"ThinkingConfig\.(level|budget_tokens) must be"):
            LLMConfig.from_yaml(path)


class TestThinkingConfigValidation:
    """``ThinkingConfig`` refuses a level or budget no provider can mean."""

    @pytest.mark.parametrize("level", ["extreme", "HIGH", "", None, 2, True])
    def test_an_unknown_level_is_refused(self, level):
        with pytest.raises(ValueError, match=r"ThinkingConfig\.level must be one of"):
            ThinkingConfig(level=level)

    @pytest.mark.parametrize("level", ["extreme", "", 2, True])
    def test_an_unknown_level_is_refused_with_a_budget_too(self, level):
        """A level string always had to name a level. A value of another type alongside a
        budget is refused too: only None says "no level"."""
        with pytest.raises(ValueError, match=r"ThinkingConfig\.level must be one of"):
            ThinkingConfig(level=level, budget_tokens=9000)

    def test_no_level_with_a_budget_is_kept(self):
        """A budget is sent without reading ``level``, so ``level=None`` alongside one (as
        in a YAML ``{level: null, budget_tokens: 9000}``) has always worked and still does."""
        config = ThinkingConfig(level=cast(ThinkingLevel, None), budget_tokens=9000)
        assert config.level is None
        assert config.budget_tokens == 9000
        assert config.get_effective_budget() == 9000
        llm = LLMConfig(model="m", thinking=config)
        assert llm.get_thinking_config() == config

    def test_no_level_with_a_budget_loads_from_yaml(self, tmp_path):
        path = tmp_path / "llm.yaml"
        path.write_text("model: m\nthinking: {level: null, budget_tokens: 9000}\n")
        thinking = LLMConfig.from_yaml(path).get_thinking_config()
        assert thinking == ThinkingConfig(level=cast(ThinkingLevel, None), budget_tokens=9000)

    def test_the_level_message_lists_every_level(self):
        with pytest.raises(ValueError) as excinfo:
            ThinkingConfig(level=cast(ThinkingLevel, "extreme"))
        for level in ThinkingLevel:
            assert repr(level.value) in str(excinfo.value)

    @pytest.mark.parametrize(
        "budget_tokens", [True, False, np.bool_(False), -2, 1024.0, "1024", float("nan")]
    )
    def test_a_budget_that_is_not_an_integer_from_minus_one_is_refused(self, budget_tokens):
        with pytest.raises(ValueError, match=r"ThinkingConfig\.budget_tokens must be None or"):
            ThinkingConfig(budget_tokens=budget_tokens)

    @pytest.mark.parametrize("budget_tokens", [None, -1, 0, 1, 1024, 50000])
    def test_a_valid_budget_is_kept_unchanged(self, budget_tokens):
        assert ThinkingConfig(budget_tokens=budget_tokens).budget_tokens == budget_tokens

    def test_a_numpy_integer_budget_is_stored_as_int(self):
        budget_tokens = cast(int, np.int32(4096))
        config = ThinkingConfig(budget_tokens=budget_tokens)
        assert config.budget_tokens == 4096
        assert type(config.budget_tokens) is int
        # The response-cache key renders the config with str(); a plain int keeps it plain.
        assert str(config) == str(ThinkingConfig(budget_tokens=4096))

    @pytest.mark.parametrize("level", ["low", ThinkingLevel.LOW])
    def test_a_level_string_or_member_is_stored_as_the_member(self, level):
        assert ThinkingConfig(level=level).level is ThinkingLevel.LOW

    def test_replace_validates_again(self):
        with pytest.raises(ValueError, match="budget_tokens"):
            replace(ThinkingConfig(), budget_tokens=-7)


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
        config = LLMConfig(model="m", thinking="high", cache_dir=Path("autorubric_cache"))

        config.to_yaml(path)

        assert path.read_text(encoding="utf-8") == (
            "model: m\n"
            "timeout: 60.0\n"
            "max_retries: 3\n"
            "retry_min_wait: 1.0\n"
            "retry_max_wait: 60.0\n"
            "cache_enabled: false\n"
            "cache_dir: autorubric_cache\n"
            "thinking: high\n"
            "prompt_caching: true\n"
        )
