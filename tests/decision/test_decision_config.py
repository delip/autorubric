"""Tests for ``DecisionModelConfig``: defaults, validation, export, and the optional SDK."""

import dataclasses
import math
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import autorubric
from autorubric import DecisionModelConfig, LLMConfig


class TestDefaults:
    def test_defaults(self):
        config = DecisionModelConfig(model="jev-latest")
        assert config.model == "jev-latest"
        assert config.api_key is None
        assert config.api_base is None
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.max_parallel_requests is None
        assert config.cache_enabled is False
        assert config.cache_dir == ".autorubric_cache"
        assert config.cache_ttl is None
        assert config.extra_headers == {}
        assert config.binary_framing == "noul_framed"
        assert config.ordinal_framing == "choice"
        assert config.decision_threshold == 0.5
        assert config.input_cost_per_token is None

    def test_field_set_is_exactly_the_documented_one(self):
        assert [f.name for f in dataclasses.fields(DecisionModelConfig)] == [
            "model",
            "api_key",
            "api_base",
            "timeout",
            "max_retries",
            "max_parallel_requests",
            "cache_enabled",
            "cache_dir",
            "cache_ttl",
            "extra_headers",
            "binary_framing",
            "ordinal_framing",
            "decision_threshold",
            "input_cost_per_token",
        ]

    def test_shared_concepts_mirror_llm_config_names_and_defaults(self):
        """Where LLMConfig has the same concept, the name and the default are the same."""
        shared = [
            "api_key",
            "api_base",
            "timeout",
            "max_retries",
            "max_parallel_requests",
            "cache_enabled",
            "cache_dir",
            "cache_ttl",
            "extra_headers",
        ]
        dm = DecisionModelConfig(model="jev-latest")
        llm = LLMConfig(model="openai/gpt-5.2")
        for name in shared:
            assert getattr(dm, name) == getattr(llm, name), name

    def test_extra_headers_default_is_not_shared(self):
        a = DecisionModelConfig(model="m")
        b = DecisionModelConfig(model="m")
        a.extra_headers["X-Trace"] = "1"
        assert b.extra_headers == {}

    def test_model_is_positional_everything_else_keyword_only(self):
        assert DecisionModelConfig("jev-latest").model == "jev-latest"
        with pytest.raises(TypeError):
            DecisionModelConfig("jev-latest", "sk-key")

    def test_api_key_is_kept_out_of_repr(self):
        config = DecisionModelConfig(model="jev-latest", api_key="sk-secret-value")
        assert "sk-secret-value" not in repr(config)
        assert "jev-latest" in repr(config)


class TestValidation:
    @pytest.mark.parametrize("model", ["", None])
    def test_model_is_required(self, model):
        with pytest.raises(ValueError, match="model"):
            DecisionModelConfig(model=model)

    @pytest.mark.parametrize("framing", ["noul", "noul_framed", "choice"])
    def test_binary_framings_accepted(self, framing):
        assert DecisionModelConfig(model="m", binary_framing=framing).binary_framing == framing

    @pytest.mark.parametrize("framing", ["score", "NOUL", "", "noul-framed"])
    def test_unknown_binary_framing_rejected(self, framing):
        with pytest.raises(ValueError, match="binary_framing"):
            DecisionModelConfig(model="m", binary_framing=framing)

    @pytest.mark.parametrize("framing", ["choice", "score"])
    def test_ordinal_framings_accepted(self, framing):
        assert DecisionModelConfig(model="m", ordinal_framing=framing).ordinal_framing == framing

    @pytest.mark.parametrize("framing", ["noul", "noul_framed", "Score", ""])
    def test_unknown_ordinal_framing_rejected(self, framing):
        with pytest.raises(ValueError, match="ordinal_framing"):
            DecisionModelConfig(model="m", ordinal_framing=framing)

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 1.0, 1])
    def test_threshold_in_unit_interval_accepted(self, threshold):
        config = DecisionModelConfig(model="m", decision_threshold=threshold)
        assert config.decision_threshold == threshold

    @pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan, math.inf, True])
    def test_threshold_outside_unit_interval_rejected(self, threshold):
        with pytest.raises(ValueError, match="decision_threshold"):
            DecisionModelConfig(model="m", decision_threshold=threshold)

    @pytest.mark.parametrize("timeout", [0, -1.0, math.nan, math.inf])
    def test_timeout_must_be_positive_and_finite(self, timeout):
        with pytest.raises(ValueError, match="timeout"):
            DecisionModelConfig(model="m", timeout=timeout)

    @pytest.mark.parametrize("max_retries", [0, -1, 2.0, True])
    def test_max_retries_must_be_a_positive_int(self, max_retries):
        with pytest.raises(ValueError, match="max_retries"):
            DecisionModelConfig(model="m", max_retries=max_retries)

    def test_single_attempt_allowed(self):
        assert DecisionModelConfig(model="m", max_retries=1).max_retries == 1

    @pytest.mark.parametrize("limit", [0, -2, 1.5, True])
    def test_max_parallel_requests_must_be_a_positive_int(self, limit):
        with pytest.raises(ValueError, match="max_parallel_requests"):
            DecisionModelConfig(model="m", max_parallel_requests=limit)

    @pytest.mark.parametrize("ttl", [0, -5])
    def test_cache_ttl_must_be_positive(self, ttl):
        with pytest.raises(ValueError, match="cache_ttl"):
            DecisionModelConfig(model="m", cache_ttl=ttl)

    @pytest.mark.parametrize("price", [0.0, 0.042e-6, 1])
    def test_non_negative_price_accepted(self, price):
        assert DecisionModelConfig(model="m", input_cost_per_token=price).input_cost_per_token == (
            price
        )

    @pytest.mark.parametrize("price", [-1e-9, math.nan, math.inf])
    def test_negative_or_non_finite_price_rejected(self, price):
        with pytest.raises(ValueError, match="input_cost_per_token"):
            DecisionModelConfig(model="m", input_cost_per_token=price)


class TestExport:
    def test_exported_from_the_package(self):
        assert autorubric.DecisionModelConfig is DecisionModelConfig
        assert "DecisionModelConfig" in autorubric.__all__

    def test_client_is_internal(self):
        assert not hasattr(autorubric, "DecisionModelClient")
        assert "DecisionModelClient" not in autorubric.__all__


def test_core_install_never_imports_the_sdk():
    """``import autorubric``, building a config (its ``api_base`` included) and classifying
    errors need no SDK, nor the SDK's HTTP stack.

    Runs in a fresh interpreter, because this test session has already imported the SDK.
    The SDK is then blocked (as if not installed): building a client must fail with an
    actionable ``ImportError``, and everything else keeps working.
    """
    script = textwrap.dedent(
        """
        import sys

        import autorubric
        from autorubric import DecisionModelConfig, classify_grading_error

        assert "typesafe_sdk" not in sys.modules, "import autorubric imported the SDK"
        config = DecisionModelConfig(model="jev-latest", api_key="k")
        DecisionModelConfig(model="jev-latest", api_base="https://dm.example.com/v2")
        assert classify_grading_error(ValueError("x")) == "parse"
        assert classify_grading_error(RuntimeError("x")) == "unknown"
        assert "typesafe_sdk" not in sys.modules, "config or classification imported the SDK"
        assert "httpx2" not in sys.modules, "checking api_base imported the SDK's HTTP stack"

        sys.modules["typesafe_sdk"] = None  # behave as if the SDK were not installed
        from autorubric.decision import DecisionModelClient

        try:
            DecisionModelClient(config)
        except ImportError as exc:
            assert "pip install 'autorubric[typesafe]'" in str(exc), str(exc)
        else:
            raise AssertionError("expected ImportError")
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_the_typesafe_extra_requires_an_sdk_that_keeps_credentials_out_of_errors():
    """``typesafe-sdk`` 0.7.1 is the first release that validates the API key when a client is
    built and redacts credentials from transport errors. Under 0.7.0 a key with a line
    break reached the HTTP layer, whose error (``Illegal header value b'Bearer ...'``) was
    recorded in every failed report's ``error`` and ``reason``, and so in checkpoints."""
    requirement = _typesafe_extra()["typesafe-sdk"]
    assert not requirement.specifier.contains("0.7.0")
    assert requirement.specifier.contains("0.7.1")


def test_the_typesafe_extra_declares_the_http_stack_the_client_imports():
    """The client checks the base URL with ``httpx2`` and each extra header with ``h11``,
    the SDK's HTTP stack and the HTTP/1.1 layer beneath it, so the extra declares both
    rather than relying on the SDK's own requirements."""
    assert set(_typesafe_extra()) == {"typesafe-sdk", "httpx2", "h11"}


def _typesafe_extra() -> dict[str, Requirement]:
    pyproject = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = [
        Requirement(spec) for spec in pyproject["project"]["optional-dependencies"]["typesafe"]
    ]
    return {requirement.name: requirement for requirement in requirements}
