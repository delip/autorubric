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

    def test_the_retry_budget_must_be_finite(self):
        """The retry budget, ``max_retries * (timeout + backoff_max)`` seconds, is built from
        ``max_retries * timeout``, which must be finite; each factor finite is not enough."""
        with pytest.raises(ValueError, match="retry budget"):
            DecisionModelConfig(model="m", timeout=1e308, max_retries=2)

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

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.typesafe.ai",
            "https://xyz.endpoints.huggingface.cloud/",
            "http://localhost:8080",
            "http://[::1]:9000/prefix",
            "https://b\u00fccher.example/v2",
            "https://dm.example.com/a%20b",
            "https://dm.example.com/@org/v2",
        ],
    )
    def test_http_base_urls_accepted(self, url):
        assert DecisionModelConfig(model="m", api_base=url).api_base == url

    @pytest.mark.parametrize(
        "url", ["", "api.typesafe.ai", "ftp://api.typesafe.ai", "https://", "https://h:notaport"]
    )
    def test_malformed_base_url_rejected(self, url):
        with pytest.raises(ValueError, match="api_base"):
            DecisionModelConfig(model="m", api_base=url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://dm.example.com\n",
            "https://dm.example.com\r\n",
            "https://dm.example.com\t",
            "\thttps://dm.example.com",
            " https://dm.example.com",
            "https://dm.exam\r\nple.com",
            "https://dm.example.com/a b",
            "https://dm.example.com\x00",
            "https://dm.example.com\x7f",
            "https://dm.example.com\u00a0",
            "https://dm.example.com\u2028",
        ],
        ids=[
            "trailing-lf",
            "trailing-crlf",
            "trailing-tab",
            "leading-tab",
            "leading-space",
            "embedded-crlf",
            "embedded-space",
            "nul",
            "del",
            "no-break-space",
            "line-separator",
        ],
    )
    def test_base_url_with_whitespace_or_control_characters_rejected(self, url):
        """A URL contains no whitespace or control characters. ``urlsplit`` silently drops
        tabs and line breaks (and leading spaces and control characters) before it parses,
        so a URL read from a file with its trailing newline would pass a parse-only check
        and then fail every request in the HTTP layer, as an error no category covers (the
        conservative worst case on every criterion of every item)."""
        with pytest.raises(ValueError, match="api_base"):
            DecisionModelConfig(model="m", api_base=url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://dm.example.com/api?tenant=a",
            "https://dm.example.com/api?",
            "https://dm.example.com?tenant=a",
            "https://dm.example.com/api#x",
            "https://dm.example.com/api#",
            "https://dm.example.com/api?tenant=a#x",
        ],
        ids=["query", "empty-query", "query-without-path", "fragment", "empty-fragment", "both"],
    )
    def test_base_url_with_a_query_or_fragment_rejected(self, url):
        """Requests go to ``{api_base}/v1/systemone``, a path appended to the URL string.
        After a query, even an empty one, the path lands in the query; after a fragment it is
        dropped with the fragment. Either way no request reaches the endpoint's path."""
        with pytest.raises(ValueError, match=r"api_base .*query or fragment"):
            DecisionModelConfig(model="m", api_base=url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://svc:SECRET@dm.example.com/api",
            "https://SECRET@dm.example.com",
            "https://:SECRET@dm.example.com:8443/v2",
            "https://SECRET:@dm.example.com",
            "https://@dm.example.com",
            "http://svc:SECRET@[::1]:9000",
        ],
        ids=["user-password", "user", "password", "user-empty-password", "empty", "ipv6"],
    )
    def test_base_url_with_credentials_rejected_without_repeating_them(self, url):
        """The HTTP layer sends URL credentials as Basic authentication in place of the
        bearer API key, so the configured key would never be sent. The message names the
        field but never repeats the URL."""
        with pytest.raises(ValueError, match=r"api_base .*credentials") as excinfo:
            DecisionModelConfig(model="m", api_base=url)
        assert "SECRET" not in str(excinfo.value)
        assert "api_key" in str(excinfo.value)

    @pytest.mark.parametrize(
        "url",
        [
            "https://svc:SECRET@dm.example.com\n",
            "https://svc:SECRET@dm.example.com:notaport",
            "https://svc:SECRET@[::1",
            "ftp://svc:SECRET@dm.example.com",
            "https://svc:SECRET@",
            "https://dm.example.com/@SECRET?x",
        ],
        ids=["line-break", "bad-port", "bad-ipv6", "scheme", "no-host", "at-in-path"],
    )
    def test_a_rejected_base_url_with_an_at_sign_is_never_repeated(self, url):
        """Whatever else is wrong with it, a URL holding an ``@`` may hold credentials, so
        no error message repeats it."""
        with pytest.raises(ValueError, match="api_base") as excinfo:
            DecisionModelConfig(model="m", api_base=url)
        assert "SECRET" not in str(excinfo.value)

    @pytest.mark.parametrize(
        "headers",
        [
            {"X-Org": "evals"},
            {},
            {"X-A": "1", "X-B": ""},
            {"X-Org": "a b\tc"},
            {"X-Json": '{"a": [1, 2]}'},
            {"!#$%&'*+-.^_`|~0-9aZ": "~!@#$%^&*()_+`-={}[]|\\:;\"'<>,.?/"},
        ],
    )
    def test_string_extra_headers_accepted(self, headers):
        assert DecisionModelConfig(model="m", extra_headers=headers).extra_headers == headers

    @pytest.mark.parametrize(
        "headers", [{"X-Org": 7}, {7: "evals"}, {"X-Org": b"evals"}, {"X-Org": None}, ["X"], None]
    )
    def test_extra_headers_must_map_strings_to_strings(self, headers):
        """A non-string header fails every request inside the SDK, as an ``unknown`` error
        (the conservative worst case on every criterion), so it is rejected up front."""
        with pytest.raises(ValueError, match="extra_headers"):
            DecisionModelConfig(model="m", extra_headers=headers)

    def test_extra_headers_error_never_shows_a_header_value(self):
        with pytest.raises(ValueError) as excinfo:
            DecisionModelConfig(model="m", extra_headers={"X-Secret": "s3cr3t", "X-Org": 7})
        assert "s3cr3t" not in str(excinfo.value)
        assert "7" not in str(excinfo.value)

    @pytest.mark.parametrize(
        "value",
        [
            "SECRET\n",
            "SECRET\r\nX-Injected: 1",
            "SECRET\r",
            "SEC\x00RET",
            "SEC\x01RET",
            "SEC\x7fRET",
            " SECRET",
            "SECRET ",
            "SECRET\t",
            "SECR\u00e9T",
            "SECRET\u00a0",
        ],
        ids=[
            "trailing-lf",
            "embedded-crlf",
            "trailing-cr",
            "nul",
            "control",
            "del",
            "leading-space",
            "trailing-space",
            "trailing-tab",
            "non-ascii",
            "no-break-space",
        ],
    )
    def test_extra_header_values_must_be_http_field_values(self, value):
        """A header value is visible ASCII with spaces or tabs only between characters (RFC
        9110). The HTTP layer refuses anything else on every request, typically a secret
        read from a file with its trailing newline, after every retry, and its error, which
        repeats the value, would be recorded in every failed report. The message names the
        header, never its value."""
        with pytest.raises(ValueError, match="extra_headers") as excinfo:
            DecisionModelConfig(model="m", extra_headers={"X-Goog-Api-Key": value})
        assert "X-Goog-Api-Key" in str(excinfo.value)
        assert "SEC" not in str(excinfo.value)

    @pytest.mark.parametrize(
        "name", ["X Org", "", "X-Org:", "X\nOrg", "X(Org)", "X/Org", "X-\u00c4rg", "X-Org\t"]
    )
    def test_extra_header_names_must_be_http_tokens(self, name):
        """A header name is a token (RFC 9110): letters, digits and ``!#$%&'*+-.^_`|~``."""
        with pytest.raises(ValueError, match="extra_headers") as excinfo:
            DecisionModelConfig(model="m", extra_headers={name: "SECRET"})
        assert "SECRET" not in str(excinfo.value)


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
    """The client parses base URLs with ``httpx2``, the SDK's HTTP stack, so the extra
    declares it rather than relying on the SDK's own requirement."""
    assert set(_typesafe_extra()) == {"typesafe-sdk", "httpx2"}


def _typesafe_extra() -> dict[str, Requirement]:
    pyproject = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = [
        Requirement(spec) for spec in pyproject["project"]["optional-dependencies"]["typesafe"]
    ]
    return {requirement.name: requirement for requirement in requirements}
