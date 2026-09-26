"""Run the live-regression parity harness for LLM judges.

One invocation grades every parity config against the library selected by ``--code`` and
writes a self-contained run directory under ``experiments/parity/<run>/``:

- ``fresh``: real provider calls into the run's own empty response cache. Paid; refused
  without ``--confirm-paid`` and a passing pre-flight.
- ``replay``: a copy of a prior fresh run's cache, with ``litellm.acompletion`` replaced by a
  sentinel that records and refuses every call. Free.
- ``synthetic``: ``litellm.acompletion`` replaced by a deterministic fake provider, the
  response cache off. Free; runnable before any paid baseline exists.

Other entry points: ``--init-manifest`` (baseline code, once) pins the datasets, the
charm-100 split and litellm's model cost map; ``--preflight`` checks what litellm sends to
each provider (``--offline`` does so against a closed loopback port, for free).

Run with ``uv run --frozen python scripts/parity/run_parity.py ...``; baseline runs prepend
``PYTHONPATH=<baseline worktree>/src``. See ``scripts/parity/README.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import ipaddress
import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
import openai
import parity_config as pc

if TYPE_CHECKING:
    from autorubric import EvalResult, MetricsResult, RubricDataset
    from autorubric.graders import CriterionGrader

Code = Literal["baseline", "after"]
Mode = Literal["fresh", "replay", "synthetic"]
Network = Literal["live", "blocked", "synthetic"]

LOGGER = logging.getLogger("parity")

SECRET_PARAMS = frozenset({"api_key", "extra_headers"})
"""Request parameters never written to disk."""

SAMPLING_PARAMS = ("temperature", "top_p", "max_tokens", "reasoning_effort", "thinking", "seed")
"""Request parameters that shape the model's output, used to key the synthetic provider."""


# =======================================================================================
# Process setup and library identity
# =======================================================================================


def prepare_litellm_environment() -> None:
    """Keep litellm off the network at import time.

    By default litellm downloads its model cost map on every import, so model routing and
    ``completion_cost`` would depend on the day a run happens. The harness loads litellm's
    bundled map instead and then installs the pinned snapshot (``pin_cost_map``).
    """
    if "litellm" in sys.modules:
        raise SystemExit("litellm was imported before the parity environment was prepared")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def configure_logging(verbose: bool) -> None:
    """Console logging; library warnings about individual judge errors are noisy."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    quiet = logging.INFO if verbose else logging.ERROR
    for name in ("autorubric", "LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx"):
        logging.getLogger(name).setLevel(quiet)


@dataclass(frozen=True)
class LibraryIdentity:
    """Which ``autorubric`` the process imported, and the state of its tree."""

    package_dir: Path
    tree: Path
    git_commit: str | None
    git_dirty: bool | None
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """JSON form for ``run.json``."""
        return {
            "package_dir": str(self.package_dir),
            "tree": str(self.tree),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "source_sha256": self.source_sha256,
        }


def _git(tree: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(tree), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _source_fingerprint(package_dir: Path) -> str:
    """SHA-256 over every source file of the package (relative path and bytes)."""
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(package_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_library(expected_tree: Path) -> LibraryIdentity:
    """Import ``autorubric`` and assert it resolves inside ``expected_tree/src``.

    Raises:
        SystemExit: If the imported package lives anywhere else.
    """
    import autorubric

    if autorubric.__file__ is None:
        raise SystemExit("autorubric has no __file__; cannot tell which library is loaded")
    package_dir = Path(autorubric.__file__).resolve().parent
    expected = (expected_tree / "src" / "autorubric").resolve()
    if package_dir != expected:
        raise SystemExit(
            f"autorubric resolved to {package_dir}, expected {expected}. "
            "Baseline runs need PYTHONPATH=<baseline tree>/src; after runs must not set it."
        )
    commit = _git(expected_tree, "rev-parse", "HEAD")
    status = _git(
        expected_tree, "status", "--porcelain", "--untracked-files=all", "--", "src/autorubric"
    )
    return LibraryIdentity(
        package_dir=package_dir,
        tree=expected_tree.resolve(),
        git_commit=commit.strip() if commit else None,
        git_dirty=None if status is None else bool(status.strip()),
        source_sha256=_source_fingerprint(package_dir),
    )


def package_versions() -> dict[str, dict[str, str | None]]:
    """Installed versions: the pinned four, plus others that shape metrics or requests."""

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    others = ("httpx", "tenacity", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels")
    return {
        "pinned": {name: version(name) for name in pc.PINNED_PACKAGES},
        "other": {name: version(name) for name in others},
    }


def pin_cost_map(manifest: dict[str, Any]) -> str:
    """Install the manifest's model cost map snapshot into litellm.

    Returns:
        The snapshot's SHA-256.

    Raises:
        SystemExit: If the snapshot is missing, altered, or does not route a judge model.
    """
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import _finalize_model_cost_map

    expected = manifest["model_cost_map"]["sha256"]
    if not pc.COST_MAP_PATH.exists() or pc.sha256_file(pc.COST_MAP_PATH) != expected:
        raise SystemExit(f"{pc.COST_MAP_PATH} is missing or does not match the manifest")
    finalized = _finalize_model_cost_map(json.loads(pc.COST_MAP_PATH.read_bytes()))
    litellm.model_cost.clear()
    litellm.model_cost.update(finalized)
    litellm.add_known_models(finalized)
    litellm.suppress_debug_info = True
    for judge in pc.JUDGES.values():
        _, provider, _, _ = litellm.get_llm_provider(judge.model)
        if provider != judge.provider:
            raise SystemExit(f"{judge.model} routes to {provider!r}, expected {judge.provider!r}")
    return expected


def load_manifest() -> dict[str, Any]:
    """Read the parity manifest, verifying the dataset snapshots it pins."""
    if not pc.MANIFEST_PATH.exists():
        raise SystemExit(
            f"{pc.MANIFEST_PATH} does not exist; create it once with "
            "--code baseline --init-manifest --baseline-tree <worktree>"
        )
    manifest = pc.read_json(pc.MANIFEST_PATH)
    for dataset_id, source in pc.DATASETS.items():
        recorded = manifest["datasets"][dataset_id]["sha256"]
        if pc.sha256_file(source.snapshot_path) != recorded:
            raise SystemExit(f"{source.snapshot_path} does not match the manifest")
    return manifest


def expected_tree(code: Code, manifest: dict[str, Any]) -> Path:
    """Library tree that ``--code`` selects."""
    if code == "baseline":
        return Path(manifest["baseline_tree"])
    return pc.AFTER_LIBRARY_TREE


def judges_fingerprint() -> str:
    """Identity of the judge definitions a pre-flight vouches for."""
    judges = {
        name: {
            "model": judge.model,
            "provider": judge.provider,
            "extra_params": [list(pair) for pair in judge.extra_params],
            "expected_request": [list(pair) for pair in judge.expected_request],
        }
        for name, judge in pc.JUDGES.items()
    }
    payload = {"temperature": pc.TEMPERATURE, "thinking": pc.THINKING, "judges": judges}
    return pc.sha256_text(pc.canonical_json(payload))


# =======================================================================================
# Manifest creation
# =======================================================================================


def init_manifest(args: argparse.Namespace) -> int:
    """Pin datasets, the charm-100 split (computed on the baseline code) and the cost map."""
    if args.code != "baseline":
        raise SystemExit("--init-manifest must run on the baseline code (--code baseline)")
    if args.baseline_tree is None:
        raise SystemExit("--init-manifest needs --baseline-tree <baseline worktree>")
    if pc.MANIFEST_PATH.exists():
        raise SystemExit(f"{pc.MANIFEST_PATH} already exists; delete it to recreate")
    baseline_tree = Path(args.baseline_tree).resolve()
    library = load_library(baseline_tree)

    import litellm

    from autorubric import RubricDataset

    pc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}
    for dataset_id, source in pc.DATASETS.items():
        shutil.copyfile(baseline_tree / source.relative_path, source.snapshot_path)
        datasets[dataset_id] = {
            "source": source.relative_path,
            "snapshot": source.snapshot_path.relative_to(pc.PARITY_DIR).as_posix(),
            "sha256": pc.sha256_file(source.snapshot_path),
        }

    charm = RubricDataset.from_file(pc.DATASETS["charm"].snapshot_path)
    train, test = charm.split_train_test(**pc.CHARM_SPLIT)
    position = {id(item): index for index, item in enumerate(charm.items)}
    split = {
        **pc.CHARM_SPLIT,
        "train_indices": [position[id(item)] for item in train.items],
        "test_indices": [position[id(item)] for item in test.items],
    }

    response = httpx.get(litellm.model_cost_map_url, timeout=60.0)
    response.raise_for_status()
    pc.COST_MAP_PATH.write_bytes(response.content)
    missing = [j.model for j in pc.JUDGES.values() if j.model not in response.json()]
    if missing:
        raise SystemExit(f"the fetched cost map has no entry for {missing}")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_tree": str(baseline_tree),
        "baseline_library": library.to_dict(),
        "datasets": datasets,
        "charm_split": split,
        "model_cost_map": {
            "url": litellm.model_cost_map_url,
            "path": pc.COST_MAP_PATH.relative_to(pc.PARITY_DIR).as_posix(),
            "sha256": pc.sha256_file(pc.COST_MAP_PATH),
        },
        "packages": package_versions(),
    }
    pc.write_json(pc.MANIFEST_PATH, manifest)
    LOGGER.info("wrote %s (charm test items: %s)", pc.MANIFEST_PATH, split["test_indices"])
    return 0


# =======================================================================================
# Request recording, attribution and network control
# =======================================================================================


def to_jsonable(value: Any) -> Any:
    """Convert a litellm request parameter into plain JSON data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, type) and hasattr(value, "model_json_schema"):
        return {"pydantic_model": value.__name__, "schema": value.model_json_schema()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {"repr": repr(value)}


def canonical_request(params: dict[str, Any]) -> dict[str, Any]:
    """Every parameter sent to ``litellm.acompletion`` except credentials and headers."""
    return {key: to_jsonable(value) for key, value in params.items() if key not in SECRET_PARAMS}


def user_prompt_of(params: dict[str, Any]) -> str | None:
    """The user message of a chat request."""
    for message in params.get("messages") or []:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _block(text: str, tag: str, *, last: bool = False) -> str | None:
    opener, closer = f"<{tag}>\n", f"\n</{tag}>"
    start = text.rfind(opener) if last else text.find(opener)
    if start < 0:
        return None
    start += len(opener)
    end = text.rfind(closer) if last else text.find(closer, start)
    return text[start:end] if end >= start else None


@dataclass
class PassContext:
    """Attribution data and network policy of the pass currently grading."""

    label: str
    config_id: str
    dataset_id: str
    construction: str
    network: Network
    submission_index: dict[str, int]
    requirement_index: dict[str, int]
    judge_by_model: dict[str, str]
    blocked_categories: dict[tuple[int, int, str], str] = field(default_factory=dict)
    attempts: Counter[str] = field(default_factory=Counter)
    in_flight: Counter[str] = field(default_factory=Counter)
    concurrent_duplicates: int = 0

    def attribute(self, params: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
        """``(item, criterion, judge)`` of a request, read from its prompt and model."""
        prompt = user_prompt_of(params) or ""
        requirement = _block(prompt, "criterion")
        if requirement is None:
            requirement = _block(prompt, "question")
        submission = _block(prompt, "submission", last=True)
        return (
            self.submission_index.get(submission) if submission is not None else None,
            self.requirement_index.get(requirement) if requirement is not None else None,
            self.judge_by_model.get(str(params.get("model"))),
        )


class NetworkBlockedError(openai.APIError):
    """Raised in place of a provider call while the network is blocked.

    An ``openai.APIError``, so ``classify_grading_error`` routes it to ``infrastructure``
    (abstain), and not one of the litellm types ``LLMClient`` retries, so no backoff sleep.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message, httpx.Request("POST", "https://network-blocked.invalid"), body=None
        )


class BlockedParseError(ValueError):
    """Blocked call standing in for a call the source run saw fail as ``parse``."""


class BlockedUnknownError(RuntimeError):
    """Blocked call standing in for a call the source run saw fail as ``unknown``."""


def blocked_error(model: str, category: str | None) -> Exception:
    """The exception a blocked call raises.

    A call is attempted while blocked only when its response is not in the cache, i.e. the
    source run's call failed. Raising an exception of the same ``classify_grading_error``
    category makes the grader route the failure exactly as it did in the source run (an
    abstention for ``infrastructure``/``parse``, the worst-case verdict for ``unknown``), so
    verdicts, scores and metrics stay comparable; only the error text differs. Calls the
    source run never made default to ``infrastructure``.
    """
    message = f"parity harness blocked a network call to {model}"
    if category == "parse":
        return BlockedParseError(f"{message} (source call failed: parse)")
    if category == "unknown":
        return BlockedUnknownError(f"{message} (source call failed: unknown)")
    if category == "infrastructure":
        return NetworkBlockedError(f"{message} (source call failed: infrastructure)")
    return NetworkBlockedError(message)


class RequestRecorder:
    """Stands in for ``litellm.acompletion``: records every call, then applies the policy."""

    def __init__(self, original: Callable[..., Any], synthetic: SyntheticProvider | None):
        self.original = original
        self.synthetic = synthetic
        self.records: list[dict[str, Any]] = []
        self.active: PassContext | None = None

    async def __call__(self, **params: Any) -> Any:
        context = self.active
        if context is None:
            raise RuntimeError("an LLM call happened outside a parity pass")
        request = canonical_request(params)
        request_sha = pc.sha256_text(pc.canonical_json(request))
        context.attempts[request_sha] += 1
        context.in_flight[request_sha] += 1
        if context.in_flight[request_sha] > 1:
            context.concurrent_duplicates += 1
        item_idx, criterion_idx, judge_id = context.attribute(params)
        record: dict[str, Any] = {
            "pass": context.label,
            "config": context.config_id,
            "dataset": context.dataset_id,
            "construction": context.construction,
            "item_idx": item_idx,
            "criterion_idx": criterion_idx,
            "judge_id": judge_id,
            "attempt": context.attempts[request_sha],
            "request_sha256": request_sha,
            "request": request,
        }
        try:
            response = await self._respond(context, params, record)
        finally:
            context.in_flight[request_sha] -= 1
            self.records.append(record)
        return response

    async def _respond(
        self, context: PassContext, params: dict[str, Any], record: dict[str, Any]
    ) -> Any:
        model = str(params.get("model"))
        if context.network == "blocked":
            triple = (record["item_idx"], record["criterion_idx"], record["judge_id"])
            category = context.blocked_categories.get(triple)
            record["outcome"] = f"blocked:{category or 'no-source-failure'}"
            raise blocked_error(model, category)
        if context.network == "synthetic":
            assert self.synthetic is not None
            response, error, outcome = self.synthetic.respond(params, record["attempt"])
            record["outcome"] = outcome
            if error is not None:
                record["error"] = str(error)
                raise error
            record["provider_model"] = response.model
            return response
        try:
            response = await self.original(**params)
        except Exception as exc:
            record["outcome"] = f"exception:{type(exc).__name__}"
            record["error"] = str(exc)[:1000]
            raise
        record["outcome"] = "ok"
        record["provider_model"] = getattr(response, "model", None)
        return response


@contextmanager
def patched_acompletion(recorder: RequestRecorder) -> Iterator[None]:
    """Route every ``litellm.acompletion`` call through the recorder."""
    import litellm

    original = litellm.acompletion
    litellm.acompletion = recorder
    try:
        yield
    finally:
        litellm.acompletion = original


class SocketGuard:
    """Tripwire for any network access other than ``litellm.acompletion``.

    While installed, name resolution of and connections to non-loopback hosts are refused
    and recorded. Replay and synthetic runs must end with no recorded attempt.
    """

    def __init__(self) -> None:
        self.attempts: list[dict[str, str]] = []
        self._saved: list[tuple[object, str, Any]] = []

    def _allowed(self, host: Any) -> bool:
        if host is None or host in ("localhost", b"localhost"):
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        try:
            return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
        except ValueError:
            return False

    def _check(self, kind: str, host: Any, target: Any) -> None:
        if not self._allowed(host):
            self.attempts.append({"kind": kind, "target": repr(target)})
            raise OSError(f"parity harness blocked network access: {kind} {target!r}")

    def install(self) -> None:
        """Patch ``socket`` name resolution and connects."""
        getaddrinfo = socket.getaddrinfo
        create_connection = socket.create_connection
        connect = socket.socket.connect
        connect_ex = socket.socket.connect_ex

        def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            self._check("getaddrinfo", host, host)
            return getaddrinfo(host, *args, **kwargs)

        def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
            self._check("create_connection", address[0], address)
            return create_connection(address, *args, **kwargs)

        def guarded_connect(sock: socket.socket, address: Any) -> Any:
            if isinstance(address, tuple):
                self._check("connect", address[0], address)
            return connect(sock, address)

        def guarded_connect_ex(sock: socket.socket, address: Any) -> Any:
            if isinstance(address, tuple):
                self._check("connect_ex", address[0], address)
            return connect_ex(sock, address)

        patches: list[tuple[object, str, Any]] = [
            (socket, "getaddrinfo", guarded_getaddrinfo),
            (socket, "create_connection", guarded_create_connection),
            (socket.socket, "connect", guarded_connect),
            (socket.socket, "connect_ex", guarded_connect_ex),
        ]
        for owner, name, replacement in patches:
            self._saved.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

    def uninstall(self) -> None:
        """Restore the original ``socket`` functions."""
        while self._saved:
            owner, name, original = self._saved.pop()
            setattr(owner, name, original)


# =======================================================================================
# Synthetic provider
# =======================================================================================


@dataclass(frozen=True)
class SyntheticShares:
    """Shares of synthetic outcomes, drawn per request from its hash.

    Attributes:
        infrastructure: Terminal provider exceptions (``InternalServerError`` / ``APIError``),
            which ``classify_grading_error`` routes to ``infrastructure``. Neither is a type
            ``LLMClient`` retries, so they cost no backoff sleep.
        transient: A ``RateLimitError`` on the first attempt only; ``LLMClient`` retries once
            after its one-second minimum backoff and then gets a valid answer.
        malformed: Unparseable or invalid output: prose, truncated JSON, an invalid verdict or
            option type, and (multi-choice) an out-of-range option number.
        null_content: ``content=None``, which the grader classifies as ``unknown`` (the
            conservative worst-case path).
    """

    infrastructure: float = 0.02
    transient: float = 0.02
    malformed: float = 0.03
    null_content: float = 0.01


SYNTHETIC_REPORTED_MODELS = {
    "gpt-6-luna": "gpt-6-luna",
    "gemini/gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
}
"""Model strings the synthetic provider reports, as the real providers do."""

SYNTHETIC_CREATED = 1767225600
"""Fixed ``created`` timestamp of synthetic responses."""


class SyntheticProvider:
    """Deterministic stand-in for a provider.

    Each response is a pure function of a SHA-256 of the canonical request (model,
    messages, response-format schema, sampling parameters) and, for the transient share,
    the attempt number. Identical requests therefore get identical responses in any run.
    """

    version = 1

    def __init__(self, shares: SyntheticShares | None = None) -> None:
        self.shares = shares or SyntheticShares()
        self.provider_of = {judge.model: judge.provider for judge in pc.JUDGES.values()}

    def describe(self) -> dict[str, Any]:
        """Parameters recorded in ``run.json``."""
        return {
            "version": self.version,
            "shares": self.shares.__dict__,
            "reported_models": SYNTHETIC_REPORTED_MODELS,
        }

    @staticmethod
    def key(params: dict[str, Any]) -> str:
        """SHA-256 of the output-shaping part of a request."""
        payload = {
            "model": params.get("model"),
            "messages": to_jsonable(params.get("messages")),
            "response_format": to_jsonable(params.get("response_format")),
            "sampling": {k: to_jsonable(params[k]) for k in SAMPLING_PARAMS if k in params},
        }
        return pc.sha256_text(pc.canonical_json(payload))

    def respond(
        self, params: dict[str, Any], attempt: int
    ) -> tuple[Any, BaseException | None, str]:
        """Return ``(response, exception, outcome label)``; exactly one of the first two."""
        import litellm

        digest = self.key(params)
        rng = random.Random(int(digest, 16))
        u_outcome, u_kind, u_choice, u_reason = (rng.random() for _ in range(4))
        model = str(params.get("model"))
        provider = self.provider_of.get(model, "openai")
        tag = digest[:10]
        shares = self.shares

        edge = shares.infrastructure
        if u_outcome < edge:
            if u_kind < 0.5:
                error: BaseException = litellm.InternalServerError(
                    message=f"synthetic upstream error ({tag})", llm_provider=provider, model=model
                )
            else:
                error = litellm.APIError(
                    status_code=502,
                    message=f"synthetic bad gateway ({tag})",
                    llm_provider=provider,
                    model=model,
                )
            return None, error, f"infrastructure:{type(error).__name__}"
        if u_outcome < edge + shares.transient and attempt == 1:
            error = litellm.RateLimitError(
                message=f"synthetic rate limit ({tag})", llm_provider=provider, model=model
            )
            return None, error, "transient:RateLimitError"
        edge += shares.transient

        prompt = user_prompt_of(params) or ""
        multi_choice = "selected_option" in _schema_fields(params.get("response_format"))
        reasoning: str | None = None
        if u_outcome < edge + shares.malformed:
            content, outcome = _malformed_content(multi_choice, prompt, u_kind, u_choice, tag)
        elif u_outcome < edge + shares.malformed + shares.null_content:
            content, outcome = None, "null_content"
        else:
            content, outcome = _valid_content(multi_choice, prompt, u_choice, tag)
            if provider == "gemini" and u_reason < 0.25:
                reasoning = f"Synthetic thought summary {tag}: weighed the evidence once."

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        prompt_chars = sum(len(str(m.get("content", ""))) for m in params.get("messages") or [])
        prompt_tokens = max(1, round(prompt_chars / 4))
        completion_tokens = max(1, round(len(content or "") / 4))
        completion_tokens += round(len(reasoning) / 4) if reasoning else 0
        response = litellm.ModelResponse(
            id=f"synthetic-{digest[:24]}",
            created=SYNTHETIC_CREATED,
            model=SYNTHETIC_REPORTED_MODELS.get(model, model),
            object="chat.completion",
            choices=[{"index": 0, "finish_reason": "stop", "message": message}],
            usage=litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        response._hidden_params = {"custom_llm_provider": provider}
        return response, None, outcome


def _schema_fields(response_format: Any) -> set[str]:
    if isinstance(response_format, dict):
        schema = (response_format.get("json_schema") or {}).get("schema") or {}
        return set(schema.get("properties") or {})
    return set(getattr(response_format, "model_fields", {}) or {})


def _options(prompt: str) -> list[bool]:
    """NA flag of each option listed in a multi-choice prompt, in presented order."""
    block = _block(prompt, "options") or ""
    lines = [line for line in block.split("\n") if line.strip()]
    tokens = ("n/a", "not applicable", "cannot assess")
    return [any(token in line.lower() for token in tokens) for line in lines]


def _valid_content(multi_choice: bool, prompt: str, u_choice: float, tag: str) -> tuple[str, str]:
    if multi_choice:
        na_flags = _options(prompt) or [False]
        weights = [1.0 if na else 3.0 for na in na_flags]
        threshold, chosen = u_choice * sum(weights), len(weights) - 1
        for index, weight in enumerate(weights):
            threshold -= weight
            if threshold < 0:
                chosen = index
                break
        body = {
            "selected_option": chosen + 1,
            "explanation": f"Synthetic judgment {tag}: option {chosen + 1} fits best…",
        }
        outcome = "valid:NA" if na_flags[chosen] else f"valid:option{chosen + 1}"
        return json.dumps(body, ensure_ascii=False), outcome
    if u_choice < 0.5:
        verdict = "MET"
    elif u_choice < 0.9:
        verdict = "UNMET"
    else:
        verdict = "CANNOT_ASSESS"
    body = {
        "criterion_status": verdict,
        "explanation": f"Synthetic judgment {tag}: the submission reads as {verdict} — noted.",
    }
    return json.dumps(body, ensure_ascii=False), f"valid:{verdict}"


def _malformed_content(
    multi_choice: bool, prompt: str, u_kind: float, u_choice: float, tag: str
) -> tuple[str, str]:
    kinds = (
        ("prose", "truncated", "bad_type", "out_of_range")
        if multi_choice
        else (
            "prose",
            "truncated",
            "bad_enum",
        )
    )
    kind = kinds[min(int(u_kind * len(kinds)), len(kinds) - 1)]
    if kind == "prose":
        return f"I believe the submission is acceptable overall ({tag}).", "malformed:prose"
    if kind == "truncated":
        valid, _ = _valid_content(multi_choice, prompt, u_choice, tag)
        return valid[: len(valid) // 2], "malformed:truncated"
    if kind == "bad_enum":
        body: dict[str, Any] = {"criterion_status": "PARTIALLY_MET", "explanation": tag}
        return json.dumps(body), "malformed:bad_enum"
    if kind == "bad_type":
        return json.dumps({"selected_option": "second", "explanation": tag}), "malformed:bad_type"
    out_of_range = len(_options(prompt)) + 1
    body = {"selected_option": out_of_range, "explanation": tag}
    return json.dumps(body), "malformed:out_of_range"


# =======================================================================================
# Passes
# =======================================================================================


@dataclass(frozen=True)
class PassSpec:
    """One ``evaluate()`` call: a config, a dataset and a construction path."""

    config_id: str
    dataset_id: str
    construction: pc.Construction

    @property
    def pass_id(self) -> str:
        """Experiment name of the pass."""
        return pc.pass_id(self.config_id, self.dataset_id, self.construction)

    @property
    def config(self) -> pc.ParityConfig:
        """The pass's grader configuration."""
        return pc.CONFIGS[self.config_id]


def plan_passes(code: Code, mode: Mode) -> list[PassSpec]:
    """Passes in execution order: every construction of parents, then derived configs."""
    specs: list[PassSpec] = []
    ordered = sorted(pc.CONFIGS.values(), key=lambda c: c.parent is not None)
    for config in ordered:
        for construction in pc.constructions_for(code, mode, config):
            for dataset_id in config.datasets:
                specs.append(PassSpec(config.config_id, dataset_id, construction))
    return specs


def warning_rows(
    caught: list[warnings.WarningMessage], *, label: str, phase: str, package_dir: Path
) -> list[dict[str, Any]]:
    """Recorded warnings, deduplicated with counts."""
    counts: Counter[tuple[str, str, str, int, bool]] = Counter()
    for item in caught:
        filename = str(Path(item.filename).resolve()) if item.filename else ""
        inside = filename.startswith(str(package_dir) + os.sep)
        category = f"{item.category.__module__}.{item.category.__qualname__}"
        counts[(category, str(item.message), filename, item.lineno, inside)] += 1
    return [
        {
            "pass": label,
            "phase": phase,
            "category": category,
            "is_deprecation": _is_deprecation(category),
            "message": message,
            "filename": filename,
            "lineno": lineno,
            "in_autorubric": inside,
            "autorubric_relpath": (
                Path(filename).relative_to(package_dir).as_posix() if inside else None
            ),
            "count": count,
        }
        for (category, message, filename, lineno, inside), count in sorted(counts.items())
    ]


def _is_deprecation(category: str) -> bool:
    return category.endswith(("DeprecationWarning", "PendingDeprecationWarning")) or (
        "Deprecat" in category
    )


@contextmanager
def recorded_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """``warnings.catch_warnings(record=True)`` with every warning recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def compute_pass_metrics(
    result: EvalResult, dataset: RubricDataset, config: pc.ParityConfig
) -> MetricsResult:
    """``compute_metrics`` with the harness's fixed bootstrap."""
    return result.compute_metrics(
        dataset,
        bootstrap=True,
        n_bootstrap=pc.METRICS_N_BOOTSTRAP,
        seed=pc.METRICS_BOOTSTRAP_SEED,
        per_judge=config.is_ensemble,
    )


def save_metrics(metrics: MetricsResult, out_dir: Path) -> None:
    """Persist every field (JSON), both summaries (text) and the frame (pickle and CSV)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pc.write_json(out_dir / "metrics.json", metrics.model_dump(mode="json"))
    (out_dir / "summary.txt").write_text(metrics.summary(), encoding="utf-8", newline="\n")
    (out_dir / "summary_verbose.txt").write_text(
        metrics.summary(verbose=True), encoding="utf-8", newline="\n"
    )
    frame = metrics.to_dataframe()
    frame.to_pickle(out_dir / "dataframe.pkl")
    frame.to_csv(out_dir / "dataframe.csv", index=False, lineterminator="\n")


def close_clients(grader: CriterionGrader) -> None:
    """Release the grader's response-cache handles."""
    for client in getattr(grader, "_clients", {}).values():
        close = getattr(client, "close", None)
        if callable(close):
            close()


def item_errors(experiment_dir: Path, label: str) -> list[dict[str, Any]]:
    """Errored votes and failed items of one experiment."""
    rows: list[dict[str, Any]] = []
    for item_idx, record in sorted(pc.read_checkpoint(experiment_dir).items()):
        if record.get("error"):
            rows.append({"pass": label, "item_idx": item_idx, "item_error": record["error"]})
        for vote in pc.vote_rows(record):
            if vote.error is not None:
                rows.append(
                    {
                        "pass": label,
                        "item_idx": vote.item_idx,
                        "criterion_idx": vote.criterion_idx,
                        "judge_id": vote.judge_id,
                        "category": vote.error_category,
                        "error": vote.error,
                    }
                )
    return rows


@dataclass
class Runner:
    """State of one parity run."""

    code: Code
    mode: Mode
    run_dir: Path
    library: LibraryIdentity
    datasets: dict[str, RubricDataset]
    recorder: RequestRecorder
    replay_source: Path | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def cache_dir(self, config: pc.ParityConfig) -> Path | None:
        """Response cache of a config (None in synthetic mode, where caching is off)."""
        if self.mode == "synthetic":
            return None
        return self.run_dir / pc.CACHE_SUBDIR / config.cache_group

    def network_for(self, config: pc.ParityConfig) -> Network:
        """Network policy of a config's passes."""
        if self.mode == "synthetic":
            return "synthetic"
        if self.mode == "fresh" and config.parent is None:
            return "live"
        return "blocked"

    def failure_source(self, spec: PassSpec) -> Path | None:
        """Experiment whose failed calls a blocked pass stands in for.

        A fresh derived pass reuses its parent pass's calls; a replay pass reuses the
        replay source's calls for the same config and dataset.
        """
        config = spec.config
        if self.mode == "fresh" and config.parent is not None:
            label = pc.pass_id(config.parent, spec.dataset_id, spec.construction)
            return self.run_dir / pc.EXPERIMENTS_SUBDIR / label
        if self.mode == "replay" and self.replay_source is not None:
            source_meta = pc.read_json(self.replay_source / pc.RUN_FILE)
            construction = source_meta["primary_construction"]
            label = pc.pass_id(spec.config_id, spec.dataset_id, construction)
            return self.replay_source / pc.EXPERIMENTS_SUBDIR / label
        return None

    def context(
        self, label: str, spec: PassSpec, network: Network, dataset: RubricDataset
    ) -> PassContext:
        """Attribution context for a pass."""
        submissions: dict[str, int] = {}
        for index, item in enumerate(dataset.items):
            if item.submission in submissions:
                raise SystemExit(f"{spec.dataset_id}: duplicate submission at item {index}")
            submissions[item.submission] = index
        assert dataset.rubric is not None
        requirements = {c.requirement: i for i, c in enumerate(dataset.rubric.rubric)}
        if len(requirements) != len(dataset.rubric.rubric):
            raise SystemExit(f"{spec.dataset_id}: duplicate criterion requirements")
        judge_by_model = {}
        for judge_id, name in spec.config.judge_names.items():
            judge_by_model[pc.JUDGES[name].model] = judge_id
        return PassContext(
            label=label,
            config_id=spec.config_id,
            dataset_id=spec.dataset_id,
            construction=spec.construction,
            network=network,
            submission_index=submissions,
            requirement_index=requirements,
            judge_by_model=judge_by_model,
        )

    def build(
        self, spec: PassSpec, *, cache_dir: Path | None, construction: pc.Construction
    ) -> CriterionGrader:
        """Build the pass's grader through ``construction``."""
        config = spec.config
        judge_configs = {
            name: pc.judge_llm_config(name, cache_dir=cache_dir) for name, _ in config.judges
        }
        return pc.build_grader(
            config,
            construction,
            judge_configs=judge_configs,
            few_shot_pool=self.datasets["charm_pool"],
        )

    async def grade(self, spec: PassSpec) -> dict[str, Any]:
        """Run one pass: grade, save metrics from memory and from the checkpoint."""
        from autorubric import EvalResult, evaluate

        config = spec.config
        dataset = self.datasets[spec.dataset_id]
        network = self.network_for(config)
        context = self.context(spec.pass_id, spec, network, dataset)
        source = self.failure_source(spec) if network == "blocked" else None
        if source is not None:
            context.blocked_categories = {
                vote.triple: vote.error_category or "unknown"
                for record in pc.read_checkpoint(source).values()
                for vote in pc.vote_rows(record)
                if vote.error is not None
            }
        experiments_dir = self.run_dir / pc.EXPERIMENTS_SUBDIR
        summary: dict[str, Any] = {
            "pass": spec.pass_id,
            "config_id": spec.config_id,
            "dataset_id": spec.dataset_id,
            "construction": spec.construction,
            "parent": config.parent,
            "cache_group": config.cache_group,
            "network": network,
            "n_items": len(dataset),
        }
        started = time.perf_counter()
        self.recorder.active = context
        grader: CriterionGrader | None = None
        try:
            with recorded_warnings() as caught:
                grader = self.build(
                    spec, cache_dir=self.cache_dir(config), construction=spec.construction
                )
                result = await evaluate(
                    dataset,
                    grader,
                    show_progress=False,
                    max_concurrent_items=pc.MAX_CONCURRENT_ITEMS,
                    experiment_name=spec.pass_id,
                    experiments_dir=experiments_dir,
                    resume=False,
                )
            self.warnings += warning_rows(
                caught, label=spec.pass_id, phase="grade", package_dir=self.library.package_dir
            )
        except Exception:
            self._fail(spec.pass_id, "grade")
            return {**summary, "status": "failed"}
        finally:
            self.recorder.active = None
            if grader is not None:
                close_clients(grader)
        summary.update(
            {
                "grade_seconds": round(time.perf_counter() - started, 3),
                "successful_items": result.successful_items,
                "failed_items": result.failed_items,
                "concurrent_duplicate_requests": context.concurrent_duplicates,
            }
        )
        experiment_dir = experiments_dir / spec.pass_id
        self.errors += item_errors(experiment_dir, spec.pass_id)
        try:
            with recorded_warnings() as caught:
                save_metrics(
                    compute_pass_metrics(result, dataset, config),
                    self.run_dir / pc.METRICS_SUBDIR / spec.pass_id,
                )
                reloaded = EvalResult.from_experiment(experiment_dir)
                save_metrics(
                    compute_pass_metrics(reloaded, dataset, config),
                    self.run_dir / pc.CHECKPOINT_METRICS_SUBDIR / spec.pass_id,
                )
            self.warnings += warning_rows(
                caught, label=spec.pass_id, phase="metrics", package_dir=self.library.package_dir
            )
        except Exception:
            self._fail(spec.pass_id, "metrics")
            return {**summary, "status": "metrics_failed"}
        LOGGER.info("  %-40s %6.1fs", spec.pass_id, time.perf_counter() - started)
        return {**summary, "status": "completed"}

    async def checkpoint_compat(self, source: Path) -> dict[str, Any]:
        """Load, re-score and resume (a copy of) every experiment of an older run."""
        from autorubric import EvalResult, evaluate

        source_meta = pc.read_json(source / pc.RUN_FILE)
        out = self.run_dir / pc.CHECKPOINT_COMPAT_SUBDIR
        results: list[dict[str, Any]] = []
        primary = pc.PRIMARY_CONSTRUCTION[self.code]
        for entry in source_meta["passes"]:
            spec = PassSpec(entry["config_id"], entry["dataset_id"], entry["construction"])
            dataset = self.datasets[spec.dataset_id]
            source_dir = source / pc.EXPERIMENTS_SUBDIR / spec.pass_id
            row: dict[str, Any] = {"pass": spec.pass_id}
            try:
                with recorded_warnings() as caught:
                    loaded = EvalResult.from_experiment(source_dir)
                    row["loaded_items"] = len(loaded.item_results)
                    save_metrics(
                        compute_pass_metrics(loaded, dataset, spec.config),
                        out / pc.METRICS_SUBDIR / spec.pass_id,
                    )
                resume_root = out / "resume"
                target = resume_root / spec.pass_id
                shutil.copytree(source_dir, target)
                before = len(pc.read_jsonl(target / "items.jsonl"))
                label = f"checkpoint-resume:{spec.pass_id}"
                context = self.context(label, spec, "blocked", dataset)
                n_records = len(self.recorder.records)
                self.recorder.active = context
                grader = None
                try:
                    with recorded_warnings() as resume_caught:
                        grader = self.build(spec, cache_dir=None, construction=primary)
                        resumed = await evaluate(
                            dataset,
                            grader,
                            show_progress=False,
                            experiment_name=spec.pass_id,
                            experiments_dir=resume_root,
                            resume=True,
                        )
                finally:
                    self.recorder.active = None
                    if grader is not None:
                        close_clients(grader)
                after = len(pc.read_jsonl(target / "items.jsonl"))
                row.update(
                    {
                        "items_before": before,
                        "items_after": after,
                        "items_graded": after - before,
                        "network_attempts": len(self.recorder.records) - n_records,
                        "resumed_items": len(resumed.item_results),
                        "total_items": resumed.total_items,
                        "status": "completed",
                    }
                )
                for rows, phase in ((caught, "checkpoint-load"), (resume_caught, "resume")):
                    self.warnings += warning_rows(
                        rows, label=label, phase=phase, package_dir=self.library.package_dir
                    )
            except Exception:
                row["status"] = "failed"
                row["traceback"] = traceback.format_exc()
            results.append(row)
        report = {"source_run": source_meta["run_id"], "passes": results}
        pc.write_json(out / "result.json", report)
        return report

    def _fail(self, label: str, phase: str) -> None:
        LOGGER.error("  %s failed during %s", label, phase)
        self.failures.append({"pass": label, "phase": phase, "traceback": traceback.format_exc()})


def call_counts(records: list[dict[str, Any]], passes: list[dict[str, Any]]) -> dict[str, Any]:
    """Logical calls (distinct triples) and attempts per pass, config and judge name."""
    by_pass: dict[str, dict[str, Any]] = {}
    for entry in passes:
        label = entry["pass"]
        rows = [r for r in records if r["pass"] == label]
        config = pc.CONFIGS[entry["config_id"]]
        per_judge: Counter[str] = Counter()
        triples = {(r["item_idx"], r["criterion_idx"], r["judge_id"]) for r in rows}
        for _, _, judge_id in triples:
            per_judge[config.judge_names.get(str(judge_id), str(judge_id))] += 1
        by_pass[label] = {
            "calls": len(triples),
            "attempts": len(rows),
            "calls_per_judge": dict(sorted(per_judge.items())),
            "unattributed": sum(
                1 for r in rows if None in (r["item_idx"], r["criterion_idx"], r["judge_id"])
            ),
        }
    return by_pass


def derived_new_requests(
    records: list[dict[str, Any]], passes: list[dict[str, Any]]
) -> dict[str, int]:
    """Per derived pass, the number of distinct requests its parent pass never issued."""
    shas: dict[str, set[str]] = {}
    for record in records:
        shas.setdefault(record["pass"], set()).add(record["request_sha256"])
    result: dict[str, int] = {}
    for entry in passes:
        parent = pc.CONFIGS[entry["config_id"]].parent
        if parent is None:
            continue
        parent_pass = pc.pass_id(parent, entry["dataset_id"], entry["construction"])
        result[entry["pass"]] = len(shas.get(entry["pass"], set()) - shas.get(parent_pass, set()))
    return result


def planned_calls(specs: list[PassSpec], datasets: dict[str, RubricDataset]) -> dict[str, int]:
    """Logical LLM calls per judge of the parent configs' primary passes.

    Derived configs reuse these calls and add none (in fresh mode they read the cache).
    """
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        key = (spec.config_id, spec.dataset_id)
        if spec.config.parent is not None or key in seen:
            continue
        seen.add(key)
        dataset = datasets[spec.dataset_id]
        assert dataset.rubric is not None
        for name, _ in spec.config.judges:
            counts[name] += len(dataset) * len(dataset.rubric.rubric)
    return dict(sorted(counts.items()))


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic order for concurrently issued requests."""

    def key(record: dict[str, Any]) -> tuple[Any, ...]:
        return (
            record["pass"],
            -1 if record["item_idx"] is None else record["item_idx"],
            -1 if record["criterion_idx"] is None else record["criterion_idx"],
            record["judge_id"] or "",
            record["request_sha256"],
            record["attempt"],
        )

    return sorted(records, key=key)


def require_fresh_preconditions(args: argparse.Namespace, cost_map_sha: str) -> None:
    """Refuse a paid run without explicit consent and a matching passing pre-flight."""
    if not args.confirm_paid:
        raise SystemExit("fresh mode makes paid provider calls; pass --confirm-paid to proceed")
    if not pc.PREFLIGHT_PATH.exists():
        raise SystemExit(f"no pre-flight result at {pc.PREFLIGHT_PATH}; run --preflight first")
    preflight = pc.read_json(pc.PREFLIGHT_PATH)
    versions = package_versions()["pinned"]
    problems = []
    if not preflight.get("passed"):
        problems.append("the recorded pre-flight did not pass")
    if preflight.get("packages") != versions:
        problems.append("package versions changed since the pre-flight")
    if preflight.get("cost_map_sha256") != cost_map_sha:
        problems.append("the cost map changed since the pre-flight")
    if preflight.get("judges_fingerprint") != judges_fingerprint():
        problems.append("the judge definitions changed since the pre-flight")
    if problems:
        raise SystemExit("fresh run refused: " + "; ".join(problems))


def source_run(run_id: str, *, purpose: str, mode: str | None = None) -> Path:
    """Directory of an existing run used as a replay or checkpoint source."""
    source = pc.PARITY_DIR / run_id
    meta_path = source / pc.RUN_FILE
    if not meta_path.exists():
        raise SystemExit(f"{purpose}: no run at {source}")
    actual = pc.read_json(meta_path).get("mode")
    if mode is not None and actual != mode:
        raise SystemExit(f"{purpose}: {run_id} is a {actual} run; it must be a {mode} run")
    return source


async def execute_run(args: argparse.Namespace) -> int:
    """Grade every pass of the run and write its artifacts."""
    code: Code = args.code
    mode: Mode = args.mode
    if not args.run:
        raise SystemExit("--run <id> is required")
    manifest = load_manifest()
    library = load_library(expected_tree(code, manifest))
    api = pc.detect_api()
    if code == "after" and not api.has_judge_model_config:
        raise SystemExit(
            "--code after requested, but the loaded library lacks judge_model_config "
            f"(CriterionGrader: {api.grader_judge_model_config}, "
            f"JudgeSpec: {api.judgespec_judge_model_config})"
        )
    cost_map_sha = pin_cost_map(manifest)
    if mode == "fresh":
        require_fresh_preconditions(args, cost_map_sha)

    replay_source: str | None = None
    if mode == "replay":
        if not args.replay_cache:
            raise SystemExit("replay mode needs --replay-cache <fresh run id>")
        source_run(args.replay_cache, purpose="--replay-cache", mode="fresh")
        replay_source = args.replay_cache
    checkpoint_source = args.checkpoint_source or replay_source
    if checkpoint_source:
        source_run(checkpoint_source, purpose="--checkpoint-source")
    run_dir = pc.PARITY_DIR / args.run
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists; choose a new --run id")
    run_dir.mkdir(parents=True)
    if replay_source:
        # Copy, so the fresh run's own cache is never touched by a replay.
        shutil.copytree(pc.PARITY_DIR / replay_source / pc.CACHE_SUBDIR, run_dir / pc.CACHE_SUBDIR)

    import litellm

    synthetic = SyntheticProvider() if mode == "synthetic" else None
    recorder = RequestRecorder(litellm.acompletion, synthetic)
    runner = Runner(
        code=code,
        mode=mode,
        run_dir=run_dir,
        library=library,
        datasets=pc.load_parity_datasets(manifest),
        recorder=recorder,
        replay_source=pc.PARITY_DIR / replay_source if replay_source else None,
    )
    specs = plan_passes(code, mode)
    planned = planned_calls(specs, runner.datasets)
    LOGGER.info(
        "%s LLM calls planned: %s (total %d, plus retries)",
        "PAID" if mode == "fresh" else "free",
        planned,
        sum(planned.values()),
    )
    guard = SocketGuard() if mode != "fresh" else None
    started_at = datetime.now(UTC)
    LOGGER.info(
        "run %s: code=%s mode=%s library=%s (%d passes)",
        args.run,
        code,
        mode,
        library.package_dir,
        len(specs),
    )
    passes: list[dict[str, Any]] = []
    compat: dict[str, Any] | None = None
    if guard is not None:
        guard.install()
    try:
        with patched_acompletion(recorder):
            for spec in specs:
                passes.append(await runner.grade(spec))
            if checkpoint_source:
                compat = await runner.checkpoint_compat(pc.PARITY_DIR / checkpoint_source)
    finally:
        if guard is not None:
            guard.uninstall()

    records = sort_records(recorder.records)
    with (run_dir / pc.REQUESTS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(pc.canonical_json(record) + "\n")
    pc.write_json(run_dir / pc.WARNINGS_FILE, runner.warnings)
    pc.write_json(run_dir / pc.ERRORS_FILE, runner.errors)

    graded = [p for p in passes if p.get("status") == "completed"]
    provider_models: dict[str, list[str]] = {}
    for record in records:
        if record.get("provider_model"):
            name = record["request"].get("model")
            provider_models.setdefault(str(name), [])
            if record["provider_model"] not in provider_models[str(name)]:
                provider_models[str(name)].append(record["provider_model"])
    meta = {
        "run_id": args.run,
        "code": code,
        "mode": mode,
        "status": "completed" if not runner.failures else "completed_with_failures",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "library": library.to_dict(),
        "api": api.__dict__,
        "primary_construction": pc.PRIMARY_CONSTRUCTION[code],
        "packages": package_versions(),
        "cost_map_sha256": cost_map_sha,
        "parity_manifest_sha256": pc.sha256_file(pc.MANIFEST_PATH),
        "harness_sha256": {
            path.name: pc.sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "judges": {
            name: {
                "model": judge.model,
                "temperature": pc.TEMPERATURE,
                "thinking": pc.THINKING,
                "extra_params": dict(judge.extra_params),
            }
            for name, judge in pc.JUDGES.items()
        },
        "judges_fingerprint": judges_fingerprint(),
        "response_cache": "off" if mode == "synthetic" else "on",
        "synthetic_provider": synthetic.describe() if synthetic else None,
        "replay_source": replay_source,
        "checkpoint_source": checkpoint_source,
        "passes": passes,
        "failures": runner.failures,
        "planned_calls": planned,
        "call_counts": call_counts(records, graded),
        "derived_new_requests": derived_new_requests(records, graded),
        "provider_models": provider_models,
        "socket_attempts": guard.attempts if guard is not None else None,
        "checkpoint_compat": compat is not None,
    }
    pc.write_json(run_dir / pc.RUN_FILE, meta)
    LOGGER.info("wrote %s (%d requests recorded)", run_dir, len(records))
    return 1 if runner.failures else 0


# =======================================================================================
# Pre-flight
# =======================================================================================


def _lookup(body: Any, dotted: str) -> Any:
    value = body
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return "<missing>"
        value = value[part]
    return value


def _closed_loopback_port() -> int:
    """A loopback port nothing listens on (bound, then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def execute_preflight(args: argparse.Namespace) -> int:
    """Send one grading call per judge and check what litellm put on the wire."""
    offline = bool(args.offline)
    if not offline and not args.confirm_paid:
        raise SystemExit("the pre-flight makes two paid calls; pass --confirm-paid (or --offline)")
    manifest = load_manifest()
    library = load_library(expected_tree(args.code, manifest))
    cost_map_sha = pin_cost_map(manifest)

    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    from autorubric import Rubric

    captured: list[dict[str, Any]] = []

    class CaptureRequest(CustomLogger):
        """Captures the provider request body litellm is about to send."""

        def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:
            additional = (kwargs or {}).get("additional_args") or {}
            captured.append(
                {
                    "model": model,
                    "api_base": additional.get("api_base"),
                    "body": to_jsonable(additional.get("complete_input_dict")),
                }
            )

    capture = CaptureRequest()
    litellm.input_callback.append(capture)
    essay = pc.load_parity_datasets(manifest)["essay"]
    assert essay.rubric is not None
    rubric = Rubric([essay.rubric.rubric[0]])
    primary = pc.PRIMARY_CONSTRUCTION[args.code]
    results: dict[str, Any] = {}
    try:
        for name, judge in pc.JUDGES.items():
            overrides = None
            if offline:
                overrides = {
                    "api_base": f"http://127.0.0.1:{_closed_loopback_port()}",
                    "api_key": "offline-preflight-dummy-key",
                    "max_retries": 1,
                    "timeout": 10.0,
                }
            judge_config = pc.judge_llm_config(name, cache_dir=None, overrides=overrides)
            config = pc.ParityConfig(f"preflight-{name}", ("essay",), ((name, 1.0),))
            captured.clear()
            with recorded_warnings():
                grader = pc.build_grader(
                    config, primary, judge_configs={name: judge_config}, few_shot_pool=None
                )
                report = await rubric.grade(
                    to_grade=essay.items[0].submission,
                    grader=grader,
                    query=essay.get_item_prompt(0),
                )
            votes = getattr(report.report[0], "votes", []) if report.report else []
            vote_error = votes[0].error if votes else "no vote recorded"
            bodies = [c["body"] for c in captured]
            checks = [
                {
                    "path": path,
                    "expected": expected,
                    "observed": [_lookup(body, path) for body in bodies],
                }
                for path, expected in judge.expected_request
            ]
            params_ok = bool(bodies) and all(
                all(observed == check["expected"] for observed in check["observed"])
                for check in checks
            )
            accepted = vote_error is None
            results[name] = {
                "model": judge.model,
                "requests_captured": len(bodies),
                "checks": checks,
                "request_parameters_ok": params_ok,
                "provider_accepted": None if offline else accepted,
                "judge_error": vote_error,
                "captured": captured[:],
                "passed": params_ok and (offline or accepted),
            }
            LOGGER.info(
                "%s: parameters %s; %s",
                name,
                "as expected" if params_ok else "NOT as expected",
                vote_error or "provider accepted",
            )
    finally:
        litellm.input_callback.remove(capture)
    passed = all(entry["passed"] for entry in results.values())
    outcome = {
        "offline": offline,
        "passed": passed,
        "created_at": datetime.now(UTC).isoformat(),
        "library": library.to_dict(),
        "packages": package_versions()["pinned"],
        "cost_map_sha256": cost_map_sha,
        "judges_fingerprint": judges_fingerprint(),
        "judges": results,
    }
    path = pc.OFFLINE_PREFLIGHT_PATH if offline else pc.PREFLIGHT_PATH
    pc.write_json(path, outcome)
    LOGGER.info("pre-flight %s; wrote %s", "passed" if passed else "FAILED", path)
    return 0 if passed else 1


# =======================================================================================
# Command line
# =======================================================================================


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--code", choices=("baseline", "after"), required=True)
    parser.add_argument("--mode", choices=("fresh", "replay", "synthetic"))
    parser.add_argument("--run", help="run id: the output directory under experiments/parity/")
    parser.add_argument("--replay-cache", help="fresh run whose response cache replay reads")
    parser.add_argument(
        "--checkpoint-source",
        help="run whose checkpoints are loaded, re-scored and resumed (replay default: the "
        "--replay-cache run)",
    )
    parser.add_argument("--init-manifest", action="store_true")
    parser.add_argument("--baseline-tree", help="baseline worktree (for --init-manifest)")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="pre-flight against a closed loopback port with a dummy key (no paid call)",
    )
    parser.add_argument("--confirm-paid", action="store_true", help="allow paid provider calls")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not (args.init_manifest or args.preflight) and args.mode is None:
        parser.error("--mode is required for a run")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    prepare_litellm_environment()
    configure_logging(args.verbose)
    if args.init_manifest:
        return init_manifest(args)
    if args.preflight:
        return asyncio.run(execute_preflight(args))
    return asyncio.run(execute_run(args))


if __name__ == "__main__":
    sys.exit(main())
