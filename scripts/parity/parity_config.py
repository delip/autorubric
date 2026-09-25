"""Fixed inputs of the live-regression parity harness: datasets, judges, configs, seeds.

Everything that decides *what* is graded lives here, so baseline and post-change runs are
built from one definition. This module never imports ``autorubric`` at import time (the
comparator reads it without loading any library); the grader builders import it lazily, so
they run against whichever library the calling process has loaded.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from autorubric import LLMConfig, RubricDataset
    from autorubric.graders import CriterionGrader

# ---------------------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository that holds this harness; its ``src/`` is the post-change library."""

AFTER_LIBRARY_TREE = REPO_ROOT
"""Tree whose ``src/autorubric`` is the library under test for ``--code after``."""

PARITY_DIR = REPO_ROOT / "experiments" / "parity"
"""Root of all parity outputs (gitignored)."""

MANIFEST_PATH = PARITY_DIR / "manifest.json"
"""Parity manifest: baseline tree, dataset snapshots, charm-100 split, cost-map snapshot."""

DATA_DIR = PARITY_DIR / "data"
"""Dataset snapshots taken from the baseline tree when the manifest is created."""

COST_MAP_PATH = PARITY_DIR / "model_cost_map.json"
"""Snapshot of litellm's remote model cost map, pinned for every run."""

PREFLIGHT_PATH = PARITY_DIR / "preflight.json"
"""Result of the paid pre-flight; fresh runs require a passing one."""

OFFLINE_PREFLIGHT_PATH = PARITY_DIR / "preflight-offline.json"
"""Result of the free loopback pre-flight (request shape only; never unlocks fresh runs)."""

# ---------------------------------------------------------------------------------------
# Seeds and run settings
# ---------------------------------------------------------------------------------------

SEED = 0
"""Master seed of every grader: pins option shuffles and few-shot selection."""

METRICS_BOOTSTRAP_SEED = 0
METRICS_N_BOOTSTRAP = 1000

MAX_CONCURRENT_ITEMS = 10
"""``evaluate(max_concurrent_items=...)``: throttles bursts of paid calls. It only affects
scheduling (results do not depend on it) and is recorded in ``eval_config``, not in the
compared ``grader_config``."""

PINNED_PACKAGES = ("litellm", "pydantic", "openai", "diskcache")
"""Runs are comparable only when these installed versions are identical."""

# ---------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSource:
    """A dataset file, addressed relative to the library tree it is snapshotted from."""

    dataset_id: str
    relative_path: str

    @property
    def snapshot_path(self) -> Path:
        """Where the manifest step stores the pinned copy."""
        return DATA_DIR / Path(self.relative_path).name


DATASETS: dict[str, DatasetSource] = {
    "essay": DatasetSource("essay", "examples/data/essay_grading_dataset.json"),
    "charm": DatasetSource("charm", "examples/data/charm100.json"),
}

CHARM_SPLIT: dict[str, Any] = {"n_train": 70, "stratify": True, "seed": 0}
"""``split_train_test`` arguments for the charm-100 subset: 30 test items graded, the other
70 are the few-shot pool of config C."""

# ---------------------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------------------

TEMPERATURE = 1.0
"""Sent explicitly: the provider default of both models, written out so both phases send the
same value regardless of ``LLMConfig.temperature``'s default."""

THINKING = "none"
"""No reasoning (or its minimum) for both judges."""


@dataclass(frozen=True)
class ParityJudge:
    """One live LLM judge.

    Attributes:
        name: Judge name used in ensembles (``JudgeSpec.judge_id``) and reports.
        model: litellm model id.
        provider: Provider name litellm routes the model to.
        extra_params: Extra litellm parameters (``LLMConfig.extra_params``).
        expected_request: Dotted paths into litellm's outgoing request body and the value
            the pre-flight requires at each.
    """

    name: str
    model: str
    provider: str
    extra_params: tuple[tuple[str, tuple[str, ...]], ...] = ()
    expected_request: tuple[tuple[str, Any], ...] = ()


JUDGES: dict[str, ParityJudge] = {
    # litellm 1.95.0 routes "gpt-6-luna" through its generic GPT parameter table, which lists
    # no ``reasoning_effort``, so ``thinking="none"`` alone is rejected client-side with
    # UnsupportedParamsError before any request is sent. ``allowed_openai_params`` is
    # litellm's documented pass-through for a parameter its table does not list; with it the
    # request carries ``reasoning_effort="none"`` as intended. It is consumed by litellm and
    # never sent to the provider.
    "luna": ParityJudge(
        name="luna",
        model="gpt-6-luna",
        provider="openai",
        extra_params=(("allowed_openai_params", ("reasoning_effort",)),),
        expected_request=(("temperature", 1.0), ("reasoning_effort", "none")),
    ),
    # For Gemini 3.x Flash models litellm maps reasoning_effort="none" to the minimal
    # thinking level.
    "flashlite": ParityJudge(
        name="flashlite",
        model="gemini/gemini-3.5-flash-lite",
        provider="gemini",
        expected_request=(
            ("generationConfig.temperature", 1.0),
            ("generationConfig.thinkingConfig.thinkingLevel", "minimal"),
        ),
    ),
}


def judge_llm_config(
    judge_name: str,
    *,
    cache_dir: Path | None,
    overrides: dict[str, Any] | None = None,
) -> LLMConfig:
    """Build a judge's ``LLMConfig``.

    Args:
        judge_name: Key of ``JUDGES``.
        cache_dir: Response-cache directory, or None to disable the response cache.
        overrides: Extra ``LLMConfig`` fields (used only by the offline pre-flight, which
            points the judge at a closed loopback port with a dummy key).

    Returns:
        The judge's configuration.
    """
    from autorubric import LLMConfig

    judge = JUDGES[judge_name]
    fields: dict[str, Any] = {
        "model": judge.model,
        "temperature": TEMPERATURE,
        "thinking": THINKING,
        "extra_params": {key: list(value) for key, value in judge.extra_params},
    }
    if cache_dir is not None:
        fields["cache_enabled"] = True
        fields["cache_dir"] = str(cache_dir)
    fields.update(overrides or {})
    return LLMConfig(**fields)


# ---------------------------------------------------------------------------------------
# Grader configurations
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityConfig:
    """One grader configuration.

    Only settings that differ from ``CriterionGrader``'s defaults are passed, so a changed
    default would show up as a parity difference rather than be masked.

    Attributes:
        config_id: Name used in pass ids and reports.
        datasets: Dataset ids graded with this config.
        judges: ``(judge name, weight)`` pairs; one pair means a single-judge grader.
        parent: For a derived config, the config whose LLM calls it reuses (it changes only
            post-call processing, so it issues no new requests).
        grader_kwargs: Non-default scalar ``CriterionGrader`` keywords.
        few_shot: Use the 70-item charm pool with ``FewShotConfig(n_examples=3)``.
        derived_scoring: Add ``LengthPenalty``, ``CannotAssessStrategy.PARTIAL`` and
            ``normalize=False``.
    """

    config_id: str
    datasets: tuple[str, ...]
    judges: tuple[tuple[str, float], ...]
    parent: str | None = None
    grader_kwargs: tuple[tuple[str, Any], ...] = ()
    few_shot: bool = False
    derived_scoring: bool = False

    @property
    def is_ensemble(self) -> bool:
        """Whether the grader is built from ``judges=[JudgeSpec, ...]``."""
        return len(self.judges) > 1

    @property
    def cache_group(self) -> str:
        """Config whose response cache this config reads (itself unless derived)."""
        return self.parent or self.config_id

    @property
    def judge_names(self) -> dict[str, str]:
        """Map from the ``judge_id`` recorded in reports to the judge name."""
        if self.is_ensemble:
            return {name: name for name, _ in self.judges}
        return {"default": self.judges[0][0]}


CONFIGS: dict[str, ParityConfig] = {
    config.config_id: config
    for config in (
        ParityConfig("A-luna", ("essay", "charm"), (("luna", 1.0),)),
        ParityConfig("A-flashlite", ("essay", "charm"), (("flashlite", 1.0),)),
        ParityConfig(
            "B",
            ("charm",),
            (("luna", 1.0), ("flashlite", 1.0)),
            grader_kwargs=(("aggregation", "majority"),),
        ),
        ParityConfig("C", ("charm",), (("luna", 1.0),), few_shot=True),
        ParityConfig(
            "A-prime-luna",
            ("essay", "charm"),
            (("luna", 1.0),),
            parent="A-luna",
            derived_scoring=True,
        ),
        ParityConfig(
            "A-prime-flashlite",
            ("essay", "charm"),
            (("flashlite", 1.0),),
            parent="A-flashlite",
            derived_scoring=True,
        ),
        ParityConfig(
            "B-prime",
            ("charm",),
            (("luna", 2.0), ("flashlite", 1.0)),
            parent="B",
            grader_kwargs=(
                ("aggregation", "weighted"),
                ("ordinal_aggregation", "median"),
                ("nominal_aggregation", "unanimous"),
            ),
        ),
    )
}
"""Parent configs first: in fresh mode a derived config reads its parent's response cache."""

PAID_CONFIGS = tuple(cid for cid, config in CONFIGS.items() if config.parent is None)
"""Configs that issue LLM calls; test L compares their verdicts."""

LENGTH_PENALTY_KWARGS: dict[str, Any] = {
    "free_budget": 60,
    "max_cap": 240,
    "penalty_at_cap": 10.0,
}
"""Derived-config length penalty in words. Essays (25-142 words) and charm responses
(30-499 words) fall below, inside and above the ramp, and the absolute 10-point cap suits
``normalize=False`` raw scores."""

# ---------------------------------------------------------------------------------------
# Construction paths
# ---------------------------------------------------------------------------------------

Construction = Literal[
    "baseline", "judge_model_config", "llm_config", "judgespec_judge_model_config"
]
"""How a grader is built.

- ``baseline``: ``CriterionGrader(llm_config=...)`` / positional ``JudgeSpec(cfg, id)``,
  the only API the baseline library has.
- ``judge_model_config``: ``CriterionGrader(judge_model_config=...)`` / positional
  ``JudgeSpec(cfg, id)`` (the primary post-change construction).
- ``llm_config``: ``CriterionGrader(llm_config=...)`` (the deprecated alias) /
  ``JudgeSpec(llm_config=..., judge_id=...)`` (the stored field's own keyword).
- ``judgespec_judge_model_config``: ensembles only,
  ``JudgeSpec(judge_model_config=..., judge_id=...)``.
"""

PRIMARY_CONSTRUCTION: dict[str, Construction] = {
    "baseline": "baseline",
    "after": "judge_model_config",
}

DEPRECATED_CONSTRUCTION: Construction = "llm_config"
"""The only construction allowed to emit a ``DeprecationWarning`` (single-judge graders)."""


def constructions_for(code: str, mode: str, config: ParityConfig) -> tuple[Construction, ...]:
    """Constructions a run builds for one config.

    Fresh runs build only the primary construction. Post-change replay and synthetic runs
    also build every alternative, which must all process the same outputs identically.
    """
    primary = PRIMARY_CONSTRUCTION[code]
    if code == "baseline" or mode == "fresh":
        return (primary,)
    extra: tuple[Construction, ...] = ("llm_config",)
    if config.is_ensemble:
        extra += ("judgespec_judge_model_config",)
    return (primary, *extra)


@dataclass(frozen=True)
class ApiSupport:
    """Which judge-configuration keywords the loaded library accepts (by introspection)."""

    grader_judge_model_config: bool
    judgespec_judge_model_config: bool

    @property
    def has_judge_model_config(self) -> bool:
        """Both ``CriterionGrader`` and ``JudgeSpec`` accept ``judge_model_config``."""
        return self.grader_judge_model_config and self.judgespec_judge_model_config


def detect_api() -> ApiSupport:
    """Inspect the loaded library's ``CriterionGrader`` and ``JudgeSpec`` signatures."""
    from autorubric.graders import CriterionGrader, JudgeSpec

    grader_params = inspect.signature(CriterionGrader.__init__).parameters
    spec_params = inspect.signature(JudgeSpec.__init__).parameters
    return ApiSupport(
        grader_judge_model_config="judge_model_config" in grader_params,
        judgespec_judge_model_config="judge_model_config" in spec_params,
    )


def build_grader(
    config: ParityConfig,
    construction: Construction,
    *,
    judge_configs: dict[str, LLMConfig],
    few_shot_pool: RubricDataset | None,
) -> CriterionGrader:
    """Build a config's grader through one construction path.

    Args:
        config: The grader configuration.
        construction: Which API spelling to use (see ``Construction``).
        judge_configs: ``LLMConfig`` per judge name.
        few_shot_pool: The 70-item charm pool (required when ``config.few_shot``).

    Returns:
        The grader.

    Raises:
        ValueError: If the construction does not apply to the config.
    """
    from autorubric import (
        CannotAssessConfig,
        CannotAssessStrategy,
        FewShotConfig,
        LengthPenalty,
    )
    from autorubric.graders import CriterionGrader, JudgeSpec

    kwargs: dict[str, Any] = {"seed": SEED, **dict(config.grader_kwargs)}
    if config.few_shot:
        if few_shot_pool is None:
            raise ValueError(f"{config.config_id} needs the few-shot pool")
        kwargs["training_data"] = few_shot_pool
        kwargs["few_shot_config"] = FewShotConfig(n_examples=3)
    if config.derived_scoring:
        kwargs["length_penalty"] = LengthPenalty(**LENGTH_PENALTY_KWARGS)
        kwargs["cannot_assess_config"] = CannotAssessConfig(strategy=CannotAssessStrategy.PARTIAL)
        kwargs["normalize"] = False

    if config.is_ensemble:
        judges = []
        for name, weight in config.judges:
            judge_config = judge_configs[name]
            if construction in ("baseline", "judge_model_config"):
                judges.append(JudgeSpec(judge_config, name, weight=weight))
            elif construction == "llm_config":
                judges.append(JudgeSpec(llm_config=judge_config, judge_id=name, weight=weight))
            elif construction == "judgespec_judge_model_config":
                judges.append(
                    JudgeSpec(judge_model_config=judge_config, judge_id=name, weight=weight)
                )
            else:
                raise ValueError(f"unknown construction {construction!r}")
        return CriterionGrader(judges=judges, **kwargs)

    judge_config = judge_configs[config.judges[0][0]]
    if construction in ("baseline", "llm_config"):
        return CriterionGrader(llm_config=judge_config, **kwargs)
    if construction == "judge_model_config":
        return CriterionGrader(judge_model_config=judge_config, **kwargs)
    raise ValueError(f"construction {construction!r} does not apply to {config.config_id}")


# ---------------------------------------------------------------------------------------
# Datasets from the manifest
# ---------------------------------------------------------------------------------------


def load_parity_datasets(manifest: dict[str, Any]) -> dict[str, RubricDataset]:
    """Load the graded datasets and the few-shot pool from the manifest's snapshots.

    Returns:
        ``{"essay": ..., "charm": <30 test items>, "charm_pool": <70 train items>}``, with
        charm items in the exact order ``split_train_test`` produced on the baseline code.
    """
    from autorubric import RubricDataset

    essay = RubricDataset.from_file(DATASETS["essay"].snapshot_path)
    charm_full = RubricDataset.from_file(DATASETS["charm"].snapshot_path)
    split = manifest["charm_split"]
    return {
        "essay": essay,
        "charm": charm_subset(charm_full, split["test_indices"]),
        "charm_pool": charm_subset(charm_full, split["train_indices"]),
    }


def charm_subset(full: RubricDataset, indices: list[int]) -> RubricDataset:
    """Rebuild a ``split_train_test`` half from item indices, preserving their order."""
    from autorubric import RubricDataset

    return RubricDataset(
        prompt=full.prompt,
        rubric=full.rubric,
        items=[full.items[i] for i in indices],
        name=full.name,
        reference_submission=full.reference_submission,
    )


# ---------------------------------------------------------------------------------------
# Run layout and artifact readers (shared by the runner and the comparator)
# ---------------------------------------------------------------------------------------

RUN_FILE = "run.json"
REQUESTS_FILE = "requests.jsonl"
WARNINGS_FILE = "warnings.json"
ERRORS_FILE = "errors.json"
EXPERIMENTS_SUBDIR = "experiments"
METRICS_SUBDIR = "metrics"
CHECKPOINT_METRICS_SUBDIR = "metrics_ckpt"
CACHE_SUBDIR = "cache"
CHECKPOINT_COMPAT_SUBDIR = "checkpoint_compat"


def pass_id(config_id: str, dataset_id: str, construction: str) -> str:
    """Name of one graded pass; also its ``evaluate()`` experiment name."""
    return f"{config_id}__{dataset_id}__{construction}"


def canonical_json(value: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, non-ASCII kept."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=True
    )


def sha256_text(text: str) -> str:
    """Hex SHA-256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Write pretty, key-sorted JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file; a missing file reads as empty."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_checkpoint(experiment_dir: Path) -> dict[int, dict[str, Any]]:
    """``items.jsonl`` records of one ``evaluate()`` experiment, keyed by item index.

    Records are appended in completion order, so they are re-keyed here.
    """
    return {record["item_idx"]: record for record in read_jsonl(experiment_dir / "items.jsonl")}


@dataclass(frozen=True)
class VoteRow:
    """One judge's vote on one criterion of one item, read from a checkpoint.

    Attributes:
        item_idx: Item index within the graded dataset.
        criterion_idx: Criterion index within the rubric.
        judge_id: ``judge_id`` as recorded in the report (``"default"`` for single judges).
        multi_choice: Whether the criterion is multi-choice.
        scale_type: The criterion's ``scale_type`` (``None`` for binary criteria).
        category: Agreement category: the binary verdict, or ``"NA"`` / the selected
            option index for multi-choice (an abstention is its own category).
        value: The selected option's ``value`` (multi-choice only).
        error: The vote's ``error`` string, set when the judge call failed.
    """

    item_idx: int
    criterion_idx: int
    judge_id: str
    multi_choice: bool
    scale_type: str | None
    category: str
    value: float | None
    error: str | None

    @property
    def triple(self) -> tuple[int, int, str]:
        """``(item, criterion, judge)`` key."""
        return (self.item_idx, self.criterion_idx, self.judge_id)

    @property
    def error_category(self) -> str | None:
        """Category prefix of ``error`` (``infrastructure`` / ``parse`` / ``unknown``)."""
        if self.error is None:
            return None
        return self.error.split(":", 1)[0].strip()


def vote_rows(record: dict[str, Any]) -> list[VoteRow]:
    """Per-judge votes of one checkpoint record (ensemble reports only).

    Every ``CriterionGrader`` report is an ensemble report; an item whose grading failed
    as a whole has no criterion reports and yields no rows.
    """
    report = record.get("report") or {}
    if report.get("report_type") != "ensemble":
        return []
    rows: list[VoteRow] = []
    for criterion_idx, ecr in enumerate(report.get("criterion_reports") or []):
        criterion = ecr.get("criterion") or {}
        scale_type = criterion.get("scale_type") if criterion.get("options") else None
        for vote in ecr.get("votes") or []:
            rows.append(
                VoteRow(
                    item_idx=record["item_idx"],
                    criterion_idx=criterion_idx,
                    judge_id=vote["judge_id"],
                    multi_choice=False,
                    scale_type=None,
                    category=str(vote.get("verdict")),
                    value=None,
                    error=vote.get("error"),
                )
            )
        for vote in ecr.get("multi_choice_votes") or []:
            category = "NA" if vote.get("na") else str(vote.get("selected_index"))
            rows.append(
                VoteRow(
                    item_idx=record["item_idx"],
                    criterion_idx=criterion_idx,
                    judge_id=vote["judge_id"],
                    multi_choice=True,
                    scale_type=scale_type,
                    category=category,
                    value=vote.get("value"),
                    error=vote.get("error"),
                )
            )
    return rows
