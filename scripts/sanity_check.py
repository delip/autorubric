#!/usr/bin/env python3
"""Fast, high-observability sanity check across three datasets + judge configs.

Grades a SMALL sample of three datasets — chosen to cover every criterion shape and
abstain mechanism — under a single judge and a cross-provider ensemble, with NO response
caching and maximum parallelism, then surfaces agreement metrics + wall-clock + LLM cost.

Datasets (loaded from examples/data/):
- ricechem  : binary MET/UNMET (Cohen's kappa, P/R/F1, CANNOT_ASSESS)
- hashemi   : multi-choice ordinal (+author NA) + nominal (weighted-kappa, NA handling)
- charm-100 : mixed ordinal + nominal + binary + NA in one rubric

Judges:
- single   : gemini/gemini-3.1-flash-lite
- ensemble : gemini/gemini-3.1-flash-lite + openai/gpt-5.4-nano (inter-judge alpha/Fleiss)

By default it runs BOTH thinking conditions (no-thinking + thinking) in a single
concurrent pool with no-thinking prioritized and thinking backfilling spare provider
capacity — maximum throughput. Output is a live Rich TUI dashboard by default, or plain
parseable text with --plain (also auto-selected on a non-TTY, e.g. when piped to an agent).

Usage:
    uv run python scripts/sanity_check.py                       # Rich TUI, N=10, both think modes
    uv run python scripts/sanity_check.py --plain --sample-size 5
    uv run python scripts/sanity_check.py --thinking-mode off --single-only
    uv run python scripts/sanity_check.py --datasets charm-100 --ricechem-question 2

Needs GEMINI_API_KEY (or GOOGLE_API_KEY) and, unless --single-only, OPENAI_API_KEY.
"""

import argparse
import asyncio
import logging
import os
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import NamedTuple

import litellm
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    EvalConfig,
    EvalRunner,
    ItemResult,
    LLMConfig,
    RubricDataset,
)
from autorubric.graders import CriterionGrader, JudgeSpec


def _fmt_opt(value: float | None, spec: str, width: int = 0) -> str:
    """None-safe fixed-width formatter: render the value, or right-aligned 'n/a' if None.

    Defined locally (rather than importing autorubric's private helper) so this dev script
    stays self-contained and runs against any autorubric version.
    """
    return format(value, spec) if value is not None else f"{'n/a':>{width}}"


# Silence known-benign third-party noise so the dashboard / plain report stays signal-only:
# - litellm serializes its non-streaming `Choices` against a `StreamingChoices` schema
# - sklearn/numpy complain on degenerate tiny samples (single-class kappa -> handled as None)
warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings")
warnings.filterwarnings("ignore", message=r".*single label was found.*")
warnings.filterwarnings("ignore", message=r".*invalid value encountered in scalar divide.*")

# --- Constants -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "examples" / "data"

# Judge model strings (LiteLLM format). The provider prefix (gemini/openai) drives the
# per-provider rate-limit bucket. NOTE: "gemini/" (not "google/") is the prefix that routes
# to Google AI Studio via GEMINI_API_KEY in standard LiteLLM.
SINGLE_MODEL = "gemini/gemini-3.1-flash-lite"
NANO_MODEL = "openai/gpt-5.4-nano"

ALL_DATASETS = ("ricechem", "hashemi", "charm-100")
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_MAX_PARALLEL = 30
DEFAULT_SEED = 42

# No-thinking vs thinking inference hyperparameters (max_tokens differ for reasoning room).
MAX_TOKENS_NO_THINK = 2048
MAX_TOKENS_THINK = 8192

logger = logging.getLogger("sanity_check")


class Job(NamedTuple):
    """One grading run = (dataset, judge mode, thinking on/off)."""

    dataset: str
    mode: str  # "single" | "ensemble"
    think: bool


class JobResult(NamedTuple):
    job: Job
    result: object | None  # EvalResult | None
    metrics: object | None  # MetricsResult | None
    error: str | None


# Display order: no-thinking before thinking, dataset alphabetical, single before ensemble.
_MODE_ORDER = {"single": 0, "ensemble": 1}


def _display_sort_key(jr: "JobResult") -> tuple[bool, str, int]:
    return (jr.job.think, jr.job.dataset, _MODE_ORDER.get(jr.job.mode, 99))


# --- Data loading & sampling -----------------------------------------------------------


def subsample_dataset(dataset: RubricDataset, max_items: int, seed: int) -> RubricDataset:
    """Return a new dataset with a reproducible random subset of items (seeded)."""
    if max_items >= len(dataset.items):
        return dataset
    rng = random.Random(seed)
    sampled = rng.sample(dataset.items, max_items)
    return RubricDataset(
        name=dataset.name,
        prompt=dataset.prompt,
        rubric=dataset.rubric,
        items=sampled,
        reference_submission=dataset.reference_submission,
    )


def load_sample(
    name: str, sample_size: int, seed: int, ricechem_question: int | None
) -> tuple[RubricDataset, str]:
    """Load a dataset and return (sampled_dataset, label).

    ricechem is 4 files with *different* rubrics, so a sample must come from ONE question:
    pick a random question (seeded) unless --ricechem-question forces one, then subsample.
    """
    if name == "ricechem":
        q = ricechem_question or random.Random(seed).choice([1, 2, 3, 4])
        ds = RubricDataset.from_file(DATA_DIR / "ricechem" / f"q{q}.json")
        return subsample_dataset(ds, sample_size, seed), f"q{q}"
    if name == "hashemi":
        ds = RubricDataset.from_file(DATA_DIR / "hashemi_etal_2024_dataset.json")
        return subsample_dataset(ds, sample_size, seed), ""
    if name == "charm-100":
        ds = RubricDataset.from_file(DATA_DIR / "charm100.json")
        return subsample_dataset(ds, sample_size, seed), ""
    raise ValueError(f"Unknown dataset: {name}")


# --- Grader construction ---------------------------------------------------------------


def build_grader(mode: str, think: bool, max_parallel: int, seed: int) -> CriterionGrader:
    """Build a single-judge or cross-provider ensemble grader.

    HP: temperature=1.0 (mandated / required by newer reasoning models), per-provider
    parallelism, SKIP abstain strategy. Thinking jobs enable "low" reasoning with more
    max_tokens headroom for the trace + structured output.
    """
    hp: dict[str, object] = {"temperature": 1.0, "max_parallel_requests": max_parallel}
    if think:
        hp["thinking"] = "low"
        hp["max_tokens"] = MAX_TOKENS_THINK
    else:
        hp["max_tokens"] = MAX_TOKENS_NO_THINK
    cac = CannotAssessConfig(strategy=CannotAssessStrategy.SKIP)
    if mode == "single":
        return CriterionGrader(
            llm_config=LLMConfig(model=SINGLE_MODEL, **hp),
            cannot_assess_config=cac,
            seed=seed,
        )
    judges = [
        JudgeSpec(llm_config=LLMConfig(model=SINGLE_MODEL, **hp), judge_id="gemini-flash-lite"),
        JudgeSpec(llm_config=LLMConfig(model=NANO_MODEL, **hp), judge_id="gpt-nano"),
    ]
    return CriterionGrader(judges=judges, cannot_assess_config=cac, seed=seed)


# --- Per-item tick seam (no progress callback exists in EvalRunner) --------------------


class _TickRunner(EvalRunner):
    """EvalRunner that fires a callback per completed item (for the live dashboard).

    Overriding the single per-item choke point keeps timing/error/report capture, real
    EvalResult aggregation, and metrics intact, while letting us drive our own display.
    """

    def __init__(self, dataset, grader, config=None, *, on_item=None, job=None):
        super().__init__(dataset, grader, config)
        self._on_item = on_item
        self._job = job

    async def _grade_item(self, idx: int, item) -> ItemResult:
        res = await super()._grade_item(idx, item)
        if self._on_item is not None:
            self._on_item(self._job, res)
        return res


async def run_job(
    job: Job,
    sample: RubricDataset,
    tmpdir: str,
    max_parallel: int,
    seed: int,
    on_item,
) -> JobResult:
    """Grade one job's sample and compute its metrics. Never raises (errors captured)."""
    try:
        grader = build_grader(job.mode, job.think, max_parallel, seed)
        cfg = EvalConfig(
            show_progress=False,  # we render our own dashboard
            resume=False,  # no checkpoint skipping -> always fresh calls (no cache)
            experiment_name=f"sanity-{job.dataset}-{job.mode}-think{int(job.think)}",
            experiments_dir=tmpdir,
        )
        runner = _TickRunner(sample, grader, cfg, on_item=on_item, job=job)
        result = await runner.run()
        metrics = result.compute_metrics(
            sample, per_judge=(job.mode == "ensemble"), na_mode="exclude"
        )
        return JobResult(job, result, metrics, None)
    except Exception as e:  # one bad job must not kill the pool
        logger.warning("Job %s failed: %s", job, e)
        return JobResult(job, None, None, str(e))


# --- Headline metric extraction --------------------------------------------------------


def headline(metrics) -> tuple[float | None, float | None, float | None, float | None]:
    """(accuracy, mean_kappa, mean Krippendorff alpha, mean Fleiss kappa) for a compact row.

    alpha/Fleiss are per-criterion (ensemble-only); we average the defined ones.
    """
    if metrics is None:
        return None, None, None, None
    alphas = [
        getattr(c, "krippendorff_alpha", None)
        for c in metrics.per_criterion
        if getattr(c, "krippendorff_alpha", None) is not None
    ]
    fleiss = [
        getattr(c, "fleiss_kappa", None)
        for c in metrics.per_criterion
        if getattr(c, "fleiss_kappa", None) is not None
    ]
    alpha = sum(alphas) / len(alphas) if alphas else None
    fl = sum(fleiss) / len(fleiss) if fleiss else None
    return metrics.criterion_accuracy, metrics.mean_kappa, alpha, fl


def think_tag(think: bool) -> str:
    return "on" if think else "off"


# --- Rich dashboard --------------------------------------------------------------------


class RichDashboard:
    """Single Live view: per-job progress bars + a growing results table + totals footer."""

    def __init__(self, jobs: list[Job], samples: dict[str, tuple[RubricDataset, str]], console):
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=22),
            MofNCompleteColumn(),
            TextColumn("{task.fields[note]}"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )
        self.task_ids: dict[Job, object] = {}
        self.started: set[object] = set()
        for job in jobs:
            total = len(samples[job.dataset][0].items)
            self.task_ids[job] = self.progress.add_task(
                self._label(job), total=total, note="", start=False
            )
        self.rows: list[JobResult] = []
        self.n_jobs = len(jobs)
        self.cost = 0.0
        self.tokens = 0
        self.items_done = 0
        self.errors = 0
        self.start = 0.0
        self.live = Live(self._render(), console=console, refresh_per_second=8)

    @staticmethod
    def _label(job: Job) -> str:
        return f"{job.dataset:<9} · {job.mode:<8} · think:{think_tag(job.think)}"

    def __enter__(self) -> "RichDashboard":
        self.start = time.perf_counter()
        self.live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self.live.__exit__(*exc)

    def tick(self, job: Job, res: ItemResult) -> None:
        task_id = self.task_ids[job]
        if task_id not in self.started:  # start the per-task timer on first real item
            self.progress.start_task(task_id)
            self.started.add(task_id)
        self.progress.advance(task_id)
        self.items_done += 1
        cost = getattr(res.report, "completion_cost", None)
        if cost:
            self.cost += cost
        usage = getattr(res.report, "token_usage", None)
        if usage:
            self.tokens += usage.total_tokens
        if res.error:
            self.errors += 1
        self.live.update(self._render())

    def job_done(self, jr: JobResult) -> None:
        self.rows.append(jr)
        note = "[red]ERR" if jr.error else "[green]done"
        self.progress.update(self.task_ids[jr.job], note=note)
        self.live.update(self._render())

    def _results_table(self) -> Panel:
        t = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        t.add_column("dataset", style="cyan", no_wrap=True)
        t.add_column("mode")
        t.add_column("think")
        t.add_column("n", justify="right")
        t.add_column("acc", justify="right")
        t.add_column("κ", justify="right")
        t.add_column("α", justify="right")
        t.add_column("fleiss", justify="right")
        t.add_column("cost$", justify="right")
        t.add_column("sec", justify="right")
        for jr in sorted(self.rows, key=_display_sort_key):
            job = jr.job
            if jr.error or jr.result is None:
                t.add_row(
                    job.dataset,
                    job.mode,
                    think_tag(job.think),
                    "—",
                    Text("ERR", style="bright_magenta"),
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                )
                continue
            acc, kappa, alpha, fl = headline(jr.metrics)
            t.add_row(
                job.dataset,
                job.mode,
                think_tag(job.think),
                str(jr.result.successful_items),
                _fmt_opt(acc, ".0%"),
                _fmt_opt(kappa, ".2f"),
                _fmt_opt(alpha, ".2f"),
                _fmt_opt(fl, ".2f"),
                _fmt_opt(jr.result.total_completion_cost, ".4f"),
                f"{jr.result.timing_stats.total_duration_seconds:.1f}",
            )
        return Panel(t, title="results (no-thinking first, then thinking)", border_style="cyan")

    def _footer(self) -> Panel:
        elapsed = time.perf_counter() - self.start
        err = f" · [red]errors {self.errors}[/red]" if self.errors else ""
        txt = (
            f"elapsed {elapsed:5.1f}s · jobs {len(self.rows)}/{self.n_jobs} · "
            f"items {self.items_done} · cost ${self.cost:.4f} · {self.tokens} tok{err}"
        )
        return Panel(Text.from_markup(txt), border_style="blue", title="totals (running)")

    def _render(self) -> Group:
        banner = Text(
            "AutoRubric sanity check — single pool, no-thinking prioritized, thinking backfills",
            style="bold",
        )
        return Group(banner, self.progress, self._results_table(), self._footer())


# --- Orchestration ---------------------------------------------------------------------


async def _launch_pool(jobs: list[Job], samples, tmpdir, max_parallel, seed, on_item):
    """Single concurrent pool: submit no-thinking jobs first (+yield) so they get FIFO
    semaphore priority, then thinking jobs backfill. Yields JobResults as they complete."""
    no_think = [j for j in jobs if not j.think]
    do_think = [j for j in jobs if j.think]
    tasks = [
        asyncio.create_task(run_job(j, samples[j.dataset][0], tmpdir, max_parallel, seed, on_item))
        for j in no_think
    ]
    await asyncio.sleep(0)  # let no-thinking items register on the per-provider semaphores first
    tasks += [
        asyncio.create_task(run_job(j, samples[j.dataset][0], tmpdir, max_parallel, seed, on_item))
        for j in do_think
    ]
    for coro in asyncio.as_completed(tasks):
        yield await coro


def print_final_reports(console: Console, rows: list[JobResult]) -> None:
    """Full metrics.summary() per job, grouped no-thinking then thinking."""
    for jr in sorted(rows, key=_display_sort_key):
        title = f"{jr.job.dataset} · {jr.job.mode} · think:{think_tag(jr.job.think)}"
        if jr.error or jr.metrics is None:
            console.print(
                Panel(Text(f"ERROR: {jr.error}", style="red"), title=title, border_style="red")
            )
        else:
            console.print(Panel(jr.metrics.summary(), title=title, border_style="cyan"))


def print_totals(console: Console, rows: list[JobResult], wall: float) -> None:
    """Grand totals + no-thinking vs thinking cost/time subtotals."""

    def subtotal(think: bool) -> tuple[float, int, int]:
        cost = sum(
            (jr.result.total_completion_cost or 0.0)
            for jr in rows
            if jr.job.think == think and jr.result is not None
        )
        toks = sum(
            (
                jr.result.total_token_usage.total_tokens
                if jr.result and jr.result.total_token_usage
                else 0
            )
            for jr in rows
            if jr.job.think == think
        )
        items = sum(jr.result.total_items for jr in rows if jr.job.think == think and jr.result)
        return cost, toks, items

    off_cost, off_tok, off_items = subtotal(False)
    on_cost, on_tok, on_items = subtotal(True)
    total_cost = off_cost + on_cost
    total_items = off_items + on_items
    n_err = sum(1 for jr in rows if jr.error)

    lines = [
        "=== TOTALS ===",
        f"wall clock      : {wall:.1f}s",
        f"jobs            : {len(rows)} ({n_err} errored)",
        f"items graded    : {total_items} ({total_items / wall:.1f} items/s)" if wall > 0 else "",
        f"total LLM cost  : ${total_cost:.4f}",
        f"  no-thinking   : ${off_cost:.4f}  ({off_tok} tok, {off_items} items)",
        f"  thinking      : ${on_cost:.4f}  ({on_tok} tok, {on_items} items)",
    ]
    console.print("\n".join(line for line in lines if line))


# --- CLI -------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Items sampled per dataset (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    p.add_argument(
        "--max-parallel-requests",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help=f"Per-provider concurrent request cap (default: {DEFAULT_MAX_PARALLEL}).",
    )
    p.add_argument(
        "--datasets",
        default=",".join(ALL_DATASETS),
        help="Comma-separated subset of: " + ", ".join(ALL_DATASETS) + " (default: all).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--single-only", action="store_true", help="Only the single judge.")
    mode.add_argument("--ensemble-only", action="store_true", help="Only the ensemble.")
    p.add_argument(
        "--thinking-mode",
        choices=["off", "on", "both"],
        default="both",
        help="Run no-thinking, thinking, or both (default: both, no-thinking first).",
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="Plain text output (no Rich TUI) — good for AI agents / piping.",
    )
    p.add_argument(
        "--ricechem-question",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Force a ricechem question (default: random, seeded).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seeds question pick, subsample, and option-shuffle (default: {DEFAULT_SEED}).",
    )
    return p.parse_args()


def check_api_keys(need_openai: bool) -> list[str]:
    missing = []
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")
    if need_openai and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    return missing


async def main() -> int:
    args = parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in ALL_DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Choose from {ALL_DATASETS}.", file=sys.stderr)
        return 2

    if args.single_only:
        modes = ["single"]
    elif args.ensemble_only:
        modes = ["ensemble"]
    else:
        modes = ["single", "ensemble"]

    thinks = {"off": [False], "on": [True], "both": [False, True]}[args.thinking_mode]

    missing = check_api_keys(need_openai="ensemble" in modes)
    if missing:
        print("Missing required API key(s): " + ", ".join(missing), file=sys.stderr)
        return 1

    use_rich = sys.stdout.isatty() and not args.plain

    # Quiet noisy LLM / library logging so it never corrupts the Live display.
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("autorubric").setLevel(logging.CRITICAL if use_rich else logging.ERROR)

    # Sample once per dataset (shared across modes + thinking conditions).
    samples = {
        name: load_sample(name, args.sample_size, args.seed, args.ricechem_question)
        for name in datasets
    }

    # Jobs, ordered no-thinking first; single/ensemble interleaved per dataset.
    jobs = [Job(name, mode, think) for think in thinks for name in datasets for mode in modes]

    console = Console()
    for name, (sample, label) in samples.items():
        tag = f" ({label})" if label else ""
        console.print(
            f"[dim]· {name}{tag}: {len(sample.items)} items, {sample.num_criteria} criteria[/dim]"
        )
    console.print(
        f"[dim]· {len(jobs)} jobs · seed={args.seed} · "
        f"max_parallel={args.max_parallel_requests} · thinking={args.thinking_mode}[/dim]\n"
    )

    wall_start = time.perf_counter()
    collected: list[JobResult] = []

    with tempfile.TemporaryDirectory(prefix="autorubric-sanity-") as tmpdir:
        if use_rich:
            with RichDashboard(jobs, samples, console) as dash:
                async for jr in _launch_pool(
                    jobs, samples, tmpdir, args.max_parallel_requests, args.seed, dash.tick
                ):
                    dash.job_done(jr)
                    collected.append(jr)
            wall = time.perf_counter() - wall_start
            print_final_reports(console, collected)
        else:
            async for jr in _launch_pool(
                jobs, samples, tmpdir, args.max_parallel_requests, args.seed, None
            ):
                collected.append(jr)
                acc, kappa, alpha, _ = headline(jr.metrics)
                cost = jr.result.total_completion_cost if jr.result else None
                status = "ERR" if jr.error else "ok"
                print(
                    f"[done] {jr.job.dataset}/{jr.job.mode}/think:{think_tag(jr.job.think)}  "
                    f"{status}  acc={_fmt_opt(acc, '.0%')} κ={_fmt_opt(kappa, '.2f')} "
                    f"α={_fmt_opt(alpha, '.2f')} cost=${_fmt_opt(cost, '.4f')}",
                    flush=True,
                )
            wall = time.perf_counter() - wall_start
            for jr in sorted(collected, key=_display_sort_key):
                title = f"{jr.job.dataset} · {jr.job.mode} · think:{think_tag(jr.job.think)}"
                print(f"\n=== {title} ===")
                print(f"ERROR: {jr.error}" if jr.metrics is None else jr.metrics.summary())

    print_totals(console, collected, wall)
    return 1 if any(jr.error for jr in collected) else 0


if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(asyncio.run(main()))
