"""Compare CriterionGrader(llm_calls="per_criterion") against llm_calls="per_item".

This script reruns the paid live experiment behind the "Grading a Whole Rubric in One
Call" cookbook recipe (docs/cookbook/single-call-grading.md): the same model, on the same
two datasets, graded with one call per criterion and with one call per item, two replicate
runs each, in the recipe's order (the first run of each mode, then the second). It compares
accuracy, agreement, cost and wall-clock time.

WARNING: this script makes real, billed API calls. With the constants below (gpt-6-luna,
REASONING="low", REPLICATES=2, 11 essay items x 5 criteria, 100 sampled RiceChem items x 8
criteria) it makes 1,932 calls, which cost about $0.31 in the recipe's run (with OpenAI's
automatic prompt-caching discount) and about $0.69 at list price, and it takes several
minutes. Set your provider's API key (e.g. OPENAI_API_KEY) before running it.

Each run is a resumable EvalRunner experiment. Its directory is named after MODEL, REASONING
and GRADER_SEED, and its name after the dataset's name, which carries the RiceChem sample's
size and seed; edits to the data files themselves are not tracked. An interrupted run resumes
without paying again for the items it graded, and a finished run is read back instead of
graded again. A resumed run never retries the items that failed, so the script stops at a run
with a failed item: fix the cause, delete the directory it names and run it again.

Usage:
    python examples/single_call_grading_comparison.py
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

from autorubric import (
    CriterionVerdict,
    EnsembleEvaluationReport,
    EvalResult,
    LLMCalls,
    LLMConfig,
    RubricDataset,
    compute_metrics,
    evaluate,
)
from autorubric.graders import CriterionGrader

load_dotenv()

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

# The recipe's numbers were measured with these values.
MODEL = "openai/gpt-6-luna"
REASONING = "low"  # LLMConfig(thinking=...): a reasoning_effort of "low"
REPLICATES = 2
RICECHEM_SAMPLE = 100
SAMPLE_SEED = 0
GRADER_SEED = 42
MAX_PARALLEL_REQUESTS = 16
MODES: tuple[LLMCalls, ...] = ("per_criterion", "per_item")

# One directory per grader configuration: EvalRunner resumes a same-named experiment whenever
# the dataset's name, prompt and size match, whatever model or seed graded it, so a change of
# MODEL, REASONING or GRADER_SEED must lead to different directories.
EXPERIMENTS_DIR = (
    Path("experiments")
    / "single_call_grading"
    / f"{MODEL.replace('/', '--')}__thinking-{REASONING}__seed{GRADER_SEED}"
)

CODE = {
    CriterionVerdict.MET: "MET",
    CriterionVerdict.UNMET: "UNMET",
    CriterionVerdict.CANNOT_ASSESS: "CA",
}


def load_datasets() -> dict[str, RubricDataset]:
    """Essay grading in full, plus a seeded random sample of RiceChem Q1."""
    essay = RubricDataset.from_file(DATA_DIR / "essay_grading_dataset.json")
    ricechem = RubricDataset.from_file(DATA_DIR / "ricechem" / "q1.json")
    idx = sorted(random.Random(SAMPLE_SEED).sample(range(len(ricechem.items)), RICECHEM_SAMPLE))
    sample = RubricDataset(
        prompt=ricechem.prompt,
        rubric=ricechem.rubric,
        items=[ricechem.items[i] for i in idx],
        # The name keys the sample's checkpoints, so it carries the sample's size and seed.
        name=f"ricechem-q1-sample{RICECHEM_SAMPLE}-seed{SAMPLE_SEED}",
    )
    return {"essay": essay, "ricechem_q1": sample}


def build_grader(mode: LLMCalls) -> CriterionGrader:
    return CriterionGrader(
        judge_model_config=LLMConfig(
            model=MODEL,
            thinking=REASONING,
            max_parallel_requests=MAX_PARALLEL_REQUESTS,
            cache_enabled=False,  # replicates must not read each other's answers
            # LiteLLM 1.95.0 recognizes OpenAI reasoning models by name (gpt-5 and the
            # o-series), so it refuses reasoning_effort for gpt-6 models unless allowed
            # explicitly; this entry is harmless on versions that already recognize gpt-6.
            extra_params={"allowed_openai_params": ["reasoning_effort"]},
        ),
        llm_calls=mode,
        seed=GRADER_SEED,
    )


def fmt(x: float | None, spec: str) -> str:
    """Format a value, or "n/a" when it is undefined (never a fabricated 0)."""
    return "n/a" if x is None else format(x, spec)


def verdict_grid(result: EvalResult, n_items: int, n_criteria: int) -> list[list[str | None]]:
    """Final verdict per (item, criterion), coded MET/UNMET/CA; None where ungraded."""
    grid: list[list[str | None]] = [[None] * n_criteria for _ in range(n_items)]
    for r in result.item_results:
        if r.error is not None or not isinstance(r.report, EnsembleEvaluationReport):
            continue
        if r.report.report is None:
            continue
        for c, cr in enumerate(r.report.report):
            if cr.error is None and cr.final_verdict is not None:
                grid[r.item_idx][c] = CODE[cr.final_verdict]
    return grid


def truth_grid(dataset: RubricDataset) -> list[list[str | None]]:
    def coded(gt: list[CriterionVerdict | str] | None) -> list[str | None]:
        if gt is None:
            return []
        return [CODE[CriterionVerdict(v)] if v is not None else None for v in gt]

    return [coded(item.ground_truth) for item in dataset.items]


def item_accuracy(grid: list[list[str | None]], truth: list[list[str | None]]) -> np.ndarray:
    """Per-item accuracy over the cells judged MET or UNMET and labelled MET or UNMET.

    As in compute_metrics' default (cannot_assess="exclude"), a cell whose verdict or label is
    CANNOT_ASSESS is left out, as is a cell that was not graded.
    """
    out = []
    for row, trow in zip(grid, truth):
        pairs = [
            (p, t)
            for p, t in zip(row, trow)
            if p is not None and t is not None and "CA" not in (p, t)
        ]
        out.append(np.mean([p == t for p, t in pairs]) if pairs else np.nan)
    return np.array(out)


def bootstrap_ci(diff: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% item-level bootstrap CI of the mean of `diff` (items with no value dropped)."""
    rng = np.random.default_rng(seed)
    d = diff[~np.isnan(diff)]
    stats = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def cell_agreement(a: list[list[str | None]], b: list[list[str | None]]) -> dict:
    pairs = [
        (x, y) for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x is not None and y is not None
    ]
    xs, ys = zip(*pairs)
    same = sum(x == y for x, y in pairs)
    kappa = cohen_kappa_score(xs, ys) if len(set(xs) | set(ys)) > 1 else None
    return {"cells": len(pairs), "agreement": same / len(pairs), "kappa": kappa}


def disagreement_direction(a: list[list[str | None]], b: list[list[str | None]]) -> dict[str, int]:
    """Count the disagreements between `a` and `b` over the cells both graded, by direction.

    "other" counts the disagreements in which either side is CANNOT_ASSESS.
    """
    counts = {"a_met_b_unmet": 0, "a_unmet_b_met": 0, "other": 0}
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            if x is None or y is None or x == y:
                continue
            if x == "MET" and y == "UNMET":
                counts["a_met_b_unmet"] += 1
            elif x == "UNMET" and y == "MET":
                counts["a_unmet_b_met"] += 1
            else:
                counts["other"] += 1
    return counts


async def run_one(
    dataset: RubricDataset, mode: LLMCalls, rep: int
) -> tuple[EvalResult, float | None]:
    """Grade one run; returns its result and wall-clock seconds (None when it was resumed)."""
    experiment = f"{dataset.name}__{mode}__rep{rep}"
    resumed = (EXPERIMENTS_DIR / experiment / "manifest.json").exists()
    start = time.perf_counter()
    result = await evaluate(
        dataset,
        build_grader(mode),
        show_progress=False,
        experiment_name=experiment,
        experiments_dir=EXPERIMENTS_DIR,
        resume=True,
    )
    wall = None if resumed else time.perf_counter() - start
    if result.failed_items:
        idx, error = result.errors[0]
        raise SystemExit(
            f"{experiment}: {result.failed_items} of {result.total_items} items failed "
            f"(item {idx}: {error}).\nA resumed run never retries its failed items: fix the "
            f"cause, delete {result.experiment_dir} and run the script again."
        )
    return result, wall


async def compare_dataset(name: str, dataset: RubricDataset) -> None:
    assert dataset.rubric is not None, "dataset must carry a global rubric"
    n_items, n_crit = len(dataset), len(dataset.rubric.rubric)
    print(f"\n{'=' * 78}\n{name} ({n_items} items x {n_crit} criteria)\n{'=' * 78}")
    truth = truth_grid(dataset)

    grids: dict[tuple[str, int], list[list[str | None]]] = {}
    print(
        f"{'run':<18} {'accuracy':>8} {'kappa':>6} {'MET':>9} {'prompt_tok':>11} "
        f"{'compl_tok':>10} {'cost':>9} {'wall_s':>7}"
    )
    # The recipe's order: the first run of each mode, then the second. Order matters for
    # recorded cost, since a provider's prompt cache may still hold an earlier run's prompts.
    for rep in range(1, REPLICATES + 1):
        for mode in MODES:
            result, wall = await run_one(dataset, mode, rep)
            metrics = compute_metrics(result, dataset)
            grid = verdict_grid(result, n_items, n_crit)
            grids[(mode, rep)] = grid
            flat = [v for row in grid for v in row if v is not None]
            usage = result.total_token_usage
            cost = result.total_completion_cost
            label = f"{mode}#{rep}"
            met = f"{flat.count('MET')}/{len(flat)}"
            cost_str = "n/a" if cost is None else f"${cost:.4f}"
            print(
                f"{label:<18} {fmt(metrics.criterion_accuracy, '.3f'):>8} "
                f"{fmt(metrics.mean_kappa, '.3f'):>6} {met:>9} "
                f"{fmt(usage.prompt_tokens if usage else None, ','):>11} "
                f"{fmt(usage.completion_tokens if usage else None, ','):>10} "
                f"{cost_str:>9} {fmt(wall, '.1f'):>7}"
            )
    print("(MET: cells judged MET out of cells judged; wall_s: n/a for a resumed run)")

    # Within-mode vs cross-mode verdict agreement (Cohen's kappa over graded cells).
    print("\nAgreement between runs:")
    for (m1, r1), (m2, r2) in itertools.combinations(grids, 2):
        agree = cell_agreement(grids[(m1, r1)], grids[(m2, r2)])
        kind = "within-mode" if m1 == m2 else "cross-mode"
        print(
            f"  {m1}#{r1} vs {m2}#{r2} ({kind}): agreement={agree['agreement']:.1%} "
            f"kappa={fmt(agree['kappa'], '.3f')} (n={agree['cells']})"
        )

    # Accuracy difference, per_item minus per_criterion: the mean over items of each item's
    # accuracy averaged over its mode's runs, the unit the bootstrap resamples. It weights items
    # equally, so it can differ from the difference of the pooled accuracy column when items
    # have different numbers of labelled cells.
    acc = {
        mode: np.nanmean(
            np.stack(
                [item_accuracy(grids[(mode, rep)], truth) for rep in range(1, REPLICATES + 1)]
            ),
            axis=0,
        )
        for mode in MODES
    }
    diff = acc["per_item"] - acc["per_criterion"]
    ci_lo, ci_hi = bootstrap_ci(diff)
    print(
        f"\nItem-level accuracy difference (per_item - per_criterion, items weighted equally): "
        f"{np.nanmean(diff):+.4f} [95% bootstrap CI {ci_lo:+.4f}, {ci_hi:+.4f}]"
    )

    # Direction of cross-mode disagreements, pairing runs with the same replicate index.
    totals = {"a_met_b_unmet": 0, "a_unmet_b_met": 0, "other": 0}
    for rep in range(1, REPLICATES + 1):
        d = disagreement_direction(grids[("per_criterion", rep)], grids[("per_item", rep)])
        for k in totals:
            totals[k] += d[k]
    print(
        "Cross-mode disagreements (paired by replicate): "
        f"per_criterion MET -> per_item UNMET: {totals['a_met_b_unmet']}, "
        f"per_criterion UNMET -> per_item MET: {totals['a_unmet_b_met']}, "
        f"involving CANNOT_ASSESS: {totals['other']}"
    )


async def main() -> None:
    for name, dataset in load_datasets().items():
        await compare_dataset(name, dataset)


if __name__ == "__main__":
    asyncio.run(main())
