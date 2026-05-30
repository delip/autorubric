"""Numerical parity check: autorubric ↔ simple-evals HealthBench grading.

REQUIRES A LOCAL CLONE OF simple-evals.
    This script reuses HealthBench grading internals (GRADER_TEMPLATE, RubricItem,
    calculate_score, parse_json_to_dict) plus the GeminiSampler grader from OpenAI's
    simple-evals (https://github.com/openai/simple-evals). Clone it at the repo root
    so the package lives at ``<repo-root>/simple-evals/`` (it is gitignored there):

        git clone https://github.com/openai/simple-evals.git

    The GeminiSampler (``sampler/gemini_sampler.py``) is a litellm-backed shim added
    on top of upstream simple-evals, so it must also be present in the clone's
    ``sampler/`` directory. Set the SIMPLE_EVALS_DIR env var to point elsewhere.

Grades the same N items (physician ideal completions) two ways:
1. simple-evals: HealthBenchEval.grade_sample() with GeminiSampler as grader.
2. autorubric:  Rubric.grade() with the same gemini model via litellm.

Both paths use gemini/gemini-3-flash-preview at temperature=0 for repeatability.
The submission is the same string (the physician's `ideal_completion`); the
rubric is the same list of (criterion, points) pairs. Only the grading
infrastructure differs.

Reports per-criterion verdict agreement and per-example score deltas, plus
the cross-correlation between simple-evals and autorubric overall scores.

Usage (paths are anchored to this file, so the working directory does not matter):
    cd health_bench
    uv run --with blobfile python healthbench_parity.py --num-examples 30 --seed 0
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# This file lives in health_bench/; the repo root is its parent.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# Load the Gemini key BEFORE importing anything that initializes litellm/openai.
load_dotenv(_REPO_ROOT / "autorubric-paper" / ".env")

# --------------------------------------------------------------------------- #
# Import HealthBench grading internals from a local simple-evals clone.
#
# simple-evals is a namespace package whose directory name ("simple-evals")
# contains a hyphen, so it cannot be loaded with a plain ``import`` statement and
# its modules use package-relative imports. We put its parent on sys.path and pull
# the submodules in via importlib, which resolves the hyphenated package name.
# --------------------------------------------------------------------------- #
_SIMPLE_EVALS_DIR = Path(os.environ.get("SIMPLE_EVALS_DIR", _REPO_ROOT / "simple-evals")).resolve()
if not (_SIMPLE_EVALS_DIR / "healthbench_eval.py").is_file():
    raise SystemExit(
        f"healthbench_parity.py needs a clone of openai/simple-evals at "
        f"{_SIMPLE_EVALS_DIR} (set SIMPLE_EVALS_DIR to override):\n"
        "  git clone https://github.com/openai/simple-evals.git\n"
        f"It also needs the GeminiSampler shim at "
        f"{_SIMPLE_EVALS_DIR / 'sampler' / 'gemini_sampler.py'}."
    )

_SE_PKG = _SIMPLE_EVALS_DIR.name  # "simple-evals"
if str(_SIMPLE_EVALS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SIMPLE_EVALS_DIR.parent))

common = importlib.import_module(f"{_SE_PKG}.common")
_healthbench_eval = importlib.import_module(f"{_SE_PKG}.healthbench_eval")
GRADER_TEMPLATE = _healthbench_eval.GRADER_TEMPLATE
RubricItem = _healthbench_eval.RubricItem
calculate_score = _healthbench_eval.calculate_score
parse_json_to_dict = _healthbench_eval.parse_json_to_dict
GeminiSampler = importlib.import_module(f"{_SE_PKG}.sampler.gemini_sampler").GeminiSampler


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_raw_main(path: Path) -> dict[str, dict]:
    """Index raw main JSONL by prompt_id."""
    out = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            out[row["prompt_id"]] = row
    return out


def pid_from_description(description: str) -> str:
    """Extract prompt_id from autorubric DataItem.description."""
    return description.split(" | ", 1)[0].split("=", 1)[1]


# --------------------------------------------------------------------------- #
# Grading paths
# --------------------------------------------------------------------------- #


def grade_via_simple_evals(
    grader,
    raw_row: dict,
    response_text: str,
    n_threads: int,
) -> tuple[float, list[dict]]:
    """Replicate HealthBenchEval.grade_sample inline (avoids its Azure download).

    Source: simple-evals/healthbench_eval.py:397-493.
    """
    rubric_items = [RubricItem.from_dict(d) for d in raw_row["rubrics"]]
    convo_with_response = list(raw_row["prompt"]) + [
        {"role": "assistant", "content": response_text}
    ]

    def grade_rubric_item(rubric_item: RubricItem) -> dict:
        convo_str = "\n\n".join([f"{m['role']}: {m['content']}" for m in convo_with_response])
        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
            "<<rubric_item>>", str(rubric_item)
        )
        messages = [{"role": "user", "content": grader_prompt}]
        while True:
            sampler_response = grader(messages)
            grading_response = sampler_response.response_text
            grading_response_dict = parse_json_to_dict(grading_response)
            if "criteria_met" in grading_response_dict:
                label = grading_response_dict["criteria_met"]
                if label is True or label is False:
                    return grading_response_dict
            print("simple-evals grading failed JSON parse, retrying...")

    grading_response_list = common.map_with_progress(
        grade_rubric_item, rubric_items, num_threads=n_threads, pbar=False
    )
    overall_score = calculate_score(rubric_items, grading_response_list)
    verdicts = [
        {
            "idx": i,
            "criterion": item.criterion,
            "points": item.points,
            "criteria_met": g["criteria_met"],
        }
        for i, (item, g) in enumerate(zip(rubric_items, grading_response_list, strict=True))
    ]
    return overall_score, verdicts


async def grade_via_autorubric(
    rubric,
    submission: str,
    prompt: str,
    grader,
):
    """Run autorubric Rubric.grade and return (score, unclipped_score, verdicts).

    ``unclipped_score`` reuses simple-evals' formula on autorubric's verdicts,
    so apples-to-apples comparison isolates verdict-level disagreement from
    autorubric's per-item [0,1] clipping (which simple-evals defers to the
    aggregate-mean stage).
    """
    report = await rubric.grade(
        to_grade=submission,
        grader=grader,
        query=prompt,
        reference_submission=None,  # parity: simple-evals never passes reference
    )
    verdicts = []
    if report.report is not None:
        for i, r in enumerate(report.report):
            verdict_str = r.final_verdict.value if r.final_verdict else "CANNOT_ASSESS"
            verdicts.append(
                {
                    "idx": i,
                    "name": r.criterion.name,
                    "criterion": r.criterion.requirement,
                    "points": r.criterion.weight,
                    "verdict": verdict_str,
                    "criteria_met": verdict_str == "MET",
                }
            )
    # Unclipped score: run autorubric's verdicts through simple-evals' formula
    rubric_items = [
        RubricItem(criterion=v["criterion"], points=v["points"], tags=[]) for v in verdicts
    ]
    grading_response_list = [{"criteria_met": v["criteria_met"]} for v in verdicts]
    unclipped = calculate_score(rubric_items, grading_response_list)
    return report.score, unclipped, verdicts


def recompute_overall_score(
    rubric_items: list[RubricItem], grading_response_list: list[dict]
) -> float:
    """simple-evals formula, applied to either set of verdicts."""
    return calculate_score(rubric_items, grading_response_list)


# --------------------------------------------------------------------------- #
# Main parity loop
# --------------------------------------------------------------------------- #


async def main_async(args: argparse.Namespace) -> None:
    # --- locate paths (this file lives in health_bench/) --------------------
    ar_ideal = _HERE / "autorubric_dataset" / "healthbench_physician_ideal.json"
    raw_main = _HERE / "raw_data" / "healthbench_main.jsonl"

    # --- load datasets ------------------------------------------------------
    from autorubric.dataset import RubricDataset

    ar_ds = RubricDataset.from_file(ar_ideal)
    print(f"Loaded autorubric dataset: {len(ar_ds.items)} items")
    raw_by_pid = load_raw_main(raw_main)
    print(f"Loaded raw main: {len(raw_by_pid)} prompts")

    # --- sample N items deterministically -----------------------------------
    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(len(ar_ds.items)), args.num_examples)
    print(f"Sampled {len(sample_indices)} item indices (seed={args.seed})")

    # --- set up graders -----------------------------------------------------
    se_grader = GeminiSampler(model=args.model, temperature=0.0, max_tokens=args.max_tokens)

    from autorubric.graders.criterion_grader import CriterionGrader
    from autorubric.llm import LLMConfig

    ar_llm = LLMConfig(
        model=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_parallel_requests=args.criteria_threads,
    )
    ar_grader = CriterionGrader(llm_config=ar_llm)

    # --- grade ---------------------------------------------------------------
    per_item: list[dict] = []
    for k, idx in enumerate(sample_indices, 1):
        item = ar_ds.items[idx]
        pid = pid_from_description(item.description)
        raw_row = raw_by_pid[pid]
        submission = item.submission  # physician ideal completion
        rubric = ar_ds.get_item_rubric(idx)
        n_crit = len(rubric.rubric)

        print(f"\n[{k}/{len(sample_indices)}] prompt_id={pid}  ({n_crit} criteria)")

        # simple-evals
        t0 = time.perf_counter()
        try:
            se_score, se_verdicts = grade_via_simple_evals(
                se_grader, raw_row, submission, args.criteria_threads
            )
            se_err = None
        except Exception as e:
            se_score, se_verdicts, se_err = None, [], str(e)
        se_t = time.perf_counter() - t0

        # autorubric
        t0 = time.perf_counter()
        try:
            ar_score, ar_unclipped, ar_verdicts = await grade_via_autorubric(
                rubric, submission, item.prompt, ar_grader
            )
            ar_err = None
        except Exception as e:
            ar_score, ar_unclipped, ar_verdicts, ar_err = None, None, [], str(e)
        ar_t = time.perf_counter() - t0

        # per-criterion agreement
        agreements = 0
        differences = []
        if se_err is None and ar_err is None and len(se_verdicts) == len(ar_verdicts):
            for sev, arv in zip(se_verdicts, ar_verdicts, strict=True):
                if sev["criteria_met"] == arv["criteria_met"]:
                    agreements += 1
                else:
                    differences.append(
                        {
                            "idx": sev["idx"],
                            "points": sev["points"],
                            "se": sev["criteria_met"],
                            "ar": arv["criteria_met"] if "criteria_met" in arv else arv["verdict"],
                            "criterion_head": sev["criterion"][:80],
                        }
                    )

        per_item.append(
            {
                "prompt_id": pid,
                "n_criteria": n_crit,
                "se_score": se_score,
                "ar_score": ar_score,  # autorubric native (clipped per-item)
                "ar_unclipped": ar_unclipped,  # autorubric verdicts via simple-evals formula
                "score_diff": (ar_score - se_score)
                if (se_score is not None and ar_score is not None)
                else None,
                "unclipped_diff": (ar_unclipped - se_score)
                if (se_score is not None and ar_unclipped is not None)
                else None,
                "agreements": agreements,
                "differences": differences,
                "se_err": se_err,
                "ar_err": ar_err,
                "se_time_s": round(se_t, 2),
                "ar_time_s": round(ar_t, 2),
            }
        )

        def fmt(x):
            return f"{x:+.3f}" if isinstance(x, float) else str(x)

        msg = (
            f"  se={fmt(se_score)} ar={fmt(ar_score)} "
            f"ar_unclip={fmt(ar_unclipped)} agree={agreements}/{n_crit} "
            f"({se_t:.1f}s/{ar_t:.1f}s)"
        )
        if differences:
            msg += f"  ({len(differences)} disagreements)"
        print(msg)

    # --- summary ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PARITY SUMMARY")
    print("=" * 70)

    total_crit = sum(p["n_criteria"] for p in per_item)
    total_agree = sum(p["agreements"] for p in per_item)
    pct = 100 * total_agree / total_crit if total_crit else 0.0
    print(f"Per-criterion verdict agreement: {total_agree}/{total_crit} = {pct:.1f}%")

    import statistics

    def summarize(label: str, diffs: list[float], a: list[float], b: list[float]) -> None:
        if not diffs:
            return
        print(f"\n{label}:")
        print(f"  mean  = {statistics.mean(diffs):+.4f}")
        print(f"  stdev = {statistics.stdev(diffs):.4f}")
        print(f"  min   = {min(diffs):+.4f}   max = {max(diffs):+.4f}")
        try:
            from scipy.stats import pearsonr

            r, pv = pearsonr(a, b)
            print(f"  Pearson r = {r:.4f} (p={pv:.2g})")
        except ImportError:
            pass

    rows = [p for p in per_item if p["score_diff"] is not None]
    summarize(
        "Score delta with autorubric NATIVE score (per-item clipped)",
        [p["score_diff"] for p in rows],
        [p["se_score"] for p in rows],
        [p["ar_score"] for p in rows],
    )
    rows = [p for p in per_item if p["unclipped_diff"] is not None]
    summarize(
        "Score delta with autorubric UNCLIPPED score (apples-to-apples)",
        [p["unclipped_diff"] for p in rows],
        [p["se_score"] for p in rows],
        [p["ar_unclipped"] for p in rows],
    )

    n_err = sum(1 for p in per_item if p["se_err"] or p["ar_err"])
    if n_err:
        print(f"Errors: {n_err}/{len(per_item)} items")

    # Save detailed report next to this script (health_bench/).
    out = _HERE / f"parity_report_n{args.num_examples}_seed{args.seed}.json"
    out.write_text(json.dumps(per_item, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed per-item report: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gemini/gemini-3-flash-preview")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--criteria-threads", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
