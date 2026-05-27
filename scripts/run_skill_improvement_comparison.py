"""Run same-judge and cross-judge skill improvement side by side.

Saves per-paper scores for both conditions so we can compute error bars.
Both conditions share the same iteration-0 reviews and grading (controlled).
"""

import asyncio
import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from autorubric import LLMConfig
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader
from autorubric.llm import LLMClient
from autorubric.types import CriterionVerdict

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "examples" / "data" / "peer_review_skill_eval.json"
OUTPUT_PATH = (
    Path(__file__).parent.parent / "examples" / "data" / "skill_improvement_comparison.json"
)

V1_SKILL = "Provide brief feedback on the text below."

GOLD_SKILL = """\
# Scientific Peer Review

## Procedure

1. **Summarize** the paper in 2-3 sentences covering contribution, methodology, and findings.
2. **Evaluate methodology** --- assess study design, appropriateness for the research question, and specific limitations.
3. **Assess statistics** --- check appropriateness of tests, sample size justification, and effect sizes.
4. **List strengths** --- identify at least 2 specific strengths with references to the paper.
5. **List weaknesses** --- identify at least 2 specific weaknesses with actionable suggestions.
6. **Pose questions** --- ask 2-3 clarifying questions for the authors.
7. **Recommend** --- state Accept, Minor Revision, Major Revision, or Reject with justification.

## Formatting
- Use section headers for each step.
- Reference specific sections, figures, and quoted results.
- Keep under 800 words."""

REVISION_SYSTEM_PROMPT = """\
You are an expert skill designer for LLM agents. Your job is to revise an agent skill \
(a system prompt that guides the agent through a task) so that the agent's outputs \
better satisfy a rubric.

Principles for effective skills:
- Use imperative verbs ("Summarize", "Evaluate", "List") not hedging ("you should consider", "it can be helpful")
- Specify concrete outputs: counts ("at least 2"), formats ("section headers"), length limits ("under 800 words")
- Make requirements observable: instead of "be thorough", say "reference specific sections, figures, and quoted results"
- Include formatting constraints that make rubric criteria easy to verify
- Keep the skill concise --- a focused procedure outperforms a long essay about good practices

Output ONLY the revised skill text. Do not include any preamble, explanation, or commentary."""

REVISION_USER_TEMPLATE = """\
## Current Skill (Iteration {iteration})

{skill}

## Rubric Criteria and Current Pass Rates

{criteria_table}

## Sample Failure Explanations

{failure_examples}

## Iteration History

{history}

Revise the skill to improve pass rates on failing criteria while maintaining performance on passing criteria. \
Output only the revised skill text."""

N_SEEDS = 3


def extract_unique_papers(dataset: RubricDataset) -> list[dict]:
    seen = set()
    papers = []
    for item in dataset.items:
        paper_id = item.description.split("] ")[1] if "] " in item.description else item.description
        if paper_id not in seen:
            seen.add(paper_id)
            papers.append({"paper_id": paper_id, "prompt": item.prompt})
    return papers


async def generate_reviews(client, skill, papers):
    tasks = [
        client.generate(system_prompt=skill, user_prompt=p["prompt"], return_result=True)
        for p in papers
    ]
    results = await asyncio.gather(*tasks)
    return [
        {"paper_id": p["paper_id"], "review": r.content, "cost": r.cost or 0.0}
        for p, r in zip(papers, results)
    ]


async def grade_reviews(rubric, grader, reviews, papers):
    tasks = [
        rubric.grade(to_grade=r["review"], grader=grader, query=p["prompt"])
        for r, p in zip(reviews, papers)
    ]
    reports = await asyncio.gather(*tasks)
    graded = []
    for review, report in zip(reviews, reports):
        per_criterion = {}
        for cr in report.report:
            verdict = cr.final_verdict if hasattr(cr, "final_verdict") else cr.verdict
            reason = cr.final_reason if hasattr(cr, "final_reason") else cr.reason
            name = cr.criterion.name if hasattr(cr, "criterion") else cr.name
            per_criterion[name] = {"verdict": verdict.value, "reason": reason}
        graded.append(
            {
                "paper_id": review["paper_id"],
                "score": report.score,
                "per_criterion": per_criterion,
            }
        )
    return graded


def compute_pass_rates(graded, criteria_names):
    return {
        name: sum(
            1 for g in graded if g["per_criterion"][name]["verdict"] == CriterionVerdict.MET.value
        )
        / len(graded)
        for name in criteria_names
    }


def format_criteria_table(criteria, pass_rates):
    lines = []
    for c in criteria:
        rate = pass_rates.get(c.name, 0.0)
        status = "PASSING" if rate >= 0.7 else "FAILING"
        lines.append(
            f"- **{c.name}** (weight={c.weight}, pass_rate={rate:.0%}, {status}): {c.requirement}"
        )
    return "\n".join(lines)


def format_failure_examples(graded, criteria, pass_rates, max_examples=3):
    sections = []
    failing = [(c.name, pass_rates[c.name]) for c in criteria if pass_rates[c.name] < 0.7]
    failing.sort(key=lambda x: x[1])
    for name, rate in failing[:5]:
        examples = [
            f"  - Paper {g['paper_id']}: {g['per_criterion'][name]['reason']}"
            for g in graded
            if g["per_criterion"][name]["verdict"] == CriterionVerdict.UNMET.value
            and g["per_criterion"][name]["reason"]
        ][:max_examples]
        if examples:
            sections.append(f"**{name}** ({rate:.0%} pass rate):\n" + "\n".join(examples))
    return "\n\n".join(sections) if sections else "No failing criteria."


async def single_revision_run(
    label,
    rubric,
    papers,
    agent_client,
    eval_grader,
    revision_client,
):
    """Run iteration 0, revise once, run iteration 1. Return per-paper scores."""
    criteria_names = [c.name for c in rubric.rubric]

    # Iteration 0: generate + grade with vague skill
    print(f"\n  [{label}] Generating reviews (vague skill)...")
    reviews_0 = await generate_reviews(agent_client, V1_SKILL, papers)
    print(f"  [{label}] Grading reviews...")
    graded_0 = await grade_reviews(rubric, eval_grader, reviews_0, papers)
    scores_0 = [g["score"] for g in graded_0]
    pass_rates_0 = compute_pass_rates(graded_0, criteria_names)
    mean_0 = np.mean(scores_0)
    print(f"  [{label}] Iter 0 mean: {mean_0:.2f}")

    # Revise
    print(f"  [{label}] Revising skill...")
    criteria_table = format_criteria_table(rubric.rubric, pass_rates_0)
    failure_examples = format_failure_examples(graded_0, rubric.rubric, pass_rates_0)
    revision_prompt = REVISION_USER_TEMPLATE.format(
        iteration=0,
        skill=V1_SKILL,
        criteria_table=criteria_table,
        failure_examples=failure_examples,
        history="This is the first iteration.",
    )
    result = await revision_client.generate(
        system_prompt=REVISION_SYSTEM_PROMPT,
        user_prompt=revision_prompt,
        return_result=True,
    )
    revised_skill = result.content.strip()

    # Iteration 1: generate + grade with revised skill
    print(f"  [{label}] Generating reviews (revised skill)...")
    reviews_1 = await generate_reviews(agent_client, revised_skill, papers)
    print(f"  [{label}] Grading reviews...")
    graded_1 = await grade_reviews(rubric, eval_grader, reviews_1, papers)
    scores_1 = [g["score"] for g in graded_1]
    pass_rates_1 = compute_pass_rates(graded_1, criteria_names)
    mean_1 = np.mean(scores_1)
    print(f"  [{label}] Iter 1 mean: {mean_1:.2f}")

    # Gold skill
    print(f"  [{label}] Evaluating gold skill...")
    reviews_g = await generate_reviews(agent_client, GOLD_SKILL, papers)
    graded_g = await grade_reviews(rubric, eval_grader, reviews_g, papers)
    scores_g = [g["score"] for g in graded_g]
    mean_g = np.mean(scores_g)
    print(f"  [{label}] Gold mean: {mean_g:.2f}")

    return {
        "vague": {"scores": scores_0, "mean": float(mean_0), "pass_rates": pass_rates_0},
        "revised": {
            "scores": scores_1,
            "mean": float(mean_1),
            "pass_rates": pass_rates_1,
            "skill": revised_skill,
        },
        "gold": {"scores": scores_g, "mean": float(mean_g)},
    }


def bootstrap_ci(values, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


async def main():
    start_time = time.time()
    dataset = RubricDataset.from_file(DATA_PATH)
    rubric = dataset.rubric
    papers = extract_unique_papers(dataset)
    print(f"Loaded {len(papers)} papers, {len(rubric.rubric)} criteria")
    print(f"Running {N_SEEDS} seeds x 2 conditions")

    agent_config = LLMConfig(
        model="groq/llama-3.1-8b-instant", temperature=0.7, max_parallel_requests=5
    )
    eval_config = LLMConfig(
        model="gemini/gemini-3-flash-preview",
        temperature=1.0,
        thinking="medium",
        max_parallel_requests=10,
    )
    same_revision_config = LLMConfig(model="gemini/gemini-3-flash-preview", temperature=1.0)
    cross_revision_config = LLMConfig(model="openai/gpt-5.4", temperature=1.0)

    agent_client = LLMClient(agent_config)
    eval_grader = CriterionGrader(llm_config=eval_config, normalize=True)
    same_revision_client = LLMClient(same_revision_config)
    cross_revision_client = LLMClient(cross_revision_config)

    all_results = {"same_judge": [], "cross_judge": []}

    for seed in range(N_SEEDS):
        print(f"\n{'=' * 60}")
        print(f"SEED {seed}")
        print(f"{'=' * 60}")

        same_result = await single_revision_run(
            f"same-judge seed={seed}",
            rubric,
            papers,
            agent_client,
            eval_grader,
            same_revision_client,
        )
        cross_result = await single_revision_run(
            f"cross-judge seed={seed}",
            rubric,
            papers,
            agent_client,
            eval_grader,
            cross_revision_client,
        )
        all_results["same_judge"].append(same_result)
        all_results["cross_judge"].append(cross_result)

    # Aggregate across seeds
    print(f"\n{'=' * 60}")
    print("AGGREGATED RESULTS")
    print(f"{'=' * 60}")

    for condition in ["same_judge", "cross_judge"]:
        runs = all_results[condition]
        label = (
            "Same-judge (Gemini→Gemini)"
            if condition == "same_judge"
            else "Cross-judge (Gemini→GPT-5.4)"
        )
        for stage in ["vague", "revised", "gold"]:
            all_scores = []
            for run in runs:
                all_scores.extend(run[stage]["scores"])
            mean = np.mean(all_scores)
            ci_lo, ci_hi = bootstrap_ci(all_scores)
            per_seed_means = [np.mean(run[stage]["scores"]) for run in runs]
            print(
                f"  {label} {stage:>8}: {mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  "
                f"per-seed: {[f'{m:.2f}' for m in per_seed_means]}"
            )

    # Save
    output = {
        "n_seeds": N_SEEDS,
        "n_papers": len(papers),
        "config": {
            "agent_model": agent_config.model,
            "eval_model": eval_config.model,
            "same_revision_model": same_revision_config.model,
            "cross_revision_model": cross_revision_config.model,
        },
        "results": all_results,
        "elapsed_seconds": time.time() - start_time,
    }

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Elapsed: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
