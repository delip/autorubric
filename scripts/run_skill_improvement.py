"""Run the skill improvement experiment.

Iteratively improves a vague peer review skill by grading agent outputs
against a rubric, analyzing per-criterion pass rates, and using a revision
LLM to rewrite the skill. Saves results to JSON for chart generation.
"""

import asyncio
import json
import time
from pathlib import Path

from autorubric import LLMConfig, Rubric
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader
from autorubric.llm import LLMClient
from autorubric.types import CriterionVerdict

DATA_PATH = Path(__file__).parent.parent / "examples" / "data" / "peer_review_skill_eval.json"
OUTPUT_PATH = Path(__file__).parent.parent / "examples" / "data" / "skill_improvement_results.json"

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

MAX_ITERATIONS = 5
CONVERGENCE_THRESHOLD = 0.02  # stop if score improves less than this


def extract_unique_papers(dataset: RubricDataset) -> list[dict]:
    """Extract one prompt per unique paper from the dataset."""
    seen = set()
    papers = []
    for item in dataset.items:
        paper_id = item.description.split("] ")[1] if "] " in item.description else item.description
        if paper_id not in seen:
            seen.add(paper_id)
            papers.append({"paper_id": paper_id, "prompt": item.prompt})
    return papers


async def generate_reviews(
    client: LLMClient,
    skill: str,
    papers: list[dict],
) -> list[dict]:
    """Generate a review for each paper using the given skill."""
    tasks = []
    for paper in papers:
        tasks.append(
            client.generate(
                system_prompt=skill,
                user_prompt=paper["prompt"],
                return_result=True,
            )
        )
    results = await asyncio.gather(*tasks)
    reviews = []
    for paper, result in zip(papers, results):
        reviews.append(
            {
                "paper_id": paper["paper_id"],
                "review": result.content,
                "cost": result.cost or 0.0,
            }
        )
    return reviews


async def grade_reviews(
    rubric: Rubric,
    grader: CriterionGrader,
    reviews: list[dict],
    papers: list[dict],
) -> list[dict]:
    """Grade each review against the rubric. Returns per-review grading data."""
    tasks = []
    for review, paper in zip(reviews, papers):
        tasks.append(
            rubric.grade(
                to_grade=review["review"],
                grader=grader,
                query=paper["prompt"],
            )
        )
    reports = await asyncio.gather(*tasks)
    graded = []
    for review, report in zip(reviews, reports):
        per_criterion = {}
        for cr in report.report:
            verdict = cr.final_verdict if hasattr(cr, "final_verdict") else cr.verdict
            reason = cr.final_reason if hasattr(cr, "final_reason") else cr.reason
            name = cr.criterion.name if hasattr(cr, "criterion") else cr.name
            per_criterion[name] = {
                "verdict": verdict.value,
                "reason": reason,
            }
        graded.append(
            {
                "paper_id": review["paper_id"],
                "score": report.score,
                "per_criterion": per_criterion,
                "cost": report.completion_cost or 0.0,
            }
        )
    return graded


def compute_pass_rates(graded: list[dict], criteria_names: list[str]) -> dict[str, float]:
    """Compute per-criterion pass rate from grading results."""
    rates = {}
    for name in criteria_names:
        met = sum(
            1 for g in graded if g["per_criterion"][name]["verdict"] == CriterionVerdict.MET.value
        )
        rates[name] = met / len(graded)
    return rates


def format_criteria_table(criteria: list, pass_rates: dict[str, float]) -> str:
    """Format rubric criteria with pass rates for the revision prompt."""
    lines = []
    for c in criteria:
        rate = pass_rates.get(c.name, 0.0)
        status = "PASSING" if rate >= 0.7 else "FAILING"
        lines.append(
            f"- **{c.name}** (weight={c.weight}, pass_rate={rate:.0%}, {status}): {c.requirement}"
        )
    return "\n".join(lines)


def format_failure_examples(
    graded: list[dict], criteria: list, pass_rates: dict[str, float], max_examples: int = 3
) -> str:
    """Format sample failure explanations for criteria with low pass rates."""
    sections = []
    failing = [(c.name, pass_rates[c.name]) for c in criteria if pass_rates[c.name] < 0.7]
    failing.sort(key=lambda x: x[1])

    for name, rate in failing[:5]:
        examples = []
        for g in graded:
            cr = g["per_criterion"][name]
            if cr["verdict"] == CriterionVerdict.UNMET.value and cr["reason"]:
                examples.append(f"  - Paper {g['paper_id']}: {cr['reason']}")
                if len(examples) >= max_examples:
                    break
        if examples:
            sections.append(f"**{name}** ({rate:.0%} pass rate):\n" + "\n".join(examples))

    return "\n\n".join(sections) if sections else "No failing criteria."


def format_history(iterations: list[dict]) -> str:
    """Format iteration history for the revision prompt."""
    if not iterations:
        return "This is the first iteration."
    lines = []
    for it in iterations:
        lines.append(f"- Iteration {it['iteration']}: mean_score={it['mean_score']:.2f}")
    return "\n".join(lines)


async def run_improvement_loop(
    rubric: Rubric,
    papers: list[dict],
    agent_client: LLMClient,
    eval_grader: CriterionGrader,
    revision_client: LLMClient,
) -> list[dict]:
    """Run the iterative skill improvement loop."""
    criteria_names = [c.name for c in rubric.rubric]
    current_skill = V1_SKILL
    iterations = []
    total_cost = 0.0

    for i in range(MAX_ITERATIONS):
        print(f"\n{'=' * 60}")
        print(f"Iteration {i}")
        print(f"{'=' * 60}")

        # Generate reviews
        print("  Generating reviews with current skill...")
        reviews = await generate_reviews(agent_client, current_skill, papers)
        gen_cost = sum(r["cost"] for r in reviews)
        total_cost += gen_cost

        # Grade reviews
        print("  Grading reviews...")
        graded = await grade_reviews(rubric, eval_grader, reviews, papers)
        grade_cost = sum(g["cost"] for g in graded)
        total_cost += grade_cost

        # Compute pass rates and mean score
        pass_rates = compute_pass_rates(graded, criteria_names)
        mean_score = sum(g["score"] for g in graded) / len(graded)

        print(f"  Mean score: {mean_score:.2f}")
        for name, rate in pass_rates.items():
            status = "OK" if rate >= 0.7 else "LOW"
            print(f"    {name}: {rate:.0%} [{status}]")

        iteration_data = {
            "iteration": i,
            "skill": current_skill,
            "mean_score": mean_score,
            "pass_rates": pass_rates,
            "sample_reviews": [
                {"paper_id": r["paper_id"], "review": r["review"][:500]} for r in reviews[:3]
            ],
            "generation_cost": gen_cost,
            "grading_cost": grade_cost,
        }
        iterations.append(iteration_data)

        # Check convergence
        if i > 0:
            prev_score = iterations[i - 1]["mean_score"]
            if mean_score - prev_score < CONVERGENCE_THRESHOLD:
                print(
                    f"  Score plateau (delta={mean_score - prev_score:.3f} < {CONVERGENCE_THRESHOLD}). Stopping."
                )
                break

        if i == MAX_ITERATIONS - 1:
            print("  Max iterations reached.")
            break

        # Revise skill
        print("  Revising skill...")
        criteria_table = format_criteria_table(rubric.rubric, pass_rates)
        failure_examples = format_failure_examples(graded, rubric.rubric, pass_rates)
        history = format_history(iterations)

        revision_prompt = REVISION_USER_TEMPLATE.format(
            iteration=i,
            skill=current_skill,
            criteria_table=criteria_table,
            failure_examples=failure_examples,
            history=history,
        )

        result = await revision_client.generate(
            system_prompt=REVISION_SYSTEM_PROMPT,
            user_prompt=revision_prompt,
            return_result=True,
        )
        current_skill = result.content.strip()
        revision_cost = result.cost or 0.0
        total_cost += revision_cost
        print(f"  Revision cost: ${revision_cost:.4f}")
        print(f"  New skill length: {len(current_skill)} chars")

    return iterations


async def evaluate_with_skill(
    skill: str,
    label: str,
    rubric: Rubric,
    papers: list[dict],
    agent_client: LLMClient,
    eval_grader: CriterionGrader,
) -> dict:
    """Generate and grade reviews for a given skill. Returns summary."""
    print(f"\nEvaluating {label}...")
    reviews = await generate_reviews(agent_client, skill, papers)
    graded = await grade_reviews(rubric, eval_grader, reviews, papers)
    criteria_names = [c.name for c in rubric.rubric]
    pass_rates = compute_pass_rates(graded, criteria_names)
    mean_score = sum(g["score"] for g in graded) / len(graded)
    cost = sum(r["cost"] for r in reviews) + sum(g["cost"] for g in graded)
    print(f"  {label} mean score: {mean_score:.2f}")
    return {
        "skill": skill,
        "mean_score": mean_score,
        "pass_rates": pass_rates,
        "cost": cost,
    }


async def main():
    start_time = time.time()

    # Load dataset
    dataset = RubricDataset.from_file(DATA_PATH)
    rubric = dataset.rubric
    papers = extract_unique_papers(dataset)
    print(f"Loaded {len(papers)} papers, {len(rubric.rubric)} criteria")

    # LLM configs
    agent_config = LLMConfig(
        model="groq/llama-3.1-8b-instant",
        temperature=0.7,
        max_parallel_requests=5,
    )
    eval_config = LLMConfig(
        model="gemini/gemini-3-flash-preview",
        temperature=1.0,
        thinking="medium",
        max_parallel_requests=10,
    )
    revision_config = LLMConfig(
        model="gemini/gemini-3-flash-preview",
        temperature=1.0,
    )

    agent_client = LLMClient(agent_config)
    revision_client = LLMClient(revision_config)
    eval_grader = CriterionGrader(llm_config=eval_config, normalize=True)

    # Run improvement loop
    iterations = await run_improvement_loop(
        rubric,
        papers,
        agent_client,
        eval_grader,
        revision_client,
    )

    # Determine convergence reason
    if len(iterations) >= MAX_ITERATIONS:
        convergence_reason = "max_iterations"
    elif len(iterations) > 1:
        last_delta = iterations[-1]["mean_score"] - iterations[-2]["mean_score"]
        convergence_reason = f"score_plateau (delta={last_delta:.3f})"
    else:
        convergence_reason = "single_iteration"

    # Evaluate curated skill for comparison
    gold_result = await evaluate_with_skill(
        GOLD_SKILL,
        "Curated Skill",
        rubric,
        papers,
        agent_client,
        eval_grader,
    )

    # Compute total cost
    total_cost = (
        sum(it.get("generation_cost", 0) + it.get("grading_cost", 0) for it in iterations)
        + gold_result["cost"]
    )

    elapsed = time.time() - start_time

    # Build output
    output = {
        "v1_skill": V1_SKILL,
        "gold_skill": GOLD_SKILL,
        "iterations": iterations,
        "gold_comparison": {
            "mean_score": gold_result["mean_score"],
            "pass_rates": gold_result["pass_rates"],
        },
        "convergence_reason": convergence_reason,
        "total_cost": total_cost,
        "elapsed_seconds": elapsed,
        "config": {
            "agent_model": agent_config.model,
            "eval_model": eval_config.model,
            "revision_model": revision_config.model,
            "max_iterations": MAX_ITERATIONS,
            "num_papers": len(papers),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Total cost: ${total_cost:.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"V1 score:       {iterations[0]['mean_score']:.2f}")
    print(f"Improved score: {iterations[-1]['mean_score']:.2f}")
    print(f"Gold score:     {gold_result['mean_score']:.2f}")
    print(f"Convergence:    {convergence_reason}")


if __name__ == "__main__":
    asyncio.run(main())
