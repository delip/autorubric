#!/usr/bin/env python3
"""Meta-rubric evaluation of the CSAP rubric.

Runs both standalone and in-context meta-rubric evaluations against
scify-phase2/csap.json using a Gemini judge, and writes HTML + JSON + Markdown
summaries to scify-phase2/reports/.
"""

import asyncio
import dataclasses
import json
import os
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

from dotenv import load_dotenv

from autorubric import LLMConfig, Rubric
from autorubric.meta import (
    evaluate_rubric_in_context,
    evaluate_rubric_standalone,
)

HERE = Path(__file__).parent
CSAP_PATH = HERE / "csap.json"
REPORTS_DIR = HERE / "reports"

TASK_PROMPT = (
    "You are asked to author a scientific feasibility CLAIM — a single, "
    "well-formed assertion about whether some scientific or technological "
    "capability is or will be achievable under specified conditions. The "
    "claim will be used as a benchmark input for an automated scientific "
    "feasibility-assessment system. A high-quality claim should be:\n"
    "  • Clear: different SMEs would interpret it similarly; single primary "
    "    capability assertion; no undefined vague terms.\n"
    "  • Specified: enough technical detail (target capability, system/method, "
    "    context, conditions, measurable success criteria, constraints) to "
    "    support decomposition into verifiable subclaims.\n"
    "  • Automatable: amenable to evidence-gathering, analysis, simulation, "
    "    or quantitative grounding by an automated system — not just "
    "    literature summarization or subjective expert judgment.\n"
    "  • Projective: requires nontrivial reasoning beyond existing consensus; "
    "    not trivially settled, not obviously impossible; involves "
    "    cross-domain or forward-in-time projection.\n"
    "Avoid bundling multiple claims, undefined vague terms (e.g., 'better', "
    "'robust', 'scalable'), policy/preference statements, and anything "
    "already settled by scientific consensus or obviously impossible."
)


def _summarize_report(report) -> dict:
    verdict_counts: Counter[str] = Counter()
    per_criterion = []
    if report.report is not None:
        for ecr in report.report:
            v = (
                ecr.final_verdict.value
                if hasattr(ecr.final_verdict, "value")
                else str(ecr.final_verdict)
            )
            verdict_counts[v] += 1
            per_criterion.append(
                {
                    "name": ecr.criterion.name,
                    "weight": ecr.criterion.weight,
                    "verdict": v,
                    "agreement": ecr.agreement,
                    "reason": ecr.final_reason,
                }
            )
    return {
        "score": report.score,
        "raw_score": report.raw_score,
        "mean_agreement": report.mean_agreement,
        "completion_cost_usd": report.completion_cost,
        "token_usage": (
            dataclasses.asdict(report.token_usage) if report.token_usage else None
        ),
        "verdict_counts": dict(verdict_counts),
        "per_criterion": per_criterion,
    }


async def main() -> None:
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set; populate .env first.")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csap = Rubric.from_file(str(CSAP_PATH))
    print(f"Loaded CSAP rubric: {len(csap.rubric)} criteria -> {[c.name for c in csap.rubric]}")

    judge = LLMConfig(
        model="gemini/gemini-3-flash-preview",
        temperature=0.0,
        thinking="medium",
        max_parallel_requests=10,
    )

    print("\n[1/2] Standalone meta-rubric evaluation...")
    standalone = await evaluate_rubric_standalone(
        csap,
        judge,
        display="html",
        output_html_path=REPORTS_DIR / "csap_standalone_report.html",
    )

    print("\n[2/2] In-context meta-rubric evaluation...")
    in_context = await evaluate_rubric_in_context(
        csap,
        TASK_PROMPT,
        judge,
        display="html",
        output_html_path=REPORTS_DIR / "csap_in_context_report.html",
    )

    summary = {
        "rubric": str(CSAP_PATH.name),
        "judge_model": judge.model,
        "standalone": _summarize_report(standalone),
        "in_context": _summarize_report(in_context),
    }
    (REPORTS_DIR / "csap_meta_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    md = [
        "# CSAP meta-rubric evaluation",
        "",
        f"Judge model: `{judge.model}`",
        f"Rubric: `{CSAP_PATH.name}` ({len(csap.rubric)} criteria)",
        "",
        "## Scores",
        "",
        "| Mode | Score | Raw | Cost (USD) |",
        "|---|---|---|---|",
        f"| Standalone | {standalone.score:.3f} | {standalone.raw_score:.2f} | "
        f"{standalone.completion_cost or 0:.4f} |",
        f"| In-context | {in_context.score:.3f} | {in_context.raw_score:.2f} | "
        f"{in_context.completion_cost or 0:.4f} |",
        "",
        "## Reports",
        "",
        "- [Standalone HTML report](csap_standalone_report.html)",
        "- [In-context HTML report](csap_in_context_report.html)",
        "- [Machine-readable summary](csap_meta_summary.json)",
    ]
    (REPORTS_DIR / "csap_meta_summary.md").write_text("\n".join(md) + "\n")

    print(
        f"\nStandalone score:  {standalone.score:.3f}  "
        f"(raw {standalone.raw_score:.2f}, cost ${standalone.completion_cost or 0:.4f})"
    )
    print(
        f"In-context score:  {in_context.score:.3f}  "
        f"(raw {in_context.raw_score:.2f}, cost ${in_context.completion_cost or 0:.4f})"
    )
    print(f"\nArtifacts written to {REPORTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
