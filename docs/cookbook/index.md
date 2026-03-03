# Cookbook

Practical recipes for evaluating text outputs with AutoRubric. Each recipe solves a specific real-world scenario with focused code snippets and complete runnable examples.

## Recipe Index

### Tier 1: Foundation

Start here if you're new to AutoRubric.

| Recipe | Domain | What You'll Learn |
|--------|--------|-------------------|
| [Your First Evaluation](first-evaluation.md) | Tech Support | Basic rubric creation and grading |
| [Managing Datasets](managing-datasets.md) | Medical Triage | Loading, saving, and splitting datasets |

| [Working with Explanations](explanations.md) | Essay Feedback | Accessing and formatting per-criterion reasons |

### Tier 2: Reliability

Improve grading consistency and accuracy.

| Recipe | Domain | What You'll Learn |
|--------|--------|-------------------|
| [Few-Shot Calibration](few-shot-calibration.md) | Legal Contracts | Calibrating judges with labeled examples |
| [Ensemble Judging](ensemble-judging.md) | Job Applications | Multi-judge voting for high-stakes decisions |
| [Handling CANNOT_ASSESS](cannot-assess.md) | RAG Responses | Strategies for uncertain verdicts |

### Tier 3: Advanced Evaluation

Sophisticated evaluation techniques.

| Recipe | Domain | What You'll Learn |
|--------|--------|-------------------|
| [Multi-Choice Rubrics](multi-choice-rubrics.md) | Restaurant Reviews | Ordinal/nominal scales with Likert ratings |
| [Extended Thinking](extended-thinking.md) | Security Assessments | Deep reasoning for complex evaluations |
| [Length Penalty](length-penalty.md) | Executive Summaries | Penalizing verbose responses |

### Tier 4: Validation & Production

Deploy with confidence.

| Recipe | Domain | What You'll Learn |
|--------|--------|-------------------|
| [Evaluating Rubric Quality](rubric-evaluation.md) | Peer Review | Meta-rubrics to validate and improve rubrics |
| [Automated Rubric Improvement](automated-rubric-improvement.md) | EV Analysis | LLM-driven iterative refinement of rubrics |
| [Judge Validation](judge-validation.md) | Content Moderation | Measuring agreement with human labels |
| [Synthetic Ground Truth](synthetic-ground-truth.md) | Product Descriptions | Bootstrapping labels from strong models |
| [Batch Evaluation](batch-evaluation.md) | Customer Feedback | Checkpointing, resumption, and cost tracking |

### Tier 5: Specialized

Advanced patterns for specific needs.

| Recipe | Domain | What You'll Learn |
|--------|--------|-------------------|
| [Per-Item Rubrics](per-item-rubrics.md) | Coding Interviews | Different rubrics for different items |
| [Cost Optimization](cost-optimization.md) | News Fact-Checking | Caching and model selection strategies |
| [Configuration Management](configuration-management.md) | Academic Papers | Sharing reproducible configs across teams |

## Quick Start

If you haven't installed AutoRubric yet:

```bash
pip install autorubric
```

Set up your API key for your preferred provider:

```bash
export OPENAI_API_KEY=your_key_here
# or
export ANTHROPIC_API_KEY=your_key_here
# or
export GEMINI_API_KEY=your_key_here
```

Then jump into [Your First Evaluation](first-evaluation.md) to get started.

## Recipe Format

Each recipe follows a consistent structure:

1. **The Scenario** - A realistic problem you might face
2. **What You'll Learn** - Key features and concepts covered
3. **The Solution** - Step-by-step implementation with focused code snippets
4. **Key Takeaways** - Summary of important points
5. **Appendix: Complete Code** - Full runnable script you can copy-paste

## Prerequisites

All recipes assume:

- Python 3.11+
- AutoRubric installed (`pip install autorubric`)
- An API key for at least one supported provider
- Basic familiarity with async/await (recipes use `asyncio.run()` for simplicity)
