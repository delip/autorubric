# Cookbook

Practical recipes for evaluating text outputs with AutoRubric. Each recipe solves a specific real-world scenario with focused code snippets and complete runnable examples.

AutoRubric grades with two kinds of AI judges. **LLM judges** take one criterion per call and explain each verdict. **Decision models**, such as TypeSafe's Jev, grade a whole rubric in one request and return probabilities instead of explanations. Both plug into the same `CriterionGrader`, and you can use either alone, mix them in an ensemble, or let a decision model take the first pass and hand its uncertain criteria to an LLM.

## Recipe Index

### Tier 1: Foundation

Start here if you're new to AutoRubric. [LLM Judges](llm-judges.md) and [Decision-Model Judges](decision-models.md) introduce the two kinds of judges, and the last recipe combines them.

| Recipe | Domain | What You'll Learn |
|--------|--------|-------------------|
| [Your First Evaluation](first-evaluation.md) | Tech Support | Basic rubric creation and grading |
| [LLM Judges](llm-judges.md) | — | Configuring, steering, and combining LLM judges; choosing a judge kind |
| [Decision-Model Judges](decision-models.md) | — | One-request judges with probabilities, mixed ensembles, and confidence cascades |
| [Managing Datasets](managing-datasets.md) | Medical Triage | Loading, saving, and splitting datasets |
| [Working with Explanations](explanations.md) | Essay Feedback | Accessing and formatting per-criterion reasons |
| [Cheap First-Pass Grading with Jev and an LLM Fallback](cascade-grading.md) | Short-Answer Grading | Calibrating and deploying a decision-model cascade |

### Tier 2: Reliability

Improve grading consistency and accuracy.

| Recipe | Domain | What You'll Learn |
|--------|--------|-------------------|
| [Few-Shot Calibration](few-shot-calibration.md) | Legal Contracts | Calibrating judges with labeled examples |
| [Ensemble Judging](ensemble-judging.md) | Job Applications | Multi-judge voting for high-stakes decisions |
| [Handling CANNOT_ASSESS](cannot-assess.md) | RAG Responses | Strategies for uncertain verdicts |
| [Fixing Seeds](fixing-seeds.md) | Product Reviews | Pinning shuffles and few-shot selection for reproducibility |

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
| [Held-Out Rubric Improvement](held-out-rubric-improvement.md) | Peer Review | Data-driven criterion refinement using grading errors |
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
| [Evaluating Agent Skills](agent-skill-evaluation.md) | Peer Review | Skill evaluation with with/without-skill comparison |
| [Improving Agent Skills](improving-agent-skills.md) | Peer Review | Automated skill refinement using rubric feedback |

## Quick Start

If you haven't installed AutoRubric yet:

```bash
pip install autorubric
```

Set up your API key for your preferred LLM provider:

```bash
export OPENAI_API_KEY=your_key_here
# or
export ANTHROPIC_API_KEY=your_key_here
# or
export GEMINI_API_KEY=your_key_here
```

For [decision-model judges](decision-models.md), install the TypeSafe extra and set its key:

```bash
pip install 'autorubric[typesafe]'
export TYPESAFE_API_KEY=your_key_here
```

Then jump into [Your First Evaluation](first-evaluation.md) to get started.

## Recipe Format

Each recipe follows a consistent structure:

1. **The Scenario** - A realistic problem you might face
2. **What You'll Learn** - Key features and concepts covered
3. **The Solution** - Step-by-step implementation with focused code snippets
4. **Key Takeaways** - Summary of important points
5. **Appendix: Complete Code** - Full runnable script you can copy-paste

The two judge chapters in Tier 1, [LLM Judges](llm-judges.md) and [Decision-Model Judges](decision-models.md), are guides rather than recipes: they explain how each kind of judge works and how to configure it.

## Prerequisites

All recipes assume:

- Python 3.11+
- AutoRubric installed (`pip install autorubric`)
- An API key for at least one supported LLM provider (recipes that use a decision model also need a TypeSafe API key)
- Basic familiarity with async/await (recipes use `asyncio.run()` for simplicity)
