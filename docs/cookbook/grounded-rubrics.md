# Grounded Rubrics

How to ensure your rubric criteria are grounded in verifiable evidence rather than requiring the judge to rely on its own knowledge.

## The Scenario

You have a rubric whose criteria use latent-quality language like "demonstrates understanding" or "contains accurate information." These criteria are problematic because an LLM judge cannot reliably verify factual claims against its own world knowledge -- it will produce inconsistent and often incorrect verdicts.

## What You'll Learn

- Why grounding matters for LLM-based evaluation
- How the metarubric detects ungrounded criteria
- How to rewrite criteria to be verifiable from the submission text alone

## Step 1: Identify Ungrounded Criteria

A criterion is **ungrounded** when it asks the judge to verify something that cannot be checked from the submission text or provided context alone. Common examples:

- "All dates mentioned are historically accurate"
- "The statistics cited are correct"
- "Demonstrates deep understanding of the topic"

These force the LLM judge to fall back on its own parametric knowledge, which is unreliable and inconsistent across runs.

## Step 2: Evaluate with the Meta-Rubric

The meta-rubric now includes two criteria that specifically target grounding issues:

- **`grounding_specified`** (positive, weight +8): Checks that criteria specify where factual claims should be grounded
- **`unverifiable_claim`** (negative, weight -8): Flags criteria that require verifying claims against external knowledge

```python
from autorubric import Rubric
from autorubric.meta import evaluate_rubric_standalone
from autorubric.llm import LLMConfig

rubric = Rubric.from_dict([
    {"weight": 10, "name": "accuracy", "requirement": "All factual claims are correct"},
    {"weight": 8, "name": "depth", "requirement": "Demonstrates deep understanding of the topic"},
])

llm = LLMConfig(model="anthropic/claude-sonnet-4-20250514")
report = await evaluate_rubric_standalone(rubric, llm)
```

The meta-rubric evaluation will flag `accuracy` as unverifiable (the judge has no reference source) and `depth` as ungrounded (relies on subjective inference).

## Step 3: Rewrite for Groundedness

Transform ungrounded criteria into verifiable ones by specifying what textual evidence constitutes satisfaction:

```python
grounded_rubric = Rubric.from_dict([
    {
        "weight": 10,
        "name": "supported_claims",
        "requirement": "Factual claims in the response are supported by citations to the provided reference material or are derivable from the context given in the prompt"
    },
    {
        "weight": 8,
        "name": "conceptual_connections",
        "requirement": "The response explicitly connects at least two concepts from the source material, explaining their relationship rather than listing them independently"
    },
])
```

Key principles:

- Point to observable textual evidence ("contains a citation," "explicitly connects")
- Name the source of truth ("provided reference material," "the prompt")
- Avoid latent qualities ("understanding," "insight," "accuracy" without a reference)

## Step 4: Automate with the Improvement Loop

You can also let the improvement loop fix grounding issues automatically:

```python
from autorubric.meta import improve_rubric

result = await improve_rubric(
    rubric,
    task_prompt="Summarize the key findings from the attached research paper.",
    eval_llm=llm,
    revision_llm=llm,
    mode="in_context",
)

# The improved rubric will have grounding issues addressed
improved = result.best_rubric
```
