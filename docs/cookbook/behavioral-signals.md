# Behavioral Signals

Measure which rubric criteria produce noisy, inconsistent judgments so you can fix them before deployment.

## The Scenario

You have a rubric for evaluating customer support responses, but some criteria produce wildly different verdicts across repeated evaluations of the same submission. A criterion like "demonstrates empathy" might flip between MET and UNMET depending on the run, while "includes a greeting" is rock-solid. You need a way to identify these unreliable criteria so you can rewrite them---or feed that information into automated rubric improvement.

AutoRubric's `compute_reward_variance` function measures per-criterion verdict variance by grading the same submissions multiple times and computing how much each criterion's binary score fluctuates.

## What You'll Learn

- How to use `compute_reward_variance` to measure criterion reliability
- How to interpret variance results (what counts as "noisy")
- How to pass behavioral evidence to meta-evaluation for richer feedback
- How this connects to the automated improvement loop

## The Solution

### Step 1: Define a Rubric with a Mix of Clear and Vague Criteria

Start with a rubric that has both precise criteria (low expected variance) and vague criteria (high expected variance):

```python
from autorubric import Rubric

rubric = Rubric.from_dict([
    {
        "name": "greeting",
        "weight": 5,
        "requirement": "Response begins with a greeting that addresses the customer by name"
    },
    {
        "name": "solution_provided",
        "weight": 10,
        "requirement": "Response provides a specific, actionable solution to the customer's problem"
    },
    {
        "name": "empathy",
        "weight": 8,
        "requirement": "Response demonstrates empathy and understanding"
    },
    {
        "name": "professional_tone",
        "weight": 6,
        "requirement": "Response maintains a professional and helpful tone throughout"
    },
    {
        "name": "follow_up",
        "weight": 7,
        "requirement": "Response includes a clear next step or follow-up action for the customer"
    },
])
```

The first two criteria are specific and observable. The last three rely on subjective interpretation---exactly the kind of criteria that tend to produce inconsistent judgments.

### Step 2: Compute Reward Variance

Call `compute_reward_variance` with a few representative submissions. The function grades each submission multiple times with different seeds and forced temperature > 0, then computes the variance of the binary verdict (1.0 for MET, 0.0 for UNMET) across all runs:

```python
import asyncio
from autorubric import LLMConfig
from autorubric.meta import compute_reward_variance

llm_config = LLMConfig(model="openai/gpt-4.1-mini", temperature=0.5)

# Representative submissions that exercise different criteria
submissions = [
    "Hi Sarah, I understand how frustrating this must be. To fix the login issue, "
    "please clear your browser cache and try again. Let me know if that works!",

    "Your account has been unlocked. Try logging in now.",

    "Hello! I'm sorry to hear about the trouble. The error you're seeing is caused "
    "by an expired session token. I've reset it on our end. Please log out, wait "
    "30 seconds, and log back in. If the issue persists, reply here and I'll "
    "escalate to our engineering team within 24 hours.",
]

async def main():
    variance = await compute_reward_variance(
        rubric,
        submissions,
        llm_config=llm_config,
        n_samples=5,
        seed=42,
    )

    print("Per-criterion variance:")
    for name, v in sorted(variance.items(), key=lambda x: -x[1]):
        label = "NOISY" if v > 0.05 else "STABLE"
        print(f"  {name:<25} {v:.4f}  [{label}]")

    return variance

variance = asyncio.run(main())
```

### Step 3: Interpret the Results

The variance for each criterion ranges from 0.0 (perfectly consistent---always the same verdict) to 0.25 (maximum variance for a binary variable---equal split between MET and UNMET).

| Variance Range | Interpretation | Action |
|----------------|----------------|--------|
| 0.00--0.02 | Highly consistent | No action needed |
| 0.02--0.05 | Minor noise | Monitor; may be acceptable |
| 0.05--0.15 | Significant noise | Rewrite the criterion for clarity |
| 0.15--0.25 | Unreliable | Criterion is too vague to use as-is |

In the example above, you would expect `greeting` and `follow_up` to have low variance (they check for specific textual features), while `empathy` and `professional_tone` are likely to show higher variance due to their subjective nature.

!!! tip "Choosing probe submissions"
    Pick submissions that span the quality spectrum for your task. Include at least one
    clearly good response, one clearly poor one, and one or two borderline cases. Borderline
    cases are where unreliable criteria are most likely to flip.

### Step 4: Pass Evidence to Meta-Evaluation

You can pass the variance data as behavioral evidence to `evaluate_rubric_standalone` or `evaluate_rubric_in_context`. The meta-judge receives these signals alongside the rubric text, allowing it to cite empirical data in its assessment:

```python
from autorubric.meta import evaluate_rubric_in_context

task_prompt = (
    "Evaluate the quality of a customer support response to a technical "
    "issue reported by a paying customer."
)

async def main():
    result = await evaluate_rubric_in_context(
        rubric,
        task_prompt,
        llm_config,
        evidence={"variance": variance},
        display="stdout",
    )
    print(f"\nMeta-rubric score: {result.score:.2f}")

asyncio.run(main())
```

When evidence is provided, the meta-judge can reference variance numbers in its reasoning---for example, noting that "empathy" has a variance of 0.12 and recommending more specific behavioral indicators.

### Step 5: Connect to the Improvement Loop

Rather than manually fixing criteria, you can feed behavioral signals directly into the automated improvement loop. The [Behavioral Improvement Loop](behavioral-improvement-loop.md) recipe shows how to define an `evidence_fn` that calls `compute_reward_variance` and wire it into `improve_rubric()`. The improvement loop then uses variance data to guide revisions---criteria with high variance get targeted for rewording.

```python
from autorubric.meta import improve_rubric

async def evidence_fn(rubric):
    variance = await compute_reward_variance(
        rubric, submissions, llm_config=llm_config, n_samples=5, seed=42
    )
    return {"variance": variance}

result = await improve_rubric(
    rubric,
    task_prompt,
    eval_llm=llm_config,
    revision_llm=LLMConfig(model="openai/gpt-4.1", temperature=0.3),
    evidence_fn=evidence_fn,
)
```

See [Behavioral Improvement Loop](behavioral-improvement-loop.md) for the full setup.

## Key Takeaways

- **`compute_reward_variance` measures criterion reliability** by grading the same submissions multiple times and computing verdict variance
- **Variance above 0.05 signals an unreliable criterion** that needs clearer, more specific wording
- **Passing evidence to meta-evaluation** lets the meta-judge cite empirical data in its assessment, not just analyze the text
- **Behavioral signals complement structural analysis**: meta-rubrics catch anti-patterns in the text; variance catches criteria that look fine on paper but fail in practice
- **Use representative submissions** that span the quality spectrum for your task, including borderline cases

## Going Further

- [Behavioral Improvement Loop](behavioral-improvement-loop.md) -- Automating variance-informed rubric refinement
- [Automated Rubric Improvement](automated-rubric-improvement.md) -- The core improvement loop without behavioral signals
- [Evaluating Rubric Quality](rubric-evaluation.md) -- Understanding meta-rubric evaluation in depth
- [Ensemble Judging](ensemble-judging.md) -- Using multiple judges to improve reliability

---

## Appendix: Complete Code

```python
"""Behavioral Signals -- Measuring Criterion Reliability"""

import asyncio
from autorubric import LLMConfig, Rubric
from autorubric.meta import compute_reward_variance, evaluate_rubric_in_context


TASK_PROMPT = (
    "Evaluate the quality of a customer support response to a technical "
    "issue reported by a paying customer."
)


def create_rubric() -> Rubric:
    """Create a rubric with a mix of clear and vague criteria."""
    return Rubric.from_dict([
        {
            "name": "greeting",
            "weight": 5,
            "requirement": "Response begins with a greeting that addresses the customer by name"
        },
        {
            "name": "solution_provided",
            "weight": 10,
            "requirement": "Response provides a specific, actionable solution to the customer's problem"
        },
        {
            "name": "empathy",
            "weight": 8,
            "requirement": "Response demonstrates empathy and understanding"
        },
        {
            "name": "professional_tone",
            "weight": 6,
            "requirement": "Response maintains a professional and helpful tone throughout"
        },
        {
            "name": "follow_up",
            "weight": 7,
            "requirement": "Response includes a clear next step or follow-up action for the customer"
        },
    ])


SUBMISSIONS = [
    "Hi Sarah, I understand how frustrating this must be. To fix the login issue, "
    "please clear your browser cache and try again. Let me know if that works!",

    "Your account has been unlocked. Try logging in now.",

    "Hello! I'm sorry to hear about the trouble. The error you're seeing is caused "
    "by an expired session token. I've reset it on our end. Please log out, wait "
    "30 seconds, and log back in. If the issue persists, reply here and I'll "
    "escalate to our engineering team within 24 hours.",
]


async def main():
    llm_config = LLMConfig(model="openai/gpt-4.1-mini", temperature=0.5)
    rubric = create_rubric()

    # Step 1: Compute per-criterion variance
    print("=" * 60)
    print("COMPUTING REWARD VARIANCE")
    print("=" * 60)

    variance = await compute_reward_variance(
        rubric,
        SUBMISSIONS,
        llm_config=llm_config,
        n_samples=5,
        seed=42,
    )

    print("\nPer-criterion variance:")
    for name, v in sorted(variance.items(), key=lambda x: -x[1]):
        label = "NOISY" if v > 0.05 else "STABLE"
        print(f"  {name:<25} {v:.4f}  [{label}]")

    # Step 2: Pass evidence to meta-evaluation
    print()
    print("=" * 60)
    print("META-EVALUATION WITH BEHAVIORAL EVIDENCE")
    print("=" * 60)

    result = await evaluate_rubric_in_context(
        rubric,
        TASK_PROMPT,
        llm_config,
        evidence={"variance": variance},
        display="stdout",
    )

    print(f"\nMeta-rubric score: {result.score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
```
