# Behavioral Improvement Loop

Use empirical behavioral signals to drive rubric refinement, so the improvement loop fixes criteria that are noisy in practice---not just structurally flawed.

## The Scenario

You are running the automated rubric improvement loop, and it converges on a rubric that scores perfectly on the meta-rubric. But when you deploy it, certain criteria still produce inconsistent verdicts across runs. The structural meta-rubric analysis says the rubric is clean, yet the grading behavior tells a different story.

You want the improvement loop itself to consume behavioral signals---like per-criterion verdict variance---so it can target criteria that look fine on paper but fail empirically. AutoRubric supports this through `evidence_fn`, `behavioral_signal_frequency`, and the `behavioral_plateau_converged` convergence function.

## What You'll Learn

- How to define an evidence function that computes behavioral signals
- How to configure `ImprovementConfig` with `evidence_fn` and `behavioral_signal_frequency`
- How to use `behavioral_plateau_converged` as a convergence function
- How to inspect artifacts showing evidence across iterations

## The Solution

### Step 1: Define an Evidence Function

An evidence function is an async callable that takes a `Rubric` and returns a dict. The most common evidence is per-criterion reward variance from `compute_reward_variance`:

```python
from autorubric import LLMConfig, Rubric
from autorubric.meta import compute_reward_variance

probe_llm = LLMConfig(model="openai/gpt-4.1-mini", temperature=0.5)

# Representative submissions for variance measurement
probe_items = [
    "Hi Sarah, I understand how frustrating this must be. To fix the login "
    "issue, please clear your browser cache and try again. Let me know if "
    "that works!",

    "Your account has been unlocked. Try logging in now.",

    "Hello! I'm sorry to hear about the trouble. The error you're seeing is "
    "caused by an expired session token. I've reset it on our end. Please log "
    "out, wait 30 seconds, and log back in. If the issue persists, reply here "
    "and I'll escalate to our engineering team within 24 hours.",
]


async def evidence_fn(rubric: Rubric) -> dict:
    """Compute per-criterion verdict variance as behavioral evidence."""
    variance = await compute_reward_variance(
        rubric,
        probe_items,
        llm_config=probe_llm,
        n_samples=5,
        seed=42,
    )
    return {"variance": variance}
```

The returned dict is stored in `IterationResult.evidence` and passed to both the meta-evaluation and the revision prompt. The revision LLM sees a "Behavioral Signals" section containing the raw numbers, so it can propose targeted edits to high-variance criteria.

!!! tip "Keep probe items small"
    Each call to `evidence_fn` grades every probe item `n_samples` times. With 3 items and
    `n_samples=5`, that is 15 grading calls per invocation. Use 3--5 representative items
    to keep costs manageable.

### Step 2: Configure the Improvement Loop

Pass `evidence_fn` and `behavioral_signal_frequency` to `improve_rubric()` or `ImprovementConfig`:

```python
import asyncio
from autorubric.meta import improve_rubric

initial_rubric = Rubric.from_dict([
    {
        "name": "greeting",
        "weight": 5,
        "requirement": "Response begins with a greeting that addresses the customer by name"
    },
    {
        "name": "empathy",
        "weight": 8,
        "requirement": "Response demonstrates empathy and understanding"
    },
    {
        "name": "solution",
        "weight": 10,
        "requirement": "Response provides a specific, actionable solution to the customer's problem"
    },
    {
        "name": "tone",
        "weight": 6,
        "requirement": "Response is professional"
    },
    {
        "name": "follow_up",
        "weight": 7,
        "requirement": "Response includes a clear next step or follow-up action"
    },
])

task_prompt = (
    "Evaluate the quality of a customer support response to a technical "
    "issue reported by a paying customer."
)

eval_llm = LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)
revision_llm = LLMConfig(model="openai/gpt-4.1", temperature=0.3)

async def main():
    result = await improve_rubric(
        initial_rubric,
        task_prompt,
        eval_llm=eval_llm,
        revision_llm=revision_llm,
        evidence_fn=evidence_fn,
        behavioral_signal_frequency="every_iter",
        save_artifacts=True,
        artifacts_dir="behavioral_improvement_run",
    )

    print(f"Converged: {result.convergence_reason}")
    print(f"Iterations: {len(result.iterations)}")
    print(f"Cost: ${result.total_completion_cost:.4f}")

    # Show variance reduction across iterations
    for it in result.iterations:
        if it.evidence and "variance" in it.evidence:
            mean_var = sum(it.evidence["variance"].values()) / len(it.evidence["variance"])
            print(f"  Iter {it.iteration}: quality={it.quality_score:.0%}, "
                  f"mean_variance={mean_var:.4f}")

asyncio.run(main())
```

The `behavioral_signal_frequency` parameter controls how often `evidence_fn` is called:

| Value | When `evidence_fn` runs | Use case |
|-------|-------------------------|----------|
| `"every_iter"` | Every iteration | Full visibility; higher cost |
| `"first_and_last"` | First and final iterations only | Compare before/after; default |
| `"on_demand"` | Never called automatically | Caller passes evidence manually |

### Step 3: Use `behavioral_plateau_converged` as Convergence Function

The built-in convergence logic checks quality score plateaus. The `behavioral_plateau_converged` function extends this to also monitor evidence stability---it waits until both the quality score and the evidence variance have stabilized:

```python
from functools import partial
from autorubric.meta import (
    ImprovementConfig,
    ImprovementRunner,
    behavioral_plateau_converged,
)

config = ImprovementConfig(
    eval_llm=eval_llm,
    revision_llm=revision_llm,
    evidence_fn=evidence_fn,
    behavioral_signal_frequency="every_iter",
    convergence_fn=partial(
        behavioral_plateau_converged,
        patience=2,
        quality_threshold=0.02,
        variance_threshold=0.01,
    ),
    save_artifacts=True,
    artifacts_dir="behavioral_improvement_run",
)

async def main():
    runner = ImprovementRunner(initial_rubric, task_prompt, config=config)
    result = await runner.run()

    print(f"Converged: {result.convergence_reason}")
    # Will print "behavioral_plateau" when both quality and variance stabilize

asyncio.run(main())
```

`behavioral_plateau_converged` returns `"behavioral_plateau"` when the quality score has changed by less than `quality_threshold` and the mean evidence variance has changed by less than `variance_threshold` for `patience` consecutive iterations. If evidence is not present on the iteration results, it falls back to quality-only plateau detection.

!!! warning "Set `behavioral_signal_frequency` to `\"every_iter\"`"
    `behavioral_plateau_converged` needs evidence on every iteration to detect variance
    stability. If you use `"first_and_last"`, only the first and last iterations will have
    evidence, and the convergence function will fall back to quality-only plateau detection.

### Step 4: Inspect Artifacts

When `save_artifacts=True`, each iteration's JSON artifact includes the evidence dict. You can analyze variance trends programmatically:

```python
import json

with open("behavioral_improvement_run/summary.json", encoding="utf-8") as f:
    summary = json.load(f)

for it_summary in summary["iterations_summary"]:
    idx = it_summary["iteration"]

    # Load the detailed iteration artifact
    with open(f"behavioral_improvement_run/iter-{idx:02d}.json", encoding="utf-8") as f:
        detail = json.load(f)

    if "evidence" in detail:
        variance = detail["evidence"].get("variance", {})
        noisy = {k: v for k, v in variance.items() if v > 0.05}
        print(f"Iter {idx}: {len(noisy)} noisy criteria")
        for name, v in sorted(noisy.items(), key=lambda x: -x[1]):
            print(f"    {name}: {v:.4f}")
    else:
        print(f"Iter {idx}: no evidence collected")
```

## Key Takeaways

- **`evidence_fn` bridges empirical measurement and rubric improvement**: the loop uses behavioral data to guide revisions, not just structural text analysis
- **`behavioral_signal_frequency` controls cost**: `"every_iter"` gives full visibility but costs more; `"first_and_last"` is a cheap before/after comparison
- **`behavioral_plateau_converged` stops the loop when both quality and variance stabilize**: this prevents wasting iterations when the rubric is no longer improving on either axis
- **Evidence is persisted in artifacts**: each iteration's `iter-{NN}.json` includes the evidence dict for post-hoc analysis

## Going Further

- [Behavioral Signals](behavioral-signals.md) -- Understanding `compute_reward_variance` in depth
- [Automated Rubric Improvement](automated-rubric-improvement.md) -- The core improvement loop
- [Evaluating Rubric Quality](rubric-evaluation.md) -- Meta-rubric evaluation without the loop

---

## Appendix: Complete Code

```python
"""Behavioral Improvement Loop -- Variance-Informed Rubric Refinement"""

import asyncio
import json
from functools import partial

from autorubric import LLMConfig, Rubric
from autorubric.meta import (
    ImprovementConfig,
    ImprovementRunner,
    behavioral_plateau_converged,
    compute_reward_variance,
    improve_rubric,
)


TASK_PROMPT = (
    "Evaluate the quality of a customer support response to a technical "
    "issue reported by a paying customer."
)

PROBE_ITEMS = [
    "Hi Sarah, I understand how frustrating this must be. To fix the login "
    "issue, please clear your browser cache and try again. Let me know if "
    "that works!",

    "Your account has been unlocked. Try logging in now.",

    "Hello! I'm sorry to hear about the trouble. The error you're seeing is "
    "caused by an expired session token. I've reset it on our end. Please log "
    "out, wait 30 seconds, and log back in. If the issue persists, reply here "
    "and I'll escalate to our engineering team within 24 hours.",
]


def create_initial_rubric() -> Rubric:
    """Create a rubric with a mix of clear and vague criteria."""
    return Rubric.from_dict([
        {
            "name": "greeting",
            "weight": 5,
            "requirement": "Response begins with a greeting that addresses the customer by name"
        },
        {
            "name": "empathy",
            "weight": 8,
            "requirement": "Response demonstrates empathy and understanding"
        },
        {
            "name": "solution",
            "weight": 10,
            "requirement": "Response provides a specific, actionable solution to the customer's problem"
        },
        {
            "name": "tone",
            "weight": 6,
            "requirement": "Response is professional"
        },
        {
            "name": "follow_up",
            "weight": 7,
            "requirement": "Response includes a clear next step or follow-up action"
        },
    ])


async def main():
    probe_llm = LLMConfig(model="openai/gpt-4.1-mini", temperature=0.5)
    eval_llm = LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)
    revision_llm = LLMConfig(model="openai/gpt-4.1", temperature=0.3)

    initial_rubric = create_initial_rubric()

    async def evidence_fn(rubric: Rubric) -> dict:
        variance = await compute_reward_variance(
            rubric,
            PROBE_ITEMS,
            llm_config=probe_llm,
            n_samples=5,
            seed=42,
        )
        return {"variance": variance}

    # Option A: Simple API
    print("=" * 60)
    print("RUNNING BEHAVIORAL IMPROVEMENT LOOP")
    print("=" * 60)

    result = await improve_rubric(
        initial_rubric,
        TASK_PROMPT,
        eval_llm=eval_llm,
        revision_llm=revision_llm,
        evidence_fn=evidence_fn,
        behavioral_signal_frequency="every_iter",
        save_artifacts=True,
        artifacts_dir="behavioral_improvement_run",
    )

    print(f"\nConverged: {result.convergence_reason}")
    print(f"Iterations: {len(result.iterations)}")
    print(f"Cost: ${result.total_completion_cost:.4f}")

    for it in result.iterations:
        if it.evidence and "variance" in it.evidence:
            mean_var = sum(it.evidence["variance"].values()) / len(
                it.evidence["variance"]
            )
            print(
                f"  Iter {it.iteration}: quality={it.quality_score:.0%}, "
                f"mean_variance={mean_var:.4f}"
            )

    # Option B: Full control with behavioral convergence
    print()
    print("=" * 60)
    print("RUNNING WITH BEHAVIORAL CONVERGENCE")
    print("=" * 60)

    config = ImprovementConfig(
        eval_llm=eval_llm,
        revision_llm=revision_llm,
        evidence_fn=evidence_fn,
        behavioral_signal_frequency="every_iter",
        convergence_fn=partial(
            behavioral_plateau_converged,
            patience=2,
            quality_threshold=0.02,
            variance_threshold=0.01,
        ),
        save_artifacts=True,
        artifacts_dir="behavioral_convergence_run",
    )

    runner = ImprovementRunner(create_initial_rubric(), TASK_PROMPT, config=config)
    result = await runner.run()

    print(f"\nConverged: {result.convergence_reason}")
    print(f"Best iteration: {result.best_iteration}")

    print("\nFinal rubric criteria:")
    for c in result.best_rubric.rubric:
        print(f"  [{c.weight:+}] {c.name}: {c.requirement}")


if __name__ == "__main__":
    asyncio.run(main())
```
