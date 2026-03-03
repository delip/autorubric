# Cost Optimization Strategies

Minimize evaluation costs while maintaining accuracy.

## The Scenario

You're fact-checking news articles at scale. Each article requires evaluating multiple claims, and costs add up quickly. You need strategies to reduce expenses: caching repeated evaluations, choosing cost-effective models, and using prompt caching.

## What You'll Learn

- Enabling response caching with `cache_enabled`
- Monitoring cache efficiency with `cache_stats()`
- Using prompt caching for system prompt efficiency
- Comparing models for cost vs accuracy trade-offs
- Batching strategies to reduce overhead

## The Solution

### Step 1: Enable Response Caching

Cache LLM responses to avoid redundant API calls:

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader

grader = CriterionGrader(
    llm_config=LLMConfig(
        model="openai/gpt-4.1-mini",
        cache_enabled=True,           # Enable disk caching
        cache_dir=".autorubric_cache", # Cache location
        cache_ttl=86400,              # 24 hours (None = no expiration)
    )
)
```

!!! tip "When Caching Helps"
    Caching is most valuable when:

    - Re-running evaluations during development
    - Evaluating the same content with different rubrics
    - Running regression tests on known content
    - Processing duplicate items in your dataset

### Step 2: Monitor Cache Performance

Check cache statistics after evaluation:

```python
# After running evaluations
client = grader._clients["default"]  # Get the LLM client
stats = client.cache_stats()

print(f"Cache Statistics:")
print(f"  Entries: {stats['count']}")
print(f"  Size: {stats['size'] / 1024 / 1024:.1f} MB")
print(f"  Directory: {stats['directory']}")
```

### Step 3: Re-run with Warm Cache

The second run uses cached responses:

```python
import asyncio
import time

async def benchmark_cache():
    # First run - populates cache
    start = time.perf_counter()
    result1 = await evaluate(dataset, grader, show_progress=False)
    cold_time = time.perf_counter() - start
    cold_cost = result1.total_completion_cost or 0

    # Second run - uses cache
    start = time.perf_counter()
    result2 = await evaluate(dataset, grader, show_progress=False)
    warm_time = time.perf_counter() - start
    warm_cost = result2.total_completion_cost or 0

    print(f"Cold run: {cold_time:.1f}s, ${cold_cost:.4f}")
    print(f"Warm run: {warm_time:.1f}s, ${warm_cost:.4f}")
    print(f"Speedup: {cold_time / warm_time:.1f}x")
    print(f"Cost savings: ${cold_cost - warm_cost:.4f}")
```

### Step 4: Clear Cache When Needed

```python
# Clear all cached responses
client = grader._clients["default"]
cleared = client.clear_cache()
print(f"Cleared {cleared} cached entries")
```

### Step 5: Enable Prompt Caching (Anthropic)

For Anthropic models, prompt caching reduces costs for repeated system prompts:

```python
grader = CriterionGrader(
    llm_config=LLMConfig(
        model="anthropic/claude-sonnet-4-5-20250929",
        prompt_caching=True,  # Enable Anthropic prompt caching (default)
    )
)

# Evaluate - system prompt cached after first call
result = await evaluate(dataset, grader)

# Check cache efficiency in token usage
if result.total_token_usage:
    usage = result.total_token_usage
    if usage.cache_read_input_tokens:
        cache_rate = usage.cache_read_input_tokens / usage.prompt_tokens * 100
        print(f"Prompt cache hit rate: {cache_rate:.1f}%")
        # Cached tokens are 90% cheaper on Anthropic!
```

!!! info "Prompt Caching by Provider"
    - **Anthropic**: Requires explicit cache_control (AutoRubric handles this)
    - **OpenAI/DeepSeek**: Automatic for prompts ≥1024 tokens
    - **Gemini**: Supported on 2.5+ models

### Step 6: Compare Model Cost vs Accuracy

Evaluate the same dataset with different models:

```python
models = [
    ("openai/gpt-4.1", "GPT-4 Turbo"),
    ("openai/gpt-4.1-mini", "GPT-4 Mini"),
    ("anthropic/claude-haiku-3-5-20241022", "Claude Haiku"),
    ("gemini/gemini-2.0-flash", "Gemini Flash"),
]

results = []
for model_id, name in models:
    grader = CriterionGrader(
        llm_config=LLMConfig(model=model_id, temperature=0.0)
    )

    result = await evaluate(dataset, grader, show_progress=False)
    metrics = result.compute_metrics(dataset)  # If ground truth available

    results.append({
        "model": name,
        "accuracy": metrics.criterion_accuracy,
        "cost": result.total_completion_cost or 0,
        "time": result.timing_stats.total_duration_seconds,
    })

# Compare
print(f"{'Model':<20} {'Accuracy':>10} {'Cost':>10} {'Time':>10}")
print("-" * 52)
for r in sorted(results, key=lambda x: x["cost"]):
    print(f"{r['model']:<20} {r['accuracy']:>9.1%} ${r['cost']:>9.4f} {r['time']:>9.1f}s")
```

Sample output:

```
Model                  Accuracy       Cost       Time
----------------------------------------------------
Gemini Flash              87.5%    $0.0012       4.2s
Claude Haiku              89.2%    $0.0018       5.1s
GPT-4 Mini                91.3%    $0.0034       6.8s
GPT-4 Turbo               94.1%    $0.0156      12.3s
```

### Step 7: Cost-Effective Production Strategy

Combine strategies for optimal cost:

```python
# Production configuration
grader = CriterionGrader(
    llm_config=LLMConfig(
        # Cost-effective model
        model="openai/gpt-4.1-mini",

        # Response caching for reruns
        cache_enabled=True,
        cache_dir=".eval_cache",
        cache_ttl=None,  # No expiration

        # Prompt caching (auto for OpenAI)
        prompt_caching=True,

        # Rate limiting to avoid throttling
        max_parallel_requests=20,

        # Low temperature for consistency
        temperature=0.0,
    )
)

# Batch evaluation with checkpointing
config = EvalConfig(
    experiment_name="production-eval",
    resume=True,  # Don't re-evaluate on restart
    max_concurrent_items=50,
)
```

### Step 8: Cost Monitoring Dashboard

Track costs over time:

```python
from pathlib import Path
from autorubric.eval import EvalResult
import json

def summarize_experiment_costs(experiments_dir: Path):
    """Summarize costs across all experiments."""

    total_cost = 0.0
    experiments = []

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        result = EvalResult.from_experiment(exp_dir)
        cost = result.total_completion_cost or 0

        experiments.append({
            "name": manifest["experiment_name"],
            "items": manifest["total_items"],
            "cost": cost,
        })
        total_cost += cost

    print(f"\nExperiment Cost Summary:")
    print("-" * 50)
    for exp in sorted(experiments, key=lambda x: x["cost"], reverse=True):
        print(f"{exp['name']:<30} {exp['items']:>6} items  ${exp['cost']:>8.4f}")

    print("-" * 50)
    print(f"{'TOTAL':<30} {'':<13} ${total_cost:>8.4f}")

# Usage
summarize_experiment_costs(Path("./experiments"))
```

## Key Takeaways

- **Response caching** (`cache_enabled=True`) avoids redundant API calls
- **Prompt caching** reduces costs for repeated system prompts
- **Model comparison** reveals cost vs accuracy trade-offs
- **Smaller models** (GPT-4-mini, Haiku, Flash) often suffice for simpler tasks
- **Checkpointing** (`resume=True`) prevents re-evaluating on restart
- **Monitor costs** across experiments to identify optimization opportunities

## Going Further

- [Batch Evaluation](batch-evaluation.md) - Checkpointing for long runs
- [Extended Thinking](extended-thinking.md) - Trade cost for accuracy
- [API Reference: LLM](../api/llm.md) - Caching configuration details

---

## Appendix: Complete Code

```python
"""Cost Optimization - News Fact-Checking Evaluation"""

import asyncio
import time
from pathlib import Path
from autorubric import (
    Rubric, RubricDataset, LLMConfig, EvalConfig, evaluate
)
from autorubric.graders import CriterionGrader


def create_factcheck_dataset() -> RubricDataset:
    """Create a fact-checking dataset."""

    rubric = Rubric.from_dict([
        {
            "name": "factual_accuracy",
            "weight": 15.0,
            "requirement": "Article's main claims are factually accurate"
        },
        {
            "name": "source_attribution",
            "weight": 10.0,
            "requirement": "Claims are properly attributed to credible sources"
        },
        {
            "name": "balanced_reporting",
            "weight": 8.0,
            "requirement": "Presents multiple perspectives fairly"
        },
        {
            "name": "misleading_framing",
            "weight": -12.0,
            "requirement": "Uses misleading headlines or framing"
        }
    ])

    dataset = RubricDataset(
        prompt="Fact-check this news article for accuracy and balance.",
        rubric=rubric,
        name="news-factcheck-v1"
    )

    articles = [
        {
            "submission": """
New Study Finds Coffee May Reduce Heart Disease Risk

A peer-reviewed study published in the New England Journal of Medicine
found that moderate coffee consumption (3-4 cups daily) was associated
with a 15% lower risk of cardiovascular disease. The study followed
500,000 participants over 10 years.

"The data suggests a protective effect, though we can't establish
causation," said lead researcher Dr. Sarah Chen from Johns Hopkins.

The American Heart Association noted the findings while emphasizing
that individual responses vary and those with certain conditions
should consult their doctors.
""",
            "description": "Well-sourced health article"
        },
        {
            "submission": """
SHOCKING: Scientists Discover Coffee Cures Heart Disease!

Doctors are STUNNED by new research proving coffee is the miracle
cure for heart problems. Start drinking more coffee TODAY to
eliminate your risk of heart attacks!

Studies show coffee drinkers NEVER get heart disease. Big Pharma
doesn't want you to know this simple trick!
""",
            "description": "Misleading sensationalized article"
        },
        {
            "submission": """
Tech Company Reports Strong Q3 Earnings

XYZ Corp reported Q3 revenue of $4.2 billion, up 12% year-over-year,
according to their SEC filing released Tuesday. The company cited
strong enterprise software sales as the primary driver.

"We're pleased with the results," CEO John Smith said in the earnings
call. Analyst consensus had expected $4.0 billion.

Some analysts expressed concern about slowing consumer segment growth,
which declined 3% from Q2.
""",
            "description": "Balanced business reporting"
        },
        {
            "submission": """
Local Community Raises $50,000 for Family After House Fire

Residents of Maple Street have raised over $50,000 for the Johnson
family after their home was destroyed by fire last week. The fire
department confirmed the cause was electrical and no injuries occurred.

"The community support has been overwhelming," said Sarah Johnson.
A GoFundMe page verified by the platform shows donations from 847 people.

The Red Cross is also providing temporary housing assistance.
""",
            "description": "Human interest with verification"
        },
        {
            "submission": """
Climate Change: Both Sides of the Debate

While 97% of climate scientists agree human activity is causing global
warming according to NASA, some politicians question the consensus.
Senator X called for "more research before economic action."

Environmental groups point to record temperatures and extreme weather
as evidence of urgent need for policy changes. Industry representatives
argue for gradual transition to protect jobs.

The IPCC's latest report projects significant impacts without emissions
reductions.
""",
            "description": "Climate article with proper context"
        },
        {
            "submission": """
Everyone Agrees: New Tax Plan is Perfect!

The proposed tax plan has received unanimous praise from all experts.
Everyone agrees this is exactly what the economy needs.

Critics have been completely silent because there's nothing to criticize.
This is the best economic policy in history according to sources.
""",
            "description": "One-sided without attribution"
        },
        {
            "submission": """
New Smartphone Released with Improved Battery

TechBrand announced their latest smartphone today, featuring a
4500mAh battery, improved from last year's 4000mAh model.

Independent testing by Consumer Reports showed 18% longer screen-on
time compared to the predecessor. The company claims "all-day battery
life" in typical usage scenarios.

Pricing starts at $799, with availability beginning next month.
Competitors Samsung and Apple offer similar specifications at
comparable price points.
""",
            "description": "Straightforward tech product coverage"
        },
        {
            "submission": """
EXPOSED: The TRUTH They Don't Want You to Know!

Insider sources reveal the shocking conspiracy behind [major event].
This changes EVERYTHING you thought you knew!

"They" have been hiding this for years but brave truth-seekers
have finally uncovered the real story. Share before it gets taken down!

Anonymous experts confirm what we've always suspected.
""",
            "description": "Conspiracy framing without sources"
        }
    ]

    for article in articles:
        dataset.add_item(**article)

    return dataset


async def compare_models(dataset: RubricDataset):
    """Compare cost and accuracy across models."""

    models = [
        ("openai/gpt-4.1", "GPT-4.1"),
        ("openai/gpt-4.1-mini", "GPT-4.1-mini"),
        ("anthropic/claude-haiku-3-5-20241022", "Claude Haiku"),
    ]

    print("\n" + "=" * 60)
    print("MODEL COST COMPARISON")
    print("=" * 60)

    results = []
    for model_id, name in models:
        grader = CriterionGrader(
            llm_config=LLMConfig(model=model_id, temperature=0.0)
        )

        start = time.perf_counter()
        result = await evaluate(
            dataset, grader,
            show_progress=False,
            experiment_name=f"cost-compare-{name.lower().replace(' ', '-')}"
        )
        elapsed = time.perf_counter() - start

        results.append({
            "model": name,
            "cost": result.total_completion_cost or 0,
            "time": elapsed,
            "tokens": result.total_token_usage.total_tokens if result.total_token_usage else 0,
        })

    print(f"\n{'Model':<18} {'Cost':>10} {'Time':>10} {'Tokens':>12}")
    print("-" * 52)
    for r in sorted(results, key=lambda x: x["cost"]):
        print(f"{r['model']:<18} ${r['cost']:>9.4f} {r['time']:>9.1f}s {r['tokens']:>12,}")


async def benchmark_caching(dataset: RubricDataset):
    """Benchmark caching performance."""

    print("\n" + "=" * 60)
    print("CACHING BENCHMARK")
    print("=" * 60)

    grader = CriterionGrader(
        llm_config=LLMConfig(
            model="openai/gpt-4.1-mini",
            temperature=0.0,
            cache_enabled=True,
            cache_dir=".benchmark_cache",
        )
    )

    # Clear cache for fair comparison
    client = grader._clients["default"]
    client.clear_cache()

    # Cold run
    print("\nCold run (cache empty)...")
    start = time.perf_counter()
    result1 = await evaluate(dataset, grader, show_progress=False)
    cold_time = time.perf_counter() - start
    cold_cost = result1.total_completion_cost or 0

    # Warm run
    print("Warm run (cache populated)...")
    start = time.perf_counter()
    result2 = await evaluate(dataset, grader, show_progress=False)
    warm_time = time.perf_counter() - start
    warm_cost = result2.total_completion_cost or 0

    # Stats
    stats = client.cache_stats()

    print(f"\nResults:")
    print(f"  Cold run: {cold_time:.1f}s, ${cold_cost:.4f}")
    print(f"  Warm run: {warm_time:.1f}s, ${warm_cost:.4f}")
    print(f"  Speedup: {cold_time / max(warm_time, 0.01):.1f}x")
    print(f"  Cost savings: ${cold_cost - warm_cost:.4f}")
    print(f"\nCache stats:")
    print(f"  Entries: {stats['count']}")
    print(f"  Size: {stats['size'] / 1024:.1f} KB")


async def main():
    # Create dataset
    dataset = create_factcheck_dataset()
    print(f"Dataset: {dataset.name}")
    print(f"Items: {len(dataset)}")

    # Compare models
    await compare_models(dataset)

    # Benchmark caching
    await benchmark_caching(dataset)

    print("\n" + "=" * 60)
    print("COST OPTIMIZATION SUMMARY")
    print("=" * 60)
    print("""
Key strategies demonstrated:
1. Response caching - Avoids redundant API calls
2. Model selection - Cheaper models for simpler tasks
3. Prompt caching - Reduces repeated system prompt costs

For production, combine:
- cache_enabled=True (response caching)
- prompt_caching=True (Anthropic system prompt caching)
- Appropriate model selection for accuracy needs
- Checkpointing to avoid re-evaluation
""")


if __name__ == "__main__":
    asyncio.run(main())
```
