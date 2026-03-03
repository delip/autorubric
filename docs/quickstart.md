# Quickstart

This guide covers installation, basic configuration, and your first evaluation with AutoRubric.

## Installation

=== "uv"

    ```bash
    uv add autorubric
    ```

=== "pip"

    ```bash
    pip install autorubric
    ```

## API Key Setup

AutoRubric uses [LiteLLM](https://docs.litellm.ai/) under the hood, providing access to 100+ LLM providers. Set up your API key for your chosen provider:

```bash
# OpenAI
export OPENAI_API_KEY=your_key_here

# Anthropic
export ANTHROPIC_API_KEY=your_key_here

# Google
export GEMINI_API_KEY=your_key_here
```

AutoRubric automatically loads environment variables from `.env` files.

### Supported Providers

| Provider | Model Format | Environment Variable |
|----------|-------------|---------------------|
| OpenAI | `openai/gpt-4.1`, `openai/gpt-4.1-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-2.5-flash`, `gemini/gemini-2.5-pro` | `GEMINI_API_KEY` |
| Azure OpenAI | `azure/openai/gpt-4.1` | `AZURE_API_KEY`, `AZURE_API_BASE` |
| Groq | `groq/llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| Ollama | `ollama/qwen3:14b`, `ollama/llama3` | (local, no key needed) |

See the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers) for the full list.

## Your First Evaluation

```python
import asyncio
from autorubric import Rubric, LLMConfig
from autorubric.graders import CriterionGrader

async def main():
    # 1. Configure the LLM judge
    grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4.1-mini"))

    # 2. Define your evaluation rubric
    rubric = Rubric.from_dict([
        {"weight": 10.0, "requirement": "States NMC cell-level energy density in the 250-300 Wh/kg range"},
        {"weight": 8.0, "requirement": "Identifies LFP thermal runaway threshold (~270°C) as higher than NMC (~210°C)"},
        {"weight": 6.0, "requirement": "States LFP cycle life advantage (2000-5000 cycles vs 1000-2000 for NMC)"},
        {"weight": -15.0, "requirement": "Incorrectly claims LFP has higher gravimetric energy density than NMC"}
    ])

    # 3. Grade a response
    result = await rubric.grade(
        to_grade="""NMC cathodes (LiNixMnyCozO2) achieve 250-280 Wh/kg at the cell level,
        while LFP (LiFePO4) typically reaches 150-205 Wh/kg. However, LFP offers superior
        thermal stability with decomposition onset at ~270°C compared to ~210°C for NMC,
        and delivers 2000-5000 charge cycles versus 1000-2000 for NMC.""",
        grader=grader,
        query="Compare NMC and LFP cathode materials for EV battery applications.",
    )

    # 4. Review results
    print(f"Score: {result.score:.2f}")  # Score is 0.0-1.0
    for criterion in result.report:
        print(f"  [{criterion.final_verdict}] {criterion.criterion.requirement}")
        print(f"    -> {criterion.final_reason}")

asyncio.run(main())
```

## Core Concepts

### Rubrics

A rubric is a list of criteria that define what you're evaluating. Each criterion has:

- **requirement**: What the response should (or shouldn't) contain
- **weight**: How important this criterion is (positive = good, negative = bad)
- **name** (optional): Identifier for the criterion

```python
from autorubric import Rubric, Criterion

# Direct construction
rubric = Rubric([
    Criterion(name="accuracy", weight=10.0, requirement="States the correct answer"),
    Criterion(name="clarity", weight=5.0, requirement="Explains reasoning clearly"),
    Criterion(name="errors", weight=-15.0, requirement="Contains factual errors"),
])

# From dictionaries (name and weight are optional)
rubric = Rubric.from_dict([
    {"weight": 10.0, "requirement": "States the correct answer"},
    {"requirement": "Explains reasoning clearly"},  # weight defaults to 10.0
])

# From files
rubric = Rubric.from_file("rubric.yaml")
rubric = Rubric.from_file("rubric.json")
```

### Graders

Graders evaluate responses against rubrics. The `CriterionGrader` is the main grader with support for:

- **Single LLM**: One judge model
- **Ensemble**: Multiple judges with aggregation
- **Few-shot**: Calibration with labeled examples

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader, JudgeSpec

# Single LLM
grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4.1-mini"))

# Ensemble with multiple judges
grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "gpt"),
        JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
    ],
    aggregation="majority",
)
```

### Verdicts

Each criterion receives one of three verdicts:

| Verdict | Meaning |
|---------|---------|
| `MET` | The requirement is satisfied |
| `UNMET` | The requirement is not satisfied |
| `CANNOT_ASSESS` | Insufficient evidence to determine |

### Scoring

The final score is calculated from weighted verdicts:

- Positive criteria: MET earns the weight, UNMET earns 0
- Negative criteria: MET subtracts the weight, UNMET contributes 0
- Score is normalized to 0-1 range by default

$$
\text{score} = \max\left(0, \min\left(1, \frac{\sum_{i=1}^{n} \mathbb{1}[\text{verdict}_i = \text{MET}] \cdot w_i}{\sum_{i=1}^{n} \max(0, w_i)}\right)\right)
$$

## LLMConfig Options

`LLMConfig` controls LLM behavior:

```python
from autorubric import LLMConfig

config = LLMConfig(
    # Required
    model="openai/gpt-4.1-mini",

    # Sampling
    temperature=0.0,           # 0.0 = deterministic (default)
    max_tokens=1024,           # Maximum response tokens

    # Rate limiting
    max_parallel_requests=10,  # Concurrent requests per provider

    # Caching
    cache_enabled=True,        # Enable response caching
    cache_dir=".autorubric_cache",
    cache_ttl=3600,            # Cache TTL in seconds

    # Extended thinking (for complex evaluations)
    thinking="high",           # "low", "medium", "high", or token budget
)
```

## Loading Rubrics from YAML

```yaml
# rubric.yaml
- name: accuracy
  weight: 10.0
  requirement: "States the correct answer"

- name: clarity
  weight: 5.0
  requirement: "Explains reasoning clearly"

- name: errors
  weight: -15.0
  requirement: "Contains factual errors"
```

```python
rubric = Rubric.from_file("rubric.yaml")
```

## Batch Evaluation

For evaluating multiple responses, use `EvalRunner` or the `evaluate()` function:

```python
from autorubric import RubricDataset, LLMConfig, evaluate
from autorubric.graders import CriterionGrader

async def batch_eval():
    # Load dataset
    dataset = RubricDataset.from_file("data.json")

    # Configure grader with rate limiting
    grader = CriterionGrader(
        llm_config=LLMConfig(
            model="openai/gpt-4.1-mini",
            max_parallel_requests=10,
        )
    )

    # Run evaluation
    result = await evaluate(dataset, grader, show_progress=True)

    print(f"Evaluated {result.successful_items}/{result.total_items}")
    print(f"Total cost: ${result.total_completion_cost:.4f}")
```

## Next Steps

- **[API Reference](api/index.md)**: Complete documentation of all classes and functions
- **[Cookbook](cookbook/index.md)**: Practical examples and recipes
- **[Ensemble Judging](api/ensemble.md)**: Reduce bias with multiple judges
- **[Metrics](api/metrics.md)**: Measure agreement with ground truth
