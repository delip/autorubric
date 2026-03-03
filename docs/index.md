# AutoRubric

A Python library for evaluating text outputs against weighted criteria using LLM-as-a-judge.

<p align="center">
  <a href="https://pypi.org/project/autorubric/">
    <img src="https://img.shields.io/pypi/v/autorubric" alt="PyPI version" />
  </a>
  <a href="https://pypi.org/project/autorubric/">
    <img src="https://img.shields.io/pypi/pyversions/autorubric" alt="Python versions" />
  </a>
  <a href="https://github.com/delip/autorubric/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
  </a>
</p>

## What is AutoRubric?

AutoRubric provides a structured, research-backed approach to evaluating LLM outputs using rubric-based grading with LLM judges. Instead of relying on vague quality assessments, AutoRubric enables you to define explicit, weighted criteria and receive detailed per-criterion verdicts with explanations.

### Key Features

- **Rubric-based evaluation**: Define weighted criteria with explicit requirements
- **Multi-provider support**: Works with OpenAI, Anthropic, Google, Azure, Groq, Ollama, and 100+ providers via LiteLLM
- **Ensemble judging**: Combine multiple LLM judges to reduce bias and improve robustness
- **Few-shot learning**: Calibrate judges with labeled examples
- **Multi-choice criteria**: Support for ordinal and nominal scales beyond binary verdicts
- **Structured outputs**: Type-safe responses with detailed per-criterion reports
- **Batch evaluation**: High-throughput processing with checkpointing and resumption
- **Metrics & validation**: Comprehensive agreement metrics and bootstrap confidence intervals

## Installation

=== "uv"

    ```bash
    uv add autorubric
    ```

=== "pip"

    ```bash
    pip install autorubric
    ```

## Quick Example

```python
import asyncio
from autorubric import Rubric, LLMConfig
from autorubric.graders import CriterionGrader

async def main():
    # Configure LLM judge
    grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4.1-mini"))

    # Define evaluation rubric
    rubric = Rubric.from_dict([
        {"weight": 10.0, "requirement": "States the correct answer"},
        {"weight": 5.0, "requirement": "Provides clear explanation"},
        {"weight": -10.0, "requirement": "Contains factual errors"}
    ])

    # Grade a response
    result = await rubric.grade(
        to_grade="The answer is 42. This is derived from...",
        grader=grader,
        query="What is the answer to life, the universe, and everything?",
    )

    print(f"Score: {result.score:.2f}")
    for criterion in result.report:
        print(f"  [{criterion.final_verdict}] {criterion.criterion.requirement}")

asyncio.run(main())
```

## Why AutoRubric?

AutoRubric is built on research findings about effective LLM-as-a-judge evaluation:

- **Analytic rubrics**: Multiple explicit criteria increase interpretability and help diagnose failure modes (Casabianca et al., 2025; Ye et al., 2023)
- **Ensemble judging**: Multi-LLM panels reduce systematic bias and self-preference (Verga et al., 2024; He et al., 2025)
- **Structured evaluation**: Form-filling with per-criterion verdicts improves repeatability (Kim et al., 2024)
- **Position bias mitigation**: Randomized option order reduces positional effects (Wang et al., 2023)
- **CANNOT_ASSESS handling**: Explicit uncertainty option prevents low-confidence guessing (Min et al., 2023)

## Documentation

| Guide | Description |
|-------|-------------|
| **[Quickstart](quickstart.md)** | Get up and running with installation, configuration, and your first evaluation |
| **[Cookbook](cookbook/index.md)** | Practical examples and recipes for common evaluation scenarios |
| **[API Reference](api/index.md)** | Complete API documentation with all classes, functions, and types |

## Requirements

- Python 3.11+
- [LiteLLM](https://docs.litellm.ai/) for multi-provider LLM support
- [Pydantic](https://docs.pydantic.dev/) for structured outputs

## References

Casabianca, J., McCaffrey, D. F., Johnson, M. S., Alper, N., and Zubenko, V. (2025). Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications. arXiv:2501.02334.

He, J., Shi, J., Zhuo, T. Y., Treude, C., Sun, J., Xing, Z., Du, X., and Lo, D. (2025). LLM-as-a-Judge for Software Engineering: Literature Review, Vision, and the Road Ahead. arXiv:2510.24367.

Kim, S., Suk, J., Longpre, S., Lin, B. Y., Shin, J., Welleck, S., Neubig, G., Lee, M., Lee, K., and Seo, M. (2024). Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4334–4353.

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., and Hajishirzi, H. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 12076–12100.

Verga, P., Hofstatter, S., Althammer, S., Su, Y., Piktus, A., Arkhangorodsky, A., Xu, M., White, N., and Lewis, P. (2024). Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models. arXiv:2404.18796.

Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., and Sui, Z. (2023). Large Language Models are not Fair Evaluators. arXiv:2305.17926.

Ye, S., Kim, D., Kim, S., Hwang, H., Kim, S., Jo, Y., Thorne, J., Kim, J., and Seo, M. (2023). FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets. In *Proceedings of the 2024 International Conference on Learning Representations (ICLR)*.

## License

MIT License - see [LICENSE](https://github.com/delip/autorubric/blob/main/LICENSE) for details.
