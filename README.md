
<p align="center">
  <a href="https://badge.fury.io/py/autorubric">
    <img src="https://badge.fury.io/py/autorubric.svg" alt="PyPI version" />
  </a>
  <a href="https://pypi.python.org/pypi/autorubric">
    <img src="https://img.shields.io/pypi/pyversions/autorubric.svg" alt="Python versions" />
  </a>
  <a href="https://autorubric.org">
    <img src="https://img.shields.io/badge/site-autorubric.org-blue.svg" alt="Website" />
  </a>
  <a href="https://arxiv.org/abs/2603.00077">
    <img src="https://img.shields.io/badge/arXiv-2603.00077-<COLOR>.svg" alt="arXiv" />
  </a>
</p>

# AutoRubric

A Python library for evaluating text outputs against weighted criteria using LLM-as-a-judge.

```bibtex
  @misc{rao2026autorubric,
        title={Autorubric: A Unified Framework for Rubric-Based LLM Evaluation},
        author={Delip Rao and Chris Callison-Burch},
        year={2026},
        eprint={2603.00077},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2603.00077},
  }
```

---


## Installation

```bash
pip install autorubric
```

## Quick Example

```python
import asyncio
from autorubric import Rubric, LLMConfig
from autorubric.graders import CriterionGrader

async def main():
    grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-5.1-mini"))

    rubric = Rubric.from_dict([
        {"weight": 10.0, "requirement": "States NMC cell-level energy density in the 250-300 Wh/kg range"},
        {"weight": 8.0, "requirement": "Identifies LFP thermal runaway threshold (~270°C) as higher than NMC (~210°C)"},
        {"weight": 6.0, "requirement": "States LFP cycle life advantage (2000-5000 cycles vs 1000-2000 for NMC)"},
        {"weight": -15.0, "requirement": "Incorrectly claims LFP has higher gravimetric energy density than NMC"}
    ])

    result = await rubric.grade(
        to_grade="""NMC cathodes (LiNixMnyCozO2) achieve 250-280 Wh/kg at the cell level,
        while LFP (LiFePO4) typically reaches 150-205 Wh/kg. However, LFP offers superior
        thermal stability with decomposition onset at ~270°C compared to ~210°C for NMC,
        and delivers 2000-5000 charge cycles versus 1000-2000 for NMC.""",
        grader=grader,
        query="Compare NMC and LFP cathode materials for EV battery applications.",
    )

    # result.score is `float | None` (None if the grade failed); guard before formatting.
    print(f"Score: {result.score:.2f}" if result.score is not None else "Score: n/a (grade failed)")
    for criterion in result.report:
        print(f"  [{criterion.final_verdict}] {criterion.criterion.requirement}")

asyncio.run(main())
```

## Documentation

Full documentation, API reference, and a cookbook with several dozen recipes are available at **[autorubric.org](https://autorubric.org/)**.

| Resource      | Link                                                                  |
| ------------- | --------------------------------------------------------------------- |
| Project site  | [autorubric.org](https://autorubric.org)                              |
| API reference | [autorubric.org/docs/api](https://autorubric.org/docs/api/)           |
| Cookbook      | [autorubric.org/docs/cookbook](https://autorubric.org/docs/cookbook/) |

## Using AutoRubric in Claude Code, Codex, Gemini CLI, and other coding agents

AutoRubric's documentation is indexed by [Context7](https://context7.com/websites/autorubric), so you can give your coding agent live access to the current API reference and cookbook. With it connected, the agent writes correct, idiomatic AutoRubric code — right imports, the async grading API, weighted criteria, ensemble and multi-choice config — instead of guessing from stale memory.

### 1. Connect the Context7 MCP server

Get a free API key at [context7.com/dashboard](https://context7.com/dashboard), then add the server to your agent (swap in your key):

**Claude Code**

```bash
claude mcp add --scope user --header "CONTEXT7_API_KEY: YOUR_API_KEY" \
  --transport http context7 https://mcp.context7.com/mcp
```

**Codex CLI**

```bash
codex mcp add context7 -- npx -y @upstash/context7-mcp --api-key YOUR_API_KEY
```

**Gemini CLI** — add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "context7": {
      "httpUrl": "https://mcp.context7.com/mcp",
      "headers": { "CONTEXT7_API_KEY": "YOUR_API_KEY" }
    }
  }
}
```

**Other clients** (Cursor, VS Code, Windsurf, Zed, …): run `npx ctx7 setup`, or see the [full client list](https://context7.com/docs/resources/all-clients). The underlying package is `@upstash/context7-mcp`.

### 2. Point the agent at AutoRubric

Reference the library in your prompt so Context7 loads the right docs — AutoRubric's library ID is `/websites/autorubric`:

> Write an AutoRubric grader that scores answers against weighted criteria with an ensemble of two judges. use context7, library /websites/autorubric

Context7 exposes a `resolve-library-id` tool (to find the library) and a docs-query tool (to fetch version-specific docs and examples); your agent invokes these automatically.

### Make it the default — drop this into your `CLAUDE.md` / `AGENTS.md`

To get high-quality AutoRubric code on every task (not just when you remember to ask), add a block like this to your agent's instruction file — `CLAUDE.md` for Claude Code, `AGENTS.md` for Codex, Gemini CLI, and most other agents:

```markdown
## Writing AutoRubric code

When writing or editing code that uses AutoRubric:

- Load the current docs via Context7 (library `/websites/autorubric`) first — rely on
  the real API, not prior memory or guesswork.
- The grading APIs are async: `await` `Rubric.grade`, `Grader.grade`, and `EvalRunner`,
  and treat `result.score` as `float | None` (guard before formatting).
- Use the feature that fits the task: weighted (±) criteria, `CriterionGrader`, ensemble
  judging with aggregation strategies, multi-choice (ordinal/nominal) criteria, batch
  `EvalRunner` with checkpointing, agreement metrics and bootstrap CIs, YAML configs, or
  meta-rubric improvement.
- Verify exact signatures and types against the API reference
  (https://autorubric.org/docs/api/) and reuse patterns from the cookbook
  (https://autorubric.org/docs/cookbook/).
- Ensure provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) are set; AutoRubric
  reaches 100+ providers via LiteLLM.
```

## Features

| Feature                    | Description                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| Weighted criteria          | Positive and negative weights with explicit requirements                 |
| Per-criterion explanations | Every verdict includes the judge's reasoning                             |
| 100+ LLM providers         | OpenAI, Anthropic, Google, Azure, Groq, Ollama, and more via LiteLLM     |
| Ensemble judging           | Combine multiple LLM judges with configurable aggregation strategies     |
| Few-shot calibration       | Provide labeled examples to improve grading consistency                  |
| Multi-choice criteria      | Ordinal and nominal scales beyond binary met/unmet verdicts              |
| Batch evaluation           | High-throughput `EvalRunner` with checkpointing and resumption           |
| Metrics & validation       | Agreement metrics, bootstrap confidence intervals, distribution analysis |
| Length penalty             | Configurable penalty for overly long responses                           |
| Thinking/reasoning support | Budget-controlled extended thinking for supported models                 |
| Response caching           | Disk-based caching to avoid redundant LLM calls                          |
| Dataset support            | Structured datasets with per-item rubrics, prompts, and ground truth     |
| YAML configuration         | Define rubrics, LLM configs, and datasets in YAML                        |
| Meta-rubric evaluation     | Evaluate and automatically improve rubric quality                        |

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This research was developed with funding from the Defense Advanced Research Projects Agency’s (DARPA) SciFy program (Agreement No. HR00112520300). The views expressed
are those of the author and do not reflect the official policy or position of the Department of Defense or the U.S. Government.