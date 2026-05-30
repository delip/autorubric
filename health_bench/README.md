# HealthBench Dataset (AutoRubric Format)

Data from OpenAI's [HealthBench](https://openai.com/index/healthbench) benchmark
converted into AutoRubric's `RubricDataset` format, with simple-evals parity
as the design goal.

> Arora, R. K., Wei, J., Soskin Hicks, R., Bowman, P., et al. (2025).
> *HealthBench: Evaluating Large Language Models Towards Improved Human Health.*
> OpenAI. [paper.pdf](./paper.pdf)

Reference implementation: [openai/simple-evals](https://github.com/openai/simple-evals)
(`healthbench_eval.py`, `healthbench_meta_eval.py`).

## Contents

```
health_bench/
├── README.md                     # this file
├── healthbench-100-README.md     # design notes for the 100-item val/test split
├── paper.pdf                     # HealthBench paper (offline copy)
├── download_raw_data.py          # stage 1 — fetch the raw HealthBench JSONL files
├── convert_to_rubric_dataset.py  # stage 2 — raw JSONL → AutoRubric RubricDatasets
├── sample_healthbench_100.py     # stage 3 — carve a stratified 100-item val/test split
├── healthbench_parity.py         # autorubric ↔ simple-evals grading parity check
├── raw_data/                     # 4 source JSONL files (downloaded)
└── autorubric_dataset/           # converted datasets, sidecar metadata, 100-item split
```

`raw_data/` and `autorubric_dataset/` are build artifacts: regenerate them with
the scripts rather than editing by hand. The first three scripts form a pipeline
(download → convert → sample); `healthbench_parity.py` is an independent
verification tool and is the only script that needs a `simple-evals` checkout.

## Overview

HealthBench measures how well chat models handle realistic health-related
conversations. **262 physicians** across 26 specialties and 60 countries
authored rubrics for **5,000 prompts**. Each prompt's response is graded
by an LLM judge against the prompt's own weighted rubric; the final score
is the mean per-prompt rubric-coverage, clipped to `[0, 1]`.

This conversion targets two requirements:

1. **Faithful entity model** — the 5,000 prompts are the single source of
   truth; hard/consensus/meta are filtered views over the same pool.
2. **Numerical parity with simple-evals** — graded with autorubric and
   `EvalConfig(use_reference_submission=False)`, scores should agree with
   simple-evals' `HealthBenchEval` within LLM stochasticity.

## Entity Model

| Entity | Count | Key |
| --- | ---: | --- |
| Prompt | 5,000 | `prompt_id` (UUID) — globally unique across all files |
| Example-level criteria (per-prompt, unique) | ~48,562 | implicit (prompt_id × position) |
| Cluster-level criteria (reusable catalog) | 37 | `cluster:*` tag in raw rubric |
| Physician ideal completions | 4,206 | prompt_id (Groups 1/2/3) |
| Physician reference completions (model-generated, shown to physicians) | 11,112 | prompt_id × ref_idx (Groups 2/3 only) |
| Meta-eval completions (separate sampling run) | 14,592 | `completion_id` (UUID) |
| Physician × criterion × completion judgments | 60,896 | (prompt_id, completion_id, criterion_name, physician_id) |

**Splits in the raw release** (`raw_data/`) are all filtered views of the
5,000-prompt pool:

| Split | Prompts | Notes |
| --- | ---: | --- |
| `healthbench_main.jsonl` | 5,000 | canonical pool, full rubrics |
| `healthbench_hard.jsonl` | 1,000 | strict subset; rubrics identical to main |
| `healthbench_consensus.jsonl` | 3,671 | strict subset; rubrics filtered to cluster-level criteria only |
| `healthbench_meta.jsonl` | 29,511 (rows) | grader meta-eval: 14,592 unique completions × 1–3 applicable consensus criteria; prompt_ids ⊂ consensus prompt_ids |

Counts above are line counts of the released JSONL files. The 60,896 figure
comes from summing the `binary_labels` lists across meta-eval rows (each row
carries 2–5 physician labels).

## Themes and Axes

- **7 themes** (per-example, `example_tags` `theme:*`): Emergency referrals,
  Context seeking, Global health, Health data tasks, Expertise-tailored
  communication, Responding under uncertainty, Response depth.
- **5 axes** (per-criterion, rubric `tags` `axis:*`): Completeness (39%),
  Accuracy (33%), Context awareness (16%), Communication quality (8%),
  Instruction following (4%).
- **`level:*` tags**: `example` vs `cluster` distinguish the two criterion
  flavors; `cluster:<id>` ties cluster-level items to one of the 37 reusable
  consensus criteria.

## Output Inventory

`convert_to_rubric_dataset.py` writes five `RubricDataset` JSONs (four datasets
plus a 2-item smoke test) and five sidecar metadata files to `autorubric_dataset/`:

| File | Items | Description |
| --- | ---: | --- |
| `healthbench.json` | 5,000 | rubrics-only template; `DataItem.submission = ""`. Fill in your model's response then grade. |
| `healthbench_physician_ideal.json` | 4,206 | `submission` = physician-written ideal completion (Groups 1+2+3). |
| `healthbench_physician_references.json` | 11,112 | `submission` = one of 4 model-generated reference completions (Groups 2+3). `reference_submission` = matching physician ideal as a quality anchor. |
| `healthbench_meta.json` | 14,592 | one item per unique meta-eval completion; per-item rubric of 1–3 applicable consensus criteria; `ground_truth` = per-criterion physician majority vote (ties → `CANNOT_ASSESS`). |
| `test_rubric_dataset.json` | 2 | first two items of `healthbench.json` for smoke-testing. |

Sidecar files (non-RubricDataset; pure JSON / JSONL):

| File | Rows | Description |
| --- | ---: | --- |
| `healthbench_tags.jsonl` | 57,237 | one row per `(prompt_id, criterion_idx)` with the raw `level:* / axis:* / cluster:*` tags. Covers all four datasets above. |
| `hard_prompt_ids.json` | 1,000 | UUIDs to reconstruct the hard subset from `healthbench.json`. |
| `consensus_prompt_ids.json` | 3,671 | UUIDs to reconstruct the consensus subset. |
| `consensus_criteria_catalog.json` | 37 | reference catalog of the 37 cluster-level criteria with (cluster_id, criterion_text, axis, points). |
| `meta_physician_labels.jsonl` | 60,896 | long-form per-physician labels for the meta-eval (one row per (prompt_id, completion_id, criterion_name, physician_id, label)). Needed to reproduce the paper's Macro-F1 grader-evaluation metric. |

### Derived 100-item split

`sample_healthbench_100.py` carves a small, stratified val/test split out of
`healthbench_meta.json` (joined with `healthbench_physician_ideal.json`) for fast
grader-config iteration. See [`healthbench-100-README.md`](./healthbench-100-README.md)
for the full sampling design, filters, stratification, and statistical notes.

| File | Items | Description |
| --- | ---: | --- |
| `healthbench-100-val.json` | 100 | stratified meta-eval items for iterating on grader prompt/config. |
| `healthbench-100-test.json` | 100 | hold-out, disjoint from val on `(prompt_id, completion_id)`; freeze and report from this set. |
| `healthbench-100-manifest.json` | — | seed, filter rules, per-cell counts, and the chosen `(prompt_id, completion_id)` pairs. Re-running the sampler at the same seed reproduces the split exactly. |

## Field Mapping

| HealthBench source | AutoRubric destination |
| --- | --- |
| `prompt` (list of chat messages) | `DataItem.prompt` — rendered as `"\n\n".join(f"{role}: {content}" for m in messages)`, matching simple-evals byte-for-byte. Lands in the judge's `<input>` tag. |
| Assistant response under evaluation | `DataItem.submission` — bare string; lands in `<submission>`. Empty for the template. |
| `prompt_id` | `DataItem.description` (lead position: `prompt_id=<uuid>`). |
| `example_tags` (`theme:*`, `physician_agreed_category:*`) | appended to `DataItem.description`, pipe-separated. |
| `ideal_completions_data.ideal_completions_group` | appended to `description` when relevant. |
| `rubrics[i].criterion` | `Criterion.requirement` (verbatim). |
| `rubrics[i].points` (signed, [-10, 10]) | `Criterion.weight`. Sign drives `<criterion_type>positive|negative</criterion_type>`. |
| `Criterion.name` | always `C{i+1}` (plain). Tag-bearing names would collide with `meta/_improve.py`'s substring-matching feedback attribution. Tags go to the sidecar. |
| `rubrics[i].tags` | `healthbench_tags.jsonl` keyed by `(prompt_id, criterion_idx)`. |
| meta `binary_labels` (per row) | majority vote → `DataItem.ground_truth` per criterion; ties → `CANNOT_ASSESS`. |
| meta `binary_labels` + `anonymized_physician_ids` (raw) | `meta_physician_labels.jsonl`. |
| `ideal_completion` (for physician_references items) | `DataItem.reference_submission`. Whether the judge actually sees it is controlled by `EvalConfig.use_reference_submission`. |
| `canary` | dropped (constant per file; preserved in `raw_data/`). |

## Scoring (matches the paper, Eq. 1)

Per-prompt score:

```
s_i = Σ_j 1{met_ij} · p_ij  /  Σ_j max(0, p_ij)
```

`s_i` can go negative when negative-point criteria are met. The final
headline number is the **clipped mean**: `clip(mean_i s_i, 0, 1)`. AutoRubric's
default normalization in `RubricDataset.compute_weighted_score` matches this
per-item formula; aggregate over `EvalResult.item_results` with mean-then-clip
to match the headline.

The optional `penalty_per_500_chars * (len - center)/500` length adjustment
from `simple-evals/healthbench_eval.py:157` is **not** in the paper — it's
an analysis hook. Apply post-hoc on `EvaluationReport.score` if needed.

## Reproducing simple-evals Parity

Grade with autorubric using these settings to match `simple-evals`'
`HealthBenchEval`:

```python
from autorubric.dataset import RubricDataset
from autorubric.eval import EvalConfig, EvalRunner, evaluate
from autorubric.graders.criterion_grader import CriterionGrader
from autorubric.llm import LLMConfig

ds = RubricDataset.from_file("health_bench/autorubric_dataset/healthbench_physician_ideal.json")
grader = CriterionGrader(llm_config=LLMConfig(model="gpt-4.1-2025-04-14"))

config = EvalConfig(
    use_reference_submission=False,   # <- parity: simple-evals never passes a reference
    show_progress=True,
)
result = await evaluate(dataset=ds, grader=grader, config=config)

# Aggregate per-item scores into the paper's headline number:
import numpy as np
per_item = [r.report.score for r in result.item_results if r.error is None]
healthbench_score = float(np.clip(np.mean(per_item), 0, 1))
```

To grade with the physician ideal as a calibration anchor (autorubric-mode,
**not** simple-evals-parity), drop `use_reference_submission=False`.

### Reproducing HealthBench Hard / Consensus

Both are filtered views; no separate dataset file is needed:

```python
import json
from pathlib import Path

ds = RubricDataset.from_file("health_bench/autorubric_dataset/healthbench.json")

# HealthBench Hard: 1,000 prompts, full rubrics.
hard_ids = set(json.loads(Path("health_bench/autorubric_dataset/hard_prompt_ids.json").read_text()))
hard_items = [it for it in ds.items if it.description.split(" | ", 1)[0].split("=", 1)[1] in hard_ids]

# HealthBench Consensus: 3,671 prompts × cluster-level criteria only.
cons_ids = set(json.loads(Path("health_bench/autorubric_dataset/consensus_prompt_ids.json").read_text()))
tags_by_prompt: dict[str, set[int]] = {}    # (prompt_id) -> set of criterion_idx that are level:cluster
for line in Path("health_bench/autorubric_dataset/healthbench_tags.jsonl").open():
    row = json.loads(line)
    if "level:cluster" in row["tags"]:
        tags_by_prompt.setdefault(row["prompt_id"], set()).add(row["criterion_idx"])
# Build new items with rubrics restricted to cluster-level indices, similar to consensus.jsonl.
```

### Reproducing the Macro-F1 grader-evaluation metric

The paper's headline grader-evaluation number is pairwise Macro-F1 between
the grader's verdicts and individual physician labels (one F1 per
`(criterion, completion, physician)` triple, balanced positive/negative).
`healthbench_meta.json` only carries the collapsed majority vote in
`DataItem.ground_truth`; the raw per-physician labels live in
`meta_physician_labels.jsonl`. Grade `healthbench_meta.json` with autorubric,
then join the grader's per-criterion verdicts against
`meta_physician_labels.jsonl` rows on
`(prompt_id, completion_id, criterion_name)` and compute F1 yourself.

## Cross-Reference Invariants

Invariants between files (verified at build time):

- `set(prompt_id) over healthbench.json` == `set(prompt_id) in raw_data/healthbench_main.jsonl` (5,000 each)
- `hard_prompt_ids.json ⊆ healthbench.json` and `consensus_prompt_ids.json ⊆ healthbench.json`, but hard and consensus are **not** nested in each other: `hard ∩ consensus = 586`
- `meta prompt_ids == consensus_prompt_ids` (both 3,671)
- `sum of binary_labels across raw_data/healthbench_meta.jsonl == 60,896` (matches paper headline)
- All 57,237 `healthbench_tags.jsonl` rows round-trip verbatim against the raw rubric tags
- Every `Criterion.name` across 272,887 emitted criteria is plain `C{i+1}` (no bracketed suffix, no rubric-improvement collision risk)

## Caveats

- **Verdict alphabet drift**: autorubric supports `CANNOT_ASSESS` (default
  `SKIP` strategy excludes from the denominator); simple-evals retries
  until the grader commits to a binary. Expect a small number of items to
  diverge. Configure `CannotAssessConfig(strategy=ZERO)` if you want strict
  binary semantics.
- **Conversation block layout**: simple-evals shows the response as the
  final `assistant:` turn in one conversation block; autorubric puts the
  conversation in `<input>` and the response in `<submission>` (two blocks).
  Same information, different packaging. Spot-check confirms the conversation
  text is character-for-character identical.
- **System prompt placement**: autorubric loads grading instructions into
  the LLM `system` message; simple-evals inlines them in the user message.
  Different placement, equivalent content. This is the largest qualitative
  divergence; could affect borderline judgments.
- **Few-shot grading is incompatible** with per-item-rubric datasets like
  this one. `CriterionGrader._prepare_examples` reads the dataset-level
  rubric. Regular (zero-shot) grading works fine.
- **`RubricDataset.criterion_names`, `num_criteria`, `total_positive_weight`
  raise** when global rubric is `None`. Consumers must use
  `get_item_rubric(idx)`.

## Downloading Raw Data

The four source JSONL files are hosted on OpenAI's public blob store (the same
URLs `simple-evals/healthbench_eval.py` and `healthbench_meta_eval.py` use).
Fetch them with the bundled downloader:

```bash
cd health_bench
uv run python download_raw_data.py            # ~240 MB; skips files already present
uv run python download_raw_data.py --force    # re-download everything
```

Files land in `raw_data/` under the short names the conversion script
expects (`healthbench_main.jsonl`, `healthbench_hard.jsonl`,
`healthbench_consensus.jsonl`, `healthbench_meta.jsonl`). Standard library
only; no `simple-evals` checkout required.

## Conversion Script

```bash
cd health_bench
uv run python convert_to_rubric_dataset.py
```

Reads `raw_data/*.jsonl`, writes `autorubric_dataset/*`. Idempotent; uses
only the standard library.

## Sampling the 100-item Split

```bash
cd health_bench
uv run python sample_healthbench_100.py
```

Reads `autorubric_dataset/healthbench_meta.json` and
`autorubric_dataset/healthbench_physician_ideal.json`, writes the three
`healthbench-100-*` files. Standard library only; deterministic at the fixed
seed baked into the script, so reruns are byte-identical. Full design notes are
in [`healthbench-100-README.md`](./healthbench-100-README.md).

## Parity Check

```bash
cd health_bench
uv run --with blobfile python healthbench_parity.py --num-examples 30 --seed 0
```

`healthbench_parity.py` grades the same physician-ideal items two ways —
autorubric's `Rubric.grade()` and simple-evals' `HealthBenchEval`, both with the
same Gemini grader at `temperature=0` — and reports per-criterion verdict
agreement, per-example score deltas, and the cross-correlation between the two
pipelines. The detailed per-item report is written next to the script.

Unlike the other scripts, this one **requires a local clone of
[openai/simple-evals](https://github.com/openai/simple-evals)**: it imports that
repo's grading internals plus a litellm-backed `GeminiSampler` shim
(`sampler/gemini_sampler.py`). Clone it at the repo root so the package lives at
`<repo-root>/simple-evals/`, or set the `SIMPLE_EVALS_DIR` environment variable
to point elsewhere.

## License

Source data and rubric content are © OpenAI, released under MIT alongside
the `simple-evals` repository. The HealthBench paper is included as `paper.pdf`
for offline reference.
