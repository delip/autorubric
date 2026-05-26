# Research-Plan-Gen Dataset (AutoRubric Format)

Data from Meta's [`facebook/research-plan-gen`](https://huggingface.co/datasets/facebook/research-plan-gen)
corpus converted into AutoRubric's `RubricDataset` format.

**Citation:**

> Goel, S., Hazra, R., Jayalath, D., Willi, T., Jain, P., Shen, W. F., Leontiadis, I., Barbieri, F., Bachrach, Y., Geiping, J., & Whitehouse, C. (2025). *Training AI Co-Scientists Using Rubric Rewards.*
> arXiv: [2512.23707](https://arxiv.org/abs/2512.23707) · HF: [`facebook/research-plan-gen`](https://huggingface.co/datasets/facebook/research-plan-gen)

## Overview

The dataset is a large, automatically-extracted corpus of **research goals** paired with
**goal-specific grading rubrics** and a **reference research plan**, mined from papers across three
domains. It is used to train and evaluate AI "co-scientists" that draft research plans, with the
rubric serving as the reward/grading signal.

Each row is a self-contained grading task: given a research `Goal`, a response (a research plan)
should be judged against a list of plain-language rubric criteria, with the paper-derived
`Reference solution` available as a reference. This is a natural fit for AutoRubric's per-item
`Criterion` / `RubricDataset` model.

The source ships **3 configs × 2 splits = 6 parquet files**, converted here to **6 JSON files**.

| Config   | Domain                         | Train | Test |
| -------- | ------------------------------ | ----: | ---: |
| `arxiv`  | arXiv preprints (all subjects) | 6,573 | 1,496 |
| `ml`     | machine-learning papers        | 6,872 |  685 |
| `pubmed` | biomedical / PubMed papers     | 6,423 |  464 |

Per item there are typically 10 binary criteria (range 1–10), and every item carries a reference
solution.

## Field Mapping

Each source row maps to one AutoRubric `DataItem`:

| HF source field      | AutoRubric field        | Notes                                                          |
| -------------------- | ----------------------- | -------------------------------------------------------------- |
| `Goal`               | `item.prompt`           | the research-goal task prompt (per-item; global prompt is null) |
| `Rubric` (list[str]) | `item.rubric`           | each string → `Criterion(name="C{i+1}", requirement=…)`         |
| `Reference solution` | `item.reference_submission` | paper-derived gold research plan                           |
| `Subdomain`/`Category`/`Identifier`/`article_id`/`q_id` | `item.description` | compact provenance string                |
| —                    | `item.submission`       | `""` (empty — see below)                                       |
| —                    | `item.ground_truth`     | `null` (source has no per-criterion verdicts)                  |

The global `rubric` and global `prompt` are `null`; every item carries its own rubric and prompt.

### Rubric / Criterion structure

Criteria are **binary** (MET/UNMET) with `name` `C1, C2, …` and the rubric string as `requirement`.
The source rubrics carry **no weights**, so every criterion uses AutoRubric's default
`weight = 10.0` (uniform). With uniform weights, the normalized score equals the fraction of
criteria met.

### Empty submission, reference provided

This is a **generation** benchmark, not a pre-labeled grading set: `submission` is empty and there
are no ground-truth verdicts. The intended workflow is to grade a *model-generated* research plan
against the per-item rubric, using `reference_submission` (the paper's own plan) as a reference.
This mirrors `examples/data/sharma_etal_2025_research_rubrics.json`.

`description` examples:

- arxiv/ml: `arxiv | cs.IR | id=2508.08404 | article=ff96… | q=3e83…`
- pubmed:   `pubmed | id=38207540 | article=e1c6… | q=2e4c…` (Subdomain/Category empty in this config)

## File Format

Each JSON file is a serialized `RubricDataset` loadable via:

```python
from autorubric import RubricDataset

dataset = RubricDataset.from_file("examples/data/research-plan-gen/pubmed_test.json")
item = dataset.items[0]
item.prompt                 # the research Goal
item.rubric.rubric          # list[Criterion] (binary, weight 10.0)
item.reference_submission   # the reference research plan
item.submission             # "" — fill in a model-generated plan to grade
```

## Reproduction

The converter is at `scripts/convert_research_plan_gen.py`. It downloads the parquet files via the
`hf` CLI into a gitignored cache (`.cache/research-plan-gen/`), then writes the 6 JSON files here.
Run inside the project's conda environment (which has `hf`, `pyarrow`, and `autorubric`):

```bash
python scripts/convert_research_plan_gen.py
```
