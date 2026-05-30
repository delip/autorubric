# HealthBench-100: val/test subsets for grader experimentation

Two 100-item `RubricDataset`s carved out of HealthBench's meta-evaluation
pool so you can iterate on grader configurations (prompt edits, ensemble
strategies, reference-anchored vs. zero-shot) without paying the cost of
running the full 14,592-item meta dataset on every change.

## Files

| File | Purpose |
| --- | --- |
| `autorubric_dataset/healthbench-100-val.json`      | 100 items. Use for iterating on grader prompt/config. |
| `autorubric_dataset/healthbench-100-test.json`     | 100 items. Disjoint from val on `(prompt_id, completion_id)`. Freeze and report from this set. |
| `autorubric_dataset/healthbench-100-manifest.json` | Seed, filter rules, per-cell counts, and the chosen `(prompt_id, completion_id)` pairs per split. Re-runs of the sampler at the same seed produce byte-identical outputs. |
| `sample_healthbench_100.py`                        | Deterministic sampler. `cd health_bench && uv run python sample_healthbench_100.py`. |

## What each item looks like

Each `DataItem` carries:

- `prompt`            — the rendered HealthBench conversation (one or more turns ending in a user message)
- `submission`        — a model-generated completion that physicians actually graded
- `reference_submission` — the **physician-written ideal** completion for the same `prompt_id`, joined from `healthbench_physician_ideal.json`
- `ground_truth`      — per-criterion physician majority vote (`MET` / `UNMET` only — see filters below)
- `rubric`            — 1–3 consensus criteria (cluster-level; same shape as `healthbench_meta.json`)
- `description`       — `prompt_id=<uuid> | completion_id=<uuid> | criteria=[...; n_phys=<k>]`

## Provenance

```
healthbench_meta.json (14,592) ──┐
                                 ├─→ sample_healthbench_100.py ──→ val (100) + test (100)
healthbench_physician_ideal.json ┘             (filters + stratify + split)
            (4,206 — supplies reference_submission and theme)
```

Each sampled item one-to-one corresponds to a row group in
`meta_physician_labels.jsonl` via `(prompt_id, completion_id, criterion_name)`,
so the paper's Macro-F1 grader-evaluation metric (Section 8.1) is computable
end-to-end without re-running the conversion.

## Filters

| Step | Items remaining |
| --- | ---: |
| raw `healthbench_meta.json`                              | 14,592 |
| drop items with any `CANNOT_ASSESS` (or null) in `ground_truth` | 9,515 |
| require a matching `prompt_id` in `healthbench_physician_ideal.json` | **8,056** |
| sample 200 stratified items, split 100/100               | 200 |

The first filter removes physician 1-1 ties; otherwise they contaminate any
hit/miss-based accuracy metric. The second filter is what lets us populate
`reference_submission` — `healthbench_physician_ideal.json` covers 4,206
unique `prompt_id`s, 540 of which are not in the meta pool; the meta pool
contains 3,671 unique prompt ids, of which 3,131 (85%) have a physician
ideal. The 14.7% gap is dropped here.

## Stratification

Two axes, applied across val+test together (then 50/50 split):

1. **Theme** (7 levels — all themes from the paper):
   `communication`, `complex_responses`, `context_seeking`,
   `emergency_referrals`, `global_health`, `health_data_tasks`, `hedging`.
2. **Verdict class**, derived from `ground_truth`:
   - `all_met`   — every criterion was met
   - `mixed`     — at least one MET and one UNMET
   - `all_unmet` — every criterion was unmet

### Why these axes

The filtered pool is dominated by `all_met` (74.4%); a proportional sample
would mostly contain trivially-passing items, which carry almost no
information about where graders disagree with physicians. **Mixed and
all-UNMET items** are the discriminating cases — they require the grader
to actually catch a failure. We oversample them deliberately.

### Combined class targets

| Class      | Pool size | Target across val+test | Per subset |
| ---------- | --------: | ---------------------: | ---------: |
| `all_met`  | 5,994     | 66                     | 33         |
| `mixed`    | 1,379     | 68                     | 34         |
| `all_unmet`| 683       | 66                     | 33         |
| **Total**  | **8,056** | **200**                | **100**    |

Within each class, slots are allocated across the 7 themes by
**Hamilton's largest-remainder method** with a floor of 4 combined items
per `(theme, class)` cell (capped at cell size). Sparse cells cascade
their deficit to other themes within the same class. Two known sparse
cells in the pool:

- `global_health × mixed`            = 0 items (this cell gets allocated 0; the slack goes to other themes within `mixed`)
- `complex_responses × all_unmet`    = 4 items (this cell gets 4 → 2 in val, 2 in test, exhausting the cell)

### Realized per-subset counts

```
val   theme totals: communication=15  complex_responses= 9  context_seeking=16
                    emergency_referrals=12  global_health=16  health_data_tasks=15
                    hedging=17

test  theme totals: communication=12  complex_responses=10  context_seeking=17
                    emergency_referrals=13  global_health=14  health_data_tasks=15
                    hedging=19

val   class totals: all_met=33  mixed=34  all_unmet=33
test  class totals: all_met=33  mixed=34  all_unmet=33
```

Per-cell counts are exact in `healthbench-100-manifest.json`.

## Intended usage

### Default: paper-aligned meta-evaluation

```python
import asyncio
from autorubric.dataset import RubricDataset
from autorubric.eval import EvalConfig, evaluate
from autorubric.graders.criterion_grader import CriterionGrader
from autorubric.llm import LLMConfig

val = RubricDataset.from_file(
    "health_bench/autorubric_dataset/healthbench-100-val.json"
)
grader = CriterionGrader(llm_config=LLMConfig(model="gpt-4.1-2025-04-14"))
result = asyncio.run(
    evaluate(
        dataset=val,
        grader=grader,
        config=EvalConfig(
            use_reference_submission=False,   # simple-evals parity
            show_progress=True,
        ),
    )
)
```

Compare `result.item_results[i].report.report[j].verdict` against
`val.items[i].ground_truth[j]` for hit/miss-based accuracy, or compute
Macro-F1 by joining grader verdicts to `meta_physician_labels.jsonl` on
`(prompt_id, completion_id, criterion_name)`.

### Reference-anchored condition

Flip `use_reference_submission=True` to expose the physician-ideal
completion to the judge as a calibration anchor. This is **not**
paper-aligned (simple-evals never passes a reference), but it's the
natural ablation for "does anchoring change grader behavior" questions.

### Macro-F1 against individual physicians

```python
import json
from collections import defaultdict

labels = defaultdict(list)
with open("health_bench/autorubric_dataset/meta_physician_labels.jsonl") as f:
    for line in f:
        r = json.loads(line)
        labels[(r["prompt_id"], r["completion_id"], r["criterion_name"])].append(
            (r["physician_id"], r["label"])
        )

# join grader verdicts to per-physician labels, compute per-(criterion, physician)
# F1 pairs, then macro-average — same recipe as paper Section 8.1.
```

## Statistical notes

Each subset contains ~200 criterion-judgments (val: 201, test: 205) and
~430 per-physician labels (val: 429, test: 437).

- **Absolute Macro-F1 vs physicians** on a single subset: 95% CI of about
  ±3–4 F1 points around an estimated F1 of ~0.70. Workable for headline
  numbers and catching regressions; **not** sharp enough to discriminate
  sub-3-point grader prompt tweaks.
- **Paired grader-config comparisons** (McNemar on the same items): the
  unit is the criterion-judgment, not the item. n=200 judgments at ~10%
  discordance gives detection power for ~4-point swings.
- For tighter bounds, supplement with bootstrap CIs **resampled at the
  item level** (not criterion level) to avoid pseudoreplication of
  multi-criterion items.
- The 50-item exploratory slice is fine as a smoke test; do not draw
  decisions from sub-5-point deltas on it.

## Reproduction

```bash
cd health_bench
uv run python sample_healthbench_100.py
```

Stdlib only; no `simple-evals` checkout required. Idempotent at the fixed
seed in the script (`SEED = 20260517`). The manifest records the exact
filter counts, allocation, and chosen item pairs so the split is auditable
without rerunning.

## Caveats

- The sampler trusts `ground_truth` (majority vote over 2–5 physicians).
  Items where all criteria have `n_phys=2` are the thinnest physician
  coverage and most susceptible to majority-vote flips. The `n_phys`
  count is preserved in each item's `description` if you want to filter
  or stratify on it post-hoc.
- The `mixed` class is heterogeneous — it lumps "mostly met with one
  miss" together with "mostly unmet with one hit". If you need finer
  control, post-filter on the MET fraction inside `ground_truth`.
- Because the meta pool is restricted to consensus criteria
  (cluster-level only), the per-item rubrics are small (1–3 criteria).
  This is fine for meta-evaluation but means each item carries less
  signal than a typical HealthBench item with ~11 criteria. If you need
  per-item-level discrimination of grader behavior with larger rubrics,
  use `healthbench_physician_ideal.json` or `healthbench.json` directly
  (no ground truth there, but more criteria per item).
