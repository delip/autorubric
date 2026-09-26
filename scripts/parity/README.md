# LLM-judge parity harness

Checks that a change to AutoRubric does not change what existing LLM-judge users get. It
grades fixed datasets with fixed judge configurations on two libraries, the unmodified
baseline and the changed code, and compares the results:

- **Replay parity (checks R1-R7), free and deterministic.** The changed code is fed the
  baseline's own recorded LLM outputs through the response cache, with the network blocked.
  It must issue no new request and process the outputs exactly as the baseline did.
- **Synthetic parity (the same checks), free, needs no paid baseline.** Both libraries grade
  against a deterministic fake provider; their request logs must be identical and their
  processing equal.
- **Live parity (check L), paid.** Two fresh baseline runs and two fresh post-change runs at
  default temperature. Post-change runs may disagree with baseline runs no more than runs of
  the same code disagree with each other.

Everything lands in `experiments/parity/` (gitignored). The scripts are excluded from the
sdist.

| File | Role |
|---|---|
| `parity_config.py` | datasets, charm-100 split, judges, grader configs, seeds, construction paths, run layout |
| `run_parity.py` | grades every config against one library and writes a run directory |
| `compare_parity.py` | compares runs and writes a Markdown and a JSON report |

## Setup

Check out the baseline library as a separate worktree, e.g.
`git worktree add <baseline> main`. Baseline runs import it through `PYTHONPATH`; post-change
runs import this repository's `src/`. Every run asserts which library it loaded and records
its git commit, a dirty flag and a hash of its sources.

Create the parity manifest once, on the baseline code:

```bash
PYTHONPATH=<baseline>/src uv run --frozen python scripts/parity/run_parity.py \
    --code baseline --init-manifest --baseline-tree <baseline>
```

This pins, for every later run:

- copies of the two datasets, taken from the baseline tree;
- the charm-100 subset: `split_train_test(n_train=70, stratify=True, seed=0)` computed on the
  baseline code, stored as ordered item indices (30 graded items, 70 few-shot pool items);
- a snapshot of litellm's model cost map. By default litellm downloads that map on every
  import, which makes model routing, request parameters and `completion_cost` depend on the
  day of the run; runs load the snapshot instead and never fetch it.

## What is graded

| Config | Grader | Datasets |
|---|---|---|
| `A-luna`, `A-flashlite` | single judge, `seed=0` | essay (11 items, 5 binary criteria, one negative) + charm subset (30 items, 4 ordinal, 1 nominal, 1 binary) |
| `B` | ensemble `[luna, flashlite]`, `aggregation="majority"`, `seed=0` | charm subset |
| `C` | `luna` with `FewShotConfig(n_examples=3)` over the 70-item pool, `seed=0` | charm subset |
| `A-prime-*` | `A-*` with `LengthPenalty`, `CannotAssessStrategy.PARTIAL`, `normalize=False` | as `A` |
| `B-prime` | `B` with `aggregation="weighted"` (weights 2, 1), `ordinal_aggregation="median"`, `nominal_aggregation="unanimous"` | charm subset |

Only non-default grader settings are passed, so a changed default shows up as a difference.
Judges: `luna` = `gpt-6-luna`, `flashlite` = `gemini/gemini-3.5-flash-lite`, both
`temperature=1.0` (their provider default, written out) and `thinking="none"`. litellm does
not list `reasoning_effort` for `gpt-6-luna`, so its config also passes
`extra_params={"allowed_openai_params": ["reasoning_effort"]}`; without it litellm rejects
every luna call before sending it.

One pass is one `evaluate()` call (config x dataset x construction). A pass per
construction path is built:

- baseline code: `CriterionGrader(llm_config=...)` / `JudgeSpec(cfg, id)`;
- post-change code: `CriterionGrader(judge_model_config=...)` / `JudgeSpec(cfg, id)`
  (primary). Replay and synthetic runs also build every config with `llm_config=` (and, for
  ensembles, `JudgeSpec(llm_config=...)` and `JudgeSpec(judge_model_config=...)`). The
  keywords are detected by introspection; `--code after` fails if they are missing.

Derived configs (`*-prime`) make no calls of their own: in fresh mode they read their
parent's response cache with the network blocked. Each parent config has its own cache
directory, so a fresh run makes exactly 1,010 logical calls (luna 595, flashlite 415) plus
any retries.

## Modes

| Mode | Provider | Response cache | Cost |
|---|---|---|---|
| `fresh` | real (`--confirm-paid` and a passing pre-flight required) | on, the run's own | about 1,010 paid calls |
| `replay` | none: every call is refused and recorded | a copy of a fresh run's cache | free |
| `synthetic` | a deterministic fake | off, so every call is recorded | free |

**Blocked calls.** In replay (and for derived configs in fresh mode) a call reaches the
provider stand-in only if its response is not cached, i.e. the source run's call failed. The
harness raises an exception of the same failure category the source run recorded for that
(item, criterion, judge): `infrastructure` and `parse` abstain, `unknown` gets the
worst-case verdict. Verdicts, scores and metrics therefore stay exactly comparable; only the
failure text differs, and the comparator masks just that text. None of these exceptions is a
type `LLMClient` retries, so no backoff sleep occurs. A socket-level guard additionally
refuses and records any other network access in replay and synthetic runs.

**The synthetic provider.** Each response is a pure function of a SHA-256 of the request
(model, messages, response-format schema, sampling parameters). About 2% of requests raise a
terminal provider error (`InternalServerError` or `APIError`, both classified
`infrastructure`), 2% raise `RateLimitError` on the first attempt and succeed on the retry
(one second of backoff), 3% return malformed output (prose, truncated JSON, an invalid
verdict or option type, an out-of-range option), 1% return `content=None` (classified
`unknown`), and the rest return valid judgments covering MET / UNMET / CANNOT_ASSESS and every
option including NA. Responses carry token usage (litellm prices them from the pinned map), a
provider-reported model string, and for a quarter of Gemini answers a reasoning trace.
Synthetic runs use the same judge configs as fresh runs except that the response cache is off.

## Commands

Run from the repository root. `BASELINE` is the baseline worktree; baseline commands put its
`src/` on `PYTHONPATH`, post-change commands set no `PYTHONPATH`.

```bash
BASELINE=<baseline worktree>

# Free check of what litellm would send (closed loopback port, dummy key; no paid call).
PYTHONPATH=$BASELINE/src uv run --frozen python scripts/parity/run_parity.py \
    --code baseline --preflight --offline

# Synthetic runs: a baseline reference, and the changed code checked against it.
PYTHONPATH=$BASELINE/src uv run --frozen python scripts/parity/run_parity.py \
    --code baseline --mode synthetic --run S0a
uv run --frozen python scripts/parity/run_parity.py \
    --code after --mode synthetic --run S1 --checkpoint-source S0a
uv run --frozen python scripts/parity/compare_parity.py synthetic --baseline S0a --after S1

# Paid pre-flight (two calls), then the paid baselines B1 and B2.
PYTHONPATH=$BASELINE/src uv run --frozen python scripts/parity/run_parity.py \
    --code baseline --preflight --confirm-paid
PYTHONPATH=$BASELINE/src uv run --frozen python scripts/parity/run_parity.py \
    --code baseline --mode fresh --run B1 --confirm-paid
PYTHONPATH=$BASELINE/src uv run --frozen python scripts/parity/run_parity.py \
    --code baseline --mode fresh --run B2 --confirm-paid

# Free replay of B1's outputs on the changed code (checkpoint checks default to B1).
uv run --frozen python scripts/parity/run_parity.py \
    --code after --mode replay --run R1 --replay-cache B1
uv run --frozen python scripts/parity/compare_parity.py replay --baseline B1 --after R1

# Paid post-change runs and the live comparison.
uv run --frozen python scripts/parity/run_parity.py --code after --mode fresh --run A1 --confirm-paid
uv run --frozen python scripts/parity/run_parity.py --code after --mode fresh --run A2 --confirm-paid
uv run --frozen python scripts/parity/compare_parity.py live --b1 B1 --b2 B2 --a1 A1 --a2 A2
```

Fresh runs are refused unless `experiments/parity/preflight.json` passed with the same
pinned packages, cost-map snapshot and judge definitions. Run ids must be new. Reports go to
`experiments/parity/reports/<kind>__<runs>.{md,json}` (`--report <stem>` overrides). The
comparator exits 0 when every gating check passes, 1 when one fails, and 2 when the runs are
not comparable (differing `litellm`, `pydantic`, `openai` or `diskcache` versions, cost-map
snapshot, manifest or judge definitions).

## Checks

| Id | Replay / synthetic check |
|---|---|
| R1 | Replay: every blocked call is one the baseline could not answer from cache, and nothing else touched the network; an unexpected call prints its prompt diffed against the baseline's prompt for the same (item, criterion, judge). Synthetic: the recorded request logs are identical, with a readable diff of the first differing requests. |
| R2 | Per vote, per criterion and per item, the processing fields are equal: verdicts or selected options, shuffle orders, reasons, reasoning, final verdicts, agreement, scores, raw scores, judge scores, CANNOT_ASSESS counts, token usage and cost. |
| R3 | The serialized checkpoint records are equal except for new keys at their defaults (`probabilities: null`, `confidence: null`, `superseded: false`, `escalated: false`). |
| R4 | Every `compute_metrics` field (fixed-seed bootstrap included) is equal except for new keys at their defaults (each per-judge entry's `coverage: "full"` and `n_pairs: null`), and both `summary()` texts and the `to_dataframe()` frames are exactly equal, for results in memory and reloaded from the checkpoint. |
| R5 | Each manifest `grader_config` is a superset of the baseline's with equal values; a newly recorded `normalize` is listed explicitly. |
| R6 | All construction paths produce identical checkpoints, metrics and request logs; only `CriterionGrader(llm_config=...)` warns; no `DeprecationWarning` is attributed to a file inside `autorubric` (unless the baseline raised the same one). |
| R7 | The baseline's checkpoints load with `EvalResult.from_experiment`, their metrics recompute equal (fields as in R4, new keys at their defaults allowed; texts and frames exactly), and resuming a copy grades no item and makes no call. |

Live check L compares (item, criterion, judge) verdicts of the paid configs (multi-choice:
the selected option, with NA as its own category), excluding any unit that errored in one of
the four runs. With `d` the disagreement rate between two runs, within-phase
`w = (d(B1,B2) + d(A1,A2)) / 2` and cross-phase `c` = the mean of the four baseline/after
pairs. It passes when `c - w` is not significantly positive (one-sided cluster bootstrap over
items, 10,000 resamples, alpha 0.05) and `c - w <= 0.03`. The overall result gates; per-config
results, a side-by-side of accuracy, mean kappa, score RMSE and error rate with bootstrap
intervals, error-rate flags by category, and provider-reported model versions are reported
alongside.

## Run directory

```
experiments/parity/<run>/
  run.json          library identity, packages, cost-map hash, passes, call counts,
                    provider-reported models, socket attempts, failures
  requests.jsonl    every LLM request (canonical JSON, credentials removed), sorted
  errors.json       errored (item, criterion, judge) votes and failed items per pass
  warnings.json     warnings recorded while building, grading and scoring, with origin
  experiments/      evaluate() experiment directories (manifest.json, items.jsonl)
  metrics/          compute_metrics per pass: metrics.json, summary texts, frame (pickle, CSV)
  metrics_ckpt/     the same, recomputed from the reloaded checkpoint
  cache/            response caches (fresh and replay)
  checkpoint_compat/  loading, re-scoring and resuming another run's checkpoints
```
