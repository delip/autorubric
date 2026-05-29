# AutoRubric Development Reference

A Python library for evaluating text outputs against weighted criteria using LLM-as-a-judge.

**For detailed documentation, examples, and usage guides, see docs/.**

## Package Structure

```
src/autorubric/
├── __init__.py             # Public exports
├── dataset.py              # DataItem, RubricDataset
├── eval.py                 # EvalRunner, EvalResult, evaluate()
├── llm.py                  # LLMConfig, LLMClient, ThinkingConfig, classify_grading_error
├── prompts.py              # Centralized prompt definitions
├── rate_limit.py           # Per-model rate limiting
├── rubric.py               # Core Rubric class
├── scoring.py              # score_reports() shared weighted-scoring core
├── types.py                # Criterion, LengthPenalty, ensemble types, etc.
├── utils.py                # JSON parsing, length penalty utilities
├── graders/
│   ├── __init__.py         # Grader exports
│   ├── base.py             # Abstract Grader base class
│   └── criterion_grader.py # Unified grader (single/ensemble/few-shot)
├── meta/
│   ├── __init__.py         # Meta-rubric evaluation exports
│   ├── _evaluate.py        # evaluate_rubric_standalone, evaluate_rubric_in_context
│   ├── _improve.py         # ImprovementRunner, improve_rubric, building blocks
│   ├── _display.py         # Rich display utilities
│   └── data/               # Meta-rubric JSON files
└── metrics/
    ├── __init__.py         # compute_metrics, result types
    ├── _compute.py         # Main compute_metrics implementation
    ├── _types.py           # MetricsResult, CriterionMetrics, etc.
    ├── _helpers.py         # Verdict extraction helpers
    └── distribution.py     # EMD, KS test, bias metrics
```

## Key Types (Quick Reference)

### Core Types (src/autorubric/types.py)

| Type | Purpose |
| --- | --- |
| `Criterion` | Single criterion: weight, requirement, optional multi-choice options. Methods: `worst_scored_option()` — score-minimizing non-NA option, weight-sign aware (canonical worst case; see *Score Calculation*); `worst_option_among(indices)` — same over a candidate subset (sign-aware key + lowest-index tie-break; single source for ensemble tie-breaking; `worst_scored_option()` delegates to it); property `na_option_index` — first NA option index or `None` (safe on binary); `with_guaranteed_na_option()` — copy with a canonical NA option **appended at end** when absent (idempotent on author NA; raises on binary) — T2-A auto-injection helper shared by grader + metrics |
| `CriterionOption` | Multi-choice option with label, value (0-1), optional `na` flag |
| `CriterionVerdict` | Enum: `MET`, `UNMET`, `CANNOT_ASSESS` |
| `CriterionReport` | Criterion + verdict + reason; optional `error` (category-prefixed) + `is_error`; optional `reasoning` (T6-B; see *Grading Flow*) |
| `EvaluationReport` | Full grading result with score, raw_score, report, token_usage, cost |
| `EnsembleEvaluationReport` | Adds judge_scores, mean_agreement, per-criterion votes |
| `JudgeVote` | Single judge's verdict + reason for a criterion; optional `error` + `is_error`; optional `reasoning` (from this judge's `CriterionReport.reasoning`, T6-B) |
| `MultiChoiceJudgeVote` | Single judge's multi-choice vote (ensemble); optional `error` + `is_error` (parity with `JudgeVote`); optional `reasoning` (T6-B). `selected_index`/`selected_label` are `int | None` / `str | None` — `None` for a genuine no-option abstain (T2-B; see *Grading Flow*); same nullability on `MultiChoiceVerdict` / `AggregatedMultiChoiceVerdict` |
| `EnsembleCriterionReport` | Per-criterion aggregate of judge votes; optional `error` (set only when ALL votes errored) + `is_error` |
| `LengthPenalty` | Config: free_budget, max_cap, penalty_at_cap, exponent, penalty_type |
| `TokenUsage` | prompt_tokens, completion_tokens, total_tokens, cache stats |

### Grader Types (src/autorubric/graders/)

| Type | Purpose |
| --- | --- |
| `CriterionGrader` | Main grader - supports single LLM, ensemble, few-shot, and custom response formats |
| `JudgeSpec` | Ensemble judge config: llm_config, judge_id, weight |

Note: `FewShotConfig` is listed in the Core Types table above (defined in `src/autorubric/types.py`).

### Dataset Types (src/autorubric/dataset.py)

| Type | Purpose |
| --- | --- |
| `DataItem` | submission, description, optional ground_truth verdicts, optional per-item rubric, optional per-item prompt, optional reference_submission |
| `RubricDataset` | optional prompt, optional global rubric, items, name, optional reference_submission; methods: get_item_rubric, get_item_prompt, get_item_reference_submission, split_train_test, to/from_file |

### LLM Types (src/autorubric/llm.py)

| Type | Purpose |
| --- | --- |
| `LLMConfig` | model, temperature, max_tokens, thinking, prompt_caching, max_parallel_requests |
| `LLMClient` | Async client with generate(), caching, rate limiting |
| `ThinkingConfig` | level (LOW/MEDIUM/HIGH) or budget_tokens |
| `ErrorCategory` | `Literal["infrastructure", "parse", "unknown"]` — classification of a grading exception |
| `classify_grading_error` | `classify_grading_error(exc) -> ErrorCategory`: `infrastructure` for `openai.APIError` subclasses (litellm API/network/timeout/rate-limit/server), `parse` for `pydantic.ValidationError` / `ValueError` (incl. `json.JSONDecodeError`), `unknown` otherwise. Routing: see *Grading Flow* |

Note: `classify_grading_error` and `ErrorCategory` are in Public Exports.

### Eval Types (src/autorubric/eval.py)

| Type | Purpose |
| --- | --- |
| `EvalConfig` | fail_fast, show_progress, experiment_name, resume |
| `EvalResult` | item_results, timing_stats, token_usage, cost; method: compute_metrics() |
| `ItemResult` | item_idx, item, report, duration_seconds, error |

### Metrics Types (src/autorubric/metrics/)

| Type | Purpose |
| --- | --- |
| `MetricsResult` | accuracy, precision, recall, f1, kappa, correlations, bias, per_criterion. Top-level `criterion_precision`/`recall`/`f1` are the BINARY MET-vs-rest metric → `float | None` (`None` for a multi-choice-only rubric; multi-class P/R/F1 lives per-option). `criterion_accuracy` (binary label accuracy if binary criteria exist, else multi-choice exact-match) and `mean_kappa` also `float | None`. See *Undefined → None* |
| `CriterionMetrics` | Per-criterion binary metrics (+ optional inter-judge agreement: `krippendorff_alpha`, `fleiss_kappa`). `accuracy`/`precision`/`recall`/`f1`/`kappa` are `float | None` (`None` when no usable paired samples — counts `support_*`/`n_samples` stay `0` — or kappa undefined) |
| `OrdinalCriterionMetrics` | weighted_kappa, adjacent_accuracy, correlations, optional `krippendorff_alpha` (ordinal-aware) + `fleiss_kappa`. `exact_accuracy`/`adjacent_accuracy`/`weighted_kappa`/`rmse`/`mae` are `float | None` |
| `NominalCriterionMetrics` | kappa, per_option metrics, optional `krippendorff_alpha` + `fleiss_kappa`. `exact_accuracy`/`kappa` are `float | None`; `OptionMetrics.precision`/`recall`/`f1` likewise `float | None` |
| `NAStats` | NA diagnostics for multi-choice criteria: `na_count_true`, `na_count_pred`, `na_false_positive`, `na_false_negative`, `na_kappa` (Cohen's kappa on dichotomized {NA, not-NA}, pred vs truth) + Landis & Koch `na_kappa_interpretation`. `na_kappa`/`na_kappa_interpretation` `None` when undefined (no paired NA, single class, NaN) |
| `CannotAssessStats` | CANNOT_ASSESS diagnostics for **binary** criteria — binary parallel of `NAStats`: `ca_count_true`, `ca_count_pred`, `ca_false_positive`, `ca_false_negative`, `ca_kappa` (Cohen's kappa on dichotomized {CANNOT_ASSESS, not-CANNOT_ASSESS}, pred vs truth) + `ca_kappa_interpretation` (`None` when undefined). A **distinct kind of abstention** from NA, so a separate type — see *CANNOT_ASSESS Handling (binary metrics)* |
| `CannotAssessMode` | `Literal["exclude", "as_unmet", "as_category"]` — how binary CANNOT_ASSESS verdicts are handled at metrics time. See *CANNOT_ASSESS Handling* below |
| `NAMode` | `Literal["exclude", "as_unmet", "as_category"]` — multi-choice NA analog of `CannotAssessMode`. See *NA Handling (multi-choice metrics)* below |

### Rubric Improvement Types (src/autorubric/meta/_improve.py)

| Type | Purpose |
| --- | --- |
| `ImprovementConfig` | eval_llm, revision_llm, mode, strategy (`"meta_rubric"` or `"held_out"`), validation_data, convergence_fn, custom prompts |
| `ImprovementResult` | original/final/best rubric, iterations, best_iteration, convergence_reason, total_completion_cost |
| `ImprovementRunner` | Full-control runner class following the EvalRunner pattern |
| `ImprovementProgressDisplay` | Rich progress display: bar, issues table, rubric panel, summary table |
| `IterationResult` | Per-iteration: iteration, rubric, quality_score, agreement, per_criterion_agreement, issues, issues_fixed, issues_introduced, accepted, rejection_reason, quality_report, token_usage, completion_cost |
| `IssueDetail` | Single issue: criterion_name, requirement, weight, is_antipattern, feedback |
| `CriterionExemplar` | A single grading case for a criterion (item_index, submission_snippet, verdicts, reason, is_disagreement) |
| `CriterionErrorReport` | Per-criterion error analysis from held-out grading (accuracy, FP/FN rates, exemplars) |
| `HeldOutValidationResult` | Result from held-out validation with per-criterion diagnostics |
| `ConvergenceFn` | Custom convergence callback type alias |

### Meta-Rubric Types (src/autorubric/meta/_evaluate.py)

| Type | Purpose |
| --- | --- |
| `MetaCriterionJudgment` | Extends `CriterionJudgment` with `affected_criteria: list[int]` field for structured criterion references (internal — not exported from `meta.__init__`) |

### Meta-Rubric Functions (src/autorubric/meta/)

| Function | Purpose |
| --- | --- |
| `evaluate_rubric_standalone` | Evaluate rubric quality in isolation (clarity, structure, etc.) |
| `evaluate_rubric_in_context` | Evaluate rubric quality relative to a task prompt |
| `get_standalone_meta_rubric` | Load the standalone meta-rubric as a Rubric object |
| `get_in_context_meta_rubric` | Load the in-context meta-rubric as a Rubric object |
| `improve_rubric` | Convenience wrapper for iterative rubric improvement |
| `extract_issues` | Extract actionable issues from a meta-rubric eval report |
| `diff_issues` | Track fixed/introduced issues between iterations |
| `format_issues_for_prompt` | Format issues into text for the revision prompt |
| `format_agreement_for_prompt` | Format per-criterion agreement data as a self-contained prompt section |
| `format_ground_truth_for_prompt` | Format ground-truth validation results as a prompt section |
| `build_revision_history` | Format recent iteration history for the revision prompt |
| `validate_agreement` | Test inter-judge agreement; returns (mean, per_criterion, cost) |
| `validate_ground_truth` | Grade validation items and compute Spearman rho against expected scores |
| `compute_expected_scores` | Compute expected scores from ground-truth verdicts and rubric weights |
| `pareto_accept` | Check revision acceptance under the Pareto constraint |
| `validate_held_out` | Grade held-out items, compare per-criterion verdicts against ground truth |
| `format_held_out_for_prompt` | Format held-out validation result into revision prompt text |
| `validate_criteria_structure` | Post-revision check that criteria count/order was preserved |
| `revise_rubric` | Revise rubric via LLM; returns (Rubric, cost) |
| `revise_rubric_held_out` | Revise rubric using held-out-specific prompt templates |
| `render_improvement_report_html` | Render consolidated HTML improvement report (in `_display.py`, private — not in `__all__`) |
| `_match_issue_to_criteria` | Best-effort match of meta-rubric issues to rubric criterion indices (private) |

## Architecture Notes

### Grading Flow
1. `Rubric.grade()` delegates to grader's `grade()` method
2. `CriterionGrader` treats single LLM as "ensemble of 1". At the start of `judge()` the rubric is normalized **once** to an *effective rubric*: with `auto_na_option=True` (default) every multi-choice criterion is guaranteed an NA abstain option via `Criterion.with_guaranteed_na_option()` (T2-A — structural analog of binary CANNOT_ASSESS; see *Multi-Choice Criteria*). Normalization is pure (no RNG), same length/order (so `criterion_idx` and the shuffle RNG key stay aligned), never mutates the user's rubric. Effective options ride through `CriterionReport.options`, shared by prompt building, verdict mapping, scoring, and aggregation. `auto_na_option=False` ⇒ forced-choice (no injection); never strips an author NA option.
3. Concurrent LLM calls per criterion per judge via `asyncio.gather()`
4. Binary criteria use `binary_response_format` (default: `CriterionJudgment`); meta-rubric evals use `MetaCriterionJudgment`, which adds a structured `affected_criteria` field
5. If the response includes `affected_criteria`, grader injects `[Affects: #1, #3]` into the reason string
6. Aggregates votes using strategy (majority/weighted/unanimous/any). For binary criteria `majority` is an **unweighted head-count** (> 50% of judges), distinct from `weighted` (summed judge weights). Multi-choice criteria aggregate via two **independent** knobs — `ordinal_aggregation` and `nominal_aggregation` — that binary `aggregation` never touches (values + axis in *Multi-Choice Criteria*).
7. Returns `EnsembleEvaluationReport`

**Judge-call failure routing.** On a failed judge call the grader calls `classify_grading_error()`. `infrastructure`/`parse` failures → `CANNOT_ASSESS` (binary) or `na=True` (multi-choice), so under the default `CannotAssessStrategy.SKIP` they leave the scoring denominator and do NOT penalize the submission. Only `unknown` errors keep the conservative worst-case verdict: UNMET for positive weight, MET for negative; for multi-choice, the score-minimizing scored (non-NA) option via `Criterion.worst_scored_option()`, never an NA option. An infra/parse abstain points `na=True` at a **genuine** NA option (`criterion.na_option_index`, guaranteed by default auto-injection). With `auto_na_option=False` and no author NA option there is none to point at, so it is a **genuine no-option abstain**: `na=True` with `selected_index=None`/`selected_label=None` (T2-B) — never `na=True` against a scored option, still excluded under SKIP. The category-prefixed message (`"infrastructure: ..."`) is stored on `JudgeVote.error` (binary) / `MultiChoiceJudgeVote.error` (multi-choice). `EnsembleCriterionReport.error` is set only when EVERY contributing vote errored (mixed → genuine verdict, `error is None`), derived via the shared helper `_aggregate_error(votes)` used by both paths (per-vote `.error` is the single source). The `is_error` property on `CriterionReport`/`EnsembleCriterionReport`/`JudgeVote`/`MultiChoiceJudgeVote` distinguishes error-induced from genuine verdicts without string-matching `reason`.

**Extended-thinking trace preservation (T6-B).** `LLMClient.generate()` injects the judge's verbose deliberation trace into `judgment.reasoning` (only when thinking is enabled). The grader copies `judgment.explanation` → `CriterionReport.reason` (concise conclusion) **and** `judgment.reasoning` → `CriterionReport.reasoning` (the deliberation it was distilled from) at both binary and multi-choice success constructions, via `getattr(judgment, "reasoning", None)` (tolerates custom `binary_response_format` models lacking the field). Ensemble aggregation forwards each judge's `cr.report.reasoning` → `JudgeVote.reasoning` / `MultiChoiceJudgeVote.reasoning` (alongside the `error` copy). Error-path reports carry `reasoning=None`. There is **no** top-level `EvaluationReport`/`EnsembleEvaluationReport.reasoning` and **no** aggregated `EnsembleCriterionReport.reasoning` — concatenating per-judge traces has no clean semantics, so per-vote granularity is the carrier (unlike `error`, whose all-failed aggregation transfers).

### Score Calculation
A **single scoring core** — `score_reports(reports, config, normalize=True)` in `scoring.py` — is the source of truth for weighted-criterion scoring, shared by all three scorers: the live grader (`CriterionGrader._calculate_score_from_reports`), `Rubric.compute_score` (ground-truth / expected scores — parses verdicts into `CriterionReport`s, delegates), `RubricDataset.compute_weighted_score` (delegates to `Rubric.compute_score`). Every path builds `CriterionReport`s and runs the same core, so they agree exactly across every `CannotAssessStrategy` x {binary, multi-choice} x {+/- weight}.

```python
# Positive criteria: MET earns weight, UNMET earns 0
# Negative criteria: MET subtracts weight, UNMET contributes 0
weighted_sum = sum(report.score_value * report.weight for non-abstained reports)
#   + sum(_abstain_contribution(report, config) for abstained reports)  # NA / CANNOT_ASSESS
score = clamp(weighted_sum / total_positive_weight, 0, 1)  # if normalized
# Negative-weight-only fallback (no positive weight): 1 + weighted_sum / total_negative_weight
# Length penalty subtracted after base calculation
```

`CannotAssessStrategy` applies uniformly to binary CANNOT_ASSESS and multi-choice NA via `_abstain_contribution`: `SKIP` excludes the criterion from **both** numerator and denominator; `ZERO`/`PARTIAL`/`FAIL` keep it in the denominator with a weight-sign-aware contribution. `FAIL` uses the score-minimizing realizable outcome — `Criterion.worst_scored_option()` for multi-choice (same canonical worst case as the `unknown`-error path and metrics `na_mode="as_unmet"`), UNMET for positive weight (0) / MET for negative weight for binary. (The legacy `Rubric._apply_cannot_assess_strategy` was removed when `compute_score` was routed through the core — it had a SKIP double-subtraction bug, both excluding the abstained criterion *and* subtracting its weight from the denominator, that the core fixes.)

### Multi-Choice Criteria
- `scale_type`: "ordinal" (weighted kappa) or "nominal" (unweighted kappa)
- Options have explicit `value` (0-1) to avoid position bias
- `shuffle_options=True` (default) mitigates position bias
- NA options (`na: true`) are the structural analog of binary CANNOT_ASSESS and flow through the same `score_reports` abstain path under whichever `CannotAssessStrategy` is configured (see *Score Calculation*)
- **First-class abstain channel (T2-A; see *Grading Flow*).** The canonical NA option auto-injected by default is `CANONICAL_NA_OPTION` (`types.py`). The rendered list marks every NA option `(cannot assess / not applicable)` (`_render_options`/`_label_signals_na` in `prompts.py`, guarded against double-marking author NA labels), and the `MULTI_CHOICE_SYSTEM_PROMPT` NA section is unconditional.
- **Ensemble aggregation has three orthogonal knobs** — binary `aggregation`, `ordinal_aggregation`, `nominal_aggregation` — sharing a *central / conservative / permissive* axis (table below). Binary `unanimous` ≡ **min** over {0,1} option values, `any` ≡ **max**; ordinal analogs are `min` (lowest selected option) / `max` (highest). Nominal `unanimous` abstains on disagreement by selecting the criterion's **genuine NA option** (`na=True`, flows through SKIP) — never `na=True` against a real option (T2-B) — falling back to `mode` with a `logger.warning` only when no NA option exists.

| Concept | Binary (`aggregation`) | Ordinal (`ordinal_aggregation`) | Nominal (`nominal_aggregation`) |
| --- | --- | --- | --- |
| Central | `majority`, `weighted` | `mean`, `median`, `weighted_mean`, `mode` | `mode`, `weighted_mode` |
| Conservative | `unanimous` (≡ min over {0,1}) | `min` (lowest selected option) | `unanimous` (abstain via NA on disagreement) |
| Permissive | `any` (≡ max over {0,1}) | `max` (highest selected option) | — (unordered ⇒ no permissive analog) |

- **Tie-breaking is uniform, deterministic, and weight-sign aware (T3-B).** On any tie — binary `majority` head-count / `weighted` equal-weight; ordinal/nominal `mode` count; nominal `weighted_mode` equal-weight; ordinal `mean`/`median` snap equidistance — the **score-minimizing** outcome wins (lowest `value` for weight ≥ 0, highest for weight < 0), with **lowest option index** as final tie-break (independent of judge/vote order). Binary → UNMET for weight ≥ 0 / MET for weight < 0 (via `_binary_worst_verdict`); multi-choice routes tied candidates through `Criterion.worst_option_among` (same sign-aware key as `worst_scored_option` / `FAIL`). `min`/`max` already resolve value ties to lowest index and are untouched.

### Reproducibility & Seed Coordination
`CriterionGrader(seed=...)` controls all non-LLM randomness (option shuffling, few-shot example selection). Auto-generated when `None` so shuffles are always pinned.

- Per-call shuffle RNGs derived from `(master_seed, content_hash, criterion_idx, judge_id)` via SHA-256 — concurrency-safe (no shared mutable state). Helper: `_derive_shuffle_rng()` in `criterion_grader.py`.
- If `FewShotConfig.seed` is unset, it is coordinated from the master seed.
- Few-shot example selection RNGs are derived per `(few_shot_seed, criterion_idx, judge_id)` via the same `_derive_shuffle_rng` helper (a constant `FEW_SHOT_DOMAIN` sits in the item-key slot — few-shot examples are a fixed property of criterion+judge, not item-specific), de-correlating selected examples and their ordering across both judges and criteria. Mirrors option shuffling and supports the ensemble-independence assumption behind the inter-judge agreement metrics; selection stays fully reproducible.
- `CriterionReport.shuffle_order` records the permutation per multi-choice criterion.
- `ExperimentManifest.grader_config` persists `master_seed`, `shuffle_options`, `auto_na_option` for checkpoint reproducibility (via `_serialize_grader_config` in `eval.py`).
- With `auto_na_option=True`, a multi-choice criterion previously lacking an NA option gains one (appended at end), so its shuffle permutation spans one more index and `shuffle_order` grows by one; criteria that already had an NA option are unaffected.

### CANNOT_ASSESS Handling
Strategies (applied by the single `score_reports` core, so binary CANNOT_ASSESS and multi-choice NA are handled identically wherever a score is computed): `SKIP` (excludes the criterion from both numerator and denominator), `ZERO`, `PARTIAL` (configurable), `FAIL` (weight-sign-aware worst case). Mechanics + `FAIL` detail: see *Score Calculation*.

Judge-call failure routing (`infrastructure`/`parse` → CANNOT_ASSESS / `na=True`, excluded under SKIP; `unknown` → conservative worst case; forced-choice no-NA infra/parse abstain → `na=True` with `selected_index=None`/`selected_label=None`, T2-B) is detailed in *Grading Flow*.

### NA Handling (multi-choice metrics)
NA is the multi-choice *structural* analog of binary CANNOT_ASSESS, though a **distinct kind of abstention** (see *CANNOT_ASSESS Handling (binary metrics)*). At metrics time, `compute_metrics(..., na_mode=NAMode)` mirrors `cannot_assess: CannotAssessMode` one-for-one:

- `"exclude"` — drop pairs where either side is NA (default).
- `"as_unmet"` — remap NA → the score-minimizing non-NA option via the shared `Criterion.worst_scored_option()` (weight-sign aware, same helper as the `unknown`-error path).
- `"as_category"` — keep NA as a distinct categorical column. **Refused for ordinal criteria with an NA option** (raises `ValueError`): NA has no ordinal position, so quadratic-weighted kappa would assign it a meaningless index-based distance.

`compute_metrics` builds criteria from author-space `dataset.rubric`, but a prediction may reference an **auto-injected** NA option at index `N = len(author.options)` (out of author range) **or be a genuine no-option abstain (`selected_index=None`, T2-B)**. Per multi-choice criterion, metrics reconstructs the effective criterion via `Criterion.with_guaranteed_na_option()` **iff an out-of-range OR `None` prediction is observed**, then normalizes any `None` prediction to that effective NA index — so injected-NA predictions and forced-choice error-abstains count as NA (FP/FN, `na_kappa`, filtering) under every `na_mode`, never as option 0, while forced-choice runs without abstains gain no spurious NA column. Consequence: once an ordinal criterion gains an NA option (authored, auto-injected + predicted, or induced by a `None` abstain), `na_mode="as_category"` is refused for it.

The old `"as_worst"` literal was renamed to `"as_category"` (it kept NA as a column, did not remap it); the old value raises a clear `ValueError`. The NAStats counts (`na_count_true`, `na_count_pred`, `na_false_positive`, `na_false_negative`, `na_kappa`) are mode-independent: FP/FN counters increment before any `exclude` skip, and `na_kappa` is computed from unfiltered per-criterion pred/true.

### CANNOT_ASSESS Handling (binary metrics) and the two kinds of abstention
Binary criteria get a `MetricsResult.cannot_assess_stats: CannotAssessStats | None` block (T2-C), the structural parallel of multi-choice `na_stats: NAStats`. Populated whenever the rubric has ≥1 binary criterion (mirroring `na_stats` for ≥1 multi-choice criterion); reports the dichotomized {CANNOT_ASSESS, not-CANNOT_ASSESS} agreement (`ca_count_true`, `ca_count_pred`, `ca_false_positive`, `ca_false_negative`, `ca_kappa` — `None` when undefined). Like NAStats these counts are **mode-independent** (from raw per-criterion verdicts before any `cannot_assess` filtering) and surfaced in `summary()` only (NOT `to_dataframe()`).

**Two kinds of abstention (why two types, not one).** CANNOT_ASSESS and NA both flow through the same SKIP path and both get a parallel dichotomized-kappa stats block, but are semantically **distinct**:

- Binary **CANNOT_ASSESS** = the judge cannot determine MET-vs-UNMET — an *epistemic* abstention on a yes/no question.
- Multi-choice **NA** = "not applicable / cannot pick an applicable option" — abstaining because no scored category fits, a statement about the *option space*.

Tracked by **separate** types (`CannotAssessStats`, `ca_`-prefixed, vs. `NAStats`, `na_`-prefixed) to keep the distinction explicit while preserving the structural analogy.

### Inter-judge Agreement (Krippendorff's α + Fleiss' κ)
`compute_metrics()` reports inter-judge agreement (judges vs. each other, independent of ground truth) for binary, ordinal, nominal criteria, populated only when the report is an **ensemble with ≥2 judges and ≥2 items** — else both stats are `None`. Per-vote errors (`JudgeVote.error` / `MultiChoiceJudgeVote.error`) are excluded so only genuine judgments count.

**Krippendorff's α (`krippendorff_alpha`) is the general, recommended statistic** — handles unequal/missing raters and is **level-aware** (`level="ordinal"` for ordinal — distance-aware — vs `level="nominal"` for binary/nominal), fixing Fleiss's order-ignoring. `_compute_krippendorff_alpha()` builds a per-criterion **reliability matrix** (rows = judges in `judge_scores` order, columns = items; cell = judge's numeric code, or `np.nan` when errored/excluded/absent). Binary codes: MET=0, UNMET=1, CANNOT_ASSESS=2 (only under `as_category`; `exclude` → `np.nan`, `as_unmet` → UNMET). Multi-choice cell = `selected_index` (genuine NA included). α uses ALL items (missing handled), no complete-case dropping. Guards `<2` units / `NaN` / exceptions → `None`.

**Fleiss' κ (`fleiss_kappa`) is the classic fixed-rater nominal measure, retained complete-case.** statsmodels requires uniform raters per subject, so `_build_fleiss_row()` includes a subject ONLY if its counted votes sum to exactly `n_judges` (items with any errored/excluded/CA-under-`exclude` vote drop from the Fleiss matrix but stay in α as missing cells). `_compute_fleiss_kappa` guards `NaN` → `None`. Binary categories follow `cannot_assess` (`exclude`/`as_unmet` → 2 columns, `as_category` → 3 incl. CANNOT_ASSESS); multi-choice uses one column per option, genuine NA as an ordinary column.

`krippendorff` (numpy-only) and `statsmodels` are both **hard dependencies**; the import guards (`HAS_KRIPPENDORFF` / `HAS_STATSMODELS`) in `_compute.py` stay for safety.

Both stats appear in `MetricsResult.summary()` and `to_dataframe()` (`_types.py`). `to_dataframe()` always emits `krippendorff_alpha` / `fleiss_kappa` columns (`None`/`NaN` where N/A — aggregate/judge rows, single-judge criteria). `summary()` appends `Kripp-α` (recommended, leads) + `Fleiss` columns to a criterion-type table only when ≥1 criterion in that group has a non-`None` value (single-judge output unchanged); per-criterion `None` renders as `n/a`.

**Framework principle: prediction-vs-ground-truth categorical agreement is reported as Cohen's kappa across the board** — binary `CriterionMetrics.kappa` (unweighted, possibly 3-class under `cannot_assess="as_category"`), ordinal `OrdinalCriterionMetrics.weighted_kappa` (quadratic-weighted), nominal `NominalCriterionMetrics.kappa` (unweighted); the orthogonal abstain decision uses `NAStats.na_kappa` ({NA, not-NA}) and the binary parallel `CannotAssessStats.ca_kappa` ({CANNOT_ASSESS, not-CANNOT_ASSESS}). All are chance-corrected and Landis & Koch interpretable via `KappaResult.interpret_kappa`. **Distinct** from inter-judge agreement (above), which uses `krippendorff_alpha` + `fleiss_kappa`. New pred-vs-truth categorical agreement metrics should land kappa-shaped, not as ad-hoc proportion metrics.

### Aggregate / per-judge scalar metrics: None when undefined, and type-handling parity (T8-B/T8-C)
**A metric is `None` when genuinely undefined / not-applicable, never a fake `0.0`** (see *Undefined → None*). Aggregate scalars `criterion_precision`/`recall`/`f1` are the **binary MET-vs-rest** metric → `None` for a multi-choice-only rubric (no MET class; multi-class P/R/F1 lives in each criterion's `per_option`). `criterion_accuracy` generalizes (binary label accuracy when ≥1 binary criterion exists, else multi-choice exact-match), `mean_kappa` is the mean of per-criterion kappas — both `None` only with no comparable pairs / no kappa contributed. Aggregate + per-judge paths share one helper **`_criterion_level_scalars(...)`** (`_compute.py`; binary label/MET flats, falls back to multi-choice exact-match with no binary criterion); `_mean_or_none(values)` is the None-safe mean. The constructor wraps each scalar `x if x is None else float(x)`.

**Per-judge metrics mirror the aggregate's type handling field-for-field (T8-B).** `_compute_judge_metrics` collects binary `cr.votes` and multi-choice `cr.multi_choice_votes` (records each judge's `selected_index` per item, NA-normalized via the SAME `effective_criteria` as the aggregate, captures the matching vote's `error` so errored MC votes skip with binary parity). It builds each judge's per-criterion kappas by reusing the aggregate's exact construction — binary `cohen_kappa_score` on prepared label reps (appended only when label pairs exist), ordinal `_compute_ordinal_criterion_metrics(...).weighted_kappa`, nominal `_compute_nominal_criterion_metrics(...).kappa` (appended after `filter_na_multi_choice`) — then calls `_criterion_level_scalars`. **Invariant: a 1-judge "ensemble" yields `per_judge[only]` equal to the aggregate field-for-field** (accuracy, mean_kappa equal; P/R/F1 both-None-or-equal). `JudgeMetrics` widened its five scalar fields to `float | None`.

`MetricsResult.summary()` renders Accuracy / Mean Kappa (and binary-only P/R/F1) through the module-level None-safe formatter **`_fmt_opt(value, spec, width=0)`** (`_types.py`), which renders `None` as right-aligned `'n/a'` and is shared with `_agreement_cells`. `to_dataframe()` stores raw values, tolerating `None` (emits JSON null / NaN).

### "Undefined → `None`, never a fabricated `0.0`" — the cross-cutting invariant
A single principle across grading → metrics → improvement loop (T8-B/T8-C are one instance): **a metric / statistic / score / coefficient is `None` when genuinely undefined (no data, degenerate input, computation failed, or not-applicable), never a stand-in `0.0`/`1.0`.** Counts (`n_samples`, `support_*`, confusion cells) stay `0`; aggregates average only defined values (`_mean_or_none`) and are `None` if none defined; renderers show `None` as `n/a` and never crash. Modeled on the pre-existing `na_kappa`/`ca_kappa`/`BiasResult.{p_value,effect_size,ci}` handling. Applies to:

- **Kappa (Cohen's, all paths).** `_kappa_or_none(y1, y2, *, weights=None)` (`_compute.py`) is the single helper for every pred-vs-truth Cohen's kappa (binary per-criterion, ordinal `weights="quadratic"`, nominal, per-judge). Returns `None` on exception **or a NaN result** (degenerate single-class data makes `cohen_kappa_score` return NaN with no exception; the old `except: 0.0` let it leak). `None` kappas are excluded from `mean_kappa` via `_mean_or_none`.
- **Empty-data criteria.** With zero usable paired samples, a criterion's metric values (`accuracy`/`precision`/`recall`/`f1`/`kappa`, ordinal `exact_accuracy`/`adjacent_accuracy`/`weighted_kappa`/`rmse`/`mae`, nominal `exact_accuracy`/`kappa`, `OptionMetrics.precision/recall/f1`) are `None`; only counts stay `0`. `_compute_adjacent_accuracy([])` returns `None`.
- **Correlations.** `_compute_correlation` returns `CorrelationResult.coefficient = None` (and `p_value = None`) for `<3` samples (`interpretation="insufficient data"`) or a constant array (NaN → `interpretation="undefined"`); `_interpret_correlation` is only called with a real float. `validate_ground_truth`'s Spearman likewise yields `None` on a constant rubric-score array. Since the constant-array case is handled, scipy's spurious `ConstantInputWarning`/`NearConstantInputWarning` is suppressed locally around the `spearmanr`/`kendalltau`/`pearsonr` calls.
- **Distribution (`distribution.py`).** `score_distribution` returns `None` for stats undefined at the sample size (`std`/`variance` at n<2, `skewness` at n<3, `kurtosis` at n<4, all central-tendency at n=0); a single point keeps its computable `mean/min/max/median/iqr`. `earth_movers_distance` on empty input returns `None` fields.
- **Bias (`distribution.py`).** `systematic_bias` computes `mean_bias` at n=1 (the single difference) with `std_bias=None`; at n=0 both `None`; `effect_size` is `None` when `std_bias == 0` (Cohen's d undefined). `BiasResult.{mean_bias,std_bias}` are `float | None`.
- **Inter-judge agreement.** `validate_agreement` returns `None` (not `0.0`) when no usable sample was measured; the empty-rubric grader path sets `EnsembleEvaluationReport.mean_agreement = None` (`float | None`, default `None`); improvement-loop consumers (`pareto_accept`, `_check_convergence`, best-tracking, formatting) treat `None` as "not measured".
- **Bootstrap.** `BootstrapResults.{accuracy_ci,kappa_ci,rmse_ci}` are `tuple[float, float] | None` — `None` when the bootstrap sample is empty; `summary()` skips a `None` CI line.
- **Grade-failure score.** `EvaluationReport.score`/`raw_score` and `EnsembleEvaluationReport.score`/`raw_score`/`llm_raw_score` are `float | None`; `None` emitted **only** on the explicit failure constructors (`EvalRunner._create_error_report`, the grader's "No judge results to aggregate" report). The normal path always computes a real float, and the scoring core never reads `report.score`. Consumers filter `error is not None` first or are `None`-guarded: `EvalResult.get_scores()` filters `score is not None`; the length-penalty path (`graders/base.py`) returns an errored report unchanged; `compute_metrics` **excludes** an errored/`None`-score report from score-level RMSE/correlation/bias arrays (no `0.0` substitution); `meta/_display` renders `None` as `n/a`; `validate_ground_truth` and the improvement loop raise `RuntimeError` on a *structural* grade failure (no score). Serialization round-trips `None` cleanly (writers coercion-free; readers index/`.get` without `float()`/`0.0` defaults).

### Improvement Loop Artifact Persistence
When `save_artifacts=True` and `artifacts_dir` is set, the improvement loop writes:
- `rubric-iter-{NN}.json` — criteria array per iteration
- `eval-iter-{NN}.html` — meta-rubric eval report (always generated, regardless of `display`)
- `iter-{NN}.json` — rich per-iteration JSON (quality report, issues, validation samples, revision prompts/response)
- `improvement_report.html` — consolidated report (always generated, regardless of `display`)
- `summary.json` — full run metadata, config snapshot, per-iteration summary

`revise_rubric()`, `validate_agreement()`, `validate_ground_truth()` accept a private `_capture` parameter for artifact collection.

## Key Conventions

- All graders return `EnsembleEvaluationReport` for consistent interface
- `raw_score` always populated regardless of `normalize` setting
- Judge-call failures classified via `classify_grading_error()`: `infrastructure`/`parse` → CANNOT_ASSESS / `na=True` (excluded under default SKIP); `unknown` → conservative worst case (forced-choice no-NA infra/parse abstain → `na=True`, `selected_index=None`/`selected_label=None`, T2-B). Full routing: see *Grading Flow*
- **Report serialization is uniformly pydantic-native (T6-D).** `EnsembleCriterionReport` / `JudgeVote` / `MultiChoiceJudgeVote` are **frozen pydantic models** (not dataclasses), so the ensemble checkpoint path collapses onto `model_dump(mode="json")` / `model_validate` exactly like the single-report `CriterionReport` path — no hand-rolled per-field plumbing. `ItemResult.to_dict` keeps its envelope (`report_type` discriminator, `criterion_reports`, `judge_scores`, nested `token_usage`); `_serialize_ensemble_criterion_report` is now a thin `ecr.model_dump(mode="json")` wrapper (shared with the meta improvement-loop artifacts), and `_deserialize_ensemble_report` rebuilds only the envelope around `[EnsembleCriterionReport.model_validate(...)]`. Adding a field to any report/vote type now round-trips for free. `EnsembleCriterionReport.agreement` is computed by a `model_validator(mode="after")` (via `object.__setattr__`, frozen-safe) — the prior `__post_init__` semantics, idempotent on reload.
- `JudgeVote.error` / `MultiChoiceJudgeVote.error` / `EnsembleCriterionReport.error` carry category-prefixed messages; serialization round-trips `error` on ensemble reports, binary judge votes, and multi-choice judge votes — automatically, via the pydantic dump/validate above
- `CriterionReport.reasoning` / `JudgeVote.reasoning` / `MultiChoiceJudgeVote.reasoning` carry the extended-thinking deliberation trace (T6-B; see *Grading Flow*); serialization round-trips it symmetrically with `error` via pydantic `model_dump`/`model_validate` on **both** the single-report and ensemble paths (T6-D), with field defaults tolerating legacy checkpoints (missing `reasoning`/`error`/`weight`/`na`/`votes`/`agreement` → field default; a missing/0.0 `agreement` recomputes from the votes)
- Filter `error is not None` results in training pipelines
- Rate limiting via `LLMConfig.max_parallel_requests` (per-provider semaphore)
- Multi-choice criteria get a guaranteed abstain option by default (`CriterionGrader(auto_na_option=True)`, auto-injected NA — binary-CANNOT_ASSESS parity, T2-A; see *Multi-Choice Criteria*); set `auto_na_option=False` for forced-choice. Never strips an author NA option

## Public Exports

See `src/autorubric/__init__.py` for complete list.
