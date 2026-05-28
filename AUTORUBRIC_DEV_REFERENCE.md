# AutoRubric Development Reference

A Python library for evaluating text outputs against weighted criteria using LLM-as-a-judge.

**For detailed documentation, examples, and usage guides, see docs/.**

## Package Structure

```
src/autorubric/
├── __init__.py              # Public exports
├── dataset.py               # DataItem, RubricDataset
├── eval.py                  # EvalRunner, EvalResult, evaluate()
├── llm.py                   # LLMConfig, LLMClient, ThinkingConfig, classify_grading_error
├── prompts.py               # Centralized prompt definitions
├── rate_limit.py            # Per-model rate limiting
├── rubric.py                # Core Rubric class
├── types.py                 # Criterion, LengthPenalty, ensemble types, etc.
├── utils.py                 # JSON parsing, length penalty utilities
├── graders/
│   ├── __init__.py          # Grader exports
│   ├── base.py              # Abstract Grader base class
│   └── criterion_grader.py  # Unified grader (single/ensemble/few-shot)
├── meta/
│   ├── __init__.py          # Meta-rubric evaluation exports
│   ├── _evaluate.py         # evaluate_rubric_standalone, evaluate_rubric_in_context
│   ├── _improve.py          # ImprovementRunner, improve_rubric, building blocks
│   ├── _display.py          # Rich display utilities
│   └── data/                # Meta-rubric JSON files
└── metrics/
    ├── __init__.py          # compute_metrics, result types
    ├── _compute.py          # Main compute_metrics implementation
    ├── _types.py            # MetricsResult, CriterionMetrics, etc.
    ├── _helpers.py          # Verdict extraction helpers
    └── distribution.py      # EMD, KS test, bias metrics
```

## Key Types (Quick Reference)

### Core Types (src/autorubric/types.py)

| Type                       | Purpose                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `Criterion`                | Single evaluation criterion with weight, requirement, optional multi-choice options. Method `worst_scored_option()` returns the score-minimizing non-NA option, weight-sign aware — shared by the grader's `unknown`-error worst-case path and the metrics' `na_mode="as_unmet"` remap |
| `CriterionOption`          | Multi-choice option with label, value (0-1), optional `na` flag                     |
| `CriterionVerdict`         | Enum: `MET`, `UNMET`, `CANNOT_ASSESS`                                               |
| `CriterionReport`          | Criterion + verdict + reason; optional `error` (category-prefixed message) + `is_error` property |
| `EvaluationReport`         | Full grading result with score, raw_score, report, token_usage, cost                |
| `EnsembleEvaluationReport` | Adds judge_scores, mean_agreement, per-criterion votes                              |
| `JudgeVote`                | Single judge's verdict + reason for a criterion; optional `error` (category-prefixed message) |
| `MultiChoiceJudgeVote`     | Single judge's multi-choice vote (ensemble); optional `error` (category-prefixed message, parity with `JudgeVote`) |
| `EnsembleCriterionReport`  | Per-criterion aggregate of judge votes; optional `error` (set only when ALL votes errored) + `is_error` property |
| `LengthPenalty`            | Config: free_budget, max_cap, penalty_at_cap, exponent, penalty_type                |
| `TokenUsage`               | prompt_tokens, completion_tokens, total_tokens, cache stats                         |

### Grader Types (src/autorubric/graders/)

| Type              | Purpose                                                                              |
| ----------------- | ------------------------------------------------------------------------------------ |
| `CriterionGrader` | Main grader - supports single LLM, ensemble, few-shot, and custom response formats   |
| `JudgeSpec`       | Ensemble judge config: llm_config, judge_id, weight                                  |

Note: `FewShotConfig` is listed in the Core Types table above (defined in `src/autorubric/types.py`).

### Dataset Types (src/autorubric/dataset.py)

| Type            | Purpose                                                                                                                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DataItem`      | submission, description, optional ground_truth verdicts, optional per-item rubric, optional per-item prompt, optional reference_submission                                                    |
| `RubricDataset` | optional prompt, optional global rubric, items, name, optional reference_submission; methods: get_item_rubric, get_item_prompt, get_item_reference_submission, split_train_test, to/from_file |

### LLM Types (src/autorubric/llm.py)

| Type             | Purpose                                                                         |
| ---------------- | ------------------------------------------------------------------------------- |
| `LLMConfig`      | model, temperature, max_tokens, thinking, prompt_caching, max_parallel_requests |
| `LLMClient`      | Async client with generate(), caching, rate limiting                            |
| `ThinkingConfig` | level (LOW/MEDIUM/HIGH) or budget_tokens                                        |
| `ErrorCategory`  | `Literal["infrastructure", "parse", "unknown"]` — classification of a grading exception |
| `classify_grading_error` | `classify_grading_error(exc) -> ErrorCategory`: `infrastructure` for `openai.APIError` subclasses (litellm API/network/timeout/rate-limit/server errors), `parse` for `pydantic.ValidationError` / `ValueError` (incl. `json.JSONDecodeError`), `unknown` otherwise |

Note: `classify_grading_error` and `ErrorCategory` are in Public Exports.

### Eval Types (src/autorubric/eval.py)

| Type         | Purpose                                                                  |
| ------------ | ------------------------------------------------------------------------ |
| `EvalConfig` | fail_fast, show_progress, experiment_name, resume                        |
| `EvalResult` | item_results, timing_stats, token_usage, cost; method: compute_metrics() |
| `ItemResult` | item_idx, item, report, duration_seconds, error                          |

### Metrics Types (src/autorubric/metrics/)

| Type                      | Purpose                                                                   |
| ------------------------- | ------------------------------------------------------------------------- |
| `MetricsResult`           | accuracy, precision, recall, f1, kappa, correlations, bias, per_criterion |
| `CriterionMetrics`        | Per-criterion binary metrics (incl. optional inter-judge agreement: `krippendorff_alpha` recommended, `fleiss_kappa` complete-case) |
| `OrdinalCriterionMetrics` | weighted_kappa, adjacent_accuracy, correlations, optional `krippendorff_alpha` (ordinal-aware, recommended) + `fleiss_kappa` |
| `NominalCriterionMetrics` | kappa, per_option metrics, optional `krippendorff_alpha` (recommended) + `fleiss_kappa` |
| `NAStats`                 | NA-handling diagnostics for multi-choice criteria: `na_count_true`, `na_count_pred`, `na_false_positive`, `na_false_negative`, plus `na_kappa` (Cohen's kappa on the dichotomized {NA, not-NA} decision, pred vs truth) and its Landis & Koch `na_kappa_interpretation`. `na_kappa` / `na_kappa_interpretation` are `None` when the dichotomy is undefined (no paired NA observations, single class, NaN). |
| `CannotAssessMode`        | `Literal["exclude", "as_unmet", "as_category"]` — how binary CANNOT_ASSESS verdicts are handled at metrics time. See *CANNOT_ASSESS Handling* below. |
| `NAMode`                  | `Literal["exclude", "as_unmet", "as_category"]` — multi-choice NA analog of `CannotAssessMode`. See *NA Handling (multi-choice metrics)* below. |

### Rubric Improvement Types (src/autorubric/meta/_improve.py)

| Type                          | Purpose                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `ImprovementConfig`           | Configuration: eval_llm, revision_llm, mode, strategy (`"meta_rubric"` or `"held_out"`), validation_data, convergence_fn, custom prompts |
| `ImprovementResult`           | Final result: original/final/best rubric, iterations, best_iteration, convergence_reason, total_completion_cost |
| `ImprovementRunner`           | Full-control runner class following the EvalRunner pattern                            |
| `ImprovementProgressDisplay`  | Rich-based progress display with bar, issues table, rubric panel, and summary table  |
| `IterationResult`             | Per-iteration: iteration, rubric, quality_score, agreement, per_criterion_agreement, issues, issues_fixed, issues_introduced, accepted, rejection_reason, quality_report, token_usage, completion_cost |
| `IssueDetail`                 | Single issue: criterion_name, requirement, weight, is_antipattern, feedback           |
| `CriterionExemplar`           | A single grading case for a criterion (item_index, submission_snippet, verdicts, reason, is_disagreement) |
| `CriterionErrorReport`        | Per-criterion error analysis from held-out grading (accuracy, FP/FN rates, exemplars) |
| `HeldOutValidationResult`     | Result from held-out validation with per-criterion diagnostics                       |
| `ConvergenceFn`               | Custom convergence callback type alias                                               |

### Meta-Rubric Types (src/autorubric/meta/_evaluate.py)

| Type                       | Purpose                                                                    |
| -------------------------- | -------------------------------------------------------------------------- |
| `MetaCriterionJudgment`    | Extends `CriterionJudgment` with `affected_criteria: list[int]` field for structured criterion references (internal — not exported from `meta.__init__`) |

### Meta-Rubric Functions (src/autorubric/meta/)

| Function                       | Purpose                                                         |
| ------------------------------ | --------------------------------------------------------------- |
| `evaluate_rubric_standalone`   | Evaluate rubric quality in isolation (clarity, structure, etc.) |
| `evaluate_rubric_in_context`   | Evaluate rubric quality relative to a task prompt               |
| `get_standalone_meta_rubric`   | Load the standalone meta-rubric as a Rubric object              |
| `get_in_context_meta_rubric`   | Load the in-context meta-rubric as a Rubric object              |
| `improve_rubric`               | Convenience wrapper for iterative rubric improvement            |
| `extract_issues`               | Extract actionable issues from a meta-rubric eval report        |
| `diff_issues`                  | Track fixed/introduced issues between iterations                |
| `format_issues_for_prompt`     | Format issues into text for the revision prompt                 |
| `format_agreement_for_prompt`  | Format per-criterion agreement data as a self-contained prompt section |
| `format_ground_truth_for_prompt` | Format ground-truth validation results as a prompt section    |
| `build_revision_history`       | Format recent iteration history for the revision prompt         |
| `validate_agreement`           | Test inter-judge agreement; returns (mean, per_criterion, cost) |
| `validate_ground_truth`        | Grade validation items and compute Spearman rho against expected scores |
| `compute_expected_scores`      | Compute expected scores from ground-truth verdicts and rubric weights |
| `pareto_accept`                | Check revision acceptance under the Pareto constraint           |
| `validate_held_out`            | Grade held-out items, compare per-criterion verdicts against ground truth |
| `format_held_out_for_prompt`   | Format held-out validation result into revision prompt text     |
| `validate_criteria_structure`  | Post-revision check that criteria count/order was preserved     |
| `revise_rubric`                | Revise rubric via LLM; returns (Rubric, cost)                   |
| `revise_rubric_held_out`       | Revise rubric using held-out-specific prompt templates          |
| `render_improvement_report_html` | Render consolidated HTML improvement report (in `_display.py`, private — not in `__all__`) |
| `_match_issue_to_criteria`       | Best-effort match of meta-rubric issues to rubric criterion indices (private) |

## Architecture Notes

### Grading Flow
1. `Rubric.grade()` delegates to grader's `grade()` method
2. `CriterionGrader` treats single LLM as "ensemble of 1"
3. Makes concurrent LLM calls per criterion per judge via `asyncio.gather()`
4. Binary criteria use `binary_response_format` (default: `CriterionJudgment`); meta-rubric evals use `MetaCriterionJudgment` which adds structured `affected_criteria` field
5. If the response includes `affected_criteria`, grader injects `[Affects: #1, #3]` tag into the reason string
6. Aggregates votes using strategy (majority/weighted/unanimous/any). For binary criteria, `majority` is an **unweighted head-count** (> 50% of judges, ties → UNMET), distinct from `weighted` (decides by summed judge weights).
7. Returns `EnsembleEvaluationReport` (consistent interface)

When a judge call fails, the grader calls `classify_grading_error()` on the exception. `infrastructure` and `parse` failures are routed to `CANNOT_ASSESS` (binary) or `na=True` (multi-choice) — so under the default `CannotAssessStrategy.SKIP` they are excluded from the scoring denominator and do NOT penalize the submission. Only `unknown` errors keep the previous conservative worst-case verdict (UNMET for positive weight, MET for negative). For multi-choice criteria, an `unknown` error selects the score-minimizing scored (non-NA) option — lowest `value` for non-negative weight, highest `value` for negative weight (mirroring the binary worst case) — and never auto-selects an NA option (NA/skip is reserved for infrastructure/parse). The failure message (category-prefixed, e.g. `"infrastructure: ..."`) is stored on `JudgeVote.error` (binary) and `MultiChoiceJudgeVote.error` (multi-choice — same parity). `EnsembleCriterionReport.error` is set only when EVERY contributing judge vote errored; a mix of failed + successful judges yields a genuine verdict with `error is None`. The `is_error` property on `CriterionReport` / `EnsembleCriterionReport` lets downstream code distinguish error-induced verdicts from genuine ones without string-matching `reason`.

### Score Calculation
```python
# Positive criteria: MET earns weight, UNMET earns 0
# Negative criteria: MET subtracts weight, UNMET contributes 0
weighted_sum = sum(verdict_value * criterion.weight for each criterion)
score = clamp(weighted_sum / total_positive_weight, 0, 1)  # if normalized
# Length penalty subtracted after base calculation
```

### Multi-Choice Criteria
- `scale_type`: "ordinal" (weighted kappa) or "nominal" (unweighted kappa)
- Options have explicit `value` (0-1) to avoid position bias
- `shuffle_options=True` (default) mitigates position bias
- NA options (`na: true`) excluded from scoring like CANNOT_ASSESS

### Reproducibility & Seed Coordination
`CriterionGrader(seed=...)` controls all non-LLM randomness (option shuffling, few-shot example selection). Auto-generated when `None` so shuffles are always pinned.

- Per-call shuffle RNGs are derived from `(master_seed, content_hash, criterion_idx, judge_id)` via SHA-256. This is concurrency-safe (no shared mutable state).
- If `FewShotConfig.seed` is unset, it is coordinated from the master seed.
- `CriterionReport.shuffle_order` records the permutation used for each multi-choice criterion.
- `ExperimentManifest.grader_config` persists `master_seed` and `shuffle_options` for checkpoint reproducibility.
- Helper: `_derive_shuffle_rng()` in `criterion_grader.py`.

### CANNOT_ASSESS Handling
Strategies: `SKIP` (adjust denominator), `ZERO`, `PARTIAL` (configurable), `FAIL` (worst case)

Judge-call failures classified as `infrastructure` or `parse` (see Grading Flow / `classify_grading_error`) are mapped to `CANNOT_ASSESS` (binary) or `na=True` (multi-choice), so under the default `SKIP` strategy they drop out of the denominator instead of penalizing the submission. Only `unknown` errors fall back to the conservative worst-case verdict. For multi-choice criteria, an `unknown` error selects the score-minimizing scored (non-NA) option via the shared `Criterion.worst_scored_option()` helper — lowest `value` for non-negative weight, highest `value` for negative weight (mirroring the binary worst case) — and never auto-selects an NA option.

### NA Handling (multi-choice metrics)
NA is the multi-choice structural analog of binary CANNOT_ASSESS. At metrics time, `compute_metrics(..., na_mode=NAMode)` mirrors `cannot_assess: CannotAssessMode` one-for-one:

- `"exclude"` — drop pairs where either side is NA (default).
- `"as_unmet"` — remap NA → the score-minimizing non-NA option via the shared `Criterion.worst_scored_option()` (same helper as the grader's `unknown`-error path, so the two layers cannot drift). Weight-sign aware: lowest `value` for non-negative weight, highest `value` for negative weight.
- `"as_category"` — keep NA as a distinct categorical column. **Refused for ordinal criteria with an NA option** (raises `ValueError`): NA has no ordinal position, so quadratic-weighted Cohen's kappa would assign NA a geometrically meaningless distance based on its index.

The old `"as_worst"` literal was renamed to `"as_category"` (it described the wrong thing — the code kept NA as a column, it did not remap NA to a worst option); passing the old value raises a clear `ValueError`. The NAStats diagnostic counts (`na_count_true`, `na_count_pred`, `na_false_positive`, `na_false_negative`, `na_kappa`) are mode-independent: the FP/FN counters increment before any `exclude` skip, and `na_kappa` is computed from unfiltered per-criterion pred/true.

### Inter-judge Agreement (Krippendorff's α + Fleiss' κ)
`compute_metrics()` reports inter-judge agreement (judges vs. each other, independent of ground truth) for binary, ordinal, and nominal criteria, populated only when the report is an **ensemble with ≥2 judges and ≥2 items** — otherwise both stats are `None`. Per-vote errors (`JudgeVote.error` / `MultiChoiceJudgeVote.error`) are excluded so only genuine judgments count.

**Krippendorff's α (`krippendorff_alpha`) is the general, recommended statistic.** It natively handles unequal/missing raters and is **level-aware** (`level="ordinal"` for ordinal criteria — distance-aware — vs `level="nominal"` for binary/nominal), fixing the latent issue that Fleiss ignores ordering. `_compute_krippendorff_alpha()` builds a per-criterion **reliability matrix** (rows = judges in `judge_scores` order, columns = items; cell = the judge's numeric code, or `np.nan` when errored/excluded/absent). Binary codes: MET=0, UNMET=1, CANNOT_ASSESS=2 (only under `as_category`; under `exclude` → `np.nan`, under `as_unmet` → coded as UNMET). Multi-choice cell = `selected_index` (genuine NA included). α uses ALL items (missing handled) — no complete-case dropping. Guards `<2` units / `NaN` / exceptions → `None`.

**Fleiss' κ (`fleiss_kappa`) is the classic fixed-rater nominal measure, retained complete-case.** statsmodels requires uniform raters per subject, so `_build_fleiss_row()` includes a subject ONLY if its counted votes sum to exactly `n_judges` (items with any errored/excluded/CA-under-`exclude` vote are dropped from the Fleiss matrix but remain in α as missing cells). `_compute_fleiss_kappa` guards `NaN` → `None`. Binary categories follow `cannot_assess` (`exclude`/`as_unmet` → 2 columns, `as_category` → 3 incl. CANNOT_ASSESS); multi-choice uses one column per option with genuine NA as an ordinary column.

`krippendorff` (numpy-only) and `statsmodels` are both **hard dependencies**; the graceful import guards (`HAS_KRIPPENDORFF` / `HAS_STATSMODELS`) in `_compute.py` stay for safety.

Both stats are surfaced in `MetricsResult.summary()` and `MetricsResult.to_dataframe()` (in `_types.py`). `to_dataframe()` always emits `krippendorff_alpha` / `fleiss_kappa` columns (`None`/`NaN` where not applicable — aggregate/judge rows, single-judge criteria). `summary()` appends `Kripp-α` (recommended, leads) and `Fleiss` columns to a criterion-type table only when at least one criterion in that group has a non-`None` value, so single-judge output is unchanged; per-criterion `None` renders as `n/a`.

**Framework principle: prediction-vs-ground-truth categorical agreement is reported as Cohen's kappa across the board.** Binary criteria use `CriterionMetrics.kappa` (unweighted Cohen's κ, possibly 3-class under `cannot_assess="as_category"`); ordinal multi-choice uses `OrdinalCriterionMetrics.weighted_kappa` (quadratic-weighted Cohen's κ); nominal multi-choice uses `NominalCriterionMetrics.kappa` (unweighted Cohen's κ); the orthogonal abstain decision uses `NAStats.na_kappa` (Cohen's κ on the {NA, not-NA} dichotomy). All are chance-corrected and Landis & Koch interpretable via `KappaResult.interpret_kappa`. This is **distinct** from inter-judge agreement (judges vs. each other), which uses `krippendorff_alpha` + `fleiss_kappa` per the section above. Any new pred-vs-truth categorical agreement metric should land kappa-shaped; do not introduce ad-hoc proportion metrics for paired categorical agreement.

### Improvement Loop Artifact Persistence
When `save_artifacts=True` and `artifacts_dir` is set, the improvement loop writes:
- `rubric-iter-{NN}.json` — criteria array per iteration
- `eval-iter-{NN}.html` — meta-rubric eval report (always generated, regardless of `display` setting)
- `iter-{NN}.json` — rich per-iteration JSON (quality report, issues, validation samples, revision prompts/response)
- `improvement_report.html` — consolidated report (always generated, regardless of `display` setting)
- `summary.json` — full run metadata, config snapshot, and per-iteration summary

`revise_rubric()`, `validate_agreement()`, and `validate_ground_truth()` accept a private `_capture` parameter for artifact collection.

## Key Conventions

- All graders return `EnsembleEvaluationReport` for consistent interface
- `raw_score` always populated regardless of `normalize` setting
- Judge-call failures are classified via `classify_grading_error()`: `infrastructure`/`parse` → CANNOT_ASSESS / `na=True` (excluded under default SKIP); only `unknown` keeps conservative defaults (UNMET for positive, MET for negative weights)
- `JudgeVote.error` / `MultiChoiceJudgeVote.error` / `EnsembleCriterionReport.error` carry category-prefixed messages; `eval.py` serialization round-trips the `error` field on ensemble reports, binary judge votes, and multi-choice judge votes
- Filter `error is not None` results in training pipelines
- Rate limiting via `LLMConfig.max_parallel_requests` (per-provider semaphore)

## Public Exports

See `src/autorubric/__init__.py` for complete list.
