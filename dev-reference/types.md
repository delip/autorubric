[← Dev Reference Index](index.md)

## Key Types (Quick Reference)

### Core Types (src/autorubric/types.py)

| Type | Purpose |
| --- | --- |
| `Criterion` | Single criterion: weight, requirement, optional multi-choice options. Methods: `worst_scored_option()` — score-minimizing non-NA option, weight-sign aware (canonical worst case; see [Score Calculation](grading-flow.md)); `worst_option_among(indices)` — same over a candidate subset (sign-aware key + lowest-index tie-break; single source for ensemble tie-breaking; `worst_scored_option()` delegates to it); property `na_option_index` — first NA option index or `None` (safe on binary); `with_guaranteed_na_option()` — copy with a canonical NA option **appended at end** when absent (idempotent on author NA; raises on binary) — auto-injection helper shared by grader + metrics |
| `CriterionOption` | Multi-choice option with label, value (0-1), optional `na` flag |
| `CriterionVerdict` | Enum: `MET`, `UNMET`, `CANNOT_ASSESS` |
| `CriterionReport` | Criterion + verdict + reason; optional `error` (category-prefixed) + `is_error`; optional `reasoning` (see [Grading Flow](grading-flow.md)) |
| `EvaluationReport` | Full grading result with score, raw_score, report, token_usage, cost |
| `EnsembleEvaluationReport` | Adds judge_scores, mean_agreement, per-criterion votes |
| `JudgeVote` | Single judge's verdict + reason for a criterion; optional `error` + `is_error`; optional `reasoning` (from this judge's `CriterionReport.reasoning`) |
| `MultiChoiceJudgeVote` | Single judge's multi-choice vote (ensemble); optional `error` + `is_error` (parity with `JudgeVote`); optional `reasoning`. `selected_index`/`selected_label` are `int | None` / `str | None` — `None` for a genuine no-option abstain (see [Grading Flow](grading-flow.md)); same nullability on `MultiChoiceVerdict` / `AggregatedMultiChoiceVerdict` |
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
| `classify_grading_error` | `classify_grading_error(exc) -> ErrorCategory`: `infrastructure` for `openai.APIError` subclasses (litellm API/network/timeout/rate-limit/server), `parse` for `pydantic.ValidationError` / `ValueError` (incl. `json.JSONDecodeError`), `unknown` otherwise. Routing: see [Grading Flow](grading-flow.md) |

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
| `MetricsResult` | accuracy, precision, recall, f1, kappa, correlations, bias, per_criterion. Top-level `criterion_precision`/`recall`/`f1` are the BINARY MET-vs-rest metric → `float | None` (`None` for a multi-choice-only rubric; multi-class P/R/F1 lives per-option). `criterion_accuracy` (binary label accuracy if binary criteria exist, else multi-choice exact-match) and `mean_kappa` also `float | None`. See [Undefined → None](metrics.md) |
| `CriterionMetrics` | Per-criterion binary metrics (+ optional inter-judge agreement: `krippendorff_alpha`, `fleiss_kappa`). `accuracy`/`precision`/`recall`/`f1`/`kappa` are `float | None` (`None` when no usable paired samples — counts `support_*`/`n_samples` stay `0` — or kappa undefined) |
| `OrdinalCriterionMetrics` | weighted_kappa, adjacent_accuracy, correlations, optional `krippendorff_alpha` (ordinal-aware) + `fleiss_kappa`. `exact_accuracy`/`adjacent_accuracy`/`weighted_kappa`/`rmse`/`mae` are `float | None` |
| `NominalCriterionMetrics` | kappa, per_option metrics, optional `krippendorff_alpha` + `fleiss_kappa`. `exact_accuracy`/`kappa` are `float | None`; `OptionMetrics.precision`/`recall`/`f1` likewise `float | None` |
| `NAStats` | NA diagnostics for multi-choice criteria: `na_count_true`, `na_count_pred`, `na_false_positive`, `na_false_negative`, `na_kappa` (Cohen's kappa on dichotomized {NA, not-NA}, pred vs truth) + Landis & Koch `na_kappa_interpretation`. `na_kappa`/`na_kappa_interpretation` `None` when undefined (no paired NA, single class, NaN) |
| `CannotAssessStats` | CANNOT_ASSESS diagnostics for **binary** criteria — binary parallel of `NAStats`: `ca_count_true`, `ca_count_pred`, `ca_false_positive`, `ca_false_negative`, `ca_kappa` (Cohen's kappa on dichotomized {CANNOT_ASSESS, not-CANNOT_ASSESS}, pred vs truth) + `ca_kappa_interpretation` (`None` when undefined). A **distinct kind of abstention** from NA, so a separate type — see [CANNOT_ASSESS Handling (binary metrics)](abstention.md) |
| `CannotAssessMode` | `Literal["exclude", "as_unmet", "as_category"]` — how binary CANNOT_ASSESS verdicts are handled at metrics time. See [CANNOT_ASSESS Handling](abstention.md) below |
| `NAMode` | `Literal["exclude", "as_unmet", "as_category"]` — multi-choice NA analog of `CannotAssessMode`. See [NA Handling (multi-choice metrics)](abstention.md) below |

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
| `MetaCriterionJudgment` | Extends `CriterionJudgment` with `affected_criteria: list[int]` field for structured criterion references (internal — not exported from `meta.__init__`). `MultiChoiceMetaJudgment` is the multi-choice analog (extends `MultiChoiceJudgment`, same field; also internal/unexported) — wired via `multi_choice_response_format` |

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
