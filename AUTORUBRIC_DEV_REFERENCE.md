# AutoRubric Development Reference

A Python library for evaluating text outputs against weighted criteria using LLM-as-a-judge.

**For detailed documentation, examples, and usage guides, see docs/.**

## Package Structure

```
src/autorubric/
├── __init__.py              # Public exports
├── dataset.py               # DataItem, RubricDataset
├── eval.py                  # EvalRunner, EvalResult, evaluate()
├── llm.py                   # LLMConfig, LLMClient, ThinkingConfig
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
| `Criterion`                | Single evaluation criterion with weight, requirement, optional multi-choice options |
| `CriterionOption`          | Multi-choice option with label, value (0-1), optional `na` flag                     |
| `CriterionVerdict`         | Enum: `MET`, `UNMET`, `CANNOT_ASSESS`                                               |
| `CriterionReport`          | Criterion + verdict + reason                                                        |
| `EvaluationReport`         | Full grading result with score, raw_score, report, token_usage, cost                |
| `EnsembleEvaluationReport` | Adds judge_scores, mean_agreement, per-criterion votes                              |
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
| `CriterionMetrics`        | Per-criterion binary metrics                                              |
| `OrdinalCriterionMetrics` | weighted_kappa, adjacent_accuracy, correlations                           |
| `NominalCriterionMetrics` | kappa, per_option metrics                                                 |

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
6. Aggregates votes using strategy (majority/weighted/unanimous/any)
7. Returns `EnsembleEvaluationReport` (consistent interface)

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

### CANNOT_ASSESS Handling
Strategies: `SKIP` (adjust denominator), `ZERO`, `PARTIAL` (configurable), `FAIL` (worst case)

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
- Parse failures use conservative defaults (UNMET for positive, MET for negative weights)
- Filter `error is not None` results in training pipelines
- Rate limiting via `LLMConfig.max_parallel_requests` (per-provider semaphore)

## Public Exports

See `src/autorubric/__init__.py` for complete list.
