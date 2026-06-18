# AutoRubric Development Reference

A Python library for evaluating text outputs against weighted criteria using LLM-as-a-judge.

**For detailed documentation, examples, and usage guides, see docs/.**

This reference is split into focused pages for gradual disclosure — load only the page you need. Start here for the package map, then follow the [Reference Map](#reference-map) below.

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

## Reference Map

Each page has a keyword-dense summary so you can route directly to the one that answers your question. The **Load when** line says what the page is for, and the **File location** line gives its path.

### Type Catalog

Quick-reference catalog of every public and internal type/function, organized by module: core types in `types.py` (`Criterion` and its `worst_scored_option`/`worst_option_among`/`na_option_index`/`with_guaranteed_na_option` methods, `CriterionOption`, the `CriterionVerdict` enum `MET`/`UNMET`/`CANNOT_ASSESS`, `CriterionReport`, `EvaluationReport`, `EnsembleEvaluationReport`, `JudgeVote`, `MultiChoiceJudgeVote` with nullable `selected_index`/`selected_label`, `EnsembleCriterionReport`, `LengthPenalty`, `TokenUsage`), grader types (`CriterionGrader`, `JudgeSpec`, `FewShotConfig`), dataset types (`DataItem`, `RubricDataset`), LLM types (`LLMConfig`, `LLMClient`, `ThinkingConfig`, `ErrorCategory`, `classify_grading_error`), eval types (`EvalConfig`, `EvalResult`, `ItemResult`), and metrics types (`MetricsResult` with its handling-mode provenance `cannot_assess_mode`/`na_mode` and aggregate scalars `n_samples`/`mean_krippendorff_alpha`/`criterion_phi`/`macro_accuracy`/`micro_kappa`/`coverage_stats` plus the per-item-heterogeneous `pooled_by_scale: list[PooledScaleMetrics]`, the pooled rubric-point type `PooledScaleMetrics` (`scale_type`/`n_points`/`exact_accuracy`/`value_rmse`/`value_mae`/`value_spearman`/`value_pearson`/`n_abstain` + binary-only `kappa`/`phi`/`precision`/`recall`/`f1`/`confusion_matrix`), `CriterionMetrics` with `confusion_matrix`/`fpr`/`fnr`/`phi`/`is_degenerate`/`coverage_stats`, `OrdinalCriterionMetrics`, `NominalCriterionMetrics` — both migrated to a single `ConfusionMatrix` (`option_labels` removed) plus `is_degenerate`/`coverage_stats`, `JudgeMetrics` (+ `phi`/`confusion_matrix`), the unified `ConfusionMatrix` and `CoverageStats` (`n_total`/`n_covered`/`n_errored` counts plus `coverage`/`judge_abstain_rate`/`gt_abstain_rate`/`union_exclusion_rate`/`error_rate`) typed diagnostics, `NAStats`, `CannotAssessStats`, plus the `CannotAssessMode`/`NAMode` literals `exclude`/`as_unmet`/`as_category`; note `AgreementSummary` has been removed). Each entry documents fields, which scalar metrics are `float | None`, the inter-judge agreement fields (`krippendorff_alpha`/`fleiss_kappa`), the φ (Matthews correlation coefficient) fields, and the `na_`/`ca_`-prefixed abstention diagnostics. It also lists rubric-improvement types from `meta/_improve.py` and the meta-rubric judgment types + full meta-rubric function roster.

**Load when:** you need the name, fields, methods, module location, or `float | None` nullability of any type/class/enum/function — core, grader, dataset, LLM, eval, metrics, rubric-improvement, or meta-rubric — or to find which module a type lives in.

**File location:** `dev-reference/types.md`

### Grading Flow & Score Calculation

The end-to-end grading pipeline: `Rubric.grade()` delegates to `CriterionGrader` (a single LLM is an "ensemble of 1"), which at `judge()` start normalizes the rubric **once** into an *effective rubric* — with `auto_na_option=True` (default) every multi-choice criterion is guaranteed an NA abstain option via `Criterion.with_guaranteed_na_option()`, a pure same-length/order transform that keeps `criterion_idx` and the shuffle RNG key aligned and never mutates the user rubric; `auto_na_option=False` gives forced-choice. Covers concurrent per-criterion/per-judge calls via `asyncio.gather()`, binary vs multi-choice response formats (`binary_response_format`/`CriterionJudgment`, `multi_choice_response_format`/`MultiChoiceJudgment`, meta overrides with `affected_criteria` and `[Affects: #1, #3]` injection via `_inject_affected_criteria()`), and vote aggregation (`majority` head-count vs `weighted`, plus independent `ordinal_aggregation`/`nominal_aggregation`). Judge-call failure routing via `classify_grading_error()` maps `infrastructure`/`parse` → `CANNOT_ASSESS`/`na=True` (excluded under `CannotAssessStrategy.SKIP`) and `unknown` → conservative worst case; forced-choice no-NA abstain → `na=True` with `selected_index=None`; `is_error` distinguishes error-induced from genuine verdicts. Extended-thinking preservation copies `judgment.explanation`→`reason` and `judgment.reasoning`→`reasoning`; the provider schema strips the unused `reasoning` slot via `_provider_response_format()` so strict-mode backends (Groq) don't force non-OpenAI models (Llama/Qwen) to emit it. Score Calculation centers on the single core `score_reports(reports, config, normalize=True)` in `scoring.py` (shared by the live grader, `Rubric.compute_score`, `RubricDataset.compute_weighted_score`) — the weighted-sum/clamp formula, negative-weight-only fallback, length penalty, and `_abstain_contribution` for `SKIP`/`ZERO`/`PARTIAL`/`FAIL`.

**Load when:** you need to understand how `Rubric.grade` routes to the grader, NA auto-injection / effective-rubric normalization, response formats and `affected_criteria` injection, vote aggregation, judge-call failure classification/routing, extended-thinking preservation, or how the single `score_reports` core computes weighted scores and applies `CannotAssessStrategy`.

**File location:** `dev-reference/grading-flow.md`

### Multi-Choice Criteria

Multi-choice criteria end to end: `scale_type` `ordinal` (weighted kappa) vs `nominal` (unweighted kappa), explicit per-option `value` (0–1) and `shuffle_options=True` to mitigate position bias, and NA options (`na: true`) as the structural analog of binary `CANNOT_ASSESS` flowing through `score_reports` under the configured `CannotAssessStrategy`. Covers the first-class abstain channel: the auto-injected `CANONICAL_NA_OPTION`, NA-label rendering via `_render_options`/`_label_signals_na` in `prompts.py`, and the unconditional `MULTI_CHOICE_SYSTEM_PROMPT` NA section. Details scale-aware, option-driven empty/refusal handling — failure/lowest-quality option when one describes an empty submission (ordinal, scored ≈0) → else NA (nominal, SKIP-excluded) → else closest forced-choice option — and the same option-shape rule for contradictory/ambiguous submissions (the judge never sees weights, so scoring applies the weight sign downstream via option `value`). Specifies the three orthogonal ensemble aggregation knobs (`aggregation`, `ordinal_aggregation`, `nominal_aggregation`) on a central/conservative/permissive axis with the full value table, including nominal `unanimous` abstaining via the genuine NA option. Closes with uniform, deterministic, weight-sign-aware tie-breaking: score-minimizing outcome + lowest-index tie-break, `_binary_worst_verdict` / `Criterion.worst_option_among`.

**Load when:** you need ordinal vs nominal scales, option `value`/shuffling and position bias, NA/abstain handling, how empty/refusal and contradictory submissions are scored, the ensemble aggregation knobs and their central/conservative/permissive values, or how ensemble ties are broken.

**File location:** `dev-reference/multi-choice.md`

### Abstention Handling

How abstention is handled across scoring and metrics for both binary `CANNOT_ASSESS` and multi-choice NA. Documents the four CannotAssessStrategy scoring strategies (`SKIP`, `ZERO`, `PARTIAL`, `FAIL`) applied by the single `score_reports` core, plus judge-call failure routing (`infrastructure`/`parse` → abstain excluded under SKIP, `unknown` → conservative worst case, forced-choice no-NA abstain → `na=True` with `selected_index=None`). Details metrics-time `na_mode=NAMode` handling (`exclude`, `as_unmet` via `Criterion.worst_scored_option()`, `as_category` which raises `ValueError` for ordinal NA), effective-criterion reconstruction via `Criterion.with_guaranteed_na_option()` when an out-of-range or `None` prediction is observed, and the `as_worst`→`as_category` rename. Describes the mode-independent `NAStats` counts (`na_count_true`/`na_count_pred`/`na_false_positive`/`na_false_negative`/`na_kappa`) and the binary parallel `CannotAssessStats` block with `ca_`-prefixed counts. Explains the metrics-time `CoverageStats` wiring (exclude-mode only, via `_build_coverage_stats`): per-criterion + aggregate coverage on a RAW denominator (`items_with_ground_truth + n_errored_items`) with union-exclusion, the `n_errored_items` counter at the error-skip site, the grading-error warning, and the top-level `n_samples`. Closes by explaining why `CANNOT_ASSESS` (epistemic, MET-vs-UNMET) and NA (option-space "not applicable") are tracked as **two separate types** despite sharing the SKIP path and dichotomized-kappa structure.

**Load when:** you need how CANNOT_ASSESS/NA strategies affect the scoring denominator, how judge-call failures route to abstain vs worst-case, how metrics treat NA via `na_mode`, when `as_category` is refused, how auto-injected/`None` abstains normalize, how exclude-mode `CoverageStats` / `n_errored_items` / `n_samples` are wired, or why CANNOT_ASSESS and NA are separate types.

**File location:** `dev-reference/abstention.md`

### Metrics

The metrics layer: inter-judge agreement — Krippendorff's α (`krippendorff_alpha`, level-aware ordinal/nominal, built from a per-criterion reliability matrix tolerating missing/errored cells) and complete-case Fleiss' κ (`fleiss_kappa` via `statsmodels`, uniform-rater subjects) — both ground-truth-independent and computed even with zero GT-paired samples. States the framework principle that prediction-vs-ground-truth categorical agreement is always Cohen's kappa (binary `CriterionMetrics.kappa`, ordinal `weighted_kappa` quadratic, nominal `kappa`), with orthogonal abstain agreement via `NAStats.na_kappa` and `CannotAssessStats.ca_kappa`. Details aggregate/per-judge scalar parity: `criterion_precision`/`recall`/`f1` are binary MET-vs-rest (`None` for multi-choice-only rubrics), `criterion_accuracy`/`mean_kappa` generalize, sharing `_criterion_level_scalars` (now a 7-tuple also yielding `phi`+`micro_kappa`) + `_mean_or_none`; per-judge metrics mirror the aggregate field-for-field (1-judge == aggregate invariant, incl. `phi`). Covers the Matthews correlation φ helper `_mcc_or_none` (mandatory `len(set(...)) < 2 → None` guard against sklearn's misleading 0.0; φ≡κ on matched marginals, φ>κ on positive-rate drift), the binary 2×2 `ConfusionMatrix` + `fpr`/`fnr`/`phi`/`is_degenerate` via `_build_binary_2x2_confusion_matrix`, the per-judge 3×3 `["MET","UNMET","CANNOT_ASSESS"]` `ConfusionMatrix` via `_build_binary_judge_confusion_matrix` (RAW pre-filter codes pooled across binary criteria; aggregate == elementwise sum), the aggregate `criterion_phi`/`macro_accuracy`/`micro_kappa`/`mean_krippendorff_alpha`, `is_degenerate` on ordinal/nominal (`weighted_kappa`/`kappa is None`), and the score-collapse + degeneracy warnings. Exhaustively specifies the cross-cutting **"undefined → `None`, never a fabricated `0.0`"** invariant across Cohen kappa (`_kappa_or_none` NaN handling), empty-data criteria, correlations (`_compute_correlation`, <3 samples / constant arrays), distribution stats, systematic bias / Cohen's d, inter-judge agreement (`validate_agreement`, `mean_agreement`), item-level bootstrap CIs for all rubric types (`_compute_bootstrap_ci`, `accuracy_ci`/`kappa_ci`/`rmse_ci`), grade-failure `None` scores, and extraction-failure `None` mapping. Also covers the **per-item heterogeneous-rubric pooled path**: `compute_metrics(..., per_item_metrics="auto"|"pooled"|"per_criterion")` dispatches to `_compute_per_item_pooled_metrics` (gated by `_has_heterogeneous_rubrics`/`_rubric_signature`) when the dataset has no global rubric and item rubrics differ (HealthBench) → `per_criterion=[]` + `MetricsResult.pooled_by_scale: list[PooledScaleMetrics]` (per-scale exact-accuracy + unified value pool; binary-only κ/φ/P/R/F1/confusion; the homogeneous per-item case stays per-criterion via the `get_item_rubric` score-step fix).

**Load when:** you need to understand or modify metric computation — including the per-item heterogeneous pooled path (`per_item_metrics`/`pooled_by_scale`/`PooledScaleMetrics`/`_compute_per_item_pooled_metrics`) — — inter-judge agreement (Krippendorff α vs Fleiss κ), pred-vs-truth Cohen's kappa across binary/ordinal/nominal, the Matthews φ helper `_mcc_or_none` / `criterion_phi` / per-judge φ, the binary 2×2 and per-judge 3×3 `ConfusionMatrix` builders, `fpr`/`fnr`/`is_degenerate`, `macro_accuracy`/`micro_kappa`/`mean_krippendorff_alpha`, score-collapse / degeneracy warnings, aggregate vs per-judge scalar parity, item-level bootstrap CIs, abstention-agreement kappas, or the "undefined → None" invariant covering correlations, distribution stats, bias, grade-failure scores, and extraction failures.

**File location:** `dev-reference/metrics.md`

### Reproducibility & Seed Coordination

How all non-LLM randomness is made deterministic. `CriterionGrader(seed=...)` controls option shuffling and few-shot selection; when `None` the `master_seed` is auto-generated so shuffles are always pinned. Per-call shuffle RNGs derive from `(master_seed, content_hash, criterion_idx, judge_id)` via SHA-256 (concurrency-safe, no shared mutable state) using `_derive_shuffle_rng()` in `criterion_grader.py`; an unset `FewShotConfig.seed` is coordinated from the master seed, and few-shot selection RNGs derive from `(few_shot_seed, criterion_idx, judge_id)` with a constant `FEW_SHOT_DOMAIN` in the item-key slot. `CriterionReport.shuffle_order` records the per-criterion permutation, and `ExperimentManifest.grader_config` persists `master_seed`/`shuffle_options`/`auto_na_option` via `_serialize_grader_config` in `eval.py`. With `auto_na_option=True`, a criterion that lacked an NA option gains one appended at the end, so its `shuffle_order` grows by one.

**Load when:** you need how seeds/RNGs are derived and coordinated, how option shuffling and few-shot selection are made deterministic and concurrency-safe, what is persisted for checkpoint reproducibility, or how auto-injected NA options affect a criterion's shuffle permutation.

**File location:** `dev-reference/reproducibility.md`

### Improvement Loop Artifacts

What the rubric improvement loop persists. When `save_artifacts=True` and `artifacts_dir` is set, it writes per-iteration files — `rubric-iter-{NN}.json` (criteria array), `eval-iter-{NN}.html` (meta-rubric eval report, always generated regardless of `display`), `iter-{NN}.json` (rich per-iteration JSON: quality report, issues, validation samples, revision prompts/response) — plus run-level `improvement_report.html` (consolidated, always generated) and `summary.json` (full run metadata, config snapshot, per-iteration summary). Also notes that `revise_rubric()`, `validate_agreement()`, and `validate_ground_truth()` accept a private `_capture` parameter used to collect these artifacts.

**Load when:** you need to know what files the improvement loop writes to disk, what `save_artifacts`/`artifacts_dir` produce, the naming/contents of per-iteration vs consolidated artifacts, or how `_capture` feeds artifact collection.

**File location:** `dev-reference/improvement-loop.md`

### Key Conventions & Public Exports

The framework's cross-cutting conventions: all graders return `EnsembleEvaluationReport`; `raw_score` is always populated regardless of `normalize`; judge-call failures route via `classify_grading_error()` (`infrastructure`/`parse` → `CANNOT_ASSESS`/`na=True` excluded under default SKIP, `unknown` → conservative worst case, forced-choice no-NA abstain → `na=True` with `selected_index=None`). Documents uniformly pydantic-native report serialization: `EnsembleCriterionReport`/`JudgeVote`/`MultiChoiceJudgeVote` are frozen pydantic models, so checkpointing collapses onto `model_dump(mode="json")`/`model_validate` like the single `CriterionReport` path, with `ItemResult.to_dict` keeping its envelope (`report_type` discriminator, `criterion_reports`, `judge_scores`, `token_usage`), `_serialize_ensemble_criterion_report`/`_deserialize_ensemble_report`, and `EnsembleCriterionReport.agreement` computed by a frozen-safe `model_validator(mode="after")`. Covers category-prefixed `error` round-tripping, extended-thinking `reasoning` round-tripping with legacy-checkpoint-tolerant field defaults, filtering `error is not None` in training pipelines, per-provider rate limiting via `LLMConfig.max_parallel_requests`, and the default guaranteed abstain option `CriterionGrader(auto_na_option=True)` vs `auto_na_option=False` forced-choice. Also states the **single-source metric-redundancy note** discipline: the conflation / different-geometry / κ−φ-gap notes live only in `MetricsResult.summary()`/`to_dataframe()`/docstrings and are never copied into the `meta/_display.py` Rich/HTML paths. Ends pointing to `src/autorubric/__init__.py` for the complete Public Exports list.

**Load when:** you need the framework-wide conventions — what graders return, `raw_score`/`normalize` behavior, judge-call failure routing, how reports/votes serialize and round-trip across checkpoints, where `error`/`reasoning` traces live, rate-limiting config, the default auto-NA behavior, the single-source metric-note discipline, or where public exports are listed.

**File location:** `dev-reference/conventions.md`

## How to update autorubric's dev reference

This reference uses **gradual disclosure**: `CLAUDE.md` `@`-imports only this `index.md`, so it loads every session as a lightweight router, while the eight detail pages load on demand when a link is followed. Preserve that property.

**Edit the page the change touches.** `types.md` for type/field changes, `grading-flow.md` for the grading pipeline or scoring core, `metrics.md` for metric computation, and so on. 

**Keep this index a router, not a copy.**

- Don't hyperlink the headings; give each page a **File location** line with its repo-relative path in backticks (e.g. `dev-reference/types.md`). Never use `@`-imports — an `@`-import would auto-load every page and defeat gradual disclosure.
- When you add, rename, or remove a page, update the **Reference Map** above with a keyword-dense TLDR, a **Load when** line, and a **File location** line. The TLDR is what makes a page discoverable, so name its key types, functions, config knobs, enum values, T-codes, and topics.
- When you add a new type, function, config knob, or T-code to a page, make sure the term also appears in that page's TLDR here — otherwise a reader scanning the index won't find it.
- Don't let the index grow toward monolith size: TLDRs + keywords route; the prose stays on the pages.

**Page conventions (so edits stay consistent).**

- Each page begins with the breadcrumb `[← Dev Reference Index](index.md)` and a blank line, then its content.
- A reference to another page's section becomes a cross-file link — `[Section Name](other-page.md)`. A reference to a section on the *same* page stays as italic emphasis (`*Section Name*`).
- **Dedup, don't drop facts.** The same mechanism is often restated in a type-table cell, an architecture page, and `conventions.md`. When trimming, consolidate to one canonical page and cross-link from the others — never delete a fact. Preserve verbatim every type/field/method/function name, enum value, table row, and code block.