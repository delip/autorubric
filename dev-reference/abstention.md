[← Dev Reference Index](index.md)

### CANNOT_ASSESS Handling
Strategies (applied by the single `score_reports` core, so binary CANNOT_ASSESS and multi-choice NA are handled identically wherever a score is computed): `SKIP` (excludes the criterion from both numerator and denominator), `ZERO`, `PARTIAL` (configurable), `FAIL` (weight-sign-aware worst case). Mechanics + `FAIL` detail: see [Score Calculation](grading-flow.md).

Judge-call failure routing (`infrastructure`/`parse` → CANNOT_ASSESS / `na=True`, excluded under SKIP; `unknown` → conservative worst case; forced-choice no-NA infra/parse abstain → `na=True` with `selected_index=None`/`selected_label=None`) is detailed in [Grading Flow](grading-flow.md).

### NA Handling (multi-choice metrics)
NA is the multi-choice *structural* analog of binary CANNOT_ASSESS, though a **distinct kind of abstention** (see *CANNOT_ASSESS Handling (binary metrics)*). At metrics time, `compute_metrics(..., na_mode=NAMode)` mirrors `cannot_assess: CannotAssessMode` one-for-one:

- `"exclude"` — drop pairs where either side is NA (default).
- `"as_unmet"` — remap NA → the score-minimizing non-NA option via the shared `Criterion.worst_scored_option()` (weight-sign aware, same helper as the `unknown`-error path).
- `"as_category"` — keep NA as a distinct categorical column. **Refused for ordinal criteria with an NA option** (raises `ValueError`): NA has no ordinal position, so quadratic-weighted kappa would assign it a meaningless index-based distance.

`compute_metrics` builds criteria from author-space `dataset.rubric`, but a prediction may reference an **auto-injected** NA option at index `N = len(author.options)` (out of author range) **or be a genuine no-option abstain (`selected_index=None`)**. Per multi-choice criterion, metrics reconstructs the effective criterion via `Criterion.with_guaranteed_na_option()` **iff an out-of-range OR `None` prediction is observed**, then normalizes any `None` prediction to that effective NA index — so injected-NA predictions and forced-choice error-abstains count as NA (FP/FN, `na_kappa`, filtering) under every `na_mode`, never as option 0, while forced-choice runs without abstains gain no spurious NA column. Consequence: once an ordinal criterion gains an NA option (authored, auto-injected + predicted, or induced by a `None` abstain), `na_mode="as_category"` is refused for it.

The old `"as_worst"` literal was renamed to `"as_category"` (it kept NA as a column, did not remap it); the old value raises a clear `ValueError`. The NAStats counts (`na_count_true`, `na_count_pred`, `na_false_positive`, `na_false_negative`, `na_kappa`) are mode-independent: FP/FN counters increment before any `exclude` skip, and `na_kappa` is computed from unfiltered per-criterion pred/true.

### CANNOT_ASSESS Handling (binary metrics) and the two kinds of abstention
Binary criteria get a `MetricsResult.cannot_assess_stats: CannotAssessStats | None` block, the structural parallel of multi-choice `na_stats: NAStats`. Populated whenever the rubric has ≥1 binary criterion (mirroring `na_stats` for ≥1 multi-choice criterion); reports the dichotomized {CANNOT_ASSESS, not-CANNOT_ASSESS} agreement (`ca_count_true`, `ca_count_pred`, `ca_false_positive`, `ca_false_negative`, `ca_kappa` — `None` when undefined). Like NAStats these counts are **mode-independent** (from raw per-criterion verdicts before any `cannot_assess` filtering) and surfaced in `summary()` only (NOT `to_dataframe()`).

**Two kinds of abstention (why two types, not one).** CANNOT_ASSESS and NA both flow through the same SKIP path and both get a parallel dichotomized-kappa stats block, but are semantically **distinct**:

- Binary **CANNOT_ASSESS** = the judge cannot determine MET-vs-UNMET — an *epistemic* abstention on a yes/no question.
- Multi-choice **NA** = "not applicable / cannot pick an applicable option" — abstaining because no scored category fits, a statement about the *option space*.

Tracked by **separate** types (`CannotAssessStats`, `ca_`-prefixed, vs. `NAStats`, `na_`-prefixed) to keep the distinction explicit while preserving the structural analogy.

### Coverage diagnostics (how much of the raw sample survived exclusion)
Under the **`exclude` handling mode only** (both `cannot_assess="exclude"` and `na_mode="exclude"`), `compute_metrics` builds per-criterion and aggregate `CoverageStats` via `_build_coverage_stats`, recording how much of the raw paired sample survived the union of all exclusion reasons. Under `as_unmet`/`as_category` nothing is union-excluded, so coverage would be trivially `1.0` and the stats are left `None`. The **raw denominator** counts every ground-truth-bearing item, including those lost to a grading error: an `n_errored_items` counter increments at the `item_result.error is not None` skip site, so `error_rate`, `judge_abstain_rate`, and `gt_abstain_rate` all share one consistent denominator (`items_with_ground_truth + n_errored_items`). `n_covered` equals the per-criterion `n_samples` (post-exclusion); `union_exclusion_rate = 1 - coverage`; every rate honours undefined→None at a zero denominator while counts stay `int`. The aggregate rollup pools across criteria (raw denominator × n_criteria; covered = sum of per-criterion `n_samples`). A warning is appended when any errored item was excluded, and the top-level `n_samples` is the sum of per-criterion `n_samples`.
