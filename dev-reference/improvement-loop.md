[← Dev Reference Index](index.md)

### Improvement Loop Artifact Persistence
When `save_artifacts=True` and `artifacts_dir` is set, the improvement loop writes:
- `rubric-iter-{NN}.json` — criteria array per iteration
- `eval-iter-{NN}.html` — meta-rubric eval report (always generated, regardless of `display`)
- `iter-{NN}.json` — rich per-iteration JSON (quality report, issues, validation samples, revision prompts/response)
- `improvement_report.html` — consolidated report (always generated, regardless of `display`)
- `summary.json` — full run metadata, config snapshot, per-iteration summary

`revise_rubric()`, `validate_agreement()`, `validate_ground_truth()` accept a private `_capture` parameter for artifact collection.

### Held-Out Validation Diagnostics

`validate_held_out()` grades held-out items and compares per-criterion verdicts against ground truth. Beyond accuracy / FP-rate / FN-rate, each `CriterionErrorReport` carries:

- `kappa` — Cohen's kappa between judge and ground-truth MET/UNMET labels over the usable pairs (via `_kappa_or_none`), or `None` when undefined (e.g. a constant array). Never a fabricated `0.0`.
- `coverage` — fraction of ground-truth-paired items that yielded a usable (non-abstained) verdict, over the **raw pre-exclusion** per-criterion denominator; `None` when nothing was paired.
- `ca_rate` — fraction of those raw paired items the judge abstained on (CANNOT_ASSESS), same denominator; `None` when nothing was paired.
- `confusion_matrix` — a 2x2 MET/UNMET `ConfusionMatrix` (reused from `autorubric.metrics`, rows=true cols=pred, `labels=["MET","UNMET"]`) over the usable verdicts; `None` when there are no usable samples (so a constructed all-zero matrix never masquerades as data).

`HeldOutValidationResult` records `cannot_assess` (the handling mode in effect), plus `mean_coverage` / `mean_ca_rate` rolled up from the per-criterion values via `_mean_or_none` (None-skipping).

**Abstention handling.** `ImprovementConfig.cannot_assess: CannotAssessMode` (default `"exclude"`, preserving the prior silent-exclude behavior) threads from the held-out runner into `validate_held_out`, which passes it as `mode=` to `filter_cannot_assess`. `"exclude"` drops abstained pairs from the confusion tallies; `"as_unmet"` folds CANNOT_ASSESS into UNMET; `"as_category"` keeps it as a distinct label. Regardless of mode, `coverage` and `ca_rate` are measured over the raw, pre-exclusion denominator — numerically aligned with `CoverageStats`.

**Serialization.** `_serialize_iteration` writes the new fields into `iter-{NN}.json`'s `held_out_diagnostics` block: `cannot_assess`, `mean_coverage`, `mean_ca_rate`, and per-criterion `kappa` / `coverage` / `ca_rate` / `confusion_matrix` (the matrix via `ConfusionMatrix.model_dump(mode="json")`, round-trippable with `ConfusionMatrix.model_validate`).

**Held-out HTML report.** `render_improvement_report_html` renders, for held-out iterations, a neutral handling-mode label (`CANNOT_ASSESS=<mode>`) plus mean coverage / CA-rate in the metrics bar, and a per-criterion table with Raw % Agreement (the judge-vs-ground-truth accuracy), Kappa, Coverage, CA-rate, Precision, FP-rate, FN-rate, and the 2x2 cells (TP/FP/TN/FN). It carries **no statistical conflation/cluster note** — single-source discipline keeps those exclusively in `MetricsResult.summary()`.
