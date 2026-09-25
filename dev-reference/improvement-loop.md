[← Dev Reference Index](index.md)

### Improvement Loop Artifact Persistence
When `save_artifacts=True` and `artifacts_dir` is set, the improvement loop writes:
- `rubric-iter-{NN}.json` — the iteration's rubric: the criteria array (`weight`, `requirement`, `name` when set), or `{"guidelines": ..., "criteria": [...]}` when the rubric has guidelines (`_save_rubric` via `_rubric_artifact_data`; `Rubric.from_dict` reads either back)
- `eval-iter-{NN}.html` — meta-rubric eval report (always generated, regardless of `display`)
- `iter-{NN}.json` — rich per-iteration JSON (quality report, issues, validation samples, revision prompts/response); its `rubric_criteria` is always the criteria array (`_rubric_criteria_data`)
- `improvement_report.html` — consolidated report (always generated, regardless of `display`); its Final Rubric panel shows the guidelines, HTML-escaped, above the criteria table when the final rubric has them
- `summary.json` — full run metadata, config snapshot, per-iteration summary; `original_rubric`/`final_rubric` use the same form as `rubric-iter-{NN}.json`

A rubric without guidelines produces exactly the artifacts it did before guidelines existed.

`revise_rubric()`, `validate_agreement()`, `validate_ground_truth()` accept a private `_capture` parameter for artifact collection.

### Rubric Guidelines in Meta-Rubric Evaluation and the Loop

`Rubric.guidelines` (see [Type Catalog](types.md)) is rubric text that applies to every criterion, so the meta paths treat it as part of the rubric:

- **Meta-judge.** `evaluate_rubric_standalone`/`evaluate_rubric_in_context` build the rubric under review with `_rubric_for_meta_judge`: `{"guidelines": ..., "criteria": [...]}` (guidelines first, as on disk) when present, `{"criteria": [...]}` otherwise. `_meta_judge_instruction` appends `_GUIDELINES_REF_INSTRUCTION` (what the field is, the rule judges get — the criterion text governs, the guidelines clarify how to apply it — and to assess each criterion as a rater would apply it with the guidelines) to `_CRITERIA_REF_INSTRUCTION` only when present. The display summary (`_rubric_summary`) reads "... criteria and guidelines".
- **Carry-over.** The revision LLM writes criteria only, so `revise_rubric`/`revise_rubric_held_out` rebuild the revised rubric with `_revised_rubric`, which carries the input rubric's guidelines unchanged. Without it every iteration after the first would grade and be meta-evaluated without them. A held-out revision that breaks the criteria structure returns the input rubric, guidelines included.
- **Revision prompt.** The loop never revises guidelines. The revision LLM still sees them, read-only, because correctness needs it: the issues and diagnostics it must act on come from judges that saw the guidelines and can refer to them, and a criterion revised blind could contradict them — a conflict the criterion text wins when grading. `_with_guidelines_block` puts `RUBRIC_REVISION_GUIDELINES_BLOCK` (`prompts.py`; the guidelines verbatim, then "They are fixed and not part of your output: revise only the criteria, and keep them consistent with the guidelines.") at the start of the user prompt, for both strategies and for custom `revision_user_prompt_template`s; the system prompt is unchanged.
- **Byte identity.** Without guidelines every meta-judge and revision prompt is unchanged. `tests/meta/test_meta_prompt_goldens.py` pins them: `tests/golden/meta/without_guidelines.json` was captured from the unmodified library, `with_guidelines.json` pins the guidelines case.

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
