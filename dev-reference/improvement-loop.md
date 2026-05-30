[← Dev Reference Index](index.md)

### Improvement Loop Artifact Persistence
When `save_artifacts=True` and `artifacts_dir` is set, the improvement loop writes:
- `rubric-iter-{NN}.json` — criteria array per iteration
- `eval-iter-{NN}.html` — meta-rubric eval report (always generated, regardless of `display`)
- `iter-{NN}.json` — rich per-iteration JSON (quality report, issues, validation samples, revision prompts/response)
- `improvement_report.html` — consolidated report (always generated, regardless of `display`)
- `summary.json` — full run metadata, config snapshot, per-iteration summary

`revise_rubric()`, `validate_agreement()`, `validate_ground_truth()` accept a private `_capture` parameter for artifact collection.
