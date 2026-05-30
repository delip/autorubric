[← Dev Reference Index](index.md)

## Key Conventions

- All graders return `EnsembleEvaluationReport` for consistent interface
- `raw_score` always populated regardless of `normalize` setting
- Judge-call failures classified via `classify_grading_error()`: `infrastructure`/`parse` → CANNOT_ASSESS / `na=True` (excluded under default SKIP); `unknown` → conservative worst case (forced-choice no-NA infra/parse abstain → `na=True`, `selected_index=None`/`selected_label=None`). Full routing: see [Grading Flow](grading-flow.md)
- **Report serialization is uniformly pydantic-native.** `EnsembleCriterionReport` / `JudgeVote` / `MultiChoiceJudgeVote` are **frozen pydantic models** (not dataclasses), so the ensemble checkpoint path collapses onto `model_dump(mode="json")` / `model_validate` exactly like the single-report `CriterionReport` path — no hand-rolled per-field plumbing. `ItemResult.to_dict` keeps its envelope (`report_type` discriminator, `criterion_reports`, `judge_scores`, nested `token_usage`); `_serialize_ensemble_criterion_report` is now a thin `ecr.model_dump(mode="json")` wrapper (shared with the meta improvement-loop artifacts), and `_deserialize_ensemble_report` rebuilds only the envelope around `[EnsembleCriterionReport.model_validate(...)]`. Adding a field to any report/vote type now round-trips for free. `EnsembleCriterionReport.agreement` is computed by a `model_validator(mode="after")` (via `object.__setattr__`, frozen-safe) — the prior `__post_init__` semantics, idempotent on reload.
- `JudgeVote.error` / `MultiChoiceJudgeVote.error` / `EnsembleCriterionReport.error` carry category-prefixed messages; serialization round-trips `error` on ensemble reports, binary judge votes, and multi-choice judge votes — automatically, via the pydantic dump/validate above
- `CriterionReport.reasoning` / `JudgeVote.reasoning` / `MultiChoiceJudgeVote.reasoning` carry the extended-thinking deliberation trace (see [Grading Flow](grading-flow.md)); serialization round-trips it symmetrically with `error` via pydantic `model_dump`/`model_validate` on **both** the single-report and ensemble paths, with field defaults tolerating legacy checkpoints (missing `reasoning`/`error`/`weight`/`na`/`votes`/`agreement` → field default; a missing/0.0 `agreement` recomputes from the votes)
- Filter `error is not None` results in training pipelines
- Rate limiting via `LLMConfig.max_parallel_requests` (per-provider semaphore)
- Multi-choice criteria get a guaranteed abstain option by default (`CriterionGrader(auto_na_option=True)`, auto-injected NA — binary-CANNOT_ASSESS parity; see [Multi-Choice Criteria](multi-choice.md)); set `auto_na_option=False` for forced-choice. Never strips an author NA option

## Public Exports

See `src/autorubric/__init__.py` for complete list.
