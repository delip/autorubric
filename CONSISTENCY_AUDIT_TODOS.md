# Consistency Audit — Prioritized TODOs

Audit of inconsistencies in how the codebase handles **analogous situations** across the
LLM-judge pipeline: prompt construction → judging → vote aggregation → scoring → metrics →
serialization. The recurring fault lines are **binary vs multi-choice** criteria, **single-judge
vs ensemble**, the **abstain mechanism** (binary `CANNOT_ASSESS` vs multi-choice `NA` option),
and **error/failure** paths.

These are to be fixed **separately**, in priority order. Each item is a checkbox; tick it when a
fix lands. This file is the working backlog — keep it updated as items are resolved or refined.

## Working directives (apply to every fix taken from this list)

> ALWAYS read CLAUDE.md and reorient yourself with the directives there before planning or starting work on anything
> **ultrathink** on each item before coding.
> - Use **Opus**, including for subtasks/subagents. **Create subagents generously** to manage context.
> - **TDD**: write a failing test that pins the inconsistency, then fix.
> - **Reuse over reinvent** — prefer stdlib / existing pypi / existing helpers; **don't create new types unnecessarily**.
> - Keep **`AUTORUBRIC_DEV_REFERENCE.md`** in sync with any code/structure change.
> - Standard gates before done: `uv run pytest`, `uv run ruff check . && uv run ruff format --diff .`, `uv run ty check src/autorubric` (no new diagnostics).
> - **No Claude/AI byline** in commits, PRs, or issues.
> - Many items are public-API/behavior changes — confirm the intended semantics with the user before implementing rather than guessing.
> - Things might have changed since these todo items were authored. If there is a discrepancy between what's claimed about the code in the todo item and what's actually in the code, the code is the source of truth.
> - Sometimes, you might have decision points that you might feel like resolving it with a human. Before doing that make sure your question cannot be already answered by the choices we have made in the branch (or being consistent with that), or by thinking harder. 

## Legend & context

- `[V]` = personally verified against source during the audit · (no tag) = sub-agent-reported, high confidence, **confirm during implementation**.
- Severity: **H** (correctness/bias or misleading output) · **M** (parity gap / latent risk) · **L** (cosmetic / maintainability).
- Baseline = branch `fix/fleiss-ensemble-metrics-issue-2` @ `9c17c64`. Recent Issue #1 (unified metrics `cannot_assess`) and Issue #2 (`krippendorff_alpha` + complete-case `fleiss_kappa`, `MultiChoiceJudgeVote.error`, per-judge alignment fix) are the **baseline** and are not re-flagged — except **T1-D**, a loose end from Issue #2.
- Line numbers are as of the baseline commit and will drift; treat them as anchors.

---

## P1 — Clear bugs / quick wins (low risk, self-contained, no API redesign)

- [x] **T1-A `[V]` (H) — Binary `"majority"` is not a majority; it equals `"weighted"`.**
  - Inconsistency: in `_aggregate_votes`, the `"majority"` and `"weighted"` branches are byte-identical (both `met_weight > unmet_weight`). With unequal judge weights, `majority` silently behaves as `weighted`; there is no head-count option. The docstring claims "Simple majority vote (> 50% must agree)".
  - Sites: `src/autorubric/graders/criterion_grader.py:1030-1033`; docstring `src/autorubric/types.py:650-656`. Contrast the multi-choice nominal path which *correctly* distinguishes `mode` (head-count) vs `weighted_mode` (`criterion_grader.py:1213-1222`).
  - Why it matters: a documented strategy does nothing distinct; users can't get an unweighted majority for binary even though multi-choice can.
  - Fix direction: make `"majority"` a true head-count (count judges, ignore weights; `> 50%`), keep `"weighted"` weight-based; decide tie behavior (see T3-B) and document both. Add tests with unequal weights proving majority ≠ weighted.
  - Effort/risk: **low / low** (behavior change for weighted ensembles using `majority` — call out in changelog).

- [x] **T1-D `[V]` (M) — `krippendorff_alpha` / `fleiss_kappa` never surfaced in `summary()` or `to_dataframe()`.** *(loose end from Issue #2)*
  - Inconsistency: both fields exist on `CriterionMetrics`/`OrdinalCriterionMetrics`/`NominalCriterionMetrics` and serialize via `to_file()`, but the two human-facing renderers never emit them.
  - Sites: fields `src/autorubric/metrics/_types.py:249-250, 353-354, 401-402`; `summary()` and `to_dataframe()` in the same file contain no reference to them.
  - Why it matters: the headline metrics of Issue #2 are invisible in the primary outputs.
  - Fix direction: add α (emphasized) and Fleiss columns/rows to `to_dataframe()` and lines to `summary()` for all three criterion types; gate on `is not None`. Consider also surfacing `per_option`/`confusion_matrix` (see T8 minors).
  - Effort/risk: **low / low**.

- [x] **T1-F (H) — Multi-choice `unknown`-error worst case ignores weight sign and flips to NA/skip.**
  - Inconsistency: binary unknown-error synthesizes a weight-aware worst case (`MET if weight<0 else UNMET`, `criterion_grader.py:606-609`). Multi-choice (`:753-771`) always picks the lowest-`value` option **and short-circuits to the first NA option if one exists**, ignoring weight sign — so an `unknown` error on a criterion that has an NA option becomes `na=True` (excluded under SKIP), the opposite of "keep a conservative worst-case verdict" (its own comment, `:746`). Negative-weight multi-choice criteria are under-penalized (lowest value is not the worst case when weight < 0).
  - Sites: `src/autorubric/graders/criterion_grader.py:606-609` (binary) vs `:753-771` (multi-choice).
  - Why it matters: error handling is not conservative for multi-choice; silently drops `unknown` errors when an NA option exists.
  - Fix direction: for `unknown` errors pick the option that minimizes score *with weight sign accounted for* (highest value for negative weight), and do **not** auto-select an NA option for `unknown` (reserve NA/skip for infra/parse). Mirror the binary worst-case intent.
  - Effort/risk: **low / med** (changes scores on errored multi-choice criteria). Related: **T2-B**.

- [x] **T1-E (M) — `na_agreement` denominator double-counts → not a clean rate.**
  - Inconsistency: `na_agreement = total_na_agreement / max(1, total_na)` where `total_na = total_na_true + total_na_pred` (a both-NA pair increments both), so the metric can't reach 1.0 even at perfect NA agreement.
  - Sites: `src/autorubric/metrics/_compute.py:1326-1330` (definition); counts at `:1308-1333`.
  - Why it matters: published agreement number is mathematically misleading; any future CANNOT_ASSESS-agreement stat (T2-C) must not copy it.
  - Fix direction: define a clean rate (e.g., agreement over the union of items where either side is NA, or Cohen-style over the NA/not-NA partition). Add a test pinning perfect-agreement → 1.0.
  - Effort/risk: **low / low**.

- [x] **T1-C `[V]` (H, naming/semantics) — `na_mode="as_worst"` is a misnomer/no-op.**
  - Inconsistency: the `"as_worst"` branch of `filter_na_multi_choice` just *keeps* NA pairs unchanged ("Keep NA but don't count as special"); it does not remap NA to a worst option. Binary's analogous knob (`cannot_assess`) has three real modes (`exclude`/`as_unmet`/`as_category`); NA has only `exclude` + this no-op.
  - Sites: `src/autorubric/metrics/_helpers.py:318-378` (esp. docstring `:335` and the branch); param surface `src/autorubric/metrics/_compute.py:809`.
  - Why it matters: misleading public parameter; NA/CANNOT_ASSESS are documented as analogous yet expose different mode sets.
  - Fix direction (decide with user): either (a) rename to `"keep"`/`"include"` and document; and/or (b) give NA real `as_unmet`-style and `as_category`-style analogs to match `cannot_assess`. Cross-ref **T1-B**, **T2-C**.
  - Effort/risk: **low / med** (public param rename = API change).

- [x] **T5-A (M-L) — `JudgeVote` / `MultiChoiceJudgeVote` lack the `is_error` property their report counterparts have.**
  - Inconsistency: `CriterionReport.is_error` and `EnsembleCriterionReport.is_error` exist; the per-vote types only carry an `error` field. Report docstrings advise "use `is_error` instead of inspecting reason" — impossible at vote level.
  - Sites: `src/autorubric/types.py` — `CriterionReport.is_error` (~`:526-533`), `EnsembleCriterionReport.is_error` (~`:740-743`); `JudgeVote` (~`:659-676`), `MultiChoiceJudgeVote` (~`:442-471`).
  - Fix direction: add a trivial `is_error` property (`self.error is not None`) to both vote dataclasses. Reuse, no new types.
  - Effort/risk: **low / low**.

- [x] **T5-B (M-L) — Multi-choice ensemble tracks errors twice; stale comment.**
  - Inconsistency: post-Issue-#2, `MultiChoiceJudgeVote.error` exists, yet the ensemble assembly still maintains a separate transient `mc_errors` list and carries a comment claiming the field doesn't exist. Binary uses a single source (`_aggregate_error(votes)`).
  - Sites: `src/autorubric/graders/criterion_grader.py:865-870, 884, 907` (multi-choice dual tracking + stale comment) vs `:84-86, 937` (binary single source).
  - Fix direction: derive the combined ensemble error from `[v.error for v in mc_votes]` (matching the binary path) and delete `mc_errors` + the stale comment; confirm the `mcv is None` guard at `:872` can't drop an errored vote (it can't today since errors always synthesize a verdict).
  - Effort/risk: **low / low**.

---

## P2 — Parity gaps needing small design calls (confirm semantics with user)

- [x] **T4-A + T1-B (H) — Two divergent scorers; `CannotAssessStrategy` not applied to multi-choice NA.**
  - Inconsistency: there are two scoring implementations — `_calculate_score_from_reports` (grader) and `Rubric.compute_score`/`_apply_cannot_assess_strategy` — and they disagree for NA under FAIL/ZERO. In the grader, multi-choice NA under `FAIL`/`ZERO` is a pass-through no-op (`[V]` `criterion_grader.py:1265-1267, 1289-1291`), so a negative-weight multi-choice criterion is never penalized under FAIL, unlike binary; `Rubric.compute_score` routes NA through `_apply_cannot_assess_strategy` and *does* apply it.
  - Sites: `src/autorubric/graders/criterion_grader.py:1243-1334` vs `src/autorubric/rubric.py:236-354`.
  - Why it matters: live-grading scores and recomputed/ground-truth scores can disagree for the same verdicts; `CannotAssessStrategy` silently doesn't cover multi-choice in one path.
  - Fix direction: unify on a single scoring function used by both (reuse over reinvent), and make `CannotAssessStrategy` apply uniformly to binary CANNOT_ASSESS and multi-choice NA (weight-sign-aware FAIL). Add a cross-check test: grader vs `Rubric.compute_score` agree across `CannotAssessStrategy` × {binary, multi-choice} × {±weight}.
  - Effort/risk: **med / med-high** (touches core scoring; needs careful tests).

- [x] **T3-A (H) — `AggregationStrategy` governs binary only; multi-choice uses separate enums; no `unanimous` analog.**
  - Sites: `src/autorubric/graders/criterion_grader.py:1015-1045` (binary reads `self._aggregation`) vs `:1047-1125` (`_aggregate_multi_choice_votes` reads `_ordinal_aggregation`/`_nominal_aggregation`, never `self._aggregation`).
  - Feedback to consider before working on this todo: the broad parity issue is real, but the wording is too broad. Nominal multi-choice already has a `unanimous` strategy in `NominalAggregation`, and `_aggregate_nominal_votes` has a `strategy == "unanimous"` branch. The sharper statement is that binary `aggregation` is orthogonal to multi-choice aggregation; ordinal multi-choice has no `unanimous`/`any` analog; nominal has `unanimous` but no `any`, and its non-unanimous fallback is mode rather than a strict failure-like outcome. Independently verify this before acting by checking `NominalAggregation` in `types.py` and the nominal aggregation branch in `criterion_grader.py`, then decide whether the fix is documentation-only, enum/API expansion, or semantic alignment.
  - Why it matters: a mixed rubric with `aggregation="unanimous"` gets strict consensus for binary but `mean`/`mode` for multi-choice silently.
  - Fix direction (decide with user): document the orthogonal knobs clearly, and/or provide analogous strategies (e.g., an ordinal/nominal `unanimous`/`any`) so the cross-type semantics line up. Reuse existing enums where possible.
  - Effort/risk: **med / med**.

- [x] **T3-B (M) — Tie-breaking diverges and is undocumented.**
  - Sites: binary tie→UNMET (`criterion_grader.py:1031`, the `>` makes equality fall to UNMET — note this *helps* a negative-weight criterion but penalizes a positive one); multi-choice tie→first-seen vote via `Counter.most_common` (`:1164, 1214, 1222`).
  - Fix direction: pick a documented, type-consistent tie rule (e.g., tie → conservative/worst-case by weight sign for both), and make it deterministic w.r.t. option order. Pairs naturally with T1-A.
  - Effort/risk: **low-med / med**.
  - **Resolved:** one uniform rule at every tie site (binary `majority`/`weighted`; ordinal/nominal `mode`; nominal `weighted_mode`; ordinal `mean`/`median` snap) — tie → score-minimizing outcome by weight sign, lowest option index as final tie-break (deterministic, independent of judge/vote order). Binary uses the shared `_binary_worst_verdict(weight)` (also reused by the `unknown`-error path); multi-choice routes tied candidates through the new `Criterion.worst_option_among(indices)`, which `worst_scored_option()` now delegates to — so aggregation, scoring `FAIL`, the `unknown`-error path, and metrics `as_unmet` share one sign-aware key and cannot drift. The mislabeled "no votes → index 0" fallback now uses `worst_scored_option()`. `min`/`max` already resolved value ties to the lowest index and were left untouched. Tie behavior documented in the three strategy-type docstrings (`types.py`). Tests: `tests/graders/test_binary_aggregation.py`, `tests/graders/test_multi_choice.py` (`TestMultiChoiceAggregation`), `tests/metrics/test_multi_choice_metrics.py` (`TestCriterionWorstOptionAmong`).

- [x] **T2-A (M) — No first-class abstain channel for multi-choice.**
  - Inconsistency: binary `CANNOT_ASSESS` is a guaranteed enum verdict + schema field + dedicated prompt section; multi-choice NA is an optional author-supplied option with conditional prompt language and no dedicated response-schema channel.
  - Sites: `src/autorubric/prompts.py:77-86` (binary) vs `:292-303` (multi-choice); `CriterionVerdict` enum vs `CriterionOption.na`.
  - Why it matters: a multi-choice criterion without an NA option has no way to express "cannot assess"; the judge is forced to pick a scored option.
  - Fix direction (decide with user): either standardize that multi-choice rubrics should include an NA option (validate/warn), or add a dedicated abstain channel to `MultiChoiceJudgment`. Pairs with T2-B.
  - Effort/risk: **med / med-high** (prompt + schema change).
  - **Resolved (decided with user — guarantee an NA option, auto-inject by default + configurable opt-out):** reused the existing NA model end-to-end — **no `MultiChoiceJudgment` schema change, no new types**. New pure `Criterion` helpers `na_option_index` + `with_guaranteed_na_option()` (appends `CANONICAL_NA_OPTION` at the end when absent; idempotent if an author NA exists). `CriterionGrader(auto_na_option=True)` (default) normalizes the rubric to effective criteria **once** in `judge()`; the effective options ride through `CriterionReport.options`, so prompt/verdict/scoring/aggregation share one option set. `auto_na_option=False` gives forced-choice and never strips an author NA. Prompts: NA options are visibly marked `(cannot assess / not applicable)` (`_render_options`/`_label_signals_na`, double-mark guarded) and the `MULTI_CHOICE_SYSTEM_PROMPT` NA section is now unconditional. `compute_metrics` reconstructs the effective criterion (via the same helper) **only when** a predicted injected-NA index is observed — recognizing it without crashing, leaving forced-choice runs unaffected. Manifest persists `auto_na_option`. This also resolves the **T2-B** contradiction in the default case (the infra/parse abstain verdict now points at a genuine NA option). Tests: `tests/graders/test_auto_na.py`, `tests/test_prompts.py`, `tests/metrics/test_multi_choice_metrics.py` (`TestCriterionNaOptionIndex`, `TestCriterionWithGuaranteedNAOption`, `TestMetricsAutoInjectedNA`), `tests/eval/test_eval_runner.py` (manifest). **T2-B residual** (forced-choice + no author NA representation) intentionally left for T2-B.

- [x] **T2-B (M) — Synthesizes `na=True` against a non-NA option when the criterion has no NA option.**
  - Inconsistency: on infra/parse error with no NA option, the multi-choice path builds a `MultiChoiceVerdict` whose `selected_index`/`selected_label` point at a real non-NA option but with `na=True`, `value=0.0` — an internally contradictory verdict.
  - Sites: `src/autorubric/graders/criterion_grader.py:772-779`.
  - Why it matters: downstream code reading `selected_index`/`selected_label` sees a legitimate-looking selection that was actually an error abstain. Cross-ref T1-F.
  - Fix direction: represent abstain without overloading a real option (depends on T2-A outcome). Minimum: don't claim `na=True` against a non-NA option.
  - Effort/risk: **med / med**.
  - **Resolved (decided with user — genuine no-option abstain):** `MultiChoiceVerdict.selected_index`/`selected_label` (and `MultiChoiceJudgeVote` / inheriting `AggregatedMultiChoiceVerdict`) are now `int | None` / `str | None`. With `auto_na_option=False` and no author NA option, an infra/parse failure yields `na=True` with `selected_index=None`/`selected_label=None` — never `na=True` against a scored option — and stays excluded under SKIP (infra/parse never penalizes). The all-NA aggregation branch prefers a vote that abstained into a genuine NA option and emits `None` only when every NA vote is itself a no-option abstain; `EnsembleCriterionReport.agreement` counts mutual abstention. Metrics normalizes a `None` multi-choice prediction to the effective NA index (reusing `with_guaranteed_na_option`, trigger extended to fire on `None`) so it is recognized as NA under every `na_mode`, never miscounted as option 0; `fill_ground_truth` raises on a `None`-label abstain. The ordinal/nominal aggregation helpers narrow indices to the assessable (non-`None`) set so the `int | None` change adds no `ty` diagnostics. Tests: `tests/graders/test_auto_na.py` (forced-choice single-judge), `tests/graders/test_error_routing.py` (ensemble all-fail + serialization round-trip), `tests/graders/test_multi_choice.py` (`TestMultiChoiceAggregation` all-NA prefer-genuine / clean-None), `tests/metrics/test_multi_choice_metrics.py` (`TestForcedChoiceNoneAbstain`).

- [x] **T2-C (M) — No metrics-level CANNOT_ASSESS stats block analogous to `NAStats`.**
  - Inconsistency: NA gets `NAStats` (counts, agreement, FP/FN) in `MetricsResult` + a `summary()` section; binary CANNOT_ASSESS gets no metrics-level reconciliation, so dropped binary pairs under `cannot_assess="exclude"` are invisible.
  - Sites: `NAStats` `src/autorubric/metrics/_types.py:286`; populated `_compute.py:1308-1333`; rendered `_types.py:785-795`.
  - Fix direction: add a CANNOT_ASSESS stats block mirroring `NAStats` (reuse the shape; avoid a brand-new type if `NAStats` can be generalized), with a clean agreement denominator (see T1-E).
  - Effort/risk: **med / low-med**.
  - **Resolved (decided with user — new parallel type, NOT a merge):** added `CannotAssessStats` (`metrics/_types.py`, exported from `metrics/__init__.py`) mirroring `NAStats` with `ca_`-prefixed fields (`ca_count_true`/`ca_count_pred`, `ca_false_positive`/`ca_false_negative`, `ca_kappa` = Cohen's κ on the {CANNOT_ASSESS, not-CA} dichotomy, `ca_kappa_interpretation`). `MetricsResult.cannot_assess_stats` is populated for any rubric with ≥1 binary criterion via a dedicated post-loop pass in `_compute.py` that counts from the **raw** per-criterion verdicts (mode-independent, before `filter_cannot_assess`), and rendered in `summary()` only (NOT `to_dataframe()`, matching `NAStats`). Per the user's guidance the two abstentions are kept as **separate types** and **documented as distinct kinds of abstention** — binary CANNOT_ASSESS is an epistemic "can't decide MET/UNMET" vs multi-choice NA's "no applicable option" — in the `CannotAssessStats` docstring and a new dev-reference subsection (*CANNOT_ASSESS Handling (binary metrics) and the two kinds of abstention*); the existing "structural analog" lines were amended to add the semantic-distinction caveat. The clean-rate concern (T1-E) is inherited: `ca_kappa` is the chance-corrected dichotomy κ, not the old union-denominator rate. Tests: `tests/metrics/test_compute_metrics.py` (`TestCaKappa`: populated/perfect/none-observed/disagreement=1/3/counts/mode-independence/summary) + `tests/metrics/test_multi_choice_metrics.py` (`TestNaKappa.test_multi_choice_only_rubric_leaves_ca_stats_none`).

- [x] **T8-B (H) — Per-judge metrics are binary-only → misleading `0.0` for multi-choice ensembles.**
  - Inconsistency: `_compute_judge_metrics` filters to binary criteria and only reads `cr.votes`; a multi-choice ensemble with `per_judge=True` yields `0.0` accuracy/precision/etc. (indistinguishable from genuine zero), even though `multi_choice_votes` carry the data (enriched in Issue #2).
  - Sites: `src/autorubric/metrics/_compute.py:715-798` (binary-only) and the per-judge collection `:1016-1036`.
  - Fix direction: either compute per-judge multi-choice metrics from `multi_choice_votes`, or return `None` (not `0.0`) for unsupported types and document. Decide with user.
  - Effort/risk: **med / med**.
  - **Resolved (decided with user — Option A, unified with T8-C under one principle):** from first principles, the top-level `precision`/`recall`/`f1` ARE the binary MET-vs-rest metric (undefined for multi-choice → `None`, not a fabricated `0.0`; the multi-class P/R/F1 story already lives per-option in `OptionMetrics`), while `accuracy`/`kappa` GENERALIZE and are computed for real. Per-judge metrics now **mirror the aggregate's type-handling field-for-field** via a single shared helper `_criterion_level_scalars(...)`: the per-judge loop collects `cr.multi_choice_votes` (NA-normalized through the SAME `effective_criteria` as the aggregate, errored votes skipped) and builds each judge's per-criterion kappas by reusing `_compute_ordinal/nominal_criterion_metrics` + `prepare_binary_metric_inputs`, so a 1-judge "ensemble" equals the aggregate **by construction** (invariant test). `MetricsResult`/`JudgeMetrics` five scalar fields widened to `float | None`. Tests: `tests/metrics/test_interjudge_agreement.py` (multi-choice per-judge accuracy/kappa hand-checked, P/R/F1 None, single-judge parity incl. a degenerate criterion, mixed-rubric, errored-MC-vote exclusion), `tests/metrics/test_multi_choice_metrics.py`. See *Undefined → None* in `AUTORUBRIC_DEV_REFERENCE.md`.

- [x] **T8-C (M-H) — Aggregate precision/recall/f1 hard-set to `0.0` (not `None`) for multi-choice-only rubrics.**
  - Sites: `src/autorubric/metrics/_compute.py:1259-1262`; rendered in `to_dataframe()` with no type guard.
  - Fix direction: use `None` for not-applicable aggregates and skip/blank them in `summary()`/`to_dataframe()`. Pairs with T1-D rendering work.
  - Effort/risk: **low-med / low**.
  - **Resolved (with user, unified with T8-B):** aggregate `criterion_precision/recall/f1` now `None` (not `0.0`) for multi-choice-only rubrics (binary MET-vs-rest is undefined without a MET class); `criterion_accuracy`/`mean_kappa` stay computed for real. `MetricsResult` fields widened to `float | None`; `summary()` renders via the None-safe `_fmt_opt` (the existing `n_binary_criteria>0` guard already skipped the lines), `to_dataframe()` already tolerates `None`. The single scoring/rendering path is shared with T8-B.

- [x] **T9 (H, resolved in the same PR as T8-B/T8-C) — "Undefined → fabricated `0.0`/`1.0`" anti-pattern swept codebase-wide.**
  - Context: investigating T8 from first principles surfaced a recurring fault — a metric/statistic/score that is genuinely undefined (no data / degenerate input / computation failed / not-applicable) was silently reported as a real-looking `0.0` (or `1.0`), corrupting aggregates (`mean_kappa`), fabricating findings ("no correlation", "no bias", "identical distributions"), and (in the improvement loop) driving wrong decisions. The codebase already did the right thing for `na_kappa`/`ca_kappa`/`BiasResult.{p_value,effect_size,ci}` (returned `None`), so most of these were internal inconsistencies.
  - **Resolved (decided with user — fix the whole family, one issue at a time, doing it right):**
    - **#1 kappa:** `_kappa_or_none` (returns `None` on exception **and NaN** — degenerate single-class made `cohen_kappa_score` return NaN that the old `except: 0.0` leaked) replaces the 5 swallow sites; `None` excluded from `mean_kappa` via `_mean_or_none`.
    - **#3 empty-data criteria:** metric values `None` (counts stay `0`) across binary/ordinal/nominal empty blocks + `OptionMetrics` + `_compute_adjacent_accuracy`.
    - **#2 correlation:** `CorrelationResult.coefficient`/`p_value` `None` for `<3` samples / constant array (NaN); also the improvement-loop ground-truth Spearman.
    - **#4 distribution / #5 bias** (`distribution.py`): low-`n` stats `None` (n=1 keeps computable mean/min/max/median/iqr); `systematic_bias` computes the real single-difference `mean_bias` at n=1 with `std_bias=None`, `effect_size=None` when `std==0`; `EMDResult`/`DistributionResult`/`BiasResult` widened.
    - **#6 agreement:** `validate_agreement` no-data → `None`; empty-rubric `EnsembleEvaluationReport.mean_agreement` → `None` (field `float | None`).
    - **#7 error score:** `EvaluationReport.score`/`EnsembleEvaluationReport.score` (+ raw_score) `float | None`, `None` only on the explicit failure constructors; the scoring core is unaffected (it computes, never reads `report.score`); `compute_metrics` excludes errored/`None`-score reports from score-level stats (no fabricated `0.0`); length-penalty/`get_scores`/`meta._display` None-guarded; serialization round-trips `None`; a structural grade failure reaching `validate_ground_truth`/the improvement loop raises a clear `RuntimeError` rather than fabricating a value.
  - Rendering uses the single None-safe `_fmt_opt`; examples/docs made None-safe. See *Undefined → None* in `AUTORUBRIC_DEV_REFERENCE.md`. `BootstrapResults` CIs widened to `tuple[float, float] | None`.

- [ ] **T6-A (Critical?, NEEDS VERIFICATION) — `AggregatedMultiChoiceVerdict.aggregated_value` may be dropped on single-report round-trip.**
  - Inconsistency: single-report deserialize uses `CriterionReport.model_validate` (`eval.py:255`) where the `MultiChoiceVerdict | AggregatedMultiChoiceVerdict` union may resolve to the base subtype and drop `aggregated_value`; the ensemble path explicitly does `AggregatedMultiChoiceVerdict.model_validate` (`eval.py:295`).
  - Feedback to consider before working on this todo: this may not be a live bug in the current environment. A direct single-report round-trip using `CriterionReport.model_dump(mode="json")` followed by `CriterionReport.model_validate(...)` preserved the concrete `AggregatedMultiChoiceVerdict` subtype and `aggregated_value` under Pydantic `2.12.3`. Treat this item as a verification task before any implementation, not as confirmed breakage. Independently verify through the real persistence path as well (`ItemResult.to_dict()` → `ItemResult.from_dict()` with a single-report `CriterionReport.multi_choice_verdict=AggregatedMultiChoiceVerdict(...)`). If `aggregated_value` survives there too, close or downgrade this to a regression-test/backcompat note rather than adding discriminators or custom deserialization.
  - **Verify first:** write a round-trip test serializing a `CriterionReport` whose `multi_choice_verdict` is an `AggregatedMultiChoiceVerdict`; check whether `aggregated_value` survives under the installed pydantic. If it survives, downgrade/close this item.
  - Fix direction (if confirmed): use a discriminated union or explicit subtype validation; or align single-report (de)serialization with the ensemble approach (see T6-D).
  - Effort/risk: **low to verify / med to fix**.

- [ ] **T6-B (M) — `reasoning` (extended-thinking trace) is never serialized.**
  - Sites: `EvaluationReport.reasoning` / `EnsembleEvaluationReport.reasoning`; `ItemResult.to_dict` (`eval.py:381-419`) writes every other top-level field but not `reasoning`; no deserialize reads it.
  - Feedback to consider before working on this todo: the loss is real, but the cited model surface appears inaccurate. `EvaluationReport` and `EnsembleEvaluationReport` do not currently define a top-level `reasoning` field. The reasoning fields exist on `CriterionJudgment` and `MultiChoiceJudgment`, and `LLMClient.generate()` can inject extracted thinking content into those structured outputs, but `CriterionGrader` copies only `judgment.explanation` into `CriterionReport.reason` and does not preserve `judgment.reasoning` anywhere. So the trace is likely lost during judgment-to-report construction before checkpoint serialization ever runs. Independently verify this data flow before adding persistence fields: create or mock a judgment with `reasoning`, run the binary and multi-choice criterion paths, and confirm whether the produced `CriterionReport` contains any place for it. The fix may need a report-level/per-criterion `reasoning` field first, followed by serialization/deserialization, rather than only adding `ItemResult.to_dict()` keys.
  - Why it matters: thinking trace is lost on checkpoint/resume for both report families.
  - Fix direction: serialize/deserialize `reasoning` symmetrically (decide whether large traces should be optional/omittable for size).
  - Effort/risk: **low / low**.

- [ ] **T6-C (L-M) — `llm_raw_score` never serialized; aliased to `raw_score` on load.**
  - Sites: `ItemResult.to_dict` writes `raw_score` only; all deserializers set `llm_raw_score=report_data.get("raw_score")` (`eval.py:260, 330, 448`).
  - Fix direction: persist `llm_raw_score` distinctly if it can ever diverge from `raw_score`; otherwise document the invariant explicitly.
  - Effort/risk: **low / low**.

---

## P3 — Larger / architectural (design + broader blast radius)

- [ ] **T6-D (structural) — Single reports use pydantic `model_dump`/`model_validate`; ensemble reports are hand-rolled per-field.**
  - Why it matters: root cause of serialization drift (T6-A/B/C) — every new ensemble field must be mirrored across two hand-written functions.
  - Fix direction: move ensemble (de)serialization onto pydantic models (or a shared schema) so field coverage is automatic and symmetric. Reuse pydantic; avoid bespoke dict plumbing.
  - Effort/risk: **high / med-high** (touches persistence format; add round-trip tests for every field of every report/vote type).

- [ ] **T7-A (H) — Few-shot example selection is not de-correlated across judges/criteria.**
  - Inconsistency: option shuffling derives a unique RNG per `(seed, item, criterion, judge)` (`criterion_grader.py:651-653`), but few-shot example *selection* uses a flat `random.Random(config.seed)` per criterion (`:341, :471`), so all ensemble judges see identical few-shot examples and ordering correlates across criteria.
  - Why it matters: undercuts the ensemble independence that the agreement metrics (Krippendorff/Fleiss, Issue #2) assume.
  - Fix direction: derive the few-shot RNG the same way as option shuffling (`_derive_shuffle_rng`-style, per judge/criterion/item). Reuse the existing helper.
  - Effort/risk: **med / med** (changes which examples judges see → may shift results; pin with seeded tests).

- [ ] **T7-B (H) — `binary_response_format` override + `affected_criteria` injection have no multi-choice analog.**
  - Inconsistency: binary supports a custom `response_format` (used by meta-eval's `MetaCriterionJudgment` with `affected_criteria`) and injects `[Affects: …]`; multi-choice hardcodes `MultiChoiceJudgment` (`criterion_grader.py:701`) with no hook (`:574/582-584` binary-only).
  - Why it matters: meta-eval / structured-output extensions can't run on multi-choice rubrics.
  - Fix direction: add a `multi_choice_response_format` parameter + symmetric `affected_criteria` handling. Avoid new types where the existing judgment models can be subclassed.
  - Effort/risk: **med-high / med**.

- [ ] **T7-C (M) — Empty-submission prompt instruction is ill-defined for nominal criteria.**
  - Inconsistency: `MULTI_CHOICE_SYSTEM_PROMPT` tells the judge to "select the lowest-quality option" on empty/refusal (`prompts.py:307-308`), meaningful only for ordinal scales; the same prompt serves nominal criteria (no quality order).
  - Fix direction: parameterize the prompt by `scale_type` (ordinal vs nominal), or give nominal its own empty-submission guidance.
  - Effort/risk: **med / low-med** (prompt change; eval-sensitive).

- [ ] **T8-A (H) — Bootstrap CIs are binary-only.**
  - Inconsistency: `bootstrap=True` on a pure multi-choice rubric silently produces nothing (`_compute.py:1277` guard requires binary labels); no weighted-kappa/exact-accuracy CI for ordinal/nominal. Undocumented.
  - Fix direction: extend bootstrap to multi-choice point estimates, or document the limitation and emit a warning when `bootstrap=True` has no effect.
  - Effort/risk: **med-high / low-med**.

---

## P3 — Minor / cosmetic (batch when convenient)

- [ ] **(M) Criterion-level support fields binary-only.** `CriterionMetrics` has `support_true`/`support_pred`; ordinal/nominal expose support only per-option (`OptionMetrics`). Consider top-level support + an abstain-drop count per criterion (binary CA drops are currently invisible).
- [ ] **(L) Two `interpret_kappa` label sets.** `_compute._interpret_kappa` returns `"poor"`; `KappaResult.interpret_kappa` returns `"poor (worse than chance)"`. Consolidate to one canonical mapping.
- [ ] **(L) Empty-data agreement-stat handling differs per type.** Binary/ordinal/nominal empty-data branches zero `fleiss_kappa`/`krippendorff_alpha` slightly differently (`_compute.py` empty branches). Normalize.
- [ ] **(L-M) Multi-choice extraction defaults to option 0 on failure.** `extract_all_verdicts_from_report` defaults a failed multi-choice extraction to index `0` (a real category) vs binary → `UNMET` (`_helpers.py:283-290`, `_compute.py:970-975`). The option-0 default silently fabricates a category vote.
- [ ] **(L) `balance_verdicts` reused for multi-choice option balancing** — semantically misnamed for the multi-choice path (`criterion_grader.py:494`). Doc or rename.
- [ ] **(L) Few-shot example XML schema differs** between binary (`<verdict>`) and multi-choice (`<selected_option>`+`<selected_label>`). Internally consistent; note for maintainers.

---

## Cross-references
- Scoring: **T4-A ↔ T1-B** (unify + apply CannotAssessStrategy to NA) ↔ **T1-F** (error worst-case) ↔ **T3-B** (tie = worst case?).
- Abstain: **T1-C** (na_mode) ↔ **T2-A/B/C** (channel, synthesized NA, CA stats) ↔ **T1-E** (agreement denominator).
- Aggregation: **T1-A** (majority) ↔ **T3-A/B**.
- Serialization: **T6-A/B/C** ↔ **T6-D** (the structural unification that subsumes them).
- Metrics rendering: **T1-D** ↔ **T8-C** (None vs 0.0) ↔ minor support/kappa items.
