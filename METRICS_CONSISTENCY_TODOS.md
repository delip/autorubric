# Metrics Consistency Audit — autorubric vs. the metrics-explainer paper

**Audit date:** 2026-05-30
**Paper audited against:** `autorubric-paper/metrics-explainer/metrics-explainer.tex`
("Agreement Metrics for LLM-as-Judge Evaluation: What to Report and Why") — the
11-item reporting checklist (Sec. 7) and the two formal results (Observation 1:
binary correlation collapse; Observation 2: ensemble α / κ_F / pairwise-φ
coincidence).
**Code audited:** `src/autorubric/metrics/{_compute,_types,_helpers,distribution}.py`,
`src/autorubric/meta/{_display,_improve}.py`, `src/autorubric/eval.py`.
**Method:** parallel read-only code audit (8 agents, file:line evidence) +
literature research on the 4 genuine "which is better" questions (verified
Observation 2 numerically against the real `krippendorff` + `statsmodels` calls).
All claims below carry `file:line` citations; the orchestrator independently
re-verified the two P0/P1 cornerstones (micro/macro mixing at `_compute.py:822-864`;
mode-not-persisted at `_compute.py:1775-1804`).

> **Scope note.** This document is a TODO list, not a code change. Every item
> that touches code must also update the dev reference in `dev-reference/` (per CLAUDE.md)
> and add/extend tests. A few items embed a genuine design choice; those are
> flagged **[DECISION]** and listed in the final section — we should walk through
> them one at a time before implementing.

## Working directives (apply to every fix taken from this list)

> ALWAYS read CLAUDE.md and reorient yourself with the directives there before planning or starting work on anything
> **ultrathink** on each item before coding.
> - Use **Opus**, including for subtasks/subagents. **Create subagents generously** to manage context.
> - **TDD**: write a failing test that pins the inconsistency, then fix.
> - **Reuse over reinvent** — prefer stdlib / existing pypi / existing helpers; **don't create new types unnecessarily**.
> - Standard gates before done: `uv run pytest`, `uv run ruff check . && uv run ruff format --diff .`, `uv run ty check src/autorubric` (no new diagnostics).
> - **No Claude/AI byline** in commits, PRs, or issues.
> - Many items are public-API/behavior changes — confirm the intended semantics with the user before implementing rather than guessing.
> - Things might have changed since these todo items were authored. If there is a discrepancy between what's claimed about the code in the todo item and what's actually in the code, the code is the source of truth.
> - Sometimes, you might have decision points that you might feel like resolving it with a human. Before doing that make sure your question cannot be already answered by the choices we have made in the branch (or being consistent with that), or by thinking harder.

---

## 1. Executive summary

autorubric's **per-criterion binary metrics are already well aligned** with the
paper: it reports accuracy + Cohen's κ + precision/recall/F1 and reports **no
redundant binary association coefficients** (no per-criterion Pearson/Spearman/
Kendall/φ/MCC), so it does not "manufacture triangulation" on binary verdicts
(checklist items 2, 3 — consistent). The "Undefined → None, never a fake 0.0"
invariant satisfies the core of item 5. The abstention *modes* are named exactly
as the paper's three estimands (exclude / as_unmet / as_category).

The gaps are concentrated in **what the reports say about a number's meaning**,
not in the numbers themselves:

| # | Gap | Checklist item | Priority |
|---|-----|----------------|----------|
| T1 | Handling mode (`cannot_assess`/`na_mode`) is **never stored on `MetricsResult` nor displayed** — every persisted/printed number is ambiguous among 3 estimands | 6 | **P0** |
| T2 | **No cluster note** where conflating stats are reported side by side (α≡Fleiss on binary/nominal; the prospective φ-bundle) — *the user's explicit requirement* | 2, 11 | **P1** |
| T3 | **Coverage and abstention/invalid rates are not surfaced** (only raw counts); exclude-mode metrics shown without their coverage | 7, 8 | **P1** |
| T4 | Aggregate scalars **mix micro (accuracy/P/R/F1) and macro (mean_kappa) levels, unlabeled** | 10 | **P1** |
| T5 | **No binary 2×2 confusion matrix / FP-FN counts** on `CriterionMetrics` (multi-choice has one; binary does not) | 4 | **P1** |
| T6 | **No per-judge confusion matrices**; `summary()` prints only RMSE+Spearman per judge, hiding the accuracy/κ/P/R it already computes | 11 | **P1** |
| T7 | Rendered reports (Rich/HTML) surface **almost no agreement metrics**; improvement HTML labels a **raw proportion as "Agreement"** (not chance-corrected); held-out report omits mode/coverage/κ | 3,6,7,8,11 | **P1** |
| T8 | **No φ/MCC for binary** — the κ–φ gap (the paper's central diagnostic = positive-rate drift) is not directly visible | 2, 3 | **P2** |
| T9 | Single-criterion / monotone score → **Pearson=Spearman=Kendall collapse** (Obs. 1) is reported as three "independent" numbers with no note | 2 | **P2** |
| T10 | Degenerate criterion not distinguished from no-data and **not flagged** in `warnings` | 5 | **P2** |
| T11 | Ordinal α (distance-aware) vs Fleiss (nominal) **diverge** — keep both but note the different geometry (or replace bare Fleiss with ordinal-weighted) | 11 | **P2/P3** |
| T12 | Score-level correlation **scale not declared**; full micro/macro/item-level triple not offered; three-class weight matrix (item 9) implicit | 1, 10, 9 | **P3** |

**Not a problem (do not "fix"):** pairwise/tie handling (items 1/6/7) — autorubric
produces no pairwise-preference verdicts, so the tie portions are N/A. The
item-level **cluster bootstrap** (resamples items, shared index across criteria)
is correct per the item-10 cluster caveat. Ordinal α correctly uses
`level="ordinal"`. Score-level Pearson/Spearman/Kendall on the *continuous*
weighted score are legitimately informative (Table 1 continuous row) and are **not**
an item-2 violation in the general multi-criterion case.

---

## 2. The clustering requirement (user's explicit ask)

> "If we are reporting multiple metrics that should conflate, we should report
> them as a cluster with that note in the generated reports."

There are exactly **three** conflation situations in autorubric. Treat them
distinctly — only the first is reported today, and none carries a note.

### 2a. Krippendorff's α (nominal) ≡ Fleiss' κ — on BINARY and NOMINAL criteria → **CLUSTER + NOTE**
- **Both are reported, side by side, with no note.** `summary()` prints a
  `Kripp-α  Fleiss` two-column block for every criterion type
  (`_types.py:1016-1023, 1035-1036, 1077-1078`); `to_dataframe()` always emits
  both columns (`_types.py:1151-1152, 1197-1198`).
- **They conflate** for binary/nominal because α is computed `level="nominal"`
  there (`_compute.py:1451-1453`) and Fleiss is always nominal
  (`_build_fleiss_row`, `_compute.py:294-359`). Per Observation 2,
  `α = κ_F + (1 − κ_F)/(NR)` — verified numerically to 1e-9 against the real
  `krippendorff.alpha(level="nominal")` + `statsmodels.fleiss_kappa` at
  (N,R) = (30,3),(100,3),(50,5). The gap is `< 1e-3` (below 3-decimal reporting
  precision) whenever `NR > 1000(1−κ_F)` — e.g. N=100,R=3 already qualifies for
  κ_F ≥ 0.57. Paper: `metrics-explainer.tex:353, 358, 399`.
- **TODO:** report **Krippendorff's α as the single primary** statistic for
  binary/nominal; keep Fleiss adjacent (backward-compat / familiarity) but under a
  **shared clustered header with a note**: *"On binary/nominal data Krippendorff's
  nominal α and Fleiss' κ coincide up to a finite-sample correction
  (1−κ_F)/(NR) ≈ <value>; they are one statistic, not corroborating evidence — α
  is primary."* Optionally surface the gap value itself for small-N readers.
  α is the right primary because it natively handles the missing/unequal raters
  autorubric produces (errored/excluded/CA votes → `np.nan` cells), whereas Fleiss
  is complete-case and silently drops those items (`_compute.py:294-307`).

### 2b. Prospective φ/MCC bundle — on BINARY criteria → **CLUSTER + NOTE** (depends on T8)
- If φ/MCC is added (T8), it must be the **single** binary association number:
  Observation 1 says φ = Pearson = Spearman = Kendall = MCC on non-degenerate
  binary data (`metrics-explainer.tex:221-229`). Do **not** also add per-criterion
  Pearson/Spearman/Kendall on binary 0/1 vectors.
- **TODO:** render φ in the **same cluster as κ**, with the note *"φ (= Pearson =
  Spearman = Kendall = MCC on binary data) is the binary association coefficient;
  κ is its marginal-sensitive companion; the κ−φ gap is the judge's positive-rate
  drift from the human's"* (`metrics-explainer.tex:255-272, 405`).

### 2c. Score-level Pearson/Spearman/Kendall — NOT a conflation in general; conflate ONLY in the degenerate case
- On the **continuous** per-item weighted score these answer different questions
  and may all be informative (Table 1 continuous row) — **no cluster note**, but
  **declare the scale** (T12). Verified continuous at `_compute.py:1618-1620`
  feeding from `report.score` / `compute_weighted_score` (`_compute.py:1262-1265`).
- **Exception (T9):** for a **single-criterion rubric** (or any rubric whose
  normalized score is monotone in one binary verdict) the per-item score is
  ≤2-valued and the three collapse to one number (Obs. 1). There, they DO conflate
  and must carry a collapse note. Today they are emitted unconditionally
  (`_types.py:924-935`) with no detection.

### 2d. Ordinal α vs Fleiss — **DO NOT cluster** (they genuinely diverge)
- For ordinal criteria α uses `level="ordinal"` (distance-aware) while Fleiss
  stays nominal; they weight near-vs-far disagreements differently and can rank
  judges oppositely (`metrics-explainer.tex:367`; code `_compute.py:1451-1453` vs
  `294-359`). Reporting both is justified — but add a **distinguishing note**
  (different geometry), so a reader does not mistake the gap for finite-sample
  noise. See T11.

**Where the cluster notes live:** `MetricsResult.summary()` and `to_dataframe()`
in `src/autorubric/metrics/_types.py` are the **only** surfaces that actually
render κ/α/Fleiss (the Rich/HTML report paths in `meta/_display.py` render no
agreement coefficients — see T7). So the metric-redundancy notes belong in
`_types.py`. The improvement HTML needs the separate fix in T7.

---

## 3. Prioritized TODOs

Each item: **what → where (file:line) → why (checklist item + paper) → how**.

### P0 — blocking ambiguity

#### T1. Persist and display the CANNOT_ASSESS / NA handling mode  *(item 6)*
- **Finding.** `compute_metrics(cannot_assess="exclude", na_mode="exclude", …)`
  (`_compute.py:1044-1045`) uses the modes to filter/transform but the returned
  `MetricsResult` **stores neither** — the constructor (`_compute.py:1775-1804`)
  has no mode field and the model (`_types.py:847-883`) has no such attribute. So
  `summary()`, `to_dataframe()`, and `to_file()` emit every accuracy/κ/F1 with its
  estimand stripped. The only type carrying `cannot_assess_mode` is
  `AgreementSummary` (`_types.py:559, 587`), which is **dead code** — never
  constructed, returned, imported, or exported (repo-wide grep). `EvalResult.compute_metrics`
  even narrows the param to `Literal["exclude","as_unmet"]` and threads it through
  without recording it (`eval.py:509, 558-567`).
- **Why P0.** The paper: "a metric reported without its handling mode is ambiguous
  among [the three estimands]" (`metrics-explainer.tex:389`). Two runs at
  `exclude` vs `as_category` are indistinguishable once serialized — this breaks
  reproducibility and cross-study comparability, and it undercuts every coverage/
  rate fix below.
- **How.** Add frozen fields `cannot_assess_mode: CannotAssessMode` and
  `na_mode: NAMode` to `MetricsResult`; set them in the constructor from the args;
  render a `Handling modes: CANNOT_ASSESS=exclude, NA=exclude` line in `summary()`
  near the criteria-type line (`_types.py:903`); add columns/keys to the
  `to_dataframe()` aggregate row so `to_file()` round-trips them. Widen
  `EvalResult.compute_metrics`'s `cannot_assess` type to include `as_category` (or
  document why it is restricted). **Retire or repurpose `AgreementSummary`.**

### P1 — high

#### T2. Cluster-note conflating inter-judge stats  *(items 2, 11 — user's explicit ask)*
- See §2a (α≡Fleiss on binary/nominal) and §2d (ordinal: note divergence, do not
  cluster). **How.** In `summary()`'s `_agreement_header`/`_agreement_cells`
  (`_types.py:1016-1023`), branch on criterion-group type: for binary/nominal
  groups, print α as primary with Fleiss under a clustered header + a one-line
  conflation note (and optionally the `(1−κ_F)/(NR)` gap); for ordinal groups,
  print both with a "different geometry" note. Mirror the note in
  `CriterionMetrics`/`NominalCriterionMetrics.krippendorff_alpha`/`fleiss_kappa`
  docstrings and `docs/api/metrics.md` (which today says only "Prefer
  krippendorff_alpha", `docs/api/metrics.md:9`, with no conflation note).
- **[DECISION]** keep-and-cluster Fleiss vs drop the bare Fleiss column for
  binary/nominal. Recommended: keep + cluster + note (non-breaking).

#### T3. Surface coverage + abstention/invalid rates  *(items 7, 8)*
- **Finding.** `NAStats`/`CannotAssessStats` are **count-only**
  (`_types.py:364-369, 415-420`); no rate, no coverage field — the docstrings even
  tell readers to derive proportions themselves (`_types.py:355, 404`). Per-criterion
  `n_samples` IS the post-exclusion covered count (binary `_compute.py:1518`;
  multi-choice `:420, :541`) and `n_items` is the full count (`:1361`), so
  `coverage = n_samples / n_items` is derivable but never surfaced, and never
  paired with the metric in `summary()` (per-criterion tables omit `n_samples`/
  support — `_types.py:1034, 1053-1056, 1076`). **Invalid/error rate is invisible:**
  errored items are silently `continue`d with no warning/count (`_compute.py:1207-1208`,
  contrast the missing-GT warning at `:1203-1204`); per-vote `is_error` drops for
  inter-judge agreement are uncounted (`:327, 351, 967-968`).
- **Why.** Items 7-8: "report abstention/tie/invalid rates as metrics in their own
  right" and "for exclude mode, report covered-subset performance AND coverage
  together" (`metrics-explainer.tex:391, 393`); selective-prediction literature
  (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017) is unanimous that the unit is
  the (coverage, selective accuracy) pair — a lone selective accuracy is trivially
  gamed by abstaining on hard cases.
- **How.** (a) Add a derived `coverage = n_samples / n_items` per-criterion and at
  aggregate; print it on the SAME block as every exclude-mode metric. (b) Add
  abstention **rates** — report judge-side (`*_count_pred / n`) and human/GT-side
  (`*_count_true / n`) **separately** (exclude drops a pair if either abstains) plus
  the union exclusion rate = 1 − coverage. (c) Add an explicit `n_errored` count +
  error rate to `MetricsResult` (tally at `_compute.py:1207`, append a warning).
  Gate coverage to exclude mode (it is trivially 1.0 under as_unmet/as_category).
  The counts are already mode-independent (`_helpers.py:410-422`;
  `_compute.py:1725-1727`), so the numerators are reliable.

#### T4. Label the aggregation level on every aggregate scalar  *(item 10)*
- **Finding (independently re-verified).** `criterion_accuracy`/`precision`/
  `recall`/`f1` are **micro** — pooled over all item×criterion decisions via
  `label_*_flat.extend(...)` then a single `accuracy_score`/`precision_score`/…
  (`_compute.py:822-846`); `mean_kappa` is **macro** — `_mean_or_none` over the
  per-criterion kappa list (`_compute.py:864`). They are printed together in one
  unlabeled "Criterion-Level Metrics" block (`_types.py:911-918`) and packed into
  one unlabeled `to_dataframe()` aggregate row (`_types.py:1107-1128`). Per-judge
  mirrors the same mix via the shared helpers (`_compute.py:992-1004`).
- **Why.** Item 10: micro-κ ≈ 0.85 vs macro-κ ≈ 0.55 can both be "the" aggregate
  κ (`metrics-explainer.tex:292-294, 397`) — an unlabeled mixed-level report is
  ambiguous and not comparable.
- **How.** Relabel `summary()` per metric: `Accuracy (micro)`, `Precision (micro)`,
  `Mean Kappa (macro)`; in `to_dataframe()` rename the aggregate columns
  (`accuracy_micro`/`kappa_macro`) or add an `aggregation` field; document the level
  on `MetricsResult.criterion_accuracy`/`mean_kappa`. Note that `mean_kappa` mixes
  per-scale kappas (binary unweighted / ordinal quadratic-weighted / nominal
  unweighted) and `accuracy` mixes binary-label and multi-choice exact-match.
  (Adding the missing aggregations is T12; labeling is the P1 part.)

#### T5. Store a binary 2×2 confusion matrix / FP-FN counts on `CriterionMetrics`  *(item 4)*
- **Finding.** The binary branch computes only accuracy/P/R/F1/κ + `support_true`/
  `support_pred` (MET marginals = TP+FN, TP+FP); **no TP/FP/FN/TN stored**
  (`_compute.py:1504-1532`; `CriterionMetrics` fields `_types.py:289-304`).
  Multi-choice criteria *do* carry `confusion_matrix` (`_types.py:476, 522`) — an
  inconsistency in the data model. FP/FN are only *algebraically* recoverable from
  {support_true, support_pred, precision, recall, n_samples} and that breaks under
  as_category and when P/R are None.
- **Why.** Item 4 prefers the explicit 2×2 because every symmetric statistic is
  FP↔FN-invariant; the confusion matrix is what separates a **strict** judge (high
  FN) from a **lenient** one (high FP), "often the most actionable diagnosis"
  (`metrics-explainer.tex:385`).
- **How.** Add `tp/fp/fn/tn: int` (or reuse `ClassificationReport`,
  `_types.py:226-247) to binary `CriterionMetrics`, populate from the existing
  0/1 vectors, surface FP/FN in `summary()`/`to_dataframe()`. The weaker half of
  item 4 (P and R shown separately) is already met (`_types.py:915-917, 1042-1043`).

#### T6. Per-judge confusion matrices + surface the per-judge metrics already computed  *(item 11)*
- **Finding.** `JudgeMetrics` carries only scalars (no confusion matrix, no
  per-criterion breakdown — `_types.py:786-799`); `_compute_judge_metrics` builds
  `pj_pred`/`pj_true` per criterion then discards them after reducing to 5 scalars
  (`_compute.py:955-1004`). Worse, `summary()` prints **only RMSE + Spearman** per
  judge (`_types.py:995-1002`) — hiding the per-judge accuracy/κ/P/R it stores, and
  Spearman is itself a redundant binary correlation (item 2). `to_dataframe()` does
  emit the per-judge scalars (`_types.py:1203-1226`) but never cell counts.
- **Why.** Item 11: "report one primary inter-judge statistic AND the per-judge
  confusion matrices" — essential once CA/NA is admitted and any scalar depends on
  a non-canonical 3×3 weight matrix (`metrics-explainer.tex:371, 399`).
- **How.** Add a per-judge confusion matrix to `JudgeMetrics` (2×2 binary; 3×3 with
  CA/NA), populated from the already-assembled `pj_pred`/`pj_true`; reuse
  `ClassificationReport`. In `summary()` print per-judge **accuracy + mean_kappa**
  (item-3 pairing) and precision+recall (item-4), demote RMSE/Spearman to secondary.

#### T7. Fix the rendered Rich/HTML reports  *(items 3, 6, 7, 8, 11)*
- **Finding.** The meta-rubric eval report (Rich + HTML) shows only verdict status
  + Score/RawScore/Cost — **no κ/α/Fleiss/φ/P/R/confusion/accuracy at all**
  (`_display.py:154-171, 122-139, 426`). The improvement reports surface an
  **"Agreement" column that is a raw proportion** of judges agreeing with the final
  verdict (`_display.py:648, 653, 682`; source `types.py:906`,
  `criterion_grader.py:1107`) — **not** chance-corrected (not κ/α/Fleiss/φ). The
  held-out report shows Accuracy + FP Rate + FN Rate (`_display.py:718-735`) but
  **no κ**, **no mode label** (silent default exclude via
  `filter_cannot_assess(...)` with no mode arg, `_improve.py:1280`), and **no
  coverage** (CA pairs dropped silently at `_improve.py:1283-1284`; raw tp/fp/tn/fn
  computed at `:1250-1298` then discarded — only rates kept). Note the shown
  FP Rate = FP/(FP+TN) is **not** precision, so precision is unrecoverable.
- **How.** (a) Relabel improvement "Agreement" → **"Raw % agreement"** so it is not
  read as a κ. (b) Add a Kappa column to the held-out HTML table (reuse
  `_kappa_or_none`) — accuracy + κ together (item 3). (c) Label the held-out
  diagnostics with the CA mode (item 6) and add coverage + CA-rate columns
  (items 7-8); thread the mode onto `HeldOutValidationResult`. (d) Store raw
  tp/fp/tn/fn on `CriterionErrorReport` (`_improve.py:113-123`) and render the 2×2
  / precision (item 4). (e) Decide whether the redundant-metric cluster note (§2)
  should also appear here once real agreement stats are wired in — by default it
  lives in `MetricsResult.summary()` (T2).

### P2 — recommended

#### T8. Add φ / MCC for binary criteria  *(items 2, 3 — operationalizes the paper's central result)*
- **Finding.** No φ/MCC anywhere (`matthews_corrcoef` not imported,
  `_compute.py:14-24`); binary agreement is κ only. The κ–φ gap = positive-rate
  drift (`metrics-explainer.tex:255-272, 405`) is therefore invisible without
  manual derivation.
- **How.** Add `_mcc_or_none(met_true, met_pred) -> float | None` next to
  `_kappa_or_none` (`_compute.py:755`). **Mandatory degenerate guard:** sklearn
  `matthews_corrcoef` returns a misleading **0.0** (not NaN, no exception) on
  constant/single-class input (scikit-learn issues #25258, #28982) — must detect
  `len(set(...)) < 2` explicitly and return None, per the "Undefined → None"
  invariant. Add `phi: float | None` to `CriterionMetrics` (None on the empty
  branch), aggregate `criterion_phi` (micro, from the pooled flats — return as a
  6th value from `_criterion_level_scalars`, None for multi-choice-only), and
  per-judge parity. Render φ **in the κ cluster** with the §2b note. Do **not** add
  per-criterion Pearson/Spearman/Kendall on binary vectors (item 2). Tests: φ==κ on
  matched marginals; φ>κ under drift (paper example 40/10/20/30 → κ=0.400, φ≈0.408);
  φ is None (not 0.0) on a constant vector.

#### T9. Detect & note the single-criterion correlation collapse  *(item 2 edge / §2c)*
- **Finding.** No guard for the ≤2-distinct-value score case; `_compute_correlation`
  only None-guards <3 samples / constant arrays (`_compute.py:607-643`); all three
  emitted unconditionally (`_types.py:924-935`).
- **How.** When the per-item score vector has ≤2 distinct values, attach a
  `warnings` entry / `summary()` note that Pearson=Spearman=Kendall collapse
  (Obs. 1) and report one — so the identical triple is not read as triangulation
  (`metrics-explainer.tex:226, 405`).

#### T10. Flag degenerate criteria distinctly from no-data  *(item 5)*
- **Finding.** Constant-label criterion → κ=None with `n_samples>0` and a real
  accuracy; no-data → κ=None with `n_samples=0`, accuracy=None
  (`_compute.py:1481-1531`). Distinguishable only via `n_samples`, **nothing written
  to `warnings`** for degeneracy (warning sites only at `:1112, 1116, 1204, 1219`);
  both render as `n/a` in the Kappa column. Minor: per-class P/R use
  `zero_division=0` (a mild coerce-to-0 for the absent class, not for an association
  coefficient).
- **Why.** Item 5: degenerate criteria "should be reported with their counts and
  marked NA, rather than silently dropped or coerced to zero"
  (`metrics-explainer.tex:387`).
- **How.** Append a `warnings` entry naming each criterion with `n_samples>0` and
  κ=None (degenerate single-class), and/or add an `is_degenerate` flag; document the
  `(n_samples>0, kappa=None)` vs `(n_samples=0)` distinction in docstrings.

### P3 — clarity / completeness

#### T11. Ordinal inter-judge geometry  *(item 11 / §2d)*  **[DECISION]**
- Either (a) add a one-line note in the ordinal `summary()`/docs that α (ordinal,
  distance-aware) and Fleiss (nominal) measure different geometries and are both
  intentionally retained; or (b) the cleaner long-term design: replace bare Fleiss
  for ordinal with an **ordinal-weighted** inter-judge statistic so the inter-judge
  column shares the ordinal geometry (then the gap collapses to the same
  finite-sample term as binary). statsmodels has no weighted multi-rater Fleiss, so
  (b) routes through `krippendorff` / a weighted-κ implementation.

#### T12. Remaining labeling / completeness  *(items 1, 10, 9)*
- **Item 1 (score scale):** label the score-correlation block "Score-Level Metrics
  (continuous per-item weighted score)" in `summary()` and the
  `ScoreCorrelationResult`/`score_spearman` docstrings (`_types.py:537-543, 778-782`)
  so they are not misread as binary-verdict correlations.
- **Item 10 (full triple):** consider adding macro accuracy (mean of per-criterion
  accuracies), micro κ (Cohen on pooled labels), and a categorical item-level
  pass/fail agreement (threshold the weighted score, or all-criteria-met
  conjunction, then κ/accuracy on per-item labels) so each metric is available at a
  consistently *named* level; document the item-level cluster bootstrap as bracketing
  the micro accuracy / macro κ.
- **Item 9 (weight matrix):** document the implicit disagreement-cost choice —
  nominal `as_category` uses unweighted Cohen κ (all confusions equal), ordinal uses
  quadratic weights, ordinal `as_category` is refused (NA has no ordinal position).
  The paper asks for an explicit W (`metrics-explainer.tex:395`).

---

## 4. Research resolutions (the genuine "which is better" questions)

All four resolved at **high confidence**; full sources in the workflow transcript.

1. **α vs Fleiss (T2/T11).** Observation 2 confirmed analytically and numerically
   against the real libraries (gap = `(1−κ_F)/(NR)`, negligible for typical
   N,R,κ_F). They conflate on binary/nominal-complete data → cluster + note, α
   primary (handles missing raters; level-aware). They genuinely diverge on ordinal
   (distance-aware α vs nominal Fleiss) → keep both, note geometry. Sources: paper
   Secs 5.1-5.2; Zapf et al. 2016 (BMC Med Res Methodol); Wikipedia/Krippendorff;
   metricgate comparison.

2. **φ/MCC vs κ (T8).** Complementary, not redundant: φ = marginal-invariant
   association (= Pearson/Spearman/Kendall/MCC on binary), κ = same association after
   a marginal-sensitive chance correction; `|κ|≤|φ|`, equality iff π=π̂; the gap =
   positive-rate drift. Worth adding (MEDIUM). Use sklearn `matthews_corrcoef` with a
   mandatory constant-input → None guard. Sources: paper Sec 4 + Obs 1; sklearn docs
   + issues #25258/#28982; Warrens 2014; Wikipedia (Cohen's κ).

3. **Micro/macro/item-level (T4/T12).** They estimate different quantities; mixing
   micro accuracy + macro κ unlabeled is a hazard. Best practice (paper Sec 3.1 +
   sklearn convention): name the level always, report all three when >a handful of
   criteria. (This research agent's structured output was partially malformed but its
   substance is fully corroborated by the T4 code finding.)

4. **Coverage / selective prediction (T3/T7).** A lone selective accuracy is
   incomplete/gameable; report the (coverage, selective accuracy) pair + separate
   judge/human abstention rates, gated to exclude mode, with an MCAR diagnostic when
   coverage < 1. Sources: paper subsec:selective + items 7-8; El-Yaniv & Wiener 2010;
   Geifman & El-Yaniv 2017; Chow 1970.

---

## 5. Open design decisions (resolve one at a time before implementing)

1. **T1 / `AgreementSummary`:** retire the dead type, or repurpose it as the
   mode-carrier? (Recommend: add fields to `MetricsResult`, delete `AgreementSummary`.)
2. **T2 (α/Fleiss binary/nominal):** keep Fleiss clustered-with-note, or drop the
   bare Fleiss column for binary/nominal? (Recommend: keep + cluster + note.)
3. **T11 (ordinal):** note-only, or replace bare Fleiss with an ordinal-weighted
   inter-judge statistic?
4. **T12 item-level:** is a categorical item-level pass/fail agreement meaningful for
   our rubrics (threshold vs all-criteria-met), or leave the item-level view as the
   continuous score correlations only?
