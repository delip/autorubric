# Metarubric Design Commitments

This document records the structural invariants and design rationale behind the autorubric metarubric system. It is intended for developers extending or modifying metarubric criteria.

## 1. Asymmetric positive/negative signals

Defect presence and quality absence are fundamentally different signals. A positive criterion like `task_aligned` (+10) rewards alignment when present; a negative criterion like `irrelevant_criteria` (-8) penalizes misalignment when detected. These are not redundant even though they assess the same concept from opposite directions:

- A rubric can be partially aligned (positive criterion MET) while still containing some irrelevant criteria (negative criterion also MET). Cross-sign pairs allow the score function to represent this mixed state.
- Collapsing them into a single criterion forces a binary verdict on a non-binary property.

Same-sign pairs that assess the same concept **are** redundant and should be consolidated (e.g., `clear_requirements` + `behavioral_language` into `unambiguous_requirements`).

## 2. Sycophancy mitigation via negative criteria

LLM judges exhibit a systematic bias toward MET on positive criteria, inflating scores. Negative-weight criteria apply downward pressure: when an anti-pattern is detected, the rubric loses points rather than merely failing to gain them.

Removing negative criteria from the metarubric degrades score discrimination. Any refactor that reduces the count or weight of negative criteria must demonstrate that sycophancy pressure has not increased (e.g., by measuring score variance on a held-out rubric corpus with known defects).

## 3. Formal properties of the score function

The metarubric score inherits the autorubric score function properties:

- **Monotonicity**: Flipping any criterion from UNMET to MET cannot decrease the score (positive weights) or increase it (negative weights).
- **Boundedness**: Score is clamped to [0, 1].
- **Decomposability**: The total score is a weighted sum of independent per-criterion verdicts divided by the sum of positive weights.
- **Bounded sensitivity**: No single criterion can move the score by more than `|weight| / sum(positive_weights)`.

Negative weights are excluded from the normalization denominator. This means negative criteria can push the score below what an all-UNMET positive rubric would produce, which is the intended mechanism for anti-pattern penalties.

## 4. Binary MET/UNMET verdicts with per-criterion decomposition

The improvement loop's revision mechanism requires per-criterion decomposition: each criterion produces an independent MET/UNMET verdict with a reason string. The `extract_issues` function identifies UNMET positive criteria and MET negative criteria as actionable issues. This decomposition would break under aggregate or ordinal scoring at the metarubric level.

CANNOT_ASSESS verdicts are handled per the configured strategy (see section 7) and do not produce actionable issues.

## 5. Self-referential consistency

The metarubric passes its own evaluation criteria. The one known exception is `generic_boilerplate`: the metarubric's own criteria are necessarily generic (they apply to all rubrics, not a specific task domain). This exception is documented rather than suppressed.

Any new metarubric criterion should be checked against the existing metarubric to verify it does not introduce a self-referential failure.

## 6. Reproducibility via master seed

Option-shuffling and few-shot selection in the meta-evaluation grader are controlled by the master seed passed to `CriterionGrader(seed=...)`. Per-call shuffle RNGs are derived from `(master_seed, content_hash, criterion_idx, judge_id)` via SHA-256, making them concurrency-safe.

This means meta-evaluation results are deterministic given the same seed, rubric content, and LLM responses. Temperature=0 does not guarantee identical LLM responses across providers, but the non-LLM randomness is fully pinned.

## 7. CANNOT_ASSESS semantics

When a meta-judge returns CANNOT_ASSESS for a criterion, four strategies control its impact on the score:

| Strategy | Behavior |
|----------|----------|
| `SKIP`   | Criterion excluded from both numerator and denominator. Score reflects only assessable criteria. |
| `ZERO`   | Treated as UNMET (positive) or MET (negative). Worst-case for the rubric. |
| `PARTIAL`| Contributes a configurable fraction of the weight (default 0.5). |
| `FAIL`   | Entire evaluation fails. Used when every criterion must be assessable. |

The default for meta-evaluation is `SKIP`. This avoids penalizing rubrics for criteria the judge cannot evaluate (e.g., task-specific criteria evaluated in standalone mode).

## 8. LLM-judge anti-pattern rationale

Each negative criterion in the metarubric targets a specific failure mode in LLM-as-a-judge evaluation:

**`no_negative_criteria`** (-6): A rubric with only positive criteria enables the judge's sycophancy bias. Without downward pressure from negative weights, scores cluster at the top of the range regardless of submission quality.

**`unfalsifiable_criteria`** (-8): Criteria that cannot be evaluated as UNMET for any plausible submission provide zero discrimination. They inflate the score floor without adding information.

**`boundary_ambiguity`** (-6): Vague thresholds (e.g., "adequately addresses") produce inconsistent verdicts across judges and runs. Criteria must specify observable, unambiguous boundaries.

**`verbosity_rewarding`** (-6): Criteria that reward length or detail count (e.g., "provides comprehensive coverage") exploit the LLM judge's verbosity bias, where longer responses receive higher scores regardless of quality.

**`poorly_anchored_ordinal`** (-6): Multi-choice criteria with poorly differentiated middle categories cause middle-category collapse: judges default to the middle option, destroying the scale's discrimination power.

**`counting_dependent`** (-6): Criteria requiring exact counting (e.g., "lists at least 5 examples") exploit a known LLM weakness. LLMs are unreliable counters, making such criteria non-deterministically assessable.

## 9. Behavioral signal path

Text-based meta-evaluation is the primary and default evaluation mode. The behavioral signal path is supplementary: it computes reward variance, judge agreement, and discrimination metrics by actually grading probe submissions with the candidate rubric, then routes these signals to the meta-judge as additional evidence.

Key design constraints:

- Behavioral signals **never replace** text-based evaluation. They are appended as a "Supplementary Behavioral Signals" section in the meta-judge prompt.
- The meta-judge decides whether and how to use them, citing relevant signals via the `evidence_cited` field on `MetaCriterionJudgment`.
- Behavioral signal computation is expensive (requires multiple grading runs). The `behavioral_signal_frequency` config controls how often signals are computed during improvement loops: `every_iter`, `first_and_last`, or `on_demand`.
- `compute_reward_variance` measures per-criterion verdict stability. High variance (approaching 0.25 for binary verdicts) indicates unreliable criteria.
- Evidence is stored in `IterationResult.evidence` and persisted in iteration artifacts for post-hoc analysis.
- `behavioral_plateau_converged` provides a convergence function that considers both quality score stability and evidence variance stability.
