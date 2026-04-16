# Changelog

## [Unreleased]

### Changed
- Consolidated same-sign redundant metarubric criteria: `clear_requirements` + `behavioral_language` -> `unambiguous_requirements`, `vague_wording` + `hedging_language` -> `imprecise_wording`, `objective_assessable` + `low_interpretation_variance` -> `rater_consistency` (in-context only)
- Narrowed `deterministic_assessability` to text-only assessability; source-grounding concerns moved to new `grounding_specified` criterion
- Narrowed `well_defined_options` to inter-option distinctness
- Sharpened `distinguishes_quality` to explicitly name the low-signal failure mode

### Added
- New metarubric criteria: `grounding_specified` (+8), `unverifiable_claim` (-8), `overly_strict_requirements` (-6, in-context only)
- `compute_reward_variance()` for measuring per-criterion verdict stability across repeated evaluations
- Behavioral evidence path: `evidence` parameter on `evaluate_rubric_standalone` and `evaluate_rubric_in_context`
- `MetaCriterionJudgment.evidence_cited` for tracking which behavioral signals informed a judgment
- `ImprovementConfig.evidence_fn` and `behavioral_signal_frequency` for periodic behavioral evidence computation during improvement loops
- `behavioral_plateau_converged` convergence function considering both quality and evidence stability
- `IssueDetail.signal_source` and `IterationResult.evidence` for tracking behavioral signal provenance
- Cookbook recipes: grounded-rubrics, behavioral-signals, behavioral-improvement-loop
- Design rationale document: `docs/design/metarubric-design-commitments.md`
