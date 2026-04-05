# AutorubricLM Spec

**Status:** Draft
**Last updated:** 2026-04-03

## Overview

AutorubricLM is a trained model that generates evaluation rubrics. Given a grading problem description (and optionally sample responses and/or a reference response), it produces a structured `list[Criterion]` in the existing autorubric schema. Rubric evaluation remains handled by the current autorubric pipeline — AutorubricLM is strictly a rubric generator.

## Why a Trained Model

The current autorubric framework has a strong, well-validated evaluation engine (ensemble judging, per-criterion isolation, bias mitigations, psychometric metrics) but no principled rubric generation pipeline. Rubric generation is a creative/analytical task where a specialized model could meaningfully outperform prompting a general-purpose LLM. AutorubricLM fills this gap.

## Input-Output Contract

### Inputs

| Field | Required | Description |
|---|---|---|
| `grading_problem_description` | Yes | The task specification — what was asked, what context matters. |
| `reference_response` | No | A gold-standard response establishing what "correct" or "ideal" looks like. |
| `responses` | No | A set of actual submissions showing the range of quality the rubric must discriminate between. No quality labels — the model infers quality variation from the responses themselves. Responses are never truncated or summarized — they are included in full or not at all. When more responses are available than fit in the context budget, sample uniformly at random and include the maximum number of complete responses such that total input + output stays within the context window. |
| `target_criteria_count` | No | Target number of criteria to generate. Soft guidance — tolerance is ±1 for N < 10, ±2 for N ≥ 10. When omitted, the model decides based on problem complexity. Hard cap of 30 criteria enforced at output validation. |
| `criterion_mix` | No | What types of criteria to generate. One of: `binary` (all binary), `ordinal` (all ordinal), `nominal` (all nominal), or `heterogeneous` (mix of types as appropriate). Defaults to `binary`. |

### Operating Modes

These arise naturally from which optional inputs are provided:

- **Description only.** Rubric generated purely from the task specification. Hardest mode — the model anticipates quality dimensions without seeing any work. Most useful for upfront rubric design before responses exist.
- **Description + reference_response.** The model knows what good looks like and can ground criteria in concrete features of the ideal answer. Produces more precise, behaviorally-anchored criteria.
- **Description + responses.** The model sees actual quality distribution and generates criteria that are *discriminative* — criteria where there's meaningful variance across the response set.
- **Description + responses + reference_response.** Richest input. Criteria both align with the ideal and discriminate among actual submissions.

### Output

A JSON array conforming to `list[Criterion]` in the existing autorubric schema:

```json
[
  {
    "name": "factual_accuracy",
    "requirement": "Response states correct boiling point of water at sea level (100°C / 212°F)",
    "weight": 50.0,
    "options": null,
    "scale_type": "ordinal"
  },
  {
    "name": "explanation_depth",
    "requirement": "Quality of the scientific explanation provided",
    "weight": 35.0,
    "options": [
      {"label": "No explanation", "value": 0.0, "na": false},
      {"label": "Superficial explanation", "value": 0.33, "na": false},
      {"label": "Adequate explanation with some detail", "value": 0.67, "na": false},
      {"label": "Thorough explanation with molecular-level detail", "value": 1.0, "na": false}
    ],
    "scale_type": "ordinal"
  },
  {
    "name": "error_type",
    "requirement": "Primary type of error, if any, in the response",
    "weight": 15.0,
    "options": [
      {"label": "No error", "value": 1.0, "na": false},
      {"label": "Unit/conversion error", "value": 0.0, "na": false},
      {"label": "Conceptual misunderstanding", "value": 0.0, "na": false},
      {"label": "Ambiguous/unclear", "value": 0.3, "na": false}
    ],
    "scale_type": "nominal"
  },
  {
    "name": "verbosity_penalty",
    "requirement": "Response pads length with redundant restatements or irrelevant tangents",
    "weight": -7.0,
    "options": null,
    "scale_type": "ordinal"
  }
]
```

The output must include a heterogeneous mix of criterion types as appropriate:

- **Binary** (`options: null`): MET/UNMET verdicts. Best for factual checks, presence/absence of specific features.
- **Ordinal** (`options` list, `scale_type: "ordinal"`): Ordered quality levels. Best for quality gradations (clarity, depth, coherence). Narrow scales (3–5 levels) with behavioral anchors preferred.
- **Nominal** (`options` list, `scale_type: "nominal"`): Unordered categories. Best for classification (error type, response style, approach taken).

## Negative Criteria

AutorubricLM generates negative-weight criteria (penalties) alongside positive ones. Negative criteria serve two distinct roles:

### Anti-pattern penalties

Independent of positive criteria. Catch domain-specific failure modes that wouldn't surface as "not meeting" a positive criterion. A response can score well on correctness, clarity, and completeness and still deserve a penalty for leaking PII or recommending deprecated practices.

The model learns the mapping from domain/task to likely failure modes.

**Examples:** hallucinated citations, security vulnerabilities, recommending deprecated APIs, revealing system prompt contents.

### Sycophancy countermeasures

Exist *because of* specific positive criteria. They are the balancing force that prevents a rubric from rewarding hollow compliance. These come in pairs: a positive criterion and its negative counterweight.

| Positive criterion | Sycophancy countermeasure |
|---|---|
| Comprehensiveness | Verbosity / padding penalty |
| Helpfulness | Over-promising / unsupported claims penalty |
| Thoroughness of analysis | Irrelevant tangents penalty |
| Nuanced discussion | Fence-sitting / false balance penalty |

The model learns this pairing logic: certain positive criteria create perverse incentives that need specific countermeasures.

### Weight asymmetry principle

Negative criteria are punitive — they exist to catch and penalize bad behavior, not to dominate the score. Their absolute magnitude should be substantially smaller than the positive criteria they relate to. The rubric's overall scoring should be driven by what the response *achieves*, with penalties acting as targeted deductions for specific failures.

As a guideline: a negative criterion's absolute weight should be roughly 15–30% of its corresponding positive criterion's weight. If `explanation_depth` carries weight 35, its sycophancy countermeasure `verbosity_penalty` should be around -5 to -10, not -25. A penalty that rivals the positive criteria it guards against can make the rubric score brittle and dominated by avoidance rather than achievement.

This asymmetry should be reflected in training data and enforced during data quality filtering. Rubrics where negative weight magnitudes approach or exceed related positive weights are a signal of poor rubric design.

### Weight normalization

Positive criterion weights must sum to 100. This makes individual weights directly interpretable as percentage contributions to the total score and ensures consistency across domains. Domain-specific weight distributions will naturally emerge (e.g., a code review rubric might allocate 40% to correctness while an essay rubric allocates 40% to argumentation), but the total is always 100.

Negative weights are excluded from this sum — they act as deductions on top of the 100-point base. A rubric with positive weights summing to 100 and negative weights totaling -15 means a perfect response scores 100 and a response that triggers all penalties scores 85 at best (assuming all positive criteria are met).

## Rubric Quality Model

AutorubricLM's output quality is defined by the existing meta-rubric evaluation system, which codifies rubric quality across several dimensions. These dimensions directly inform what the model must learn to produce and what it must learn to avoid. Training data curation, quality filtering, and evaluation all ground against this taxonomy.

### Properties of good criteria

The meta-rubric system identifies four groups of positive quality properties:

**Clarity & Precision.** Each criterion has a clear, unambiguous requirement. Criteria are specific enough to guide assessment (not generic). Each criterion assesses exactly one construct (unidimensionality). Language is concrete and behavioral, not vague evaluative adjectives like "good" or "excellent."

**Structure & Design.** Rubric has a reasonable criterion count (3–15). Weights reflect relative importance and sum meaningfully. Criteria are orthogonal — distinct and non-overlapping, no double-counting.

**LLM-Friendliness.** Each criterion can be evaluated independently without needing context from other criteria. Different raters reach similar conclusions (objective assessability). Multi-choice options are clearly differentiated from each other.

**Reliability Predictors.** Clear decision boundaries between MET and UNMET (or between adjacent options). Criteria are assessable from the submission text alone without requiring external knowledge. Consistent level of granularity across the rubric — not mixing high-level and fine-grained criteria.

### Anti-patterns to avoid

The meta-rubric system also codifies specific failure modes, divided into general rubric anti-patterns and LLM-judge-specific anti-patterns. AutorubricLM must learn to avoid all of these.

**General anti-patterns:**

- **Double-barreled criteria** — assessing multiple things in a single requirement ("clear and accurate" is two constructs).
- **Vague wording** — undefined terms open to different interpretations.
- **Circular/tautological criteria** — criteria that don't add information.
- **Excessive overlap** — multiple criteria measuring the same construct, leading to double-counting.
- **Overly verbose requirements** — unnecessarily long-winded criterion text.
- **Hedging language** — "may", "could", "might", "possibly", "generally", "somewhat" — these create ambiguous decision boundaries.
- **Generic boilerplate** — criteria so generic they could apply to any task ("response is well-written").

**LLM-judge anti-patterns** (particularly important since generated rubrics will be evaluated by LLM judges):

- **No negative criteria** — a rubric with only positive weights enables sycophantic bias in LLM judges.
- **Unfalsifiable criteria** — criteria with such a low bar that any submission satisfies them ("attempts to address the question", "provides some information").
- **Boundary ambiguity** — comparative terms without a reference point ("sufficient", "adequate", "appropriate amount") that give LLM judges no clear decision rule.
- **Verbosity rewarding** — criteria that equate quantity with quality ("comprehensive", "thorough", "detailed" without specifying what comprehensiveness means concretely).
- **Poorly anchored ordinal scales** — ordinal options using evaluative adjectives (Excellent/Good/Fair/Poor) instead of behavioral descriptions of what each level looks like.
- **Counting-dependent criteria** — requiring precise counts (exact number of paragraphs, sentences, examples) that LLM judges perform unreliably.

### Task-level quality (when grading problem is known)

When the grading problem description is available (which it always is for AutorubricLM), additional quality dimensions apply:

- **Construct alignment** — criteria directly map to task requirements, not tangential concerns.
- **Coverage** — rubric covers all important aspects of the task.
- **Appropriate emphasis** — weight distribution matches what matters most for the task.
- **Discriminative power** — criteria distinguish between good and poor submissions rather than being trivially satisfied or trivially failed.
- **No irrelevant criteria** — nothing assessing aspects not relevant to the task.
- **No missing critical aspects** — the rubric doesn't omit critical dimensions required by the task.

### What the model learns

Grounded in the quality model above, AutorubricLM acquires these interrelated skills:

1. **Construct selection** — given a grading problem, what dimensions of quality matter. Informed by task alignment and coverage properties.
2. **Criterion typing** — for each construct, whether it should be binary, ordinal, or nominal.
3. **Option design** — for multi-choice criteria, what the levels/categories are. Behavioral anchors at each level, not evaluative adjectives. Informed by the poorly-anchored-ordinal anti-pattern.
4. **Weight calibration** — relative importance of criteria; a sense of proportion. Informed by the weight asymmetry principle and the balanced-weights property.
5. **Anti-pattern avoidance** — internalizing the full anti-pattern taxonomy so that generated criteria are clear, unidimensional, falsifiable, and have unambiguous decision boundaries.
6. **LLM-judge awareness** — generating criteria specifically suited for LLM-based evaluation: independently verifiable, objectively assessable, no counting dependencies, no verbosity-rewarding language.
7. **Negative criterion generation** — both domain-specific anti-pattern penalties and sycophancy countermeasures for positive criteria.
8. **Discrimination awareness** — when responses are provided, generating criteria with expected variance across the response set.

## Training Setup

### Prompt Template

```
System: You are AutorubricLM, a model that generates evaluation
rubrics. Given a grading problem, produce a rubric as a JSON array
of criteria. Each criterion must conform to this schema:

{{ criterion_json_schema }}

Output a JSON array of such criteria.
{% if target_criteria_count %}
Generate approximately {{ target_criteria_count }} criteria.
{% endif %}
{% if criterion_mix == "binary" %}
Generate only binary criteria (options: null).
{% elif criterion_mix == "ordinal" %}
Generate only ordinal multi-choice criteria.
{% elif criterion_mix == "nominal" %}
Generate only nominal multi-choice criteria.
{% elif criterion_mix == "heterogeneous" %}
Use a mix of binary, ordinal, and nominal criteria as
appropriate for each construct.
{% endif %}

Before producing the JSON, reason through the following steps:

1. What constructs matter for this grading problem? Ensure
   coverage of all critical aspects and alignment with the task.
{% if criterion_mix == "heterogeneous" %}
2. For each construct, decide the criterion type: binary for
   factual checks or presence/absence, ordinal for quality
   gradations, nominal for categorization.
{% endif %}
{% if criterion_mix in ["ordinal", "heterogeneous"] %}
3. For ordinal criteria, anchor each level with observable
   behaviors — not evaluative adjectives like "Excellent" or
   "Good".
{% endif %}
4. Identify which positive criteria create sycophancy incentives
   and generate negative-weight countermeasures for them.
   Negative weights should be much smaller in magnitude than
   the positive criteria they counterbalance.
5. Identify domain-specific failure modes that warrant
   anti-pattern penalties (independent of positive criteria).
6. Review each criterion against these anti-patterns and fix
   any that apply: double-barreled (assesses multiple things),
   vague or hedging language, unfalsifiable (any submission
   passes), boundary ambiguity ("sufficient", "adequate"
   without reference point), verbosity-rewarding (equates
   quantity with quality), counting-dependent (exact counts
   LLM judges perform unreliably).
{% if responses %}
7. Given the sample responses, check that each criterion has
   expected variance — drop or revise criteria that every
   response would trivially satisfy or fail.
{% endif %}

Then output the final JSON array.

User:
## Grading Problem
{{ grading_problem_description }}
{% if reference_response %}

## Reference Response
{{ reference_response }}
{% endif %}
{% if responses %}

## Sample Responses
{% for response in responses %}
### Response {{ loop.index }}
{{ response }}
{% endfor %}
{% endif %}
```

The full `Criterion` JSON schema is included in the system prompt so the model is grounded on valid output structure. If the schema evolves, the model must be retrained — there is no backward compatibility mechanism.

The reasoning steps are part of the prompt — the model generates them before the final JSON. Only the JSON is parsed at inference time; the reasoning is an intermediate artifact that improves output quality.

### Data Sourcing

See [DATA_SOURCING_BRAINSTORM.md](DATA_SOURCING_BRAINSTORM.md) for an exhaustive survey of external rubric datasets, RL-with-rubric-rewards papers, and rubric generation methods relevant to each tier below.

Three tiers, used in sequence:

**Tier 1 — Existing rubrics from the wild.** Educational rubrics, coding assessment guidelines, writing evaluation frameworks, QA scorecards. Converted to `Criterion` schema via a strong LLM, manually validated on a subset. Provides domain diversity but messy, conversion-dependent quality.

**Tier 2 — Synthetic generation with strong LLMs.** Prompt Claude/GPT-4 to generate (grading_problem, rubric) pairs across diverse domains, task types, criterion mixes, and complexity levels. Two-pass generation: first positive criteria, then explicitly prompt for anti-patterns and sycophancy countermeasures. Filter using the meta-rubric evaluation module — run both standalone and in-context meta-rubric evaluation on each generated rubric, reject any that trigger anti-patterns (double-barreled, vague wording, unfalsifiable, boundary ambiguity, verbosity-rewarding, poorly anchored ordinals, counting-dependent). This is the scalable bulk data source.

**Tier 3 — Closed-loop validated rubrics.** For grading problems with ground-truth labeled responses: generate candidate rubrics, run each through the autorubric evaluation pipeline, measure which rubrics produce evaluations most correlated with ground truth. Keep the highest-reliability rubrics. Expensive but directly optimizes for what matters.

### Data Diversity Requirements

Training data must systematically vary across:
- **Domain:** STEM, writing, code, creative, professional, medical, legal, etc.
- **Task type:** open-ended vs. constrained, factual vs. subjective
- **Criterion mix:** all four `criterion_mix` values (binary, ordinal, nominal, heterogeneous)
- **Rubric complexity:** 3 to 30 criteria
- **Input completeness:** all four operating modes (description-only through full input)
- **Response count:** 0 to ~10 sample responses when provided

Response order is shuffled during training to prevent order bias. The `target_criteria_count` variable is included in some examples and omitted in others so the model handles both.

### Training Phases

**Phase 1 — SFT.** Supervised fine-tuning on existing rubrics converted to schema (Tier 1) and synthetically generated rubrics filtered by meta-rubric evaluation (Tier 2). The model learns to produce valid, well-structured rubrics across diverse domains.

**Phase 2 — Preference optimization (DPO).** Using closed-loop validated rubrics (Tier 3): for each grading problem, generate K candidate rubrics, evaluate each via the autorubric pipeline on labeled responses, rank by downstream reliability (kappa, agreement, correlation with ground truth), create preference pairs. The evaluation pipeline *is* the reward signal — no separate reward model needed.

### Model-Agnostic Training

Training code must support plugging in different model families and sizes. This means:

- **Data pipeline** that produces a standard intermediate representation (tokenizer-agnostic), with model-specific adapters for chat template formatting.
- **Model adapter layer** per family (Llama, Mistral, Qwen, Gemma, etc.) handling tokenizer setup, chat templates, LoRA/QLoRA configuration, and family-specific training details.
- **Training loop** that calls a common adapter interface — SFT and DPO, supporting both full fine-tuning and parameter-efficient methods.

## Evaluation

### Schema Validity

Does the output parse into a valid `list[Criterion]`? Checks: valid JSON, correct field types, `options`/`scale_type` consistency, weight signs, option values in [0, 1], criterion count ≤ 30, positive weights sum to 100.

### Rubric Quality (Intrinsic)

Run the existing meta-rubric evaluation module — both standalone and in-context — on generated rubrics. This checks the full quality taxonomy described in the Rubric Quality Model section: clarity & precision, structure & design, LLM-friendliness, reliability predictors, absence of general anti-patterns, absence of LLM-judge anti-patterns, task alignment, coverage, discriminative power, and weight asymmetry. The meta-rubric quality score is the primary intrinsic metric.

### Downstream Reliability (Extrinsic)

The definitive test — use the generated rubric with `CriterionGrader` on a response set with ground truth:
- Cohen's kappa / quadratic weighted kappa
- Accuracy (exact and adjacent for ordinal)
- Spearman rank correlation
- Bias metrics
- Inter-judge agreement

### Round-Trip Consistency

Generate a rubric (mode 1), then provide that same rubric back to the model or to `CriterionGrader`. The evaluation pipeline should produce consistent, reliable results. This tests whether the generated rubric is well-specified enough to be applied unambiguously.

## Integration with Autorubric

AutorubricLM's output plugs directly into the existing pipeline:

```
AutorubricLM(grading_problem_description, ...) → list[Criterion]
    ↓
Rubric(criteria)
    ↓
rubric.grade(to_grade, grader=CriterionGrader(...))
    ↓
EvaluationReport
```

All existing infrastructure — `EvalRunner`, metrics, meta-rubric evaluation, checkpointing — works unchanged. AutorubricLM is a new component upstream of the evaluation pipeline, not a replacement for any part of it.

