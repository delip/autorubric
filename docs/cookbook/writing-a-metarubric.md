# Writing a Metarubric

Understand the built-in metarubric criterion set, and customize it for your domain when the defaults don't fit.

## The Scenario

You've been using `evaluate_rubric_standalone` and `evaluate_rubric_in_context` to vet your rubrics, and the feedback is useful. But you're starting to want more: you'd like to know *which* criteria your meta-judge is checking, *why* the criterion set is shaped the way it is, and whether you can adapt it for a specialized domain (say, medical-grading rubrics that need a stricter provenance axis).

This recipe walks through the consolidated criterion set section by section, explains the design patterns behind it, and shows how to author a custom metarubric when the built-in ones don't fit.

## What You'll Learn

- What criteria the built-in standalone and in-context metarubrics check
- Why some pairs of criteria are kept separate even though they look redundant
- How to inspect the built-in metarubric programmatically
- How to author a custom metarubric JSON and run it through the meta-evaluation pipeline

## The Built-In Criterion Set

AutoRubric ships two metarubrics:

- **`meta_rubric_standalone.json`** — judges a rubric in isolation (no task context).
- **`meta_rubric_in_context.json`** — judges a rubric together with the task prompt it's meant to evaluate.

The in-context metarubric is a superset: it contains all the standalone criteria plus sections that only make sense when a task prompt is available.

### Inspecting the Criteria

```python
from autorubric.meta import get_standalone_meta_rubric, get_in_context_meta_rubric

standalone = get_standalone_meta_rubric()
in_context = get_in_context_meta_rubric()

for criterion in standalone.rubric:
    sign = "+" if criterion.weight > 0 else ""
    print(f"  [{sign}{criterion.weight:.0f}] {criterion.name}")
```

You can always browse the JSON files directly under `src/autorubric/meta/data/`.

## Standalone Sections

### Clarity & Precision

Three positive criteria that check whether each rubric criterion communicates one observable requirement.

| Criterion | Weight | What it checks |
|---|---|---|
| `unambiguous_requirements` | +10 | Clear, concrete requirements using behavioral language, not evaluative adjectives |
| `specific_actionable` | +10 | Criteria are specific to the task, not generic boilerplate |
| `unidimensional` | +10 | Each criterion assesses exactly one construct |

`unambiguous_requirements` merges two earlier criteria (`clear_requirements` and `behavioral_language`). If your rubric uses phrases like "good," "high-quality," or "appropriate" without defining them, this criterion will fire.

### Structure & Design

Three criteria that check the rubric as a whole rather than individual items.

| Criterion | Weight | What it checks |
|---|---|---|
| `reasonable_count` | +6 | Total criterion count is in a usable range (roughly 3–15) |
| `balanced_weights` | +6 | Weight distribution reflects relative importance |
| `orthogonal_criteria` | +8 | Criteria don't redundantly measure the same quality |

### LLM-Friendliness

Criteria that catch failures specific to LLM-as-judge evaluation.

| Criterion | Weight | What it checks |
|---|---|---|
| `independently_verifiable` | +10 | Each criterion can be judged without cross-referencing others |
| `objective_assessable` | +8 | Different raters reach similar conclusions |
| `well_defined_options` | +6 | For multi-choice criteria, options are distinct from each other |

`well_defined_options` is scoped narrowly to *inter-option distinctness*. MET-vs-UNMET boundary concerns are handled by `boundary_clarity` in the next section.

### Reliability Predictors

Criteria that predict grading stability under LLM judges.

| Criterion | Weight | What it checks |
|---|---|---|
| `boundary_clarity` | +8 | Clear MET/UNMET decision boundary per criterion |
| `deterministic_assessability` | +8 | Judgable from the submission text alone |
| `consistent_granularity` | +6 | Criteria operate at similar levels of detail |
| `grounding_specified` | +8 | Factual claims specify where they should be grounded (source) |

`grounding_specified` is new. It fires when a criterion says something like "the response is factually accurate" without naming a reference source, which forces the judge to fall back on its own world knowledge.

### Anti-Patterns

Negative-weight criteria that fire when the rubric exhibits a known failure mode. A MET verdict here means the anti-pattern was detected.

| Criterion | Weight | What it checks |
|---|---|---|
| `double_barreled` | -8 | A criterion assesses multiple things ("clear AND concise") |
| `imprecise_wording` | -8 | Vague terms or hedging words that make assessment inconsistent |
| `circular_tautological` | -6 | Criterion defines quality in terms of itself |
| `excessive_overlap` | -6 | Multiple criteria measure the same quality (double-counting) |
| `overly_verbose` | -6 | Requirement text is longer than it needs to be |
| `generic_boilerplate` | -8 | Criterion could apply to any task without change |
| `unverifiable_claim` | -8 | Asks the judge to verify facts against external knowledge |

`imprecise_wording` merges earlier `vague_wording` and `hedging_language` — both are cases where different raters interpret the same criterion differently.

### LLM-Judge Anti-Patterns

A second set of anti-patterns targeting LLM-specific judge behaviors. These apply to any rubric used with an LLM judge.

| Criterion | Weight | What it checks |
|---|---|---|
| `no_negative_criteria` | -6 | Rubric has no negative-weight criteria (enables sycophancy) |
| `unfalsifiable_criteria` | -8 | Sets a bar so low that almost any submission passes |
| `boundary_ambiguity` | -8 | Uses comparative terms ("sufficient," "adequate") without anchors |
| `verbosity_rewarding` | -6 | Rewards longer responses via words like "comprehensive," "thorough" |
| `poorly_anchored_ordinal` | -6 | Multi-choice levels use "Excellent/Good/Fair" without behavioral anchors |
| `counting_dependent` | -4 | Depends on precise counting (words, paragraphs) — LLMs are unreliable here |

Details on each anti-pattern's rationale live in [Metarubric Design Commitments](../design/metarubric-design-commitments.md).

## In-Context Additions

The in-context metarubric swaps `objective_assessable` for `rater_consistency` (which absorbs the old `low_interpretation_variance`), and adds three task-aware sections.

### Construct Alignment

Does the rubric measure what the task asks for?

| Criterion | Weight | What it checks |
|---|---|---|
| `task_aligned` | +12 | Criteria map to the task prompt's stated requirements |
| `covers_key_aspects` | +10 | All important aspects of the task are represented |
| `appropriate_emphasis` | +8 | Weight distribution matches task priorities |

### Discriminative Power

Will the rubric actually separate good from bad submissions?

| Criterion | Weight | What it checks |
|---|---|---|
| `distinguishes_quality` | +10 | Materially different-quality submissions get materially different scores |
| `avoids_trivial` | +6 | No criteria that all submissions trivially pass or fail |

### Anti-Patterns (In-Context)

Task-alignment failures.

| Criterion | Weight | What it checks |
|---|---|---|
| `irrelevant_criteria` | -10 | Criteria assess aspects the task never asks for |
| `missing_critical` | -10 | Rubric fails to cover something the task explicitly requires |
| `overly_strict_requirements` | -6 | Criteria are stricter than the task demands (rigid rubric) |

`overly_strict_requirements` is new. Rigidity is distinct from misalignment: a rigid rubric may grade the right objective, just too harshly (e.g., demanding exhaustive coverage when the task asks for an overview).

## Cross-Sign Pairs: Why Some Look Redundant

At a glance you might notice pairs like:

- `orthogonal_criteria` (+8) and `excessive_overlap` (-6)
- `boundary_clarity` (+8) and `boundary_ambiguity` (-8)
- `task_aligned` (+12) and `irrelevant_criteria` (-10)

These look redundant. They aren't. A rubric can use precise, non-overlapping language on most criteria (satisfying `orthogonal_criteria`) while still having *one* pair that double-counts (firing `excessive_overlap`). The positive criterion signals average quality across the rubric; the negative criterion signals that a specific defect is present somewhere.

Removing either half collapses information. The [design commitments doc](../design/metarubric-design-commitments.md) covers the asymmetric-signal argument in more depth.

## Writing a Custom Metarubric

The built-in metarubrics are opinionated for general-purpose LLM grading. If you're grading in a specialized domain — medical reasoning, legal analysis, code correctness — you may want to extend or replace parts of the criterion set.

A metarubric is just a rubric. Any sectioned JSON that `Rubric.from_file` can load works.

```python
import json
from pathlib import Path

from autorubric import Rubric
from autorubric.graders import CriterionGrader
from autorubric.llm import LLMConfig
from autorubric.meta import get_in_context_meta_rubric

# Start from the built-in and extend it
meta_data = json.loads(
    Path("src/autorubric/meta/data/meta_rubric_in_context.json").read_text()
)

# Add a domain-specific section
meta_data["rubric"]["sections"].append({
    "name": "Medical Grading",
    "criteria": [
        {
            "name": "cites_clinical_guideline",
            "weight": 10,
            "requirement": "Each factual-accuracy criterion names the clinical guideline or evidence base against which correctness is evaluated"
        },
        {
            "name": "distinguishes_severity",
            "weight": 8,
            "requirement": "Criteria distinguish between life-threatening and minor inaccuracies"
        }
    ]
})

custom_meta = Rubric.from_dict(meta_data)
```

You can then grade any rubric against your custom metarubric by going through `Rubric.grade` directly instead of the `evaluate_rubric_*` helpers:

```python
from autorubric.meta._evaluate import MetaCriterionJudgment

grader = CriterionGrader(
    llm_config=LLMConfig(model="anthropic/claude-sonnet-4-6"),
    binary_response_format=MetaCriterionJudgment,
)

rubric_under_test = Rubric.from_file("my_medical_rubric.json")
submission = json.dumps(
    [{"name": c.name, "weight": c.weight, "requirement": c.requirement}
     for c in rubric_under_test.rubric],
    indent=2,
)

report = await custom_meta.grade(to_grade=submission, grader=grader)
```

When authoring criteria, the same rules apply that the metarubric itself enforces: one construct per criterion, behavioral language, no hedging, clear MET/UNMET boundaries. The metarubric passes its own criteria (except `generic_boilerplate`, which is exempt because metarubric criteria are intentionally task-agnostic).

## Adding Behavioral Signals

Text-based meta-evaluation doesn't catch every failure mode. A criterion that always scores MET regardless of submission quality looks fine on the page — you can only detect it by *running* the rubric on real submissions and watching what it does.

AutoRubric exposes a thin behavioral-signal path for exactly this case. See [Behavioral Signals](behavioral-signals.md) for how to compute reward variance and feed it into meta-evaluation as supplementary evidence.

## When to Customize vs. Extend vs. Replace

- **Use the built-in** when you're grading general LLM outputs and want sensible defaults.
- **Extend it** (add sections) when your domain has structural requirements the defaults don't cover — medical provenance, legal citation, code determinism.
- **Replace it** only when you genuinely need a different measurement philosophy (e.g., ordinal-only evaluation, non-LLM judges). Most users never need this.

Whatever you do, run your custom metarubric against the built-in metarubric as a sanity check. If your metarubric flags the built-in on `imprecise_wording` or `excessive_overlap`, your criteria are likely over-sensitive — tighten their definitions before you ship.
