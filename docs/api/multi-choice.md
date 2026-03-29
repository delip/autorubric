# Multi-Choice

Ordinal and nominal scales beyond binary MET/UNMET verdicts.

## Overview

Multi-choice criteria support evaluation beyond binary verdicts:

- **Ordinal scales**: Satisfaction ratings, quality levels with ordered values
- **Nominal scales**: Categorical judgments where options may share values
- **NA options**: Options excluded from scoring

!!! tip "Research Background"

    Multiple sources recommend low-precision ordinal scales (0-3 or 1-5) rather than high-precision numeric scales (1-10). Broad scales invite central-tendency and anchoring problems. Multi-choice criteria with explicit option values provide clear behavioral anchors (Kim et al., 2024; Zheng et al., 2023).

## Ordinal Scale Example

```yaml
# rubric.yaml
- name: satisfaction
  requirement: "How satisfied would you be with this response?"
  weight: 10.0
  scale_type: ordinal
  options:
    - label: "1"
      value: 0.0
    - label: "2"
      value: 0.33
    - label: "3"
      value: 0.67
    - label: "4"
      value: 1.0
```

## Nominal Scale Example

```yaml
- name: efficiency
  requirement: "Is the number of exchange turns appropriate?"
  weight: 5.0
  scale_type: nominal
  options:
    - label: "Too few interactions"
      value: 0.0
    - label: "Too many interactions"
      value: 0.0
    - label: "Just right"
      value: 1.0
```

## NA Options

Exclude options from scoring (like CANNOT_ASSESS for binary):

```yaml
options:
  - label: "None"
    value: 0.0
  - label: "All claims"
    value: 1.0
  - label: "NA - No references provided"
    na: true
```

## Ensemble Aggregation

```python
from autorubric.graders import CriterionGrader

grader = CriterionGrader(
    judges=[...],
    aggregation="majority",           # For binary criteria
    ordinal_aggregation="mean",       # "mean", "median", "weighted_mean", "mode"
    nominal_aggregation="mode",       # "mode", "weighted_mode", "unanimous"
)
```

## Position Bias Mitigation

LLM judges exhibit position bias in multi-choice settings. AutoRubric shuffles options by default:

```python
# Default: shuffling enabled, seed auto-generated
grader = CriterionGrader(llm_config=config)

# Pin the seed for reproducible shuffles
grader = CriterionGrader(llm_config=config, seed=42)

# Disable shuffling entirely
grader = CriterionGrader(llm_config=config, shuffle_options=False)
```

The shuffle order for each criterion is recorded in `CriterionReport.shuffle_order` and persisted in experiment checkpoints.

## Ground Truth Format

```python
dataset.add_item(
    submission="Response text...",
    description="Good response",
    ground_truth=[
        CriterionVerdict.MET,  # Binary criterion
        "4",                    # Multi-choice ordinal
        "Just right",           # Multi-choice nominal
    ]
)
```

---

## CriterionOption

Single option for multi-choice criteria.

::: autorubric.CriterionOption
    options:
      show_source: true
      members_order: source

---

## MultiChoiceVerdict

Verdict for a multi-choice criterion.

::: autorubric.MultiChoiceVerdict
    options:
      show_source: true
      members_order: source

---

## AggregatedMultiChoiceVerdict

Aggregated verdict from ensemble for multi-choice criteria.

::: autorubric.AggregatedMultiChoiceVerdict
    options:
      show_source: true
      members_order: source

---

## MultiChoiceJudgment

LLM judgment for a multi-choice criterion.

::: autorubric.MultiChoiceJudgment
    options:
      show_source: true
      members_order: source

---

## MultiChoiceJudgeVote

Individual judge vote for multi-choice criteria in ensemble.

::: autorubric.MultiChoiceJudgeVote
    options:
      show_source: true
      members_order: source

---

## OrdinalAggregation

Aggregation strategy for ordinal multi-choice criteria in ensemble.

::: autorubric.OrdinalAggregation
    options:
      show_source: true

---

## NominalAggregation

Aggregation strategy for nominal multi-choice criteria in ensemble.

::: autorubric.NominalAggregation
    options:
      show_source: true

---

## References

Kim, S., Suk, J., Longpre, S., Lin, B. Y., Shin, J., Welleck, S., Neubig, G., Lee, M., Lee, K., and Seo, M. (2024). Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4334–4353.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685.
