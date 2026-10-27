# Few-Shot

Calibrate LLM judges with labeled examples for improved grading consistency.

## Overview

Few-shot learning provides the judge with graded examples before evaluation, helping calibrate its understanding of the rubric criteria. This is particularly effective for subjective criteria or domain-specific evaluation.

!!! tip "Research Background"

    Casabianca et al. (2025) and Ashktorab et al. (2025) recommend graded exemplars ("gold anchors") including negative examples of common failure modes for both human and LLM judge calibration. Few-shot examples reduce rater error and improve agreement metrics.

## Quick Example

```python
from autorubric import LLMConfig, FewShotConfig, RubricDataset
from autorubric.graders import CriterionGrader

# Load dataset with ground truth
dataset = RubricDataset.from_file("labeled_data.json")

# Split into training (for few-shot) and test
train_data, test_data = dataset.split_train_test(n_train=100, stratify=True, seed=42)

# Configure few-shot grader
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
    training_data=train_data,
    few_shot_config=FewShotConfig(
        n_examples=3,
        balance_verdicts=True,  # Balance examples across label classes (verdicts for binary, option indices for multi-choice)
        include_reason=True,    # Show each example's written reason, when its item has one
        seed=42,
    ),
)

# Grade with few-shot calibration
result = await rubric.grade(to_grade=response, grader=grader)
```

`FewShotConfig` checks its fields when it is built and raises `ValueError` for a value that
selection would misread. `n_examples` must be an integer of at least 1. When fewer training
items are eligible, all of them are shown. `seed` must be `None` or an integer, and
`balance_verdicts` and `include_reason` must be `True` or `False`. A `bool` is never accepted
as an integer, and a float such as `3.0` is refused too. Numpy integers are accepted and
stored as plain `int`. `CriterionGrader` validates the
config again and keeps its own copy, so a field changed after the config was built is refused
by the grader, and a change made after the grader was built does not affect it. Releases
before this check accepted a negative `n_examples`, which silently showed every eligible item
except the last few.

## Example Reasons

With `include_reason=True`, an example also shows why it got its label: the training item's
written reason for that criterion, from `DataItem.ground_truth_reasons`. Give an item one entry
per criterion, in the order of its `ground_truth`, with `None` where there is no reason:

```python
from autorubric import CriterionVerdict

dataset.add_item(
    submission="Water boils at 100 °C at sea level, and at lower temperatures at altitude.",
    description="Correct and qualified",
    ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
    ground_truth_reasons=["Gives 100 °C and the sea-level condition.", None],
)
```

A dataset file keeps them beside the ground truth, with `null` for `None`:

```json
{
  "submission": "Water boils at 100 °C at sea level, and at lower temperatures at altitude.",
  "description": "Correct and qualified",
  "ground_truth": ["MET", "UNMET"],
  "ground_truth_reasons": ["Gives 100 °C and the sea-level condition.", null]
}
```

An example whose item has no reason for the criterion is shown with its label alone, so without
reasons in the training data the prompts are the same whether `include_reason` is `True` or
`False`. Reasons never change which examples are selected.

A multi-choice label names one of the options the judge is shown. With `auto_na_option=True`
(the default) that includes the NA option the grader adds to a criterion without one, so an
item that `fill_ground_truth` labelled with that option can be an example, shown as that
option. With `auto_na_option=False` the judge is shown no such option, and the item is not used
as an example for that criterion. An item whose label names no option shown is left out.

## Which Criteria Get Examples

A training label answers the question its criterion poses, so examples are drawn for the
criteria of the training data's global rubric. A criterion of the rubric being graded is
shown the examples drawn for the training criterion at the same index only when the two pose
the same question, as the judge is shown each:

- both binary, with the same requirement and the same criterion type (a negative weight or
  not, since for a negative criterion MET means the submission makes the error), or
- both multi-choice, with the same requirement and the same options (labels and NA flags) in
  the same order. With `auto_na_option=True` this compares the options after the grader adds
  its NA option, so a criterion that lists that option itself still matches.

The weight's magnitude, the name, option values, `scale_type` and `aggregation` are not
shown to the judge and do not change what a label means, so changing them keeps the
examples. A graded criterion that poses another question, such as one with reordered or
relabelled options, gets no examples, and so does a criterion beyond the length of the
training rubric. The same rule applies to a training item with its own rubric
(`DataItem.rubric`), whose labels answer that rubric's questions. Its label for a criterion
is used only when its own criterion at that index poses the training rubric's question.
Both rubrics are compared as the judge is shown them, and the training rubric as it was when
the grader was built: like the examples, which are drawn then, the comparison does not change
when the training rubric is replaced or edited afterwards.

A graded criterion that gets no examples this way raises a `UserWarning` when it is graded,
under either `llm_calls` setting. The warning names the criterion (its id, such as `c1`, and
its name) and says why: it differs from the training rubric's criterion at its index in its
requirement, criterion type or options, it is of the other kind (binary or multi-choice), or
the training rubric has no criterion at its index. A grader warns once per criterion index
and graded question, not once per item, and warns again if a later grade puts another
question at that index. A training item whose own rubric lends no label does not warn,
because that item is training data the grader was built with, not part of the rubric being
graded. Its label is left out of the examples for that criterion, as a missing label is.

```text
UserWarning: Few-shot examples are not shown for criterion c1 ('tone'): it differs from the training rubric's criterion at its index in its options, so the training labels do not answer it
```

When the graded rubric is the training rubric, every criterion gets its examples. Releases
before this check showed the training criterion's examples for whatever criterion stood at
that index. A different binary criterion got another criterion's verdicts, reordered options
got the wrong option labels, and fewer options raised `KeyError` while the prompt was built.

## Ensemble + Few-Shot

Few-shot works orthogonally with ensemble mode. Each LLM judge gets its own examples, drawn from
the training data per criterion and per `judge_id` with a generator seeded by `FewShotConfig.seed`
(the grader's `seed` when unset). Judges therefore generally see different examples in a
different order, and a fixed seed draws the same ones again:

```python
from autorubric.graders import JudgeSpec

grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini"),
        JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
    ],
    aggregation="majority",
    training_data=train_data,
    few_shot_config=FewShotConfig(n_examples=3),
)
```

## Item-Level Examples Under `llm_calls="per_item"`

`llm_calls` is a `CriterionGrader` setting, unrelated to `compute_metrics(per_item_metrics=...)`
and to per-item rubrics: it names how many LLM calls a judge makes per graded item. Under
`CriterionGrader(llm_calls="per_item")`, where each LLM judge grades an item's whole rubric in
one call, few-shot selection changes shape. Instead of drawing examples per criterion, an
example is a whole training item, shown once in the call with its label on every criterion:

```python
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
    llm_calls="per_item",
    training_data=train_data,
    few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=True, seed=42),
)
```

- **Eligibility.** An item is shown only if every one of its ground-truth labels resolves for
  its criterion — a verdict (or its value, such as `"MET"`) for a binary criterion, an option
  for a multi-choice one. An item with a label that does not resolve, even for one criterion,
  is not used as an example at all. A multi-choice label resolves among the options the judge is
  shown, so with `auto_na_option=True` (the default) the NA option the grader adds counts
  too: an item that `fill_ground_truth` labelled with that option is a whole example. An
  item with its own rubric has a label only for the criteria at which its rubric poses the
  training rubric's question (see *Which Criteria Get Examples* above), so its example shows
  no judgment for the others; it is eligible when it has at least one label. Labels for
  criteria beyond the training rubric are not read.
- **Judgments shown.** An example has a judgment only for the graded criteria that pose the
  training rubric's question at their index and that it has a label for. A criterion that poses another question, or one
  the training rubric lacks, gets no judgment, and an example left with no judgment is not
  shown.
- **`n_examples`** counts training items per call, not per criterion: `n_examples=3` shows 3
  whole items in each call, however many criteria the rubric has.
- **`balance_verdicts=True`** (the default) selects by greedy label coverage rather than
  per-criterion balance, which is impossible once every example carries every criterion's
  label: each pick is the item that shows the most (criterion, label) pairs no already-picked
  item shows, ties broken by the seeded order; no item's submission is ever shown twice; once
  no remaining item adds a new pair, the rest of the slots are filled in the seeded order.
- **`balance_verdicts=False`** takes the first items of that seeded order.
- **Multi-choice labels** are numbered in the call's shuffled option order, matching how that
  criterion's options are shown.
- **`include_reason`** still shows each example's `DataItem.ground_truth_reasons` entry for
  the criterion, where the item has one.
- **The draw is deterministic** for a given seed and judge: each judge (including a cascade's
  escalation judges) draws its own examples, so a panel's judges generally see different items
  in a different order. The examples are taken when the grader is built, as per-criterion
  examples are: changing a training item afterwards changes no prompt.
- **Cost.** Each call carries k example submissions (`n_examples`), against N × k submissions
  per item under the default `llm_calls="per_criterion"`, where N is the number of criteria.

---

## FewShotConfig

Configuration for few-shot example selection.

::: autorubric.FewShotConfig
    options:
      show_source: true
      members_order: source

---

## FewShotExample

A single few-shot example with submission and ground truth verdict.

::: autorubric.FewShotExample
    options:
      show_source: true
      members_order: source

---

## References

Ashktorab, Z., Daly, E. M., Miehling, E., Geyer, W., Santillán Cooper, M., Pedapati, T., Desmond, M., Pan, Q., and Do, H. J. (2025). EvalAssist: A Human-Centered Tool for LLM-as-a-Judge. arXiv:2507.02186.

Casabianca, J., McCaffrey, D. F., Johnson, M. S., Alper, N., and Zubenko, V. (2025). Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications. arXiv:2501.02334.
