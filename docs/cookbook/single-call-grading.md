# Holistic Rubric grading

!!! warning "Not recommended: offered mainly for completeness"
    We recommend the default, one call per criterion, and offer holistic grading mainly for completeness. When one call judges every criterion, the criteria share a context window and their verdicts can shift each other; AutoRubric dropped this mode at its fork from the upstream `rubric` library for that reason. The live comparison in this recipe shows the shift is real: on a RiceChem sample the single call marked 45% of cells MET, against 56% for per-criterion calls and 56% in the ground truth, and every holistic run had a lower kappa and a higher score error than every per-criterion run. A single call also fails as a unit, so one failed call leaves every criterion of the item ungraded, and one thinking budget covers the whole rubric instead of each criterion. Use it only where its lower cost matters more than these risks, and only after Step 3 shows no shift you care about on your own labelled data.

Let an LLM judge answer every criterion of an item in one call instead of one call per criterion, and check on labelled data that the cheaper calls still grade the way you need.

## The Scenario

You grade a few thousand student answers each term against an eight-criterion rubric with an LLM judge. By default the judge makes one call per criterion, and every call re-sends the grading instructions and the whole answer, so each answer is sent eight times. A provider's prompt cache discounts the repeats, but the judge's output is still billed in full. `CriterionGrader(llm_calls="per_item")` asks the same judge for all eight verdicts in a single call. It is clearly cheaper; what you need to know is whether it grades as well. Teaching assistants have labelled a few hundred answers criterion by criterion, and you want to check the single call against them before switching.

## What You'll Learn

- Switching a judge, a panel or a cascade to one call per item with `llm_calls="per_item"`
- What a single call changes: tokens and cost, latency, `max_tokens`, failure scope
- Validating the switch on labelled data with `compute_metrics`, the judge's MET rate, and verdict agreement within and across modes over replicate runs
- What a live comparison on an essay set and a RiceChem sample found: cheaper and faster on both, stricter on RiceChem
- Deciding when one call per item is a good trade, and what to watch after switching

## The Solution

One call per item is a cost tool. It pays off when the single call grades your rubric the way per-criterion calls do, and whether it does depends on the rubric and the model. This recipe measures that on labelled data before anything is switched over.

### Step 1: Switch the Judge to One Call per Item

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader

grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-5.6-luna"),
    llm_calls="per_item",  # the default is "per_criterion"
)
report = await rubric.grade(to_grade=answer, grader=grader, query=question)  # one call
```

`llm_calls` applies to every LLM judge of the grader and never to a decision-model judge: each judge of a panel (`judges=[...]`) makes one call per item, and so does a [cascade](cascade-grading.md)'s escalation judge for an item with any escalated criterion. [Enabling it](llm-judges.md#enabling-it) covers these cases and the settings it cannot be combined with.

The name counts LLM calls per item. It is unrelated to [per-item rubrics](per-item-rubrics.md), where each dataset item carries its own `Rubric`, and to `compute_metrics(per_item_metrics=...)`.

### Step 2: Know What Changes

The judge gets the same per-criterion instructions, wrapped for a whole rubric: the system prompt embeds the instructions for each kind of criterion the rubric has, the user prompt lists every criterion under an id (`c0`, `c1`, ...) and the submission once, and the answer holds one structured judgment per criterion. [Grading a whole rubric in one call](llm-judges.md#grading-a-whole-rubric-in-one-call) describes the call in full, including its cost model and how few-shot examples and thinking traces work under it. What matters for the switch:

- **Tokens and cost.** The system prompt and the submission are sent once per item instead of once per criterion, so input tokens no longer multiply with the number of criteria. Output shrinks less, since every criterion still gets an explanation.
- **Latency.** One request per item instead of one per criterion raises throughput under a concurrency cap or a rate limit, but one response writes every criterion's explanation in turn, so a single item can take longer.
- **`max_tokens` and `timeout`.** Output grows with the rubric, about 80-100 tokens per criterion plus any thinking. Leave `max_tokens` unset, or size it at about 100 tokens per criterion plus the thinking budget: a reply cut off at `max_tokens` is not valid JSON and fails every criterion of the item. The default 60-second `timeout` can also be too short for a thinking model.
- **Failure scope.** A failed call fails every criterion of the item, and a single judge's item then has no score. A missing or unusable answer for one criterion still fails that criterion alone.

### Step 3: Validate Before Switching

Grade the same labelled sample with both modes, twice each, and hold everything else fixed: the same model and settings, the same grader `seed`, and the response cache off, as it is by default, so that a mode's second run does not replay its first. The seed pins multi-choice option order, which is the same in both modes. It also makes each mode's few-shot selection reproducible, but the two modes select examples differently (a `per_item` example is a whole training item), so with `training_data` the comparison also changes the examples the judge sees.

The second run is what makes the comparison readable. An LLM judge is not deterministic: a reasoning model samples at its provider's default temperature, which is what it gets when `temperature` is unset (AutoRubric then sends none), and even at a low temperature most providers do not guarantee identical outputs. Two runs of the same mode therefore disagree on some verdicts, and a difference between the modes means something only if it is larger than that.

```python
from autorubric import LLMConfig, RubricDataset, evaluate
from autorubric.graders import CriterionGrader

dataset = RubricDataset.from_file("labelled_answers.json")  # items with ground truth
judge = LLMConfig(model="openai/gpt-5.6-luna", max_parallel_requests=16)
MODES, REPLICATES = ("per_criterion", "per_item"), (1, 2)

runs = {}
for rep in REPLICATES:
    for mode in MODES:
        grader = CriterionGrader(judge_model_config=judge, llm_calls=mode, seed=42)
        runs[mode, rep] = await evaluate(dataset, grader, show_progress=False)
```

Each `evaluate()` call without an `experiment_name` is a new experiment, so re-running the block grades again. If you name the runs, remember that `evaluate()` resumes an experiment of the same name whenever the dataset's name, prompt, number of items and number of criteria match, without checking the model, its settings, the rubric's wording or the labels, and that a resumed run does not retry the items that failed: give each configuration its own names.

Score each run against the labels, and compare how often the judge says MET with how often the labels do:

```python
from autorubric import CriterionVerdict, compute_metrics

MET, UNMET = CriterionVerdict.MET, CriterionVerdict.UNMET


def verdicts(result):
    """Final verdict of each judged binary (item, criterion) cell; failed judgments left out."""
    return {
        (r.item_idx, c): cr.final_verdict
        for r in result.item_results
        if r.error is None
        for c, cr in enumerate(r.report.report)
        if not cr.is_error and cr.final_verdict is not None
    }


def fmt(x, spec=".3f"):
    return "n/a" if x is None else format(x, spec)


truth = [v for item in dataset for v in item.ground_truth]
print(f"ground truth: MET rate {truth.count(MET) / len(truth):.1%}")
for (mode, rep), result in runs.items():
    metrics = compute_metrics(result, dataset)
    judged = list(verdicts(result).values())
    print(
        f"{mode}#{rep}: accuracy {fmt(metrics.criterion_accuracy)}, "
        f"kappa {fmt(metrics.mean_kappa)}, MET rate {judged.count(MET) / len(judged):.1%}, "
        f"cost ${fmt(result.total_completion_cost, '.4f')}"
    )
```

Then compare the runs with each other, cell by cell: two runs of the same mode give the run-to-run baseline, and runs of different modes give the mode effect on top of it. The direction of the cross-mode disagreements tells you whether one mode is stricter.

```python
import itertools

from sklearn.metrics import cohen_kappa_score


def agreement(a, b):
    """Share of the cells both runs judged on which they agree, and Cohen's kappa."""
    va, vb = verdicts(a), verdicts(b)
    cells = sorted(va.keys() & vb.keys())
    x, y = [va[k].value for k in cells], [vb[k].value for k in cells]
    return sum(p == q for p, q in zip(x, y)) / len(cells), cohen_kappa_score(x, y)


for (m1, r1), (m2, r2) in itertools.combinations(runs, 2):
    kind = "within" if m1 == m2 else "across"
    share, kappa = agreement(runs[m1, r1], runs[m2, r2])
    print(f"{kind} modes, {m1}#{r1} vs {m2}#{r2}: {share:.1%} agree, kappa {kappa:.2f}")

for rep in REPLICATES:
    a, b = verdicts(runs["per_criterion", rep]), verdicts(runs["per_item", rep])
    cells = a.keys() & b.keys()
    stricter = sum(a[k] == MET and b[k] == UNMET for k in cells)
    laxer = sum(a[k] == UNMET and b[k] == MET for k in cells)
    print(f"run {rep}: per_item UNMET where per_criterion MET: {stricter}; the reverse: {laxer}")
```

The snippets compare binary verdicts and skip multi-choice criteria; to compare those too, key their cells on `cr.final_multi_choice_verdict.selected_index` instead of `cr.final_verdict`. Read the output this way:

- **Cross-mode agreement clearly below both within-mode agreements** means the modes grade differently, beyond run-to-run noise. A gap of a few cells on a small sample is not enough, as the essay set in Step 4 shows, and agreement within the range of the within-mode pairs means this sample cannot tell them apart.
- **A MET rate that moves the same way in both runs**, with the disagreements lopsided in one direction, says which way the single call leans.
- **Accuracy and kappa against the labels** say whether the shift costs or gains agreement with your graders. Put an interval on the difference before reading much into it: in Step 4, on 100 answers, the 95% interval of the accuracy difference was [−0.8, +3.3] points around +1.3, and its width depends on your sample. `compute_metrics(..., bootstrap=True)` gives each run's confidence intervals, and the companion script in the appendix computes a paired, item-level interval for the difference.
- **`metrics.per_criterion`** shows where the shift lands: it can help some criteria and hurt others.

!!! tip "Grade one item first"
    Before paying for four full runs, grade one item with each mode and check that no criterion failed (`cr.error is None` on every criterion report). A judge that is misconfigured fails every call and, by default, the run still finishes; `evaluate(..., fail_fast=True)` stops it at the first failed item instead. See the LiteLLM warning in Step 4.

### Step 4: What a Live Comparison Found

We ran Step 3 on 2026-10-30 on two labelled datasets that ship in `examples/data/`, with `openai/gpt-6-luna` at a low reasoning effort. The measured configuration, for each `mode`:

```python
judge = LLMConfig(
    model="openai/gpt-6-luna",
    thinking="low",               # sent as reasoning_effort="low"
    max_parallel_requests=16,
    cache_enabled=False,          # replicates must not replay each other's answers
    extra_params={"allowed_openai_params": ["reasoning_effort"]},  # see the warning below
)
grader = CriterionGrader(judge_model_config=judge, llm_calls=mode, seed=42)
```

`temperature` and `max_tokens` were left unset, so no temperature and no output cap were sent. Each mode ran twice, and the eight runs ran one after another: for each dataset, the first run of each mode, then the second. The whole experiment, including a one-item smoke test, cost about $0.23.

| Dataset | Items | Criteria | Ground truth | Calls per run (per_criterion / per_item) |
|---|---|---|---|---:|
| Essay grading (`essay_grading_dataset.json`) | All 11 | 5 binary: `Causes`, `Effects`, `Structure`, `Britain`, `Errors` | 55 cells, 4 of them `CANNOT_ASSESS`; MET rate 41.8% | 55 / 11 |
| RiceChem Q1 (`ricechem/q1.json`) | 100 of 327, sampled with `random.Random(0).sample(range(327), 100)` | 8 binary | Teaching assistants' labels ([Sonkar et al., 2024](https://arxiv.org/abs/2404.14316)), 800 cells; MET rate 56.1% | 800 / 100 |

!!! warning "LiteLLM and `reasoning_effort` on new OpenAI models"
    LiteLLM 1.95.0 recognizes OpenAI reasoning models by name (`gpt-5` and the o-series). For `openai/gpt-6-luna` it therefore refuses `reasoning_effort`, which `thinking="low"` sends, with `UnsupportedParamsError`, raised client-side before any request is made. So the judge's first call fails, and the grade raises `ValueError` naming the judge ([Parameters LiteLLM refuses](llm-judges.md#parameters-litellm-refuses), [#40](https://github.com/delip/autorubric/issues/40)). Before that fix, such a judge abstained on every criterion, and in a panel it contributed nothing without any item failing. `extra_params={"allowed_openai_params": ["reasoning_effort"]}` lets the parameter through.

How to read the tables and figures below:

- Each table gives both runs of each mode. Where the text quotes one number per mode, it is the mean of that mode's two runs, and a ratio is a ratio of those means.
- Accuracy is `criterion_accuracy`, kappa is `mean_kappa` and score RMSE is `score_rmse`, all from `compute_metrics` with its defaults, which leave out cells whose verdict or ground truth is `CANNOT_ASSESS`. MET rates count every cell.
- An accuracy difference with an interval is item-level: each item's accuracy is averaged over its mode's two runs, the items' differences are averaged, and items are resampled 5,000 times for a 95% bootstrap interval. It weights every item equally, so where items have different numbers of labelled cells (the essay set) it differs slightly from the difference of the pooled accuracies.
- Recorded cost is what LiteLLM reported. List-price cost bills every prompt token at the uncached rate: $0.10 per million prompt tokens and $0.50 per million completion tokens.
- Cached share is the share of a run's prompt tokens that OpenAI reported reading from its prompt cache. For this model those tokens cost $0.01 per million, and tokens written to the cache cost $0.125 per million. `TokenUsage` does not record these counts for OpenAI models, so we read them, and reasoning tokens, from each response's usage details.
- Wall-clock time is each run's elapsed time. Values are rounded half up.

**RiceChem Q1, 100 answers × 8 criteria**

| | per_criterion #1 | per_criterion #2 | per_item #1 | per_item #2 |
|---|---:|---:|---:|---:|
| Criterion accuracy | 0.721 | 0.720 | 0.730 | 0.738 |
| Mean kappa | 0.482 | 0.485 | 0.471 | 0.490 |
| Score RMSE | 0.168 | 0.155 | 0.199 | 0.196 |
| Judge MET rate (ground truth 56.1%) | 52.0% | 53.1% | 47.1% | 46.9% |
| Prompt tokens | 1,892,828 | 1,892,828 | 302,941 | 302,941 |
| Cached share of prompt tokens | 96.6% | 98.5% | 76.3% | 99.9% |
| Completion tokens | 90,643 | 92,210 | 65,073 | 64,058 |
| Recorded cost | $0.0709 | $0.0677 | $0.0438 | $0.0351 |
| List-price cost | $0.2346 | $0.2354 | $0.0628 | $0.0623 |
| Wall-clock | 98.9 s | 101.5 s | 48.4 s | 53.3 s |

**Essay grading, 11 essays × 5 criteria**

| | per_criterion #1 | per_criterion #2 | per_item #1 | per_item #2 |
|---|---:|---:|---:|---:|
| Criterion accuracy | 0.863 | 0.843 | 0.902 | 0.863 |
| Mean kappa | 0.769 | 0.726 | 0.803 | 0.735 |
| Score RMSE | 0.317 | 0.336 | 0.330 | 0.352 |
| Judge MET rate (ground truth 41.8%) | 32.7% | 30.9% | 36.4% | 32.7% |
| Prompt tokens | 118,847 | 118,847 | 30,255 | 30,255 |
| Cached share of prompt tokens | 96.4% | 97.6% | 83.8% | 99.9% |
| Completion tokens | 3,739 | 3,619 | 3,655 | 3,417 |
| Recorded cost | $0.0035 | $0.0033 | $0.0027 | $0.0020 |
| List-price cost | $0.0138 | $0.0137 | $0.0049 | $0.0047 |
| Wall-clock | 6.6 s | 6.3 s | 5.9 s | 4.9 s |

No run had a failed item or a failed criterion. One verdict was `CANNOT_ASSESS`, in essay run `per_criterion` #1, on a cell the ground truth also labels `CANNOT_ASSESS`.

**Verdict agreement between runs** (cells on which two runs gave the same verdict, and Cohen's kappa between them)

| Runs compared | Kind | Essay (55 cells) | RiceChem (800 cells) |
|---|---|---:|---:|
| per_criterion #1 vs #2 | within a mode | 53 (96.4%), κ 0.92 | 709 (88.6%), κ 0.77 |
| per_item #1 vs #2 | within a mode | 53 (96.4%), κ 0.92 | 742 (92.8%), κ 0.85 |
| per_criterion #1 vs per_item #1 | across modes | 50 (90.9%), κ 0.80 | 685 (85.6%), κ 0.71 |
| per_criterion #1 vs per_item #2 | across modes | 50 (90.9%), κ 0.80 | 683 (85.4%), κ 0.71 |
| per_criterion #2 vs per_item #1 | across modes | 52 (94.5%), κ 0.88 | 684 (85.5%), κ 0.71 |
| per_criterion #2 vs per_item #2 | across modes | 52 (94.5%), κ 0.87 | 684 (85.5%), κ 0.71 |

#### RiceChem: cheaper, faster, and stricter

**Cost and time.** Per run, `per_item` sent 6.2x fewer prompt tokens and 1.4x fewer completion tokens than `per_criterion`. It cost 1.8x less as recorded ($0.0394 against $0.0693; 1.6x between the first runs and 1.9x between the second) and 3.8x less at list price ($0.0626 against $0.2350), and it finished 2.0x sooner (50.9 s against 100.2 s).

Completion tokens fell less than prompt tokens, since every criterion still gets its own explanation. Reasoning accounts for about two thirds of the drop: one pass now covers the whole rubric, at about 24,500 reasoning tokens per run against about 42,400.

The recorded ratio is smaller than the list-price one because caching discounts prompt tokens alone. `per_criterion` sends the system prompt and the answer first, so an item's calls share them, and OpenAI served 96.6% and 98.5% of its prompt tokens from the cache. Completion tokens then made up about two thirds of its recorded bill. Each mode's second run cost less than its first, since it read what earlier runs had written and wrote nothing new to the cache. Recorded costs therefore depend on what ran before.

**The single call graded stricter.** `per_item` marked 47% of cells MET, against 53% for `per_criterion` and 56% in the ground truth, and did so in both runs. Comparing the two modes' first runs with each other, and their second runs with each other, they disagreed on 231 of 1,600 cells. On 160 of them `per_criterion` said MET and `per_item` UNMET, and on 71 the reverse. The ground truth sided with `per_criterion` on 105 of the 231 and with `per_item` on 126. Stricter calls traded recall for precision: pooled over both runs, MET precision rose from 0.77 to 0.81 and MET recall fell from 0.72 to 0.68.

**Beyond run-to-run noise.** Two runs of the same mode agreed on 88.6% of cells (κ 0.77) for `per_criterion` and 92.8% (κ 0.85) for `per_item`, while runs of different modes agreed on 85.4% to 85.6% (κ 0.71). Every cross-mode pair agreed less than either within-mode pair, so the shift is larger than the disagreement between two runs of the same mode. With one pair of runs per mode, that baseline was measured once.

**Accuracy and kappa level; score error worse in every run.** Criterion accuracy was 0.734 for `per_item` against 0.721 for `per_criterion`, an item-level difference of +1.3 points with a 95% bootstrap interval of [−0.8, +3.3], which is not significant on 100 answers. Mean kappa was 0.48 for both, with the runs overlapping (0.471 and 0.490 against 0.482 and 0.485). Score RMSE was 0.197 against 0.161, and each `per_item` run had a higher RMSE than each `per_criterion` run, but no interval was computed for that difference. Every RiceChem criterion has a positive weight, so stricter verdicts pull item scores down. Averaged over each mode's two runs, the two modes' item scores correlate at r = 0.907.

**Where the shift lands.** The mode columns pool both runs, 200 verdicts per criterion; the ground truth covers the 100 sampled answers.

| Criterion | Ground truth MET | per_criterion MET | per_item MET | per_criterion accuracy | per_item accuracy |
|---|---:|---:|---:|---:|---:|
| `decreased_repulsion` | 82.0% | 80.0% | 80.0% | 0.960 | 0.960 |
| `repulsion_potential_energy` | 57.0% | 27.5% | 23.0% | 0.665 | 0.630 |
| `same_core_charge` | 47.0% | 46.0% | 46.0% | 0.860 | 0.920 |
| `same_shell_radius` | 54.0% | 15.0% | 20.0% | 0.610 | 0.650 |
| `higher_core_charge` | 53.0% | 57.0% | 46.0% | 0.810 | 0.790 |
| `smaller_radius` | 83.0% | 67.0% | 59.5% | 0.780 | 0.745 |
| `full_pe_ie_explanation` | 52.0% | 42.5% | 42.0% | 0.825 | 0.860 |
| `partial_pe_ie_explanation` | 21.0% | 85.5% | 59.5% | 0.255 | 0.315 |

`per_item` lowered the MET rate on five of the eight criteria, raised it on `same_shell_radius`, and matched it on `decreased_repulsion` and `same_core_charge`. It lost 2.0 to 3.5 points of accuracy on three: `repulsion_potential_energy` and `smaller_radius`, on which `per_criterion` already marked MET well below the ground truth's rate, and `higher_core_charge`, which `per_criterion` called at about the right rate. It gained 3.5 to 6.0 points on four, including `partial_pe_ie_explanation`, which both modes over-call. With 100 answers per criterion, none of these per-criterion differences was tested for significance.

**The full and partial explanation criteria.** `full_pe_ie_explanation` reads "correctly explains relationship of potential energy to ionization energy" and `partial_pe_ie_explanation` "partially explains relationship between potential energy and ionization energy". The teaching assistants treated them as mutually exclusive: none of the 327 Q1 answers has both MET (175 have only the full one, 56 only the partial one, 96 neither). The rubric does not say so, and neither mode graded them that way:

| (full, partial) | Ground truth (100 answers) | per_criterion (200 verdict pairs) | per_item (200 verdict pairs) |
|---|---:|---:|---:|
| MET, UNMET | 52 | 1 | 0 |
| UNMET, MET | 21 | 87 | 35 |
| UNMET, UNMET | 27 | 28 | 81 |
| MET, MET | 0 | 84 | 84 |

The ground truth's most common pattern, full credit without partial credit, almost never appears in either mode: both judges treat a full explanation as a partial one too. The two modes mark both MET equally often (84 of 200), so `per_item`'s better accuracy on `partial_pe_ie_explanation` (0.315 against 0.255) comes from marking both UNMET far more often (81 against 28), part of its general strictness, rather than from treating the pair as exclusive. In this comparison, seeing both criteria in the same call did not make the judge apply a relation the rubric does not state.

#### Essay set: no clear difference overall

On the essay set the modes' overall results overlap, and one criterion holds most of the gap between them:

- **MET rate.** `per_item` marked 34.5% of cells MET against 31.8% for `per_criterion` (ground truth 41.8%).
- **Disagreements.** Pairing runs as above, the modes disagreed on 8 of 110 cells: 2 that `per_criterion` marked MET and `per_item` UNMET, 5 the reverse, and 1 that `per_criterion` marked `CANNOT_ASSESS`. The ground truth sided with `per_criterion` on 3 and with `per_item` on 5.
- **Agreement.** Cross-mode agreement, 90.9% to 94.5% (κ 0.80 to 0.88), is below both within-mode pairs, 96.4% each (κ 0.92), by 1 to 3 cells of 55.
- **Accuracy.** Criterion accuracy was 0.882 against 0.853. The item-level difference is +2.7 points, 95% interval [−2.7, +8.2]. Averaged over each mode's two runs, item scores correlate at r = 0.971.
- **Per criterion.** The modes' MET rates and accuracies were identical on `Britain` and `Errors`. Both under-call `Structure` (9.1% MET for `per_criterion` and 31.8% for `per_item`, against 63.6% in the ground truth), and the 5 disagreements toward MET all fell on it, with the ground truth on `per_item`'s side each time.
- **Cost and time.** The single call was cheaper and faster here too: 3.9x fewer prompt tokens, 1.4x lower recorded cost (2.9x at list price), and 1.2x faster (5.4 s against 6.5 s).

Eleven essays, though, can reveal only a large difference.

#### Reading the results

One call per item was cheaper and faster on both datasets. On RiceChem, the same model also graded measurably stricter in one call: a lower MET rate in both runs, disagreements lopsided toward UNMET, and cross-mode agreement below run-to-run agreement. Its accuracy there was 1.3 points higher, within noise on 100 answers, kappa was level, and score RMSE was higher in every run (0.199 and 0.196 against 0.168 and 0.155). On the 11-essay set the modes' overall results overlapped, with `Structure` the one criterion where they clearly parted. The results do not say which mode is right in general. They say that grading in one call changed how this model graded the RiceChem answers.

The limits of this comparison:

- One model, `openai/gpt-6-luna`, at `reasoning_effort="low"`, and one provider, OpenAI; Anthropic, Gemini and Groq models were not tested.
- Two runs per mode. RiceChem is a seeded random sample of 100 of Q1's 327 answers, and the essay set has only 11 items (55 cells, 4 with `CANNOT_ASSESS` ground truth).
- Both rubrics are binary, with no multi-choice criteria, no few-shot examples, no guidelines and no cascade.
- Recorded costs include OpenAI's prompt-caching discount, and the cached shares depend on what ran before and how long the provider kept it.
- Wall-clock times are for runs made one after another, with `max_parallel_requests=16` and the default item concurrency; they depend on rate limits and provider load.

### Step 5: Decide

`per_item` is a good trade when:

- **Your own Step 3 shows no shift you care about**: cross-mode agreement close to within-mode agreement, a MET rate that matches `per_criterion`'s, and accuracy and kappa within noise.
- **Cost or throughput is the constraint.** Before prompt caching, the savings grow with the number of criteria and the length of the submissions. A cache narrows that gap by discounting the repeated prompt, and fewer requests still help most under a rate limit.
- **A larger failure unit is acceptable**, with `max_tokens` and `timeout` sized for the whole rubric.

What to watch:

- **The MET-rate shift.** Compare each mode's MET rate with the ground truth's, overall and per criterion. A shift in the same direction in both runs, with the disagreements lopsided, is systematic; on RiceChem it was about 6 points.
- **Criteria the judge already under-calls.** Two of RiceChem's three accuracy losses were on criteria `per_criterion` already marked MET well below the labels' rate; the third, `higher_core_charge`, it called at about the right rate. Per-criterion differences on 100 answers were not tested for significance, so treat this as where to look first, not as a rule.
- **Interdependent criteria.** On RiceChem, the full and partial explanation criteria, exclusive in the labels but not in the rubric, were not graded as exclusive in either mode. If your rubric has criteria meant to exclude or imply each other, check their joint pattern against your labels in both modes. Stating the relation in the requirements or the rubric's [guidelines](../api/core-grading.md#rubric-guidelines) may help; this comparison did not test that.
- **Scores as well as verdicts.** If you use item scores, a shift in verdicts moves every score: on RiceChem, score RMSE rose from 0.161 to 0.197 while accuracy moved 1.3 points.
- **Panels as a hedge.** A panel of `per_item` judges from different model families (`judges=[...]` with `llm_calls="per_item"`) can dilute one model's shift if the models do not all shift the same way, and each judge still sends an item only once. This comparison did not measure a panel.
- **Every change of rubric, data or model.** Whether a shift appears is not a property of the model alone: with the same model, the 11-essay set's MET rate rose slightly (from 31.8% to 34.5%) while RiceChem's fell about 6 points, and this comparison cannot say which difference between the datasets matters. Re-run Step 3 when you change the rubric, the model or its thinking level.
- **A new model's configuration.** Before a paid run on a model LiteLLM may not know yet, grade one item and check that no criterion failed, or run with `fail_fast=True`, as the LiteLLM warning in Step 4 shows.

## Key Takeaways

- **One call per item is a cost tool.** On the RiceChem sample it cost 1.8x less as recorded (3.8x at list price) and ran 2.0x faster than one call per criterion.
- **It can change how the judge grades.** On RiceChem the same model graded stricter in one call: 47% of cells MET against 53%, and 160 of 231 disagreements toward UNMET. On the 11-essay set the modes' overall accuracy could not be told apart.
- **Validate with replicates.** Grade the same labelled sample twice with each mode; a mode effect must exceed the disagreement between two runs of the same mode.
- **Look past accuracy.** A 1.3-point accuracy difference within noise came with a MET rate about 6 points lower in both runs, lower MET recall, and a higher score RMSE in every run.
- **Check interdependent criteria yourself.** In this comparison, seeing every criterion at once did not make the judge treat full and partial credit as exclusive.
- **Size `max_tokens` and `timeout` for the whole rubric**, since a failed call fails every criterion of the item.

## Going Further

- [LLM Judges: Grading a whole rubric in one call](llm-judges.md#grading-a-whole-rubric-in-one-call): the call, its cost model, failure scoping, few-shot examples and thinking traces
- [Judge Validation](judge-validation.md): metrics against human labels and bootstrap confidence intervals
- [Cost Optimization](cost-optimization.md): response caching, prompt caching and model choice
- [Cheap First-Pass Grading with Jev and an LLM Fallback](cascade-grading.md): `per_item` escalation judges in a cascade
- [Extended Thinking](extended-thinking.md): one reasoning trace per call under `llm_calls="per_item"`
- [API Reference: Graders](../api/graders.md)

---

## Appendix: Complete Code

The validation of Step 3 as one script:

```python
"""Validate llm_calls="per_item" against per-criterion grading on labelled data."""

import asyncio
import itertools

from sklearn.metrics import cohen_kappa_score

from autorubric import CriterionVerdict, LLMConfig, RubricDataset, compute_metrics, evaluate
from autorubric.graders import CriterionGrader

MODES, REPLICATES = ("per_criterion", "per_item"), (1, 2)
MET, UNMET = CriterionVerdict.MET, CriterionVerdict.UNMET


def verdicts(result):
    """Final verdict of each judged binary (item, criterion) cell; failed judgments left out."""
    return {
        (r.item_idx, c): cr.final_verdict
        for r in result.item_results
        if r.error is None
        for c, cr in enumerate(r.report.report)
        if not cr.is_error and cr.final_verdict is not None
    }


def agreement(a, b):
    """Share of the cells both runs judged on which they agree, and Cohen's kappa."""
    va, vb = verdicts(a), verdicts(b)
    cells = sorted(va.keys() & vb.keys())
    x, y = [va[k].value for k in cells], [vb[k].value for k in cells]
    return sum(p == q for p, q in zip(x, y)) / len(cells), cohen_kappa_score(x, y)


def fmt(x, spec=".3f"):
    return "n/a" if x is None else format(x, spec)


async def main():
    dataset = RubricDataset.from_file("labelled_answers.json")  # items with ground truth
    judge = LLMConfig(model="openai/gpt-5.6-luna", max_parallel_requests=16)

    # Two runs of each mode, first runs first; each call is a new experiment.
    runs = {}
    for rep in REPLICATES:
        for mode in MODES:
            grader = CriterionGrader(judge_model_config=judge, llm_calls=mode, seed=42)
            runs[mode, rep] = await evaluate(dataset, grader, show_progress=False)

    # Each run against the labels.
    truth = [v for item in dataset for v in item.ground_truth]
    print(f"ground truth: MET rate {truth.count(MET) / len(truth):.1%}")
    for (mode, rep), result in runs.items():
        metrics = compute_metrics(result, dataset)
        judged = list(verdicts(result).values())
        print(
            f"{mode}#{rep}: accuracy {fmt(metrics.criterion_accuracy)}, "
            f"kappa {fmt(metrics.mean_kappa)}, MET rate {judged.count(MET) / len(judged):.1%}, "
            f"cost ${fmt(result.total_completion_cost, '.4f')}"
        )

    # Run-to-run baseline (within a mode) against the mode effect (across modes).
    for (m1, r1), (m2, r2) in itertools.combinations(runs, 2):
        kind = "within" if m1 == m2 else "across"
        share, kappa = agreement(runs[m1, r1], runs[m2, r2])
        print(f"{kind} modes, {m1}#{r1} vs {m2}#{r2}: {share:.1%} agree, kappa {kappa:.2f}")

    # Which way the cross-mode disagreements go.
    for rep in REPLICATES:
        a, b = verdicts(runs["per_criterion", rep]), verdicts(runs["per_item", rep])
        cells = a.keys() & b.keys()
        stricter = sum(a[k] == MET and b[k] == UNMET for k in cells)
        laxer = sum(a[k] == UNMET and b[k] == MET for k in cells)
        print(f"run {rep}: per_item UNMET where per_criterion MET: {stricter}; the reverse: {laxer}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Reproducing the live comparison

[`examples/single_call_grading_comparison.py`](https://github.com/delip/autorubric/blob/main/examples/single_call_grading_comparison.py) reruns Step 4: both datasets, both modes, two runs each in Step 4's order, with the configuration shown there. For each dataset it prints every run's accuracy, kappa, MET count (MET cells out of judged cells), tokens, cost and wall-clock time, the agreement between every pair of runs, the item-level accuracy difference with its bootstrap interval, and the direction of the cross-mode disagreements. The model, the reasoning level, the number of runs, the sample, the seeds and the concurrency cap are constants at the top of the script (`MODEL`, `REASONING`, `REPLICATES`, `RICECHEM_SAMPLE`, `SAMPLE_SEED`, `GRADER_SEED`, `MAX_PARALLEL_REQUESTS`). Run it from a checkout of the repository, since `examples/` is not part of the installed package:

```bash
export OPENAI_API_KEY=your_key_here
uv run python examples/single_call_grading_comparison.py
```

!!! warning "The script makes paid calls"
    Every run sends real, billed requests to the model's provider. With the constants as shipped, the script makes the same 1,932 calls as Step 4 (two runs of each mode: 55 or 11 calls per essay run, 800 or 100 per RiceChem run), which cost about $0.23 in our run with OpenAI's caching discount, or about $0.63 at list price. The judge is sampled, so your numbers will differ from Step 4's.

    Each run is an `EvalRunner` experiment under `experiments/single_call_grading/`, in a directory named after the model, the reasoning level and the grader seed, and each run's name includes its dataset's name, which carries the RiceChem sample's size and seed. An interrupted run resumes without paying again for the items it graded, and re-running the script reads finished runs back instead of grading them again (their wall-clock time then shows as `n/a`); changing one of those constants starts new runs, which are paid for. A run with a failed item stops the script, since a resumed run never retries its failed items: fix the cause, delete the directory the script names, and run it again.
