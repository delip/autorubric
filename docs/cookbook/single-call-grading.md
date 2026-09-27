# Grading a Whole Rubric in One Call

Let an LLM judge answer every criterion of an item in one call instead of one call per criterion, and check on labelled data that the cheaper calls still grade the way you need.

## The Scenario

You grade a few thousand student answers each term against an eight-criterion rubric with an LLM judge. By default the judge makes one call per criterion, and every call re-sends the grading instructions and the whole answer, so each answer is paid for eight times. `CriterionGrader(llm_calls="per_item")` asks the same judge for all eight verdicts in a single call. It is clearly cheaper; what you need to know is whether it grades as well. Teaching assistants have labelled a few hundred answers criterion by criterion, and you want to check the single call against them before switching.

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
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
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
judge = LLMConfig(model="openai/gpt-4.1-mini", max_parallel_requests=16)
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

- **Cross-mode agreement below both within-mode agreements** means the modes grade differently, beyond run-to-run noise. Agreement within the range of the within-mode pairs means this sample cannot tell them apart.
- **A MET rate that moves the same way in both runs**, with the disagreements lopsided in one direction, says which way the single call leans.
- **Accuracy and kappa against the labels** say whether the shift costs or gains agreement with your graders. Put an interval on the difference before reading much into it: in Step 4, on 100 answers, the 95% interval of the accuracy difference was [−4.1, +0.4] points around −1.8, and its width depends on your sample. `compute_metrics(..., bootstrap=True)` gives each run's confidence intervals, and the companion script in the appendix computes a paired, item-level interval for the difference.
- **`metrics.per_criterion`** shows where the shift lands: it can help some criteria and hurt others.

!!! tip "Grade one item first"
    Before paying for four full runs, grade one item with each mode and check that no criterion failed (`cr.error is None` on every criterion report). A judge that is misconfigured fails every call and, by default, the run still finishes; `evaluate(..., fail_fast=True)` stops it at the first failed item instead. See the LiteLLM warning in Step 4.

### Step 4: What a Live Comparison Found

We ran Step 3 on 2026-09-27 on two labelled datasets that ship in `examples/data/`, with `openai/gpt-6-luna` at a low reasoning effort. The measured configuration, for each `mode`:

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

`temperature` and `max_tokens` were left unset, so no temperature and no output cap were sent. Each mode ran twice, and the eight runs ran one after another: for each dataset, the first run of each mode, then the second. The whole experiment, including a one-item smoke test, cost about $0.31.

| Dataset | Items | Criteria | Ground truth | Calls per run (per_criterion / per_item) |
|---|---|---|---|---:|
| Essay grading (`essay_grading_dataset.json`) | All 11 | 5 binary: `Causes`, `Effects`, `Structure`, `Britain`, `Errors` | 55 cells, 4 of them `CANNOT_ASSESS`; MET rate 41.8% | 55 / 11 |
| RiceChem Q1 (`ricechem/q1.json`) | 100 of 327, sampled with `random.Random(0).sample(range(327), 100)` | 8 binary | Teaching assistants' labels ([Sonkar et al., 2024](https://arxiv.org/abs/2404.14316)), 800 cells; MET rate 56.1% | 800 / 100 |

!!! warning "LiteLLM and `reasoning_effort` on new OpenAI models"
    LiteLLM 1.95.0 recognizes OpenAI reasoning models by name (`gpt-5` and the o-series). For `openai/gpt-6-luna` it therefore refuses `reasoning_effort`, which `thinking="low"` sends, with `UnsupportedParamsError`, raised client-side before any request is made. AutoRubric files that as an `infrastructure` failure, so every criterion abstains; with a single judge, every item ends with no score and the error `Every criterion's judgment failed: infrastructure: ...`. Nothing raises by default: each failed call is only logged as a warning, easy to miss, and the run's `failed_items` counts the failed items once it has finished; `fail_fast=True` stops at the first failed item. In a panel, such a judge contributes nothing without any item failing ([#40](https://github.com/delip/autorubric/issues/40)). `extra_params={"allowed_openai_params": ["reasoning_effort"]}` lets the parameter through.

How to read the tables and figures below:

- Each table gives both runs of each mode. Where the text quotes one number per mode, it is the mean of that mode's two runs, and a ratio is a ratio of those means.
- Accuracy is `criterion_accuracy`, kappa is `mean_kappa` and score RMSE is `score_rmse`, all from `compute_metrics` with its defaults, which leave out cells whose verdict or ground truth is `CANNOT_ASSESS`. MET rates count every cell.
- An accuracy difference with an interval is item-level: each item's accuracy is averaged over its mode's two runs, the items' differences are averaged, and items are resampled 5,000 times for a 95% bootstrap interval. It weights every item equally, so where items have different numbers of labelled cells (the essay set) it differs slightly from the difference of the pooled accuracies.
- Recorded cost is what LiteLLM reported. It is below list price, which we attribute to OpenAI's automatic prompt caching; cached-token counts were not recorded, so that is inferred. List-price cost bills every prompt token at the uncached rate: $0.10 per million prompt tokens and $0.50 per million completion tokens (for this model, cached prompt tokens are billed at $0.01 per million, 90% less).
- Wall-clock time is each run's elapsed time. Values are rounded half up.

**RiceChem Q1, 100 answers × 8 criteria**

| | per_criterion #1 | per_criterion #2 | per_item #1 | per_item #2 |
|---|---:|---:|---:|---:|
| Criterion accuracy | 0.739 | 0.751 | 0.726 | 0.728 |
| Mean kappa | 0.524 | 0.542 | 0.470 | 0.469 |
| Score RMSE | 0.154 | 0.138 | 0.204 | 0.201 |
| Judge MET rate (ground truth 56.1%) | 56.3% | 56.0% | 45.3% | 45.1% |
| Prompt tokens | 1,889,628 | 1,889,628 | 302,941 | 302,941 |
| Completion tokens | 148,550 | 146,838 | 64,879 | 64,780 |
| Recorded cost | $0.1284 | $0.0925 | $0.0438 | $0.0354 |
| List-price cost | $0.2632 | $0.2624 | $0.0627 | $0.0627 |
| Wall-clock | 126.8 s | 123.3 s | 36.6 s | 38.9 s |

**Essay grading, 11 essays × 5 criteria**

| | per_criterion #1 | per_criterion #2 | per_item #1 | per_item #2 |
|---|---:|---:|---:|---:|
| Criterion accuracy | 0.922 | 0.882 | 0.922 | 0.902 |
| Mean kappa | 0.855 | 0.726 | 0.846 | 0.803 |
| Score RMSE | 0.322 | 0.333 | 0.310 | 0.330 |
| Judge MET rate (ground truth 41.8%) | 34.5% | 34.5% | 38.2% | 36.4% |
| Prompt tokens | 118,627 | 118,627 | 30,255 | 30,255 |
| Completion tokens | 4,087 | 4,042 | 3,928 | 3,287 |
| Recorded cost | $0.0042 | $0.0032 | $0.0028 | $0.0019 |
| List-price cost | $0.0139 | $0.0139 | $0.0050 | $0.0047 |
| Wall-clock | 5.9 s | 7.1 s | 4.0 s | 3.6 s |

No run had a failed item or a failed criterion, and no verdict was `CANNOT_ASSESS`.

**Verdict agreement between runs** (cells on which two runs gave the same verdict, and Cohen's kappa between them)

| Runs compared | Kind | Essay (55 cells) | RiceChem (800 cells) |
|---|---|---:|---:|
| per_criterion #1 vs #2 | within a mode | 51 (92.7%), κ 0.84 | 720 (90.0%), κ 0.80 |
| per_item #1 vs #2 | within a mode | 54 (98.2%), κ 0.96 | 739 (92.4%), κ 0.85 |
| per_criterion #1 vs per_item #1 | across modes | 51 (92.7%), κ 0.84 | 666 (83.3%), κ 0.67 |
| per_criterion #1 vs per_item #2 | across modes | 52 (94.5%), κ 0.88 | 667 (83.4%), κ 0.67 |
| per_criterion #2 vs per_item #1 | across modes | 53 (96.4%), κ 0.92 | 682 (85.3%), κ 0.71 |
| per_criterion #2 vs per_item #2 | across modes | 54 (98.2%), κ 0.96 | 683 (85.4%), κ 0.71 |

#### RiceChem: cheaper, faster, and stricter

**Cost and time.** Per run, `per_item` sent 6.2x fewer prompt tokens and 2.3x fewer completion tokens than `per_criterion`. It cost 2.8x less as recorded ($0.0396 against $0.1104; 2.9x between the first runs and 2.6x between the second) and 4.2x less at list price ($0.0627 against $0.2628), and it finished 3.3x sooner (37.7 s against 125.0 s).

Completion tokens fell less than prompt tokens, since every criterion still gets its own explanation. That they fell at all is probably because one reasoning pass now covers the whole rubric instead of one per criterion; reasoning tokens were not recorded separately.

The recorded ratio is smaller than the list-price one, which is consistent with prompt caching: caching discounts only prompt tokens, and at list price those are most of `per_criterion`'s bill, since it repeats the system prompt and the answer in every call, while completion tokens are about half of `per_item`'s. Each mode's second run also cost less than its first, consistent with OpenAI's cache still holding prompts the first run had sent word for word; recorded costs depend on what ran before.

**The single call graded stricter.** `per_item` marked 45% of cells MET, against 56% for `per_criterion` and 56% in the ground truth, and did so in both runs. Comparing the two modes' first runs with each other, and their second runs with each other, they disagreed on 251 of 1,600 cells; on 213 of them `per_criterion` said MET and `per_item` UNMET, and on 38 the reverse. The ground truth sided with `per_criterion` on 140 of the 251 and with `per_item` on 111. Stricter calls traded recall for precision: pooled over both runs, MET precision rose from 0.77 to 0.82 and MET recall fell from 0.77 to 0.66.

**Beyond run-to-run noise.** Two runs of the same mode agreed on 90.0% of cells (κ 0.80) for `per_criterion` and 92.4% (κ 0.85) for `per_item`; runs of different modes agreed on 83.3% to 85.4% (κ 0.67 to 0.71). Every cross-mode pair agreed less than either within-mode pair, so the shift is larger than the disagreement between two runs of the same mode: it comes from the mode, not from the judge's run-to-run randomness. With one pair of runs per mode, that baseline was measured once.

**Accuracy lower, but not significantly; kappa and scores worse in every run.** Criterion accuracy was 0.727 for `per_item` against 0.745 for `per_criterion`, an item-level difference of −1.8 points with a 95% bootstrap interval of [−4.1, +0.4]: not significant on 100 answers. Mean kappa was 0.47 against 0.53, and score RMSE 0.202 against 0.146; each `per_item` run had a lower kappa and a higher RMSE than each `per_criterion` run, but no interval was computed for either difference. Every RiceChem criterion has a positive weight, so stricter verdicts pull item scores down. Averaged over each mode's two runs, the two modes' item scores correlate at r = 0.845.

**Where the shift lands.** The mode columns pool both runs, 200 verdicts per criterion; the ground truth covers the 100 sampled answers.

| Criterion | Ground truth MET | per_criterion MET | per_item MET | per_criterion accuracy | per_item accuracy |
|---|---:|---:|---:|---:|---:|
| `decreased_repulsion` | 82.0% | 77.5% | 80.0% | 0.935 | 0.960 |
| `repulsion_potential_energy` | 57.0% | 35.5% | 24.0% | 0.745 | 0.650 |
| `same_core_charge` | 47.0% | 52.5% | 44.5% | 0.895 | 0.925 |
| `same_shell_radius` | 54.0% | 26.0% | 13.5% | 0.680 | 0.595 |
| `higher_core_charge` | 53.0% | 54.0% | 41.5% | 0.860 | 0.775 |
| `smaller_radius` | 83.0% | 64.0% | 57.0% | 0.780 | 0.720 |
| `full_pe_ie_explanation` | 52.0% | 49.0% | 41.5% | 0.800 | 0.865 |
| `partial_pe_ie_explanation` | 21.0% | 90.5% | 59.5% | 0.265 | 0.325 |

`per_item` lowered the MET rate on seven of the eight criteria. It lost 6.0 to 9.5 points of accuracy on four: three on which `per_criterion` already marked MET well below the ground truth's rate (`repulsion_potential_energy`, `same_shell_radius` and `smaller_radius`), and `higher_core_charge`, which `per_criterion` called at about the right rate. It gained 2.5 to 6.5 points on the other four, including `partial_pe_ie_explanation`, which both modes over-call. With 100 answers per criterion, none of these per-criterion differences was tested for significance.

**The full and partial explanation criteria.** `full_pe_ie_explanation` reads "correctly explains relationship of potential energy to ionization energy" and `partial_pe_ie_explanation` "partially explains relationship between potential energy and ionization energy". The teaching assistants treated them as mutually exclusive: none of the 327 Q1 answers has both MET (175 have only the full one, 56 only the partial one, 96 neither). The rubric does not say so, and neither mode graded them that way:

| (full, partial) | Ground truth (100 answers) | per_criterion (200 verdict pairs) | per_item (200 verdict pairs) |
|---|---:|---:|---:|
| MET, UNMET | 52 | 0 | 1 |
| UNMET, MET | 21 | 83 | 37 |
| UNMET, UNMET | 27 | 19 | 80 |
| MET, MET | 0 | 98 | 82 |

The ground truth's most common pattern, full credit without partial credit, almost never appears in either mode: both judges treat a full explanation as a partial one too. `per_item` marks both MET somewhat less often (82 of 200 against 98), but its better accuracy on `partial_pe_ie_explanation` (0.325 against 0.265) comes mostly from marking both UNMET far more often (80 against 19), part of its general strictness, rather than from treating the pair as exclusive. In this comparison, seeing both criteria in the same call did not make the judge apply a relation the rubric does not state.

#### Essay set: no detectable difference

On the essay set the modes cannot be told apart:

- **MET rate.** `per_item` marked 37.3% of cells MET against 34.5% for `per_criterion` (ground truth 41.8%).
- **Disagreements.** Pairing runs as above, the modes disagreed on 5 of 110 cells: 1 that `per_criterion` marked MET and `per_item` UNMET, 4 the reverse. The ground truth sided with `per_criterion` on 2 and with `per_item` on 3.
- **Agreement.** Cross-mode agreement, 92.7% to 98.2% (κ 0.84 to 0.96), lies within the range of the two within-mode pairs: 92.7% (κ 0.84) for `per_criterion` and 98.2% (κ 0.96) for `per_item`.
- **Accuracy.** Criterion accuracy was 0.912 against 0.902; the item-level difference is +0.9 points, 95% interval [−3.6, +5.5]. Averaged over each mode's two runs, item scores correlate at r = 0.986 between the modes.
- **Per criterion.** The modes' MET rates and accuracies were identical on three of the five criteria, and both under-call `Structure` (36.4% MET against 63.6% in the ground truth).
- **Cost and time.** The single call was cheaper and faster here too: 3.9x fewer prompt tokens, 1.6x lower recorded cost (2.9x at list price), and 1.7x faster (3.8 s against 6.5 s).

Eleven essays, though, can reveal only a large difference.

#### Reading the results

One call per item was cheaper and faster on both datasets. On RiceChem, the same model also graded measurably stricter in one call: a lower MET rate in both runs, disagreements lopsided toward UNMET, and cross-mode agreement below run-to-run agreement. Its accuracy there was 1.8 points lower, within noise on 100 answers, while kappa was lower and score RMSE higher in every run (kappa 0.470 and 0.469 against 0.524 and 0.542; RMSE 0.204 and 0.201 against 0.154 and 0.138). On the 11-essay set the two modes could not be told apart. The results do not say which mode is right in general; they say that grading in one call changed how this model graded the RiceChem answers.

The limits of this comparison:

- One model, `openai/gpt-6-luna`, at `reasoning_effort="low"`, and one provider, OpenAI; Anthropic, Gemini and Groq models were not tested.
- Two runs per mode. RiceChem is a seeded random sample of 100 of Q1's 327 answers, and the essay set has only 11 items (55 cells, 4 with `CANNOT_ASSESS` ground truth).
- Both rubrics are binary, with no multi-choice criteria, no few-shot examples, no guidelines and no cascade.
- Recorded costs include OpenAI's automatic prompt-caching discount. Cached-token counts were not recorded per run, so the discount is inferred from recorded cost against list price.
- Wall-clock times are for runs made one after another, with `max_parallel_requests=16` and the default item concurrency; they depend on rate limits and provider load.

### Step 5: Decide

`per_item` is a good trade when:

- **Your own Step 3 shows no shift you care about**: cross-mode agreement close to within-mode agreement, a MET rate that matches `per_criterion`'s, and accuracy and kappa within noise.
- **Cost or throughput is the constraint.** The savings grow with the number of criteria and the length of the submissions, and fewer requests help most under a rate limit.
- **A larger failure unit is acceptable**, with `max_tokens` and `timeout` sized for the whole rubric.

What to watch:

- **The MET-rate shift.** Compare each mode's MET rate with the ground truth's, overall and per criterion. A shift in the same direction in both runs, with the disagreements lopsided, is systematic; on RiceChem it was about 11 points.
- **Criteria the judge already under-calls.** Three of RiceChem's four accuracy losses were on criteria `per_criterion` already marked MET well below the labels' rate; the fourth, `higher_core_charge`, it called at about the right rate. Per-criterion differences on 100 answers were not tested for significance, so treat this as where to look first, not as a rule.
- **Interdependent criteria.** On RiceChem, the full and partial explanation criteria, exclusive in the labels but not in the rubric, were not graded as exclusive in either mode. If your rubric has criteria meant to exclude or imply each other, check their joint pattern against your labels in both modes. Stating the relation in the requirements or the rubric's [guidelines](../api/core-grading.md#rubric-guidelines) may help; this comparison did not test that.
- **Scores as well as verdicts.** If you use item scores, a shift in verdicts moves every score: on RiceChem, score RMSE rose from 0.146 to 0.202 while accuracy moved 1.8 points.
- **Panels as a hedge.** A panel of `per_item` judges from different model families (`judges=[...]` with `llm_calls="per_item"`) can dilute one model's shift if the models do not all shift the same way, and each judge still sends an item only once. This comparison did not measure a panel.
- **Every change of rubric, data or model.** Whether a shift appears is not a property of the model alone: with the same model, the 11-essay set showed no detectable shift (its MET rate rose slightly, from 34.5% to 37.3%) while RiceChem's fell about 11 points, and this comparison cannot say which difference between the datasets matters. Re-run Step 3 when you change the rubric, the model or its thinking level.
- **A new model's configuration.** Before a paid run on a model LiteLLM may not know yet, grade one item and check that no criterion failed, or run with `fail_fast=True`, as the LiteLLM warning in Step 4 shows.

## Key Takeaways

- **One call per item is a cost tool.** On the RiceChem sample it cost 2.8x less as recorded (4.2x at list price) and ran 3.3x faster than one call per criterion.
- **It can change how the judge grades.** On RiceChem the same model graded stricter in one call: 45% of cells MET against 56%, and 213 of 251 disagreements toward UNMET. On the 11-essay set the modes were indistinguishable.
- **Validate with replicates.** Grade the same labelled sample twice with each mode; a mode effect must exceed the disagreement between two runs of the same mode.
- **Look past accuracy.** A 1.8-point accuracy difference within noise came with a MET rate about 11 points lower in both runs, lower MET recall, and a lower kappa and higher score RMSE in every run.
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
    judge = LLMConfig(model="openai/gpt-4.1-mini", max_parallel_requests=16)

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
    Every run sends real, billed requests to the model's provider. With the constants as shipped, the script makes the same 1,932 calls as Step 4 (two runs of each mode: 55 or 11 calls per essay run, 800 or 100 per RiceChem run), which cost about $0.31 in our run with OpenAI's caching discount, or about $0.69 at list price. The judge is sampled, so your numbers will differ from Step 4's.

    Each run is an `EvalRunner` experiment under `experiments/single_call_grading/`, in a directory named after the model, the reasoning level and the grader seed, and each run's name includes its dataset's name, which carries the RiceChem sample's size and seed. An interrupted run resumes without paying again for the items it graded, and re-running the script reads finished runs back instead of grading them again (their wall-clock time then shows as `n/a`); changing one of those constants starts new runs, which are paid for. A run with a failed item stops the script, since a resumed run never retries its failed items: fix the cause, delete the directory the script names, and run it again.
