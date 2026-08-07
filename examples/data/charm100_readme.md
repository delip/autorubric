# Appendix: CHARM-100 — Chatbot Assessment with Mixed Rubric Metrics

## A.1 Dataset Overview

Several chatbot evaluation benchmarks exist, but they uniformly adopt a single scale type across all criteria. MT Bench (Zheng et al., 2023) and Vicuna Bench (Chiang et al., 2023) use 1-5 Likert scales; Arena-Hard Auto (Li et al., 2024) uses pairwise ordinal judgments; WildBench (Lin et al., 2024) uses binary checklists aggregated into a composite score; and HelpSteer2 (Wang et al., 2024) and LLM-Rubric (Hashemi et al., 2024) use multi-dimensional Likert ratings. None of these combine ordinal, nominal, and binary criteria within a single rubric while providing per-sample criterion-level reference labels for end-to-end framework testing.

The CHARM-100 dataset (**Ch**atbot **A**ssessment with Mixed **R**ubric **M**etrics, 100 samples) was created to fill this gap and to exercise the capabilities of rubric evaluation frameworks such as `autorubric` that support heterogeneous criterion types. It contains 100 annotated single-turn chatbot conversations, each with operational reference labels across six evaluation criteria spanning ordinal, nominal, and binary measurement scales. All conversations were synthetically authored in English and serialized as JSON conforming to the `RubricDataset` schema.

| Property | Value                                        |
| -------- | -------------------------------------------- |
| Name     | `charm-100`                                  |
| Size     | 100 annotated samples                        |
| Format   | JSON (`RubricDataset` schema)                |
| Language | English                                      |
| Source   | Synthetically authored chatbot conversations |

## A.2 Annotation Schema

The dataset uses a hybrid rubric combining ordinal, nominal, and binary criteria. Each sample receives exactly six operational reference labels, one per criterion. Real-world evaluation rubrics rarely consist of a single criterion type, so a benchmark restricted to one measurement scale would give an incomplete picture of judge capabilities.

### A.2.1 Criteria summary

| #   | Criterion          | Type    | Weight | Scale               | Options                                                                                                   |
| --- | ------------------ | ------- | ------ | ------------------- | --------------------------------------------------------------------------------------------------------- |
| 1   | `satisfaction`     | Ordinal | 10.0   | 4-point             | Very dissatisfied (0.0), Somewhat dissatisfied (0.33), Somewhat satisfied (0.67), Very satisfied (1.0)    |
| 2   | `helpfulness`      | Ordinal | 8.0    | 4-point             | Not helpful at all (0.0), Slightly helpful (0.33), Moderately helpful (0.67), Very helpful (1.0)          |
| 3   | `naturalness`      | Ordinal | 5.0    | 4-point             | Robotic/unnatural (0.0), Somewhat mechanical (0.33), Mostly natural (0.67), Very natural/human-like (1.0) |
| 4   | `response_length`  | Nominal | 4.0    | 3-class             | Too brief (0.0), Too verbose (0.0), Just right (1.0)                                                      |
| 5   | `factual_accuracy` | Binary  | 10.0   | MET/UNMET           | Binary verdict on factual correctness                                                                     |
| 6   | `specificity`      | Ordinal | 6.0    | 5-point (incl. N/A) | Very vague (0.0), Somewhat vague (0.33), Moderately specific (0.67), Very specific (1.0), N/A (excluded)  |

The two highest-weighted criteria, satisfaction (10.0) and factual accuracy (10.0), pair a subjective holistic impression with an objective verifiable property. Helpfulness (8.0) and specificity (6.0) sit in the middle; naturalness (5.0) and response length (4.0) carry the lowest weights. The final score is therefore most sensitive to satisfaction and factual accuracy, while still penalizing deficiencies in tone and length.

### A.2.2 Scoring semantics

The four ordinal criteria (`satisfaction`, `helpfulness`, `naturalness`, `specificity`) are scored using the `value` field (0.0-1.0) associated with each option. Because these criteria have a natural ordering, inter-rater agreement is measured with weighted (quadratic) kappa, which penalizes large disagreements more heavily than adjacent ones.

The nominal criterion `response_length` has three options, but two of them (`Too brief` and `Too verbose`) both receive value 0.0. Only `Just right` receives 1.0. Because the two failure modes have no ordering relative to each other, agreement is measured with unweighted Cohen's kappa.

The binary criterion `factual_accuracy` uses a MET/UNMET verdict, scored as 1.0 and 0.0 respectively. Despite being the simplest scale in the rubric, it requires the judge to verify factual content rather than rely on surface-level heuristics.

The `specificity` criterion includes an N/A option (flagged with `"na": true`) for cases where the question does not call for concrete recommendations, such as purely definitional questions. Samples labeled N/A are excluded from the specificity score denominator. The rubric does not specify a custom CANNOT_ASSESS strategy, so the default `SKIP` strategy applies (the denominator is adjusted to exclude unassessable criteria).

The final score is a weighted sum of per-criterion scores, normalized by total positive weight. All weights in this rubric are positive.

## A.3 Data collection methodology

### A.3.1 Construction process

All 100 samples were synthetically authored to cover a controlled distribution of quality levels, topic domains, and response failure modes. No samples were collected from deployed chatbot systems, and no crowdsourcing was used. Synthetic construction provides precise control over the joint distribution of quality labels across criteria, which would be prohibitively difficult to achieve by collecting real chatbot conversations.

### A.3.2 Design principles

The dataset was constructed according to four principles.

The item descriptions span the full quality spectrum. For audit sampling, a deterministic classifier maps description prefixes and failure keywords into five quality strata plus an Edge Cases stratum; these post hoc tags are not explicit fields in the released JSON.

Cross-criteria conflicts were deliberately introduced. A response might be factually wrong but naturally written, or technically correct but robotic. These conflicts are designed to expose reliance on a single "overall quality" heuristic projected across all criteria.

Non-trivial responses were required even for descriptions classified into the lowest-quality strata. The "poor" samples contain substantive text with identifiable flaws, not empty outputs or obviously broken formatting. The benchmark therefore tests identification of subtle problems in realistic-looking text.

Diversity was maximized across multiple dimensions: 100 unique system prompts, more than 35 topic domains, and more than 25 distinct assistant personas. This reduces the risk that a judge's accuracy is inflated by topic-specific heuristics or memorized patterns.

## A.4 Topic and domain coverage

For descriptive coverage, each sample was assigned post hoc to one mutually exclusive primary-topic cluster using the main information need in the user message. The assignment covers all 100 items and is not intended to estimate deployment traffic.

| Cluster                                        | Domains                                                                                                            | Count |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----- |
| Computing & Technology                         | programming, ML/AI, cybersecurity, digital privacy, software operations, computer support                        | 9     |
| Science & Mathematics                          | mathematics, statistics, physics, chemistry, biology, astronomy, environmental science                           | 14    |
| Health & Wellness                              | medicine, nutrition, mental health, sleep, exercise, rehabilitation, first aid                                    | 12    |
| Home & Practical Skills                        | cooking, baking, home repair, electrical safety, automotive maintenance, gardening                               | 17    |
| Finance, Law & Consumer Affairs                | budgeting, saving and investing, taxation, legal rights, major purchases                                          | 9     |
| Work & Business                                | project management, presentations, interviewing, career development, negotiation, entrepreneurship               | 6     |
| Humanities, Social Sciences & Education        | history, philosophy, literature, economics, geography, art history, psychology, study and research methods       | 12    |
| Arts, Hobbies & Animal Care                    | photography, music and audio production, gaming, crafts, creative writing, pets, aquariums                        | 10    |
| Language, Culture & Social Life                | travel, etiquette, language learning and translation, media literacy, parenting, relationships                   | 11    |

Every sample uses a unique system prompt, preventing a judge from learning system-prompt-specific shortcuts.

## A.5 Description-derived quality strata

The released JSON has no explicit quality-tier field. The companion second-annotation audit uses a fixed, deterministic classifier over each item's description prefix and failure keywords to assign six post hoc strata for sampling. These tags are heuristic metadata, not independently authored annotations.

| Stratum        | Count | Description                                                                                                                          |
| -------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Excellent      | 15    | All or nearly all top marks. Specific, accurate, well-structured, natural.                                                           |
| Good           | 12    | Mostly high ratings with one or two minor shortcomings (e.g., slightly verbose or mildly mechanical).                                |
| Mediocre/Mixed | 27    | Split verdicts across criteria with deliberate conflicts.                                                                            |
| Below Average  | 15    | Generic, mechanical, missing key information, or deflective. Partially addresses the question but falls short on several dimensions. |
| Poor           | 11    | Useless, evasive, broken, or containing dangerous misinformation.                                                                    |
| Edge Cases     | 20    | Borderline or debatable labels designed to challenge automated judges.                                                               |

The classifier yields counts 15/12/27/15/11/20 for Excellent, Good, Mediocre/Mixed, Below Average, Poor, and Edge Cases, respectively.

## A.6 Reference label distribution

Reference labels were distributed broadly across options for each criterion. If a single label dominated (e.g., 95% "Very satisfied"), a judge could achieve high accuracy by always predicting the majority class. The four ordinal criteria were constructed with relatively broad distributions, while response length and factual accuracy have majority-class shares of 66% and 72%, respectively; per-criterion interpretation should therefore consider the displayed class balance rather than accuracy alone.

### A.6.1 Satisfaction (ordinal, 4-point)

| Label                 | Count | Percentage |
| --------------------- | ----- | ---------- |
| Very dissatisfied     | 20    | 20%        |
| Somewhat dissatisfied | 33    | 33%        |
| Somewhat satisfied    | 28    | 28%        |
| Very satisfied        | 19    | 19%        |

The distribution is nearly uniform with a slight skew toward the middle categories (Somewhat dissatisfied and Somewhat satisfied account for 61%). Normalized entropy: 0.98.

### A.6.2 Helpfulness (ordinal, 4-point)

| Label              | Count | Percentage |
| ------------------ | ----- | ---------- |
| Not helpful at all | 19    | 19%        |
| Slightly helpful   | 27    | 27%        |
| Moderately helpful | 34    | 34%        |
| Very helpful       | 20    | 20%        |

Near-uniform distribution (normalized entropy: 0.98) with a mild peak at "Moderately helpful," reflecting the many responses that partially address the user's question.

### A.6.3 Naturalness (ordinal, 4-point)

| Label                   | Count | Percentage |
| ----------------------- | ----- | ---------- |
| Robotic/unnatural       | 9     | 9%         |
| Somewhat mechanical     | 25    | 25%        |
| Mostly natural          | 30    | 30%        |
| Very natural/human-like | 36    | 36%        |

Naturalness is the most skewed ordinal criterion (normalized entropy: 0.93), leaning toward the upper end of the scale. Even synthetically authored responses tend to sound at least somewhat natural, making "Robotic/unnatural" the hardest label to construct convincingly. The 9 samples in that category are deliberately extreme cases of formulaic output.

### A.6.4 Response length (nominal, 3-class)

| Label       | Count | Percentage |
| ----------- | ----- | ---------- |
| Too brief   | 20    | 20%        |
| Too verbose | 14    | 14%        |
| Just right  | 66    | 66%        |

Response length has the lowest entropy (normalized: 0.79), driven by the dominance of "Just right" at 66%. This is intentional: most chatbot responses are roughly appropriate in length, and a more balanced distribution would misrepresent the base rate. The 34 samples with length failures (20 too brief, 14 too verbose) provide enough signal to measure a judge's sensitivity to this criterion.

### A.6.5 Factual accuracy (binary)

| Label | Count | Percentage |
| ----- | ----- | ---------- |
| MET   | 72    | 72%        |
| UNMET | 28    | 28%        |

The 72/28 split reflects a realistic base rate. Normalized entropy is 0.86, which is high for a binary variable. The 28 UNMET samples span a range of factual failure modes (see Section A.7.1).

### A.6.6 Specificity (ordinal, 5-point including N/A)

| Label               | Count | Percentage |
| ------------------- | ----- | ---------- |
| Very vague          | 13    | 13%        |
| Somewhat vague      | 27    | 27%        |
| Moderately specific | 26    | 26%        |
| Very specific       | 25    | 25%        |
| N/A                 | 9     | 9%         |

Specificity has the highest raw entropy (2.21 / 2.32, normalized: 0.95) because it has the most options. The four substantive labels are distributed fairly evenly. The 9 N/A samples represent questions that do not call for concrete recommendations (e.g., definitional or philosophical questions) and are excluded from scoring.

## A.7 Response anti-pattern taxonomy

The dataset includes a structured taxonomy of response failure modes distributed across the description-derived strata in Section A.5. These anti-patterns provide coverage of the failures that automated judges must detect, and they enable fine-grained error analysis when a judge underperforms on a particular criterion.

### A.7.1 Factual failures

The 28 UNMET samples for factual accuracy exhibit six failure patterns. Outright errors involve confidently stating incorrect facts (e.g., wrong chemical formulas, misattributed artworks). Subtle hallucinations are mostly correct responses with one non-obvious factual error requiring domain knowledge to catch. Outdated information covers facts that were previously correct but are no longer accurate.

Three patterns test the boundary between truth and deception. Confident confabulation involves inventing plausible-sounding but fictitious details: nonexistent laws, fabricated studies, wrong historical dates. Partially correct with critical error describes sound reasoning that depends on one wrong premise. Misleading framing covers cases where every individual statement is technically true but the statements are arranged to support a false conclusion.

### A.7.2 Helpfulness failures

Helpfulness failures test whether a judge can distinguish between responses that look informative and those that actually address the user's need. Evasive/deflective responses ask for information the user already provided. Tangential responses answer a related but different question. Over-hedged responses pile on qualifiers until no actionable advice remains. Circular responses restate the question as the answer. Information dumps provide encyclopedic text on the general topic but fail to address the specific question.

### A.7.3 Naturalness failures

Naturalness failures cover ways a response can feel artificial despite correct content. Overly formal/stilted responses read like textbook excerpts inappropriate for conversational settings. Excessive disclaimers front-load paragraphs of caveats before any substance. Bullet-only walls present information as raw outlines without connective prose. Patronizing tone manifests as condescending language. Robotic enumeration presents numbered lists without context or transitions.

### A.7.4 Length failures

Length failures fall into three subcategories. Too-brief responses provide one-liners or fragments for questions requiring detailed answers. Too-verbose responses suffer from bloated repetition, unnecessary preamble, or kitchen-sink coverage. Preamble-heavy responses contain useful content buried under paragraphs of unnecessary context-setting. This third category tests whether a judge evaluates length based on total word count or on the ratio of useful content to filler.

### A.7.5 Compound anti-patterns

The most diagnostically valuable samples combine failures across criteria. A response that is factually wrong but naturally written tests whether the judge can separate content from presentation. A correct but robotic response tests the reverse. Other combinations: helpful but excessively verbose; natural but vague; condescending but technically specific; empathetic but without actionable advice. These compound patterns force the judge to evaluate each criterion independently rather than relying on a single quality signal.

## A.8 Edge cases and challenging samples

Between 10 and 20 samples are deliberately ambiguous or borderline. These are expected to produce lower inter-rater agreement and serve as stress tests for automated judges.

Criteria conflicts pit quality dimensions against each other: factually wrong but well-written, or correct but robotic. These test whether a judge maintains independent assessments or collapses into a single "overall quality" judgment.

Factual borderlines probe the gray area between correct and incorrect: mostly correct with one subtle error, reasonable rounding versus wrong numbers, previously-correct-but-now-outdated information.

Length borderlines test context-dependent judgment. Three sentences may be appropriate for a simple question but insufficient for a complex one. These samples test whether the judge adjusts length expectations to question complexity.

N/A ambiguity targets the specificity criterion. Some questions are clearly definitional (N/A is appropriate), others clearly call for actionable advice (N/A would be wrong), and a subset falls between these extremes.

Context mismatches present correct advice for the wrong context (wrong jurisdiction, wrong climate, wrong cultural setting). Polite refusals are appropriate safety redirects that nonetheless fail to answer the question. Satisfaction-helpfulness divergence separates the subjective reading experience from objective utility: some responses are correct but unsatisfying, others entertaining but unhelpful. Oversimplification versus accessibility tests the boundary at which simplification becomes misleading.

These samples are intended to expose the discrimination limits of LLM-as-a-judge systems. A judge that achieves high accuracy on unambiguous samples but fails on edge cases is likely relying on shallow heuristics.

## A.9 Sample format

Each item in the `items` array has three fields:

```json
{
  "submission": "<JSON-encoded string of messages array>",
  "description": "<brief description of quality characteristics>",
  "ground_truth": [
    "<satisfaction label>",
    "<helpfulness label>",
    "<naturalness label>",
    "<response_length label>",
    "<factual_accuracy verdict: MET or UNMET>",
    "<specificity label>"
  ]
}
```

The `submission` field contains a JSON-encoded string representing an array of chat messages. All conversations are single-turn (one system message, one user message, one assistant response):

```json
[
  {"role": "system", "content": "You are a helpful programming assistant."},
  {"role": "user", "content": "How do I handle exceptions in Python?"},
  {"role": "assistant", "content": "Great question! Exception handling in Python uses try/except blocks..."}
]
```

The `description` field provides a human-readable summary of the sample's quality characteristics (e.g., "Factually accurate and specific but overly formal tone"). It is not used in scoring but aids manual inspection. The `ground_truth` array contains exactly six labels in the fixed order of the criteria table in Section A.2.1. Label strings must match the rubric's option labels exactly.

## A.10 Summary statistics

| Statistic                                      | Value                                                   |
| ---------------------------------------------- | ------------------------------------------------------- |
| Total samples                                  | 100                                                     |
| Unique topic domains                           | 35+                                                     |
| Unique system prompts                          | 100                                                     |
| User message length (median)                   | 19 words                                                |
| Assistant response length (min / median / max) | 9 / 180 / 497 words                                     |
| Assistant response length (mean / stdev)       | 167 / 98 words                                          |
| Criteria per sample                            | 6                                                       |
| Ordinal criteria                               | 4 (satisfaction, helpfulness, naturalness, specificity) |
| Nominal criteria                               | 1 (response_length)                                     |
| Binary criteria                                | 1 (factual_accuracy)                                    |
| Samples with N/A labels                        | 9 (specificity only)                                    |
| Mean normalized entropy (across criteria)      | 0.92                                                    |

User messages have a median length of 19 words. Assistant responses range from 9 to 497 words (mean 167, stdev 98). The high coefficient of variation reflects the intentional diversity of response lengths across description-derived strata.

Mean normalized entropy is 0.92 across all six criteria. The four ordinal criteria are comparatively balanced (majority-class baselines 27--36%), whereas response length and factual accuracy are more skewed (66% and 72%); the benchmark therefore spans both balanced and imbalanced label distributions.

## A.11 Intended use

This dataset benchmarks LLM-as-a-judge systems on their ability to reproduce the dataset's operational reference labels across multiple criteria simultaneously. Because it requires the judge to handle ordinal, nominal, and binary criteria in a single pass, it provides a more varied measurement task than benchmarks restricted to a single scale type.

The dataset also supports per-criterion discrimination analysis. Factual accuracy and naturalness, for instance, test very different capabilities (fact verification versus stylistic judgment), and a judge may perform well on one while struggling with the other. The hybrid rubric enables criterion-appropriate agreement metrics: weighted (quadratic) kappa for ordinal criteria, unweighted Cohen's kappa for nominal and binary criteria, adjacent accuracy, Spearman and Kendall rank correlations, and per-option precision/recall/F1.

The edge cases and compound anti-patterns (Sections A.7.5 and A.8) test sensitivity to criteria conflicts, subtle factual errors, and context-dependent quality assessments. The dataset also serves as a reference implementation for the `autorubric` library's multi-choice criterion format.

### A.11.1 Usage example

```python
from autorubric.dataset import RubricDataset

dataset = RubricDataset.from_file(
    "examples/data/charm100.json"
)
print(f"Loaded {len(dataset.items)} items")
```

## A.12 Limitations

All conversations are synthetic: authored by a language model rather than collected from a production chatbot system. The distribution of response patterns, error types, and conversational styles may therefore not fully represent real-world deployments. Synthetic authoring provides precise control over the joint distribution of quality labels but reduces ecological validity.

Reference labels were assigned by the authoring LLM, so they are operational annotations rather than independently adjudicated truth. The initial dataset release included no agreement measurement. A subsequent seed-42, quality-tier-stratified second annotation pass covered 50 items; its recorded subset, label key, guidelines, and summary are preserved in the [companion paper repository](https://github.com/delip/autorubric-paper/tree/45581b07f0b45bcdfe31b5dbfbf2671c1694db58/data). Agreement with the reference labels ranged from Cohen's kappa 0.506 to 0.870 by criterion. The descriptive macro-average (0.687) mixes quadratic-weighted and unweighted coefficients and should not be treated as a common-scale reliability statistic or as full-dataset human--human agreement. On edge cases, disagreements may represent legitimate differences of interpretation.

The dataset is English-only. All prompts, queries, and responses are in English, and the evaluation criteria assume English-language conventions for naturalness, formality, and specificity. Results may not generalize to multilingual settings.

All conversations are single-turn (one user message, one assistant response). Multi-turn dialogue quality dimensions (coherence across turns, context tracking, topic management, conversational repair) are not captured.

The rubric is fixed at six criteria with predetermined scales and weights. Alternative rubric designs (different granularity, different criteria such as creativity or safety, domain-specific criteria) are not represented. The benchmark tests a judge's ability to apply this particular rubric, not to adapt to arbitrary rubrics.

The dataset is a static snapshot. It does not capture how chatbot quality evolves over time, across model versions, or in response to changing user expectations.

## Evaluation results

The following results were obtained by running `autorubric` on CHARM-100 with Gemini 3 Flash Preview (`gemini/gemini-3-flash-preview`) as the judge on all 100 samples. These results illustrate how the hybrid rubric design exposes criterion-type-specific strengths and weaknesses that a single-scale benchmark would obscure.

### Aggregate metrics

| Metric                        | Value                | 95% Bootstrap CI |
| ----------------------------- | -------------------- | ---------------- |
| Accuracy (binary criterion)   | 87.0%                | [80.0%, 93.0%]   |
| Mean kappa (all criteria)     | 0.623                | [0.451, 0.794]   |
| Spearman correlation (scores) | 0.810                | --               |
| Kendall correlation (scores)  | 0.663                | --               |
| RMSE (scores)                 | 0.246                | [0.212, 0.274]   |
| Mean bias                     | +0.170               | --               |

The model has a positive mean bias of +0.17, rating responses higher than the reference labels on average. No inferential test is reported for this statistic. The pattern manifests differently across criterion types, as the per-criterion breakdown reveals.

### Per-criterion breakdown

| Criterion        | Type    | Exact Acc. | Adj. Acc. | Kappa | Spearman | RMSE  |
| ---------------- | ------- | ---------- | --------- | ----- | -------- | ----- |
| factual_accuracy | Binary  | 87.0%      | --        | 0.642 | --       | --    |
| naturalness      | Ordinal | 58.0%      | 93.0%     | 0.719 | 0.743    | 0.265 |
| satisfaction     | Ordinal | 42.0%      | 85.0%     | 0.648 | 0.786    | 0.339 |
| specificity      | Ordinal | 39.5%      | 86.4%     | 0.549 | 0.698    | 0.356 |
| helpfulness      | Ordinal | 38.0%      | 85.0%     | 0.625 | 0.747    | 0.345 |
| response_length  | Nominal | 81.0%      | --        | 0.552 | --       | --    |

The binary criterion (factual accuracy) achieves 87% accuracy with F1 = 0.92 and high recall (0.97), indicating that the model rarely fails to flag correct responses but occasionally marks incorrect responses as correct (precision 0.86). The ordinal criteria tell a different story: exact accuracy ranges from 38% to 58%, though adjacent accuracy (within one step on the scale) is consistently high (85-93%). Naturalness is the easiest ordinal criterion for the model (weighted kappa 0.719); specificity is the hardest (0.549). The nominal criterion, response length, achieves 81% accuracy but only moderate kappa (0.552).

**Factual accuracy confusion matrix** (rows = ground truth, columns = predicted):

|           |  MET | UNMET |
| --------- | ---: | ----: |
| **MET**   |   70 |     2 |
| **UNMET** |   11 |    17 |

### Middle-category collapse

The confusion matrices for ordinal criteria reveal a consistent pattern: the model struggles to use intermediate categories, instead pulling predictions toward the extremes of the scale (particularly the positive end). For satisfaction, the model never predicts "Somewhat satisfied" (0/100 predictions) and predicts "Very satisfied" for 61 samples when only 19 are labeled as such in the ground truth. All 28 "Somewhat satisfied" samples and 15 of the 33 "Somewhat dissatisfied" samples are misclassified as "Very satisfied." Helpfulness shows the same pattern: "Moderately helpful" receives only 10 predictions versus 34 in the ground truth, while "Very helpful" receives 64 predictions versus 20.

**Satisfaction confusion matrix** (rows = ground truth, columns = predicted):

|                           | Very dissatisfied | Somewhat dissatisfied | Somewhat satisfied | Very satisfied |
| ------------------------- | ----------------: | --------------------: | -----------------: | -------------: |
| **Very dissatisfied**     |                16 |                     4 |                  0 |              0 |
| **Somewhat dissatisfied** |                 3 |                     7 |                  8 |             15 |
| **Somewhat satisfied**    |                 0 |                     1 |                  0 |             27 |
| **Very satisfied**        |                 0 |                     0 |                  0 |             19 |

**Helpfulness confusion matrix** (rows = ground truth, columns = predicted):

|                        | Not helpful at all | Slightly helpful | Moderately helpful | Very helpful |
| ---------------------- | -----------------: | ---------------: | -----------------: | -----------: |
| **Not helpful at all** |                 13 |                5 |                  1 |            0 |
| **Slightly helpful**   |                  4 |                3 |                  7 |           13 |
| **Moderately helpful** |                  1 |                0 |                  2 |           31 |
| **Very helpful**       |                  0 |                0 |                  0 |           20 |

This middle-category collapse produces a characteristic signature in the metrics: high adjacent accuracy (the model is usually within one step) and strong rank correlations (the ordering is largely preserved), but low exact accuracy (the model cannot reliably distinguish adjacent categories). The weighted kappa values remain in the substantial range (0.6-0.7) because this metric downweights adjacent disagreements, but per-option F1 scores for middle categories are near zero.

### Ordinal confusion matrices: naturalness and specificity

**Naturalness confusion matrix** (rows = ground truth, columns = predicted):

|                             | Robotic/unnatural | Somewhat mechanical | Mostly natural | Very natural/human-like |
| --------------------------- | ----------------: | ------------------: | -------------: | ----------------------: |
| **Robotic/unnatural**       |                 7 |                   1 |              1 |                       0 |
| **Somewhat mechanical**     |                 4 |                  11 |              5 |                       5 |
| **Mostly natural**          |                 1 |                   2 |              5 |                      22 |
| **Very natural/human-like** |                 0 |                   0 |              1 |                      35 |

**Specificity confusion matrix** (rows = ground truth, columns = predicted, N/A excluded):

|                         | Very vague | Somewhat vague | Moderately specific | Very specific |
| ----------------------- | ---------: | -------------: | ------------------: | ------------: |
| **Very vague**          |          4 |              6 |                   1 |             2 |
| **Somewhat vague**      |          1 |              4 |                   9 |             8 |
| **Moderately specific** |          0 |              0 |                   0 |            21 |
| **Very specific**       |          0 |              0 |                   1 |            24 |

### Asymmetric sensitivity in nominal criteria

The response_length criterion exposes an asymmetry that would be invisible in an ordinal or binary scale. The model detects brevity with reasonable recall (0.70 for "Too brief") but almost completely fails to detect verbosity (0.14 recall for "Too verbose"). Of 14 truly verbose responses, 11 are misclassified as "Just right." In contrast, the model never labels a truly appropriate-length response as verbose (precision 1.00 for "Too verbose"). This pattern suggests the model applies a length threshold that is too permissive on the long side.

**Response length confusion matrix** (rows = ground truth, columns = predicted):

|                 | Too brief | Too verbose | Just right |
| --------------- | --------: | ----------: | ---------: |
| **Too brief**   |        14 |           0 |          6 |
| **Too verbose** |         1 |           2 |         11 |
| **Just right**  |         1 |           0 |         65 |

### N/A handling

The model predicts N/A for the specificity criterion 16 times versus 9 reference N/A labels. The N/A-status confusion counts are TP=6, FP=10, FN=3, and TN=81: raw status agreement is 87.0% and Cohen's kappa is 0.412. Treating N/A as positive, precision is 37.5%, recall is 66.7%, and Jaccard similarity is 31.6%. Specificity-label metrics use the 81 cases in which neither side is N/A.

### Connection to dataset design

These results validate the motivation for a mixed-criterion benchmark described in Section A.1. A purely Likert-scale benchmark would not reveal the middle-category collapse pattern, since it requires per-option analysis within ordinal scales. A purely binary benchmark would not expose the asymmetric sensitivity to different failure modes within the nominal response_length criterion. The divergence across criterion types (kappa ranging from 0.549 to 0.719 across ordinal criteria, with qualitatively different error patterns for nominal and binary criteria) confirms that evaluating a judge on a single scale type gives an incomplete picture of its capabilities.
