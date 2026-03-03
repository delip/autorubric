# ResearcherBench Dataset (AutoRubric Format)

Data from the [ResearcherBench](https://github.com/GAIR-NLP/ResearcherBench) benchmark converted into AutoRubric's `RubricDataset` format.

> Xu, T., Lu, P., Ye, L., Hu, X., & Liu, P. (2025). *ResearcherBench: Evaluating Deep AI Research Systems on the Frontiers of Scientific Inquiry.* arXiv:2507.16280

## Overview

ResearcherBench evaluates Deep AI Research Systems (DARS) on genuinely open-ended, frontier scientific questions — the kind that require synthesis across papers, expert judgment, and nuanced reasoning rather than simple factual recall.

The benchmark contains:
- **65** expert-curated frontier AI research questions across **34** subjects
- **3** question categories: Technical Details, Literature Review, Open Consulting
- Expert-designed per-question rubrics with weighted binary criteria
- Responses from **11** AI systems (7 evaluated in the paper + 4 additional from the repository)

The data was converted to AutoRubric format to enable automated rubric-based evaluation using AutoRubric's grading pipeline. Each question has per-item rubrics with weighted binary criteria — a natural fit for AutoRubric's `Criterion` / `RubricDataset` model.

## Question Categories

| Category          | Count | Description                                                                               |
| ----------------- | ----- | ----------------------------------------------------------------------------------------- |
| Technical Details | 12    | Precise explanations of methodologies, implementations, or theoretical concepts           |
| Literature Review | 20    | Synthesizing findings across multiple papers, comparing methodologies, identifying trends |
| Open Consulting   | 33    | Emerging trends, strategic insights, subjective interpretation and expert judgment        |

## Rubric Structure

Each `DataItem` carries its own per-item rubric (the global rubric is `null`). Per-item rubrics are lists of `Criterion` objects with:

| Field         | Description                                              |
| ------------- | -------------------------------------------------------- |
| `name`        | Auto-generated label (C1, C2, ...)                       |
| `weight`      | Integer 1–3 (1 = nice-to-have, 2 = supporting, 3 = core) |
| `requirement` | What the response must address                           |

**Statistics across the 65 questions:**
- Criteria per question: min 6, max 21, average ~14
- Weight distribution: 35% weight-1, 51% weight-2, 14% weight-3
- All criteria are binary (MET/UNMET)

Rubrics were designed by experienced AI researchers (masters/PhDs) following a 3-step process: insight extraction, human annotation, and quality control.

## Coverage Score

The paper defines a Coverage Score (Eq. 1) that maps directly to AutoRubric's normalized weighted score:

```
Coverage Score = Σ(wᵢ · cᵢ) / Σ(wᵢ)
```

where `wᵢ` is the criterion weight and `cᵢ ∈ {0, 1}` is whether the criterion is met. This is identical to AutoRubric's `raw_score` calculation for binary criteria with positive weights.

## Model Name Mapping

### Deep Research Systems (evaluated in paper)

| Output File                       | Full Model Name                       |
| --------------------------------- | ------------------------------------- |
| `OpenAI_rubric_dataset.json`      | OpenAI Deep Research                  |
| `Google_rubric_dataset.json`      | Gemini Deep Research (Gemini-2.5-Pro) |
| `Grok3_rubric_dataset.json`       | Grok3 DeepSearch                      |
| `Grok3deeper_rubric_dataset.json` | Grok3 DeeperSearch                    |
| `Perplexity_rubric_dataset.json`  | Perplexity Deep Research              |

### LLMs with Search Tools (evaluated in paper)

| Output File                                 | Full Model Name                 |
| ------------------------------------------- | ------------------------------- |
| `gpt-4o-search-preview_rubric_dataset.json` | GPT-4o Search Preview           |
| `sonar-reasoning-pro_rubric_dataset.json`   | Perplexity: Sonar Reasoning Pro |

### Additional Models (from repository, not evaluated in paper)

| Output File                            | Source File           |
| -------------------------------------- | --------------------- |
| `Claude_rubric_dataset.json`           | Claude.json           |
| `Doubao_rubric_dataset.json`           | Doubao.json           |
| `Mita_rubric_dataset.json`             | Mita.json             |
| `perplexity-sonar_rubric_dataset.json` | perplexity-sonar.json |

## Dataset Format

Each output file in `output/` is a single `RubricDataset` JSON:

```json
{
  "name": "OpenAI",
  "prompt": "Provide a response to the question.",
  "rubric": null,
  "items": [
    {
      "submission": "{\"question\": \"...\", \"response\": \"...\"}",
      "description": "Q1 [Open Consulting] Synthetic Data",
      "ground_truth": null,
      "rubric": [
        {"name": "C1", "weight": 2, "requirement": "Explains the importance of ..."},
        {"name": "C2", "weight": 2, "requirement": "Describes at least three ..."}
      ]
    }
  ]
}
```

- `submission`: Serialized JSON string containing `{"question": "...", "response": "..."}`
- `description`: Formatted as `Q{id} [{category}] {subject}`
- `ground_truth`: Always `null` (no gold-standard verdicts)
- `rubric`: Per-item list of `Criterion` dicts

A `test_rubric_dataset.json` with 2 items is included for quick validation.

## Conversion Script

`convert_to_rubric_dataset.py` converts the raw ResearcherBench data into AutoRubric format.

**Source data:**
- `eval_data/questions.json` — Question metadata (id, question, category, subject)
- `eval_data/rubric.json` — Per-question rubrics (id, list of {point, weight})
- `user_data/*.json` — System responses (id, question, response)

**Usage:**
```bash
python convert_to_rubric_dataset.py
```

Output is written to `output/`.


## Evaluation Results

We evaluated three Deep AI Research Systems — OpenAI DeepResearch, Gemini DeepResearch (Gemini-2.5-Pro), and Grok3 DeepSearch — across all 65 ResearcherBench questions using AutoRubric's automated grading pipeline. The 65 questions carry a total of **931 binary rubric criteria**, with the number of criteria per question ranging from 6 to 21 (mean 14.3, median 14, SD 3.1; interquartile range 12–16.5). Each system's response was independently judged against every criterion by two LLM judges — Claude Sonnet-4.5 and Gemini-3-Flash — yielding **5,586 individual criterion-level judgments** across the full experiment (931 criteria × 3 systems × 2 judges). We report the mean coverage score (weighted rubric satisfaction averaged across all 65 questions) alongside the evaluation cost in USD for each judge. For reference, we include the rubric coverage assessment scores reported by Du et al. (2025), which were obtained using Sonnet-3.5 as the judge.

### Results

|                                         | Sonnet-4.5 | Cost (USD) | Gemini-3-Flash | Cost (USD) | Du et al. (2025) Sonnet-3.5 |
| --------------------------------------- | ----------:| ----------:| --------------:| ----------:| ----------------------------:|
| OpenAI DeepResearch                     |      0.620 |      37.98 |          0.771 |       6.72 |                       0.7032 |
| Gemini DeepResearch                     |      0.692 |      61.23 |          0.810 |       9.77 |                       0.6929 |
| Grok3 DeepSearch                        |      0.579 |      12.37 |          0.618 |       1.99 |                       0.4414 |
|                                         |            |            |                |            |                              |
| Spearman Correlation with Du et al. (2025) | 0.5     |      —     |          0.5   |       —    |                          1.0 |

### Analysis

Both Sonnet-4.5 and Gemini-3-Flash produce the same system ranking: Gemini DeepResearch > OpenAI DeepResearch > Grok3 DeepSearch. This cross-judge agreement in system ordering lends confidence to the robustness of the automated evaluation. However, the two judges differ in absolute score calibration — Gemini-3-Flash assigns consistently higher coverage scores than Sonnet-4.5 across all three systems, suggesting a more lenient grading posture.

The system-level ranking produced by AutoRubric with either Sonnet-4.5 or Gemini-3-Flash yields perfect rank correlation with itself (Spearman's ρ = 1.0; Kendall's τ = 1.0), though this result is descriptive given the small sample size (n = 3). In contrast, alignment between the AutoRubric-based rankings and the Du et al. (2025) rubric-coverage assessment with Sonnet-3.5 is limited, with weak-to-moderate rank agreement (Spearman's ρ = 0.50; Kendall's τ = 0.33). The observed disagreement arises from a reversal in the top two systems relative to Du et al., while agreement is retained for the lowest-ranked system.

Evaluation cost varies substantially across judge models. Gemini-3-Flash is considerably cheaper than Sonnet-4.5 — roughly 5–6× lower cost — while producing concordant system rankings. For the full 65-question evaluation, Sonnet-4.5 costs range from $12.37 (Grok3) to $61.23 (Gemini), whereas Gemini-3-Flash costs range from $1.99 to $9.77. The cost disparity across systems reflects differences in response length: longer responses require more tokens per criterion evaluation, inflating the total cost.
