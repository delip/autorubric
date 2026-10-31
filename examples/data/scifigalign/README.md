# SciFigAlign Demo Dataset

## Overview

SciFigAlign (Xu et al., 2026) scores a figure from a machine-learning paper on four peer-review dimensions, Clarity (CL), Relevance (RE), Informativeness (IN) and Structure (ST), each scored from 1 to 5. Its GitHub repository ships a demo of 300 human-rated figures from ICLR 2024 papers, one figure from each of 300 papers. Each figure is a PNG crop extracted automatically from the paper's PDF, with its caption and one score per dimension. The demo is split by paper into 240 train, 30 val and 30 test figures. By the rows' `figure_type` field it holds 117 architecture diagrams, 100 figures typed `other` and 83 qualitative examples.

The files in this directory turn that demo into AutoRubric `RubricDataset` files. Each submission is a document of two parts, the figure (an `Image`) followed by its caption. They are for grading with a vision LLM judge or with a decision model that accepts images.

The demo is not the paper's test set. The paper reports its results on 396 human-rated test figures drawn from a 3,857-figure corpus, and its models also read the paragraphs that cite each figure. The demo carries no such context (see *Caveats*).

**Citation:**

> Xu, C., Deng, Z., Liang, H., Yue, C., Cui, Z., Ye, P., & Cai, W. (2026). SciFigAlign: Scoring Scientific Figures by Fine-tuned Alignment of Visuals with Manuscript Evidence. arXiv: [2607.27066](https://arxiv.org/abs/2607.27066)
> Repository: [github.com/FrankDengAI/SciFigAlign](https://github.com/FrankDengAI/SciFigAlign)

**Licenses.** The SciFigAlign repository is released under the MIT license (its `LICENSE` file), and the demo files used here come from that repository. The full corpus on Hugging Face ([haihanlamu/SciFigAlign](https://huggingface.co/datasets/haihanlamu/SciFigAlign)) is released under Apache-2.0. It is out of scope here, and nothing in this directory comes from it.

## Generating the Files

Only this README is tracked. The dataset files and figures are git-ignored, and the converter writes them:

```bash
uv run python scripts/convert_scifigalign.py [--cache-dir .cache/scifigalign] \
    [--out-dir examples/data/scifigalign] [--force]
```

The script fetches the repository's `data/` folder at the pinned commit `a1f0ce7c2bea5ddd4f380771d3cd477b1ada3ba0` into `.cache/scifigalign` with a shallow, blob-less, sparse `git` clone, which needs git 2.27 or later. Nothing from the clone is executed. It checks every row and every figure, copies the 300 PNGs (about 32 MB) and writes

```
examples/data/scifigalign/{train,val,test,all}.json
examples/data/scifigalign/figures/<paper_id>__<figure_id>.png
```

`all.json` holds the 300 figures of the three splits. The script then loads each file back with `RubricDataset.from_file`, checks that every image part names a PNG in `figures/` and that every label names an option of its criterion, and prints the score histograms. A later run reuses the cache. It refuses to overwrite earlier output unless given `--force`, which also replaces a cache that is not a clean checkout of the pinned commit.

## File Format

Each file is a serialized `RubricDataset` with the rubric embedded:

```python
from autorubric import RubricDataset

dataset = RubricDataset.from_file("examples/data/scifigalign/test.json")
```

In the JSON an item's submission is `[{"image": "figures/<paper_id>__<figure_id>.png"}, "<caption>"]`, the image path relative to the dataset file. No image is read when the file loads. The dataset's prompt is empty, so an LLM judge's prompt has no `<input>` block and a decision model's state has no `input` field.

| Upstream field | In the dataset |
| --- | --- |
| `figure_image_path` | The submission's first part, an `Image` of `figures/<paper_id>__<figure_id>.png` |
| `caption_text` | The submission's second part, verbatim |
| `scores.<dimension>.score` | `ground_truth`, one option label per dimension in the order CL, RE, IN, ST |
| `scores.<dimension>.rationale` | `ground_truth_reasons`, aligned with `ground_truth`, stripped, `null` where blank; the field is left out when all four are blank |
| `paper_id`, `figure_id`, `figure_type`, `venue`, `label_source` | `description`, `"<paper_id> <figure_id> (<figure_type>, <venue>, <label_source>)"` |
| `paper_title` | Dropped. In every row it holds the venue string `ICLR_2024`, not a title |
| `citing_paragraphs`, `nearby_context`, `abstract`, `section_title` | Dropped. They are empty in every row, and the converter refuses a row in which one is not |

Every row's `label_source` is `human` (the converter refuses any other), and every score is a whole number from 1 to 5, written upstream as a float such as `4.0`.

## The Rubric

`build_rubric()` in the converter defines the rubric, and every file embeds the same copy. It has four ordinal criteria of weight 1.0, one per dimension, each with five options for scores 1 to 5, valued 0, 0.25, 0.5, 0.75 and 1. There is no "not applicable" option, because the annotators had none. Each criterion's requirement is the dimension's definition in Section 3.1 of the paper, verbatim except that its typographic apostrophes are written as ASCII.

An option's label is `"Score k: <descriptor>"`. A `CriterionOption` has no description field, and the judge needs to see what each score means, so the descriptor goes into the label. Each descriptor starts with the paper's Table 2 entry for that score. At scores 1, 3 and 5 the sentence the human annotators used as a rationale template for that tier follows it, where one exists. Each template is attached to a fixed level (low-tier templates to 1, mid-tier to 3, high-tier to 5), although the annotators used them with neighbouring scores too.

The templates and their counts come from `frontend/public/dataset_overview.json` at the pinned commit, which lists all 3,857 figures of the corpus with their rationales. The converter does not fetch that file (`git -C .cache/scifigalign show HEAD:frontend/public/dataset_overview.json` prints it). A count is the number of the file's 1,982 human-rated rows whose rationale for that dimension is exactly the template.

| Dimension | Level | Template appended to the descriptor | Human rows | Their scores |
| --- | :---: | --- | :---: | --- |
| Clarity | 1 | "The large area is unclear, and key elements are difficult to identify." | 2 | 1, 2 |
| Clarity | 3 | "A little area is unclear, and key elements are difficult to identify." | 1 | 3 |
| Clarity | 5 | "Almost all elements are clear and easy to read, allowing readers to quickly identify the content in the diagram." | 51 | 4 (23), 5 (28) |
| Relevance | 1 | none | | |
| Relevance | 3 | "The relevant parts show a connection, but their supporting role is limited." | 25 | 2 (3), 3 (17), 4 (5) |
| Relevance | 5 | "Highly relevant, closely aligned with the caption and main text content, with images and text supporting each other." | 52 | 4 (21), 5 (31) |
| Informativeness | 1 | "There is very little information; it is hardly helpful for understanding." | 7 | 1 (2), 2 (4), 3 (1) |
| Informativeness | 3 | "There is some information, but it is incomplete, or valid information coexists with redundant information." | 20 | 2 (3), 3 (14), 4 (3) |
| Informativeness | 5 | "The information is comprehensive and effective, with no obvious redundancy, clearly conveys the core content, and significantly aids understanding." | 60 | 4 (24), 5 (36) |
| Structure | 1 | "The structure is chaotic, the reading path is unclear, and the module relationships are difficult to determine." | 15 | 1 (8), 2 (5), 3 (2) |
| Structure | 3 | "The structure is generally followable, but the sense of hierarchy or organization is average." | 13 | 3 (10), 4 (3) |
| Structure | 5 | "The hierarchy is clear, the layout is logical, and the processes or relationships are easily understandable." | 29 | 4 (4), 5 (25) |

Relevance has no low-tier template, so its score-1 option carries only the Table 2 descriptor ("Clearly unrelated to paper.").

One frequent Structure rationale is not used. "There is very little information; it is primarily decorative and hardly helpful for understanding." is written as the Structure rationale of 33 human rows, 31 of them scored 4 or 5 on Structure. It describes informativeness, not structure, and contradicts the scores it accompanies, so we treat it as a template copied into the wrong dimension.

The templates come from the human rows only. The corpus's 1,875 rows labelled by GPT-4o carry free-text rationales. None of them is one of the templates above, and in each dimension 1,873 to 1,875 of the 1,875 are distinct.

The rubric's guidelines tell the judge that it reviews a figure cropped automatically from an ICLR 2024 paper, followed by its caption, and that it should score the four dimensions independently. They say that the annotators also read the citing paragraphs, which the judge does not see, that a caption may be cut at 300 characters or carry PDF-extraction artefacts that are not the figure's fault, that an incomplete crop is scored as shown, and that legibility is judged at the size the figure has in the paper. The full text is `GUIDELINES` in the converter.

## Score Distributions

Counts of figures per score, computed from the converted files (the converter prints the same table at the end of a run):

| Split | Dimension | 1 | 2 | 3 | 4 | 5 |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| all (300) | Clarity | 12 | 29 | 89 | 102 | 68 |
| | Relevance | 10 | 17 | 46 | 144 | 83 |
| | Informativeness | 19 | 20 | 93 | 120 | 48 |
| | Structure | 22 | 31 | 100 | 92 | 55 |
| train (240) | Clarity | 9 | 20 | 71 | 82 | 58 |
| | Relevance | 7 | 16 | 35 | 118 | 64 |
| | Informativeness | 15 | 15 | 70 | 97 | 43 |
| | Structure | 17 | 22 | 80 | 76 | 45 |
| val (30) | Clarity | 2 | 6 | 7 | 11 | 4 |
| | Relevance | 3 | 1 | 5 | 13 | 8 |
| | Informativeness | 3 | 4 | 10 | 11 | 2 |
| | Structure | 3 | 6 | 8 | 8 | 5 |
| test (30) | Clarity | 1 | 3 | 11 | 9 | 6 |
| | Relevance | 0 | 0 | 6 | 13 | 11 |
| | Informativeness | 1 | 1 | 13 | 12 | 3 |
| | Structure | 2 | 3 | 12 | 8 | 5 |

Scores of 3 and 4 dominate every dimension. The test split has no Relevance score below 3, so its Relevance metrics are measured on scores 3 to 5 only.

## Caveats

- **Captions are cut.** 225 of the 300 captions (184 train, 21 val, 20 test) are exactly 300 characters long, cut off upstream. Some also carry text extracted from the figure itself, and words broken across lines in the PDF ("classifica- tion", "de- scribed"). The guidelines tell the judge not to penalize the figure for either.
- **No citing context.** The annotators read the paragraphs that cite each figure, and the paper's models use them. The demo has none, so the judge assesses Relevance and Informativeness against the caption and the figure alone.
- **Few rationales.** Only 49 rationale cells are non-blank, all on 16 train figures (11 each for CL, RE and IN, 16 for ST). Eight of them hold another dimension's template, on seven figures. Seven are the misplaced Structure sentence described above, on `ICLR_2024_00548 fig_04`, `ICLR_2024_02831 fig_07`, `ICLR_2024_03080 fig_13`, `ICLR_2024_04706 fig_02`, `ICLR_2024_05175 fig_04`, `ICLR_2024_06425 fig_06` and `ICLR_2024_07308 fig_02`. The eighth is the Relevance high-tier template given as the Clarity rationale of `ICLR_2024_03080 fig_13`. They are kept verbatim in `ground_truth_reasons`. Few-shot examples show reasons only with `FewShotConfig(include_reason=True)`, which `examples/scifigalign_demo.py` does not set.
- **Broken crops score low.** Some crops are incomplete or are not figures at all, and the annotators scored them low instead of leaving them out. Four train figures have a Structure score of 1 with rationales such as "Image incomplete" and "Strictly speaking, it is not a scientific image."
- **Small crops.** The crops are 287 to 1,275 px wide and 89 to 535 px high. 75 of them (56 train, 10 val, 9 test) have a side under 280 px, the smallest side Jev accepts. AutoRubric scales such images up before sending them to a decision model and pads the 18 crops too elongated to reach 280 px by scaling (2 of them in test) with white, so the decision model sees the fitted image. LLM judges get the crop as it is.
- **Small test split.** With 30 test figures, a per-dimension MAE moves by about 0.03 for each figure whose score is off by one more point. Bootstrap intervals (`compute_metrics(bootstrap=True)`) are wide.

## Mapping the Paper's Metrics

The paper reports MAE, Spearman's rank correlation (SRCC) and within-paper pairwise accuracy (PA), in score points on the 1 to 5 scale. AutoRubric's metrics map onto them as follows.

| Paper | AutoRubric |
| --- | --- |
| MAE of one dimension, 1 to 5 scale | `4 × OrdinalCriterionMetrics.mae`; option values are `(score − 1) / 4`, so an error on them times 4 is an error in score points. RMSE likewise: `4 × OrdinalCriterionMetrics.rmse` |
| Macro MAE | The mean of the four per-dimension MAEs above |
| SRCC of the baselines (SigLIP2-Ridge, LLM-as-judge): the macro average of the per-dimension Spearman coefficients, as the caption of the paper's Table 4 says | The mean of the four `OrdinalCriterionMetrics.spearman.coefficient` values, which the demo prints as "macro SRCC" |
| SRCC of SciFigAlign: Spearman on the overall score | `MetricsResult.score_spearman`, which the demo prints as "SRCC of the overall score". The four weights are equal and the rubric has no abstain option, so an item's normalized score is the mean of its four option values, a linear function of the mean of the four scores, and the rank correlation is the same |
| PA | Not computable. PA counts pairs of figures from the same paper whose gold scores differ by at least 0.5, and the demo's 300 figures come from 300 different papers, so there are 0 such pairs |

Per dimension, `OrdinalCriterionMetrics` also gives `exact_accuracy`, `adjacent_accuracy` (within one point), `weighted_kappa` (quadratic) and `spearman`, which the demo prints.

## Reference Results and Baselines

The paper's results on its own 396-figure test set, as given in the repository (`index.html`, `README.md` and `config/paper_config.json` at the pinned commit):

| Method | Macro MAE | SRCC | Which SRCC | PA |
| --- | :---: | :---: | :---: | :---: |
| Mean / median predictor | 0.90–0.94 | — | — | 0% |
| SigLIP2-Ridge | 0.877 | 0.266 | macro | 65.8% |
| LLM-as-judge (v9) | 0.864 | 0.198 | macro | 63.2% |
| SciFigAlign (fine-tuned CLIP + SciBERT) | 0.3524 | 0.3088 | overall | 81.64% |

The repository gives the three SRCCs in one column, but they are two statistics. The caption of the paper's Table 4 says the baselines' SRCC is the macro average of the per-dimension Spearman coefficients and SciFigAlign's is on the overall score. Compare an LLM judge's run with the LLM-as-judge row on its macro SRCC, not on `score_spearman`. The demo prints each reference row next to the same kind of SRCC from the run.

The paper also gives the LLM judge's per-dimension MAE as 0.902 for Clarity, 0.828 for Informativeness and 0.947 for Structure. These three numbers come from the paper and are not in the repository.

None of these rows can be compared directly with a run on this directory's test split. They were measured on 396 different figures, with citing paragraphs as input, and SciFigAlign was trained on the full corpus.

Predicting the same scores for every figure costs nothing. On this directory's test split, the macro MAE of such predictions is as follows (`examples/scifigalign_demo.py` prints this table without spending anything):

| Constant prediction | Macro MAE | Macro RMSE |
| --- | :---: | :---: |
| Train mean per dimension (3.667, 3.900, 3.575, 3.458) | 0.772 | 0.937 |
| Train mean rounded to a score (4, 4, 4, 3) | 0.742 | 1.000 |
| Always 4 | 0.775 | 1.028 |
| Always 3 | 0.892 | 1.164 |

A judge whose macro MAE on the test split is not below about 0.74 does no better than a constant.

## Running the Demo

`examples/scifigalign_demo.py` grades a split with an LLM judge (`--judge llm`, default `openai/gpt-5.6-luna`) or with Jev (`--judge jev`, default `typesafe/jev-1.14.0`). Without `--run` it spends nothing. It prints the split's histograms, the constant baselines and an estimate of the run's tokens and cost, then exits.

```bash
uv run python examples/scifigalign_demo.py                       # LLM judge, dry run
uv run python examples/scifigalign_demo.py --llm-calls per_criterion
uv run python examples/scifigalign_demo.py --few-shot 3          # examples from train.json
uv run python examples/scifigalign_demo.py --judge jev --input-cost 0.042e-6
uv run python examples/scifigalign_demo.py --judge jev --limit 3 --run
```

The LLM judge grades all four dimensions in one call per figure by default (`--llm-calls per_item`), so each figure is sent once. With `--llm-calls per_criterion` it is sent four times. Jev makes one request per figure. `--input-cost` gives Jev's price per input token (`DecisionModelConfig.input_cost_per_token`), since autorubric has no price table for decision models. `typesafe/jev-1.13.0` takes no images, and autorubric's decision-model registry refuses it with a configuration error before any request is sent. With `--run` the demo prints that error. Both judges are built with `auto_na_option=False`, so neither can abstain, as the annotators could not. Responses are cached on disk (`cache_enabled=True`), so a repeated run is free.
