# RiceChem Dataset

## Overview

The RiceChem dataset (Sonkar et al., 2024) comprises 1,240 human-graded, long-form student responses to four college-level chemistry exam questions. Originally introduced to study *Automated Long Answer Grading* (ALAG) as a rubric entailment problem, the dataset provides binary (TRUE/FALSE) annotations for 27 rubric criteria across four questions, along with numerical scores assigned by teaching assistants. The average response length is approximately 120 words, substantially longer than prior short-answer grading corpora such as SciEntsBank (~13 words) or Beetle (~10 words).

The files in this directory contain the RiceChem data converted into the AutoRubric `RubricDataset` format, enabling direct use for LLM-as-a-judge evaluation and benchmarking.

**Citation:**

> Sonkar, S., Ni, K., Tran Lu, L., Kincaid, K., Hutchinson, J. S., & Baraniuk, R. G. (2024). Automated Long Answer Grading with RiceChem Dataset. In *Proceedings of the 25th International Conference on Artificial Intelligence in Education (AIED 2024)*, LNCS vol. 14829. Springer.
> arXiv: [2404.14316](https://arxiv.org/abs/2404.14316)

## Questions and Rubric Structure

The four questions cover distinct topics from introductory chemistry, each graded against an independent rubric. Questions 1--3 expect responses of approximately 150 words; Question 4 expects approximately 75 words.

| Question  | Topic                                                         | Criteria | Students  | Max Score |
| --------- | ------------------------------------------------------------- | :------: | :-------: | :-------: |
| Q1        | Ionization energies of silicon via Coulomb's Law              |    8     |    327    |    ~8     |
| Q2        | Quantized absorption vs. continuous photoejection frequencies |    6     |    317    |    ~8     |
| Q3        | Hybrid orbital analysis in methanimine (CH₂NH)                |    7     |    298    |     9     |
| Q4        | Law of Multiple Proportions and atomic theory                 |    6     |    298    |    ~8     |
| **Total** |                                                               |  **27**  | **1,240** |           |

Each criterion is binary: a student response either satisfies it (MET) or does not (UNMET). The original dataset also contains several negative/flag columns per question (e.g., "incorrect", "Blank") that indicate error patterns rather than positive achievements. These flag columns are excluded from the rubric criteria and are instead used only to filter blank or degenerate submissions.

### Q1: Silicon Ionization Energies

Students explain why successive ionization energies of silicon increase, and why the jump from the 4th to 5th ionization energy is disproportionately large, using core charge calculations and Coulomb's Law.

| Criterion                                                 | Weight | MET Rate |
| --------------------------------------------------------- | :----: | :------: |
| Correctly cites decreased electron-electron repulsion     |  1.01  |  83.8%   |
| Relates decreased repulsion to decreased potential energy |  1.03  |  63.3%   |
| 3rd and 4th electrons feel same core charge               |  0.96  |  58.7%   |
| 3rd and 4th electrons ionized from n=3 shell, same radius |  1.00  |  55.0%   |
| 5th electron from n=2 shell feels higher core charge      |  0.95  |  71.3%   |
| 5th electron from n=2 shell has smaller radius            |  0.98  |  83.2%   |
| Correctly explains PE–IE relationship (full)              |  1.97  |  41.9%   |
| Partially explains PE–IE relationship                     |  0.99  |  17.1%   |

### Q2: Light Absorption vs. Photoejection

Students reconcile two observations: only certain frequencies excite electrons to higher levels (quantized transitions), while any frequency above a threshold can eject electrons (photoelectric effect).

| Criterion                                     | Weight | MET Rate |
| --------------------------------------------- | :----: | :------: |
| Frequency proportional to energy of light     |  1.93  |  57.4%   |
| Energy levels of an electron are quantized    |  0.96  |  49.8%   |
| Fully explains energy/frequency condition     |  2.04  |  27.1%   |
| Partially explains energy/frequency condition |  1.02  |  24.9%   |
| Minimum energy needed to eject electron       |  0.96  |  72.6%   |
| Additional energy becomes kinetic energy      |  1.98  |  42.0%   |

### Q3: Hybrid Orbitals in Methanimine

Students assess a (deliberately flawed) peer response about hybrid orbitals in CH₂NH, identifying errors in the claimed sp3 hybridization of carbon and the claim that nitrogen does not hybridize.

| Criterion                                        | Weight | MET Rate |
| ------------------------------------------------ | :----: | :------: |
| Sentence 1 correct: VBT half-filled orbitals     |  2.00  |  51.7%   |
| Sentence 2: correct number of hybrid orbitals    |  2.00  |  40.9%   |
| Sentence 2: correct type (sp2)                   |  1.00  |  58.7%   |
| Sentence 3: nitrogen is hybridized               |  1.00  |  69.1%   |
| Sentence 3: correct hybridization type (sp2)     |  1.00  |  69.1%   |
| Sentence 3: hybrid orbital bonds described       |  1.00  |  15.1%   |
| Sentence 3: unhybridized orbital bonds described |  1.00  |  24.5%   |

### Q4: Law of Multiple Proportions

Students explain how the Law of Multiple Proportions provided evidence that matter is composed of atoms.

| Criterion                                | Weight | MET Rate |
| ---------------------------------------- | :----: | :------: |
| Fixed mass of one element                |  0.98  |  89.3%   |
| Mass data in Law of Multiple Proportions |  0.98  |  88.6%   |
| Combine to form compounds                |  0.98  |  88.3%   |
| Integer/whole number ratio               |  1.01  |  93.0%   |
| Whole numbers mean indivisible/discrete  |  1.98  |  80.5%   |
| Indivisible unit of mass = atom          |  2.00  |  67.8%   |

## Conversion Process

The raw data consists of eight CSV files: four *Student Answers* files containing student identifiers (`sis_id`) and free-text responses, and four *Graded Rubric* files containing per-student binary criterion annotations, numerical scores, and optional TA adjustments.

### Criterion Selection

The original graded rubrics contain both *positive* criteria (knowledge the student demonstrated) and *negative/flag* columns (error indicators and blanks). Following the approach in Sonkar et al. (2024), which reports 27 total rubric items, we retain only the positive criteria:

- **Q1:** 8 of 11 columns retained (excluding `incorrect`, `Blank`, `Core charge calculation error`)
- **Q2:** 6 of 9 columns retained (excluding `Incorrect statement included`, `Incorrect`, `Blank`)
- **Q3:** 7 of 9 columns retained (excluding `Correct response`, `Incorrect/Blank response`)
- **Q4:** 6 of 8 columns retained (excluding `incorrect/misleading statement`, `incorrect/missing answer`)

### Weight Inference

The raw data does not explicitly encode per-criterion point values. We infer weights by solving a least-squares regression per question:

$$\text{Score}_i - \text{Adjustment}_i = \sum_{j=1}^{m} w_j \cdot \mathbf{1}[\text{criterion}_j = \text{TRUE}]$$

where the `Adjustment` column captures manual TA score modifications. The inferred weights cluster tightly around integer values (1 or 2 points per criterion), consistent with the grading rubric design. Goodness-of-fit is measured by $R^2$:

| Question | $R^2$ | Inferred Weight Range |
| -------- | :---: | :-------------------: |
| Q1       | 0.994 |     0.95 -- 1.97      |
| Q2       | 0.986 |     0.96 -- 2.04      |
| Q3       | 0.542 |     1.00 -- 2.00      |
| Q4       | 0.985 |     0.98 -- 2.00      |

**Q3's lower $R^2$** is attributable to a grading shortcut in the original annotations: 13 of 15 students marked with the flag column `Correct response = TRUE` received a full score of 9 but have *all individual criteria marked FALSE*. These students were evidently graded holistically rather than criterion-by-criterion. Since the `Correct response` column is a summary flag (not a rubric criterion), it is excluded from the converted rubric, leaving these 13 responses with ground truth labels that undercount their actual performance. An additional 51 responses (16.3%) carry manual `Adjustment` values of $\pm1$ or $\pm2$ points, introducing further variance not captured by the binary criteria alone. Together, these two annotation artifacts account for the majority of the unexplained variance. For Q1, Q2, and Q4, the near-perfect $R^2$ values ($\geq 0.985$) confirm that the inferred weights faithfully recover the original scoring rubric.

### Blank Filtering

Student submissions that are empty or flagged as blank in the graded rubric are excluded. This removes 1 submission from Q1, 11 from Q2, 10 from Q3 (combining blank and those present only in the rubric file but absent from the answers file), and 2 from Q4.

### Ground Truth Encoding

Each binary criterion annotation is mapped to a `CriterionVerdict`: `TRUE` becomes `MET` and `FALSE` becomes `UNMET`. The resulting ground truth vectors enable direct computation of agreement metrics (accuracy, Cohen's kappa, F1) between human annotations and LLM-as-a-judge predictions.

## Score Distributions

| Question | Mean Score | Std Dev |  Min  |  Max  |
| -------- | :--------: | :-----: | :---: | :---: |
| Q1       |    5.09    |  2.10   |  0.0  |  7.9  |
| Q2       |    3.92    |  2.20   |  0.0  |  7.9  |
| Q3       |    4.18    |  2.60   |  0.0  |  9.0  |
| Q4       |    6.65    |  1.75   |  0.0  |  7.9  |

Scores are computed as $\sum_j w_j \cdot \mathbf{1}[\text{criterion}_j = \text{MET}]$ using the inferred weights. Q4 exhibits the highest average score (6.65), reflecting that most criteria have MET rates above 80%. Q2 is the most challenging, with several criteria below 30% MET rate.

## File Format

Each JSON file is a serialized `RubricDataset` loadable via:

```python
from autorubric import RubricDataset

dataset = RubricDataset.from_file("examples/data/ricechem/q1.json")
```

The files contain the question prompt, rubric criteria with inferred weights, and all non-blank student submissions with ground truth verdict labels.

## Reproduction

The conversion script is located at `scripts/convert_ricechem.py` and requires the raw CSV files in `ricechem-rawdata/`. To regenerate the JSON files:

```bash
python scripts/convert_ricechem.py
```

## Prior Results

Sonkar et al. (2024) evaluate on the rubric entailment task (per-criterion binary classification) using an **80-10-10 split** per question: 80% of student responses for training, 10% for validation, and 10% for testing, stratified by student so that all rubric-item labels for a given response remain in the same split. Fine-tuned model results are averaged across 5 random seeds; metrics are micro-averaged across all rubric-response pairs in the test set.

**In-distribution** (train and test on all 4 questions):

| Model                | Accuracy |  F1   |
| -------------------- | :------: | :---: |
| RoBERTa-large + MNLI |  86.8%   | 0.888 |
| BART-large + MNLI    |  85.4%   | 0.876 |
| RoBERTa-large        |  84.1%   | 0.864 |
| GPT-4 (zero-shot)    |  70.9%   | 0.689 |

**Cold start** (train on 3 questions, test on the held-out 4th):

| Held-out Question | Accuracy |  F1   |
| ----------------- | :------: | :---: |
| Q1                |  65.9%   | 0.705 |
| Q2                |  68.7%   | 0.629 |
| Q3                |  66.7%   | 0.633 |
| Q4                |  60.6%   | 0.717 |

The NLI-based transfer learning approach (fine-tuning on MNLI, then on RiceChem) substantially outperforms zero-shot LLM prompting in the in-distribution setting, though both formulations treat grading as pairwise entailment between the student response and each rubric criterion. The cold-start results highlight the difficulty of generalizing to unseen question types without in-domain training data.

## Evaluation Results

We evaluate AutoRubric with `gemini/gemini-3-flash-preview` as the backbone LLM on the RiceChem rubric entailment task. Following Sonkar et al. (2024), we use the same 80-10-10 split protocol (seed=42) and report micro-averaged accuracy and F1 across all rubric-criterion pairs on the 10% held-out test set. Few-shot examples are drawn from the 80% training split.

### Comparison with Baselines

| Model                                       | Accuracy | F1    | Source              |
| ------------------------------------------- | :------: | :---: | :-----------------: |
| GPT-4 (zero-shot)                           |  70.9%   | 0.689 | Sonkar et al. (2024) |
| RoBERTa-large + MNLI                        |  86.8%   | 0.888 | Sonkar et al. (2024) |
| AutoRubric + Gemini-3-flash (zero-shot)     |  77.2%   | 0.832 | Ours                |
| AutoRubric + Gemini-3-flash (3-shot)        |  79.0%   | 0.841 | Ours                |

AutoRubric zero-shot surpasses GPT-4 zero-shot by +6.3 accuracy / +14.3 F1, demonstrating the effectiveness of structured rubric-based prompting over direct LLM grading. With 3 few-shot examples, accuracy improves a further +1.8 and F1 +0.9. A gap to the supervised RoBERTa-large + MNLI baseline remains, as expected for a zero/few-shot approach compared to a model fine-tuned on 80% of the training data.

### Few-Shot Ablation

| Shots | Accuracy |  Cost  |
| :---: | :------: | :----: |
|   0   |  77.2%   | $0.51  |
|   3   |  79.0%   | $0.77  |
|   5   |  80.0%   | $0.92  |
|  10   |  79.7%   | $0.84  |
|  20   |  80.8%   | $1.07  |

Accuracy gains are steepest from 0 to 5 shots (+2.8 pp), with diminishing returns beyond that. Cost grows sublinearly with the number of exemplars: going from 0 to 20 shots only doubles the cost ($0.51 → $1.07), whereas linear extrapolation from the initial marginal rate would predict ~$2.25. This sublinearity is attributable to prompt caching and KV-cache reuse — few-shot exemplars form a shared prompt prefix that is cached across per-criterion LLM calls, so each additional exemplar incurs diminishing marginal cost. The 5-shot setting offers a favorable accuracy–cost tradeoff.
