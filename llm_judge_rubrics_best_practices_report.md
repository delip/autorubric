# Best Practices for Rubric-Based LLM-as-a-Judge Evaluation

## Abstract
LLM-as-a-judge has become a dominant paradigm for scalable evaluation of generative model outputs, with reports of >80% agreement with human preferences when evaluation is carefully designed. This paper consolidates best practices for using rubrics (explicit criteria, anchored scoring schemes, and repeatable judging protocols) in LLM-based evaluation. Across surveyed research and industry guidance (G-Eval, MT-Bench, FLASK, FActScore, Prometheus 2, CheckEval, Gecko, and human-centered tooling), evaluation quality depends as much on rubric design, bias mitigation, prompt engineering, and validation as on the judge model itself. We organize recommendations spanning (i) rubric architecture and scoring scales, (ii) judging prompts and protocols, (iii) bias and robustness controls, and (iv) reliability, validity, and operational monitoring.

## 1 Introduction
Rubric-based evaluation is increasingly implemented via LLM-as-a-judge for tasks where human evaluation is costly or slow. Recent work treats rubric-based evaluation as a measurement instrument: an explicit set of criteria (often multi-dimensional) paired with an anchored scoring scheme and a repeatable judging protocol, sometimes with humans in the loop. This framing motivates a "crisp evaluation contract": specify the decision the evaluation will drive (ship gate, regression detection, prompt iteration, model selection, safety monitoring), the artifact being judged (single response, multi-turn dialog, tool-call trace, RAG answer-with-citations), and the context available to the judge (prompt/history, retrieved passages, and/or reference answers).

A central design choice is the judgment format:
* Pointwise scoring evaluates one output in isolation and is commonly used for absolute tracking and monitoring.
* Pairwise (or listwise) comparison ranks candidates and is often preferred for subtle style and quality differences when absolute scores are unstable.

Because LLM judges exhibit systematic failure modes (biases, drift, and domain limitations), best practice is to treat the judge as requiring calibration, meta-evaluation, and ongoing monitoring.

## 2 Background: Rubrics, Formats, and Scales

### 2.1 Pointwise vs. pairwise evaluation
Empirical guidance distinguishes contexts where each format is most appropriate. Pairwise comparisons often correlate better with human preferences for subjective qualities such as helpfulness, coherence, and style; MT-Bench reports human-level agreement rates for GPT-4 pairwise judgments. Pairwise evaluation is vulnerable to position/order inconsistency (judgments can flip under response-order swapping). Pointwise scoring can be preferable for objective checks (factuality, safety compliance) and for production monitoring where single outputs require assessment; recent work also reports lower flip rates under adversarial manipulation for pointwise protocols than pairwise protocols in certain settings.

### 2.2 Scale granularity and "false precision"
Multiple sources converge on avoiding high-precision numeric scales. Best practice is to use low-precision ordinal scales (commonly 0-3 or 1-5) or binary/ternary schemes when fine discrimination is unnecessary. Broad numeric scales (1-10) invite central-tendency and anchoring problems and are difficult to define behaviorally. A recurring recommendation is to include an explicit "cannot assess / insufficient information" option when the judge lacks evidence (especially for factuality/groundedness), reducing forced and noisy guesses.

## 3 Rubric Architecture and Design

### 3.1 Link criteria to the intended construct
Rubric criteria should be explicitly mapped to the construct being measured. Evidence-centered design (ECD) and subject-matter expert (SME) review are repeatedly recommended to support interpretability, construct validity, and defensible score meaning (Casabianca et al., 2025). Teams should design tasks and rubrics jointly, document the linkage from construct to criteria to scoring, and clarify how scores will be used.

### 3.2 Prefer analytic, multi-dimensional rubrics
For open-ended generation, best practice is to use analytic rubrics (multiple criteria) rather than relying only on a holistic score. Analytic rubrics increase interpretability and help diagnose failure modes. Common patterns include "N dimension scores + 1 overall summary question," with the overall item treated as a summary rather than the only score.

Dimensions should be kept as orthogonal as possible to avoid double-counting defects (overlapping "clarity" and "readability" criteria unless anchors distinguish them). Several sources emphasize making rubrics task-specific by default, including question-specific rubrics for code evaluation where correctness depends strongly on the task statement (Rubric Is All You Need, arXiv:2503.23989).

### 3.3 Decompose criteria into atomic checks and checklists
A widely shared recommendation is to decompose evaluation into atomic, observable criteria-often checklist-style boolean items-especially for complex tasks. Frameworks such as CheckEval and factuality-oriented approaches advocate replacing vague dimensions ("completeness") with concrete yes/no questions ("Does the summary mention the protagonist?"). Atomic decomposition for factuality (FActScore) breaks text into minimal verifiable units before scoring support, enabling fine-grained measurement that holistic ratings cannot capture.

Rubrics-as-reward work recommends instance-specific rubrics or per-item checklists rather than rubric-free Likert-only scoring, with reported gains in pairwise preference accuracy and the largest improvements for smaller judge models when rubric guidance is present (Gunjal et al., 2025).

### 3.4 Use hierarchical and "gatekeeper" rubrics
For cost and accuracy, sources recommend hierarchical ("gatekeeper") rubric structures that short-circuit evaluation when prerequisite criteria fail. Example logic: first check if code is executable; if not, assign a failing score before evaluating efficiency. This saves judge calls and prevents downstream criteria from being applied when evidence is absent.

### 3.5 Explicitly include negative constraints
Scoring should address what should not happen, with explicit rubric items for hallucinations, safety violations, and verbosity/over-apologizing. Including negative constraints helps prevent rubrics from rewarding superficially fluent but harmful or ungrounded outputs.

### 3.6 Write behaviorally anchored score descriptions (and use exemplars)
Score points should be defined with behavioral descriptors rather than abstract "vibes." Describing only the extreme anchors (highest and lowest score) can be more reliable than ambiguous intermediate anchors. Graded exemplars ("gold anchors"), including negative examples of common failure modes, are repeatedly recommended for both human and LLM judge calibration (Casabianca et al., 2025; Ashktorab et al., 2025).

## 4 Judging Prompts and Protocols

### 4.1 Document prompts, decoding, and few-shot examples
Prompt wording, context presentation, and decoding parameters materially affect judge outputs. Best practice is to document the exact prompt text, few-shot examples, model versions, and sampling settings (temperature/top-k) for reproducibility and auditability (Casabianca et al., 2025).

### 4.2 Structure judging as form-filling with per-criterion scoring
A recurring recommendation is to structure the judge task as form-filling (structured output schema) that returns per-criterion scores (and optionally brief evidence/rationale) in a strict format. This improves repeatability and parsing reliability and encourages the judge to score each dimension rather than jumping to a holistic verdict. Best practice is to judge each criterion separately (often in separate calls) and compute overall scores via explicit aggregation afterward.

### 4.3 Use chain-of-thought and stepwise procedures appropriately
G-Eval-style prompting emphasizes stepwise evaluation procedures and reasoning before scoring. Enforcing chain-of-thought (CoT) so that the judge produces an explicit reasoning argument before the final score is recommended; others recommend using CoT selectively, enabling it only when rubric criteria require stepwise reasoning and avoiding unnecessary CoT when criteria are already clear.

Related protocols add critique/self-review cycles: instruct the judge to critique and revise an initial judgment. Empirical results in REGAI report reduced error magnitude and improved differentiation under a critique cycle, while also motivating iterative exploration and calibration to human raters (Johnson and Straub, 2024).

### 4.4 Use reference-aware judging when applicable
When high-quality references exist, sources recommend reference-grounded rubrics and judge prompts that explicitly measure overlap between the generated answer and the reference (especially for RAG and math). For creative or reference-free tasks, rubrics should focus on intrinsic constraints (tone, format, logic) rather than reference overlap. Access to reference answers and expert grounding substantially improves rubric quality (Gunjal et al., 2025).

### 4.5 Prefer repeated sampling, distributions, and probabilistic scoring when needed
Several sources recommend not relying on a single deterministic judge sample for subjective or borderline cases. Instead:
* Aggregate multiple stochastic evaluations to reduce noise.
* When possible, work with distributions rather than only point estimates; distribution-aware meta-evaluation is recommended where LLM errors are systematic and reproducible (He et al., 2025).
* Use probabilistic scoring (G-Eval method): compute scores from token probability mass over score options, yielding continuous values (e.g., 4.23) rather than only the single sampled integer.

## 5 Bias, Robustness, and Adversarial Considerations

### 5.1 Position/order bias and counterbalancing
LLM judges can exhibit strong position bias in pairwise settings, sometimes preferring a fixed position independent of content. Best practice is swap augmentation: evaluate (A,B) and (B,A) and only accept preferences that select the same content regardless of position. Randomizing presentation order and measuring positional effects is recommended as part of routine diagnostics.

### 5.2 Verbosity bias and length control
LLM judges often prefer longer answers. Recommended mitigations include adding conciseness as an explicit rubric dimension, constraining response length, and using length-controlled win rates or normalization (conditioning on comparable lengths) to reduce verbosity-driven inflation.

### 5.3 Self-preference and preference leakage
Multiple sources document self/egocentric biases where a model rates its own family higher. Best practice is cross-family judging (avoid using the same model family as generator and judge), and maintaining independence between (i) models used for synthetic data generation, (ii) student models being trained, and (iii) judge models used for evaluation to prevent preference leakage contamination.

### 5.4 Sycophancy and false positives; adversarial testing
"Safety-style" failure modes exist where judges accept invalid outputs (low true-negative rates) and can be fooled by fluent but incorrect answers. Recommended mitigations include incorporating negative examples into calibration materials, explicitly defining negative constraints in rubrics, and performing adversarial testing with "trick" inputs to ensure rubrics actively detect incorrect-but-fluent failures.

### 5.5 Fairness analyses across subgroups
For high-stakes or user-facing evaluation, sources recommend fairness checks and subgroup analyses. Suggested procedures include standardized mean difference (SMD) and quadratic-weighted kappa (QWK) by subgroup, empirical-Bayesian SMD for small samples, and differential item/algorithmic functioning analyses comparing human vs machine patterns (Casabianca et al., 2025).

### 5.6 Debiasing modules, ensembles, and multi-agent protocols
Advanced debiasing approaches cited include reasoning-based bias detection and unsupervised debiasing alignment, as well as training-time augmentation approaches (JudgeLM). For subjective criteria or high-stakes settings, best practice is to use multi-LLM ensembles/panels (majority vote or meta-judge pipelines) and, in some settings, multi-agent debate/discussion-based judging (ChatEval, CourtEval) to diversify perspectives-while noting that ensembles do not automatically remove shared systematic biases.

## 6 Reliability, Validity, and Calibration

### 6.1 Plan calibration and double-scoring a priori
Rubric-based evaluation should specify a calibration plan in advance:
* Define a double-scored subsample.
* Train/certify raters (human or machine-judge prompts).
* Measure agreement with appropriate statistics (ICC, Krippendorff's alpha, QWK; Casabianca et al., 2025).

For LLM judges, best practice is to benchmark judge outputs against a human-labeled subset (meta-evaluation) and iteratively refine rubric text and prompts until agreement is acceptable.

### 6.2 Use both agreement metrics and distribution-aware comparisons
Sources emphasize that correlation alone can mask systematic bias. In addition to agreement/correlation, distribution-aware metrics (Earth Mover's Distance) and separate reporting for high- vs low-agreement subsets can reveal systematic deviations (He et al., 2025).

### 6.3 Operational validation, monitoring, and drift
Operational protocols include maintaining a "gold set" of human-graded examples (often suggested at 50-100 items) and using Cohen's kappa as an alignment check with thresholds (refine rubrics if kappa < 0.7). Another recurring recommendation is to sample 1-5% of production traffic for continuous human validation and to monitor judge drift over time, especially after rubric or model updates.

To reduce Goodharting and test-set leak, sources recommend treating the evaluation set as a defended benchmark: match real usage distributions, include edge cases and adversarial prompts, keep hidden holdouts, and refresh prompts periodically as models and user behavior change.

## 7 Model and Framework Selection

### 7.1 Capability, cost, and domain considerations
Frontier models (GPT-4-class and Claude-class models) are reported to achieve the strongest agreement with humans on many preference tasks, but they still exhibit systematic biases and cost constraints. Open-source judges such as Prometheus 2 are cited as approaching frontier performance (including reported correlations comparable to GPT-4 in some evaluations), enabling local deployment and reducing API costs.

Agreement can drop substantially in specialized domains (medicine, law, low-resource languages), motivating hybrid human-machine workflows for expert evaluation and safety-critical settings.

### 7.2 Framework selection guide

The sources explicitly enumerate several rubric-based judge frameworks and their suggested use cases:

| Framework    | Best use case            | Key innovation (as described)                                                   |
| ------------ | ------------------------ | ------------------------------------------------------------------------------- |
| G-Eval       | General purpose          | Stepwise, rubric-driven judging; probabilistic scoring via token probabilities. |
| CheckEval    | Factuality / constraints | Decomposition into granular Boolean sub-questions.                              |
| RAGAS        | RAG systems              | Specialized rubric dimensions (faithfulness, context recall, answer relevance). |
| Gecko        | Multimodal (images)      | VQA-as-judge decomposition into text Q&A checks (object presence).              |
| Prometheus 2 | Open-source / local      | Fine-tuned model specialized to follow rubrics as a judge.                      |

Recent work also emphasizes evaluating more than the final output in agentic settings (Agent-as-a-Judge), and using multi-agent protocols (ChatEval, CourtEval) for nuanced tradeoffs.

## 8 Implementation, Tooling, and Reporting

### 8.1 Human-in-the-loop rubric authoring and stakeholder deliberation
Human-centered tooling (EvalAssist) emphasizes that stakeholders vary in rubric specification-some overfit criteria to single examples while others leave criteria too vague-so best practice is explicit deliberation and iterative refinement with exemplars (Ashktorab et al., 2025). Shareable metrics guides and checklist practices can surface evaluator blind spots.

### 8.2 Store provenance and evaluation artifacts
For traceability and audits, best practice is to persist rubric versions, prompts, judge settings, raw outputs, scores, rationales, and calibration data in a queryable store (including graph database designs). This supports error analysis, reproducibility, and defensible comparisons across time.

### 8.3 Report sufficient detail for reproducibility
Recommended reporting items include: rubric text (with anchors/examples), exact judge prompt(s), judge model name/version, sampling settings, aggregation rule, evaluation dataset specification, and human baselines (including human-human agreement). Without these details, rubric-based scores are difficult to interpret or compare across runs.

## 9 Limitations and Outlook
Across sources, a consistent conclusion is that LLM judges can reach human-like agreement on well-designed tasks, but performance is bounded by systematic biases, domain expertise limits, and vulnerability to adversarial inputs-particularly for safety evaluation. The forward path emphasized is not replacing humans but augmenting them: use LLM judges for scalable approximate evaluation while reserving human expertise for calibration, edge cases, expert domains, and high-stakes decisions.

## 10 Conclusion
The consolidated best practices recommend:
* Define a clear evaluation contract (decision, artifact, context) and deliberately choose pointwise vs pairwise protocols.
* Use analytic, task-specific rubrics with atomic criteria, hierarchical gating, and explicit negative constraints.
* Prefer low-precision anchored scales; include "cannot assess" where evidence is insufficient; provide graded exemplars (including negative examples).
* Implement judge prompts as structured form-filling; score criteria separately; document prompts and decoding settings.
* Use CoT and critique cycles appropriately; adopt repeated sampling/distributional reasoning and probabilistic scoring where needed.
* Engineer bias controls (order swapping, length control, cross-family judging) and stress-test with adversarial inputs.
* Validate with agreement and distribution-aware metrics; calibrate against human-labeled sets; monitor drift and maintain defended benchmarks.
* Support implementation with human-centered rubric tooling, provenance storage, and reproducible reporting.

---

## Table 1: Compact best-practice checklist

| Category                 | Practice                                        | Specifics / procedure                                                                                    | Metrics or evidence                                                                         | Key sources                                                           |
| ------------------------ | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Rubric design            | Link criteria to construct (ECD)                | Map each rubric criterion to the target construct with SME review and evidence-centered-design artifacts | Requires validity evidence; supports interpretability and construct alignment               | Casabianca et al. (2025)                                              |
| Rubric design            | Provide annotated exemplars                     | Supply multiple scored examples with annotations/chain-of-thought to train/calibrate judges              | Reduces rater error and improves calibration (MAE/QWK improvements reported)                | Casabianca et al. (2025); Johnson and Straub (2024)                   |
| Rubric design            | Use instance-specific checklists                | Derive per-item checklists (check/uncheck) for complex tasks instead of single Likert totals             | Improves pairwise preference accuracy in RaR experiments (20.4% to 35.9%)              | Gunjal et al. (2025)                                                  |
| Rubric design            | Ground rubrics in references/SME answers        | Create rubrics or criteria using reference answers and SME input rather than entirely synthetic rules    | Reference-grounded rubrics yield higher scoring accuracy                                    | Gunjal et al. (2025)                                                  |
| Judging protocols        | Document prompts, sampling & decoding           | Record exact prompt text, model settings, and any few-shot examples for reproducibility                  | Prompt/config choices materially affect scores                                              | Casabianca et al. (2025)                                              |
| Judging protocols        | Prefer pairwise for fine discriminations        | Use pairwise comparisons when subtle quality differences are required                                    | Pairwise often yields more reliable rankings than pointwise in practice                     | MT-Bench / Chatbot Arena (arXiv:2306.05685)                           |
| Judging protocols        | Explore non-deterministic sampling              | Aggregate multiple samples per response to reduce evaluation noise                                       | Non-deterministic sampling can improve alignment with humans vs single deterministic decode | He et al. (2025)                                                      |
| Judging protocols        | Use CoT sparingly & when needed                 | Enable stepwise reasoning only when rubric criteria require it                                           | CoT can add noise if misapplied                                                             | G-Eval (arXiv:2303.16634)                                             |
| Judging protocols        | Add critique / self-review cycles               | Instruct judges to critique and revise initial scores                                                    | Reduces MAE and improves QWK versus single-pass scoring in reported study                   | Johnson and Straub (2024)                                             |
| Reliability & validity   | Plan double-scoring and rater calibration       | Predefine double-scored items, run rater training/calibration, and certify raters                        | Use ICC, Krippendorff's alpha, QWK to quantify agreement                                        | Casabianca et al. (2025)                                              |
| Reliability & validity   | Report distribution-aware comparisons           | Compare full score distributions between humans and judges using metrics like Earth Mover's Distance     | Reveals systematic shifts missed by point estimates                                         | He et al. (2025)                                                      |
| Reliability & validity   | Monitor human-LLM concordance over time         | Track correlations, MAE, QWK after model/rubric updates                                                  | Flags drift and regression                                                                  | Casabianca et al. (2025)                                              |
| Bias mitigation          | Detect & mitigate position/order bias           | Randomize or swap ordering in pairwise tasks and measure positional effects                              | Swap tests detect and mitigate position bias                                                | "Large Language Models are not Fair Evaluators" (arXiv:2305.17926)    |
| Bias mitigation          | Control length/verbosity effects                | Normalize for response length or include length as an explicit rubric dimension                          | Length correlates with inflated scores if unchecked                                         | AlpacaEval (as cited)                                                 |
| Bias mitigation          | Address teacher-preference & self-bias          | Use ensembles, external judges, or debiasing modules; avoid same-family judge                            | Reduces favoritism toward training-source answers                                           | He et al. (2025); EvalAssist (Ashktorab et al., 2025)                 |
| Bias mitigation          | Run fairness analyses across subgroups          | Evaluate SMD and QWK by subgroup; use EB-SMD; run DIF-style analyses                                     | Recommended fairness/validity evidence                                                      | Casabianca et al. (2025)                                              |
| Implementation & tooling | Use multi-LLM ensembles / panels                | Combine independent judges and/or meta-judges                                                            | Improves robustness vs single-LLM verdicts                                                  | "Replacing Judges with Juries" (as cited in EvalAssist)               |
| Implementation & tooling | Store evaluation artifacts & provenance         | Persist prompts, rubric versions, outputs, scores, and rationales in a queryable store                   | Enables audits and error analysis                                                           | EvalAssist (Ashktorab et al., 2025); REGAI (Johnson and Straub, 2024) |
| Implementation & tooling | Human-in-the-loop rubric authoring & SME review | Interactive workflows for stakeholders to co-design criteria and review annotations                      | Avoids under/over-specific rubrics and blind spots                                          | EvalAssist (Ashktorab et al., 2025)                                   |
| Implementation & tooling | Document model selection & standards alignment  | Record rationale for chosen LLMs and align validation to testing standards                               | Strengthens validity arguments for scores                                                   | Casabianca et al. (2025)                                              |

---

## References

Ashktorab, Zahra, Elizabeth M. Daly, Erik Miehling, Werner Geyer, Martin Santillan Cooper, Tejaswini Pedapati, Michael Desmond, Qian Pan, and Hyo Jin Do. 2025. *EvalAssist: A Human-Centered Tool for LLM-as-a-Judge.* arXiv:2507.02186.

Casabianca, J., Daniel F. McCaffrey, Matthew S. Johnson, Naim Alper, and Vladimir Zubenko. 2025. *Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications.* arXiv:2501.02334.

Databricks Blog. *Best Practices for LLM Evaluation (RAG).*

Dietz, Laura. 2025. *Principles and Guidelines for the Use of LLM Judges.*

FActScore. *Atomic fact decomposition for factual precision.*

FLASK. *Fine-grained skill taxonomy for evaluation.*

G-Eval. 2023. *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* arXiv:2303.16634.

Gecko. *VQA-as-Judge decomposition for multimodal evaluation.*

Gunjal, Anisha, Anthony Wang, Elaine Lau, Vaskar Nath, Bing Liu, and Sean M. Hendryx. 2025. *Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains.* arXiv:2507.17746.

Gu et al. 2024. *Comprehensive survey on LLM-as-a-judge (taxonomy and gaps).*

He, Junda, Jieke Shi, Terry Yue Zhuo, Christoph Treude, Jiamou Sun, Zhenchang Xing, Xiaoning Du, and David Lo. 2025. *LLM-as-a-Judge for Software Engineering: Literature Review, Vision, and the Road Ahead.* arXiv:2510.24367.

Johnson, Zach, and Jeremy Straub. 2024. *Development of REGAI: Rubric Enabled Generative Artificial Intelligence.* arXiv:2408.02811.

JudgeLM. *Training-time augmentation to address judge biases.*

Judge's Verdict Benchmark. *Correlation test and human-likeness z-score test for judges.*

Li et al. 2024. *Comprehensive survey on LLM-as-a-judge (taxonomy and gaps).*

LLM-RUBRIC. *Distribution over rubric responses and calibration/aggregation.*

Large Language Models are not Fair Evaluators. 2023. arXiv:2305.17926.

Microsoft Learn. *Evaluating the performance of LLM summarization prompts with G-Eval.*

MT-Bench and Chatbot Arena. 2023. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* arXiv:2306.05685.

Prometheus 2. *Fine-tuned open-source judge model.*

RAGAS. *RAG metrics (faithfulness, context recall, answer relevance).*

Reasoning-based Bias Detector (RBD). *Reasoning-guided judge self-correction.*

Replacing Judges with Juries. *Ensemble/panel judging to diversify judgments.*

Rubric Is All You Need. 2025. *Rubric Is All You Need: Enhancing LLM-based Code Evaluation With Question-Specific Rubrics.* arXiv:2503.23989.

Sycophancy bias in LLM judges. *High false-positive acceptance of invalid outputs.*

Through the Judge's Eyes. 2025. *Through the Judge's Eyes: Inferred Thinking Traces Improve Reliability of LLM Raters.* arXiv:2510.25860.

Tripathi et al. 2025. *Pointwise vs. pairwise robustness under adversarial manipulation.*

Unsupervised Debiasing Alignment (UDA). *Dynamic scoring adjustment to reduce variance.*

Zheng et al. 2023. *MT-Bench pairwise judging and agreement results.*

Zhuge et al. 2024. *Agent-as-a-Judge for evaluating agentic systems/processes.*

ChatEval. 2023. *ChatEval: Towards Better LLM-based Evaluators through multi-agent debate.* arXiv:2308.07201.

CourtEval. *Debate/judicial-proceedings style multi-agent judge.*

MAJ-EVAL. *Multi-agent judging protocol.*

CodeJudgeBench. 2025. *Coding judgment benchmark; order effects.*

MLLM-as-a-Judge. 2024. *Multimodal judging; GPT-4V agreement results.*

R-Judge. *Safety evaluation scenarios and adversarial cases.*
