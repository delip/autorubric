# Data Sourcing Brainstorm for AutorubricLM

**Status:** Brainstorm (not a spec update)
**Date:** 2026-04-03

---

## 1. Datasets Already in the Codebase

| Dataset | Domain | Size | Criterion Types | Ground Truth |
|---------|--------|------|-----------------|--------------|
| RiceChem (Sonkar et al., 2024) | Chemistry education | 1,240 responses, 27 criteria, 4 questions | Binary | Human-graded (TA annotations) |
| CHARM-100 | Chatbot evaluation | 100 items | Heterogeneous (binary + ordinal + nominal) | Synthetic ground truth |
| ResearcherBench (Xu et al., 2025) | AI research | 65 questions, 931 criteria (6–21 per question, mean 14.3), 34 AI subjects | Binary (per-item, weighted 1–3) | Expert-curated |
| Hashemi et al. 2024 (LLM-Rubric) | Dialogue/QA | 3.2 MB, multi-dimensional Likert | Ordinal (Likert) | Human-annotated |
| Sharma et al. 2025 (ResearchRubrics) | Deep research | 986 KB | Mixed | Research rubrics |
| Essay grading | Humanities (Industrial Revolution) | ~11 items, 5 criteria | Binary | Human scores |
| Peer review skill eval | Scientific peer review | 10 papers × 3 conditions | Mixed | Skill evaluation |

---

## 2. Exhaustive Survey of External Rubric Datasets

Compiled from a systematic survey of arxiv papers on LLM-as-judge with rubrics, RL with rubric rewards, rubric generation, and educational assessment (2023–2026). Organized by tier relevance to AutorubricLM training.

### 2.1 Tier 1 — Existing Rubrics Convertible to Criterion Schema

These datasets contain structured evaluation criteria created by humans or well-validated benchmarks. Primary use: convert to `list[Criterion]` format for SFT training data.

| Dataset | Paper / Source | arxiv ID | Year | Size | Domain | Criteria Structure | Availability |
|---------|---------------|----------|------|------|--------|-------------------|--------------|
| **Prometheus Feedback Collection** | Prometheus (Kim et al., 2023) | 2310.08491 | 2023 | 1K score rubrics, 20K instructions, 100K response+feedback instances | General instruction-following | Per-instance score rubric with criterion description + 5 score-band descriptions (1–5); each example includes instruction, reference answer, response, feedback rationale, and score. Uniform score distribution (20K per tier) | [GitHub](https://github.com/prometheus-eval/prometheus) |
| **Prometheus 2 Preference Collection** | Prometheus 2 (Kim et al., 2024) | 2405.01535 | 2024 | 1K evaluation criteria, 20K instructions, 200K pairwise instances with verbal feedback | General instruction-following | Pairwise ranking data from 5 responses per instruction; balanced A/B labels; verbal comparative feedback per pair. Shares rubrics with Feedback Collection | [GitHub](https://github.com/prometheus-eval/prometheus) |
| **BiGGen-Bench** | Kim et al. (2025, NAACL Best Paper) | 2406.05761 | 2025 | 77 tasks, 765 instances, 9 core capabilities | Instruction following, grounding, planning, reasoning, refinement, safety, theory of mind, tool usage, multilingualism | Instance-specific fine-grained evaluation criteria per example; 5-point rubrics with behavioral anchors. Evaluated on 103 frontier models by 5 evaluator LMs | [HuggingFace](https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench) |
| **FLASK** | Ye et al. (2023, ICLR 2024 Spotlight) | 2307.10928 | 2023 | 12 fine-grained skills across 4 primary abilities | General LLM evaluation | Skill-based decomposition: Logical Thinking (3 sub-skills), Background Knowledge (2 sub-skills), Problem Handling (multiple). Task-agnostic, instance-wise skill set metrics | [GitHub](https://github.com/kaistAI/FLASK) |
| **Auto-J** | Li et al. (2023) | — | 2023 | 332 hand-crafted evaluation criteria, 58 real-world scenarios | Diverse real-world tasks | Task-matched criteria sets supporting both pairwise comparison and single-response evaluation with natural language critiques | [GitHub](https://github.com/GAIR-NLP/auto-j) |
| **HelpSteer2** | Wang et al. (2024, NeurIPS D&B) | 2406.08673 | 2024 | 10K response pairs, 5 dimensions | General helpfulness | 5-point Likert (0–4) on: helpfulness, correctness, coherence, complexity, verbosity. Multi-turn English conversations. Nemotron-4-340B-Reward outputs float per dimension | [HuggingFace](https://huggingface.co/datasets/nvidia/HelpSteer2) (CC-BY-4.0) |
| **SedarEval** | Fan et al. (2025) | 2501.15595 | 2025 | 1,000 questions, 8 major categories | Long-tail knowledge, mathematics, coding, logical reasoning | Self-adaptive per-question rubrics with scoring points, penalty points, and background knowledge. Primary/secondary criteria with scoring trajectories. Discriminative filtering (retain high-variance questions) | Paper release |
| **ResearchRubrics (Scale AI)** | Sharma et al. (2025) | 2511.07685 | 2025 | ~101 tasks, 2,593 expert-written criteria (~26 rubrics/task), 9 domains, 2,800+ hours human labor | Business planning, historical analysis, technical documentation, consumer questions, etc. | Expert-written fine-grained rubrics with positive and negative criteria; mandatory vs. optional; ternary grading (full/partial/no credit). Factual grounding, reasoning coherence, completeness, relevance, clarity, citation quality | [HuggingFace](https://huggingface.co/ScaleAI/researchrubrics) (auth required) |
| **RubricBench** | Zhang et al. (2026) | 2603.01562 | 2026 | 1,147 pairwise comparisons | General chat (36.5%), coding (23.9%), STEM reasoning (23.8%), instruction following (8.8%), safety (7.0%) | Human-annotated atomic rubrics: 2–10 binary Yes/No checks per instance (majority 4–6). Criteria mapped to: Reasoning, Content, Expression, Alignment, Safety. Key finding: 27% accuracy gap between model-generated and human rubrics | [HuggingFace](https://huggingface.co/datasets/DonJoey/rubricbench) |
| **AdvancedIF** | He et al. (Meta, 2025) | 2511.10507 | 2025 | 1,645 prompts: 402 complex IF + 736 multi-turn + 507 system-prompt steerability | Instruction following | Per-prompt expert-written rubrics averaging 6–10 criteria (up to 20). Binary compliance via o3-mini verifier. Covers format, style, structure, length, negative constraints, spelling, inter-conditional constraints | Paper release |
| **HealthBench** | Used in multiple papers (2025) | — | 2025 | 5,000 dialogue instances; 1,000-question Hard subset | Medical dialogue / healthcare | Task-specific rubrics annotated by physicians. Per-query rubric lists evaluated criterion-by-criterion. Used as seed data for RaR, InfiMed-ORBIT, Self-Rewarding RL, RubricRAG | Benchmark release |
| **WildBench** | Lin et al. (2024) | 2406.04770 | 2024 | 1,024 tasks, 5–10 checklist items each | Real user queries | YES/NO task-specific checklists finalized from GPT-4-Turbo + Claude-3-Opus. WB-Reward (pairwise, 5 outcomes) + WB-Score (overall). 0.98 Pearson with Chatbot Arena | [GitHub](https://github.com/allenai/WildBench) |
| **TIGERScore** | Jiang et al. (2023, TMLR 2024) | 2310.00752 | 2023 | 42K quadruples across 23 datasets, 6 text generation tasks | Text generation (summarization, translation, data-to-text, etc.) | Error-based with penalty scores [−5, −0.5]: error location, aspect, explanation, penalty per error. **Only major dataset with explicit negative-weight criteria** | [GitHub](https://github.com/TIGER-AI-Lab/TIGERScore) |
| **DREsS** | (2024) | 2402.16733 | 2024 | 48.9K samples (2.3K real classroom + synthetic augmentation) | EFL essay scoring | Multi-level rubric-based scoring for writing proficiency. Real undergraduate essays + standardized rubrics | Public |
| **ExpertLongBench** | (2026) | 2506.01241 | 2026 | 1,050 samples, 11 tasks, 9 domains | Expert-level long-form: law, material science, education, healthcare, chemistry, biology, medicine, finance, cybersecurity | CLEAR checklist-based evaluation framework. Expert-validated rubrics. Gemini-2.5-Pro achieves only 33.4 F1 | [HuggingFace](https://huggingface.co/datasets/launch/ExpertLongBench) (CC-BY-NC-SA) |
| **DRACO (Perplexity)** | Perplexity AI (2026) | 2602.11685 | 2026 | ~40 criteria per task, multiple professional domains | Deep research | 4 dimensions: factual accuracy, breadth/depth of analysis, presentation quality, citation quality | [HuggingFace](https://huggingface.co/datasets/perplexity-ai/draco) |
| **DeepResearch Bench II** | (2026) | — | 2026 | 132 tasks, 22 domains, **9,430 fine-grained binary rubrics** | Expert-level research | Expert-written articles decomposed into hierarchical rubrics covering information recall, analysis, presentation quality | [GitHub](https://github.com/imlrz/DeepResearch-Bench-II) |
| **ASAP / ASAP 2.0** | Automated Student Assessment Prize | — | 2012+ | 10K+ essays, multiple prompts | Essay scoring | Holistic + analytic rubrics (organization, style, conventions, etc.) | [Kaggle](https://www.kaggle.com/c/asap-aes) |
| **SummEval** | Fabbri et al. (2020) | 2007.12626 | 2020 | CNN/DailyMail model outputs + annotations | Summarization | 14+ evaluation metrics unified: coherence, consistency, fluency, relevance | [GitHub](https://github.com/Yale-LILY/SummEval) |
| **MQM Translation Gold** | MQM Framework | — | — | Thousands of annotated translation units, 11 language pairs (bio-medical) | Translation quality | Severity levels (Minor/Major/Critical) × error categories. Structured error taxonomy | [HuggingFace](https://huggingface.co/datasets/alconost/mqm-translation-gold), [Google WMT-MQM](https://github.com/google/wmt-mqm-human-evaluation) |
| **LLM-Rubric (Microsoft)** | Hashemi et al. (2024, ACL 2024) | 2501.00274 | 2024 | Multi-dimensional dialogue evaluation | Dialogue systems | 9 rubric questions (naturalness, conciseness, citation quality) + calibration neural network. Judge-specific and judge-independent parameters. RMS error < 0.5 (2× improvement) | [GitHub](https://github.com/microsoft/LLM-Rubric) |
| **OneMillion-Bench** | — | — | — | 400 expert-level entries, 5 professional domains (bilingual) | Professional (Global/Chinese) | Weighted rubric-based binary criteria in single-call evaluation | [HuggingFace](https://huggingface.co/datasets/humanlaya-data-lab/OneMillion-Bench) |
| **ProImage-Bench** | (2025) | 2512.12220 | 2025 | 654 tasks, 6,076 criteria, 44,131 binary checks | Professional image generation (biology, engineering) | Binary YES/NO conditions on task + domain dimensions | Paper release |
| **MOPRD** | (2022) | 2212.04972 | 2022 | Multidisciplinary papers + reviews | Peer review | Paper metadata, manuscripts, review comments, meta-reviews, rebuttals, editorial decisions | Public |
| **ORB (Open Review-Based)** | — | — | — | 36K+ papers, 89K+ reviews | Academic peer review | Structured review dimensions | Public |
| **Creative-Rubrics** | — | — | — | Creative responses from GPT-4.5, o3-mini, DeepSeek-R1 | Creative writing (movie reviews, short stories) | Multiple compliance levels (high/mid/low score) | [HuggingFace](https://huggingface.co/datasets/vicgalle/creative-rubrics) |

### 2.2 Tier 2 — Synthetic / Semi-Synthetic Rubric Collections

Large-scale rubric datasets generated by LLMs or hybrid methods. Primary use: direct SFT training data, or as scaffold for re-generation with meta-rubric quality filtering.

| Dataset | Paper / Source | arxiv ID | Year | Size | Domain | Generation Method | Rubric Structure | Availability |
|---------|---------------|----------|------|------|--------|-------------------|------------------|--------------|
| **RubricHub** | RubricHub (2026) | 2601.08430 | 2026 | **~110K rubrics** | Writing, medical, code, math, science | Coarse-to-fine: principle-guided synthesis → multi-model aggregation (GPT-5.1, Gemini 3 Pro) → difficulty evolution (excellent vs. exceptional). Post-trained via RuFT + RuRL | 30+ fine-grained criteria per query in complex domains. Qwen3-14B post-trained achieves 69.3 on HealthBench (SOTA, surpassing GPT-5) | [HuggingFace](https://huggingface.co/datasets/sojuL/RubricHub_v1) |
| **OpenRubrics** | Liu et al. (2025) | 2510.07743 | 2025 | **~35.6K rubric entries** | Multi-domain: instruction following, reasoning, general helpfulness. Source corpora: UltraFeedback, Tulu 2.5, HelpSteer3, Skywork, MegaScience, Medical-o1 | Contrastive Rubric Generation from preference triplets (prompt, chosen, rejected). Consistency-based filtering: re-query to predict preference, retain only when prediction matches human label | Two-part format: explicit Hard Rules (prompt-specified requirements) + higher-level Principles (qualitative: reasoning, factuality, style). Diverse via t-SNE on Qwen-3-Embedding | [HuggingFace](https://huggingface.co/datasets/OpenRubrics/OpenRubrics) |
| **WildChecklists** | Viswanathan et al. (2025) | 2507.18624 | 2025 | **130,000 instructions** with paired checklists | General instruction-following / non-verifiable tasks | Candidate-based two-stage: generate responses of varying quality, enumerate failure modes, assign importance weights. Desiderata: comprehensiveness, naturalness, objectiveness, atomicity | Dynamic YES/NO checklist sequences with importance weights. Universal anti-hacking requirement appended | Paper release (planned) |
| **Rubicon Rubric Bank** | Huang et al. (2025) | 2508.12790 | 2025 | **10,000+ rubrics**, used with ~5K training samples | Open-ended, humanities-centric, social, creative tasks | Human experts + LLMs + iterative human-LLM hybrids. Three scopes: dataset-grounded, task-level, instance-specific. Two-stage RL: stage 1 = static multi-dimensional rubrics; stage 2 = instance-specific from stronger agentic workflows | Multi-dimensional rubrics; rubric diversity, granularity, and quantity critically affect RL success. Single rubrics invite exploitation | Paper release |
| **RaR-Medicine & RaR-Science** | Gunjal et al. (2025) | 2507.17746 | 2025 | **~20K prompts each** (~40K total) | Medicine; Science (aligned with GPQA-Diamond) | LLM-generated (o3-mini, GPT-4o) conditioned on reference answers. Instance-specific rubric per prompt | 7–20 weighted binary checklist items per prompt. Weights: numeric + categorical (Essential, Important, Optional, Pitfall). Reward via explicit weighted-sum or implicit LLM-judge holistic scoring | Public release |
| **DR Tulu Evolving Rubrics** | Shao et al. (2025) | 2511.19399 | 2025 | ~9K RL prompts (~5K SearchArena/OpenScholar + ~4K RaR); 16K SFT trajectories | Deep research: healthcare, scientific literature, general-domain | Persistent rubrics initialized via search; evolving rubrics from analyzing rollouts. Variance-based filtering retains only most discriminative rubrics. Buffer management: remove zero-variance, keep top-Kmax by std dev | Per-prompt weighted rubric sets with positive and negative rubrics. Judge outputs 0/0.5/1 per rubric. Separate persistent + evolving buffers | Pipeline described |
| **QuRL** | Wei et al. | — | — | 1,200 questions (800 train / 400 test) | 10 popular topical domains / open-ended QA | Web-sourced: retrieve webpages, distill into meta-descriptions, construct case-wise rubrics. Four design principles: Content Focus, Writing Quality, Case-Wise Specificity, Meta-Description Referencing. Discriminative power filtering | Case-wise rubrics with point-based criteria; include illustrative good/bad examples per criterion | Paper release |
| **ResearchPlanGen-ML** | Goel et al. (2025) | 2512.23707 | 2025 | 6,872 ML research goals (NeurIPS 2023–2024, ICLR 2025) | ML research planning | Automated: sample selector scores rubrics by diversity + quality, filters to top-10 items per goal. Grader checks against 7 general guidelines; score = fraction of satisfied rubric items | Goal-specific rubrics; binary per-item satisfaction. Length-limited plans with unlimited thinking tokens | Pipeline described |
| **Auto-Rubric (AgentScope)** | Xie et al. (2025) | 2510.17314 | 2025 | Hierarchical rubric sets from HelpSteer3-Preference + UltraFeedback-Binarized | General preference evaluation | Two-stage: Propose-Evaluate-Revise loop + information-theoretic generalization. 70 preference pairs → RewardBench2 SOTA (80.91% with Qwen3-8B) | Hierarchical "Theme-Tips" format: themes (e.g., "Factual Accuracy and Canonical Consistency") with multiple concrete tips per theme | [HuggingFace](https://huggingface.co/datasets/agentscope-ai/Auto-Rubric) |
| **R3 Datasets (D14k, D4k)** | Anugraha et al. (2025) | 2505.13388 | 2025 | 14K + 4K quality-filtered samples | Rubric-agnostic reward modeling | Quality-filtered from larger preference corpus. LoRA-compatible, training-efficient | Rubric-agnostic with interpretable reasoning. R3-Qwen3-14B-LoRA surpasses Nemotron-49B | [GitHub](https://github.com/rubricreward/r3) |
| **Rubrik's CUBE** | (2025) | 2503.23899 | 2025 | 26K explanations (human + 6 LLM-generated) | Reasoning and language explanation quality | Education-inspired rubric for evaluating explanation quality | Quality-annotated by both humans and LLMs | Paper release |
| **RubricEval** | Pan et al. (2026) | 2603.25133 | 2026 | 3,486 instances (Easy/Hard subsets) | Instruction following | Quality-controlled meta-evaluation benchmark. Rubric types: Form, Topic Scope, Quality Requirements, Task Completion | Human-labeled consensus rubric-level judgments. Key finding: even GPT-4o achieves only 55.97% on Hard subset | Paper release |

### 2.3 Tier 3 — Closed-Loop / RL-Validated Rubric Systems

These papers describe methods where rubrics are validated via downstream task performance (RL reward, ground-truth correlation, or preference agreement). Directly relevant to AutorubricLM's Tier 3 (DPO via evaluation pipeline). Some release data; others describe replicable pipelines.

| Paper | arxiv ID | Year | RL Algorithm | Approach | Reward Dimensions / Rubric Structure | Scale | Data Released? |
|-------|----------|------|-------------|----------|--------------------------------------|-------|----------------|
| **Rubrics as Rewards (RaR)** | 2507.17746 | 2025 | GRPO | On-policy RL with instance-specific weighted binary checklists as reward. Explicit aggregation (weighted sum) or implicit (LLM-judge holistic). +31% on HealthBench, +7% on GPQA-Diamond | 7–20 weighted binary items per prompt; categorical weights (Essential/Important/Optional/Pitfall) | ~40K prompts (medicine + science) | Yes (RaR-Medicine, RaR-Science) |
| **Rubicon** | 2508.12790 | 2025 | Two-stage RL | Stage 1: static multi-dimensional rubrics for constraint handling. Stage 2: instance-specific rubrics from stronger agentic workflows for open-ended tasks. Rubric diversity/granularity critical | Multi-dimensional, three scopes (dataset-grounded, task-level, instance-specific) | 10K+ rubrics, ~5K training samples | Rubric bank described |
| **DR Tulu (RLER)** | 2511.19399 | 2025 | GRPO | Evolving rubric buffers: persistent (search-grounded) + evolving (from rollout analysis). Variance-based filtering retains discriminative rubrics. Positive + negative rubrics | Per-prompt weighted rubric sets; judge scores 0/0.5/1 per rubric; separate persistent + evolving buffers | ~9K RL prompts + 16K SFT trajectories | Pipeline described |
| **RIFL / AdvancedIF** | 2511.10507 | 2025 | RLHF-style | Rubric generator → finetuned rubric verifier → reward shaping. Binary per-criterion compliance. +6.7% absolute on AdvancedIF | Expert-written rubrics, 6–20 criteria per prompt covering format/style/structure/length/negative constraints | 1,645 prompts | Benchmark + pipeline |
| **WildChecklists (RLCF)** | 2507.18624 | 2025 | RL (RLCF) | Reinforcement Learning from Checklist Feedback. Dynamic checklists with importance weights. Candidate-based extraction outperforms direct prompting. Universal anti-hacking requirement | YES/NO checklists; atomicity + objectiveness enforced | 130K instructions | Yes (WildChecklists) |
| **Rubric-ARM** | 2602.01511 | 2026 | Alternating GRPO | Jointly optimizes rubric generator + judge as generalized EM with rubrics as latent variables. Stage I: SFT warmup on synthetic rubric/judge trajectories. Stage II: alternating RL updates. +4.7% avg on reward-modeling benchmarks | Prompt-conditioned structured criteria (factual correctness, tone, presentation). Shaped reward: Rj = Racc + Rfmt | Pairwise preference datasets (UltraFeedback, SkyWork, Magpie, Synthetic IF) | [HuggingFace](https://huggingface.co/OpenRubrics/rubricarm) |
| **ARL with Contextual Rubric Rewards** | 2603.15646 | 2026 | Alternating GRPO | Alternates optimization across rubric meta-classes; variance contraction effect. Eliminates fixed scalarization | Meta-classes: accuracy, completeness, instruction following, context awareness, communication quality | 1.7B–14B model scales on HealthBench | Pipeline described |
| **RRD: Rethinking Rubric Generation** | 2602.05125 | 2026 | Used as reward signal (RFT) | Recursive decompose-filter cycle: decomposes rubrics that apply to too many rollouts into finer-grained criteria; misalignment filter removes rubrics that prefer weaker models. +17.7% on JudgeBench | Fine-grained decomposed criteria; upper bound on misclassification via (wᵀμ)²/(wᵀΣw) optimization | +160% reward boost for Qwen3-4B, +60% for Llama3.1-8B | Pipeline described |
| **R-GRPO** | 2511.12344 | 2025 | GRPO | Expert LLM (O3) generates reference answer → rubrics. Rubrics divided into Factual Criteria + Process Criteria. Binary verification per criterion, weighted aggregation into normalized scalar reward. Failed criteria trigger off-policy refinements | Factual Criteria (intermediate steps, sub-answers, final results) + Process Criteria (reasoning steps, logic). Adaptive weights | — | Pipeline described |
| **Self-Rewarding Rubric-Based RL** | 2509.25534 | 2025 | GRPO | Generative reward model (static Qwen3-32B) judges rubric satisfaction on HealthBench. Self-rewarding: same model family scores own outputs | HealthBench task-specific rubrics; physician-annotated Hard subset | 5K dialogue instances; 4K easy SFT + 1K scoring data | Pipeline described |
| **Training AI Co-Scientists** | 2512.23707 | 2025 | GRPO | Self-rewarding RL for research plan generation. Sample selector filters to top-10 rubric items per goal by diversity + quality. Score = fraction of satisfied items. Length-limited with unlimited thinking | Goal-specific rubrics × 7 general guidelines. Binary per-item satisfaction | 6,872 ML research goals | Pipeline described |
| **InfiMed-ORBIT** | 2510.15859 | 2025 | GRPO | RAG-based dynamic rubric generation from HealthBench-derived seed data. Retrieves exemplar rubrics, generates multi-dimensional query-specific rubrics. Two-stage filtering: intermediate difficulty + high standards | Dynamic, multi-dimensional, query-specific rubrics from RAG. External judge scores per rubric | HealthBench-derived | Pipeline described |
| **Open Rubric System (OpenRS)** | 2602.14069 | 2026 | Asymmetric GRPO | Hierarchical: static Meta Rubrics (general + domain) + per-pair Adaptive Rubrics from semantic differences. Evolutionary GA-style beam search for rubric refinement. Pointwise Verifiable Rubrics as hard gates | Weighted criteria with comparative scores vₖ ∈ {−2,−1,0,1,2}. Preference = weighted average. Actions: ADD/DELETE/MODIFY | — | Pipeline described |
| **Chasing the Tail** | 2509.21500 | 2025 | — | Dynamic rubrics for tail-case discrimination. Off-policy examples + LLM-generated grading rubrics per prompt. Addresses reward over-optimization | Per-prompt rubrics distinguishing excellent from merely great | — | Pipeline described |
| **QuRL** | — | — | GRPO | Web-sourced rubrics for open-ended QA. Mine internet text → meta-descriptions → case-wise rubrics. Discriminative power filtering | Case-wise rubrics with point-based criteria; content + writing quality dimensions | 1,200 questions (800 train / 400 test) | Paper release |
| **RuCL** | 2602.21628 | 2026 | Curriculum RL | Stratified rubric-based curriculum learning for multimodal LLMs. Rubrics stratified by difficulty; curriculum scheduling based on model competence. +12.97% on WeMATH, +5.16% on MathVerse | Generalized, reusable rubrics from foundational perception to advanced deduction | Visual reasoning tasks | Pipeline described |
| **Learning Query-Specific Rubrics** | 2602.03619 | 2026 | GRPO | RL-trained query-specific rubric generator for DeepResearch reports. Hybrid reward: human preference supervision + LLM-based rubric evaluation. 16 expert annotators (master's+) do pairwise preferences | Weighted evaluation rubrics per research query; judged on usefulness, coherence, completeness, alignment | Diverse research queries + multi-domain human preference data | Pipeline described |
| **Fine-Grained RLHF** | 2306.01693 | 2023 | PPO | Dense per-sentence rewards across multiple error categories. Two dimensions: density (sentence-level) + multiple feedback types. Lowest toxicity + perplexity while maintaining diversity | Error categories: irrelevance, repetition, incoherence, factual incorrectness (sentence-level), incompleteness (response-level) | — | [Website](https://finegrainedrlhf.github.io/) |
| **ArmoRM** | 2406.12845 | 2024 | — | Multi-objective reward via MoE gating for context-dependent aggregation of dimensions. 8B params approaches Nemotron-4 340B. SOTA on RewardBench | Multiple human-interpretable objectives (honesty, verbosity, safety, etc.) | — | [HuggingFace model](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) |
| **Self-Rewarding LMs** | 2401.10020 | 2024 | Iterative DPO | LLM self-generates rewards via 5 additive criteria. Reward generation ability improves alongside instruction following over iterations | 5 criteria: relevance, coverage, usefulness, clarity, expertise | — | Pipeline described |
| **CARMO** | 2410.21545 | 2025 | — | Dynamic per-query criteria generation mitigating reward hacking. Distillable to smaller models. +2.1% on RewardBench zero-shot | Context-dependent (logical consistency, clarity, depth) | — | Pipeline described (ACL 2025) |
| **Health-SCORE** | 2601.18706 | 2026 | — | Adaptive rubric selection mechanism for health domain. Rubric quality as training signal | Health-specific criteria with adaptive prompt-level selection | — | Pipeline described |
| **AdaRubric** | 2603.21362 | 2026 | Used for DPO | Task-adaptive rubrics on-the-fly from task descriptions. DimensionAwareFilter prevents high-scoring dimensions from masking failures. 0.79 Pearson with human judgment; +6.8 to +8.5% for DPO | Task-specific criteria for agent evaluation (WebArena, ToolBench) | — | Pipeline described |
| **Configurable Preference Tuning** | 2506.11702 | 2025 | — | Rubric-guided synthetic preference data conditioned on system prompts. LLMs modulate outputs at inference without retraining | System-prompt-conditioned rubric criteria | — | Pipeline described |
| **RuscaRL** | 2508.16949 | 2025 | RL with scaffolding decay | Checklist-style rubrics as explicit exploration scaffolding. Different rubrics guide diverse responses; guidance decayed over time for internalization. Qwen-2.5-7B: 23.6→50.3 on HealthBench-500 (surpasses GPT-4.1) | Checklist-based per-task criteria with verifiable rewards | — | [GitHub](https://github.com/IANNXANG/RuscaRL) |

---

## 3. Key Papers on Rubric Generation

These papers specifically train or prompt models to *generate* rubrics — the closest prior work to AutorubricLM's core task.

| Paper | arxiv ID | Year | Key Idea | Training Data / Method | Relevance to AutorubricLM |
|-------|----------|------|----------|------------------------|---------------------------|
| **RubricHub** | 2601.08430 | 2026 | Coarse-to-fine pipeline; ~110K rubrics; post-training via RuFT + RuRL | GPT-5.1 + Gemini 3 Pro multi-model aggregation; difficulty evolution | Largest rubric corpus. Same generate-then-filter philosophy. Quality filtering via your meta-rubric would be the key value-add |
| **OpenRubrics** | 2510.07743 | 2025 | Contrastive Rubric Generation from preference pairs. Hard Rules + Principles format | UltraFeedback, Tulu 2.5, HelpSteer3, MegaScience, Medical-o1. Consistency filtering | Demonstrates that rubrics derived from seeing *what went wrong* in rejected responses are more discriminative |
| **Auto-Rubric** | 2510.17314 | 2025 | Propose-Evaluate-Revise + information-theoretic generalization. Theme-Tips hierarchy | 70 preference pairs → RewardBench2 SOTA (80.91%) | Shows that very few examples suffice when rubric extraction is principled. The hierarchical format is interesting for AutorubricLM's weight calibration |
| **RRD** | 2602.05125 | 2026 | Recursive decompose-filter; decomposes rubrics that apply to too many rollouts. Misalignment filter | General open-ended tasks; tested on JudgeBench, PPE, HealthBench, BiGGen-Bench | The decomposition principle maps to AutorubricLM's "discriminative power" quality property. Could be a post-processing step |
| **GER-Eval** | 2602.08672 | 2026 | Two-stage: LLMs generate rubric names, definitions, scoring scales, then apply them | Task descriptions → generated rubrics | Closest architectural match to AutorubricLM's generate-then-evaluate design |
| **RubricRAG** | 2603.20882 | 2026 | Retrieval-augmented rubric generation from similar queries | Existing rubric corpus as retrieval index | Finding: retrieval-based prompting more effective than SFT for rubric quality. Suggests RAG mode could complement trained AutorubricLM |
| **AdaRubric** | 2603.21362 | 2026 | Task-adaptive rubrics on-the-fly; DimensionAwareFilter | Agent benchmark tasks (WebArena, ToolBench) | The DimensionAwareFilter preventing high-scoring dimensions from masking failures parallels AutorubricLM's discriminative power concern |
| **WritingBench** | 2503.05244 | 2025 | Fine-tuned dedicated critic model for query-dependent evaluation dimensions | Writing benchmark tasks | Trains a *model* to generate evaluation dimensions — same paradigm as AutorubricLM but domain-specific |
| **Rubric-ARM** | 2602.01511 | 2026 | Jointly trains rubric generator + judge via alternating RL (generalized EM, rubrics as latent variables) | Pairwise preference datasets (UltraFeedback, SkyWork, Magpie, Synthetic IF) | Most sophisticated co-training of generator + evaluator. The EM framing is theoretically interesting for AutorubricLM's Tier 3 |
| **Open Rubric System** | 2602.14069 | 2026 | Evolutionary GA-style refinement of meta-rubrics; asymmetric GRPO | Two-level: static Meta Rubrics + per-pair Adaptive Rubrics | Hierarchical rubric architecture (general → domain → instance) is a design AutorubricLM could adopt |
| **AutoChecklist** | 2603.07019 | 2026 | Composable pipelines with 5 distinct checklist generation abstractions | General evaluation tasks | Framework-level contribution: composable generation pipelines |
| **iRULER** | 2602.12779 | 2026 | Recursive rubric refinement via meta-rubric ("rubric-of-rubrics"); 6 design principles: Specific, Scaffolded, Justified, Actionable, Qualified, Refinable | User-defined rubrics | Meta-rubric-of-rubrics parallels your standalone meta-rubric. The 6 design principles overlap with your quality model |
| **RULERS** | 2601.08654 | 2026 | Compiler-executor: NL rubrics → executable specifications; deterministic evidence verification; Wasserstein calibration | Essay + summarization rubrics | The "compile rubric to executable spec" idea could inform output validation |
| **OptimSyn** | 2604.00536 | 2026 | Rubric construction as learnable component driven by target-model feedback | Synthetic data generation | Interesting framing: rubric optimization as inner loop of data generation |
| **Automated Essay Rubric Refinement** | 2510.09030 | 2025 | Reflect-and-Revise: iterative refinement by reflecting on scoring rationales and human score discrepancies. QWK improvements up to 0.47 | ASAP-style essay scoring | Directly relevant to Tier 3: rubric refinement grounded in scoring agreement |
| **Learning Query-Specific Rubrics** | 2602.03619 | 2026 | RL-trained rubric generator with hybrid reward (human preference + LLM evaluation) | Research queries + expert pairwise preferences (16 annotators, master's+) | Training a rubric *generator* via RL preference signal — the exact AutorubricLM Tier 3 strategy |
| **SedarEval** | 2501.15595 | 2025 | Self-adaptive per-question rubrics with scoring points + penalty points + background knowledge | 1,000 questions, discriminative filtering | Penalty points concept maps to AutorubricLM's negative criteria. Scoring trajectories as instruction-following |
| **M-Prometheus** | — | 2025 | Multilingual extension of Prometheus with natively multilingual feedback data | Original 1K score rubrics applied cross-lingually | Multilingual rubric application — relevant if AutorubricLM targets non-English |

---

## 4. Coverage Gap Analysis

Mapping external datasets against the spec's Data Diversity Requirements:

| Diversity Axis | Current Codebase Coverage | Best External Sources to Fill Gaps | Gap Severity |
|---------------|--------------------------|-----------------------------------|--------------|
| **STEM** | RiceChem (chemistry only) | ExpertLongBench (9 domains), BiGGen-Bench (reasoning), R-GRPO (math reasoning) | Medium |
| **Writing** | Essay grading (~11 items) | DREsS (48.9K), ASAP (10K+), WritingBench, Creative-Rubrics | **High** |
| **Code** | None | RubricHub (code subset), RubricBench (23.9% coding), "Rubric Is All You Need" (CS courses) | **High** |
| **Creative** | None | Creative-Rubrics, WritingBench, Rubicon rubric bank (humanities/creative focus) | **High** |
| **Professional** | None | OneMillion-Bench (5 domains), ExpertLongBench (law, finance), DRACO | **High** |
| **Medical** | None | HealthBench (5K physician-annotated), RaR-Medicine (~20K), RubricHub medical subset, InfiMed-ORBIT, Health-SCORE | **High** |
| **Legal** | None | ExpertLongBench (law subset) | Medium |
| **Translation** | None | MQM Translation Gold (11 language pairs, structured error taxonomy) | Medium |
| **Deep Research** | ResearcherBench (65 questions) | ResearchRubrics (2,593 criteria), DRACO, DeepResearch Bench II (9,430 rubrics), DR Tulu, QuRL | Low |
| **Dialogue/Chat** | CHARM-100 (100, synthetic) | HelpSteer2 (10K), Prometheus (100K responses), RubricBench (36.5% general chat) | Medium |
| **Instruction Following** | None | WildBench (1K), AdvancedIF (1,645), WildChecklists (130K), RubricEval (3,486) | **High** |
| **Agent Tasks** | None | AdaRubric (WebArena, ToolBench) | Medium |
| **Ordinal criteria** | CHARM-100 only | Prometheus (1–5 Likert per rubric), HelpSteer2 (5-dim Likert), BiGGen-Bench (instance-specific 5-point), SedarEval | **High** |
| **Nominal criteria** | CHARM-100 only | TIGERScore (error categories), MQM (error typology), Open Rubric System (vₖ ∈ {−2..2}) | **High** |
| **Negative criteria** | None | TIGERScore (−5 to −0.5 penalties), SedarEval (penalty points), DR Tulu (negative rubrics), ResearchRubrics (negative rubrics), Open Rubric System (−2/−1 scores) | **Critical** |
| **Heterogeneous criterion mix** | CHARM-100 only | RubricHub (~110K, mixed), OpenRubrics (Hard Rules + Principles) | **High** |
| **Weighted criteria** | ResearcherBench (weights 1–3), RiceChem (inferred) | RaR (Essential/Important/Optional/Pitfall), AdvancedIF, Open Rubric System, ResearchRubrics (mandatory/optional) | Medium |

---

## 5. High-Priority Acquisition Targets

Ranked by value-to-effort ratio for AutorubricLM training:

1. **RubricHub** (~110K rubrics, HuggingFace) — Largest rubric dataset. Multi-domain. Already in (prompt, rubric) format. Needs quality filtering via meta-rubric, but provides enormous Tier 2 scale. Their coarse-to-fine pipeline is philosophically similar to Tier 2, so the delta is whether their rubrics pass your quality filter.

2. **OpenRubrics** (~35.6K, HuggingFace) — Contrastive generation teaches what makes rubrics discriminative. The Hard Rules + Principles format offers a natural decomposition. Consistency filtering already applied.

3. **WildChecklists** (130K instructions) — Massive scale. Binary-only and uniform-weight (a degenerate case of your schema), but provides breadth. Could be used selectively for binary-criterion training. The importance-weight annotations partially address the weight calibration concern.

4. **HelpSteer2** (10K, CC-BY-4.0) — Clean 5-dimension Likert annotations. Maps naturally to ordinal criteria. The only large-scale dataset with explicit multi-dimensional annotations under a permissive license. Already the standard in RL reward modeling.

5. **Prometheus Feedback Collection** (1K rubrics, 100K responses) — Custom per-task scoring rubrics with explicit score-band descriptions. Closest existing dataset to AutorubricLM's ordinal output format. The 1K rubrics cover diverse evaluation criteria; each has behavioral anchors at 5 levels.

6. **RaR-Medicine + RaR-Science** (~40K prompts, public) — Instance-specific weighted binary checklists. The categorical weight system (Essential/Important/Optional/Pitfall) is a natural fit for teaching weight calibration. Already validated via on-policy RL.

7. **Rubicon Rubric Bank** (10K+ rubrics) — Human-LLM hybrid construction spanning three scopes. The emphasis on rubric diversity and granularity directly addresses your quality model. Key insight: "single rubrics invite exploitation" validates the need for diverse rubric generation.

8. **DeepResearch Bench II** (9,430 binary rubrics, 22 domains) — Expert-derived hierarchical rubrics. High quality, directly usable for Tier 1 conversion.

9. **TIGERScore** (42K error analyses) — **Only major dataset with explicit negative-weight criteria** (penalty scores −5 to −0.5). Critical for teaching AutorubricLM negative criterion generation. Also provides nominal criterion structure (error categories).

10. **AdvancedIF** (1,645 prompts with expert rubrics) — Expert-written, per-prompt, with up to 20 criteria. The rubric verifier pipeline (generator → finetuned verifier → reward shaping) is directly applicable to Tier 3 validation.

11. **DREsS** (48.9K essays) — Fills the writing domain gap with real classroom data + standardized rubrics. Largest education-specific rubric dataset.

12. **BiGGen-Bench** (765 instances, instance-specific rubrics, NAACL Best Paper) — High-quality behavioral anchors on 5-point scales. Good training signal for ordinal option design. 9 capability types provide natural domain diversity.

13. **ExpertLongBench** (1,050 samples, 9 expert domains) — Fills professional domain gaps (law, medicine, finance, cybersecurity). CLEAR framework aligns with checklist-style evaluation.

14. **ResearchRubrics** (2,593 criteria, expert-written, auth required) — Most carefully crafted rubric dataset. Includes both positive and negative criteria, mandatory/optional distinction, ternary grading. The positive/negative + mandatory/optional structure maps well to your weighted Criterion schema.

15. **SedarEval** (1,000 questions with self-adaptive rubrics) — Scoring points + penalty points + background knowledge. The penalty point concept directly maps to negative criteria. Discriminative filtering during construction aligns with your quality model.

---

## 6. Observations for the Spec Discussion

### The negative criteria problem

The spec devotes significant space to negative criteria (anti-pattern penalties, sycophancy countermeasures, weight asymmetry principle). The survey reveals this is genuinely novel territory — almost no external dataset has negative-weight criteria. Sources with partial coverage:

- **TIGERScore**: Penalty scores −5 to −0.5 (error-based, closest match)
- **SedarEval**: Penalty points deducting for deviations from expected tendencies
- **DR Tulu**: Explicit "negative rubrics" summarizing undesirable behaviors
- **ResearchRubrics (Scale)**: Negative rubrics penalizing extraneous/incorrect content
- **Open Rubric System**: Comparative scores allowing negative values (−2/−1)

Even with all these sources combined, Tier 2 synthetic generation will carry the majority of the weight for teaching negative criterion generation with proper weight asymmetry.

### Scale reality check

The spec doesn't mention target dataset sizes. From the survey:
- SFT corpora in this space range from 1K rubrics (Prometheus) to 130K instructions (WildChecklists)
- Successful RL training uses 5K–40K prompts (RaR: ~40K; Rubicon: ~5K; DR Tulu: ~9K; AdvancedIF: 1.6K)
- DPO preference pairs typically need 5K–50K examples
- Your existing labeled data totals ~1,400 tasks — enough for validation and Tier 3 test sets, but Tier 2 must produce the SFT bulk

**Concrete target:** ~30K–50K high-quality (prompt, rubric) pairs for SFT (achievable by filtering RubricHub + OpenRubrics through your meta-rubric, supplemented by targeted synthetic generation for underrepresented domains and negative criteria). ~5K–10K preference pairs for DPO from Tier 3 closed-loop validation.

### RubricHub as potential shortcut vs. quality risk

At 110K rubrics, RubricHub could substantially reduce the Tier 2 generation burden. But their rubrics were generated by GPT-5.1 + Gemini 3 Pro without your meta-rubric's anti-pattern taxonomy. The key question: what fraction passes your meta-rubric filter (both standalone and in-context)? Running the meta-rubric evaluation module on a RubricHub sample would be a high-value experiment.

### The RL-with-rubric-rewards explosion

The survey found 15+ papers in 2025–2026 using rubric-based RL rewards. The dominant pattern: GRPO + instance-specific weighted binary checklists + LLM-as-judge scoring. Key insights from this literature:

- **Rubric diversity prevents exploitation.** Rubicon and DR Tulu both find that single/static rubrics are gamed. AutorubricLM should generate diverse rubrics for the same task.
- **Evolving rubrics outperform static ones.** DR Tulu's variance-based buffer management (keep rubrics where responses vary, drop where all responses score the same) is a principled approach to discriminative power.
- **Weight matters.** RaR's categorical weights (Essential/Important/Optional/Pitfall) and R-GRPO's adaptive weights both show that non-uniform weights improve RL outcomes. This validates the spec's weight calibration emphasis.
- **Negative rubrics are emerging.** DR Tulu and ResearchRubrics both explicitly use negative rubrics. The field is converging on the spec's insight that rubrics need both reward and penalty dimensions.

### Checklist vs. structured rubric

Many papers (WildBench, TICK, WildChecklists, ExpertLongBench, RuscaRL) use YES/NO checklists — binary, often equal-weight. These are a degenerate case of the Criterion schema. Options for training:

- **Include as binary-criterion data** with uniform weights (teaches format, hurts weight calibration)
- **Convert to weighted binary** by using importance annotations where available (WildChecklists has importance weights)
- **Exclude** and only use for evaluation holdouts

Recommendation: include selectively, tagged as `criterion_mix: binary`, and ensure ordinal/nominal/heterogeneous examples dominate training proportionally.

### The retrieval angle

RubricRAG found retrieval-based prompting more effective than SFT for rubric quality. Rubicon's "instance-specific rubrics from stronger agentic workflows" and DR Tulu's "persistent rubrics initialized via search" both use retrieval. This suggests AutorubricLM could benefit from a hybrid mode: trained model + rubric retrieval index. The retrieval index could be seeded from the training corpus itself.

### Rubric-as-latent-variable framing

Rubric-ARM's formulation (rubrics as latent variables in a generalized EM, with alternating GRPO updates for generator and judge) is theoretically interesting for AutorubricLM's Tier 3. Instead of the spec's current approach (generate K rubrics → evaluate each → rank → DPO), Rubric-ARM's alternating optimization could potentially co-improve the rubric generator and the evaluation pipeline simultaneously.

### RubricBench's 27% gap finding

RubricBench reports a 27% accuracy gap between model-generated and human-authored rubrics. This quantifies the value proposition of AutorubricLM: a well-trained rubric generator could close this gap. It also suggests that rubric quality is currently the bottleneck in LLM-as-judge systems, not judge quality — making rubric generation the highest-leverage intervention.
