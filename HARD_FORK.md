# autorubric: why hard fork?

`autorubric` is a <a href="https://producingoss.com/en/forks.html#:~:text=Hard%20forks%20(also,cooperative%20development%20fork.">hard fork</a> of [`rubric` v1.2.8](https://github.com/The-LLM-Data-Company/rubric/releases/tag/v1.2.8) by The LLM Data Company. We are grateful to The LLM Data Company for their work and for making the code available under the MIT license. While `autorubric` shares a conceptual similarity with `rubric` in the sense that it is a library for evaluating text outputs against criteria via LLM-as-a-judge, it should be considered categorically distinct, as this document explains.

Note (added on 3/27): rubric has continued active development since v1.2.8 and is currently on v2.2.0. This document describes the state of the library at the time of the fork, and many of the observations below have since changed.

---

## Where rubric v1.2.8 fell short

`rubric` v1.2.8 was a clean, minimal implementation of rubric-based LLM grading: define weighted criteria, call an LLM, get MET/UNMET verdicts. About 800 lines of Python across 9 files. As a starting point it was useful, but several gaps made it unsuitable for evaluation work where you need to trust the scores.

The library used a generate_fn callable pattern that allowed any LLM provider. The package used Gemini by default and parsed responses with regex-based JSON extraction. While you could swap in another model by writing a custom async callable, but the friction discouraged the systematic model comparisons that evaluation work requires.

There was also no way to know whether a verdict was reliable. A single judge produced a single MET or UNMET per criterion, with no mechanism for agreement measurement or confidence estimation. You couldn't tell whether a score reflected a stable judgment or an artifact of one model's behavior on one prompt variation.

The criterion model was limited to binary verdicts. There was no "cannot assess" option, so when the grader failed to parse a response it silently defaulted to UNMET regardless of criterion polarity, a bug for negative-weight criteria. Ordinal and multi-choice scales were absent.

Beyond grading, there was no infrastructure for running evaluations at scale: no dataset abstraction, no batch runner, no ground-truth comparison, no metrics. Each experiment required bespoke scripts for parallelism, checkpointing, and analysis. And there was no mechanism for assessing rubric quality itself, whether criteria were clear enough for an LLM to apply consistently, or whether they contained anti-patterns that quietly inflated or suppressed scores.

These gaps compounded. Ensemble judging was the prerequisite for measuring reliability, but metrics were the prerequisite for detecting low reliability, richer criteria for expressing the distinctions that would raise it, and meta-evaluation for diagnosing whether the rubric was the bottleneck. Each missing piece made the others harder to see.

---

## What we built instead

### Provider-agnostic LLM layer

We replaced the Gemini dependency with a provider-agnostic backend (via LiteLLM) that routes to Anthropic, OpenAI, Google, Azure, and others through a single interface. Responses use Pydantic structured outputs rather than regex parsing, which eliminated a persistent source of failures on malformed model output. Token usage and cost are tracked per call and aggregated, because when you're running thousands of evaluations the bill matters as much as the methodology.

### From three graders to one

`rubric` shipped three grading strategies: per-criterion (one LLM call per criterion), one-shot (all criteria in a single call), and holistic (a single 0-100 score). We kept only the per-criterion approach. In the one-shot variant, criteria evaluations could contaminate each other within a shared context window: a judge that decided criterion 3 was unmet might let that color its reading of criterion 4. The holistic approach discarded per-criterion transparency entirely and asked the LLM to mentally compute weighted sums.

The surviving grader treats a solo LLM as an ensemble of one, so the same code path handles both single-judge and multi-judge configurations. Ensemble judging is the default operating mode; single-judge evaluation is the degenerate case.

### Ensemble judging and agreement

Verga et al. (2024) showed that panels of diverse models reduce the systematic errors inherent in any single judge. autorubric implements this directly: you compose a jury from different model families, each with its own temperature and weight. Inter-judge agreement is reported primarily via Krippendorff's alpha — the general, recommended statistic, which handles unequal/missing raters (errored or excluded votes) and is level-aware (nominal for binary/nominal criteria, ordinal for ordered scales). Fleiss' kappa is retained alongside it as the classic fixed-rater nominal measure, computed complete-case. Agreement-with-ground-truth uses Cohen's kappa (binary), quadratic-weighted kappa (ordinal), interpreted on the Landis & Koch (1977) scale.

### Richer criteria

Binary MET/UNMET is now one option among several. Criteria can define multi-choice options with explicit values on a 0-1 scale, typed as ordinal or nominal, with per-option behavioral anchors. This addresses Zheng et al.'s (2023) finding that low-precision ordinal scales suffer from central-tendency collapse, and Kim et al.'s (2024, Prometheus 2) evidence that concrete anchor descriptions improve consistency.

CANNOT_ASSESS is a first-class verdict, following Min et al. (2023, FActScore), who demonstrated that an explicit "cannot determine" option is essential for atomic evaluation. The system handles it with configurable strategies: exclude from the denominator, treat as zero, assign partial credit, or fail conservatively.

Position bias (Wang et al. 2023) is mitigated by shuffling option presentation order independently per judge and per call, so that each judge in an ensemble sees a different permutation and residual positional effects average out rather than compound.

### Bias countermeasures

LLM judges tend toward positive verdicts (sycophantic bias) and prefer longer outputs (verbosity bias, per Dubois et al. 2024). autorubric addresses these at two levels. A length penalty mechanism applies configurable deductions when submission length exceeds a free budget. Separately, the meta-rubric system flags rubrics that lack negative-weight criteria, contain unfalsifiable requirements, or inadvertently reward verbosity: the structural properties that let these biases through in the first place.

### Few-shot calibration

Ashktorab et al. (2025, EvalAssist) found that graded exemplars spanning the verdict space improve judge calibration. autorubric selects few-shot examples stratified across MET/UNMET/CANNOT_ASSESS per criterion, so the judge sees balanced anchors rather than a skewed sample. When training data is available, this calibration happens automatically.

### Evaluation infrastructure

Datasets with optional ground-truth verdicts, per-item rubrics, and stratified train/test splitting replace the bare one-item-at-a-time grading of `rubric`. A batch evaluation runner handles concurrent grading, progress display, experiment checkpointing, and resumption from partial runs.

The metrics layer computes what you need for a validity argument in the sense of Casabianca et al. (2025): Cohen's kappa at the criterion level, Spearman and Kendall correlations at the score level, systematic bias detection via paired t-tests with Cohen's d effect sizes, and bootstrap confidence intervals. Distribution comparison uses Earth Mover's Distance and Kolmogorov-Smirnov tests. He et al. (2025) noted that correlation alone can mask systematic bias: a judge whose scores track ground truth but are shifted upward will show high Spearman rho and low accuracy simultaneously. The distributional metrics catch exactly this failure mode.

### Rubric improvement loop

The meta-evaluation module treats the rubric itself as an object of study. Two sets of meta-rubric criteria, one for standalone quality (clarity, structure, anti-patterns) and one for task-aligned quality (coverage, discriminative power), grade the rubric the same way the rubric grades submissions. An iterative improvement loop evaluates quality, validates against ground truth (Spearman rho) or inter-judge agreement, revises via LLM, and accepts the revision only under a Pareto constraint: quality may improve, but reliability must not regress. The loop terminates when issues are resolved, scores plateau, or a cost budget is exhausted. Each iteration's rubric, evaluation, and revision prompt are persisted as artifacts for post-hoc analysis.

This operationalizes the evidence-centered design process that Casabianca et al. (2025) advocate: iterative refinement against gold sets, with systematic reliability measurement at each step.

For more details of how `autorubric` addresses these problems, we encourage readers to our preprint:
```bibtex
  @misc{rao2026autorubric,
        title={Autorubric: A Unified Framework for Rubric-Based LLM Evaluation},
        author={Delip Rao and Chris Callison-Burch},
        year={2026},
        eprint={2603.00077},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2603.00077},
  }
```
