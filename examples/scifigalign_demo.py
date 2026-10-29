#!/usr/bin/env python3
"""Grade SciFigAlign figures with an LLM judge or with Jev, a decision model.

SciFigAlign (Xu et al., 2026, arXiv:2607.27066) asks for four peer-review scores from 1 to
5 (Clarity, Relevance, Informativeness, Structure) for a figure cropped from an ICLR 2024
paper. ``scripts/convert_scifigalign.py`` turns the 300 human-rated figures of its GitHub
demo into AutoRubric datasets, each submission a figure followed by its caption; see
``examples/data/scifigalign/README.md`` for the data, the rubric and how the paper's metrics
map onto AutoRubric's.

By default the demo spends nothing. It loads the split and prints its score histograms, the
error of constant predictors and an estimate of what grading the split would cost, then
exits. ``--run`` grades the split and prints the paper's metrics next to its reference rows.

Usage:

    uv run python scripts/convert_scifigalign.py            # once, to write the datasets
    uv run python examples/scifigalign_demo.py              # dry run, LLM judge
    uv run python examples/scifigalign_demo.py --judge jev --input-cost 0.042e-6
    uv run python examples/scifigalign_demo.py --limit 3 --run    # grades 3 figures

    [--split train|val|test|all] [--judge llm|jev] [--model M]
    [--llm-calls per_item|per_criterion] [--ordinal-framing choice|score]
    [--input-cost USD_PER_TOKEN] [--image-tokens N] [--limit N] [--few-shot N] [--run]

The LLM judge needs the provider's key (e.g. ``OPENAI_API_KEY`` for the default model) and
a model that reads images. Jev needs ``pip install 'autorubric[typesafe]'``,
``TYPESAFE_API_KEY`` and a model that autorubric's decision-model registry lists as
accepting images: ``typesafe/jev-1.14.0`` does, ``typesafe/jev-1.13.0`` does not, and
autorubric refuses it with its own error before any request is sent. Keys may be set in a
``.env`` file. Responses are cached on disk, so a repeated run costs nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import litellm
import PIL.Image as pil_image

from autorubric import (
    Content,
    DecisionModelConfig,
    EvalResult,
    FewShotConfig,
    Image,
    LLMConfig,
    MetricsResult,
    RubricDataset,
    decision_model_capabilities,
    evaluate,
)
from autorubric.decision import build_state
from autorubric.graders import CriterionGrader
from autorubric.metrics import OrdinalCriterionMetrics
from autorubric.prompts import (
    DECISION_MODEL_TASK_INSTRUCTION,
    MULTI_CHOICE_FEW_SHOT_ADDITION,
    MULTI_CHOICE_SYSTEM_PROMPT,
    build_multi_choice_user_prompt,
    build_rubric_system_prompt,
    build_rubric_user_prompt,
)
from autorubric.types import Criterion

DATA_DIR = Path(__file__).resolve().parent / "data" / "scifigalign"
SPLITS = ("train", "val", "test", "all")

DEFAULT_LLM_MODEL = "openai/gpt-4.1-mini"
DEFAULT_JEV_MODEL = "typesafe/jev-1.14.0"

# Assumptions behind the cost estimate. Providers count an image's input tokens in their
# own way (by tiles, patches or pixels), so one allowance per image stands in for all of
# them; override it with --image-tokens. Output is a JSON judgment per criterion.
DEFAULT_IMAGE_TOKENS = 1_500
OUTPUT_TOKENS_PER_CRITERION = 150

SCORES = (1, 2, 3, 4, 5)
# Option values run from 0 to 1 in steps of 0.25, so an error on them times 4 is an error
# in score points on the paper's 1 to 5 scale.
POINTS_PER_UNIT = 4

# Rows of the paper's results on its own human-rated test set: 396 figures, each judged with
# its citing paragraphs. They are for orientation, not comparison (see the README).
# Columns: macro MAE, SRCC, pairwise accuracy (PA), and per-dimension MAE where reported.
REFERENCE_ROWS = (
    ("LLM-as-judge (v9), paper", 0.864, 0.198, 0.632, {"CL": 0.902, "IN": 0.828, "ST": 0.947}),
    ("SciFigAlign (fine-tuned), paper", 0.3524, 0.3088, 0.8164, {}),
)
SHORT_NAMES = {
    "clarity": "CL",
    "relevance": "RE",
    "informativeness": "IN",
    "structure": "ST",
}


# =============================================================================
# Arguments
# =============================================================================


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be an integer >= 1, got {text}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be an integer >= 0, got {text}")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"must be a number >= 0, got {text}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade SciFigAlign figures. Without --run, only estimates and exits."
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--judge", choices=("llm", "jev"), default="llm")
    parser.add_argument(
        "--model",
        help=f"the judge's model (default {DEFAULT_LLM_MODEL} for llm, "
        f"{DEFAULT_JEV_MODEL} for jev)",
    )
    parser.add_argument(
        "--llm-calls",
        choices=("per_item", "per_criterion"),
        help="LLM judge only: one call per figure for all four scores (default), or one "
        "call per score, which sends the figure four times",
    )
    parser.add_argument(
        "--ordinal-framing",
        choices=("choice", "score"),
        help="Jev only: pose each dimension as a choice among its five options (default) "
        "or as a score on an ordered scale",
    )
    parser.add_argument(
        "--input-cost",
        type=_non_negative_float,
        help="Jev only: your price in USD per input token, e.g. 0.042e-6 "
        "(DecisionModelConfig.input_cost_per_token); without it no cost is reported",
    )
    parser.add_argument(
        "--image-tokens",
        type=_non_negative_int,
        default=DEFAULT_IMAGE_TOKENS,
        help=f"input tokens assumed per image in the estimate (default {DEFAULT_IMAGE_TOKENS})",
    )
    parser.add_argument("--limit", type=_positive_int, help="grade only the first N figures")
    parser.add_argument(
        "--few-shot",
        type=_positive_int,
        help="LLM judge only: show N training figures with their human scores in each call "
        "(examples drawn from train.json)",
    )
    parser.add_argument(
        "--run", action="store_true", help="grade the split (spends on the judge's API)"
    )
    args = parser.parse_args(argv)

    if args.judge == "jev":
        for flag, value in (("--llm-calls", args.llm_calls), ("--few-shot", args.few_shot)):
            if value is not None:
                parser.error(
                    f"{flag} applies to LLM judges only (a decision model makes one request "
                    "per figure and takes no few-shot examples)"
                )
        args.model = args.model or DEFAULT_JEV_MODEL
        args.ordinal_framing = args.ordinal_framing or "choice"
    else:
        for flag, value in (
            ("--ordinal-framing", args.ordinal_framing),
            ("--input-cost", args.input_cost),
        ):
            if value is not None:
                parser.error(f"{flag} applies to --judge jev only")
        args.model = args.model or DEFAULT_LLM_MODEL
        args.llm_calls = args.llm_calls or "per_item"
    if args.few_shot is not None and args.split in ("train", "all"):
        parser.error(
            "--few-shot draws its examples from train.json; grade val or test with it, so "
            "that no figure is shown as an example of itself"
        )
    return args


# =============================================================================
# Data
# =============================================================================


def load_dataset(split: str) -> RubricDataset:
    path = DATA_DIR / f"{split}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Write the datasets first:\n"
            "    uv run python scripts/convert_scifigalign.py"
        )
    return RubricDataset.from_file(path)


def first_items(dataset: RubricDataset, limit: int | None) -> RubricDataset:
    """``dataset`` with only its first ``limit`` items (all of them for ``None``)."""
    if limit is None or limit >= len(dataset):
        return dataset
    return RubricDataset(
        prompt=dataset.prompt,
        rubric=dataset.rubric,
        items=dataset.items[:limit],
        name=f"{dataset.name}-first-{limit}",
        reference_submission=dataset.reference_submission,
    )


def criteria_of(dataset: RubricDataset) -> list[Criterion]:
    if dataset.rubric is None:
        raise SystemExit(f"{dataset.name}: the dataset has no rubric")
    return dataset.rubric.rubric


def human_scores(dataset: RubricDataset) -> list[list[int]]:
    """Each item's four human scores, 1 to 5, read through its labels' options."""
    criteria = criteria_of(dataset)
    scores = []
    for item in dataset.items:
        labels = item.ground_truth or []
        scores.append(
            [
                SCORES[criterion.find_option_by_label(str(label))]
                for criterion, label in zip(criteria, labels, strict=True)
            ]
        )
    return scores


def paper_id(description: str) -> str:
    """The paper id that starts an item's description (``"<paper_id> <figure_id> (...)"``)."""
    return description.split(" ", 1)[0]


def same_paper_pairs(dataset: RubricDataset) -> int:
    """How many pairs of figures in ``dataset`` come from the same paper."""
    counts = Counter(paper_id(item.description) for item in dataset.items)
    return sum(n * (n - 1) // 2 for n in counts.values())


def content_parts(content: Content) -> list[str | Image]:
    """The parts of a submission or prompt: text, an image, or a document of both."""
    if isinstance(content, str | Image):
        return [content]
    return list(content)


def figure_of(content: Content) -> list[Image]:
    """The images of a submission."""
    return [part for part in content_parts(content) if isinstance(part, Image)]


def pixel_size(path: Path) -> tuple[int, int]:
    """An image file's width and height in pixels (only its header is read)."""
    with pil_image.open(path) as opened:
        return opened.size


def text_and_image_count(content: Content) -> tuple[str, int]:
    """A prompt's text, with its images left out, and how many images it holds."""
    parts = content_parts(content)
    text = "".join(part for part in parts if isinstance(part, str))
    return text, sum(1 for part in parts if isinstance(part, Image))


# =============================================================================
# What the dry run prints
# =============================================================================


def print_histograms(dataset: RubricDataset, split: str) -> None:
    criteria = criteria_of(dataset)
    scores = human_scores(dataset)
    print(f"\nHuman scores in {split} ({len(dataset)} figures)")
    print(f"  {'dimension':<16} " + " ".join(f"{s:>4}" for s in SCORES) + "    mean")
    for index, criterion in enumerate(criteria):
        column = [row[index] for row in scores]
        counts = Counter(column)
        row = " ".join(f"{counts[s]:>4}" for s in SCORES)
        print(f"  {criterion.name or '':<16} {row}   {statistics.mean(column):.2f}")


def macro_errors(
    predictions: Sequence[float], truth: Sequence[Sequence[int]]
) -> tuple[float, float]:
    """Macro MAE and macro RMSE, in score points, of one constant prediction per dimension."""
    maes, rmses = [], []
    for index, prediction in enumerate(predictions):
        errors = [prediction - row[index] for row in truth]
        maes.append(statistics.mean(abs(e) for e in errors))
        rmses.append(statistics.mean(e * e for e in errors) ** 0.5)
    return statistics.mean(maes), statistics.mean(rmses)


def print_constant_baselines(dataset: RubricDataset, train: RubricDataset, split: str) -> None:
    """The error of predicting the same scores for every figure, which costs nothing."""
    truth = human_scores(dataset)
    train_scores = human_scores(train)
    means = [statistics.mean(row[i] for row in train_scores) for i in range(len(truth[0]))]
    baselines = {
        "train mean per dimension": means,
        "train mean, rounded": [float(round(m)) for m in means],
        "always 3": [3.0] * len(means),
        "always 4": [4.0] * len(means),
    }
    print(f"\nConstant predictions on {split} (score points on the 1 to 5 scale)")
    if split in ("train", "all"):
        print("  (the train means are measured on figures they were computed from)")
    print(f"  {'prediction':<26} {'macro MAE':>9} {'macro RMSE':>10}")
    for name, predictions in baselines.items():
        mae, rmse = macro_errors(predictions, truth)
        print(f"  {name:<26} {mae:>9.3f} {rmse:>10.3f}")
    print(f"  train means: {', '.join(f'{m:.3f}' for m in means)} (CL, RE, IN, ST)")


def llm_call_prompts(
    dataset: RubricDataset, llm_calls: str, with_examples: bool
) -> list[tuple[str, Content]]:
    """The system and user prompt of every call the LLM judge makes for ``dataset``.

    Built with the prompt builders the grader uses, without few-shot examples and with the
    options in rubric order (the grader shuffles them, which changes no token count).
    """
    criteria = criteria_of(dataset)
    guidelines = dataset.rubric.guidelines if dataset.rubric is not None else None
    guide = MULTI_CHOICE_SYSTEM_PROMPT + (MULTI_CHOICE_FEW_SHOT_ADDITION if with_examples else "")
    prompts: list[tuple[str, Content]] = []
    if llm_calls == "per_item":
        system = build_rubric_system_prompt(None, guide, with_examples=with_examples)
        ids = [(f"c{index}", criterion) for index, criterion in enumerate(criteria)]
        for item in dataset.items:
            user = build_rubric_user_prompt(ids, item.submission, guidelines=guidelines)
            prompts.append((system, user))
    else:
        for item in dataset.items:
            for criterion in criteria:
                user = build_multi_choice_user_prompt(
                    criterion, item.submission, guidelines=guidelines
                )
                prompts.append((guide, user))
    return prompts


def example_tokens(train: RubricDataset, model: str) -> float:
    """Text tokens one few-shot example adds: its caption and its four labels, on average."""
    counts = []
    for item in train.items:
        text, _ = text_and_image_count(item.submission)
        labels = "\n".join(str(label) for label in item.ground_truth or [])
        counts.append(litellm.token_counter(model=model, text=f"{text}\n{labels}"))
    return statistics.mean(counts)


def _usd(amount: float) -> str:
    return f"${amount:.4f}"


def print_llm_estimate(
    dataset: RubricDataset, args: argparse.Namespace, train: RubricDataset
) -> None:
    prompts = llm_call_prompts(dataset, args.llm_calls, with_examples=args.few_shot is not None)
    n_criteria = len(criteria_of(dataset))
    text_tokens = images = 0
    for system, user in prompts:
        text, n_images = text_and_image_count(user)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        text_tokens += litellm.token_counter(model=args.model, messages=messages)
        images += n_images
    shots = args.few_shot or 0
    if shots:
        text_tokens += round(len(prompts) * shots * example_tokens(train, args.model))
        images += len(prompts) * shots
    image_tokens = images * args.image_tokens
    criteria_per_call = n_criteria if args.llm_calls == "per_item" else 1
    output_tokens = len(prompts) * criteria_per_call * OUTPUT_TOKENS_PER_CRITERION
    input_tokens = text_tokens + image_tokens

    print(f"\nEstimate for {args.model}, {args.llm_calls} ({len(prompts)} calls)")
    print(f"  text input tokens   {text_tokens:>10,}  (litellm.token_counter)")
    print(
        f"  image input tokens  {image_tokens:>10,}  ({images} images x {args.image_tokens}, "
        "an assumed allowance)"
    )
    if shots:
        print(f"  ({shots} few-shot examples per call, each an image, a caption and 4 labels)")
    print(
        f"  output tokens       {output_tokens:>10,}  "
        f"({OUTPUT_TOKENS_PER_CRITERION} per score, assumed)"
    )
    try:
        input_cost, output_cost = litellm.cost_per_token(
            model=args.model, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
    except Exception as error:  # LiteLLM raises its own errors for a model it cannot price
        print(f"  cost: LiteLLM has no price for {args.model} ({type(error).__name__})")
        return
    print(
        f"  cost                {_usd(input_cost + output_cost):>10}  "
        f"(${input_cost:.4f} input + ${output_cost:.4f} output, LiteLLM's prices, "
        "before any provider-side prompt caching)"
    )


def print_jev_estimate(dataset: RubricDataset, args: argparse.Namespace) -> bool:
    """Print the estimate; return whether the registry lists the model as taking images."""
    capabilities = decision_model_capabilities(args.model)
    criteria = criteria_of(dataset)
    guidelines = dataset.rubric.guidelines if dataset.rubric is not None else None
    questions = "\n".join(
        [DECISION_MODEL_TASK_INSTRUCTION]
        + [
            "\n".join([criterion.requirement, *(o.label for o in criterion.options or [])])
            for criterion in criteria
        ]
    )
    text_tokens = images = 0
    for item in dataset.items:
        state = build_state(item.submission, guidelines=guidelines)
        text_tokens += litellm.token_counter(text=json.dumps(state) + "\n" + questions)
        images += len(figure_of(item.submission))
    image_tokens = images * args.image_tokens
    input_tokens = text_tokens + image_tokens

    print(f"\nEstimate for {args.model} ({len(dataset)} requests, one per figure)")
    print(
        f"  text input tokens   {text_tokens:>10,}  (state and questions, counted with "
        "LiteLLM's default tokenizer, an approximation)"
    )
    print(
        f"  image input tokens  {image_tokens:>10,}  ({images} images x {args.image_tokens}, "
        "an assumed allowance)"
    )
    if args.input_cost is None:
        print("  cost: pass --input-cost (your price per input token) to price the run")
    else:
        print(
            f"  cost                {_usd(input_tokens * args.input_cost):>10}  "
            f"({input_tokens:,} input tokens x ${args.input_cost:g})"
        )

    if capabilities is None or not capabilities["images"]:
        listed = "does not list it" if capabilities is None else "lists it as taking no images"
        print(
            f"\nautorubric's decision-model registry {listed}: with --run, every figure fails "
            f"with autorubric's configuration error for {args.model}, raised before any "
            "request is sent."
        )
        return False
    low, high = capabilities["min_side_px"], capabilities["max_side_px"]
    sizes = [
        pixel_size(image.path)
        for item in dataset.items
        for image in figure_of(item.submission)
        if image.path is not None
    ]
    small = sum(1 for size in sizes if min(size) < low)
    elongated = sum(1 for size in sizes if max(size) / min(size) > high / low)
    large = sum(1 for size in sizes if max(size) > high)
    print(
        f"  fitting: {small} of {len(sizes)} crops have a side under {low} px; autorubric "
        f"scales them up before sending, and pads with white the {elongated} too elongated "
        f"to reach {low} px that way (aspect ratio over {high}/{low}). {large} have a side "
        f"over {high} px and are scaled down. Jev itself never resizes an image."
    )
    return True


# =============================================================================
# Running
# =============================================================================


def build_grader(args: argparse.Namespace, train: RubricDataset) -> CriterionGrader:
    # The annotators had no "not applicable" choice, so neither judge gets one.
    if args.judge == "jev":
        return CriterionGrader(
            judge_model_config=DecisionModelConfig(
                args.model,
                ordinal_framing=args.ordinal_framing,
                input_cost_per_token=args.input_cost,
                cache_enabled=True,
            ),
            auto_na_option=False,
        )
    few_shot_config = (
        None
        if args.few_shot is None
        else FewShotConfig(n_examples=args.few_shot, include_reason=False)
    )
    return CriterionGrader(
        judge_model_config=LLMConfig(args.model, cache_enabled=True, max_parallel_requests=8),
        llm_calls=args.llm_calls,
        auto_na_option=False,
        training_data=train if few_shot_config is not None else None,
        few_shot_config=few_shot_config,
    )


def experiment_name(args: argparse.Namespace, dataset: RubricDataset) -> str:
    settings = json.dumps(
        [args.judge, args.model, args.llm_calls, args.ordinal_framing, args.few_shot],
        sort_keys=True,
    )
    digest = hashlib.sha256(settings.encode()).hexdigest()[:8]
    return f"scifigalign-{dataset.name}-{args.judge}-{digest}"


def _fmt(value: float | None, scale: float = 1.0, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value * scale, spec)


def print_metrics(metrics: MetricsResult, dataset: RubricDataset) -> None:
    per_dimension = [m for m in metrics.per_criterion if isinstance(m, OrdinalCriterionMetrics)]
    print("\nPer dimension (MAE and RMSE in score points on the 1 to 5 scale)")
    print(
        f"  {'dimension':<16} {'n':>3} {'MAE':>6} {'RMSE':>6} {'exact':>6} {'+-1':>6} "
        f"{'w.kappa':>8} {'SRCC':>6}"
    )
    for m in per_dimension:
        print(
            f"  {m.name:<16} {m.n_samples:>3} {_fmt(m.mae, POINTS_PER_UNIT):>6} "
            f"{_fmt(m.rmse, POINTS_PER_UNIT):>6} {_fmt(m.exact_accuracy, 1, '.1%'):>6} "
            f"{_fmt(m.adjacent_accuracy, 1, '.1%'):>6} {_fmt(m.weighted_kappa):>8} "
            f"{_fmt(m.spearman.coefficient):>6}"
        )
    maes = [m.mae * POINTS_PER_UNIT for m in per_dimension if m.mae is not None]
    srccs = [m.spearman.coefficient for m in per_dimension if m.spearman.coefficient is not None]
    pairs = same_paper_pairs(dataset)
    print("\nThe paper's metrics")
    print(f"  macro MAE                      {_fmt(statistics.mean(maes) if maes else None)}")
    print(f"  macro SRCC (mean over dims)    {_fmt(statistics.mean(srccs) if srccs else None)}")
    print(f"  SRCC of the overall score      {_fmt(metrics.score_spearman.coefficient)}")
    if pairs:
        print(f"  PA                             not computed ({pairs} same-paper pairs)")
    else:
        print("  PA                             not computable (0 same-paper pairs)")
    if metrics.bootstrap is not None and metrics.bootstrap.kappa_ci is not None:
        low, high = metrics.bootstrap.kappa_ci
        print(f"  mean weighted kappa {_fmt(metrics.mean_kappa)}, 95% CI [{low:.3f}, {high:.3f}]")

    print("\nReference rows (the paper's 396-figure test set, with citing paragraphs)")
    print(f"  {'method':<34} {'MAE':>6} {'SRCC':>6} {'PA':>7}  per-dimension MAE")
    for name, mae, srcc, pa, dims in REFERENCE_ROWS:
        per_dim = ", ".join(f"{dim} {value}" for dim, value in dims.items()) or "-"
        print(f"  {name:<34} {mae:>6} {srcc:>6} {pa:>7.2%}  {per_dim}")


def print_usage(result: EvalResult) -> None:
    usage = result.total_token_usage
    cost = result.total_completion_cost
    print(f"\nGraded {result.successful_items} of {result.total_items} figures")
    if usage is not None:
        print(
            f"  tokens: {usage.prompt_tokens:,} input, {usage.completion_tokens:,} output "
            f"({usage.total_tokens:,} in all)"
        )
    print(f"  cost: {'unknown' if cost is None else f'${cost:.4f}'}")


async def run(args: argparse.Namespace, dataset: RubricDataset, train: RubricDataset) -> None:
    grader = build_grader(args, train)
    result = await evaluate(
        dataset,
        grader,
        show_progress=True,
        experiment_name=experiment_name(args, dataset),
        resume=False,
    )
    print_usage(result)
    if result.successful_items == 0:
        # Every item failed alike, e.g. for a model the registry refuses images for: show
        # autorubric's own error rather than a metrics error about having nothing to score.
        first = next((r.error for r in result.item_results if r.error), "no error recorded")
        raise SystemExit(f"No figure was graded. The first error recorded:\n{first}")
    metrics = result.compute_metrics(dataset, bootstrap=True, seed=42)
    print_metrics(metrics, dataset)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = first_items(load_dataset(args.split), args.limit)
    train = load_dataset("train")

    judge = f"{args.model} ({args.llm_calls})" if args.judge == "llm" else args.model
    print(f"SciFigAlign demo: {dataset.name}, {len(dataset)} figures, judge {judge}")
    print_histograms(dataset, args.split)
    print_constant_baselines(dataset, train, args.split)
    accepted = True
    if args.judge == "llm":
        print_llm_estimate(dataset, args, train)
    else:
        accepted = print_jev_estimate(dataset, args)

    if not args.run:
        advice = (
            "Re-run with --run to grade these figures."
            if accepted
            else f"Use --model {DEFAULT_JEV_MODEL} to grade these figures."
        )
        print(f"\nNothing was sent. {advice}")
        return
    asyncio.run(run(args, dataset, train))


if __name__ == "__main__":
    main(sys.argv[1:])
