"""Grade retinal fundus photographs for image quality with an LLM judge.

A proof of concept for the "Fundus Image Quality (EyeQ)" cookbook recipe. It draws a few
images of each EyeQ quality grade, grades them with gemini-3.8-flash against a two-criterion
rubric built from EyeQ's grading rules, prints the evaluation report, and saves every image
whose criterion the judge got wrong next to the judge's reasoning.

EyeQ ships labels only. You need:

1. The EyeQ label file Label_EyeQ_train.csv from https://github.com/hzfu/EyeQ (data/).
2. A few of the matching EyePACS training images in one folder. The Kaggle dataset
   tanlikesmath/diabetic-retinopathy-resized keeps their original filenames and can be
   downloaded one file at a time:

       kaggle datasets download tanlikesmath/diabetic-retinopathy-resized \\
           -f resized_train/resized_train/11937_left.jpeg -p eyepacs

   Only images whose files are in the folder are sampled.

Usage:

    uv run python examples/eyeq_fundus_quality.py \\
        --labels Label_EyeQ_train.csv --images eyepacs --per-grade 2

Needs GEMINI_API_KEY. Six images cost about $0.06.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import random
from pathlib import Path

import PIL.Image as pil_image

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    Image,
    LLMConfig,
    Rubric,
    RubricDataset,
    evaluate,
)
from autorubric.graders import CriterionGrader

# EyeQ's label codes and the grading rules its paper gives for each grade.
GRADES = {
    0: "Good: no low-quality indicators, and every retinal structure is clearly visible",
    1: (
        "Usable: the optic disc and macula are present, but low-quality indicators "
        "(blur, uneven illumination, over- or under-exposure, artifacts) affect at most "
        "20% of the fundus"
    ),
    2: (
        "Reject: the optic disc or the macula is missing, or low-quality indicators "
        "cover more than 20% of the fundus"
    ),
}

RUBRIC = Rubric(
    [
        Criterion(
            name="quality_grade",
            weight=10.0,
            requirement=(
                "Grade the quality of this retinal fundus photograph for diabetic "
                "retinopathy screening."
            ),
            scale_type="ordinal",
            options=[
                CriterionOption(label=GRADES[0], value=1.0),
                CriterionOption(label=GRADES[1], value=0.5),
                CriterionOption(label=GRADES[2], value=0.0),
            ],
        ),
        Criterion(
            name="gradeable",
            weight=5.0,
            requirement=(
                "The photograph can be used for screening: the optic disc and the macula "
                "are both in view, and low-quality indicators cover at most 20% of the fundus"
            ),
        ),
    ]
)

PROMPT = "A retinal fundus photograph taken for diabetic retinopathy screening."


def sample_rows(labels: Path, images: Path, per_grade: int, seed: int) -> list[dict]:
    """Draw `per_grade` labelled rows of each grade whose image file exists."""
    by_grade: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    with labels.open(newline="") as f:
        for row in csv.DictReader(f):
            if (images / row["image"]).exists():
                by_grade[int(row["quality"])].append(row)
    rng = random.Random(seed)
    picked = []
    for grade, rows in by_grade.items():
        if len(rows) < per_grade:
            raise SystemExit(f"only {len(rows)} images of grade {grade} found in {images}")
        picked += rng.sample(rows, per_grade)
    return picked


def downsized(path: Path, max_side: int = 1024) -> Image:
    """EyePACS photos run to ~4000 px; a 1024 px JPEG keeps requests small."""
    with pil_image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return Image(buf.getvalue())


def build_dataset(rows: list[dict], images: Path) -> RubricDataset:
    dataset = RubricDataset(prompt=PROMPT, rubric=RUBRIC, name="eyeq-poc")
    for row in rows:
        grade = int(row["quality"])
        dataset.add_item(
            submission=downsized(images / row["image"]),
            description=row["image"],
            ground_truth=[GRADES[grade], "UNMET" if grade == 2 else "MET"],
        )
    return dataset


def verdict_text(cr) -> str | None:
    if cr.final_multi_choice_verdict is not None:
        return cr.final_multi_choice_verdict.selected_label
    return cr.final_verdict.value if cr.final_verdict is not None else None


def truth_text(truth) -> str:
    return truth.value if isinstance(truth, CriterionVerdict) else str(truth)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--per-grade", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("eyeq_failures"))
    parser.add_argument("--model", default="gemini/gemini-3.8-flash")
    args = parser.parse_args()

    rows = sample_rows(args.labels, args.images, args.per_grade, args.seed)
    dataset = build_dataset(rows, args.images)

    print("== Data ==")
    for item in dataset.items:
        print(f"{item.description:24s} truth: {item.ground_truth[0].split(':')[0]}")

    grader = CriterionGrader(judge_model_config=LLMConfig(model=args.model), seed=args.seed)
    result = await evaluate(
        dataset,
        grader,
        max_concurrent_items=4,
        show_progress=False,
        experiment_name=f"eyeq-poc-{args.model.rsplit('/', 1)[-1]}",
    )

    print("\n== Evaluation report ==")
    print(result.compute_metrics(dataset).summary())

    print("\n== Criteria the judge got wrong ==")
    args.out.mkdir(exist_ok=True)
    for r in result.item_results:
        if r.error is not None:
            print(f"{r.item.description}: grading failed: {r.error}")
            continue
        for cr, truth in zip(r.report.report, r.item.ground_truth, strict=True):
            predicted = verdict_text(cr)
            if predicted == truth_text(truth):
                continue
            image_path = args.out / f"{Path(r.item.description).stem}.jpg"
            image_path.write_bytes(r.item.submission.read_bytes())
            print(f"\n{r.item.description} [{cr.criterion.name}]  image: {image_path}")
            print(f"  truth:     {truth_text(truth)}")
            print(f"  predicted: {predicted}")
            print(f"  reasoning: {cr.final_reason}")


if __name__ == "__main__":
    asyncio.run(main())
