#!/usr/bin/env python3
"""Convert the SciFigAlign human-rated demo to AutoRubric dataset files.

Source: SciFigAlign (arXiv:2607.27066), https://github.com/FrankDengAI/SciFigAlign, whose
``data/`` folder holds 300 figures from ICLR 2024 papers, each cropped from the paper's PDF
and scored by human annotators from 1 to 5 on Clarity, Relevance, Informativeness and
Structure, split by paper into 240 train, 30 val and 30 test figures.

The script fetches ``data/`` at a pinned commit with ``git`` into a cache directory (a
shallow, blob-less, sparse clone; nothing from it is executed), checks every row and figure,
copies the figures and writes one dataset file per split:

    examples/data/scifigalign/{train,val,test,all}.json
    examples/data/scifigalign/figures/<paper_id>__<figure_id>.png

Each item's submission is the figure followed by its caption, its ground truth the four
human scores as option labels of the rubric ``build_rubric`` returns (embedded in every
file), and its ground-truth reasons the annotators' rationales where they wrote one.
The output is git-ignored; only the folder's README is tracked.

Usage:

    uv run python scripts/convert_scifigalign.py [--cache-dir .cache/scifigalign]
        [--out-dir examples/data/scifigalign] [--force]

``--force`` replaces existing output (the four JSON files and ``figures/``; nothing else in
the output folder is touched) and a cache that is not a clean checkout of the pinned commit.
Requires git 2.27 or later (sparse checkout of a blob-less clone).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autorubric.content import Image
from autorubric.dataset import DataItem, RubricDataset
from autorubric.rubric import Rubric
from autorubric.types import Criterion, CriterionOption, CriterionVerdict

REPO_URL = "https://github.com/FrankDengAI/SciFigAlign.git"
COMMIT = "a1f0ce7c2bea5ddd4f380771d3cd477b1ada3ba0"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / ".cache" / "scifigalign"
DEFAULT_OUT_DIR = ROOT / "examples" / "data" / "scifigalign"

# Upstream row files under data/splits/, and the dataset file each one becomes.
SPLITS = ("train", "val", "test")
SOURCE_FILES = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
ALL_SOURCE_FILE = "demo_all.jsonl"
OUTPUT_NAMES = ("train.json", "val.json", "test.json", "all.json")

# The four dimensions, in the order of each item's ground truth (CL, RE, IN, ST).
DIMENSIONS = ("clarity", "relevance", "informativeness", "structure")
SCORES = (1, 2, 3, 4, 5)
# Option values for scores 1 to 5.
OPTION_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)

ROW_KEYS = frozenset(
    {
        "paper_id",
        "figure_id",
        "paper_title",
        "venue",
        "figure_type",
        "figure_image_path",
        "caption_text",
        "citing_paragraphs",
        "abstract",
        "section_title",
        "nearby_context",
        "scores",
        "label_source",
    }
)
# Paper context the demo leaves empty; the dataset has no query, so a non-empty value
# would be dropped without notice.
EMPTY_LIST_FIELDS = ("citing_paragraphs", "nearby_context")
EMPTY_TEXT_FIELDS = ("abstract", "section_title")
SCORE_KEYS = frozenset({"score", "rationale"})
# A paper or figure id becomes part of a file name.
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Bit depths each PNG colour type allows (PNG specification, section 11.2.2).
_PNG_BIT_DEPTHS = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}

GIT_TIMEOUT_SECONDS = 600

# Each criterion's requirement: the dimension's definition in the paper's section 3.1.
REQUIREMENTS = {
    "clarity": (
        "Clarity (CL): whether the figure is clear, readable, and cleanly laid out (fonts, "
        "contrast, axis labels). A figure may be scientifically dense yet still score poorly "
        "here if clutter or rendering artefacts obstruct reading."
    ),
    "relevance": (
        "Relevance (RE): whether the figure matches the caption, citing paragraphs, and the "
        "paper's experimental narrative. High-relevance figures do argumentative work; "
        "low-relevance ones feel incidental or decorative."
    ),
    "informativeness": (
        "Informativeness (IN): how much the figure contributes to the paper's argument "
        "(comparisons, curves, method details vs. removable schematics)."
    ),
    "structure": "Structure (ST): subfigure organization, reading path, and visual hierarchy.",
}

# Each score's descriptor: the paper's Table 2, followed at scores 1, 3 and 5 by the
# sentence the human annotators used as a rationale template for that tier, where one exists.
DESCRIPTORS = {
    "clarity": {
        1: (
            "Most content is illegible. The large area is unclear, and key elements are "
            "difficult to identify."
        ),
        2: "Partially readable, with difficulty.",
        3: (
            "Readable but with noticeable issues. A little area is unclear, and key elements "
            "are difficult to identify."
        ),
        4: "Mostly clear, minor defects.",
        5: (
            "Fully clear and easy to read. Almost all elements are clear and easy to read, "
            "allowing readers to quickly identify the content in the diagram."
        ),
    },
    "relevance": {
        1: "Clearly unrelated to paper.",
        2: "Weakly related.",
        3: (
            "Partially relevant. The relevant parts show a connection, but their supporting "
            "role is limited."
        ),
        4: "Well aligned with text.",
        5: (
            "Highly consistent with paper. Highly relevant, closely aligned with the caption "
            "and main text content, with images and text supporting each other."
        ),
    },
    "informativeness": {
        1: (
            "Negligible information content. There is very little information; it is hardly "
            "helpful for understanding."
        ),
        2: "Limited useful content.",
        3: (
            "Moderate information. There is some information, but it is incomplete, or valid "
            "information coexists with redundant information."
        ),
        4: "Informative.",
        5: (
            "Rich and effective scientific content. The information is comprehensive and "
            "effective, with no obvious redundancy, clearly conveys the core content, and "
            "significantly aids understanding."
        ),
    },
    "structure": {
        1: (
            "Chaotic layout, no clear reading path. The structure is chaotic, the reading path "
            "is unclear, and the module relationships are difficult to determine."
        ),
        2: "Poorly organised.",
        3: (
            "Basically reasonable structure. The structure is generally followable, but the "
            "sense of hierarchy or organization is average."
        ),
        4: "Logical hierarchy.",
        5: (
            "Clear layout, logical flow. The hierarchy is clear, the layout is logical, and the "
            "processes or relationships are easily understandable."
        ),
    },
}

GUIDELINES = """\
You are a peer reviewer assessing one figure from a paper published at ICLR 2024. The \
submission is the figure as it was automatically cropped from the paper's PDF, followed by \
the figure's caption.

Score each of the four dimensions on its own. A polished figure can still carry little \
information, and a dense, informative figure can still be hard to read.

Only the figure and its caption are available to you. The human annotators also read the \
paragraphs of the paper that cite the figure, so judge Relevance and Informativeness against \
the caption and the figure itself.

A caption longer than 300 characters is cut off there, and a caption may contain artefacts \
of extracting text from a PDF, such as broken words or stray symbols. Do not penalize the \
figure for them.

If the crop is incomplete or does not show a figure, score what it shows. The annotators \
scored such crops low rather than skipping them.

Judge legibility as a reader of the printed paper would see the figure, at the size it has \
in the paper."""


# =============================================================================
# Rubric
# =============================================================================


def option_label(dimension: str, score: int) -> str:
    """The label of ``dimension``'s option for ``score``: ``"Score k: <descriptor>"``.

    The descriptor is in the label because an option has no other text the judge sees.

    Raises:
        KeyError: If ``dimension`` is not one of ``DIMENSIONS`` or ``score`` not 1 to 5.
    """
    return f"Score {score}: {DESCRIPTORS[dimension][score]}"


def build_rubric() -> Rubric:
    """The SciFigAlign rubric: four equally weighted ordinal criteria with guidelines.

    Each criterion is named after its dimension, has the paper's definition as its
    requirement and five options, scores 1 to 5 valued 0, 0.25, 0.5, 0.75 and 1. There is
    no NA option, because the annotators had none.
    """
    criteria = [
        Criterion(
            name=dimension,
            weight=1.0,
            requirement=REQUIREMENTS[dimension],
            options=[
                CriterionOption(label=option_label(dimension, score), value=value)
                for score, value in zip(SCORES, OPTION_VALUES, strict=True)
            ],
            scale_type="ordinal",
        )
        for dimension in DIMENSIONS
    ]
    return Rubric(criteria, guidelines=GUIDELINES)


# =============================================================================
# Validation
# =============================================================================


def score_to_int(value: object) -> int:
    """A score as the integer 1 to 5 it holds.

    An integer, or a float with an integral value (upstream writes ``4.0``), is accepted.

    Raises:
        ValueError: For a ``bool``, a string, NaN, infinity, a fractional value such as
            4.5, a value outside 1 to 5, or any other type.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"score must be a number from 1 to 5, got {value!r}")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"score must be a whole number from 1 to 5, got {value!r}")
    score = int(value)
    if score not in SCORES:
        raise ValueError(f"score must be from 1 to 5, got {value!r}")
    return score


def _row_name(row: Mapping[str, Any], where: str) -> str:
    """``where``, followed by the row's paper and figure ids when they are strings."""
    paper_id, figure_id = row.get("paper_id"), row.get("figure_id")
    if isinstance(paper_id, str) and isinstance(figure_id, str):
        return f"{where} ({paper_id} {figure_id})"
    return where


def validate_row(row: Any, where: str) -> dict[str, Any]:
    """Check one upstream row and return it.

    A row has exactly the keys of ``ROW_KEYS``; non-empty string ids, title, venue and
    figure type, the ids safe in a file name; ``label_source`` ``"human"``; a caption that
    is not blank; empty paper context (``citing_paragraphs``, ``nearby_context``,
    ``abstract``, ``section_title``); ``scores`` with exactly the four dimensions, each
    ``{"score": <1 to 5>, "rationale": <str>}`` (``score_to_int``); and the image path
    ``figures/<paper_id>__<figure_id>.png``.

    Args:
        row: The parsed JSON value of one line of a row file.
        where: The row's place, for messages (e.g. ``"train.jsonl line 3"``).

    Raises:
        ValueError: Naming the row and what is wrong with it.
    """
    if not isinstance(row, dict):
        raise ValueError(f"{where}: a row must be a JSON object, got {type(row).__name__}")
    name = _row_name(row, where)
    missing, extra = ROW_KEYS - row.keys(), row.keys() - ROW_KEYS
    if missing or extra:
        raise ValueError(
            f"{name}: unexpected keys (missing {sorted(missing)}, extra {sorted(extra)})"
        )
    for key in ("paper_id", "figure_id", "paper_title", "venue", "figure_type"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"{name}: {key} must be a non-empty string, got {row[key]!r}")
    for key in ("paper_id", "figure_id"):
        if not _ID_PATTERN.fullmatch(row[key]):
            raise ValueError(f"{name}: {key} {row[key]!r} is not usable in a file name")
    if row["label_source"] != "human":
        raise ValueError(f"{name}: label_source must be 'human', got {row['label_source']!r}")
    caption = row["caption_text"]
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"{name}: caption_text must be non-blank text, got {caption!r}")
    for key in EMPTY_LIST_FIELDS:
        if row[key] != []:
            raise ValueError(f"{name}: {key} must be an empty list, got {row[key]!r}")
    for key in EMPTY_TEXT_FIELDS:
        if row[key] != "":
            raise ValueError(f"{name}: {key} must be empty text, got {row[key]!r}")

    scores = row["scores"]
    if not isinstance(scores, dict) or scores.keys() != set(DIMENSIONS):
        found = sorted(scores) if isinstance(scores, dict) else scores
        raise ValueError(f"{name}: scores must hold exactly {list(DIMENSIONS)}, got {found!r}")
    for dimension in DIMENSIONS:
        entry = scores[dimension]
        if not isinstance(entry, dict) or entry.keys() != SCORE_KEYS:
            raise ValueError(
                f"{name}: scores[{dimension!r}] must be an object with keys "
                f"{sorted(SCORE_KEYS)}, got {entry!r}"
            )
        try:
            score_to_int(entry["score"])
        except ValueError as error:
            raise ValueError(f"{name}: scores[{dimension!r}]: {error}") from None
        if not isinstance(entry["rationale"], str):
            rationale = entry["rationale"]
            raise ValueError(
                f"{name}: scores[{dimension!r}] rationale must be text, got {rationale!r}"
            )

    expected_path, path = f"figures/{figure_name(row)}", row["figure_image_path"]
    if path != expected_path:
        raise ValueError(f"{name}: figure_image_path must be {expected_path!r}, got {path!r}")
    return row


def figure_name(row: Mapping[str, Any]) -> str:
    """The file name of a row's figure: ``<paper_id>__<figure_id>.png``."""
    return f"{row['paper_id']}__{row['figure_id']}.png"


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return row["paper_id"], row["figure_id"]


def _keyed_rows(rows: Sequence[Mapping[str, Any]], name: str) -> dict[tuple[str, str], Any]:
    """The rows by ``(paper_id, figure_id)``; a key repeated in ``rows`` raises."""
    keyed: dict[tuple[str, str], Any] = {}
    for row in rows:
        key = _row_key(row)
        if key in keyed:
            raise ValueError(f"{name}: figure {key[0]} {key[1]} appears more than once")
        keyed[key] = row
    return keyed


def validate_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    all_rows: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> None:
    """Check the splits against each other, the full row list and the corpus metadata.

    Each split has the number of rows ``demo_corpus_meta.json`` gives, and the full list
    ``n_figures`` rows from ``n_papers`` papers. No ``(paper_id, figure_id)`` repeats within
    a list, the splits share none, and together they hold exactly the full list's rows, each
    equal to its row there.

    Args:
        splits: The validated rows of each of ``SPLITS``.
        all_rows: The validated rows of ``demo_all.jsonl``.
        meta: The parsed ``demo_corpus_meta.json``.

    Raises:
        ValueError: Naming the split or row and what is wrong.
    """
    if set(splits) != set(SPLITS):
        raise ValueError(f"expected the splits {list(SPLITS)}, got {sorted(splits)}")
    expected_counts = {**{split: split for split in SPLITS}, "all": "n_figures"}
    for name, key in expected_counts.items():
        expected = meta.get(key)
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError(f"demo_corpus_meta.json: {key} must be an integer, got {expected!r}")
        rows = all_rows if name == "all" else splits[name]
        if len(rows) != expected:
            raise ValueError(
                f"{name}: expected {expected} rows ({key} in metadata), got {len(rows)}"
            )
    n_papers = meta.get("n_papers")
    papers = {row["paper_id"] for row in all_rows}
    if len(papers) != n_papers:
        raise ValueError(f"all: expected {n_papers!r} papers (n_papers), got {len(papers)}")

    keyed_all = _keyed_rows(all_rows, "all")
    seen: dict[tuple[str, str], str] = {}
    for split in SPLITS:
        for key, row in _keyed_rows(splits[split], split).items():
            if key in seen:
                raise ValueError(
                    f"{split}: figure {key[0]} {key[1]} is also in the {seen[key]} split"
                )
            seen[key] = split
            if key not in keyed_all:
                raise ValueError(f"{split}: figure {key[0]} {key[1]} is not in demo_all")
            if row != keyed_all[key]:
                raise ValueError(
                    f"{split}: figure {key[0]} {key[1]} differs from its row in demo_all"
                )
    missing = sorted(set(keyed_all) - set(seen))
    if missing:
        names = ", ".join(f"{paper} {figure}" for paper, figure in missing[:5])
        raise ValueError(f"all: {len(missing)} figures are in no split (first: {names})")


def check_png(path: Path) -> tuple[int, int]:
    """Check that ``path`` is a regular file starting like a PNG; return its size.

    The file must not be a symbolic link, and must start with the PNG signature and a
    well-formed ``IHDR`` chunk: length 13, positive width and height, and a colour type and
    bit depth the PNG specification allows.

    Returns:
        ``(width, height)`` in pixels.

    Raises:
        ValueError: Naming the file and what is wrong with it.
    """
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ValueError(f"{path}: cannot be read ({error})") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"{path}: not a regular file")
    with open(path, "rb") as file:
        header = file.read(33)
    if len(header) < 33 or header[:8] != _PNG_SIGNATURE:
        raise ValueError(f"{path}: not a PNG file (no PNG signature)")
    length, chunk_type, width, height, bit_depth, colour_type = struct.unpack(
        ">I4sIIBB", header[8:26]
    )
    if length != 13 or chunk_type != b"IHDR":
        raise ValueError(f"{path}: the PNG does not start with an IHDR chunk")
    if width == 0 or height == 0:
        raise ValueError(f"{path}: the PNG has size {width}x{height}")
    if bit_depth not in _PNG_BIT_DEPTHS.get(colour_type, set()):
        raise ValueError(
            f"{path}: the PNG has colour type {colour_type} with bit depth {bit_depth}"
        )
    return width, height


# =============================================================================
# Items
# =============================================================================


def make_description(row: Mapping[str, Any]) -> str:
    """An item's description.

    ``"<paper_id> <figure_id> (<figure_type>, <venue>, <label_source>)"``.
    """
    return (
        f"{row['paper_id']} {row['figure_id']} "
        f"({row['figure_type']}, {row['venue']}, {row['label_source']})"
    )


def row_to_item(row: Mapping[str, Any], figures_dir: Path) -> DataItem:
    """The dataset item of a validated row whose figure is in ``figures_dir``.

    The submission is the figure followed by the caption, verbatim. The ground truth is
    the option label of each dimension's score, in the order of ``DIMENSIONS``, and the
    ground-truth reasons the rationales, stripped, aligned with it (``None`` for a blank
    one), or ``None`` when every rationale is blank. No file is read.
    """
    scores = row["scores"]
    labels: list[CriterionVerdict | str] = [
        option_label(dimension, score_to_int(scores[dimension]["score"]))
        for dimension in DIMENSIONS
    ]
    reasons = [scores[dimension]["rationale"].strip() or None for dimension in DIMENSIONS]
    return DataItem(
        submission=[Image(figures_dir / figure_name(row)), row["caption_text"]],
        description=make_description(row),
        ground_truth=labels,
        ground_truth_reasons=reasons if any(reasons) else None,
    )


# =============================================================================
# Fetching
# =============================================================================


def git_fetch_commands(
    cache_dir: Path, repo_url: str = REPO_URL, commit: str = COMMIT
) -> list[list[str]]:
    """The git commands that fetch ``data/`` of ``repo_url`` at ``commit`` into ``cache_dir``.

    A new repository with a sparse checkout of ``data/``, then a shallow (depth 1),
    blob-less fetch of the one commit, checked out detached. Blobs outside the sparse
    checkout are not downloaded.
    """
    git = ["git", "-C", str(cache_dir)]
    return [
        ["git", "init", "--quiet", str(cache_dir)],
        [*git, "remote", "add", "origin", repo_url],
        [*git, "sparse-checkout", "set", "data"],
        [*git, "fetch", "--depth", "1", "--filter=blob:none", "origin", commit],
        [*git, "checkout", "--quiet", "--detach", "FETCH_HEAD"],
    ]


def _git_environment() -> dict[str, str]:
    """The environment git runs in: never prompting for credentials."""
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def _git_output(cache_dir: Path, *args: str) -> str | None:
    """The output of ``git -C cache_dir <args>``, or ``None`` when git fails."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cache_dir), *args],
            check=True,
            capture_output=True,
            text=True,
            env=_git_environment(),
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def _is_pinned_checkout(cache_dir: Path) -> bool:
    """Whether ``cache_dir`` is a clean checkout of ``COMMIT`` (its own repository)."""
    if not (cache_dir / ".git").is_dir():
        return False
    return (
        _git_output(cache_dir, "rev-parse", "HEAD") == COMMIT
        and _git_output(cache_dir, "status", "--porcelain") == ""
    )


def fetch(cache_dir: Path, *, force: bool) -> Path:
    """Fetch the pinned commit into ``cache_dir`` unless it is already there.

    A cache that is a clean checkout of the pinned commit is reused. Any other existing
    cache is replaced only with ``force``, and only when it is a clone of ``REPO_URL``, so
    a directory named by mistake is never deleted.

    Returns:
        The fetched ``data/`` directory.
    """
    if _is_pinned_checkout(cache_dir):
        print(f"Reusing {cache_dir} at {COMMIT}")
        return cache_dir / "data"
    if cache_dir.exists():
        is_clone = (cache_dir / ".git").is_dir() and (
            _git_output(cache_dir, "remote", "get-url", "origin") == REPO_URL
        )
        if not (force and is_clone):
            raise SystemExit(
                f"{cache_dir} exists but is not a clean checkout of {COMMIT}. "
                + (
                    "Pass --force to fetch it again."
                    if is_clone
                    else "It is not a clone of the SciFigAlign repository; remove it or "
                    "choose another --cache-dir."
                )
            )
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching data/ of {REPO_URL} at {COMMIT} into {cache_dir}")
    for command in git_fetch_commands(cache_dir):
        subprocess.run(command, check=True, env=_git_environment(), timeout=GIT_TIMEOUT_SECONDS)
    if not _is_pinned_checkout(cache_dir):
        raise SystemExit(f"{cache_dir} is not a clean checkout of {COMMIT} after fetching")
    return cache_dir / "data"


# =============================================================================
# Reading, writing and checking
# =============================================================================


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The validated rows of a JSON Lines row file (blank lines skipped)."""
    rows = []
    with open(path, encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            where = f"{path.name} line {number}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: not valid JSON ({error})") from None
            rows.append(validate_row(value, where))
    return rows


def check_figures(figures_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Check that ``figures_dir`` holds exactly the rows' figures, each a valid PNG."""
    expected = {figure_name(row) for row in rows}
    present = {entry.name for entry in figures_dir.iterdir()}
    if present != expected:
        missing, extra = sorted(expected - present), sorted(present - expected)
        raise ValueError(f"{figures_dir}: missing {missing[:5]}, unexpected {extra[:5]}")
    for name in sorted(expected):
        check_png(figures_dir / name)


def _clear_output(out_dir: Path, *, force: bool) -> None:
    """Remove the output a previous run wrote (``figures/`` and the four JSON files).

    Raises:
        SystemExit: If there is such output and ``force`` is false.
    """
    figures = out_dir / "figures"
    existing = [out_dir / name for name in OUTPUT_NAMES if (out_dir / name).exists()]
    if figures.exists() or figures.is_symlink():
        existing.append(figures)
    if not existing:
        return
    if not force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"{out_dir} already holds {names}. Pass --force to replace them.")
    for path in existing:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _same_rubric(left: Rubric, right: Rubric) -> bool:
    return left.rubric == right.rubric and left.guidelines == right.guidelines


def verify_output(out_dir: Path, rubric: Rubric, datasets: Mapping[str, RubricDataset]) -> None:
    """Load each written file and check it against the dataset it was written from.

    Each file loads with ``RubricDataset.from_file`` to the same items and the same rubric,
    every image part names a PNG file in ``figures/``, and every ground-truth label names an
    option of its criterion (``Criterion.find_option_by_label``).

    Raises:
        ValueError: Naming the file and what does not match.
    """
    figures = (out_dir / "figures").absolute()
    for split, written in datasets.items():
        path = out_dir / f"{split}.json"
        loaded = RubricDataset.from_file(path)
        if loaded.name != written.name or loaded.items != written.items:
            raise ValueError(f"{path}: does not load back to the items written")
        if loaded.rubric is None or not _same_rubric(loaded.rubric, rubric):
            raise ValueError(f"{path}: does not load back to the rubric written")
        for index, item in enumerate(loaded.items):
            parts = item.submission
            if not (
                isinstance(parts, list)
                and len(parts) == 2
                and isinstance(parts[0], Image)
                and parts[0].path is not None
                and parts[0].path.parent == figures
                and isinstance(parts[1], str)
            ):
                raise ValueError(f"{path}: item {index}'s submission is not a figure and caption")
            image_path = parts[0].path
            check_png(image_path)
            if parts[0].media_type != "image/png":
                raise ValueError(f"{path}: item {index}'s figure is not read as a PNG")
            labels = item.ground_truth or []
            for criterion, label in zip(loaded.rubric.rubric, labels, strict=True):
                criterion.find_option_by_label(str(label))


def print_histograms(datasets: Mapping[str, RubricDataset]) -> None:
    """Print, for each split, how many items have each score on each dimension."""
    print(f"\n{'split':<6} {'dimension':<16} " + " ".join(f"{s:>4}" for s in SCORES))
    for split, dataset in datasets.items():
        criteria = dataset.rubric.rubric if dataset.rubric is not None else []
        for index, criterion in enumerate(criteria):
            counts = Counter(
                SCORES[criterion.find_option_by_label(str(item.ground_truth[index]))]
                for item in dataset.items
                if item.ground_truth is not None
            )
            row = " ".join(f"{counts[s]:>4}" for s in SCORES)
            print(f"{split:<6} {criterion.name or '':<16} {row}")


def convert(cache_dir: Path, out_dir: Path, *, force: bool) -> dict[str, RubricDataset]:
    """Fetch, check, copy the figures, write the four dataset files and check them."""
    data_dir = fetch(cache_dir, force=force)
    meta = json.loads((data_dir / "demo_corpus_meta.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("demo_corpus_meta.json: must be a JSON object")
    splits = {split: read_rows(data_dir / "splits" / SOURCE_FILES[split]) for split in SPLITS}
    all_rows = read_rows(data_dir / "splits" / ALL_SOURCE_FILE)
    validate_splits(splits, all_rows, meta)
    check_figures(data_dir / "figures", all_rows)

    figures_dir = out_dir.absolute() / "figures"
    rubric = build_rubric()
    datasets = {
        split: RubricDataset(
            name=f"scifigalign-demo-{split}",
            # A dataset needs a prompt; an empty one gives the judge no <input> block, as the
            # demo has no paper context to put there.
            prompt="",
            rubric=rubric,
            items=[row_to_item(row, figures_dir) for row in rows],
        )
        for split, rows in {**splits, "all": all_rows}.items()
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_output(out_dir, force=force)
    figures_dir.mkdir()
    for row in all_rows:
        name = figure_name(row)
        shutil.copyfile(data_dir / "figures" / name, figures_dir / name)

    for split, dataset in datasets.items():
        dataset.to_file(out_dir.absolute() / f"{split}.json")
    verify_output(out_dir, rubric, datasets)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the SciFigAlign human-rated demo to AutoRubric dataset files."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing output and a cache that is not the pinned checkout",
    )
    args = parser.parse_args()
    datasets = convert(args.cache_dir, args.out_dir, force=args.force)
    for split, dataset in datasets.items():
        print(f"Wrote {args.out_dir / f'{split}.json'} ({len(dataset)} items)")
    print_histograms(datasets)


if __name__ == "__main__":
    main()
