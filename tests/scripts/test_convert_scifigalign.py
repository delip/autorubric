"""Tests for ``scripts/convert_scifigalign.py``, the SciFigAlign demo converter.

The script is not part of the package, so it is loaded from its file. Every test runs on
synthetic rows and a tiny PNG written to ``tmp_path``: nothing is fetched and no SciFigAlign
file is read.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from autorubric import CriterionVerdict, Image, RubricDataset

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "convert_scifigalign.py"


def _load_converter() -> ModuleType:
    name = "convert_scifigalign"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONVERT = _load_converter()

# A 2x1 RGB PNG (colour type 2, bit depth 8), as Pillow writes it.
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"{@\xe8\xdd\x00\x00\x00\x0fIDATx\x9cc\xfc\xff\xff?\x03\x03\x03\x00\x0e\xff\x02\xff\xb8"
    b"\xc5\x9e\x99\x00\x00\x00\x00IEND\xaeB`\x82"
)

CAPTION = "Figure 2: Accuracy of the three models on the held-out set."


def make_row(paper: str = "ICLR_2024_00001", figure: str = "fig_02", **changes: Any) -> dict:
    """A valid upstream row; ``changes`` replace its fields."""
    row: dict[str, Any] = {
        "paper_id": paper,
        "figure_id": figure,
        "paper_title": "ICLR_2024",
        "venue": "ICLR_2024",
        "figure_type": "other",
        "figure_image_path": f"figures/{paper}__{figure}.png",
        "caption_text": CAPTION,
        "citing_paragraphs": [],
        "abstract": "",
        "section_title": "",
        "nearby_context": [],
        "scores": {
            "clarity": {"score": 4.0, "rationale": ""},
            "relevance": {"score": 5.0, "rationale": "  Closely aligned with the caption.  "},
            "informativeness": {"score": 3.0, "rationale": ""},
            "structure": {"score": 1.0, "rationale": " "},
        },
        "label_source": "human",
    }
    row.update(changes)
    return row


def with_scores(row: dict, dimension: str, entry: Any) -> dict:
    """``row`` with ``scores[dimension]`` replaced by ``entry``."""
    changed = copy.deepcopy(row)
    changed["scores"][dimension] = entry
    return changed


# =============================================================================
# The rubric
# =============================================================================


class TestBuildRubric:
    def test_four_equally_weighted_ordinal_criteria_in_dimension_order(self):
        rubric = CONVERT.build_rubric()
        assert [c.name for c in rubric.rubric] == [
            "clarity",
            "relevance",
            "informativeness",
            "structure",
        ]
        for criterion in rubric.rubric:
            assert criterion.weight == 1.0
            assert criterion.scale_type == "ordinal"
            assert criterion.requirement == CONVERT.REQUIREMENTS[criterion.name]

    def test_each_requirement_is_the_papers_section_3_1_definition(self):
        """Typed from the paper's Section 3.1 (arXiv:2607.27066), its typographic
        apostrophes written as ASCII."""
        assert CONVERT.REQUIREMENTS == {
            "clarity": (
                "Clarity (CL): whether the figure is clear, readable, and cleanly laid out "
                "(fonts, contrast, legends, axis labels). A figure may be scientifically dense "
                "yet still score poorly here if clutter or rendering artefacts obstruct reading."
            ),
            "relevance": (
                "Relevance (RE): whether the figure matches the caption, citing paragraphs, and "
                "the paper's experimental narrative. High-relevance figures do argumentative "
                "work; low-relevance ones feel incidental or decorative."
            ),
            "informativeness": (
                "Informativeness (IN): how much the figure contributes to the paper's argument "
                "(comparisons, curves, method details vs. removable schematics)."
            ),
            "structure": (
                "Structure (ST): subfigure organization, reading path, and visual hierarchy."
            ),
        }

    def test_five_options_labelled_by_score_and_valued_0_to_1(self):
        for criterion in CONVERT.build_rubric().rubric:
            assert criterion.options is not None
            assert [o.label for o in criterion.options] == [
                CONVERT.option_label(criterion.name, score) for score in (1, 2, 3, 4, 5)
            ]
            assert [o.value for o in criterion.options] == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_no_option_is_an_na_option(self):
        for criterion in CONVERT.build_rubric().rubric:
            assert criterion.na_option_index is None
            assert not any(o.na for o in criterion.options or [])

    def test_the_guidelines_are_the_scripts(self):
        rubric = CONVERT.build_rubric()
        assert rubric.guidelines == CONVERT.GUIDELINES
        assert "300 characters" in rubric.guidelines

    def test_a_label_starts_with_its_score_and_holds_its_descriptor(self):
        label = CONVERT.option_label("relevance", 5)
        assert label.startswith("Score 5: Highly consistent with paper.")
        assert label == f"Score 5: {CONVERT.DESCRIPTORS['relevance'][5]}"

    def test_every_label_resolves_to_its_scores_option(self):
        for criterion in CONVERT.build_rubric().rubric:
            for score in (1, 2, 3, 4, 5):
                label = CONVERT.option_label(criterion.name, score)
                assert criterion.find_option_by_label(label) == score - 1

    @pytest.mark.parametrize(
        ("dimension", "score"), [("colour", 3), ("clarity", 0), ("clarity", 6)]
    )
    def test_an_unknown_dimension_or_score_has_no_label(self, dimension, score):
        with pytest.raises(KeyError):
            CONVERT.option_label(dimension, score)


# =============================================================================
# Scores
# =============================================================================


class TestScoreToInt:
    @pytest.mark.parametrize(("value", "expected"), [(1, 1), (5, 5), (4.0, 4), (1.0, 1)])
    def test_accepts_whole_numbers_from_1_to_5(self, value, expected):
        score = CONVERT.score_to_int(value)
        assert score == expected and type(score) is int

    @pytest.mark.parametrize(
        "value", [True, False, "4", 4.5, math.nan, math.inf, 0, 6, -1, 0.0, None, [4]]
    )
    def test_rejects_anything_else(self, value):
        with pytest.raises(ValueError, match="score must be"):
            CONVERT.score_to_int(value)


# =============================================================================
# Rows
# =============================================================================


class TestValidateRow:
    def test_a_valid_row_is_returned(self):
        row = make_row()
        assert CONVERT.validate_row(row, "test.jsonl line 1") is row

    def test_a_row_that_is_not_an_object(self):
        with pytest.raises(ValueError, match="test.jsonl line 1: a row must be a JSON object"):
            CONVERT.validate_row(["paper"], "test.jsonl line 1")

    def test_a_missing_key(self):
        row = make_row()
        del row["abstract"]
        with pytest.raises(ValueError, match=r"missing \['abstract'\]"):
            CONVERT.validate_row(row, "w")

    def test_an_extra_key(self):
        with pytest.raises(ValueError, match=r"extra \['doi'\]"):
            CONVERT.validate_row(make_row(doi="10.1/x"), "w")

    def test_the_message_names_the_row(self):
        with pytest.raises(ValueError, match=r"^train.jsonl line 7 \(ICLR_2024_00001 fig_02\)"):
            CONVERT.validate_row(make_row(doi="x"), "train.jsonl line 7")

    def test_a_missing_dimension(self):
        row = make_row()
        del row["scores"]["structure"]
        with pytest.raises(ValueError, match="scores must hold exactly"):
            CONVERT.validate_row(row, "w")

    def test_an_extra_dimension(self):
        row = with_scores(make_row(), "aesthetics", {"score": 3, "rationale": ""})
        with pytest.raises(ValueError, match="scores must hold exactly"):
            CONVERT.validate_row(row, "w")

    def test_scores_that_are_not_an_object(self):
        with pytest.raises(ValueError, match="scores must hold exactly"):
            CONVERT.validate_row(make_row(scores=[4, 4, 4, 4]), "w")

    def test_a_score_entry_with_other_keys(self):
        row = with_scores(make_row(), "clarity", {"score": 4, "rationale": "", "rater": "a"})
        with pytest.raises(ValueError, match=r"scores\['clarity'\] must be an object"):
            CONVERT.validate_row(row, "w")

    def test_a_fractional_score(self):
        row = with_scores(make_row(), "informativeness", {"score": 3.5, "rationale": ""})
        with pytest.raises(ValueError, match=r"scores\['informativeness'\]: score must be"):
            CONVERT.validate_row(row, "w")

    def test_a_rationale_that_is_not_text(self):
        row = with_scores(make_row(), "clarity", {"score": 4, "rationale": None})
        with pytest.raises(ValueError, match="rationale must be text"):
            CONVERT.validate_row(row, "w")

    @pytest.mark.parametrize(
        "path",
        [
            "figures/ICLR_2024_00001_fig_02.png",
            "figures/ICLR_2024_00001__fig_02.jpg",
            "ICLR_2024_00001__fig_02.png",
            "/data/figures/ICLR_2024_00001__fig_02.png",
            "figures/../figures/ICLR_2024_00001__fig_02.png",
            "../figures/ICLR_2024_00001__fig_02.png",
        ],
    )
    def test_an_image_path_other_than_the_figures_name(self, path):
        with pytest.raises(ValueError, match="figure_image_path must be"):
            CONVERT.validate_row(make_row(figure_image_path=path), "w")

    @pytest.mark.parametrize("paper", ["../x", "/abs", "a/b", "", "_lead", "a b"])
    def test_an_id_unusable_in_a_file_name(self, paper):
        with pytest.raises(ValueError, match="paper_id"):
            CONVERT.validate_row(make_row(paper=paper), "w")

    @pytest.mark.parametrize("caption", ["", "   \n", None, 7])
    def test_a_blank_or_missing_caption(self, caption):
        with pytest.raises(ValueError, match="caption_text must be non-blank text"):
            CONVERT.validate_row(make_row(caption_text=caption), "w")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("citing_paragraphs", ["As Figure 2 shows, ..."]),
            ("nearby_context", ["Results."]),
            ("abstract", "We study ..."),
            ("section_title", "Experiments"),
        ],
    )
    def test_paper_context_that_is_not_empty(self, field, value):
        with pytest.raises(ValueError, match=f"{field} must be"):
            CONVERT.validate_row(make_row(**{field: value}), "w")

    @pytest.mark.parametrize("source", ["gpt", "gpt4o", "", None])
    def test_a_label_source_other_than_human(self, source):
        with pytest.raises(ValueError, match="label_source must be 'human'"):
            CONVERT.validate_row(make_row(label_source=source), "w")

    @pytest.mark.parametrize("field", ["paper_title", "venue", "figure_type"])
    def test_an_empty_text_field(self, field):
        with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
            CONVERT.validate_row(make_row(**{field: ""}), "w")


# =============================================================================
# Splits
# =============================================================================


def make_splits() -> tuple[dict[str, list[dict]], list[dict], dict[str, Any]]:
    """Valid splits of four rows (two train, one val, one test), each from its own paper."""
    rows = [make_row(paper=f"ICLR_2024_0000{i}") for i in range(4)]
    splits = {"train": rows[:2], "val": rows[2:3], "test": rows[3:]}
    meta = {"n_figures": 4, "n_papers": 4, "train": 2, "val": 1, "test": 1}
    return splits, [copy.deepcopy(row) for row in rows], meta


class TestValidateSplits:
    def test_valid_splits_pass(self):
        CONVERT.validate_splits(*make_splits())

    def test_a_missing_split(self):
        splits, all_rows, meta = make_splits()
        del splits["val"]
        with pytest.raises(ValueError, match="expected the splits"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_split_whose_count_differs_from_the_metadata(self):
        splits, all_rows, meta = make_splits()
        meta["train"] = 3
        with pytest.raises(ValueError, match=r"train: expected 3 rows \(train in metadata\)"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_full_list_whose_count_differs_from_the_metadata(self):
        splits, all_rows, meta = make_splits()
        meta["n_figures"] = 5
        with pytest.raises(ValueError, match="all: expected 5 rows"):
            CONVERT.validate_splits(splits, all_rows, meta)

    @pytest.mark.parametrize("count", [None, "2", 2.0, True])
    def test_a_metadata_count_that_is_not_an_integer(self, count):
        splits, all_rows, meta = make_splits()
        meta["val"] = count
        with pytest.raises(ValueError, match="val must be an integer"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_paper_count_that_differs_from_the_metadata(self):
        splits, all_rows, meta = make_splits()
        meta["n_papers"] = 3
        with pytest.raises(ValueError, match="expected 3 papers"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_figure_repeated_within_a_split(self):
        splits, all_rows, meta = make_splits()
        splits["train"] = [splits["train"][0], splits["train"][0]]
        with pytest.raises(ValueError, match="train: figure ICLR_2024_00000 fig_02 appears more"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_figure_repeated_in_the_full_list(self):
        splits, all_rows, meta = make_splits()
        all_rows[3] = all_rows[0]
        meta["n_papers"] = 3
        with pytest.raises(ValueError, match="all: figure ICLR_2024_00000 fig_02 appears more"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_figure_in_two_splits(self):
        splits, all_rows, meta = make_splits()
        splits["test"] = [splits["train"][0]]
        with pytest.raises(ValueError, match="test: figure ICLR_2024_00000 fig_02 is also in"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_split_figure_missing_from_the_full_list(self):
        splits, all_rows, meta = make_splits()
        splits["test"] = [make_row(paper="ICLR_2024_09999")]
        with pytest.raises(ValueError, match="is not in demo_all"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_split_row_that_differs_from_its_full_list_row(self):
        splits, all_rows, meta = make_splits()
        all_rows[2]["caption_text"] = "Another caption."
        with pytest.raises(ValueError, match="differs from its row in demo_all"):
            CONVERT.validate_splits(splits, all_rows, meta)

    def test_a_full_list_figure_in_no_split(self):
        splits, all_rows, meta = make_splits()
        all_rows[3] = make_row(paper="ICLR_2024_09999")
        splits["test"] = [make_row(paper="ICLR_2024_00003")]
        all_rows.append(make_row(paper="ICLR_2024_00003"))
        meta.update(n_figures=5, n_papers=5)
        with pytest.raises(ValueError, match="all: 1 figures are in no split"):
            CONVERT.validate_splits(splits, all_rows, meta)


# =============================================================================
# Items
# =============================================================================


class TestRowToItem:
    def test_the_description_names_the_figure_and_its_provenance(self, tmp_path):
        item = CONVERT.row_to_item(make_row(), tmp_path)
        assert item.description == "ICLR_2024_00001 fig_02 (other, ICLR_2024, human)"

    def test_the_submission_is_the_figure_then_the_caption_verbatim(self, tmp_path):
        item = CONVERT.row_to_item(make_row(), tmp_path)
        assert item.submission == [Image(tmp_path / "ICLR_2024_00001__fig_02.png"), CAPTION]

    def test_no_file_is_read(self, tmp_path):
        # The figure does not exist; building the item must not need it.
        item = CONVERT.row_to_item(make_row(), tmp_path / "absent")
        image = item.submission[0]
        assert isinstance(image, Image) and image.path == tmp_path / "absent" / (
            "ICLR_2024_00001__fig_02.png"
        )

    def test_the_labels_are_each_dimensions_option_label_in_order(self, tmp_path):
        item = CONVERT.row_to_item(make_row(), tmp_path)
        assert item.ground_truth == [
            CONVERT.option_label("clarity", 4),
            CONVERT.option_label("relevance", 5),
            CONVERT.option_label("informativeness", 3),
            CONVERT.option_label("structure", 1),
        ]

    def test_the_reasons_are_the_stripped_rationales_none_where_blank(self, tmp_path):
        item = CONVERT.row_to_item(make_row(), tmp_path)
        assert item.ground_truth_reasons == [None, "Closely aligned with the caption.", None, None]

    def test_no_reasons_when_every_rationale_is_blank(self, tmp_path):
        row = with_scores(make_row(), "relevance", {"score": 5, "rationale": ""})
        assert CONVERT.row_to_item(row, tmp_path).ground_truth_reasons is None


# =============================================================================
# Writing one item
# =============================================================================


def write_one_item(out_dir: Path) -> RubricDataset:
    """Write the fixture row's figure and a one-item dataset file to ``out_dir``."""
    row = make_row()
    figures = out_dir / "figures"
    figures.mkdir(parents=True)
    (figures / CONVERT.figure_name(row)).write_bytes(TINY_PNG)
    dataset = RubricDataset(
        name="scifigalign-demo-test",
        prompt="",
        rubric=CONVERT.build_rubric(),
        items=[CONVERT.row_to_item(row, figures)],
    )
    dataset.to_file(out_dir / "test.json")
    return dataset


class TestWriting:
    def test_the_file_names_the_figure_relative_to_itself_then_the_caption(self, tmp_path):
        write_one_item(tmp_path)
        raw = json.loads((tmp_path / "test.json").read_text(encoding="utf-8"))
        (item,) = raw["items"]
        assert item["submission"] == [{"image": "figures/ICLR_2024_00001__fig_02.png"}, CAPTION]
        assert raw["rubric"]["guidelines"] == CONVERT.GUIDELINES
        assert item["ground_truth_reasons"] == [
            None,
            "Closely aligned with the caption.",
            None,
            None,
        ]

    def test_the_file_loads_back_to_the_items_and_rubric_written(self, tmp_path):
        written = write_one_item(tmp_path)
        loaded = RubricDataset.from_file(tmp_path / "test.json")
        assert loaded.items == written.items
        assert loaded.rubric is not None
        assert loaded.rubric.rubric == CONVERT.build_rubric().rubric
        assert loaded.rubric.guidelines == CONVERT.GUIDELINES
        submission = loaded.items[0].submission
        assert isinstance(submission, list)
        image = submission[0]
        assert isinstance(image, Image) and image.media_type == "image/png"
        assert image.read_bytes() == TINY_PNG

    def test_the_converters_own_check_accepts_the_file(self, tmp_path):
        written = write_one_item(tmp_path)
        CONVERT.verify_output(tmp_path, CONVERT.build_rubric(), {"test": written})

    def test_every_label_resolves_after_loading(self, tmp_path):
        write_one_item(tmp_path)
        loaded = RubricDataset.from_file(tmp_path / "test.json")
        assert loaded.rubric is not None
        labels = loaded.items[0].ground_truth or []
        indices = [
            criterion.find_option_by_label(str(label))
            for criterion, label in zip(loaded.rubric.rubric, labels, strict=True)
        ]
        assert indices == [3, 4, 2, 0]
        assert not any(isinstance(label, CriterionVerdict) for label in labels)


# =============================================================================
# PNG checks
# =============================================================================


def png_with_ihdr(
    width: int = 2, height: int = 1, bit_depth: int = 8, colour_type: int = 2, length: int = 13
) -> bytes:
    """The PNG signature and an ``IHDR`` chunk with the given fields (CRC not checked)."""
    return (
        TINY_PNG[:8]
        + length.to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([bit_depth, colour_type, 0, 0, 0])
        + b"\x00" * 4
    )


class TestCheckPng:
    def test_returns_the_size_of_a_valid_png(self, tmp_path):
        path = tmp_path / "x.png"
        path.write_bytes(TINY_PNG)
        assert CONVERT.check_png(path) == (2, 1)

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be read"):
            CONVERT.check_png(tmp_path / "absent.png")

    def test_a_symbolic_link(self, tmp_path):
        (tmp_path / "x.png").write_bytes(TINY_PNG)
        (tmp_path / "link.png").symlink_to(tmp_path / "x.png")
        with pytest.raises(ValueError, match="not a regular file"):
            CONVERT.check_png(tmp_path / "link.png")

    def test_a_directory(self, tmp_path):
        (tmp_path / "d.png").mkdir()
        with pytest.raises(ValueError, match="not a regular file"):
            CONVERT.check_png(tmp_path / "d.png")

    @pytest.mark.parametrize("content", [b"", b"GIF89a" + b"\x00" * 40, TINY_PNG[:20]])
    def test_a_file_without_the_signature_and_a_header(self, tmp_path, content):
        path = tmp_path / "x.png"
        path.write_bytes(content)
        with pytest.raises(ValueError, match="not a PNG file"):
            CONVERT.check_png(path)

    def test_a_first_chunk_that_is_not_ihdr(self, tmp_path):
        path = tmp_path / "x.png"
        path.write_bytes(png_with_ihdr(length=12))
        with pytest.raises(ValueError, match="does not start with an IHDR chunk"):
            CONVERT.check_png(path)

    @pytest.mark.parametrize(("width", "height"), [(0, 1), (2, 0)])
    def test_an_empty_size(self, tmp_path, width, height):
        path = tmp_path / "x.png"
        path.write_bytes(png_with_ihdr(width=width, height=height))
        with pytest.raises(ValueError, match="has size"):
            CONVERT.check_png(path)

    @pytest.mark.parametrize(("colour_type", "bit_depth"), [(2, 4), (6, 1), (3, 16), (5, 8)])
    def test_a_bit_depth_its_colour_type_does_not_allow(self, tmp_path, colour_type, bit_depth):
        path = tmp_path / "x.png"
        path.write_bytes(png_with_ihdr(colour_type=colour_type, bit_depth=bit_depth))
        with pytest.raises(ValueError, match=f"colour type {colour_type} with bit depth"):
            CONVERT.check_png(path)


# =============================================================================
# Fetching
# =============================================================================


class TestGitFetchCommands:
    def test_a_sparse_shallow_blobless_fetch_of_the_pinned_commit(self, tmp_path):
        cache = tmp_path / "cache"
        git = ["git", "-C", str(cache)]
        assert CONVERT.git_fetch_commands(cache) == [
            ["git", "init", "--quiet", str(cache)],
            [*git, "remote", "add", "origin", "https://github.com/FrankDengAI/SciFigAlign.git"],
            [*git, "sparse-checkout", "set", "data"],
            [
                *git,
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                "a1f0ce7c2bea5ddd4f380771d3cd477b1ada3ba0",
            ],
            [*git, "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        ]

    def test_the_pin_is_a_full_commit_sha(self):
        assert len(CONVERT.COMMIT) == 40
        assert all(character in "0123456789abcdef" for character in CONVERT.COMMIT)

    def test_the_repository_and_commit_can_be_given(self, tmp_path):
        commands = CONVERT.git_fetch_commands(tmp_path, repo_url="file:///r.git", commit="abc")
        assert commands[1][-1] == "file:///r.git"
        assert commands[3][-1] == "abc"
