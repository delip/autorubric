"""Images in dataset files: how ``RubricDataset`` reads and writes submission content.

A submission, prompt or reference submission is written as a string (text), an image object
``{"image": "<path>"}`` (``"image"`` its only key, a string value), or an array of both (a
document). Any other JSON value loads unchanged, as it did before images existed. Image paths
are POSIX paths, relative to the dataset file's directory; no image file is read at load or
when writing JSON, and a bytes-backed image is saved beside the file by ``to_file`` only.
"""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import PIL.Image as pil_image
import pytest

from autorubric import Criterion, CriterionVerdict, DataItem, Image, Rubric, RubricDataset
from autorubric.content import _content_digest, _normalize_content

ESSAYS = Path(__file__).resolve().parents[2] / "examples" / "data" / "essay_grading_dataset.json"
RUBRIC = Rubric([Criterion(name="legible", weight=1.0, requirement="The figure is legible")])
MET = CriterionVerdict.MET


def encoded(color: str, image_format: str = "PNG") -> bytes:
    """A tiny image of one ``color``, encoded by Pillow."""
    buffer = io.BytesIO()
    pil_image.new("RGB", (2, 2), color).save(buffer, image_format)
    return buffer.getvalue()


def dataset_json(items: list[dict[str, Any]], **fields: Any) -> str:
    """A dataset file's text with the one-criterion rubric and ``items``."""
    data = {
        "prompt": "Rate the figure.",
        **fields,
        "rubric": [{"name": "legible", "weight": 1.0, "requirement": "The figure is legible"}],
        "items": [{"description": "d", **item} for item in items],
    }
    return json.dumps(data)


def one_item(dataset: RubricDataset) -> DataItem:
    (item,) = dataset.items
    return item


# =============================================================================
# Reading
# =============================================================================


class TestShapes:
    def test_a_string_is_text(self, tmp_path):
        item = one_item(RubricDataset.from_json(dataset_json([{"submission": "An answer."}])))
        assert item.submission == "An answer."

    def test_an_image_object_is_an_image(self, tmp_path):
        loaded = RubricDataset.from_json(
            dataset_json([{"submission": {"image": "figures/fig1.png"}}]), base_dir=tmp_path
        )
        assert one_item(loaded).submission == Image(tmp_path / "figures" / "fig1.png")

    def test_an_array_of_strings_and_image_objects_is_a_document(self, tmp_path):
        value = [{"image": "a.png"}, "Figure 1: accuracy.", {"image": "b.png"}]
        loaded = RubricDataset.from_json(dataset_json([{"submission": value}]), base_dir=tmp_path)
        assert one_item(loaded).submission == [
            Image(tmp_path / "a.png"),
            "Figure 1: accuracy.",
            Image(tmp_path / "b.png"),
        ]

    def test_an_array_of_strings_stays_a_list_of_strings(self):
        loaded = RubricDataset.from_json(dataset_json([{"submission": ["one", "two"]}]))
        assert one_item(loaded).submission == ["one", "two"]

    @pytest.mark.parametrize("field", ["prompt", "reference_submission"])
    def test_an_items_prompt_and_reference_take_images(self, tmp_path, field):
        value = [{"image": "q.png"}, "What does the figure show?"]
        loaded = RubricDataset.from_json(
            dataset_json([{"submission": "x", field: value}]), base_dir=tmp_path
        )
        assert getattr(one_item(loaded), field) == [
            Image(tmp_path / "q.png"),
            "What does the figure show?",
        ]

    @pytest.mark.parametrize("field", ["prompt", "reference_submission"])
    def test_the_datasets_prompt_and_reference_take_images(self, tmp_path, field):
        loaded = RubricDataset.from_json(
            dataset_json([{"submission": "x"}], **{field: {"image": "ref.png"}}),
            base_dir=tmp_path,
        )
        assert getattr(loaded, field) == Image(tmp_path / "ref.png")

    @pytest.mark.parametrize(
        "value",
        [
            {"image": "a.png", "caption": "a second key"},
            {"image": 3},
            {"image": None},
            {"picture": "a.png"},
            {},
            3,
            2.5,
            True,
            [],
            ["text", 3],
            ["text", ["nested"]],
            [{"image": "a.png", "alt": "x"}],
        ],
    )
    def test_any_other_value_loads_unchanged(self, value):
        loaded = RubricDataset.from_json(dataset_json([{"submission": value}]))
        assert one_item(loaded).submission == value

    def test_a_value_that_is_not_content_raises_when_graded_not_when_loaded(self):
        loaded = RubricDataset.from_json(dataset_json([{"submission": ["text", 3]}]))
        with pytest.raises(TypeError, match="got int"):
            _normalize_content(one_item(loaded).submission)

    def test_an_empty_image_path_is_refused_naming_the_item_and_field(self):
        with pytest.raises(ValueError, match="Item 0 reference_submission: an image object's"):
            RubricDataset.from_json(
                dataset_json([{"submission": "x", "reference_submission": {"image": ""}}])
            )


class TestPathResolution:
    def test_from_file_resolves_against_the_files_directory_not_the_cwd(
        self, tmp_path, monkeypatch
    ):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        path = data_dir / "set.json"
        path.write_text(dataset_json([{"submission": {"image": "figures/f.png"}}]))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        item = one_item(RubricDataset.from_file(path))
        assert item.submission == Image(data_dir / "figures" / "f.png")

    def test_from_file_with_a_relative_path_resolves_against_its_directory(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "set.json").write_text(
            dataset_json([{"submission": {"image": "f.png"}}])
        )
        monkeypatch.chdir(tmp_path)
        item = one_item(RubricDataset.from_file("data/set.json"))
        assert item.submission == Image(tmp_path / "data" / "f.png")

    def test_from_json_resolves_against_base_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base = tmp_path / "base"
        loaded = RubricDataset.from_json(
            dataset_json([{"submission": {"image": "f.png"}}]), base_dir=base
        )
        assert one_item(loaded).submission == Image(base / "f.png")

    def test_from_json_without_base_dir_resolves_against_the_cwd_at_the_call(
        self, tmp_path, monkeypatch
    ):
        text = dataset_json([{"submission": {"image": "sub/f.png"}}])
        monkeypatch.chdir(tmp_path)
        assert one_item(RubricDataset.from_json(text)).submission == Image(
            tmp_path / "sub" / "f.png"
        )

    def test_an_absolute_path_is_kept(self, tmp_path):
        absolute = (tmp_path / "abs" / "f.png").as_posix()
        loaded = RubricDataset.from_json(
            dataset_json([{"submission": {"image": absolute}}]), base_dir=tmp_path / "other"
        )
        assert one_item(loaded).submission == Image(absolute)

    def test_dot_dot_is_kept_as_written(self, tmp_path):
        loaded = RubricDataset.from_json(
            dataset_json([{"submission": {"image": "../shared/f.png"}}]),
            base_dir=tmp_path / "data",
        )
        submission = one_item(loaded).submission
        assert isinstance(submission, Image)
        assert submission.path == tmp_path / "data" / ".." / "shared" / "f.png"

    def test_loading_reads_no_image_file_so_missing_files_load(self, tmp_path, monkeypatch):
        def no_reads(self: Path) -> bytes:
            raise AssertionError(f"read {self}")

        monkeypatch.setattr(Path, "read_bytes", no_reads)
        loaded = RubricDataset.from_json(
            dataset_json([{"submission": [{"image": "missing.png"}, "caption"]}]),
            base_dir=tmp_path,
        )
        assert not (tmp_path / "missing.png").exists()
        assert one_item(loaded).submission == [Image(tmp_path / "missing.png"), "caption"]


# =============================================================================
# Writing
# =============================================================================


def image_dataset(submission: Any, **fields: Any) -> RubricDataset:
    data = RubricDataset(prompt="Rate the figure.", rubric=RUBRIC, **fields)
    data.add_item(submission=submission, description="d", ground_truth=[MET])
    return data


def written_submission(text: str) -> Any:
    return json.loads(text)["items"][0]["submission"]


class TestToJson:
    def test_an_image_inside_base_dir_is_written_relative_with_slashes(self, tmp_path):
        data = image_dataset([Image(tmp_path / "figures" / "f.png"), "caption"])
        assert written_submission(data.to_json(base_dir=tmp_path)) == [
            {"image": "figures/f.png"},
            "caption",
        ]

    def test_an_image_outside_base_dir_is_written_absolute(self, tmp_path):
        image = Image(tmp_path / "outside" / "f.png")
        data = image_dataset(image)
        assert written_submission(data.to_json(base_dir=tmp_path / "data")) == {
            "image": (tmp_path / "outside" / "f.png").as_posix()
        }

    def test_without_base_dir_every_image_is_written_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = image_dataset(Image("f.png"))
        assert written_submission(data.to_json()) == {"image": (tmp_path / "f.png").as_posix()}

    def test_writing_reads_no_image_file(self, tmp_path):
        data = image_dataset(Image(tmp_path / "missing.png"))
        assert written_submission(data.to_json(base_dir=tmp_path)) == {"image": "missing.png"}

    @pytest.mark.parametrize(
        ("fields", "item_fields", "where"),
        [
            ({}, {"submission": "s", "prompt": "BYTES"}, "Item 0 prompt"),
            ({}, {"submission": ["s", "BYTES"]}, "Item 0 submission"),
            ({}, {"submission": "s", "reference_submission": "BYTES"}, "Item 0 reference_sub"),
            ({"prompt": "BYTES"}, {"submission": "s"}, "The dataset prompt"),
            ({"reference_submission": "BYTES"}, {"submission": "s"}, "The dataset reference"),
        ],
    )
    def test_a_bytes_backed_image_is_refused_naming_where_it_is(self, fields, item_fields, where):
        image = Image(encoded("red"))

        def swap(value: Any) -> Any:
            if value == "BYTES":
                return image
            if isinstance(value, list):
                return [swap(part) for part in value]
            return value

        data = RubricDataset(
            prompt=swap(fields.get("prompt", "Rate the figure.")),
            rubric=RUBRIC,
            reference_submission=swap(fields.get("reference_submission")),
        )
        data.add_item(description="d", **{k: swap(v) for k, v in item_fields.items()})
        with pytest.raises(ValueError, match=f"^{where}.*to_file"):
            data.to_json()

    def test_a_text_dataset_is_written_the_same_with_or_without_base_dir(self, tmp_path):
        data = RubricDataset.from_file(ESSAYS)
        assert data.to_json(base_dir=tmp_path) == data.to_json()

    def test_a_tuple_document_is_written_as_an_array(self, tmp_path):
        data = image_dataset((Image(tmp_path / "f.png"), "caption"))
        assert written_submission(data.to_json(base_dir=tmp_path)) == [
            {"image": "f.png"},
            "caption",
        ]


class TestToFile:
    def test_a_bytes_backed_image_is_saved_beside_the_file_by_its_digest(self, tmp_path):
        png = Image(encoded("red"))
        jpeg = Image(encoded("blue", "JPEG"))
        data = image_dataset([png, "caption", jpeg])
        path = tmp_path / "set.json"
        data.to_file(path)
        assert written_submission(path.read_text()) == [
            {"image": f"set_images/{png.digest}.png"},
            "caption",
            {"image": f"set_images/{jpeg.digest}.jpg"},
        ]
        assert (tmp_path / "set_images" / f"{png.digest}.png").read_bytes() == png.read_bytes()
        assert (tmp_path / "set_images" / f"{jpeg.digest}.jpg").read_bytes() == (jpeg.read_bytes())

    def test_a_saved_image_file_is_written_only_when_absent(self, tmp_path):
        png = Image(encoded("red"))
        saved = tmp_path / "set_images" / f"{png.digest}.png"
        saved.parent.mkdir()
        saved.write_bytes(b"already here")
        image_dataset(png).to_file(tmp_path / "set.json")
        assert saved.read_bytes() == b"already here"

    def test_the_same_bytes_are_saved_once(self, tmp_path):
        png = encoded("red")
        image_dataset([Image(png), "and again", Image(png)]).to_file(tmp_path / "set.json")
        assert [p.name for p in (tmp_path / "set_images").iterdir()] == [f"{Image(png).digest}.png"]

    def test_a_file_backed_image_is_written_relative_to_the_files_directory(self, tmp_path):
        figure = tmp_path / "figures" / "f.png"
        image_dataset(Image(figure)).to_file(tmp_path / "set.json")
        assert written_submission((tmp_path / "set.json").read_text()) == {"image": "figures/f.png"}
        assert not (tmp_path / "set_images").exists()

    def test_the_digest_survives_a_round_trip_through_a_moved_directory(self, tmp_path):
        figure = tmp_path / "data" / "figures" / "f.png"
        figure.parent.mkdir(parents=True)
        figure.write_bytes(encoded("green"))
        submission = [Image(figure), "caption", Image(encoded("red"))]
        image_dataset(submission, reference_submission=Image(encoded("blue"))).to_file(
            tmp_path / "data" / "set.json"
        )
        expected = _content_digest(submission)
        moved = tmp_path / "moved"
        (tmp_path / "data").rename(moved)
        loaded = RubricDataset.from_file(moved / "set.json")
        assert _content_digest(one_item(loaded).submission) == expected
        reference = loaded.reference_submission
        assert isinstance(reference, Image)
        assert reference.digest == Image(encoded("blue")).digest
        assert reference.path is not None and reference.path.parent == moved / "set_images"

    def test_a_text_dataset_file_is_unchanged(self, tmp_path):
        data = RubricDataset.from_file(ESSAYS)
        data.to_file(tmp_path / "set.json")
        assert (tmp_path / "set.json").read_text(encoding="utf-8") == data.to_json()
        assert [p.name for p in tmp_path.iterdir()] == ["set.json"]


class TestItems:
    def test_an_item_holding_images_pickles_by_its_image_sources(self, tmp_path):
        item = DataItem(submission=[Image(tmp_path / "f.png"), "caption"], description="d")
        assert pickle.loads(pickle.dumps(item)) == item
