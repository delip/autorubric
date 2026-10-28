"""Tests for ``autorubric.content``: the ``Image`` value, ``Content`` and the content helpers.

The images are real but tiny: each format is encoded by Pillow in a fixture, so every case
exercises the same format detection a user's file goes through.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import io
import pickle
from pathlib import Path

import PIL.Image as pil_image
import pytest

import autorubric
from autorubric.content import (
    Content,
    Image,
    _concat,
    _content_digest,
    _content_identity,
    _content_key,
    _content_preview,
    _content_text,
    _has_images,
    _images,
    _load_content,
    _normalize_content,
)
from autorubric.types import ToGradeInput


def _encoded(fmt: str, color: str = "red", size: tuple[int, int] = (3, 2)) -> bytes:
    """A tiny ``size`` image of one ``color``, encoded by Pillow as ``fmt``."""
    buffer = io.BytesIO()
    pil_image.new("RGB", size, color).save(buffer, fmt)
    return buffer.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def png() -> bytes:
    return _encoded("PNG")


@pytest.fixture
def other_png() -> bytes:
    return _encoded("PNG", color="blue")


@pytest.fixture
def png_file(tmp_path: Path, png: bytes) -> Path:
    path = tmp_path / "fig.png"
    path.write_bytes(png)
    return path


SUPPORTED = [
    ("PNG", "image/png"),
    ("JPEG", "image/jpeg"),
    ("GIF", "image/gif"),
    ("WEBP", "image/webp"),
]


class TestConstruction:
    def test_a_relative_path_is_stored_absolute(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        image = Image("figures/fig.png")
        assert image.path == tmp_path / "figures" / "fig.png"
        assert image.path is not None and image.path.is_absolute()

    def test_a_path_and_its_string_make_equal_images(self, tmp_path: Path) -> None:
        path = tmp_path / "fig.png"
        assert Image(path) == Image(str(path))
        assert hash(Image(path)) == hash(Image(str(path)))

    def test_a_path_like_object_is_accepted(self, tmp_path: Path) -> None:
        class PathLike:
            def __fspath__(self) -> str:
                return str(tmp_path / "fig.png")

        assert Image(PathLike()) == Image(tmp_path / "fig.png")

    def test_a_missing_path_constructs_compares_and_hashes_without_reading(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing.png"
        first, second = Image(missing), Image(str(missing))
        assert first == second
        assert len({first, second}) == 1
        assert "missing.png" in repr(first)
        assert not missing.exists()

    def test_a_missing_path_raises_when_its_bytes_are_needed(self, tmp_path: Path) -> None:
        image = Image(tmp_path / "missing.png")
        with pytest.raises(FileNotFoundError):
            image.read_bytes()
        with pytest.raises(FileNotFoundError):
            _ = image.media_type
        with pytest.raises(FileNotFoundError):
            _ = image.digest

    def test_bytes_bytearray_and_memoryview_make_equal_images(self, png: bytes) -> None:
        images = [Image(png), Image(bytearray(png)), Image(memoryview(png))]
        assert images[0] == images[1] == images[2]
        assert all(image.path is None for image in images)
        assert all(image.read_bytes() == png for image in images)

    def test_the_bytes_are_copied(self, png: bytes) -> None:
        buffer = bytearray(png)
        image = Image(buffer)
        buffer[:] = b"\0" * len(buffer)
        assert image.read_bytes() == png

    def test_bytes_are_validated_at_construction(self) -> None:
        with pytest.raises(ValueError, match="not an image"):
            Image(b"this is not an image")

    def test_empty_bytes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Image(b"")

    def test_an_empty_path_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Image("")

    @pytest.mark.parametrize("source", [1, 1.5, None, ["fig.png"], {"image": "fig.png"}])
    def test_any_other_source_is_a_type_error(self, source: object) -> None:
        with pytest.raises(TypeError, match=type(source).__name__):
            Image(source)  # ty: ignore[invalid-argument-type]

    def test_an_image_is_frozen(self, png: bytes) -> None:
        image = Image(png)
        with pytest.raises(dataclasses.FrozenInstanceError):
            image._data = b"x"  # ty: ignore[invalid-assignment]

    def test_the_package_exports_image_and_content(self) -> None:
        assert autorubric.Image is Image
        assert autorubric.Content is Content
        assert "Image" in autorubric.__all__
        assert "Content" in autorubric.__all__

    def test_to_grade_input_admits_content(self) -> None:
        assert ToGradeInput == Content | autorubric.ThinkingOutputDict


class TestFormatDetection:
    @pytest.mark.parametrize(("fmt", "media_type"), SUPPORTED)
    def test_bytes_of_each_supported_format(self, fmt: str, media_type: str) -> None:
        assert Image(_encoded(fmt)).media_type == media_type

    @pytest.mark.parametrize(("fmt", "media_type"), SUPPORTED)
    def test_a_file_of_each_supported_format(
        self, tmp_path: Path, fmt: str, media_type: str
    ) -> None:
        path = tmp_path / "figure"
        path.write_bytes(_encoded(fmt))
        assert Image(path).media_type == media_type

    def test_a_multi_picture_jpeg_is_a_jpeg(self) -> None:
        buffer = io.BytesIO()
        first, second = pil_image.new("RGB", (3, 2), "red"), pil_image.new("RGB", (3, 2))
        first.save(buffer, "MPO", save_all=True, append_images=[second])
        data = buffer.getvalue()
        with pil_image.open(io.BytesIO(data)) as opened:
            assert opened.format == "MPO"
        assert Image(data).media_type == "image/jpeg"

    @pytest.mark.parametrize("fmt", ["BMP", "TIFF"])
    def test_an_unsupported_format_is_named(self, fmt: str) -> None:
        with pytest.raises(ValueError, match=f"{fmt} image") as raised:
            Image(_encoded(fmt))
        assert "PNG, JPEG, GIF or WebP" in str(raised.value)

    def test_a_file_that_is_not_an_image_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("plain text")
        image = Image(path)
        with pytest.raises(ValueError, match="notes.txt") as raised:
            _ = image.media_type
        assert "not an image" in str(raised.value)

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            Image(path).read_bytes()


class TestBytesAndDigest:
    def test_the_digest_is_the_sha256_of_the_bytes(self, png: bytes, png_file: Path) -> None:
        assert Image(png).digest == _sha256(png)
        assert Image(png_file).digest == _sha256(png)

    def test_a_path_backed_image_rereads_its_file(
        self, png_file: Path, other_png: bytes, png: bytes
    ) -> None:
        image = Image(png_file)
        assert image.read_bytes() == png
        png_file.write_bytes(other_png)
        assert image.read_bytes() == other_png

    def test_a_path_backed_image_keeps_the_type_and_digest_of_its_first_read(
        self, png_file: Path, png: bytes
    ) -> None:
        image = Image(png_file)
        assert image.media_type == "image/png"
        png_file.unlink()
        assert image.digest == _sha256(png)
        assert image.media_type == "image/png"

    def test_a_loaded_image_holds_its_bytes(
        self, png_file: Path, png: bytes, other_png: bytes
    ) -> None:
        loaded = Image(png_file)._loaded()
        png_file.write_bytes(other_png)
        assert loaded.read_bytes() == png
        assert loaded.digest == _sha256(png)
        png_file.unlink()
        assert loaded.read_bytes() == png
        assert loaded.media_type == "image/png"

    def test_a_loaded_image_equals_its_source(self, png_file: Path) -> None:
        image = Image(png_file)
        loaded = image._loaded()
        assert loaded == image
        assert hash(loaded) == hash(image)
        assert loaded.path == image.path

    def test_loading_is_idempotent(self, png_file: Path, png: bytes) -> None:
        loaded = Image(png_file)._loaded()
        assert loaded._loaded() is loaded
        image = Image(png)
        assert image._loaded() is image

    def test_loading_reads_and_checks_the_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Image(tmp_path / "missing.png")._loaded()
        path = tmp_path / "notes.txt"
        path.write_text("plain text")
        with pytest.raises(ValueError, match="not an image"):
            Image(path)._loaded()

    def test_the_data_url_carries_the_bytes(self, png: bytes, png_file: Path) -> None:
        for image in (Image(png), Image(png_file)):
            prefix = "data:image/png;base64,"
            assert image._data_url.startswith(prefix)
            assert base64.b64decode(image._data_url[len(prefix) :]) == png

    def test_a_path_and_bytes_of_the_same_image_are_different_values(
        self, png: bytes, png_file: Path
    ) -> None:
        assert Image(png) != Image(png_file)
        assert Image(png).digest == Image(png_file).digest

    def test_the_repr_shows_the_source(self, png: bytes, png_file: Path) -> None:
        assert repr(Image(png_file)) == f"Image({str(png_file)!r})"
        assert repr(Image(png)) == f"Image(<{len(png)} bytes, image/png>)"


class TestPickleAndCopy:
    def test_a_bytes_backed_image_round_trips_without_its_caches(self, png: bytes) -> None:
        image = Image(png)
        _ = image.digest, image._data_url
        restored = pickle.loads(pickle.dumps(image))
        assert restored == image
        assert "digest" not in restored.__dict__
        assert "_data_url" not in restored.__dict__
        assert restored.read_bytes() == png

    def test_the_pickle_holds_the_source_only(self, png: bytes) -> None:
        image = Image(png)
        _ = image._data_url
        assert len(pickle.dumps(image)) < len(png) + 200

    def test_a_loaded_path_backed_image_unpickles_path_backed(
        self, png_file: Path, png: bytes, other_png: bytes
    ) -> None:
        loaded = Image(png_file)._loaded()
        _ = loaded.digest, loaded._data_url
        restored = pickle.loads(pickle.dumps(loaded))
        assert restored == loaded
        assert restored.path == png_file
        assert not {"_held", "digest", "media_type", "_data_url"} & restored.__dict__.keys()
        png_file.write_bytes(other_png)
        assert restored.read_bytes() == other_png

    def test_a_missing_path_pickles_without_reading(self, tmp_path: Path) -> None:
        image = Image(tmp_path / "missing.png")
        assert pickle.loads(pickle.dumps(image)) == image

    def test_a_copy_is_the_same_image(self, png_file: Path) -> None:
        loaded = Image(png_file)._loaded()
        assert copy.copy(loaded) is loaded
        assert copy.deepcopy([loaded])[0] is loaded


class TestNormalizeContent:
    def test_a_string_is_returned_as_the_same_object(self) -> None:
        text = "".join(["a", "b"])
        assert _normalize_content(text) is text

    def test_adjacent_strings_merge(self) -> None:
        assert _normalize_content(("a", "b")) == "ab"
        assert _normalize_content(["a", "b"]) == "ab"

    def test_empty_strings_drop_and_text_around_an_image_stays(self, png: bytes) -> None:
        image = Image(png)
        assert _normalize_content(["a", image, "", "b"]) == ("a", image, "b")
        assert _normalize_content(["a", "", "b", image, "c", "d"]) == ("ab", image, "cd")

    def test_images_keep_their_order(self, png: bytes, other_png: bytes) -> None:
        image, other = Image(png), Image(other_png)
        assert _normalize_content([image, other]) == (image, other)
        assert _normalize_content([other, image]) == (other, image)

    def test_a_bare_image_is_a_one_part_document(self, png: bytes) -> None:
        image = Image(png)
        assert _normalize_content(image) == (image,)

    def test_an_image_free_sequence_collapses_to_text(self) -> None:
        assert _normalize_content([]) == ""
        assert _normalize_content(["", ""]) == ""

    @pytest.mark.parametrize(
        ("value", "named"),
        [
            ([["a"]], "list"),
            ([("a",)], "tuple"),
            ([b"x"], "bytes"),
            ([1], "int"),
            (["a", None], "NoneType"),
        ],
    )
    def test_a_part_that_is_neither_text_nor_an_image(self, value: object, named: str) -> None:
        with pytest.raises(TypeError, match=named):
            _normalize_content(value)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        ("value", "named"), [(1, "int"), (None, "NoneType"), (b"x", "bytes"), ({}, "dict")]
    )
    def test_a_value_that_is_not_content(self, value: object, named: str) -> None:
        with pytest.raises(TypeError, match=named):
            _normalize_content(value)  # ty: ignore[invalid-argument-type]


class TestContentHelpers:
    def test_concat_of_text_is_the_joined_text(self) -> None:
        assert _concat("<a>", "body", "</a>") == "<a>body</a>"
        assert _concat() == ""

    def test_concat_splices_documents(self, png: bytes) -> None:
        image = Image(png)
        assert _concat("<s>", [image, "cap"], "</s>") == ("<s>", image, "cap</s>")
        assert _concat("x", image) == ("x", image)

    def test_has_images_and_images(self, png: bytes, other_png: bytes) -> None:
        image, other = Image(png), Image(other_png)
        assert not _has_images("text")
        assert not _has_images(["a", "b"])
        assert _has_images(["a", image])
        assert _images("text") == ()
        assert _images([other, "x", image, other]) == (other, image, other)

    def test_content_text_puts_the_marker_where_each_image_was(self, png: bytes) -> None:
        image = Image(png)
        text = "plain"
        assert _content_text(text) is text
        assert _content_text(["one", image, "two"]) == "one\ntwo"
        assert _content_text([image, "caption"], image_marker="") == "caption"
        assert _content_text([image, image], image_marker=" ") == "  "

    def test_content_preview_marks_each_image(self, png: bytes, png_file: Path) -> None:
        assert _content_preview("plain") == "plain"
        assert _content_preview(["See ", Image(png_file), " here"]) == ("See [image: fig.png] here")
        assert _content_preview([Image(png), "caption"]) == (f"[image: {len(png)} bytes]caption")

    def test_content_preview_reads_no_file(self, tmp_path: Path) -> None:
        assert _content_preview([Image(tmp_path / "gone.png")]) == "[image: gone.png]"


class TestContentDigest:
    def test_a_text_digest_is_the_sha256_of_its_utf8(self) -> None:
        assert _content_digest("x") == hashlib.sha256(b"x").hexdigest()
        assert _content_digest("héllo") == _sha256("héllo".encode())

    def test_text_only_sequences_digest_as_their_text(self) -> None:
        assert _content_digest(["a", "", "b"]) == _content_digest("ab")

    def test_a_document_digest_follows_the_framing(self, png: bytes) -> None:
        image = Image(png)
        expected = hashlib.sha256(
            b"autorubric-content-v1\0"
            + b"T"
            + (2).to_bytes(8, "big")
            + "é".encode()
            + b"I"
            + hashlib.sha256(png).digest()
        ).hexdigest()
        assert _content_digest(["é", image]) == expected

    def test_a_document_digest_is_not_its_text_digest(self, png: bytes) -> None:
        document = ["caption", Image(png)]
        assert _content_digest(document) != _content_digest("caption")
        assert _content_digest(document) != _content_digest(_content_text(document))

    def test_the_digest_changes_with_image_bytes(self, png: bytes, other_png: bytes) -> None:
        assert _content_digest(["c", Image(png)]) != _content_digest(["c", Image(other_png)])

    def test_the_digest_changes_with_part_order(self, png: bytes) -> None:
        image = Image(png)
        assert _content_digest(["c", image]) != _content_digest([image, "c"])
        assert _content_digest(["a", image, "b"]) != _content_digest(["ab", image])

    def test_the_digest_depends_on_bytes_not_source(self, png: bytes, png_file: Path) -> None:
        assert _content_digest([Image(png)]) == _content_digest([Image(png_file)])

    def test_content_key(self, png: bytes) -> None:
        text = "".join(["a", "b"])
        assert _content_key(text) is text
        assert _content_key(None) is None
        assert _content_key(["a", "b"]) == "ab"
        document = ["a", Image(png)]
        assert _content_key(document) == {"content_sha256": _content_digest(document)}

    def test_content_identity_of_text_is_the_text(self) -> None:
        # A lone surrogate cannot be encoded, so it could not be hashed.
        unencodable = "Plants use light \ud83d to make sugar."
        assert _content_identity(unencodable) is unencodable
        assert _content_identity(["Li", "ght."]) == "Light."
        assert _content_identity([]) == ""

    def test_content_identity_of_a_document_is_its_digest(
        self, png: bytes, other_png: bytes, png_file: Path
    ) -> None:
        document = [Image(png), "x"]
        identity = _content_identity(document)
        assert identity == ("content_sha256", _content_digest(document))
        assert _content_identity((Image(png_file), "x")) == identity
        assert _content_identity([Image(other_png), "x"]) != identity
        assert _content_identity(["x", Image(png)]) != identity
        # No text, its digest included, is a document's identity.
        assert _content_identity(_content_digest(document)) != identity


class TestLoadContent:
    def test_text_is_returned_unchanged(self) -> None:
        text = "".join(["a", "b"])
        assert _load_content(text) is text
        assert _load_content(["a", "b"]) == "ab"

    def test_every_image_is_loaded(self, png_file: Path, png: bytes, other_png: bytes) -> None:
        bytes_backed = Image(other_png)
        loaded = _load_content(["a", Image(png_file), bytes_backed])
        assert loaded == ("a", Image(png_file), bytes_backed)
        assert isinstance(loaded, tuple)
        assert loaded[2] is bytes_backed
        png_file.unlink()
        part = loaded[1]
        assert isinstance(part, Image)
        assert part.read_bytes() == png

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_content(["a", Image(tmp_path / "missing.png")])

    def test_a_file_that_is_not_an_image_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("plain text")
        with pytest.raises(ValueError, match="notes.txt"):
            _load_content(Image(path))
