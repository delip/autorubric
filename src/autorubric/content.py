"""Submission content: text, images, and interleaved documents of both.

A submission (and a query or reference submission) is ``Content``: a string, an ``Image``,
or a sequence of strings and images read in order (a *document*, such as a figure followed
by its caption). The helpers here are the one place that reads content: they put it in a
canonical form (``_normalize_content``), join it (``_concat``), take its text
(``_content_text``, ``_content_preview``), take what follows a prefix
(``_content_removeprefix``), tell it from other values (``_is_content``),
identify it (``_content_digest``, ``_content_key``) and read its image files
(``_load_content``, ``_load_content_async``).

This module imports nothing from ``autorubric``, so every other module can import it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

import PIL.Image as pil_image

# Pillow's format names for the formats an Image may hold, and their media types. Pillow
# reports a JPEG that carries several pictures (MPO, as cameras and phones write) as "MPO";
# it is still a JPEG file, read as its first picture by anything that reads JPEG.
_MEDIA_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

_DIGEST_DOMAIN = b"autorubric-content-v1\0"


def _media_type_of(data: bytes, what: str) -> str:
    """The media type of the image ``data`` holds, as Pillow identifies its format.

    Only the format is read (``PIL.Image.open(...).format``); the pixels are not decoded.

    Args:
        data: The image bytes.
        what: How messages name the bytes ("Image data", "Image file /x/fig.png").

    Raises:
        ValueError: If ``data`` is empty, is not an image Pillow can identify, or is an
            image of a format other than PNG, JPEG, GIF or WebP.
    """
    if not data:
        raise ValueError(f"{what} is empty")
    try:
        with pil_image.open(io.BytesIO(data)) as opened:
            image_format = opened.format
    except OSError as error:
        # UnidentifiedImageError's own message names only the in-memory buffer.
        reason = "" if isinstance(error, pil_image.UnidentifiedImageError) else f": {error}"
        raise ValueError(
            f"{what} is not an image Pillow can identify ({len(data)} bytes starting "
            f"{data[:16]!r}){reason}"
        ) from error
    media_type = _MEDIA_TYPES.get(image_format or "")
    if media_type is None:
        raise ValueError(
            f"{what} is a {image_format} image; an Image must be PNG, JPEG, GIF or WebP"
        )
    return media_type


@dataclass(frozen=True, init=False, repr=False)
class Image:
    """An image in submission content, read from a file or held as bytes.

    ``Image(path)`` refers to an image file. The path (a ``str`` or ``os.PathLike``) is made
    absolute (``Path.absolute()``, against the current working directory, symlinks and
    ``..`` kept as written) and nothing is read: the file need not exist until its bytes
    are needed. ``Image(data)`` holds a copy of ``data`` (``bytes``, ``bytearray`` or
    ``memoryview``), whose format is checked at once.

    An image may be PNG, JPEG, GIF or WebP, as Pillow identifies the format; a JPEG holding
    several pictures (MPO) counts as a JPEG.

    Two images are equal, and hash alike, when their sources are: the same absolute path,
    or the same bytes. Comparing and hashing never read a file, so images of missing files
    compare without raising. A file's image and the same bytes held in memory are different
    values with the same ``digest``; content that must be identified by what it shows, not
    where it comes from, uses the digest (see ``_content_digest``).

    A file-backed image reads its file each time ``read_bytes()`` is called. Its
    ``media_type`` and ``digest`` come from the first read that needs them (both from the
    same read) and are kept; so is ``_data_url``. Nothing else is kept, so an image that is
    only compared, hashed or identified does not hold its bytes in memory. ``_loaded()``
    reads the file once and returns an equal image that holds those bytes, whose
    ``read_bytes()``, ``media_type``, ``digest`` and ``_data_url`` all describe that one
    read.

    An image pickles as its source alone: a file-backed image (loaded or not) unpickles as a
    file-backed image that has read nothing, a bytes-backed one as its bytes. Copies
    (``copy.copy``, ``copy.deepcopy``) are the image itself, as for any immutable value.

    Example:
        >>> from autorubric import Image
        >>> figure = Image("figures/fig1.png")  # read when graded
        >>> submission = [figure, "Figure 1: accuracy by model size."]
    """

    _path: Path | None
    _data: bytes | None

    def __init__(self, source: str | os.PathLike[str] | bytes | bytearray | memoryview) -> None:
        """Make an image from a file path or from image bytes.

        Args:
            source: A path to an image file (``str`` or ``os.PathLike``), or the image's
                bytes (``bytes``, ``bytearray`` or ``memoryview``, copied).

        Raises:
            TypeError: If ``source`` is neither a path nor bytes.
            ValueError: If ``source`` is an empty path, or bytes that are empty, not an
                image, or not PNG, JPEG, GIF or WebP.
        """
        if isinstance(source, bytes | bytearray | memoryview):
            data = bytes(source)
            media_type = _media_type_of(data, "Image data")
            object.__setattr__(self, "_path", None)
            object.__setattr__(self, "_data", data)
            self.__dict__["media_type"] = media_type
            return
        if isinstance(source, str | os.PathLike):
            path = os.fspath(source)
            if not isinstance(path, str):
                raise TypeError(f"An image path must be text, got {type(path).__name__}")
            if not path:
                raise ValueError("An image path must not be empty")
            object.__setattr__(self, "_path", Path(path).absolute())
            object.__setattr__(self, "_data", None)
            return
        raise TypeError(
            "Image() takes a file path (str or os.PathLike) or image bytes "
            f"(bytes, bytearray or memoryview), got {type(source).__name__}"
        )

    @property
    def path(self) -> Path | None:
        """The image file's absolute path, or ``None`` for an image held as bytes."""
        return self._path

    def read_bytes(self) -> bytes:
        """The image's bytes.

        A bytes-backed image, or one returned by ``_loaded()``, returns the bytes it holds.
        Otherwise the file is read (each call) and its format checked.

        Raises:
            OSError: If the file cannot be read (``FileNotFoundError`` when it is missing).
            ValueError: If the file is empty, not an image, or not PNG, JPEG, GIF or WebP.
        """
        if self._data is not None:
            return self._data
        held = self.__dict__.get("_held")
        if held is not None:
            return held
        data, media_type = self._read_file()
        self.__dict__.setdefault("media_type", media_type)
        if "digest" not in self.__dict__:
            self.__dict__["digest"] = hashlib.sha256(data).hexdigest()
        return data

    def _read_file(self) -> tuple[bytes, str]:
        """The file's bytes and their media type, read now (a file-backed image only)."""
        path = self._path
        assert path is not None  # an image holds either bytes or a path
        data = path.read_bytes()
        return data, _media_type_of(data, f"Image file {path}")

    @cached_property
    def media_type(self) -> str:
        """The image's media type: ``image/png``, ``image/jpeg``, ``image/gif`` or ``image/webp``.

        Raises:
            OSError: If a file-backed image's file cannot be read.
            ValueError: If that file is not a supported image.
        """
        # A bytes-backed image has it from construction and a loaded one from its read;
        # only a file-backed image that has not read its file gets here.
        self.read_bytes()
        return self.__dict__["media_type"]

    @cached_property
    def digest(self) -> str:
        """The SHA-256 of the image's bytes, as 64 lowercase hex digits.

        Raises:
            OSError: If a file-backed image's file cannot be read.
            ValueError: If that file is not a supported image.
        """
        if self._data is not None:
            return hashlib.sha256(self._data).hexdigest()
        self.read_bytes()
        return self.__dict__["digest"]

    @property
    def _bytes_held(self) -> bool:
        """Whether the image holds its bytes (bytes-backed, or returned by ``_loaded()``), so
        that reading them reads no file."""
        return self._data is not None or "_held" in self.__dict__

    def _loaded(self) -> Image:
        """This image with its bytes held: an equal image that reads no file again.

        A bytes-backed image, or one already loaded, is returned itself. A file-backed one
        reads its file once; the returned image's ``read_bytes()``, ``media_type`` and
        ``digest`` all come from that read.

        Raises:
            OSError: If the file cannot be read (``FileNotFoundError`` when it is missing).
            ValueError: If the file is empty, not an image, or not PNG, JPEG, GIF or WebP.
        """
        if self._bytes_held:
            return self
        data, media_type = self._read_file()
        assert self._path is not None  # read above
        loaded = type(self)(self._path)
        loaded.__dict__.update(
            _held=data, media_type=media_type, digest=hashlib.sha256(data).hexdigest()
        )
        return loaded

    @cached_property
    def _data_url(self) -> str:
        """The image as a ``data:`` URL (``data:<media type>;base64,<bytes>``).

        Raises:
            As ``_loaded()``.
        """
        loaded = self._loaded()
        encoded = base64.b64encode(loaded.read_bytes()).decode("ascii")
        return f"data:{loaded.media_type};base64,{encoded}"

    def __repr__(self) -> str:
        if self._path is not None:
            return f"Image({str(self._path)!r})"
        return f"Image(<{len(self._data or b'')} bytes, {self.media_type}>)"

    def __reduce__(self) -> tuple[type[Image], tuple[Path | bytes]]:
        source = self._path if self._path is not None else self._data
        assert source is not None  # an image holds either bytes or a path
        return (type(self), (source,))

    def __copy__(self) -> Image:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Image:
        return self


Content = str | Image | Sequence[str | Image]
"""What a submission, query or reference submission may be.

A string; an ``Image``; or a sequence (a list or tuple) of strings and images read in
order, a document such as ``[Image("fig1.png"), "Figure 1: accuracy by model size."]``.
A sequence holding only strings is the same content as their concatenation.
"""

_Normalized = str | tuple[str | Image, ...]


def _normalize_content(value: Content) -> _Normalized:
    """``value`` in canonical form: its text, or a document of parts.

    A ``str`` is returned as it is (the same object). A bare ``Image`` becomes the
    one-part document ``(image,)``. The parts of a sequence are taken in order: empty
    strings are dropped and adjacent strings are joined with no separator, so a document
    never holds two strings in a row; when no image is left, the result is the joined
    text (``""`` for an empty sequence) rather than a document.

    Raises:
        TypeError: If ``value`` is not content, or a part of a sequence is neither a
            ``str`` nor an ``Image`` (nested sequences and ``bytes`` included). The
            message names the offending type.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Image):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray | memoryview):
        raise TypeError(
            "Content must be a str, an Image, or a sequence of str and Image parts, "
            f"got {type(value).__name__}"
        )
    parts: list[str | Image] = []
    text: list[str] = []
    for part in value:
        if isinstance(part, str):
            if part:
                text.append(part)
        elif isinstance(part, Image):
            if text:
                parts.append("".join(text))
                text = []
            parts.append(part)
        else:
            raise TypeError(f"A content part must be a str or an Image, got {type(part).__name__}")
    if not parts:
        return "".join(text)
    if text:
        parts.append("".join(text))
    return tuple(parts)


def _is_content(value: object) -> bool:
    """Whether ``value`` is ``Content``, as ``_normalize_content`` accepts it.

    A value it refuses, such as a ``{thinking, output}`` dict submission or a JSON value a
    dataset file kept as it was, is not content; the helpers that must handle such values
    as they did before images existed ask this first.
    """
    try:
        _normalize_content(cast(Content, value))
    except TypeError:
        return False
    return True


def _concat(*pieces: Content) -> _Normalized:
    """The content of ``pieces`` read one after another, in canonical form.

    Strings alone join exactly as ``"".join(pieces)``; documents are spliced in place, with
    the text on either side of each joining the text it meets.

    Raises:
        TypeError: As ``_normalize_content``, for a piece or part that is not content.
    """
    parts: list[str | Image] = []
    for piece in pieces:
        normalized = _normalize_content(piece)
        if isinstance(normalized, str):
            parts.append(normalized)
        else:
            parts.extend(normalized)
    return _normalize_content(parts)


def _content_removeprefix(value: Content, prefix: Content) -> _Normalized | None:
    """What follows ``prefix`` in ``value``, in canonical form, or ``None`` if ``value`` does
    not start with it.

    Text works as ``str.startswith`` and slicing do. For a document, the parts of ``prefix``
    before its last must equal ``value``'s first parts (images by ``Image`` equality, so by
    source, with no I/O), and its last part must start the part at that position: the same
    image, or text the part begins with. That is how a prompt built with ``_concat`` from
    ``prefix`` and more content starts, since ``_concat`` joins the text the two meet.

    Raises:
        TypeError: As ``_normalize_content``, for a part that is not content.
    """
    whole = _normalize_content(value)
    start = _normalize_content(prefix)
    if isinstance(whole, str) and isinstance(start, str):
        return whole[len(start) :] if whole.startswith(start) else None
    if start == "":
        return whole
    whole_parts = [whole] if isinstance(whole, str) else list(whole)
    start_parts = [start] if isinstance(start, str) else list(start)
    last = len(start_parts) - 1
    if len(whole_parts) <= last or whole_parts[:last] != start_parts[:last]:
        return None
    meeting, ending = whole_parts[last], start_parts[last]
    if isinstance(ending, str):
        if not isinstance(meeting, str) or not meeting.startswith(ending):
            return None
        rest: list[str | Image] = [meeting[len(ending) :]]
    elif meeting == ending:
        rest = []
    else:
        return None
    return _normalize_content([*rest, *whole_parts[last + 1 :]])


def _has_images(value: Content) -> bool:
    """Whether ``value`` holds an image."""
    return not isinstance(_normalize_content(value), str)


def _images(value: Content) -> tuple[Image, ...]:
    """The images in ``value``, in order, each as often as it occurs."""
    normalized = _normalize_content(value)
    if isinstance(normalized, str):
        return ()
    return tuple(part for part in normalized if isinstance(part, Image))


def _content_text(value: Content, *, image_marker: str = "\n") -> str:
    """The text of ``value``, with ``image_marker`` in place of each image.

    The default marker keeps the words on either side of an image apart. Text is returned
    as it is (a ``str`` as the same object).
    """
    normalized = _normalize_content(value)
    if isinstance(normalized, str):
        return normalized
    return "".join(image_marker if isinstance(part, Image) else part for part in normalized)


def _image_marker(image: Image) -> str:
    if image.path is not None:
        return f"[image: {image.path.name}]"
    return f"[image: {len(image.read_bytes())} bytes]"


def _content_preview(value: Content) -> str:
    """The text of ``value`` for people to read, with a marker in place of each image.

    The marker is ``[image: <file name>]`` for a file-backed image and
    ``[image: <N> bytes]`` for one held as bytes, written exactly where the image was. No
    file is read.
    """
    normalized = _normalize_content(value)
    if isinstance(normalized, str):
        return normalized
    return "".join(_image_marker(part) if isinstance(part, Image) else part for part in normalized)


def _content_digest(value: Content) -> str:
    """The SHA-256 (64 lowercase hex digits) identifying ``value`` by what it holds.

    Text, including a sequence of strings alone, digests as
    ``hashlib.sha256(text.encode("utf-8")).hexdigest()``, the digest the library keyed text
    submissions by before images existed. A document digests as the SHA-256 of
    ``b"autorubric-content-v1\\0"`` followed by each part in order: text as ``b"T"``, its
    UTF-8 length in 8 big-endian bytes and the UTF-8 itself; an image as ``b"I"`` and the 32
    bytes of its ``digest``. So a document's digest depends on its images' bytes (not on
    where they are read from) and on the order of its parts, and is never the digest of any
    text.

    Raises:
        OSError: If a file-backed image's file cannot be read.
        TypeError: As ``_normalize_content``, for a value that is not content.
        ValueError: If an image file is not a supported image.
    """
    normalized = _normalize_content(value)
    if isinstance(normalized, str):
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    for part in normalized:
        if isinstance(part, Image):
            digest.update(b"I" + bytes.fromhex(part.digest))
        else:
            encoded = part.encode("utf-8")
            digest.update(b"T" + len(encoded).to_bytes(8, "big") + encoded)
    return digest.hexdigest()


def _content_key(value: Content | None) -> object:
    """``value`` in a form to embed in JSON that is hashed to identify it.

    Text (and ``None``) is returned unchanged, so a key built from text is the key built
    before images existed; a sequence of strings alone gives its joined text. A document
    gives ``{"content_sha256": _content_digest(value)}``. A value that is not content (a
    ``{thinking, output}`` dict submission, say) is returned unchanged too, as it was
    embedded before images existed.

    Raises:
        OSError: If a file-backed image's file cannot be read.
        ValueError: If an image file is not a supported image.
    """
    if value is None or isinstance(value, str):
        return value
    try:
        normalized = _normalize_content(value)
    except TypeError:
        return value
    if isinstance(normalized, str):
        return normalized
    return {"content_sha256": _content_digest(normalized)}


def _content_identity(value: Content) -> str | tuple[str, str]:
    """A hashable value identifying ``value`` by what it holds, to compare or collect it.

    Text is its own identity, returned as it is (a sequence of strings alone gives its
    joined text) and never hashed, so any ``str`` serves, an unencodable one included. A
    document gives ``("content_sha256", _content_digest(value))``, which no text equals, so
    documents whose images hold the same bytes share an identity wherever the images are
    read from.

    Raises:
        As ``_content_digest``, for a document.
    """
    normalized = _normalize_content(value)
    if isinstance(normalized, str):
        return normalized
    return ("content_sha256", _content_digest(normalized))


def _load_content(value: Content) -> _Normalized:
    """``value`` in canonical form with every image's bytes held (``Image._loaded()``).

    Each image file is read once, here, so that grading reads no file afterwards and a
    missing or unreadable file fails before any judge is called. Text is returned as
    ``_normalize_content`` returns it (a ``str`` as the same object).

    Raises:
        OSError: If an image file cannot be read (``FileNotFoundError`` when it is missing).
        TypeError: As ``_normalize_content``, for a value that is not content.
        ValueError: If an image file is empty, not an image, or not PNG, JPEG, GIF or WebP.
    """
    normalized = _normalize_content(value)
    if isinstance(normalized, str):
        return normalized
    return tuple(part._loaded() if isinstance(part, Image) else part for part in normalized)


async def _load_content_async(value: Content) -> _Normalized:
    """``_load_content(value)``, reading image files in a worker thread.

    Only content holding a file-backed image whose bytes are not held yet reads a file;
    it is loaded in ``asyncio.to_thread``, so an event loop grading other items keeps
    running while the files are read. Any other content (text, bytes-backed images,
    content loaded already) is put in canonical form at once, on the caller's thread,
    reading nothing: text is returned as the same object, and loading loaded content again
    gives equal content.

    Raises:
        As ``_load_content``.
    """
    normalized = _normalize_content(value)
    if isinstance(normalized, str) or all(
        part._bytes_held for part in normalized if isinstance(part, Image)
    ):
        return _load_content(normalized)
    return await asyncio.to_thread(_load_content, normalized)
