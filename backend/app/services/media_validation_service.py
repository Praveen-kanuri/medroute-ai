"""Phase 2C: secure validation and normalization of a directly-uploaded
image or video file, before any vision-model call is ever made.

Only raw bytes the client uploaded are accepted — never a filesystem path
or remote URL. Format is determined by sniffing file-signature ("magic
bytes"), never by trusting the client-declared filename extension or MIME
type. Uploaded media, extracted frames, and any temporary file this module
creates are never persisted beyond this module's own processing — video
temp files are removed in a `finally` block on every path, including
failure. Nothing in this module logs filenames or media bytes.
"""

import io
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import cv2
from PIL import Image, ImageOps

from app.config.settings import Settings

MediaKind = Literal["image", "video"]

_JPEG_SIGNATURE = b"\xff\xd8\xff"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_RIFF_TAG = b"RIFF"
_WEBP_TAG = b"WEBP"
_EBML_SIGNATURE = b"\x1a\x45\xdf\xa3"  # WebM/Matroska container header
_ISO_BMFF_BOX = b"ftyp"  # MP4/MOV (ISO base media file format)
_QUICKTIME_BRAND = b"qt  "


class EmptyMediaUploadError(ValueError):
    """Raised when the uploaded file has zero bytes."""


class MediaTooLargeError(ValueError):
    """Raised when the uploaded file exceeds the configured size limit."""


class UnsupportedMediaFormatError(ValueError):
    """Raised when the file signature does not match a supported image or
    video format — regardless of what the filename or client-declared
    content type claimed."""


class MalformedMediaError(ValueError):
    """Raised when a file with a supported-looking signature cannot
    actually be decoded (corrupt, truncated, or otherwise invalid)."""


class MediaDimensionExceededError(ValueError):
    """Raised when decoded pixel dimensions/count exceed the configured
    conservative limit (guards against decompression-bomb-style images and
    absurdly large video frames)."""


class VideoDurationExceededError(ValueError):
    """Raised when a video's decoded duration exceeds the configured
    maximum."""


@dataclass(frozen=True)
class MediaFrame:
    """One still image ready for vision analysis: a normalized JPEG for a
    direct image upload, or one deterministically sampled video frame.
    Never written to disk or logged — held only in memory for the
    duration of the current request."""

    data: bytes
    timestamp_seconds: float | None


@dataclass(frozen=True)
class PreparedMedia:
    """The validated, normalized result of one upload — the only thing
    passed on to vision analysis. Raw upload bytes are never retained
    beyond producing this."""

    kind: MediaKind
    frames: list[MediaFrame]


def detect_media_kind(data: bytes) -> MediaKind:
    """Sniff the file signature to classify the upload — never trusts a
    filename extension or client-declared Content-Type."""
    if not data:
        raise EmptyMediaUploadError("Uploaded file is empty.")

    if data.startswith(_JPEG_SIGNATURE) or data.startswith(_PNG_SIGNATURE):
        return "image"
    if len(data) >= 12 and data[0:4] == _RIFF_TAG and data[8:12] == _WEBP_TAG:
        return "image"
    if len(data) >= 12 and data[4:8] == _ISO_BMFF_BOX:
        return "video"
    if data.startswith(_EBML_SIGNATURE):
        return "video"

    raise UnsupportedMediaFormatError(
        "Unsupported file format. Supported: JPEG, PNG, WebP images; MP4, MOV, WebM video."
    )


def _video_temp_suffix(data: bytes) -> str:
    if data.startswith(_EBML_SIGNATURE):
        return ".webm"
    if len(data) >= 12 and data[8:12] == _QUICKTIME_BRAND:
        return ".mov"
    return ".mp4"


def process_image_upload(data: bytes, *, settings: Settings) -> MediaFrame:
    """Validate, orient, and re-encode an uploaded image — stripping
    metadata (EXIF, etc.) in the process. Raises a typed error for an
    oversized, malformed, or excessively large (decompression-bomb-style)
    image; never fabricates a usable image on failure."""
    if len(data) > settings.image_max_upload_bytes:
        raise MediaTooLargeError(
            f"Uploaded image exceeds the {settings.image_max_upload_bytes}-byte limit."
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except Exception as exc:
        raise MalformedMediaError("Uploaded image could not be decoded.") from exc

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width > settings.image_max_dimension_px or height > settings.image_max_dimension_px:
                raise MediaDimensionExceededError(
                    f"Image dimensions exceed the {settings.image_max_dimension_px}px-per-side "
                    "limit."
                )
            if width * height > settings.image_max_pixels:
                raise MediaDimensionExceededError(
                    f"Image pixel count exceeds the {settings.image_max_pixels}-pixel limit."
                )

            # Correct orientation using EXIF before dropping the EXIF data
            # itself; convert to RGB so the re-encode below never carries
            # transparency/palette metadata forward.
            normalized = ImageOps.exif_transpose(image)
            if normalized is None:
                normalized = image
            normalized = normalized.convert("RGB")

            buffer = io.BytesIO()
            # No `exif=` kwarg is passed, so no metadata is carried into
            # the re-encoded output.
            normalized.save(buffer, format="JPEG", quality=85)
    except MediaDimensionExceededError:
        raise
    except Exception as exc:
        raise MalformedMediaError("Uploaded image could not be processed.") from exc

    return MediaFrame(data=buffer.getvalue(), timestamp_seconds=None)


@contextmanager
def _temp_video_file(data: bytes, *, suffix: str) -> Iterator[str]:
    """A securely-scoped temp file that is always removed on exit, even if
    the caller raises — video is never left on disk beyond one request's
    processing."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _sample_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Deterministic, evenly-spread frame indices from the first to the
    last frame — never unbounded, never random."""
    if sample_count <= 1 or frame_count <= 1:
        return [0]
    step = (frame_count - 1) / (sample_count - 1)
    indices = {round(i * step) for i in range(sample_count)}
    return sorted(i for i in indices if 0 <= i < frame_count)


def process_video_upload(data: bytes, *, settings: Settings) -> list[MediaFrame]:
    """Validate a video and deterministically sample a bounded number of
    frames for analysis — the raw video is never sent to a vision model.
    The temporary file used for decoding is always removed, including on
    every failure path."""
    if len(data) > settings.video_max_upload_bytes:
        raise MediaTooLargeError(
            f"Uploaded video exceeds the {settings.video_max_upload_bytes}-byte limit."
        )

    with _temp_video_file(data, suffix=_video_temp_suffix(data)) as path:
        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                raise MalformedMediaError("Uploaded video could not be decoded.")

            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                raise MalformedMediaError("Uploaded video could not be decoded.")

            duration_seconds = frame_count / fps
            if duration_seconds > settings.video_max_duration_seconds:
                raise VideoDurationExceededError(
                    f"Video duration exceeds the {settings.video_max_duration_seconds}-second "
                    "limit."
                )
            if width > settings.video_max_dimension_px or height > settings.video_max_dimension_px:
                raise MediaDimensionExceededError(
                    f"Video frame dimensions exceed the {settings.video_max_dimension_px}"
                    "px-per-side limit."
                )

            sample_count = min(settings.video_max_sampled_frames, frame_count)
            frames: list[MediaFrame] = []
            for index in _sample_frame_indices(frame_count, sample_count):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = capture.read()
                if not success:
                    continue
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    continue
                frames.append(
                    MediaFrame(data=buffer.tobytes(), timestamp_seconds=round(index / fps, 2))
                )

            if not frames:
                raise MalformedMediaError("No frames could be extracted from the uploaded video.")
            return frames
        finally:
            capture.release()


def prepare_media_upload(data: bytes, *, settings: Settings) -> PreparedMedia:
    """Validate and normalize an uploaded image or video into a bounded
    list of still frames ready for vision analysis. Raises a typed error
    (see the exceptions above) for any empty, oversized, unsupported,
    malformed, or excessive-dimension/duration upload."""
    kind = detect_media_kind(data)
    if kind == "image":
        return PreparedMedia(kind="image", frames=[process_image_upload(data, settings=settings)])
    return PreparedMedia(kind="video", frames=process_video_upload(data, settings=settings))
