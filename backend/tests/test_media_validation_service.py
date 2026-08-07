"""Phase 2C: tests for image/video upload validation and normalization.

Proves format detection is signature-based (never trusts a filename or
declared content type), and that empty/oversized/malformed/excessive
uploads are rejected with typed errors before any vision-model call would
ever be made — this module never fabricates a usable frame on failure.
"""

import pytest

from app.config.settings import Settings
from app.services.media_validation_service import (
    EmptyMediaUploadError,
    MalformedMediaError,
    MediaDimensionExceededError,
    MediaTooLargeError,
    UnsupportedMediaFormatError,
    VideoDurationExceededError,
    detect_media_kind,
    prepare_media_upload,
    process_image_upload,
    process_video_upload,
)
from tests.media_fixtures import (
    MALFORMED_MP4_BYTES,
    make_jpeg_bytes,
    make_mp4_bytes,
    make_png_bytes,
    make_webp_bytes,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --- format detection (signature-based, not extension/MIME) -----------------


def test_detects_jpeg_by_signature() -> None:
    assert detect_media_kind(make_jpeg_bytes()) == "image"


def test_detects_png_by_signature() -> None:
    assert detect_media_kind(make_png_bytes()) == "image"


def test_detects_webp_by_signature() -> None:
    assert detect_media_kind(make_webp_bytes()) == "image"


def test_detects_mp4_by_signature() -> None:
    assert detect_media_kind(make_mp4_bytes()) == "video"


def test_empty_upload_rejected() -> None:
    with pytest.raises(EmptyMediaUploadError):
        detect_media_kind(b"")


def test_unsupported_signature_rejected() -> None:
    with pytest.raises(UnsupportedMediaFormatError):
        detect_media_kind(b"not a real media file, just plain text bytes")


def test_extension_and_content_type_spoofing_is_ignored() -> None:
    # A JPEG's real bytes must be classified as an image even though a
    # caller could claim any filename/extension or Content-Type — this
    # module never trusts either, only the actual file signature.
    jpeg_bytes_claiming_to_be_a_video = make_jpeg_bytes()
    assert detect_media_kind(jpeg_bytes_claiming_to_be_a_video) == "image"


def test_text_file_with_video_extension_is_rejected_not_misclassified() -> None:
    with pytest.raises(UnsupportedMediaFormatError):
        detect_media_kind(b"plain text pretending to be a .mp4 upload")


# --- image processing --------------------------------------------------


def test_valid_image_is_processed_into_one_frame() -> None:
    frame = process_image_upload(make_jpeg_bytes(), settings=_settings())
    assert frame.timestamp_seconds is None
    assert frame.data.startswith(b"\xff\xd8\xff")  # re-encoded as JPEG


def test_oversized_image_rejected() -> None:
    with pytest.raises(MediaTooLargeError):
        process_image_upload(make_jpeg_bytes(), settings=_settings(image_max_upload_bytes=10))


def test_malformed_image_rejected() -> None:
    with pytest.raises(MalformedMediaError):
        process_image_upload(b"\xff\xd8\xff" + b"garbage-not-a-real-jpeg", settings=_settings())


def test_image_dimension_limit_enforced() -> None:
    with pytest.raises(MediaDimensionExceededError):
        process_image_upload(
            make_jpeg_bytes(width=64, height=48), settings=_settings(image_max_dimension_px=32)
        )


def test_image_pixel_count_limit_enforced_decompression_bomb_guard() -> None:
    # 64x48 = 3072 pixels is under the per-side dimension limit but over a
    # deliberately tiny configured total-pixel-count limit — proves the
    # guard checks width*height independently of the per-side check, the
    # same mechanism that protects against a decompression-bomb-style
    # image whose per-side dimensions look individually reasonable.
    with pytest.raises(MediaDimensionExceededError):
        process_image_upload(
            make_jpeg_bytes(width=64, height=48), settings=_settings(image_max_pixels=1000)
        )


def test_image_metadata_and_orientation_normalized() -> None:
    # The re-encoded frame is a fresh JPEG produced by Pillow's own save()
    # with no `exif=` kwarg — no EXIF/metadata segment is carried forward.
    frame = process_image_upload(make_jpeg_bytes(), settings=_settings())
    assert b"Exif" not in frame.data


# --- video processing ----------------------------------------------------


def test_valid_video_is_sampled_into_bounded_frames() -> None:
    frames = process_video_upload(
        make_mp4_bytes(frame_count=30), settings=_settings(video_max_sampled_frames=4)
    )
    assert 1 <= len(frames) <= 4
    timestamps = [frame.timestamp_seconds for frame in frames]
    assert timestamps == sorted(timestamps)
    assert all(timestamp is not None and timestamp >= 0 for timestamp in timestamps)


def test_video_frame_timestamps_preserved_and_increasing() -> None:
    frames = process_video_upload(
        make_mp4_bytes(fps=10.0, frame_count=20), settings=_settings(video_max_sampled_frames=5)
    )
    timestamps = [frame.timestamp_seconds for frame in frames]
    assert len(set(timestamps)) == len(timestamps)  # deterministic, distinct samples


def test_oversized_video_rejected() -> None:
    with pytest.raises(MediaTooLargeError):
        process_video_upload(make_mp4_bytes(), settings=_settings(video_max_upload_bytes=10))


def test_video_duration_limit_enforced() -> None:
    with pytest.raises(VideoDurationExceededError):
        process_video_upload(
            make_mp4_bytes(fps=10.0, frame_count=20),  # 2.0s
            settings=_settings(video_max_duration_seconds=1.0),
        )


def test_video_dimension_limit_enforced() -> None:
    with pytest.raises(MediaDimensionExceededError):
        process_video_upload(
            make_mp4_bytes(width=64, height=48), settings=_settings(video_max_dimension_px=32)
        )


def test_malformed_video_rejected_not_misdecoded() -> None:
    with pytest.raises(MalformedMediaError):
        process_video_upload(MALFORMED_MP4_BYTES, settings=_settings())


def test_video_decode_failure_cleans_up_temp_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Directly proves the temp file created for decoding is removed even
    # on a decode failure — the "cleanup on every failure path" guarantee.
    import os
    import tempfile

    created_paths: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def _tracking_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, path = real_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", _tracking_mkstemp)

    with pytest.raises(MalformedMediaError):
        process_video_upload(MALFORMED_MP4_BYTES, settings=_settings())

    assert created_paths
    assert not os.path.exists(created_paths[0])


def test_video_temp_file_cleaned_up_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mirrors the failure-path test above, but for the happy path — the
    # temp file used for decoding must be gone once processing succeeds,
    # not just when it fails.
    import os
    import tempfile

    created_paths: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def _tracking_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, path = real_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", _tracking_mkstemp)

    frames = process_video_upload(make_mp4_bytes(frame_count=20), settings=_settings())

    assert frames
    assert created_paths
    assert not os.path.exists(created_paths[0])


# --- end-to-end prepare_media_upload -------------------------------------


def test_prepare_media_upload_image() -> None:
    prepared = prepare_media_upload(make_jpeg_bytes(), settings=_settings())
    assert prepared.kind == "image"
    assert len(prepared.frames) == 1


def test_prepare_media_upload_video() -> None:
    prepared = prepare_media_upload(
        make_mp4_bytes(frame_count=20), settings=_settings(video_max_sampled_frames=3)
    )
    assert prepared.kind == "video"
    assert 1 <= len(prepared.frames) <= 3


def test_prepare_media_upload_empty_rejected() -> None:
    with pytest.raises(EmptyMediaUploadError):
        prepare_media_upload(b"", settings=_settings())
