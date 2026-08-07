"""Phase 2C: tests for vision-provider orchestration.

Never calls Groq — every test injects a FakeVisionProvider directly.
Proves: schema-valid observations are accepted, schema-invalid/diagnostic
model output is rejected (not sanitized into a fabricated-but-safe
substitute), substantially repeated observations across frames are
deduplicated, a missing configuration or provider failure yields an empty
list plus a safe note rather than raising or fabricating, and the observed
frame_timestamp_seconds/source_type are set correctly for images vs. video
frames.
"""

import pytest

from app.config.settings import Settings
from app.providers.vision.fake import FakeVisionProvider
from app.schemas.vision import MediaSourceType
from app.services.media_validation_service import MediaFrame
from app.services.vision_analysis_service import (
    VisionNotConfiguredError,
    analyze_media_frames,
    build_vision_provider,
)

_VALID_OBSERVATION = {
    "observation_type": "skin_appearance",
    "body_area": "forearm",
    "visual_description": "mild redness on the forearm",
    "visible_attributes": ["redness"],
    "confidence": "low",
    "limitations": "a visual review only, not a diagnosis",
}


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


async def test_valid_observation_accepted_for_image() -> None:
    frame = MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])

    observations, note = await analyze_media_frames(
        [frame], kind="image", settings=_settings(), provider=provider
    )

    assert len(observations) == 1
    assert observations[0].source_type == MediaSourceType.IMAGE
    assert observations[0].frame_timestamp_seconds is None
    assert "not a diagnosis" in note


async def test_valid_observation_accepted_for_video_frame_with_timestamp() -> None:
    frame = MediaFrame(data=b"fake-jpeg-frame", timestamp_seconds=1.5)
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])

    observations, note = await analyze_media_frames(
        [frame], kind="video", settings=_settings(), provider=provider
    )

    assert len(observations) == 1
    assert observations[0].source_type == MediaSourceType.VIDEO_FRAME
    assert observations[0].frame_timestamp_seconds == 1.5


async def test_schema_invalid_observation_rejected_not_fabricated() -> None:
    diagnostic_observation = {**_VALID_OBSERVATION, "visual_description": "diagnosis of eczema"}
    frame = MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)
    provider = FakeVisionProvider(observations=[diagnostic_observation])

    observations, note = await analyze_media_frames(
        [frame], kind="image", settings=_settings(), provider=provider
    )

    assert observations == []
    assert "no specific visual findings" in note.lower() or "could not be analyzed" in note.lower()


async def test_malformed_shape_observation_rejected() -> None:
    frame = MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)
    provider = FakeVisionProvider(observations=[{"unexpected": "shape"}])

    observations, _note = await analyze_media_frames(
        [frame], kind="image", settings=_settings(), provider=provider
    )

    assert observations == []


async def test_non_dict_items_in_provider_output_ignored() -> None:
    frame = MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)
    provider = FakeVisionProvider(observations=["not-a-dict", 42, None])

    observations, _note = await analyze_media_frames(
        [frame], kind="image", settings=_settings(), provider=provider
    )

    assert observations == []


async def test_substantially_repeated_observations_deduplicated() -> None:
    frame_a = MediaFrame(data=b"frame-a", timestamp_seconds=0.0)
    frame_b = MediaFrame(data=b"frame-b", timestamp_seconds=1.0)
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])

    observations, _note = await analyze_media_frames(
        [frame_a, frame_b], kind="video", settings=_settings(), provider=provider
    )

    assert len(observations) == 1


async def test_observation_count_capped_by_settings() -> None:
    many_frames = [
        MediaFrame(data=f"frame-{i}".encode(), timestamp_seconds=float(i)) for i in range(5)
    ]

    def _distinct_observation(index: int) -> dict[str, object]:
        return {**_VALID_OBSERVATION, "visual_description": f"distinct finding number {index}"}

    class _SequentialProvider:
        def __init__(self) -> None:
            self._calls = 0

        async def analyze_image(self, image_bytes: bytes, *, content_type: str = "image/jpeg"):
            observation = _distinct_observation(self._calls)
            self._calls += 1
            return [observation]

    observations, _note = await analyze_media_frames(
        many_frames,
        kind="video",
        settings=_settings(vision_max_observations=2),
        provider=_SequentialProvider(),
    )

    assert len(observations) == 2


async def test_provider_not_configured_returns_empty_with_safe_note() -> None:
    observations, note = await analyze_media_frames(
        [MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)],
        kind="image",
        settings=_settings(groq_api_key=None),
        provider=None,
    )
    assert observations == []
    assert "not available" in note.lower()


async def test_all_frames_failing_returns_empty_with_safe_note() -> None:
    provider = FakeVisionProvider(error=RuntimeError("synthetic provider failure"))
    observations, note = await analyze_media_frames(
        [MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)],
        kind="image",
        settings=_settings(),
        provider=provider,
    )
    assert observations == []
    assert "could not be analyzed" in note.lower()


async def test_partial_frame_failure_still_returns_successful_observations() -> None:
    calls = {"n": 0}

    class _FlakyProvider:
        async def analyze_image(self, image_bytes: bytes, *, content_type: str = "image/jpeg"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic failure on first frame")
            return [_VALID_OBSERVATION]

    frames = [
        MediaFrame(data=b"frame-1", timestamp_seconds=0.0),
        MediaFrame(data=b"frame-2", timestamp_seconds=1.0),
    ]
    observations, note = await analyze_media_frames(
        frames, kind="video", settings=_settings(), provider=_FlakyProvider()
    )
    assert len(observations) == 1
    assert "not a diagnosis" in note.lower()


def test_build_vision_provider_raises_when_not_configured() -> None:
    with pytest.raises(VisionNotConfiguredError):
        build_vision_provider(_settings(groq_api_key=None))


def test_build_vision_provider_succeeds_when_configured() -> None:
    provider = build_vision_provider(
        _settings(vision_mode="groq", groq_api_key="fake-test-key-not-real")
    )
    assert provider is not None


def test_build_vision_provider_raises_when_key_present_but_mode_not_groq() -> None:
    # Mirrors routing_mode/response_mode: a real API key alone must never
    # be enough to trigger a live call — vision_mode must be explicitly
    # "groq" too. This is what keeps automated tests (and any environment
    # with a real GROQ_API_KEY configured for local development) from ever
    # making a live vision call by default.
    with pytest.raises(VisionNotConfiguredError):
        build_vision_provider(_settings(groq_api_key="fake-test-key-not-real"))
