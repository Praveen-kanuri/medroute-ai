"""Phase 2C: turns validated media frames into controlled VisionObservation
objects via the configured vision provider.

Best-effort only, mirroring the project's existing optional-provider
pattern (Groq routing/response rephrasing, Deepgram TTS): any missing
configuration, provider failure, timeout, or schema-invalid model output
is handled here and never propagated as a crash — the caller always gets
back a (possibly empty) list of already-validated observations plus a
short, safe, human-readable status note. This module never fabricates an
observation when analysis is unavailable or fails.
"""

import logging
from typing import Any

from pydantic import ValidationError

from app.config.settings import Settings
from app.providers.vision.base import VisionProvider
from app.schemas.vision import MediaSourceType, VisionObservation
from app.services.media_validation_service import MediaFrame

logger = logging.getLogger(__name__)

_NOT_CONFIGURED_NOTE = (
    "Visual analysis is not available right now; continuing using any text or voice "
    "information provided."
)
_ALL_FRAMES_FAILED_NOTE = (
    "The uploaded media could not be analyzed. Continuing using any text or voice "
    "information provided."
)
_NO_FINDINGS_NOTE = "No specific visual findings were identified from the uploaded media."


class VisionNotConfiguredError(Exception):
    """Raised when vision analysis is not enabled: either vision_mode is
    not "groq", or no Groq API key is configured."""


def build_vision_provider(settings: Settings) -> VisionProvider:
    """Construct the configured vision provider.

    Mirrors routing_mode/response_mode's opt-in gate: vision_mode must be
    explicitly "groq" (not just a configured API key) before a real
    provider is built — the default "deterministic" never calls a vision
    model, so a real Groq API key present in the environment (e.g. for
    local development) never causes an automated test to make a live
    vision call. The Groq import is local to this function so importing
    this module never requires the groq package to be exercised (e.g. in
    tests that inject a fake provider directly).
    """
    if not (settings.vision_mode == "groq" and settings.groq_configured):
        raise VisionNotConfiguredError("Vision analysis is not enabled or configured.")

    from app.providers.vision.groq import GroqVisionProvider

    assert settings.groq_api_key is not None
    return GroqVisionProvider(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_vision_model,
        timeout_seconds=settings.groq_vision_timeout_seconds,
        max_output_tokens=settings.vision_max_output_tokens,
    )


def _normalize_for_dedup(text: str) -> str:
    return " ".join(text.lower().split())


def _media_processed_note(kind: str, frame_count: int) -> str:
    label = "image" if kind == "image" else f"{frame_count} sampled video frame(s)"
    return (
        f"Reviewed the uploaded {label}; see the visual observations below. This is a "
        "visual review only, not a diagnosis."
    )


async def analyze_media_frames(
    frames: list[MediaFrame],
    *,
    kind: str,
    settings: Settings,
    provider: VisionProvider | None = None,
) -> tuple[list[VisionObservation], str]:
    """Analyze each frame and return deduplicated, schema-validated
    observations plus a short status note. Never raises — any provider or
    validation failure is absorbed here.
    """
    if provider is None:
        try:
            provider = build_vision_provider(settings)
        except VisionNotConfiguredError:
            return [], _NOT_CONFIGURED_NOTE

    source_type = MediaSourceType.IMAGE if kind == "image" else MediaSourceType.VIDEO_FRAME
    collected: list[VisionObservation] = []
    seen: set[str] = set()
    failed_frames = 0

    for frame in frames:
        if len(collected) >= settings.vision_max_observations:
            break
        try:
            raw_observations = await provider.analyze_image(frame.data, content_type="image/jpeg")
        except Exception:
            logger.warning("vision_frame_analysis_failed kind=%s", kind)
            failed_frames += 1
            continue

        for raw in raw_observations:
            if len(collected) >= settings.vision_max_observations:
                break
            if not isinstance(raw, dict):
                continue

            candidate: dict[str, Any] = {
                **raw,
                "source_type": source_type.value,
                "frame_timestamp_seconds": (frame.timestamp_seconds if kind == "video" else None),
            }
            try:
                observation = VisionObservation.model_validate(candidate)
            except ValidationError:
                logger.warning("vision_observation_rejected_by_schema kind=%s", kind)
                continue

            key = _normalize_for_dedup(observation.visual_description)
            if key in seen:
                continue
            seen.add(key)
            collected.append(observation)

    if not collected:
        if frames and failed_frames == len(frames):
            return [], _ALL_FRAMES_FAILED_NOTE
        return [], _NO_FINDINGS_NOTE

    return collected, _media_processed_note(kind, len(frames))
