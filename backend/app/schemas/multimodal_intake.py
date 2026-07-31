"""Phase 1C multimodal intake contract.

This module defines a stateless, non-diagnostic intake request/response
shape. It validates and normalizes user-supplied text, a pre-generated
voice transcript, and image/video URL *references* — it never interprets
any of it. Speech-to-text, vision analysis, LLM calls, specialty
inference, and persistence are all out of scope here; see
docs/roadmap.md's Phase 1D notes.

Callers must not submit identifying information (name, date of birth,
phone, email, SSN, insurance ID, medical-record ID, payment details, or
other government identifiers) — this module does not attempt to detect
or redact such data, it only validates the fields defined below.
"""

import ipaddress
import re
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAX_SYMPTOMS = 10
_MAX_SYMPTOM_LENGTH = 200
_MAX_MAIN_CONCERN_LENGTH = 300
_MAX_TRANSCRIPT_LENGTH = 5000
_MAX_SPECIALTY_SLUG_LENGTH = 64
_MAX_IMAGE_URLS = 5
_MAX_VIDEO_URLS = 2
_MAX_URL_LENGTH = 2048

_SPECIALTY_SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
# Conservative language-tag format: "en", "en-US", "zh-Hant", etc.
_LANGUAGE_TAG_PATTERN = r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$"


class DurationUnit(StrEnum):
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"


class EmergencySignal(StrEnum):
    """User-declared only. MedRoute AI never infers these from text, a
    transcript, or media, and this list is not medically exhaustive."""

    DIFFICULTY_BREATHING = "difficulty_breathing"
    CHEST_PAIN_OR_PRESSURE = "chest_pain_or_pressure"
    NEW_CONFUSION_OR_UNRESPONSIVE = "new_confusion_or_unresponsive"
    BLUE_OR_GRAY_LIPS_OR_FACE = "blue_or_gray_lips_or_face"
    OTHER_IMMEDIATE_DANGER = "other_immediate_danger"


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = " ".join(value.split())
    return trimmed or None


def _validate_media_url(url: str) -> str:
    """Contract-level hardening only — not a complete SSRF defense.

    Performs string-level checks with no DNS resolution and no network
    access. A future media-processing service must independently
    revalidate destinations at fetch time.
    """
    if len(url) > _MAX_URL_LENGTH:
        raise ValueError("URL exceeds maximum length")

    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError("URL must use https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not contain embedded credentials")
    if parsed.fragment:
        raise ValueError("URL must not contain a fragment")
    if parsed.query:
        raise ValueError("URL must not contain a query string")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a host")
    if hostname.lower() == "localhost":
        raise ValueError("URL must not reference localhost")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        pass  # Not an IP literal — an ordinary hostname, allowed.
    else:
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_private
        ):
            raise ValueError("URL must not reference a private or reserved network address")

    return url


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class Duration(BaseModel):
    """How long the user has been experiencing their main concern."""

    model_config = ConfigDict(extra="forbid")

    value: int = Field(gt=0, le=1000)
    unit: DurationUnit


class Location(BaseModel):
    """Used for future provider-discovery preferences only — never for
    medical reasoning."""

    model_config = ConfigDict(extra="forbid")

    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str | None = "US"

    @field_validator("city", "postal_code", mode="before")
    @classmethod
    def _trim(cls, value: str | None) -> str | None:
        return _blank_to_none(value)

    @field_validator("state", "country", mode="before")
    @classmethod
    def _trim_and_uppercase(cls, value: str | None) -> str | None:
        trimmed = _blank_to_none(value)
        return trimmed.upper() if trimmed is not None else None


class VoiceInput(BaseModel):
    """A transcript the *caller* already generated. MedRoute AI does not
    perform speech-to-text and does not verify this transcript's accuracy."""

    model_config = ConfigDict(extra="forbid")

    transcript: str | None = Field(default=None, max_length=_MAX_TRANSCRIPT_LENGTH)
    language: str = Field(default="en", pattern=_LANGUAGE_TAG_PATTERN)

    @field_validator("transcript", mode="before")
    @classmethod
    def _normalize_transcript(cls, value: str | None) -> str | None:
        return _blank_to_none(value)


class VisionInputs(BaseModel):
    """Image/video URL *references* only. MedRoute AI does not fetch,
    open, probe, or analyze this media in Phase 1C."""

    model_config = ConfigDict(extra="forbid")

    image_urls: list[str] = Field(default_factory=list, max_length=_MAX_IMAGE_URLS)
    video_urls: list[str] = Field(default_factory=list, max_length=_MAX_VIDEO_URLS)

    @field_validator("image_urls", "video_urls")
    @classmethod
    def _validate_urls(cls, urls: list[str]) -> list[str]:
        validated = [_validate_media_url(url) for url in urls]
        return _dedupe_preserve_order(validated)


class MultimodalIntakeRequest(BaseModel):
    """Phase 1C intake request. Validated and normalized only — never
    interpreted. Do not submit identifying information (name, date of
    birth, phone, email, SSN, insurance ID, medical-record ID, payment
    details, or other government identifiers)."""

    model_config = ConfigDict(extra="forbid")

    symptoms: list[str] = Field(default_factory=list, max_length=_MAX_SYMPTOMS)
    main_concern: str | None = Field(default=None, max_length=_MAX_MAIN_CONCERN_LENGTH)
    duration: Duration | None = None
    location: Location | None = None
    preferred_specialty: str | None = Field(default=None, max_length=_MAX_SPECIALTY_SLUG_LENGTH)
    emergency_concern: bool = False
    emergency_signals: list[EmergencySignal] = Field(default_factory=list)
    voice_input: VoiceInput | None = None
    vision_inputs: VisionInputs | None = None

    @field_validator("symptoms")
    @classmethod
    def _normalize_symptoms(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in values:
            collapsed = " ".join(raw.split())
            if not collapsed:
                raise ValueError("symptom entries must not be blank")
            if len(collapsed) > _MAX_SYMPTOM_LENGTH:
                raise ValueError(f"symptom entry exceeds {_MAX_SYMPTOM_LENGTH} characters")
            normalized.append(collapsed)

        seen: set[str] = set()
        deduped: list[str] = []
        for entry in normalized:
            key = entry.casefold()
            if key not in seen:
                seen.add(key)
                deduped.append(entry)
        return deduped

    @field_validator("main_concern", mode="before")
    @classmethod
    def _normalize_main_concern(cls, value: str | None) -> str | None:
        return _blank_to_none(value)

    @field_validator("preferred_specialty", mode="before")
    @classmethod
    def _normalize_preferred_specialty(cls, value: str | None) -> str | None:
        return _blank_to_none(value.lower()) if value is not None else None

    @field_validator("preferred_specialty")
    @classmethod
    def _validate_specialty_slug(cls, value: str | None) -> str | None:
        if value is not None and not re.match(_SPECIALTY_SLUG_PATTERN, value):
            raise ValueError(
                "preferred_specialty must be a normalized slug (lowercase, hyphenated)"
            )
        return value

    @field_validator("emergency_signals")
    @classmethod
    def _dedupe_emergency_signals(cls, values: list[EmergencySignal]) -> list[EmergencySignal]:
        seen: set[EmergencySignal] = set()
        result: list[EmergencySignal] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result


class IntakeStatus(StrEnum):
    EMERGENCY = "emergency"
    NEEDS_CLARIFICATION = "needs_clarification"
    READY_FOR_MULTIMODAL_PROCESSING = "ready_for_multimodal_processing"


class MultimodalStatus(BaseModel):
    text_received: bool
    voice_received: bool
    vision_received: bool
    image_count: int
    video_count: int


class NextAction(BaseModel):
    code: str
    message: str


class NormalizedIntake(BaseModel):
    """Echoes back only validated, normalized, user-supplied fields — no
    inferred symptoms, visual observations, medical conclusions, or
    generated specialty."""

    symptoms: list[str]
    main_concern: str | None
    duration: Duration | None
    location: Location | None
    preferred_specialty: str | None
    emergency_concern: bool
    emergency_signals: list[EmergencySignal]
    voice_input: VoiceInput | None
    vision_inputs: VisionInputs | None


class MultimodalIntakeResponse(BaseModel):
    intake_id: str
    status: IntakeStatus
    normalized_intake: NormalizedIntake
    multimodal_status: MultimodalStatus
    missing_fields: list[str]
    clarification_questions: list[str]
    next_action: NextAction
    safety_message: str | None
    disclaimer: str
