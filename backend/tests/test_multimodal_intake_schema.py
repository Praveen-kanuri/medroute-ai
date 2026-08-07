"""Unit tests for Phase 1C request validation and normalization.

Pure Pydantic — no FastAPI app, no database, no network, no model API.
"""

import pytest
from pydantic import ValidationError

from app.schemas.multimodal_intake import (
    Duration,
    DurationUnit,
    Location,
    MultimodalIntakeRequest,
    VisionInputs,
    VoiceInput,
)


def test_valid_text_only_intake() -> None:
    request = MultimodalIntakeRequest(symptoms=["knee pain"], main_concern="Discomfort")
    assert request.symptoms == ["knee pain"]
    assert request.main_concern == "Discomfort"


def test_valid_voice_only_intake() -> None:
    request = MultimodalIntakeRequest(voice_input=VoiceInput(transcript="synthetic transcript"))
    assert request.voice_input is not None
    assert request.voice_input.transcript == "synthetic transcript"


def test_valid_image_only_intake() -> None:
    request = MultimodalIntakeRequest(
        vision_inputs=VisionInputs(image_urls=["https://media.example.org/intake/a.jpg"])
    )
    assert request.vision_inputs is not None
    assert request.vision_inputs.image_urls == ["https://media.example.org/intake/a.jpg"]


def test_valid_video_only_intake() -> None:
    request = MultimodalIntakeRequest(
        vision_inputs=VisionInputs(video_urls=["https://media.example.org/intake/a.mp4"])
    )
    assert request.vision_inputs is not None
    assert request.vision_inputs.video_urls == ["https://media.example.org/intake/a.mp4"]


def test_valid_mixed_modality_intake() -> None:
    request = MultimodalIntakeRequest(
        symptoms=["knee pain"],
        voice_input=VoiceInput(transcript="synthetic transcript"),
        vision_inputs=VisionInputs(image_urls=["https://media.example.org/intake/a.jpg"]),
    )
    assert request.symptoms == ["knee pain"]
    assert request.voice_input is not None
    assert request.vision_inputs is not None


def test_symptoms_whitespace_is_normalized() -> None:
    request = MultimodalIntakeRequest(symptoms=["  knee   pain  "])
    assert request.symptoms == ["knee pain"]


def test_duplicate_symptoms_removed_case_insensitively() -> None:
    request = MultimodalIntakeRequest(symptoms=["Knee Pain", "knee pain", "swelling"])
    assert request.symptoms == ["Knee Pain", "swelling"]


def test_symptom_order_preserved_after_dedup() -> None:
    request = MultimodalIntakeRequest(symptoms=["swelling", "knee pain", "SWELLING"])
    assert request.symptoms == ["swelling", "knee pain"]


def test_symptom_blank_entry_rejected() -> None:
    with pytest.raises(ValidationError):
        MultimodalIntakeRequest(symptoms=["knee pain", "   "])


def test_symptom_count_limit() -> None:
    with pytest.raises(ValidationError):
        MultimodalIntakeRequest(symptoms=[f"symptom {i}" for i in range(11)])


def test_symptom_count_limit_boundary_allowed() -> None:
    request = MultimodalIntakeRequest(symptoms=[f"symptom {i}" for i in range(10)])
    assert len(request.symptoms) == 10


def test_symptom_length_limit() -> None:
    with pytest.raises(ValidationError):
        MultimodalIntakeRequest(symptoms=["x" * 201])


def test_main_concern_length_limit() -> None:
    with pytest.raises(ValidationError):
        MultimodalIntakeRequest(main_concern="x" * 301)


def test_main_concern_blank_becomes_none() -> None:
    request = MultimodalIntakeRequest(main_concern="   ")
    assert request.main_concern is None


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MultimodalIntakeRequest.model_validate({"unexpected_field": "value"})


def test_unknown_nested_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MultimodalIntakeRequest.model_validate(
            {"duration": {"value": 3, "unit": "days", "extra": "nope"}}
        )


@pytest.mark.parametrize("unit", ["hours", "days", "weeks", "months", "years"])
def test_duration_accepts_each_unit(unit: str) -> None:
    duration = Duration(value=1, unit=DurationUnit(unit))
    assert duration.unit == unit


def test_duration_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        Duration(value=0, unit=DurationUnit.DAYS)


def test_duration_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        Duration(value=-1, unit=DurationUnit.DAYS)


def test_duration_rejects_unsupported_unit() -> None:
    with pytest.raises(ValidationError):
        Duration.model_validate({"value": 1, "unit": "decades"})


def test_duration_rejects_unreasonably_large_value() -> None:
    with pytest.raises(ValidationError):
        Duration(value=10_000, unit=DurationUnit.DAYS)


def test_duration_rejects_non_integer_value() -> None:
    with pytest.raises(ValidationError):
        Duration.model_validate({"value": 3.5, "unit": "days"})


def test_location_normalizes_state_to_uppercase() -> None:
    location = Location(state="tx")
    assert location.state == "TX"


def test_location_normalizes_country_to_uppercase() -> None:
    location = Location(country="us")
    assert location.country == "US"


def test_location_defaults_country_to_us() -> None:
    location = Location()
    assert location.country == "US"


def test_location_preserves_leading_zero_postal_code() -> None:
    location = Location(postal_code="00501")
    assert location.postal_code == "00501"


def test_location_all_fields_optional() -> None:
    location = Location()
    assert location.city is None
    assert location.state is None
    assert location.postal_code is None


def test_preferred_specialty_normalized_to_lowercase() -> None:
    request = MultimodalIntakeRequest(preferred_specialty="Cardiology")
    assert request.preferred_specialty == "cardiology"


def test_preferred_specialty_rejects_malformed_slug() -> None:
    with pytest.raises(ValidationError):
        MultimodalIntakeRequest(preferred_specialty="Not A Slug!")


def test_preferred_specialty_accepts_hyphenated_slug() -> None:
    request = MultimodalIntakeRequest(preferred_specialty="general-surgery")
    assert request.preferred_specialty == "general-surgery"


def test_preferred_specialty_blank_becomes_none() -> None:
    request = MultimodalIntakeRequest(preferred_specialty="   ")
    assert request.preferred_specialty is None


def test_voice_language_defaults_to_en() -> None:
    voice_input = VoiceInput(transcript="synthetic transcript")
    assert voice_input.language == "en"


@pytest.mark.parametrize("language", ["en", "en-US", "es", "zh-Hant"])
def test_voice_language_accepts_valid_tags(language: str) -> None:
    voice_input = VoiceInput(language=language)
    assert voice_input.language == language


def test_voice_language_rejects_invalid_tag() -> None:
    with pytest.raises(ValidationError):
        VoiceInput.model_validate({"language": "not_a_valid_tag_at_all"})


def test_voice_transcript_length_bound() -> None:
    with pytest.raises(ValidationError):
        VoiceInput(transcript="x" * 5001)


def test_voice_transcript_blank_becomes_none() -> None:
    voice_input = VoiceInput(transcript="   ")
    assert voice_input.transcript is None


def test_image_url_limit() -> None:
    urls = [f"https://media.example.org/intake/{i}.jpg" for i in range(6)]
    with pytest.raises(ValidationError):
        VisionInputs(image_urls=urls)


def test_image_url_limit_boundary_allowed() -> None:
    urls = [f"https://media.example.org/intake/{i}.jpg" for i in range(5)]
    vision_inputs = VisionInputs(image_urls=urls)
    assert len(vision_inputs.image_urls) == 5


def test_video_url_limit() -> None:
    urls = [f"https://media.example.org/intake/{i}.mp4" for i in range(3)]
    with pytest.raises(ValidationError):
        VisionInputs(video_urls=urls)


def test_video_url_limit_boundary_allowed() -> None:
    urls = [f"https://media.example.org/intake/{i}.mp4" for i in range(2)]
    vision_inputs = VisionInputs(video_urls=urls)
    assert len(vision_inputs.video_urls) == 2


def test_url_rejects_non_https_scheme() -> None:
    with pytest.raises(ValidationError):
        VisionInputs(image_urls=["http://media.example.org/intake/a.jpg"])


def test_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValidationError):
        VisionInputs(image_urls=["https://user:pass@media.example.org/intake/a.jpg"])


def test_url_rejects_localhost() -> None:
    with pytest.raises(ValidationError):
        VisionInputs(image_urls=["https://localhost/intake/a.jpg"])


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "169.254.1.1",
        "10.0.0.5",
        "192.168.1.1",
        "224.0.0.1",
        "0.0.0.0",
        "[::1]",
    ],
)
def test_url_rejects_private_or_reserved_ip_literals(host: str) -> None:
    with pytest.raises(ValidationError):
        VisionInputs(image_urls=[f"https://{host}/intake/a.jpg"])


def test_url_rejects_fragment() -> None:
    with pytest.raises(ValidationError):
        VisionInputs(image_urls=["https://media.example.org/intake/a.jpg#section"])


def test_url_rejects_query_string() -> None:
    with pytest.raises(ValidationError):
        VisionInputs(image_urls=["https://media.example.org/intake/a.jpg?token=abc123"])


def test_url_ordinary_public_hostname_accepted() -> None:
    vision_inputs = VisionInputs(image_urls=["https://media.example.org/intake/a.jpg"])
    assert vision_inputs.image_urls == ["https://media.example.org/intake/a.jpg"]


def test_duplicate_media_urls_removed_preserving_order() -> None:
    url_a = "https://media.example.org/intake/a.jpg"
    url_b = "https://media.example.org/intake/b.jpg"
    vision_inputs = VisionInputs(image_urls=[url_a, url_b, url_a])
    assert vision_inputs.image_urls == [url_a, url_b]


def test_url_exceeding_max_length_rejected() -> None:
    long_url = "https://media.example.org/" + ("a" * 2100) + ".jpg"
    with pytest.raises(ValidationError):
        VisionInputs(image_urls=[long_url])


def test_emergency_signals_deduplicated() -> None:
    request = MultimodalIntakeRequest.model_validate(
        {
            "emergency_signals": [
                "chest_pain_or_pressure",
                "chest_pain_or_pressure",
                "difficulty_breathing",
            ]
        }
    )
    assert request.emergency_signals == ["chest_pain_or_pressure", "difficulty_breathing"]


def test_emergency_signals_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        MultimodalIntakeRequest.model_validate({"emergency_signals": ["not_a_real_signal"]})


def test_empty_request_is_structurally_valid() -> None:
    request = MultimodalIntakeRequest()
    assert request.symptoms == []
    assert request.emergency_concern is False
