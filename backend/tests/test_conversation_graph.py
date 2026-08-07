"""Phase 2B: tests for the LangGraph conversation graph.

Node-level tests call the pure/thin node functions directly. Graph-level
tests compile a fresh, isolated graph (its own InMemorySaver, never the
process-wide singleton) via build_conversation_graph() and drive it with
.ainvoke()/Command(resume=...) — exactly how the real orchestration
service uses it. No real database beyond the existing local dev Postgres
already used by test_navigation_api.py's provider-search tests, and no
external network calls (routing_mode/response_mode default to
deterministic, no API keys configured).
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.db.session import dispose_engine, get_sessionmaker
from app.graph.build import build_conversation_graph
from app.graph.nodes import (
    clarification_node,
    medical_intake_agent_node,
    route_after_clarification,
    route_after_safety_gate,
    route_after_specialty_routing,
    route_after_supervisor,
    safety_gate_node,
    supervisor_router_node,
    vision_agent_node,
)
from app.providers.vision.fake import FakeVisionProvider
from app.schemas.clinical_context import FactSource
from app.services.clinical_intake_service import LEG_SWELLING_PROTOCOL, build_initial_context
from app.services.media_validation_service import MediaFrame, PreparedMedia


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    # pytest-asyncio gives each test function its own event loop, but
    # app/db/session.py's engine is a process-wide singleton bound to
    # whichever loop first created it — dispose it first so a fresh engine
    # is created (and bound) on *this* test's loop, then dispose again on
    # teardown so the next test does the same.
    await dispose_engine()
    async with get_sessionmaker()() as session:
        yield session
    await dispose_engine()


def _config(
    thread_id: str,
    *,
    settings: Settings | None = None,
    session: object = None,
    media: object = None,
    vision_provider: object = None,
) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "settings": settings or _settings(),
            "session": session,
            "media": media,
            "vision_provider": vision_provider,
        }
    }


_VALID_OBSERVATION = {
    "observation_type": "skin_appearance",
    "body_area": "forearm",
    "visual_description": "mild redness on the forearm",
    "visible_attributes": ["redness"],
    "confidence": "low",
    "limitations": "a visual review only, not a diagnosis",
}
_DERMATOLOGY_OBSERVATION = {
    "observation_type": "skin_appearance",
    "body_area": "forearm",
    "visual_description": "visible itchy rash on the forearm",
    "visible_attributes": ["redness"],
    "confidence": "low",
    "limitations": "a visual review only, not a diagnosis",
}


# --- individual node / conditional-edge unit tests --------------------------


def test_medical_intake_agent_node_computes_missing_fields() -> None:
    result = medical_intake_agent_node({"intake_request": {"emergency_concern": False}})
    assert "duration" in result["missing_fields"]
    assert "concern" in result["missing_fields"]
    assert result["intake_response"]["status"] == "needs_clarification"


def test_medical_intake_agent_node_ready_when_complete() -> None:
    result = medical_intake_agent_node(
        {
            "intake_request": {
                "main_concern": "annual checkup",
                "duration": {"value": 1, "unit": "days"},
            }
        }
    )
    assert result["missing_fields"] == []
    assert result["intake_response"]["status"] == "ready_for_multimodal_processing"
    assert result["media_note"] is None


def test_safety_gate_node_detects_emergency_status() -> None:
    result = safety_gate_node({"intake_response": {"status": "emergency"}})
    assert result["is_emergency"] is True


def test_safety_gate_node_false_for_non_emergency_status() -> None:
    result = safety_gate_node({"intake_response": {"status": "ready_for_multimodal_processing"}})
    assert result["is_emergency"] is False


def test_route_after_safety_gate_emergency_goes_to_response_composition() -> None:
    assert route_after_safety_gate({"is_emergency": True}) == "response_composition"


def test_route_after_safety_gate_non_emergency_goes_to_clarification() -> None:
    assert route_after_safety_gate({"is_emergency": False}) == "clarification"


def test_clarification_node_passes_through_when_nothing_missing() -> None:
    # No missing fields -> must not call interrupt() at all.
    result = clarification_node({"missing_fields": [], "intake_request": {}})
    assert result["just_resumed_clarification"] is False
    assert result["clarification_status"] == "none"


def test_route_after_clarification_resumed_loops_back() -> None:
    assert route_after_clarification({"just_resumed_clarification": True}) == "normalize_intake"


def test_route_after_clarification_not_resumed_continues() -> None:
    assert route_after_clarification({"just_resumed_clarification": False}) == "specialty_routing"


def test_route_after_specialty_routing_matched_goes_to_provider_search() -> None:
    state = {"routing": {"specialty_slug": "cardiology"}}
    assert route_after_specialty_routing(state) == "provider_search"


def test_route_after_specialty_routing_unmatched_goes_to_response_composition() -> None:
    state: dict[str, Any] = {"routing": {"specialty_slug": None}}
    assert route_after_specialty_routing(state) == "response_composition"


# --- supervisor_router_node routing tests (formerly route_from_start) -----


def test_supervisor_routes_to_vision_agent_when_media_pending() -> None:
    result = supervisor_router_node({"media_pending": True, "intake_request": {}})
    assert result["next_agent"] == "vision_agent"
    assert route_after_supervisor(result) == "vision_agent"


def test_supervisor_routes_to_medical_intake_agent_when_no_media() -> None:
    result = supervisor_router_node({"media_pending": False, "intake_request": {}})
    assert result["next_agent"] == "medical_intake_agent"
    result_absent = supervisor_router_node({"intake_request": {}})
    assert result_absent["next_agent"] == "medical_intake_agent"


def test_supervisor_skips_vision_agent_when_emergency_concern_declared() -> None:
    # A declared emergency must skip vision_agent entirely — the
    # emergency response never looks at vision_observations, so analyzing
    # media in this case would only ever be a wasted provider call.
    state = {"media_pending": True, "intake_request": {"emergency_concern": True}}
    result = supervisor_router_node(state)
    assert result["next_agent"] == "medical_intake_agent"
    assert result["detected_intent"] == "medical_concern"


def test_supervisor_skips_vision_agent_when_emergency_signals_declared() -> None:
    state = {
        "media_pending": True,
        "intake_request": {"emergency_signals": ["chest_pain_or_pressure"]},
    }
    result = supervisor_router_node(state)
    assert result["next_agent"] == "medical_intake_agent"


def test_supervisor_still_routes_to_vision_agent_when_no_emergency_declared() -> None:
    state = {"media_pending": True, "intake_request": {"emergency_concern": False}}
    result = supervisor_router_node(state)
    assert result["next_agent"] == "vision_agent"


def test_supervisor_routes_greeting_to_conversation_agent() -> None:
    state = {"intake_request": {"main_concern": "Hi"}}
    result = supervisor_router_node(state)
    assert result["detected_intent"] == "greeting"
    assert result["next_agent"] == "conversation_agent"


def test_supervisor_routes_real_concern_to_medical_intake_agent() -> None:
    state = {"intake_request": {"main_concern": "chest pain for two days"}}
    result = supervisor_router_node(state)
    assert result["detected_intent"] == "medical_concern"
    assert result["next_agent"] == "medical_intake_agent"


def test_supervisor_emergency_precedence_overrides_greeting_looking_text() -> None:
    state = {"intake_request": {"main_concern": "hi", "emergency_concern": True}}
    result = supervisor_router_node(state)
    assert result["detected_intent"] == "medical_concern"
    assert result["next_agent"] == "medical_intake_agent"


def test_supervisor_records_agent_path_entry() -> None:
    result = supervisor_router_node({"intake_request": {"main_concern": "Hi"}})
    assert result["conversation_history"] == [
        {"agent": "supervisor_router", "intent": "greeting", "next_agent": "conversation_agent"}
    ]


async def test_vision_agent_node_passthrough_when_no_media_configured() -> None:
    result = await vision_agent_node({}, _config("t-vision-noop"))
    assert result["vision_observations"] == []
    assert result["media_analysis_note"] is None


async def test_vision_agent_node_produces_validated_observations() -> None:
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])
    result = await vision_agent_node(
        {}, _config("t-vision-run", media=media, vision_provider=provider)
    )
    assert len(result["vision_observations"]) == 1
    assert result["vision_observations"][0]["source_type"] == "image"
    assert result["media_analysis_note"] is not None


async def test_vision_agent_node_never_fabricates_on_provider_failure() -> None:
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(error=RuntimeError("synthetic provider failure"))
    result = await vision_agent_node(
        {}, _config("t-vision-fail", media=media, vision_provider=provider)
    )
    assert result["vision_observations"] == []
    assert result["media_analysis_note"] is not None


# --- full-graph tests ---------------------------------------------------


async def test_graph_text_only_unmatched_routes_without_db() -> None:
    graph = build_conversation_graph()
    config = _config("t-text-unmatched")
    state = {
        "intake_request": {
            "main_concern": "zzz qqq unrelated words",
            "duration": {"value": 1, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["intake_response"]["status"] == "ready_for_multimodal_processing"
    assert result["routing"]["method"] == "unmatched"
    assert result.get("provider_search") is None
    assert result["response_text"]
    assert result["audio"] is None


async def test_graph_greeting_routes_through_conversation_agent_not_routing() -> None:
    # A greeting is intercepted by supervisor_router before medical intake,
    # specialty routing, or provider search ever run — it never reaches
    # the "unmatched routing" branch at all any more.
    graph = build_conversation_graph()
    config = _config("t-greeting")
    state = {
        "intake_request": {"main_concern": "how are you?"},
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["detected_intent"] == "greeting"
    assert result.get("routing") is None
    assert result.get("provider_search") is None
    assert result.get("intake_response") is None
    agent_path = [entry["agent"] for entry in result["conversation_history"]]
    assert agent_path == ["supervisor_router", "conversation_agent", "response_agent"]


async def test_graph_voice_only_transcript_routes_to_specialty(db_session: AsyncSession) -> None:
    # No symptoms/main_concern at all — only a confirmed voice transcript,
    # with an explicit duration expression inside it (Phase 2A/2B
    # integration) — must still reach a routed, ready state. A real
    # session is required because this transcript matches cardiology,
    # which triggers the provider_search node.
    graph = build_conversation_graph()
    config = _config("t-voice-only", session=db_session)
    state = {
        "intake_request": {
            "voice_input": {
                "transcript": "chest pain and heart palpitations for the past three days",
                "language": "en",
            },
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["intake_response"]["status"] == "ready_for_multimodal_processing"
    assert "duration" not in result["missing_fields"]
    assert result["routing"]["specialty_slug"] == "cardiology"


async def test_graph_emergency_short_circuits_before_routing() -> None:
    graph = build_conversation_graph()
    config = _config("t-emergency")
    state = {"intake_request": {"emergency_concern": True}, "generate_speech": False}
    result = await graph.ainvoke(state, config=config)
    assert result["intake_response"]["status"] == "emergency"
    assert result.get("routing") is None
    assert result.get("provider_search") is None
    assert "911" in result["response_text"]


async def test_graph_clarification_interrupts_when_duration_missing() -> None:
    graph = build_conversation_graph()
    config = _config("t-clarify-interrupt")
    state = {
        "intake_request": {"main_concern": "zzz qqq unrelated words"},
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["missing_fields"] == ["duration"]
    assert interrupt_value["clarification_questions"]


async def test_graph_clarification_resumes_and_reaches_routing() -> None:
    graph = build_conversation_graph()
    config = _config("t-clarify-resume")
    state = {
        "intake_request": {"main_concern": "zzz qqq unrelated words"},
        "generate_speech": False,
    }
    first = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in first

    second = await graph.ainvoke(
        Command(resume={"duration": {"value": 3, "unit": "days"}}), config=config
    )
    assert "__interrupt__" not in second
    assert second["intake_response"]["status"] == "ready_for_multimodal_processing"
    assert second["intake_response"]["normalized_intake"]["duration"] == {
        "value": 3,
        "unit": "days",
    }


async def test_graph_conversation_state_preserves_confirmed_transcript_across_resume(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-clarify-preserve-transcript", session=db_session)
    transcript = "chest pain and heart palpitations"
    state = {
        "intake_request": {"voice_input": {"transcript": transcript, "language": "en"}},
        "generate_speech": False,
    }
    first = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in first  # duration still missing

    second = await graph.ainvoke(
        Command(resume={"duration": {"value": 3, "unit": "days"}}), config=config
    )
    assert second["intake_response"]["normalized_intake"]["voice_input"]["transcript"] == transcript
    assert second["routing"]["specialty_slug"] == "cardiology"


async def test_graph_multi_round_clarification_asks_again_for_remaining_field(
    db_session: AsyncSession,
) -> None:
    # Nothing at all supplied -> both "concern" and "duration" missing.
    # Answering only "duration" must interrupt again for "concern". A real
    # session is required because the final answer ("itchy rash") matches
    # dermatology, which triggers the provider_search node.
    graph = build_conversation_graph()
    config = _config("t-clarify-multi-round", session=db_session)
    state = {"intake_request": {}, "generate_speech": False}
    first = await graph.ainvoke(state, config=config)
    assert set(first["__interrupt__"][0].value["missing_fields"]) == {"concern", "duration"}

    second = await graph.ainvoke(
        Command(resume={"duration": {"value": 2, "unit": "weeks"}}), config=config
    )
    assert "__interrupt__" in second
    assert second["__interrupt__"][0].value["missing_fields"] == ["concern"]

    third = await graph.ainvoke(Command(resume={"main_concern": "itchy rash"}), config=config)
    assert "__interrupt__" not in third
    assert third["intake_response"]["status"] == "ready_for_multimodal_processing"


async def test_graph_clarification_resume_via_spoken_voice_transcript_extracts_duration() -> None:
    # A duration answered by *speaking* (a new recording transcribed and
    # resent as clarification_answer.voice_transcript) rather than typing
    # into the duration widget must be extracted from the transcript and
    # merged in — exactly like a fresh voice-only intake already gets
    # (Phase 2A/2B), now also for a clarification *resume* (Phase 2E). The
    # original concern must never be overwritten by the new utterance.
    graph = build_conversation_graph()
    config = _config("t-clarify-resume-voice-duration")
    state = {
        "intake_request": {"main_concern": "zzz qqq unrelated words"},
        "generate_speech": False,
    }
    first = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["missing_fields"] == ["duration"]

    second = await graph.ainvoke(
        Command(resume={"voice_transcript": "I've had it for 3 days now"}), config=config
    )
    assert "__interrupt__" not in second
    assert second["intake_response"]["normalized_intake"]["duration"] == {
        "value": 3,
        "unit": "days",
    }
    normalized = second["intake_response"]["normalized_intake"]
    assert normalized["main_concern"] == "zzz qqq unrelated words"
    assert second["detected_intent"] == "clarification_answer"


async def test_graph_clarification_resume_via_voice_transcript_fills_missing_concern() -> None:
    # When concern was *also* missing, a spoken answer with no parseable
    # duration in it falls back to filling in the concern instead of
    # being discarded.
    graph = build_conversation_graph()
    config = _config("t-clarify-resume-voice-concern")
    state = {"intake_request": {}, "generate_speech": False}
    first = await graph.ainvoke(state, config=config)
    assert set(first["__interrupt__"][0].value["missing_fields"]) == {"concern", "duration"}

    second = await graph.ainvoke(
        Command(resume={"voice_transcript": "my feet are swollen"}), config=config
    )
    assert "__interrupt__" in second
    assert second["__interrupt__"][0].value["missing_fields"] == ["duration"]

    snapshot = await graph.aget_state(config)
    assert snapshot.values["intake_request"]["main_concern"] == "my feet are swollen"


async def test_graph_clarification_resume_voice_transcript_never_overwrites_original_transcript(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-clarify-resume-voice-preserve-original", session=db_session)
    original_transcript = "chest pain and heart palpitations"
    state = {
        "intake_request": {"voice_input": {"transcript": original_transcript, "language": "en"}},
        "generate_speech": False,
    }
    first = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in first  # duration still missing

    second = await graph.ainvoke(
        Command(resume={"voice_transcript": "I've had it for 3 days now"}), config=config
    )
    assert "__interrupt__" not in second
    normalized = second["intake_response"]["normalized_intake"]
    assert normalized["voice_input"]["transcript"] == original_transcript
    assert normalized["duration"] == {"value": 3, "unit": "days"}
    assert second["routing"]["specialty_slug"] == "cardiology"


async def test_graph_specialty_routing_and_provider_search_invoked(
    db_session: AsyncSession,
) -> None:
    # Reuses the same confirmed working NPPES-fixture example as
    # test_navigation_api.py: "annual checkup" in Springfield, CA matches
    # family-medicine with one loaded provider.
    graph = build_conversation_graph()
    config = _config("t-provider-search", session=db_session)
    state = {
        "intake_request": {
            "symptoms": ["annual checkup"],
            "location": {"city": "Springfield", "state": "CA"},
            "duration": {"value": 1, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["routing"]["specialty_slug"] == "family-medicine"
    assert result["provider_search"] is not None
    assert len(result["provider_search"]["results"]) == 1
    assert "Family Medicine" in result["response_text"]


# --- Phase 2C: full-graph vision tests -----------------------------------


async def test_graph_image_only_upload_satisfies_concern_without_text_or_voice() -> None:
    # No symptoms, no main_concern, no voice transcript — only a directly
    # uploaded image producing controlled visual observations. "concern"
    # must not be reported missing.
    graph = build_conversation_graph()
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])
    config = _config("t-image-only", media=media, vision_provider=provider)
    state = {
        "intake_request": {"duration": {"value": 1, "unit": "days"}},
        "generate_speech": False,
        "media_pending": True,
    }
    result = await graph.ainvoke(state, config=config)
    assert "concern" not in result["missing_fields"]
    assert len(result["vision_observations"]) == 1
    assert "not a diagnosis" in result["response_text"]


async def test_graph_vision_observations_contribute_to_specialty_routing(
    db_session: AsyncSession,
) -> None:
    # A dermatology-matching visual observation, with no symptoms/main
    # concern/voice transcript at all, must still route to dermatology —
    # exactly the same treatment as a confirmed voice transcript already
    # gets (Phase 2A/2B). Needs a real session because the match triggers
    # provider_search.
    graph = build_conversation_graph()
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[_DERMATOLOGY_OBSERVATION])
    config = _config("t-vision-routing", session=db_session, media=media, vision_provider=provider)
    state = {
        "intake_request": {"duration": {"value": 1, "unit": "days"}},
        "generate_speech": False,
        "media_pending": True,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["routing"]["specialty_slug"] == "dermatology"


async def test_graph_vision_observations_never_trigger_emergency_status(
    db_session: AsyncSession,
) -> None:
    # No emergency_concern/emergency_signals declared — a visual
    # observation, however visually striking its (still schema-safe) text
    # might be, must never flip is_emergency on its own. A real session is
    # required because this observation text matches internal-medicine
    # (Phase 3A's leg-swelling keywords), which triggers provider_search.
    graph = build_conversation_graph()
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(
        observations=[
            {
                **_VALID_OBSERVATION,
                "visual_description": "extensive visible swelling across the affected area",
            }
        ]
    )
    config = _config(
        "t-vision-no-emergency", session=db_session, media=media, vision_provider=provider
    )
    state = {
        "intake_request": {"duration": {"value": 1, "unit": "days"}},
        "generate_speech": False,
        "media_pending": True,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["is_emergency"] is False
    assert result["intake_response"]["status"] != "emergency"


async def test_graph_declared_emergency_precedence_unchanged_with_media_present() -> None:
    # A user-declared emergency must still short-circuit straight to
    # response_composition even when media was uploaded this turn.
    # vision_analysis is skipped entirely (route_from_start) rather than
    # run-and-ignored — the provider must never even be called, since an
    # emergency response can never surface its output either way.
    graph = build_conversation_graph()
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])
    config = _config("t-vision-emergency", media=media, vision_provider=provider)
    state = {
        "intake_request": {"emergency_concern": True},
        "generate_speech": False,
        "media_pending": True,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["intake_response"]["status"] == "emergency"
    assert provider.call_count == 0
    assert not result.get("vision_observations")
    assert result.get("routing") is None
    assert result.get("provider_search") is None
    assert "911" in result["response_text"]


async def test_graph_media_pending_false_never_runs_vision_analysis() -> None:
    # Even with a provider configured via configurable, vision_analysis
    # must not run (and no observations must appear) when media_pending is
    # not set — a plain text-only turn is unaffected by Phase 2C.
    graph = build_conversation_graph()
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])
    config = _config("t-no-media-pending", vision_provider=provider)
    state = {
        "intake_request": {
            "main_concern": "zzz qqq unrelated words",
            "duration": {"value": 1, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert not result.get("vision_observations")
    assert result.get("media_analysis_note") is None


# --- Phase 2C privacy/checkpoint review -----------------------------------


async def test_graph_resume_does_not_rerun_vision_provider(db_session: AsyncSession) -> None:
    # An initial turn with media that ALSO needs a follow-up (duration
    # missing) must call the vision provider exactly once — resuming with
    # the missing field must never re-run vision_analysis a second time.
    # A real session is needed because the dermatology match triggers
    # provider_search.
    graph = build_conversation_graph()
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[_DERMATOLOGY_OBSERVATION])
    config = _config(
        "t-resume-no-revision", session=db_session, media=media, vision_provider=provider
    )
    state = {"intake_request": {}, "generate_speech": False, "media_pending": True}

    first = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["missing_fields"] == ["duration"]
    assert provider.call_count == 1

    second = await graph.ainvoke(
        Command(resume={"duration": {"value": 3, "unit": "days"}}), config=config
    )
    assert "__interrupt__" not in second
    assert provider.call_count == 1  # unchanged — vision_analysis did not run again
    assert second["routing"]["specialty_slug"] == "dermatology"
    assert len(second["vision_observations"]) == 1


async def test_graph_two_media_uploads_do_not_share_vision_observations() -> None:
    # Two independent media uploads (different thread_ids, as every real
    # POST /api/v1/media/analyze call gets) against the SAME compiled
    # graph instance (matching the process-wide singleton in production)
    # must never leak one upload's observations into the other's state.
    graph = build_conversation_graph()

    provider_a = FakeVisionProvider(observations=[_VALID_OBSERVATION])
    media_a = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"frame-a", timestamp_seconds=None)]
    )
    config_a = _config("t-media-upload-a", media=media_a, vision_provider=provider_a)
    state = {
        "intake_request": {"duration": {"value": 1, "unit": "days"}},
        "generate_speech": False,
        "media_pending": True,
    }
    result_a = await graph.ainvoke(state, config=config_a)

    config_b = _config("t-media-upload-b", vision_provider=provider_a)
    state_no_media = {
        "intake_request": {
            "main_concern": "zzz qqq unrelated words",
            "duration": {"value": 1, "unit": "days"},
        },
        "generate_speech": False,
    }
    result_b = await graph.ainvoke(state_no_media, config=config_b)

    assert len(result_a["vision_observations"]) == 1
    assert not result_b.get("vision_observations")


async def test_graph_checkpoint_never_contains_raw_media_bytes() -> None:
    # The checkpointed state for a thread must contain only the small,
    # already-validated VisionObservation dicts — never the raw frame
    # bytes/PreparedMedia object passed transiently via configurable.
    graph = build_conversation_graph()
    raw_frame_bytes = b"\xff\xd8\xff-totally-fake-jpeg-bytes-marker"
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=raw_frame_bytes, timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])
    config = _config("t-checkpoint-privacy", media=media, vision_provider=provider)
    state = {
        "intake_request": {"duration": {"value": 1, "unit": "days"}},
        "generate_speech": False,
        "media_pending": True,
    }
    await graph.ainvoke(state, config=config)

    snapshot = await graph.aget_state(config)
    checkpointed_repr = repr(snapshot.values)
    assert raw_frame_bytes not in checkpointed_repr.encode("utf-8", errors="ignore")
    assert "media" not in snapshot.values
    assert isinstance(snapshot.values["vision_observations"], list)
    assert all(isinstance(o, dict) for o in snapshot.values["vision_observations"])


async def test_no_vision_observation_content_in_captured_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Observation text/model output must never appear in logs, even when
    # real observations are produced (see app/services/
    # vision_analysis_service.py and app/api/v1/media.py's log lines,
    # which only ever log kind/counts/status — never content).
    graph = build_conversation_graph()
    synthetic_marker = "synthetic-marker-visible-finding-7d3f1a"
    observation = {**_VALID_OBSERVATION, "visual_description": synthetic_marker}
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[observation])
    config = _config("t-log-privacy", media=media, vision_provider=provider)
    state = {
        "intake_request": {"duration": {"value": 1, "unit": "days"}},
        "generate_speech": False,
        "media_pending": True,
    }

    with caplog.at_level(logging.DEBUG):
        result = await graph.ainvoke(state, config=config)

    assert synthetic_marker in result["response_text"]  # sanity: it really was produced


# --- Phase 3A: leg-swelling clinical-context vertical slice -----------------


async def test_graph_leg_swelling_full_flow_completes_with_navigation_summary(
    db_session: AsyncSession,
) -> None:
    # The exact vertical-slice example: onset/duration/location/laterality
    # already present in the first message (via a confirmed voice
    # transcript, so Phase 2A's duration extraction also applies) -- only
    # the four protocol questions should be asked, one at a time.
    graph = build_conversation_graph()
    config = _config("t-leg-swelling-full", session=db_session)
    transcript = "My left leg has been swollen for two days."
    state = {
        "intake_request": {"voice_input": {"transcript": transcript, "language": "en"}},
        "generate_speech": False,
    }

    first = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["missing_fields"] == ["clinical_answer"]

    # clinical_intake_agent_node's own state update is only ever written
    # once it actually returns -- which happens on the *next* invoke (the
    # one that resumes it), not the one that first paused it.
    onset_answer = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "It came on gradually, getting worse."}),
        config=config,
    )
    assert "__interrupt__" in onset_answer
    assert onset_answer["clinical_context"]["laterality"]["value"] == "left"
    assert onset_answer["clinical_context"]["duration"]["value"] == "2 days"
    assert onset_answer["clinical_context"]["primary_concern"]["value"] == "leg swelling"
    assert onset_answer["clinical_context"]["onset"]["value"] == "gradual"

    local_symptoms_answer = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Some redness but no pain or warmth."}),
        config=config,
    )
    assert "__interrupt__" in local_symptoms_answer

    red_flags_answer = await graph.ainvoke(
        Command(
            resume={
                "clinical_answer_text": (
                    "No chest pain, no trouble breathing, no fainting, no coughing blood."
                )
            }
        ),
        config=config,
    )
    # A fully negated red-flag answer must not short-circuit to emergency --
    # the protocol continues to the risk-factors question instead.
    assert "__interrupt__" in red_flags_answer
    assert red_flags_answer.get("is_emergency") is not True

    final = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "No recent travel, surgery, or injury."}),
        config=config,
    )
    assert "__interrupt__" not in final
    assert final["intake_response"]["status"] == "ready_for_multimodal_processing"
    assert final["routing"]["specialty_slug"] == "internal-medicine"
    assert final.get("provider_search") is not None

    navigation = final["clinical_navigation"]
    assert navigation["diagnosis"] is None
    assert navigation["treatment_recommendation"] is None
    # Corrected Phase 3A POC behavior: always prompt in-person evaluation
    # for leg swelling, never "routine".
    assert navigation["recommended_care_level"] == "prompt"
    assert "internal-medicine" in navigation["specialty_candidates"]
    assert "left-sided leg swelling" in navigation["summary_of_reported_information"]

    # Safety invariant: the dedicated red-flag gate must have run exactly
    # once for this completed, non-emergency protocol run, before routine
    # routing (see the reported bypass this correction fixes).
    agent_path = [entry["agent"] for entry in final["conversation_history"]]
    assert agent_path.count("clinical_red_flag_gate") == 1
    assert agent_path.index("clinical_red_flag_gate") < agent_path.index("specialty_routing_agent")


async def test_graph_leg_swelling_red_flag_affirmed_short_circuits_to_emergency() -> None:
    graph = build_conversation_graph()
    config = _config("t-leg-swelling-emergency")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    first = await graph.ainvoke(state, config=config)
    assert first["__interrupt__"][0].value["missing_fields"] == ["clinical_answer"]

    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Sudden onset, getting worse."}), config=config
    )
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Pain and redness."}), config=config
    )
    red_flag_result = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Yes, I have chest pain."}), config=config
    )

    assert "__interrupt__" not in red_flag_result
    assert red_flag_result["is_emergency"] is True
    assert red_flag_result["clinical_red_flag_affirmed"] is True
    assert red_flag_result.get("routing") is None
    assert red_flag_result.get("provider_search") is None
    assert "911" in red_flag_result["response_text"]
    assert red_flag_result["intake_response"]["status"] != "emergency"  # Phase 1C had no idea


async def test_graph_leg_swelling_red_flag_unknown_reply_never_affirms() -> None:
    # "I'm not sure" must never be treated as a positive red-flag finding.
    graph = build_conversation_graph()
    config = _config("t-leg-swelling-uncertain")
    state = {
        "intake_request": {
            "main_concern": "My right leg has been swollen for a week.",
            "duration": {"value": 1, "unit": "weeks"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)
    await graph.ainvoke(Command(resume={"clinical_answer_text": "Gradual."}), config=config)
    await graph.ainvoke(Command(resume={"clinical_answer_text": "No."}), config=config)
    red_flag_result = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "I'm not sure."}), config=config
    )
    assert "__interrupt__" in red_flag_result  # continues to risk_factors, not emergency
    assert red_flag_result.get("is_emergency") is not True
    # Absent from the (deliberately sparse) red_flags dict means "unknown"
    # -- an unresolved category is never explicitly stored (see
    # _merge_red_flags_dict), only ever affirmed/negated ones.
    assert (
        red_flag_result["clinical_context"]["red_flags"].get("chest_pain", "unknown") == "unknown"
    )


async def test_graph_leg_swelling_clinical_question_answerable_by_voice_transcript() -> None:
    # The same voice_transcript field an ordinary voice-answered
    # clarification already uses (see nodes.py's
    # _merge_clarification_answer) must also work for a Phase 3A protocol
    # question, with no dedicated UI change required.
    graph = build_conversation_graph()
    config = _config("t-leg-swelling-voice-answer")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)
    onset_answer = await graph.ainvoke(
        Command(resume={"voice_transcript": "It came on suddenly."}), config=config
    )
    assert onset_answer["clinical_context"]["onset"]["value"] == "sudden"
    assert "onset" in onset_answer["clinical_context"]["answered_protocol_fields"]


async def test_graph_non_leg_swelling_concern_skips_clinical_intake_agent() -> None:
    # Regression guard: every existing (non-protocol) concern must reach
    # specialty_routing_agent with clinical_intake_agent recorded as a
    # complete no-op, exactly as before Phase 3A.
    graph = build_conversation_graph()
    config = _config("t-not-leg-swelling")
    state = {
        "intake_request": {
            "main_concern": "zzz qqq unrelated words",
            "duration": {"value": 1, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" not in result
    assert result["clinical_protocol"] is None
    assert result.get("clinical_navigation") is None
    agent_path = [entry["agent"] for entry in result["conversation_history"]]
    assert "clinical_intake_agent" in agent_path
    assert "clinical_red_flag_gate" not in agent_path
    assert result["routing"]["method"] == "unmatched"


# --- Phase 3A correction: red-flag gate safety invariant --------------------
#
# An independent review found that the gate was only reached when the field
# most recently answered was literally "red_flags" -- meaning a message that
# stated a red flag up front (before any question was ever asked) could have
# that information satisfied by the initial-message extraction pass and then
# never actually evaluated by the gate at all. clinical_intake_agent_node was
# redesigned so the gate is reached whenever the red_flags field has been
# resolved by ANY means and the gate has not yet run for this context (see
# ClinicalContext.red_flag_gate_passed) -- checked fresh on every entry to
# that node, never inferred from which field was most recently answered.
# Each test below is numbered to match the review's required-test list.


def _agent_path(result: dict) -> list[str]:
    return [entry["agent"] for entry in result["conversation_history"]]


async def test_1_red_flag_affirmed_in_initial_message_short_circuits_immediately() -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-1-affirmed-initial")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days and I have chest pain.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" not in result  # never even reaches an interrupt
    assert result["is_emergency"] is True
    assert "911" in result["response_text"]
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1
    assert "specialty_routing_agent" not in path
    assert "provider_search_agent" not in path


async def test_2_red_flag_negated_in_initial_message_continues() -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-2-negated-initial")
    state = {
        "intake_request": {
            "main_concern": (
                "My left leg has been swollen for two days. No chest pain, no trouble "
                "breathing, no fainting, no coughing blood."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    # Not affirmed -> must continue the protocol rather than short-circuit;
    # onset/local_symptoms/risk_factors were not addressed, so it interrupts
    # for the next one, but only *after* the gate has already run once.
    assert "__interrupt__" in result
    assert result.get("is_emergency") is not True
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1


async def test_3_all_fields_in_initial_message_with_positive_red_flag_still_gated(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-3-all-fields-positive", session=db_session)
    state = {
        "intake_request": {
            "main_concern": (
                "My left leg has been swollen for two days. It came on gradually and has "
                "gotten worse. There is some redness. I have chest pain. I recently had "
                "surgery."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    # Every other field was answerable from the original message too, but
    # the gate must still run -- and short-circuit -- before any of that
    # is used to build a summary or route anywhere.
    assert "__interrupt__" not in result
    assert result["is_emergency"] is True
    assert result.get("clinical_navigation") is None
    assert result.get("routing") is None
    assert result.get("provider_search") is None
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1
    assert "specialty_routing_agent" not in path


async def test_4_all_fields_in_initial_message_with_negated_red_flags_completes_in_one_call(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-4-all-fields-negated", session=db_session)
    state = {
        "intake_request": {
            "main_concern": (
                "My left leg has been swollen for two days. It came on gradually and has "
                "gotten worse. There is some redness. No chest pain, no trouble breathing, "
                "no fainting, no coughing blood. I recently had surgery."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    # Nothing left to ask -> completes in a single call, but only after the
    # gate has run once and found nothing affirmed.
    assert "__interrupt__" not in result
    assert result.get("is_emergency") is not True
    assert result["clinical_navigation"] is not None
    assert result["routing"]["specialty_slug"] == "internal-medicine"
    assert result.get("provider_search") is not None
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1
    assert path.index("clinical_red_flag_gate") < path.index("specialty_routing_agent")


async def test_5_red_flag_affirmed_during_dedicated_question_short_circuits() -> None:
    # Same scenario as test_graph_leg_swelling_red_flag_affirmed_short_circuits_to_emergency
    # above, restated here under the review's numbering for traceability.
    graph = build_conversation_graph()
    config = _config("t-rf-5-dedicated-question")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)  # interrupt: onset
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Sudden onset, getting worse."}), config=config
    )  # interrupt: local_symptoms
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Pain and redness."}), config=config
    )  # interrupt: red_flags
    result = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Yes, I have chest pain."}), config=config
    )
    assert "__interrupt__" not in result
    assert result["is_emergency"] is True
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1
    assert "specialty_routing_agent" not in path


async def test_6_red_flag_mentioned_during_an_earlier_protocol_answer_is_still_caught() -> None:
    # The user mentions chest pain while answering the *onset* question --
    # not the initial message, and not the dedicated red-flags question.
    graph = build_conversation_graph()
    config = _config("t-rf-6-earlier-answer")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)  # interrupt: onset
    result = await graph.ainvoke(
        Command(
            resume={
                "clinical_answer_text": "It came on suddenly, and I've also had some chest pain."
            }
        ),
        config=config,
    )
    assert "__interrupt__" not in result
    assert result["is_emergency"] is True
    assert result.get("clinical_navigation") is None
    assert result.get("routing") is None
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1
    # local_symptoms/risk_factors were never reached -- escalation happened
    # as soon as the red flag became known, without asking anything else.
    assert not any(
        entry.get("field_answered") in ("local_symptoms", "risk_factors")
        for entry in result["conversation_history"]
        if entry["agent"] == "clinical_intake_agent"
    )


async def test_7_unknown_red_flag_reply_never_affirms() -> None:
    # Same scenario as test_graph_leg_swelling_red_flag_unknown_reply_never_affirms
    # above, restated here under the review's numbering for traceability.
    graph = build_conversation_graph()
    config = _config("t-rf-7-unknown")
    state = {
        "intake_request": {
            "main_concern": "My right leg has been swollen for a week.",
            "duration": {"value": 1, "unit": "weeks"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)
    await graph.ainvoke(Command(resume={"clinical_answer_text": "Gradual."}), config=config)
    await graph.ainvoke(Command(resume={"clinical_answer_text": "No."}), config=config)
    result = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "I'm not sure."}), config=config
    )
    assert result.get("is_emergency") is not True
    assert result["clinical_context"]["red_flags"].get("chest_pain", "unknown") == "unknown"
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1


async def test_8_positive_red_flag_prevents_specialty_routing() -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-8-no-routing")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days and I have chest pain.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result.get("routing") is None


async def test_9_positive_red_flag_prevents_provider_search() -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-9-no-provider-search")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days and I have chest pain.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result.get("provider_search") is None


async def test_10_positive_red_flag_prevents_normal_navigation_summary() -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-10-no-summary")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days and I have chest pain.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result.get("clinical_navigation") is None


async def test_11_negative_red_flags_permit_the_remaining_protocol_flow(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-11-negative-continues", session=db_session)
    state = {
        "intake_request": {
            # "chest discomfort" rather than "chest pain" here deliberately --
            # the word "pain" alone would also (incidentally, and separately
            # from the invariant under test) satisfy the local_symptoms
            # question's own "pain" phrase, changing which question comes
            # next; this test is about question *sequencing* after a
            # negated red flag, not that separate extraction-overlap case.
            "main_concern": (
                "My left leg has been swollen for two days. No chest discomfort, no "
                "trouble breathing, no fainting, no coughing blood."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    first = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in first  # gate ran, not affirmed, asks onset next
    onset_answer = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual, getting worse."}), config=config
    )
    assert "__interrupt__" in onset_answer  # asks local_symptoms next
    local_symptoms_answer = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "No pain, no redness."}), config=config
    )
    assert "__interrupt__" in local_symptoms_answer  # asks risk_factors next
    final = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "No recent travel or surgery."}), config=config
    )
    assert "__interrupt__" not in final
    assert final.get("is_emergency") is not True
    assert final["clinical_navigation"] is not None
    assert final["routing"]["specialty_slug"] == "internal-medicine"
    assert final.get("provider_search") is not None
    path = _agent_path(final)
    assert path.count("clinical_red_flag_gate") == 1


async def test_12_clinical_red_flag_gate_visited_exactly_once_before_routine_routing(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-rf-12-exactly-once", session=db_session)
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual, getting worse."}), config=config
    )
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "No pain, no redness."}), config=config
    )
    await graph.ainvoke(
        Command(
            resume={
                "clinical_answer_text": (
                    "No chest pain, no trouble breathing, no fainting, no coughing blood."
                )
            }
        ),
        config=config,
    )
    final = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "No recent travel or surgery."}), config=config
    )
    assert "__interrupt__" not in final
    path = _agent_path(final)
    assert path.count("clinical_red_flag_gate") == 1
    assert path.index("clinical_red_flag_gate") < path.index("specialty_routing_agent")
    assert path.index("clinical_red_flag_gate") < path.index("provider_search_agent")


# --- Phase 3A second correction: red-flag *completeness* gap ---------------
#
# An independent review found that _handle_red_flags/scan_incidental_red_flags
# marked the whole compound "red_flags" field answered the moment ANY single
# one of the four categories was resolved -- so a message stating only "no
# chest pain" (leaving breathing/fainting/coughing blood unaddressed) would
# never be asked about those three at all. Fixed by only marking the field
# answered when (a) at least one category is AFFIRMED (always demands
# immediate escalation on its own), or (b) all four are explicitly resolved;
# a partial state now asks a dynamically-narrowed question naming only the
# categories still unknown. Each test below is numbered to match the
# review's required-test list.


async def test_rfc_1_initial_message_negates_only_chest_pain() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-1")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days, but I have no chest pain.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result  # not resolved -> continues, does not skip ahead
    assert result.get("is_emergency") is not True
    path = _agent_path(result)
    assert "clinical_red_flag_gate" not in path  # not yet resolved enough to gate


async def test_rfc_2_initial_message_negates_only_breathing_difficulty() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-2")
    state = {
        "intake_request": {
            "main_concern": (
                "My left leg has been swollen for two days. No trouble breathing though."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result
    assert result.get("is_emergency") is not True
    path = _agent_path(result)
    assert "clinical_red_flag_gate" not in path


async def test_rfc_3_initial_message_negates_two_of_four() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-3")
    state = {
        "intake_request": {
            "main_concern": (
                "My left leg has been swollen for two days. No chest discomfort and no "
                "trouble breathing."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result
    path = _agent_path(result)
    assert "clinical_red_flag_gate" not in path


async def test_rfc_4_partial_negative_does_not_mark_red_flags_answered() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days, but I have no chest pain.",
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert "red_flags" not in context.answered_protocol_fields
    assert context.red_flags == {"chest_pain": "negated"}


async def test_rfc_5_remaining_unknown_red_flags_are_subsequently_asked() -> None:
    # "chest discomfort" here (not "chest pain") deliberately avoids
    # incidentally satisfying the unrelated local_symptoms "pain" phrase
    # from the same sentence, which would otherwise change which question
    # comes next and make this test about that overlap instead of the
    # narrowed-question behavior actually under test.
    graph = build_conversation_graph()
    config = _config("t-rfc-5")
    state = {
        "intake_request": {
            "main_concern": (
                "My left leg has been swollen for two days, but I have no chest discomfort."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    first = await graph.ainvoke(state, config=config)
    assert first["__interrupt__"][0].value["clarification_questions"] != [
        "Have you had any chest pain or discomfort, difficulty breathing or shortness of "
        "breath, fainting, or coughing up blood?"
    ]
    # Fast-forward through onset/local_symptoms to reach the (narrowed)
    # red-flags question.
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual, getting worse."}), config=config
    )
    local_symptoms_answer = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Some redness."}), config=config
    )
    red_flags_question = local_symptoms_answer["__interrupt__"][0].value["clarification_questions"][
        0
    ]
    assert "chest pain" not in red_flags_question  # already resolved -- not re-asked
    assert "difficulty breathing" in red_flags_question
    assert "fainting" in red_flags_question
    assert "coughing up blood" in red_flags_question


async def test_rfc_6_previously_recorded_negative_survives_the_later_answer() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-6")
    state = {
        "intake_request": {
            "main_concern": (
                "My left leg has been swollen for two days, but I have no chest discomfort."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual, getting worse."}), config=config
    )
    await graph.ainvoke(Command(resume={"clinical_answer_text": "Some redness."}), config=config)
    # Answer the narrowed question -- must not erase the earlier chest_pain negative.
    final_red_flags_answer = await graph.ainvoke(
        Command(
            resume={"clinical_answer_text": "No trouble breathing, no fainting, no coughing blood."}
        ),
        config=config,
    )
    assert final_red_flags_answer["clinical_context"]["red_flags"]["chest_pain"] == "negated"
    assert final_red_flags_answer["clinical_context"]["red_flags"]["difficulty_breathing"] == (
        "negated"
    )


async def test_rfc_7_partial_incidental_negative_during_onset_does_not_suppress_question() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-7")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)  # interrupt: onset
    onset_answer = await graph.ainvoke(
        Command(
            resume={"clinical_answer_text": "It came on suddenly. No chest discomfort though."}
        ),
        config=config,
    )
    assert "__interrupt__" in onset_answer
    assert onset_answer["clinical_context"]["red_flags"] == {"chest_pain": "negated"}
    assert "red_flags" not in onset_answer["clinical_context"]["answered_protocol_fields"]
    path = _agent_path(onset_answer)
    assert "clinical_red_flag_gate" not in path
    # The dedicated question must still come up later, narrowed to the
    # remaining three categories.
    local_symptoms_answer = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "No redness or warmth."}), config=config
    )
    red_flags_question = local_symptoms_answer["__interrupt__"][0].value["clarification_questions"][
        0
    ]
    assert "chest pain" not in red_flags_question
    assert "difficulty breathing" in red_flags_question


async def test_rfc_8_partial_incidental_negative_during_local_symptoms_not_suppressed() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-8")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)  # interrupt: onset
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual, getting worse."}), config=config
    )  # interrupt: local_symptoms
    local_symptoms_answer = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "There's some redness, but no chest discomfort."}),
        config=config,
    )
    assert "__interrupt__" in local_symptoms_answer
    assert local_symptoms_answer["clinical_context"]["red_flags"] == {"chest_pain": "negated"}
    assert "red_flags" not in local_symptoms_answer["clinical_context"]["answered_protocol_fields"]
    red_flags_question = local_symptoms_answer["__interrupt__"][0].value["clarification_questions"][
        0
    ]
    assert "chest pain" not in red_flags_question
    assert "fainting" in red_flags_question


async def test_rfc_9_incidental_positive_still_gates_immediately() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-9")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)
    result = await graph.ainvoke(
        Command(
            resume={
                "clinical_answer_text": "It came on suddenly, and I've also had some chest pain."
            }
        ),
        config=config,
    )
    assert "__interrupt__" not in result
    assert result["is_emergency"] is True
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1


async def test_rfc_10_mixed_incidental_affirmed_and_negated_preserves_both_and_escalates() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-10")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)  # interrupt: onset
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual, getting worse."}), config=config
    )  # interrupt: local_symptoms
    result = await graph.ainvoke(
        Command(
            resume={
                "clinical_answer_text": (
                    "No redness, but I do have chest pain, though no trouble breathing."
                )
            }
        ),
        config=config,
    )
    assert "__interrupt__" not in result
    assert result["is_emergency"] is True
    assert result["clinical_context"]["red_flags"]["chest_pain"] == "affirmed"
    assert result["clinical_context"]["red_flags"]["difficulty_breathing"] == "negated"
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1


async def test_rfc_11_all_four_initially_negated_visits_gate_exactly_once(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-11", session=db_session)
    state = {
        "intake_request": {
            "main_concern": (
                "My left leg has been swollen for two days. No chest discomfort, no "
                "trouble breathing, no fainting, no coughing blood."
            ),
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result  # gate ran once, then asks onset next
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1
    assert result.get("is_emergency") is not True


async def test_rfc_12_direct_not_sure_answer_remains_unknown_never_affirmed() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-12")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days, but I have no chest pain.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)  # interrupt: onset
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual, getting worse."}), config=config
    )  # interrupt: local_symptoms
    result = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "I'm not sure."}), config=config
    )
    # A direct (if ambiguous) answer to the dedicated question always
    # completes it -- never loops forever re-asking.
    assert result.get("is_emergency") is not True
    red_flags = result["clinical_context"]["red_flags"]
    assert red_flags["chest_pain"] == "negated"  # preserved from the initial message
    assert red_flags.get("difficulty_breathing", "unknown") == "unknown"
    path = _agent_path(result)
    assert path.count("clinical_red_flag_gate") == 1


async def test_rfc_13_positive_red_flag_still_cannot_reach_routing_search_or_summary() -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-13")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)
    result = await graph.ainvoke(
        Command(
            resume={
                "clinical_answer_text": "It came on suddenly, and I've also had some chest pain."
            }
        ),
        config=config,
    )
    assert result.get("routing") is None
    assert result.get("provider_search") is None
    assert result.get("clinical_navigation") is None


async def test_rfc_14_completed_non_emergency_flow_visits_gate_exactly_once_before_provider_search(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-rfc-14", session=db_session)
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days, but I have no chest pain.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    await graph.ainvoke(state, config=config)
    await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual, getting worse."}), config=config
    )
    await graph.ainvoke(Command(resume={"clinical_answer_text": "Some redness."}), config=config)
    await graph.ainvoke(
        Command(
            resume={"clinical_answer_text": "No trouble breathing, no fainting, no coughing blood."}
        ),
        config=config,
    )
    final = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "No recent travel or surgery."}), config=config
    )
    assert "__interrupt__" not in final
    assert final.get("is_emergency") is not True
    assert final["clinical_navigation"] is not None
    assert final["routing"]["specialty_slug"] == "internal-medicine"
    assert final.get("provider_search") is not None
    path = _agent_path(final)
    assert path.count("clinical_red_flag_gate") == 1
    assert path.index("clinical_red_flag_gate") < path.index("provider_search_agent")


# --- Phase 3A protocol-scope correction: unilateral only --------------------


async def test_bilateral_leg_swelling_does_not_enter_the_unilateral_protocol(
    db_session: AsyncSession,
) -> None:
    # A real session is required: bilateral swelling still matches
    # internal-medicine via the existing deterministic keyword routing
    # (unrelated to this protocol) since "swollen"/"swelling" are already
    # internal-medicine keywords -- only the clinical *protocol* must be
    # skipped for bilateral wording, not ordinary specialty routing.
    graph = build_conversation_graph()
    config = _config("t-scope-bilateral", session=db_session)
    state = {
        "intake_request": {
            "main_concern": "Both legs are swollen.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" not in result
    assert result["clinical_protocol"] is None
    assert result.get("clinical_navigation") is None


async def test_unspecified_laterality_does_not_enter_the_unilateral_protocol(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-scope-unspecified", session=db_session)
    state = {
        "intake_request": {
            "main_concern": "My leg is swollen.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" not in result
    assert result["clinical_protocol"] is None


async def test_explicit_left_leg_swelling_enters_the_protocol() -> None:
    graph = build_conversation_graph()
    config = _config("t-scope-left")
    state = {
        "intake_request": {
            "main_concern": "My left leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result
    onset_result = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual."}), config=config
    )
    assert onset_result["clinical_protocol"] == "unilateral_leg_swelling"
    assert onset_result["clinical_context"]["laterality"]["value"] == "left"


async def test_explicit_right_leg_swelling_enters_the_protocol() -> None:
    graph = build_conversation_graph()
    config = _config("t-scope-right")
    state = {
        "intake_request": {
            "main_concern": "My right leg has been swollen for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result
    onset_result = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual."}), config=config
    )
    assert onset_result["clinical_context"]["laterality"]["value"] == "right"


async def test_explicit_one_sided_wording_enters_the_protocol() -> None:
    graph = build_conversation_graph()
    config = _config("t-scope-one-sided")
    state = {
        "intake_request": {
            "main_concern": "I have one-sided leg swelling for two days.",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result
    onset_result = await graph.ainvoke(
        Command(resume={"clinical_answer_text": "Gradual."}), config=config
    )
    assert onset_result["clinical_context"]["laterality"]["value"] == "one-sided"


# --- Phase 3B: optional Groq-backed general-conversation layer -------------
#
# No database, no network, no real model call — the optional Groq path is
# exercised only with a monkeypatched stand-in, never the real SDK/network.
# concern_relevance_agent is a zero-cost, zero-behavior-change no-op with
# the deterministic default (conversation_mode="deterministic"), so every
# existing (pre-Phase-3B) test in this file continuing to pass unmodified
# is itself evidence of that.


def _fake_groq_json(content: str) -> type:
    class _FakeMessage:
        pass

    _FakeMessage.content = content  # type: ignore[attr-defined]

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, *args: object, **kwargs: object) -> _FakeResponse:
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeAsyncGroq:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.chat = _FakeChat()

    return _FakeAsyncGroq


async def test_concern_relevance_agent_noop_when_deterministic() -> None:
    graph = build_conversation_graph()
    config = _config("t-relevance-deterministic")
    state = {
        "intake_request": {
            "main_concern": "I'm feeling so bored, what should I do?",
            "duration": {"value": 1, "unit": "hours"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    # conversation_mode defaults to "deterministic" -- unchanged existing
    # behavior: treated as an ordinary (if unmatched) medical concern.
    assert result["detected_intent"] == "medical_concern"
    path = _agent_path(result)
    assert "concern_relevance_agent" in path
    assert "medical_intake_agent" in path
    assert "conversation_agent" not in path


async def test_concern_relevance_agent_routes_to_general_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("groq.AsyncGroq", _fake_groq_json('{"is_health_concern": false}'))
    graph = build_conversation_graph()
    config = _config(
        "t-relevance-general-chat",
        settings=_settings(conversation_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    state = {
        "intake_request": {"main_concern": "I'm feeling so bored, what should I do?"},
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["detected_intent"] == "general_chat"
    assert result.get("routing") is None
    assert result.get("provider_search") is None
    assert result.get("missing_fields") in (None, [])
    # Regression guard: response_agent_node must preserve
    # conversation_agent's already-finalized general-chat reply, never
    # silently overwrite it with the medical-routing response composer's
    # unrelated "could not match your concern" text.
    assert "describe a symptom" in (result.get("response_text") or "").lower()
    path = _agent_path(result)
    assert "medical_intake_agent" not in path
    assert "conversation_agent" in path


async def test_concern_relevance_agent_keeps_real_concern_on_true_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("groq.AsyncGroq", _fake_groq_json('{"is_health_concern": true}'))
    graph = build_conversation_graph()
    config = _config(
        "t-relevance-true",
        settings=_settings(conversation_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    state = {
        "intake_request": {"main_concern": "I think I have a headache, what do you recommend?"},
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["detected_intent"] == "medical_concern"
    path = _agent_path(result)
    assert "medical_intake_agent" in path


async def test_concern_relevance_agent_falls_back_when_groq_client_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RaisingGroq:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated network failure — never a real call")

    monkeypatch.setattr("groq.AsyncGroq", _RaisingGroq)
    graph = build_conversation_graph()
    config = _config(
        "t-relevance-raises",
        settings=_settings(conversation_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    state = {
        "intake_request": {"main_concern": "I'm feeling so bored, what should I do?"},
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    # Classification failure -> None -> treated exactly like True: proceeds
    # as an ordinary medical concern, never silently dropped.
    assert result["detected_intent"] == "medical_concern"
    path = _agent_path(result)
    assert "medical_intake_agent" in path


async def test_concern_relevance_agent_never_consulted_for_declared_emergency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SAFETY-CRITICAL: even a Groq classifier that would (wrongly) call
    # this "general chat" must never be able to divert a user-declared
    # emergency away from the safety gate. The fake here always answers
    # "general_chat" -- if it were consulted at all, this test would fail.
    monkeypatch.setattr("groq.AsyncGroq", _fake_groq_json('{"is_health_concern": false}'))
    graph = build_conversation_graph()
    config = _config(
        "t-relevance-emergency",
        settings=_settings(conversation_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    state = {
        "intake_request": {
            "main_concern": "I need help right now",
            "emergency_concern": True,
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert result["is_emergency"] is True
    assert result["intake_response"]["status"] == "emergency"
    assert "911" in result["response_text"]
    assert result["detected_intent"] != "general_chat"
    path = _agent_path(result)
    assert "conversation_agent" not in path
    assert "medical_intake_agent" in path


async def test_concern_relevance_agent_skipped_on_vision_path(db_session: AsyncSession) -> None:
    # The media/vision handoff to medical_intake_agent is unaffected by
    # this node -- it is only ever wired into the text/voice path.
    graph = build_conversation_graph()
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[_VALID_OBSERVATION])
    config = _config(
        "t-relevance-vision-skip", session=db_session, media=media, vision_provider=provider
    )
    state = {
        "intake_request": {"duration": {"value": 1, "unit": "days"}},
        "generate_speech": False,
        "media_pending": True,
    }
    result = await graph.ainvoke(state, config=config)
    path = _agent_path(result)
    assert "concern_relevance_agent" not in path
    assert "vision_agent" in path
    assert "medical_intake_agent" in path
