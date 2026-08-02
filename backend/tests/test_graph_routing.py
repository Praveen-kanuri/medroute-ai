"""Phase 2D: graph-routing tests for the multi-agent conversation graph.

Proves the *structural* routing guarantees the multi-agent architecture
exists to provide (see app/graph/build.py, app/graph/nodes.py): a greeting
is routed by supervisor_router straight to conversation_agent and can
never reach medical intake, specialty routing, or provider search; a real
medical concern always passes through medical_intake_agent; response_agent
cannot change a routing/safety decision another agent already made; and a
duplicate turn submission (the server-side backing for "a Streamlit rerun
must not repeat an already-completed turn") never re-invokes the graph.

Node-level and full end-to-end behavioral tests already live in
test_conversation_graph.py — this file is scoped specifically to the
routing/orchestration guarantees this task's architecture correction adds.
"""

from collections.abc import AsyncGenerator

import pytest
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

import app.graph.nodes as nodes
from app.config.settings import Settings
from app.db.session import dispose_engine, get_sessionmaker
from app.graph.build import build_conversation_graph
from app.graph.nodes import response_agent_node
from app.providers.text_to_speech.fake import FakeTextToSpeechProvider
from app.providers.vision.fake import FakeVisionProvider
from app.schemas.multimodal_intake import (
    Duration,
    DurationUnit,
    MultimodalIntakeRequest,
    VoiceInput,
)
from app.services.conversation_service import run_conversation_turn
from app.services.media_validation_service import MediaFrame, PreparedMedia

_DERMATOLOGY_OBSERVATION = {
    "observation_type": "skin_appearance",
    "body_area": "forearm",
    "visual_description": "visible itchy rash on the forearm",
    "visible_attributes": ["redness"],
    "confidence": "low",
    "limitations": "a visual review only, not a diagnosis",
}


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
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


def _agent_path(result: dict) -> list[str]:
    return [entry["agent"] for entry in result["conversation_history"]]


# 1. "Hi" routes supervisor -> conversation_agent -> response_agent.
async def test_greeting_routes_supervisor_conversation_agent_response_agent() -> None:
    graph = build_conversation_graph()
    config = _config("t-route-hi")
    state = {"intake_request": {"main_concern": "Hi"}, "generate_speech": False}
    result = await graph.ainvoke(state, config=config)
    assert _agent_path(result) == ["supervisor_router", "conversation_agent", "response_agent"]
    assert result["detected_intent"] == "greeting"


# 2. A greeting never calls provider search.
async def test_greeting_never_reaches_specialty_routing_or_provider_search() -> None:
    graph = build_conversation_graph()
    config = _config("t-route-greeting-no-provider")
    state = {"intake_request": {"main_concern": "hello there"}, "generate_speech": False}
    result = await graph.ainvoke(state, config=config)
    agent_path = _agent_path(result)
    assert "specialty_routing_agent" not in agent_path
    assert "provider_search_agent" not in agent_path
    assert result.get("routing") is None
    assert result.get("provider_search") is None


# 3. A medical concern routes through medical intake.
async def test_medical_concern_routes_through_medical_intake_agent(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-route-medical-concern", session=db_session)
    state = {
        "intake_request": {
            "main_concern": "chest pain for two days",
            "duration": {"value": 2, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    agent_path = _agent_path(result)
    assert agent_path[0] == "supervisor_router"
    assert "medical_intake_agent" in agent_path
    assert "conversation_agent" not in agent_path
    assert result["detected_intent"] == "medical_concern"


# 4. Missing information creates a clarification loop.
async def test_missing_information_creates_a_clarification_loop() -> None:
    graph = build_conversation_graph()
    config = _config("t-route-clarify-loop")
    state = {
        "intake_request": {"main_concern": "zzz qqq unrelated words"},
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    assert "__interrupt__" in result
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["missing_fields"] == ["duration"]
    assert interrupt_value["clarification_questions"]


# 5. A clarification answer resumes the same thread.
async def test_clarification_answer_resumes_the_same_thread() -> None:
    graph = build_conversation_graph()
    config = _config("t-route-clarify-resume")
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
    assert second["detected_intent"] == "clarification_answer"

    # Same checkpointed thread -- the original concern text was preserved,
    # never re-submitted as a fresh/unrelated turn.
    snapshot = await graph.aget_state(config)
    assert snapshot.values["intake_request"]["main_concern"] == "zzz qqq unrelated words"


# 6. Complete intake reaches specialty routing and provider search.
async def test_complete_intake_reaches_specialty_routing_and_provider_search(
    db_session: AsyncSession,
) -> None:
    graph = build_conversation_graph()
    config = _config("t-route-complete-intake", session=db_session)
    state = {
        "intake_request": {
            "symptoms": ["annual checkup"],
            "location": {"city": "Springfield", "state": "CA"},
            "duration": {"value": 1, "unit": "days"},
        },
        "generate_speech": False,
    }
    result = await graph.ainvoke(state, config=config)
    agent_path = _agent_path(result)
    assert "specialty_routing_agent" in agent_path
    assert "provider_search_agent" in agent_path
    assert result["selected_specialty"] == "family-medicine"
    assert result["provider_results"] is not None


# 7. Vision observations hand off to medical intake.
async def test_vision_observations_hand_off_to_medical_intake(db_session: AsyncSession) -> None:
    graph = build_conversation_graph()
    media = PreparedMedia(
        kind="image", frames=[MediaFrame(data=b"fake-jpeg", timestamp_seconds=None)]
    )
    provider = FakeVisionProvider(observations=[_DERMATOLOGY_OBSERVATION])
    config = _config(
        "t-route-vision-handoff", session=db_session, media=media, vision_provider=provider
    )
    state = {
        "intake_request": {"duration": {"value": 1, "unit": "days"}},
        "generate_speech": False,
        "media_pending": True,
    }
    result = await graph.ainvoke(state, config=config)
    agent_path = _agent_path(result)
    assert agent_path[0] == "supervisor_router"
    assert agent_path[1] == "vision_agent"
    assert "medical_intake_agent" in agent_path
    assert "concern" not in result["missing_fields"]
    # The vision-derived observation actually influenced routing -- proof
    # the handoff carried real content, not just an empty pass-through.
    assert result["selected_specialty"] == "dermatology"


# 8. Response Agent cannot change selected specialty/provider data.
async def test_response_agent_cannot_change_specialty_or_provider_data() -> None:
    state = {
        "detected_intent": "medical_concern",
        "is_emergency": False,
        "missing_fields": [],
        "intake_request": {"main_concern": "chest pain"},
        "intake_response": {"clarification_questions": []},
        "routing": {
            "specialty_slug": "cardiology",
            "specialty_display_name": "Cardiology",
            "method": "keyword_match",
            "note": "x",
        },
        "provider_search": {"results": [], "disclaimer": "x"},
        "selected_specialty": "cardiology",
        "provider_results": {"results": [], "disclaimer": "x"},
    }
    config = _config("t-route-response-agent-bounds")
    result = await response_agent_node(state, config)
    # response_agent's return value is the only thing that can ever affect
    # state (LangGraph merges by key) -- it must never include any of the
    # routing/provider/safety keys, no matter what it read.
    forbidden_keys = {
        "routing",
        "provider_search",
        "selected_specialty",
        "provider_results",
        "is_emergency",
        "detected_intent",
        "missing_fields",
    }
    assert forbidden_keys.isdisjoint(result.keys())
    assert set(result.keys()) <= {"active_agent", "response_text", "conversation_history"}


# 9. Voice-originated responses automatically request TTS.
async def test_voice_originated_response_automatically_requests_tts() -> None:
    response = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(
            voice_input=VoiceInput(transcript="zzz qqq unrelated words", language="en"),
            duration=Duration(value=2, unit=DurationUnit.DAYS),
        ),
        clarification_answer=None,
        generate_speech=True,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
        tts_provider_override=FakeTextToSpeechProvider(),
    )
    assert response.intent == "medical_concern"
    assert response.audio is not None


# 10. Streamlit reruns do not repeat an already completed graph turn.
async def test_duplicate_turn_fingerprint_does_not_reinvoke_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}
    original_evaluate_intake = nodes.evaluate_intake

    def counting_evaluate_intake(*args: object, **kwargs: object) -> object:
        call_count["n"] += 1
        return original_evaluate_intake(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(nodes, "evaluate_intake", counting_evaluate_intake)

    settings = _settings()
    fingerprint = "synthetic-dedup-fingerprint-1"
    intake = MultimodalIntakeRequest(
        main_concern="zzz qqq unrelated words", duration=Duration(value=1, unit=DurationUnit.DAYS)
    )

    first = await run_conversation_turn(
        thread_id="t-route-dedup-thread",
        intake=intake,
        clarification_answer=None,
        generate_speech=False,
        settings=settings,
        session=None,  # type: ignore[arg-type]
        turn_fingerprint=fingerprint,
    )
    assert call_count["n"] == 1

    second = await run_conversation_turn(
        thread_id="t-route-dedup-thread",
        intake=intake,
        clarification_answer=None,
        generate_speech=False,
        settings=settings,
        session=None,  # type: ignore[arg-type]
        turn_fingerprint=fingerprint,
    )
    # Unchanged -- the graph (and therefore medical_intake_agent's own
    # evaluate_intake call) was never re-invoked for the duplicate submission.
    assert call_count["n"] == 1
    assert second.response_text == first.response_text
    assert second.thread_id == first.thread_id == "t-route-dedup-thread"
