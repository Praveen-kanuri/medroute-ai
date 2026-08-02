"""Direct-call tests for app.services.conversation_service.run_conversation_turn,
focused on the greeting short-circuit (see
app.services.intent_classification_service). A greeting/thanks message
must never touch the LangGraph conversation graph, intake validation,
specialty routing, or provider search — yet must still honor
generate_speech exactly like every other turn.

Calls run_conversation_turn directly (not via HTTP) so a fake
TextToSpeechProvider can be injected. Since a greeting never reaches the
graph, it also never needs a database session — passing None in its place
is itself evidence the short-circuit works (a real turn would fail fast
against a None session the moment provider_search_node ran).
"""

from app.config.settings import Settings
from app.providers.text_to_speech.fake import FakeTextToSpeechProvider
from app.schemas.multimodal_intake import MultimodalIntakeRequest, VoiceInput
from app.services.conversation_service import run_conversation_turn


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


async def test_greeting_short_circuits_before_graph_with_no_database() -> None:
    response = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(main_concern="Hi"),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
    )
    assert response.intent == "greeting"
    assert response.status == "ready_for_multimodal_processing"
    assert response.routing is None
    assert response.provider_search is None
    assert response.missing_fields == []
    assert response.clarification_questions == []


async def test_voice_originated_greeting_requests_and_receives_speech() -> None:
    response = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(
            voice_input=VoiceInput(transcript="Good evening", language="en")
        ),
        clarification_answer=None,
        generate_speech=True,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
        tts_provider_override=FakeTextToSpeechProvider(),
    )
    assert response.intent == "greeting"
    assert response.audio is not None
    assert response.response_text is not None
    assert "evening" in response.response_text.lower()


async def test_non_voice_greeting_does_not_request_speech() -> None:
    response = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(main_concern="hello"),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
        tts_provider_override=FakeTextToSpeechProvider(),
    )
    assert response.audio is None


async def test_emergency_declared_overrides_greeting_looking_text() -> None:
    response = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(main_concern="hi", emergency_concern=True),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
    )
    assert response.status == "emergency"
    assert response.intent == "medical_concern"
    assert response.response_text is not None
    assert "911" in response.response_text


async def test_thread_id_is_preserved_for_a_greeting_reply() -> None:
    response = await run_conversation_turn(
        thread_id="caller-supplied-thread-id",
        intake=MultimodalIntakeRequest(main_concern="thanks"),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
    )
    assert response.thread_id == "caller-supplied-thread-id"
