"""Phase 3A correction: regression tests proving clinical state (clinical_
protocol/clinical_context/clinical_next_step/clinical_red_flag_affirmed/
clinical_navigation/is_emergency) cannot leak between separate conversation
threads, a later unrelated concern reusing the same thread_id, or a
greeting following a completed clinical flow.

Calls app.services.conversation_service.run_conversation_turn directly (the
same entry point the API layer uses) rather than raw graph.ainvoke, since
the state-isolation fix under test lives in that function's construction of
a brand-new turn's initial graph state.
"""

from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.db.session import dispose_engine, get_sessionmaker
from app.schemas.multimodal_intake import Duration, DurationUnit, MultimodalIntakeRequest
from app.services.conversation_service import run_conversation_turn

_LEG_SWELLING_CONCERN = (
    "My left leg has been swollen for two days. It came on gradually and has gotten "
    "worse. There is some redness. No chest pain, no trouble breathing, no fainting, "
    "no coughing blood. I recently had surgery."
)
_LEG_SWELLING_EMERGENCY_CONCERN = "My left leg has been swollen for two days and I have chest pain."


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    await dispose_engine()
    async with get_sessionmaker()() as session:
        yield session
    await dispose_engine()


async def test_separate_new_threads_never_share_clinical_context(
    db_session: AsyncSession,
) -> None:
    first = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(
            main_concern=_LEG_SWELLING_CONCERN, duration=Duration(value=2, unit=DurationUnit.DAYS)
        ),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert first.clinical_navigation is not None
    assert first.thread_id != ""

    second = await run_conversation_turn(
        thread_id=None,  # a genuinely separate, brand-new thread
        intake=MultimodalIntakeRequest(
            main_concern="annual checkup", duration=Duration(value=1, unit=DurationUnit.DAYS)
        ),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert second.thread_id != first.thread_id
    assert second.clinical_navigation is None
    assert second.status != "emergency"


async def test_reusing_a_thread_id_for_a_new_unrelated_concern_does_not_leak_clinical_state(
    db_session: AsyncSession,
) -> None:
    completed = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(
            main_concern=_LEG_SWELLING_CONCERN, duration=Duration(value=2, unit=DurationUnit.DAYS)
        ),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert completed.clinical_navigation is not None
    assert completed.status != "emergency"

    reused = await run_conversation_turn(
        thread_id=completed.thread_id,  # same thread_id, but a brand-new intake, not a resume
        intake=MultimodalIntakeRequest(
            main_concern="annual checkup", duration=Duration(value=1, unit=DurationUnit.DAYS)
        ),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert reused.thread_id == completed.thread_id
    assert reused.clinical_navigation is None
    assert reused.status != "emergency"
    assert reused.routing is None or reused.routing.specialty_slug != "internal-medicine"


async def test_reusing_a_thread_id_after_an_emergency_does_not_leak_is_emergency(
    db_session: AsyncSession,
) -> None:
    emergency = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(
            main_concern=_LEG_SWELLING_EMERGENCY_CONCERN,
            duration=Duration(value=2, unit=DurationUnit.DAYS),
        ),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert emergency.status == "emergency"

    reused = await run_conversation_turn(
        thread_id=emergency.thread_id,
        intake=MultimodalIntakeRequest(
            main_concern="annual checkup", duration=Duration(value=1, unit=DurationUnit.DAYS)
        ),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert reused.status != "emergency"
    assert reused.clinical_navigation is None


async def test_greeting_after_a_completed_clinical_flow_shows_no_stale_clinical_state(
    db_session: AsyncSession,
) -> None:
    completed = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(
            main_concern=_LEG_SWELLING_CONCERN, duration=Duration(value=2, unit=DurationUnit.DAYS)
        ),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert completed.clinical_navigation is not None

    greeting = await run_conversation_turn(
        thread_id=completed.thread_id,
        intake=MultimodalIntakeRequest(main_concern="Hi"),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert greeting.intent == "greeting"
    assert greeting.status == "ready_for_multimodal_processing"
    assert greeting.clinical_navigation is None
    assert greeting.routing is None
    assert greeting.provider_search is None


async def test_greeting_after_a_completed_emergency_shows_no_stale_emergency_status(
    db_session: AsyncSession,
) -> None:
    emergency = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(
            main_concern=_LEG_SWELLING_EMERGENCY_CONCERN,
            duration=Duration(value=2, unit=DurationUnit.DAYS),
        ),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert emergency.status == "emergency"

    greeting = await run_conversation_turn(
        thread_id=emergency.thread_id,
        intake=MultimodalIntakeRequest(main_concern="hello"),
        clarification_answer=None,
        generate_speech=False,
        settings=_settings(),
        session=db_session,
    )
    assert greeting.intent == "greeting"
    assert greeting.status == "ready_for_multimodal_processing"
