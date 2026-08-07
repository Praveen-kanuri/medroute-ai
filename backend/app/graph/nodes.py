"""Phase 2B-2D: LangGraph node implementations for the MedRoute AI
multi-agent conversation graph.

Every node is a thin wrapper around an already-existing, already-tested
service function — this module contains no business logic of its own,
only orchestration glue and the JSON-dict <-> pydantic-model boundary
LangGraph's checkpointer needs. Dependencies that don't belong in
checkpointed state (Settings, the DB session, and — only in tests — a
fake TTS provider) are threaded through via each node's `config`
parameter (`config["configurable"]`), never a module-level global.

Each node has a bounded responsibility and a restricted set of tools it
calls, matching the graph's multi-agent shape (see app/graph/build.py):
supervisor_router (routes only, no domain logic), conversation_agent
(greeting/small-talk replies only — no provider search, no intake
validation), medical_intake_agent (intake normalization only),
specialty_routing_agent (specialty selection only), provider_search_agent
(deterministic provider lookup only — no medical reasoning),
response_agent (text finalization only — never changes routing/safety
decisions already made upstream). text_to_speech is deliberately not
named/treated as an agent: it is an external modality service, not a
reasoning step.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.graph.state import ConversationState
from app.providers.text_to_speech.base import TextToSpeechProvider
from app.providers.vision.base import VisionProvider
from app.schemas.clinical_context import (
    ClinicalContext,
    ClinicalFact,
    FactConfidence,
    FactSource,
    MentionStatus,
)
from app.schemas.conversation import ConversationIntent
from app.schemas.multimodal_intake import IntakeStatus, MultimodalIntakeRequest
from app.services.clinical_intake_service import (
    apply_field_answer,
    build_initial_context,
    build_navigation_summary,
    get_protocol,
    match_protocol,
    next_missing_field,
    question_for,
    red_flags_question_text,
    scan_incidental_red_flags,
)
from app.services.duration_extraction import extract_duration_from_transcript
from app.services.intent_classification_service import (
    classify_concern_relevance,
    classify_new_turn_intent,
    compose_general_chat_reply,
    greeting_reply_text,
)
from app.services.intent_classification_service import (
    concern_text as _concern_text,
)
from app.services.media_validation_service import PreparedMedia
from app.services.multimodal_intake_service import evaluate_intake
from app.services.provider_search_service import (
    DEFAULT_LIMIT,
    build_provider_search_response,
    search_providers_page,
)
from app.services.response_composition_service import compose_response
from app.services.specialty_routing_service import route_to_specialty
from app.services.text_to_speech_service import build_text_to_speech_provider, synthesize_speech
from app.services.vision_analysis_service import analyze_media_frames

MEDIA_UNAVAILABLE_NOTE = (
    "Image/video analysis is not available in this demo. Only text and voice-transcript "
    "content were used for routing."
)


def _configurable(config: RunnableConfig) -> dict[str, Any]:
    return config.get("configurable") or {}


def _concern_already_present(intake_request: dict[str, Any]) -> bool:
    return (
        bool(intake_request.get("main_concern"))
        or bool(intake_request.get("symptoms"))
        or bool((intake_request.get("voice_input") or {}).get("transcript"))
    )


def _merge_clarification_answer(intake_request: dict[str, Any], answer: object) -> dict[str, Any]:
    """Merge a typed *or spoken* clarification answer into the intake
    request dict.

    Only the two structured fields the clarification node ever asks about
    (duration, main_concern) are ever considered; anything else in
    `answer` is ignored, except `voice_transcript` (Phase 2E) — the
    speech-to-text transcript of the user's *spoken* answer, used only as
    a *source* to derive one of those two fields when the caller didn't
    already supply them directly:

    - `duration` is extracted from it with the exact same conservative,
      never-guessing logic a fresh voice-only intake already gets (see
      app.services.duration_extraction) — never a diagnosis, treatment,
      urgency, or emergency inference.
    - `main_concern` falls back to the raw transcript only when this
      intake genuinely has no concern yet (see _concern_already_present)
      — an already-confirmed concern from earlier in the same turn is
      never overwritten by a later duration-only spoken answer.

    Every other key already in intake_request — including a confirmed
    voice_input — is carried over unchanged."""
    merged = dict(intake_request)
    if not isinstance(answer, dict):
        return merged

    duration = answer.get("duration")
    main_concern = answer.get("main_concern")
    voice_transcript = answer.get("voice_transcript")

    if not duration and voice_transcript:
        extracted = extract_duration_from_transcript(voice_transcript)
        if extracted is not None:
            duration = extracted.model_dump(mode="json")

    if not main_concern and voice_transcript and not _concern_already_present(merged):
        main_concern = voice_transcript

    if duration:
        merged["duration"] = duration
    if main_concern:
        merged["main_concern"] = main_concern
    return merged


def supervisor_router_node(state: ConversationState) -> dict[str, Any]:
    """The graph's single entry point and only routing authority.

    Inspects the current normalized input, its modality, whether media is
    pending, and whether an emergency was declared, then makes one
    explicit structured decision (detected_intent + next_agent) — never
    the other way around (no downstream agent second-guesses this).
    Greeting/small-talk detection is deterministic (see
    app.services.intent_classification_service) and requires no model
    call, but the *routing decision itself* — and the fact that it is
    graph state, not a Streamlit if/else — is what makes this multi-agent
    orchestration: a greeting is routed to conversation_agent and can
    never reach medical_intake_agent, specialty_routing_agent, or
    provider_search_agent this turn.

    A resumed clarification turn never re-enters here (LangGraph resumes
    execution at the interrupted node itself — see clarification_node) —
    this node only ever runs once, for a brand-new turn.
    """
    intake_request = state.get("intake_request") or {}
    is_emergency = bool(intake_request.get("emergency_concern")) or bool(
        intake_request.get("emergency_signals")
    )
    concern = _concern_text(intake_request)
    current_input = state.get("current_input")
    if current_input is None:
        current_input = concern

    if is_emergency:
        # Safety precedence: an emergency-declared turn always proceeds
        # through medical_intake_agent -> safety_gate, regardless of
        # whether the accompanying text happens to look like a greeting
        # or media is pending -- never short-circuited to conversation_agent.
        detected_intent = ConversationIntent.MEDICAL_CONCERN.value
        next_agent = "medical_intake_agent"
    elif state.get("media_pending"):
        detected_intent = ConversationIntent.MEDICAL_CONCERN.value
        next_agent = "vision_agent"
    else:
        detected_intent = classify_new_turn_intent(
            concern=concern, has_media=False, is_emergency=False
        ).value
        next_agent = (
            "conversation_agent"
            if detected_intent == ConversationIntent.GREETING.value
            else "medical_intake_agent"
        )

    return {
        "active_agent": "supervisor_router",
        "next_agent": next_agent,
        "detected_intent": detected_intent,
        "current_input": current_input,
        "conversation_history": [
            {"agent": "supervisor_router", "intent": detected_intent, "next_agent": next_agent}
        ],
    }


def route_after_supervisor(state: ConversationState) -> str:
    return state.get("next_agent") or "medical_intake_agent"


async def concern_relevance_agent_node(
    state: ConversationState, config: RunnableConfig
) -> dict[str, Any]:
    """Phase 3B (optional, opt-in via Settings.conversation_mode="groq"):
    the only place in the graph that may reclassify an ordinary
    medical_concern turn as general chat, inserted between
    supervisor_router and medical_intake_agent (see app/graph/build.py) —
    never bypassed, but a complete zero-cost, zero-risk no-op whenever
    conversation_mode stays "deterministic" (the default) or Groq isn't
    configured, in which case this always falls through to
    medical_intake_agent exactly as before this node existed.

    SAFETY: never consulted for a user-declared emergency turn — checked
    directly here, independent of supervisor_router_node, so an LLM
    misclassification can never divert a declared emergency away from the
    safety gate. This is the only node in the graph allowed to route a
    medical_concern turn to conversation_agent instead of
    medical_intake_agent; it never diagnoses, and a failed/unconfigured
    check always defaults to treating the turn as a genuine concern
    (proceeding to medical_intake_agent), never the other way around."""
    intake_request = state.get("intake_request") or {}
    is_emergency = bool(intake_request.get("emergency_concern")) or bool(
        intake_request.get("emergency_signals")
    )
    settings: Settings | None = _configurable(config).get("settings")
    concern = _concern_text(intake_request)

    is_general_chat = False
    if not is_emergency and settings is not None and concern is not None:
        if settings.conversation_mode == "groq" and settings.groq_configured:
            is_health_concern = await classify_concern_relevance(concern, settings)
            is_general_chat = is_health_concern is False

    if is_general_chat:
        return {
            "next_agent": "conversation_agent",
            "detected_intent": ConversationIntent.GENERAL_CHAT.value,
            "active_agent": "concern_relevance_agent",
            "conversation_history": [{"agent": "concern_relevance_agent", "is_general_chat": True}],
        }
    return {
        "next_agent": "medical_intake_agent",
        "active_agent": "concern_relevance_agent",
        "conversation_history": [{"agent": "concern_relevance_agent", "is_general_chat": False}],
    }


def route_after_concern_relevance(state: ConversationState) -> str:
    return state.get("next_agent") or "medical_intake_agent"


async def conversation_agent_node(
    state: ConversationState, config: RunnableConfig
) -> dict[str, Any]:
    """Handles greeting AND general-chat turns. Restricted tools: for a
    greeting, a fixed, reviewed reply-text lookup (see
    app.services.intent_classification_service.greeting_reply_text) —
    unchanged, still zero-cost, zero-model-call. For general chat (Phase
    3B, only ever reachable via concern_relevance_agent_node — see that
    node's docstring for the safety guarantees this relies on), a
    deterministic-by-default, optionally Groq-backed redirect reply (see
    compose_general_chat_reply) — never provider search, never intake
    validation, never a diagnosis or medical advice. Its output is
    finalized (not re-derived) by response_agent."""
    if state.get("detected_intent") == ConversationIntent.GENERAL_CHAT.value:
        settings: Settings = _configurable(config)["settings"]
        reply = await compose_general_chat_reply(state.get("current_input"), settings)
        return {
            "active_agent": "conversation_agent",
            "response_text": reply,
            "conversation_history": [{"agent": "conversation_agent", "intent": "general_chat"}],
        }

    reply = greeting_reply_text(state.get("current_input"))
    return {
        "active_agent": "conversation_agent",
        "response_text": reply,
        "conversation_history": [{"agent": "conversation_agent", "intent": "greeting"}],
    }


def _vision_observation_text(observations: list[dict[str, Any]]) -> str | None:
    """Flattens already schema-validated VisionObservation dicts into a
    plain string for specialty-routing keyword matching — the same
    treatment already given to a confirmed voice transcript."""
    if not observations:
        return None
    parts: list[str] = []
    for observation in observations:
        description = observation.get("visual_description")
        if description:
            parts.append(description)
        parts.extend(observation.get("visible_attributes") or [])
    text = " ".join(parts).strip()
    return text or None


async def vision_agent_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
    """Phase 2C: analyzes already-validated media frames (see
    app/api/v1/media.py, which performs all upload validation before the
    graph ever runs) into controlled VisionObservation dicts via the
    configured vision provider. Restricted tools: the vision provider
    only — no authority to diagnose (the provider's own schema rejects
    diagnostic/treatment/urgency language; see app/schemas/vision.py).
    Only ever runs when supervisor_router routed here. Never fabricates
    an observation: any missing configuration or provider failure yields
    an empty list plus a safe status note instead (see
    app.services.vision_analysis_service.analyze_media_frames). Always
    hands off into medical_intake_agent next — vision observations only
    ever *feed* intake normalization, never bypass it."""
    configurable = _configurable(config)
    media: PreparedMedia | None = configurable.get("media")
    if media is None:
        return {
            "vision_observations": [],
            "media_analysis_note": None,
            "active_agent": "vision_agent",
            "conversation_history": [{"agent": "vision_agent", "observation_count": 0}],
        }

    settings: Settings = configurable["settings"]
    provider: VisionProvider | None = configurable.get("vision_provider")
    observations, note = await analyze_media_frames(
        media.frames, kind=media.kind, settings=settings, provider=provider
    )
    observation_dicts = [observation.model_dump(mode="json") for observation in observations]
    return {
        "vision_observations": observation_dicts,
        "media_analysis_note": note,
        "active_agent": "vision_agent",
        "conversation_history": [
            {"agent": "vision_agent", "observation_count": len(observation_dicts)}
        ],
    }


def medical_intake_agent_node(state: ConversationState) -> dict[str, Any]:
    """Phase 1C intake normalization, reused unchanged, plus one additive
    Phase 2C signal: controlled visual observations already produced by
    vision_agent (if it ran this turn) satisfy the "concern" requirement
    exactly like a confirmed voice transcript already does — see
    evaluate_intake's vision_concern_present parameter. Restricted tools:
    deterministic intake evaluation only — no specialty inference, no
    provider lookup."""
    request = MultimodalIntakeRequest.model_validate(state["intake_request"])
    vision_concern_present = bool(state.get("vision_observations"))
    response = evaluate_intake(request, vision_concern_present=vision_concern_present)
    media_note = MEDIA_UNAVAILABLE_NOTE if response.multimodal_status.vision_received else None
    return {
        "active_agent": "medical_intake_agent",
        "intake_response": response.model_dump(mode="json"),
        "missing_fields": response.missing_fields,
        "media_note": media_note,
        "normalized_intake": response.normalized_intake.model_dump(mode="json"),
        "clarification_status": "awaiting" if response.missing_fields else "none",
        "conversation_history": [
            {"agent": "medical_intake_agent", "missing_fields": response.missing_fields}
        ],
    }


def safety_gate_node(state: ConversationState) -> dict[str, Any]:
    """Emergency precedence check. Reads only the already-evaluated intake
    status (itself derived only from user-declared emergency_concern /
    emergency_signals — see evaluate_intake) — never inspects transcript
    or symptom text directly, and never infers an emergency from it. Part
    of the existing safety path, not a reasoning agent."""
    intake_response = state.get("intake_response") or {}
    is_emergency = intake_response.get("status") == IntakeStatus.EMERGENCY.value
    return {
        "active_agent": "safety_gate",
        "is_emergency": is_emergency,
        "conversation_history": [{"agent": "safety_gate", "is_emergency": is_emergency}],
    }


def route_after_safety_gate(state: ConversationState) -> str:
    return "response_composition" if state.get("is_emergency") else "clarification"


def clarification_node(state: ConversationState) -> dict[str, Any]:
    """Pauses the graph (via LangGraph's interrupt()) whenever a required
    field is missing, reporting the same typed missing_fields and
    human-readable clarification_questions the API layer already returns.
    Resuming with Command(resume=<ClarificationAnswer-shaped dict>) merges
    the answer into intake_request and loops back to medical_intake_agent
    so missing_fields is recomputed with the new information — this may
    interrupt again if something else is still missing. A resume never
    re-enters supervisor_router (LangGraph resumes execution here
    directly) — this node itself records the "clarification_answer"
    intent for a resumed turn."""
    missing = state.get("missing_fields") or []
    if not missing:
        return {"just_resumed_clarification": False, "clarification_status": "none"}

    intake_response = state.get("intake_response") or {}
    answer = interrupt(
        {
            "missing_fields": missing,
            "clarification_questions": intake_response.get("clarification_questions", []),
        }
    )
    merged_request = _merge_clarification_answer(state["intake_request"], answer)
    return {
        "intake_request": merged_request,
        "just_resumed_clarification": True,
        "clarification_status": "resumed",
        "detected_intent": ConversationIntent.CLARIFICATION_ANSWER.value,
        "active_agent": "clarification",
        "conversation_history": [{"agent": "clarification", "clarification_status": "resumed"}],
    }


def route_after_clarification(state: ConversationState) -> str:
    if state.get("just_resumed_clarification"):
        return "normalize_intake"
    return "specialty_routing"


def clinical_intake_agent_node(state: ConversationState) -> dict[str, Any]:
    """Phase 3A: extensible, per-complaint clinical-context intake — a
    complete no-op (routes straight to specialty routing) for any concern
    that matches no known protocol (see
    app.services.clinical_intake_service.match_protocol), preserving every
    other concern's existing behavior unchanged.

    For a matched protocol, asks its ordered questions one at a time across
    successive interrupt/resume round-trips on this same thread, mirroring
    clarification_node's proven single-interrupt-per-invocation pattern
    (self-loop via route_after_clinical_intake) so both typed and
    voice-originated answers already work via the existing resume path —
    see app.schemas.conversation.ClarificationAnswer's
    clinical_answer_text/voice_transcript fields, and
    conversation_service.run_conversation_turn, which needs no change to
    recognize this node's interrupt payload (it reuses the same
    missing_fields/clarification_questions shape clarification_node
    already produces).

    SAFETY INVARIANT: whenever the protocol's "red_flags" question has been
    resolved by ANY means -- a clarification answer, or (see
    app.services.clinical_intake_service.build_initial_context) simply
    having been mentioned in the original message -- and
    clinical_red_flag_gate_node has not yet actually run for this context
    (ClinicalContext.red_flag_gate_passed), routing to that gate is this
    node's *first* priority: before asking about anything else still
    missing, and before ever building a ClinicalNavigationSummary. This is
    checked fresh on every single entry to this node, never inferred from
    "which field was most recently answered" -- a prior version of this
    check only fired when the red_flags question was itself just resumed,
    which meant a message that stated a red flag up front (e.g. "...and I
    also have chest pain") could have its red-flag information silently
    satisfied by the initial-message extraction pass and then never
    actually evaluated by the gate at all. Only
    clinical_red_flag_gate_node itself may ever set red_flag_gate_passed —
    see that node.

    Restricted tools: the matched protocol's own deterministic
    detect/extract/merge functions only (app.services.clinical_intake_service)
    — never a model call, never a diagnosis, and never the final emergency
    decision (that is clinical_red_flag_gate_node's job)."""
    intake_request = state.get("intake_request") or {}
    concern_text = _concern_text(intake_request) or ""
    voice_input = intake_request.get("voice_input") or {}
    source = FactSource.VOICE if voice_input.get("transcript") else FactSource.TEXT

    context_dict = state.get("clinical_context")
    if context_dict is None:
        protocol = match_protocol(concern_text)
        if protocol is None:
            return {
                "clinical_protocol": None,
                "clinical_context": None,
                "clinical_just_resumed": False,
                "clinical_next_step": "specialty_routing",
                "active_agent": "clinical_intake_agent",
                "conversation_history": [{"agent": "clinical_intake_agent", "protocol": None}],
            }
        intake_response = state.get("intake_response") or {}
        normalized_duration = (intake_response.get("normalized_intake") or {}).get("duration")
        duration_fact = (
            ClinicalFact(
                value=f"{normalized_duration['value']} {normalized_duration['unit']}",
                confidence=FactConfidence.EXPLICIT,
                source=source,
            )
            if normalized_duration
            else None
        )
        context = build_initial_context(
            protocol, concern_text, duration_fact=duration_fact, source=source
        )
    else:
        protocol = get_protocol(state.get("clinical_protocol"))
        assert protocol is not None
        context = ClinicalContext.model_validate(context_dict)

    if "red_flags" in context.answered_protocol_fields and not context.red_flag_gate_passed:
        return {
            "clinical_protocol": protocol.name,
            "clinical_context": context.model_dump(mode="json"),
            "clinical_just_resumed": False,
            "clinical_next_step": "red_flag_gate",
            "active_agent": "clinical_intake_agent",
            "conversation_history": [
                {"agent": "clinical_intake_agent", "pending_red_flag_gate": True}
            ],
        }

    missing_field = next_missing_field(protocol, context)
    if missing_field is None:
        # Reachable only once every question is answered AND (whenever the
        # protocol has a red_flags question at all) the gate above has
        # already run without an affirmed flag -- an affirmed gate routes
        # straight to response_agent and never returns here (see
        # route_after_clinical_red_flag_gate), so building a navigation
        # summary here can never follow an unevaluated or affirmed red flag.
        navigation = build_navigation_summary(protocol, context)
        return {
            "clinical_protocol": protocol.name,
            "clinical_context": context.model_dump(mode="json"),
            "clinical_just_resumed": False,
            "clinical_next_step": "specialty_routing",
            "clinical_navigation": navigation.model_dump(mode="json"),
            "active_agent": "clinical_intake_agent",
            "conversation_history": [
                {"agent": "clinical_intake_agent", "protocol": protocol.name, "complete": True}
            ],
        }

    # red_flags gets a dynamically-narrowed question (only the categories
    # still unknown) rather than the protocol's static prompt, since a
    # partial statement earlier (the original message, or an answer to a
    # different question) may have already resolved some of the four
    # categories -- see app.services.clinical_intake_service.
    # red_flags_question_text and the completeness-gap correction this
    # addresses.
    question = (
        red_flags_question_text(context)
        if missing_field == "red_flags"
        else question_for(protocol, missing_field)
    )
    answer = interrupt(
        {"missing_fields": ["clinical_answer"], "clarification_questions": [question]}
    )
    answer_text = ""
    if isinstance(answer, dict):
        answer_text = str(
            answer.get("clinical_answer_text") or answer.get("voice_transcript") or ""
        )

    updated_context = apply_field_answer(
        context, protocol, missing_field, answer_text, source=FactSource.CLARIFICATION_ANSWER
    )
    # Safety net: an answer to a *different* question (e.g. onset) may
    # still incidentally mention a red-flag symptom ("it came on suddenly,
    # and I've also had chest pain") -- scan every answer for that,
    # regardless of which field was actually being asked, so it is never
    # missed until the dedicated red-flags question happens to come up.
    updated_context = scan_incidental_red_flags(
        updated_context, answer_text, source=FactSource.CLARIFICATION_ANSWER
    )

    return {
        "clinical_protocol": protocol.name,
        "clinical_context": updated_context.model_dump(mode="json"),
        "clinical_just_resumed": True,
        # Always loop back through this same node rather than special-casing
        # "was red_flags the field just answered" here -- the invariant
        # check at the top of this function is what actually decides
        # whether to go to the gate next, on the *next* entry.
        "clinical_next_step": "clinical_intake_agent",
        "detected_intent": ConversationIntent.CLARIFICATION_ANSWER.value,
        "active_agent": "clinical_intake_agent",
        "conversation_history": [
            {"agent": "clinical_intake_agent", "field_answered": missing_field}
        ],
    }


def route_after_clinical_intake(state: ConversationState) -> str:
    return state.get("clinical_next_step") or "specialty_routing"


def clinical_red_flag_gate_node(state: ConversationState) -> dict[str, Any]:
    """Phase 3A: dedicated, deterministic red-flag safety gate (see
    app.safety.red_flag_rules) — a separate graph step from the existing
    user-declared safety_gate above. Reached whenever
    clinical_intake_agent_node's own invariant check (ClinicalContext.
    red_flags answered but red_flag_gate_passed still False) fires — which
    covers both "the dedicated red-flag question was just answered" and
    "red-flag information was already present in the original message" —
    always before specialty/provider routing or a navigation summary.
    Reads only the already-classified clinical_context.red_flags dict
    (itself produced entirely by deterministic keyword/negation matching,
    never a model call) — this node makes no independent judgment beyond
    "was at least one of these four already-classified flags AFFIRMED."
    Always marks red_flag_gate_passed True in the persisted
    clinical_context (regardless of outcome) so this exact check can never
    fire again for this same context — clinical_intake_agent_node relies on
    this to avoid re-entering the gate in an infinite loop once it has
    already run and found nothing affirmed. Reuses the existing
    is_emergency flag so response_agent_node's existing emergency handling
    (compose_response) applies automatically, without duplicating that
    logic."""
    context_dict = dict(state.get("clinical_context") or {})
    red_flags = context_dict.get("red_flags") or {}
    any_affirmed = any(status == MentionStatus.AFFIRMED.value for status in red_flags.values())
    context_dict["red_flag_gate_passed"] = True
    return {
        "clinical_context": context_dict,
        "is_emergency": any_affirmed or bool(state.get("is_emergency")),
        "clinical_red_flag_affirmed": any_affirmed,
        "active_agent": "clinical_red_flag_gate",
        "conversation_history": [{"agent": "clinical_red_flag_gate", "any_affirmed": any_affirmed}],
    }


def route_after_clinical_red_flag_gate(state: ConversationState) -> str:
    if state.get("clinical_red_flag_affirmed"):
        return "response_composition"
    return "clinical_intake_agent"


async def specialty_routing_agent_node(
    state: ConversationState, config: RunnableConfig
) -> dict[str, Any]:
    """Phase 1D deterministic specialty routing, reused unchanged as a
    controlled tool — it only ever selects a specialty already present in
    the curated catalog. Restricted tools: the specialty catalog matcher
    only — no medical reasoning, no provider lookup."""
    settings: Settings = _configurable(config)["settings"]
    request = state["intake_request"]
    voice_input = request.get("voice_input") or {}

    result = await route_to_specialty(
        preferred_specialty=request.get("preferred_specialty"),
        symptoms=request.get("symptoms") or [],
        main_concern=request.get("main_concern"),
        voice_transcript=voice_input.get("transcript"),
        vision_observation_text=_vision_observation_text(state.get("vision_observations") or []),
        settings=settings,
    )
    routing = {
        "specialty_slug": result.specialty_slug,
        "specialty_display_name": result.specialty_display_name,
        "method": result.method.value,
        "note": result.note,
    }
    return {
        "active_agent": "specialty_routing_agent",
        "routing": routing,
        "selected_specialty": result.specialty_slug,
        "conversation_history": [
            {"agent": "specialty_routing_agent", "specialty_slug": result.specialty_slug}
        ],
    }


def route_after_specialty_routing(state: ConversationState) -> str:
    routing = state.get("routing") or {}
    return "provider_search" if routing.get("specialty_slug") else "response_composition"


async def provider_search_agent_node(
    state: ConversationState, config: RunnableConfig
) -> dict[str, Any]:
    """Phase 1B deterministic provider search, reused unchanged as a
    controlled tool — no LLM call, no fabricated results, no medical
    reasoning. Only ever runs after specialty_routing_agent has already
    selected a specialty (see route_after_specialty_routing)."""
    session: AsyncSession = _configurable(config)["session"]
    routing = state["routing"] or {}
    specialty_slug = routing["specialty_slug"]

    page = await search_providers_page(session, specialty_slug=specialty_slug, limit=DEFAULT_LIMIT)
    response = build_provider_search_response(page, specialty_slug=specialty_slug)
    response_dict = response.model_dump(mode="json")
    return {
        "active_agent": "provider_search_agent",
        "provider_search": response_dict,
        "provider_results": response_dict,
        "conversation_history": [
            {
                "agent": "provider_search_agent",
                "result_count": len(response_dict.get("results") or []),
            }
        ],
    }


async def response_agent_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
    """Concise, non-diagnostic response text for a completed turn
    (greeting, emergency, unmatched, or routed with/without providers).
    Restricted tools: text composition only — it reads routing/
    provider_search/safety decisions already made upstream but never
    writes to them, so it cannot change a selected specialty, provider
    results, or the emergency determination. Never called for a paused
    clarification turn — that question is returned directly by the
    orchestration service before this node would run."""
    if state.get("detected_intent") in (
        ConversationIntent.GREETING.value,
        ConversationIntent.GENERAL_CHAT.value,
    ):
        # conversation_agent already produced the final, safe reply for
        # either intent -- finalize it unchanged rather than re-deriving
        # new text via the medical-routing response composer below, which
        # has no notion of either.
        text = state.get("response_text") or ""
        return {
            "active_agent": "response_agent",
            "response_text": text,
            "conversation_history": [
                {"agent": "response_agent", "intent": state.get("detected_intent")}
            ],
        }

    settings: Settings = _configurable(config)["settings"]
    text = await compose_response(
        is_emergency=state.get("is_emergency", False),
        missing_fields=state.get("missing_fields") or [],
        clarification_questions=(state.get("intake_response") or {}).get(
            "clarification_questions", []
        ),
        routing=state.get("routing"),
        provider_search=state.get("provider_search"),
        settings=settings,
        media_analysis_note=state.get("media_analysis_note"),
        vision_observations=state.get("vision_observations"),
        concern_text=_concern_text(state.get("intake_request") or {}),
        clinical_navigation=state.get("clinical_navigation"),
    )
    return {
        "active_agent": "response_agent",
        "response_text": text,
        "conversation_history": [
            {"agent": "response_agent", "intent": state.get("detected_intent")}
        ],
    }


async def text_to_speech_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
    """Best-effort speech synthesis over the final response text. Always
    returns a state update (never raises) — a failed or skipped synthesis
    still lets the text response reach the caller. An external modality
    service, not a reasoning agent: it never reads or influences routing,
    safety, or specialty/provider decisions."""
    if not state.get("generate_speech"):
        return {"audio": None, "audio_result": None, "active_agent": "text_to_speech"}

    configurable = _configurable(config)
    settings: Settings = configurable["settings"]
    provider: TextToSpeechProvider | None = configurable.get("tts_provider")
    if provider is None:
        provider = build_text_to_speech_provider(settings)

    audio = await synthesize_speech(
        text=state.get("response_text"), settings=settings, provider=provider
    )
    audio_dict = audio.model_dump(mode="json") if audio else None
    return {"audio": audio_dict, "audio_result": audio_dict, "active_agent": "text_to_speech"}
