# MedRoute AI — Project Guide for Claude Code

## Purpose

MedRoute AI is an open-model-first, multimodal healthcare appointment
navigation assistant. It collects symptoms and patient preferences, routes
the user to an appropriate medical specialty, searches available doctors,
and simulates appointment booking.

## Safety boundary (non-negotiable)

- This system is **not** a diagnostic or treatment system.
- It must **never** diagnose conditions, prescribe medication, or claim to
  replace a medical professional.
- Any output that suggests a specialty, urgency level, or next step is
  advisory only and must be clearly labeled as LLM-derived, not a medical
  determination.
- The system must support emergency escalation (directing users to
  emergency services) as a first-class path, separate from routine routing.
- LLM-generated content (routing suggestions, rationale text) must always
  be structurally and visibly separated from deterministic data (doctor
  records, appointment slots, booking confirmations). Never let generated
  text overwrite or masquerade as deterministic fields.
- All patient data used in development and tests is synthetic. Never use
  real patient information.

## Architecture

```
backend/app/
  api/            FastAPI routers (versioned under /api/v1)
  config/         Environment-driven settings (pydantic-settings)
  graph/          LangGraph multi-agent conversation graph (state.py, nodes.py,
                    build.py) — supervisor_router -> conversation_agent |
                    vision_agent | medical_intake_agent -> safety_gate ->
                    clarification -> clinical_intake_agent (Phase 3A, a
                    no-op unless a known complaint protocol matches) ->
                    clinical_red_flag_gate -> specialty_routing_agent ->
                    provider_search_agent -> response_agent -> text_to_speech
  providers/      Abstract provider interfaces + fake implementations
    llm/                TextLLMProvider
    speech_to_text/      SpeechToTextProvider (real: Deepgram primary, Groq fallback)
    text_to_speech/      TextToSpeechProvider (real: Deepgram primary, Groq fallback)
    vision/              VisionProvider (real: Groq multimodal chat model)
  safety/         Emergency disclaimers + user-declared emergency constants +
                    shared forbidden-medical-claim-term lists + Phase 3A's
                    dedicated deterministic red-flag rules (red_flag_rules.py)
  schemas/        Pydantic v2 domain models (intake, multimodal_intake, routing,
                    doctor, booking, provider_search, voice_intake, conversation,
                    vision, clinical_context)
  services/       Pure business logic (provider ranking/search, multimodal intake,
                    deterministic specialty routing, voice transcription orchestration,
                    duration extraction, response composition, text-to-speech
                    orchestration, conversation-graph orchestration, media upload
                    validation/normalization, vision-provider orchestration,
                    per-complaint clinical-context intake (clinical_intake_service.py),
                    negation-aware phrase classification (mention_classification_service.py))
  tools/          LangGraph tool implementations (future milestone)
  main.py         FastAPI app factory
streamlit_app/    Lightweight demo UI (calls the API only; no logic of its own)
```

Provider interfaces are abstract and async so real integrations (Groq,
Deepgram, etc.) can be dropped in later without changing calling code.
Fake implementations exist purely for local development and tests — they
must never be used to imply the app is production-ready.

## Coding standards

- Python 3.11, full type hints throughout.
- Pydantic v2 for all data models; no bare dicts crossing module boundaries.
- Keep provider-specific logic isolated inside `providers/<kind>/`. Callers
  depend only on the abstract base class.
- No premature abstraction: don't add config, flags, or layers for
  hypothetical future needs.
- No authentication, cloud deployment, or real external API calls until a
  milestone explicitly calls for them.

## Required validation commands

Run from `backend/`:

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
```

All must pass before considering a change complete.

## Current milestone

Phase 3A — extensible per-complaint clinical-context intake and a
dedicated, deterministic red-flag safety gate, vertically sliced through
one complaint: unilateral/one-sided leg swelling. `POST /api/v1/converse`
(and `/api/v1/media/analyze`) route through the multi-agent graph's
`clinical_intake_agent` node, inserted between the existing `clarification`
and `specialty_routing_agent` nodes — a complete no-op (falls straight
through, exactly as before Phase 3A) for any concern matching no known
protocol (`app/services/clinical_intake_service.py`'s `ClinicalProtocol`
registry, currently one entry). For a matched protocol, it asks its own
ordered questions one at a time via the same `interrupt()`/
`Command(resume=...)` mechanism `clarification` already uses (self-looping
via a conditional edge back to itself), skipping any question the original
message already answered, and accepting either a typed
(`clarification_answer.clinical_answer_text`) or spoken
(`clarification_answer.voice_transcript`) answer through the existing
resume path — no dedicated voice-UI change was needed.

A dedicated `clinical_red_flag_gate` node (`app/safety/red_flag_rules.py`)
runs immediately after the leg-swelling protocol's red-flag question is
answered, always before specialty/provider routing — deterministic
keyword/negation matching only (`app/services/mention_classification_service.py`),
no model call, distinguishing affirmed ("I have chest pain") from negated
("no chest pain") from unknown ("I'm not sure" — never treated as
affirmed). An affirmed flag reuses the existing `is_emergency`/
`EMERGENCY_SAFETY_MESSAGE` path unchanged; it never identifies a disease
and never overrides the existing user-declared emergency precedence
(checked first, unconditionally, by the pre-existing `safety_gate` node).

Once a protocol's questions are all answered with no red flag affirmed, a
bounded `ClinicalNavigationSummary` (`app/schemas/clinical_context.py`,
`diagnosis`/`treatment_recommendation` always `None`) is produced from a
small, reviewable per-protocol definition — never RAG, never scraped
content — and exposed as `ConversationResponse.clinical_navigation`, folded
into the existing response text and the Streamlit chat feed. Specialty
candidates are drawn only from the existing Phase 1B/1D catalog (extended
with a few leg-swelling-relevant keywords on Internal Medicine — no new
specialty, no bypassed routing contract).

`POST /api/v1/converse`, `POST /api/v1/navigate`, and `POST
/api/v1/media/analyze` are otherwise unchanged. See `docs/roadmap.md` for
the full Phase 3A entry (and the Phase 2C entry for image/video
understanding, delivered earlier).
