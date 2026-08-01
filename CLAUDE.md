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
  graph/          Phase 2B LangGraph conversation graph (state.py, nodes.py, build.py)
  providers/      Abstract provider interfaces + fake implementations
    llm/                TextLLMProvider
    speech_to_text/      SpeechToTextProvider (real: Groq Whisper)
    text_to_speech/      TextToSpeechProvider (real: Deepgram Aura)
  safety/         Emergency disclaimers + user-declared emergency constants
  schemas/        Pydantic v2 domain models (intake, multimodal_intake, routing,
                    doctor, booking, provider_search, voice_intake, conversation)
  services/       Pure business logic (provider ranking/search, multimodal intake,
                    deterministic specialty routing, voice transcription orchestration,
                    duration extraction, response composition, text-to-speech
                    orchestration, conversation-graph orchestration)
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

Phase 2B — LangGraph conversational orchestration with spoken output. A
stateless-per-turn `POST /api/v1/converse` endpoint runs a compiled
LangGraph graph (`app/graph/`) that composes every existing service as a
controlled tool: `normalize_intake` (Phase 1C `evaluate_intake`) ->
`safety_gate` (user-declared emergency precedence only, never inferred
from text) -> `clarification` (pauses via LangGraph's `interrupt()` when
a required field is missing, reporting typed `missing_fields`) ->
`specialty_routing` (Phase 1D deterministic routing) -> `provider_search`
(Phase 1B deterministic search) -> `response_composition` (a concise,
non-diagnostic summary; deterministic by default, optional validated Groq
rephrasing) -> `text_to_speech` (optional, Deepgram Aura). A `thread_id` +
an `InMemorySaver` checkpointer let the caller resume a paused
clarification turn with a typed answer (`Command(resume=...)`), merging
it into the same conversation state and looping back through
`normalize_intake` — the confirmed voice transcript and every other
intake field are preserved automatically; no re-upload or re-transcription
is ever needed. `POST /api/v1/navigate` (Phase 1D) is unchanged and still
available. Text-to-speech is best-effort: any missing configuration,
oversized text, or provider failure returns no audio but always still
returns the text response; tests always inject a fake TTS provider and
never call Deepgram. See `docs/roadmap.md` for Phase 2C (vision input).
