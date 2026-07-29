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
  graph/          LangGraph routing graph (future milestone)
  providers/      Abstract provider interfaces + fake implementations
    llm/                TextLLMProvider
    speech_to_text/      SpeechToTextProvider
    text_to_speech/      TextToSpeechProvider
  safety/         Emergency escalation & disclaimer logic (future milestone)
  schemas/        Pydantic v2 domain models (intake, routing, doctor, booking)
  services/       Business logic orchestration (future milestone)
  tools/          LangGraph tool implementations (future milestone)
  main.py         FastAPI app factory
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

Phase 0 — repository and engineering foundation only. No LLM calls, no
voice/image processing, no database, no real appointment logic. See
`docs/roadmap.md` for what comes next.
