# Roadmap

## Phase 0 — Repository & Engineering Foundation (complete)

- Backend package skeleton (api, graph, providers, safety, schemas,
  services, tools, config).
- Abstract provider interfaces with fake implementations.
- Core Pydantic domain schemas.
- `/health` and `/api/v1/system/info` endpoints.
- Environment-based configuration.
- Docker, Docker Compose, CI workflow, docs scaffolding.

## Phase 0.2 — PostgreSQL Persistence Foundation (current)

- Async SQLAlchemy 2.x engine, session factory, and FastAPI session
  dependency; Alembic owns all schema changes (no `metadata.create_all()`
  at runtime).
- Minimal infrastructure-only schema used to validate the persistence
  stack end-to-end — no business-domain tables yet.
- `GET /api/v1/health/readiness` distinguishing process liveness from
  database-backed readiness.
- Docker Compose PostgreSQL service for local development.
- Doctor/specialty/appointment persistence tables are explicitly out of
  scope here; they arrive in Phase 1 once that domain schema is defined.

## Phase 1 — Symptom Intake & Deterministic Doctor Data (planned)

- Static/synthetic doctor and specialty dataset, persisted via the
  Phase 0.2 database foundation.
- Doctor search service and endpoint (deterministic, no LLM).
- Intake endpoint accepting `SymptomIntake`, no routing yet.

## Phase 2 — LLM-Backed Routing (planned)

- Real `TextLLMProvider` implementation (Groq).
- LangGraph graph: intake → routing decision.
- Safety subsystem: emergency keyword/escalation detection.

## Phase 3 — Booking Simulation (planned)

- Simulated appointment booking flow using deterministic slot data.
- `BookingRequest` → `BookingConfirmation` end-to-end (still simulated).

## Phase 4 — Multimodal Input (planned)

- Real `SpeechToTextProvider` (Deepgram) and `TextToSpeechProvider`.
- Voice-based intake flow.

## Phase 5 — Frontend (planned)

- React/TypeScript client consuming the FastAPI backend.

## Later (unscheduled)

- Authentication.
- Deployment/hosting.
