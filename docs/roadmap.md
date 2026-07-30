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

## Phase 1A — NPPES Provider-Directory Ingestion Foundation (complete)

- Normalized PostgreSQL schema for provider identity, locations, and
  taxonomies (`providers`, `provider_locations`, `provider_taxonomies`),
  plus `ingestion_runs` tracking — built on the Phase 0.2 async
  SQLAlchemy/Alembic foundation.
- NPPES is a public government provider-directory registry (real
  doctor/organization identities, not patient data), so this does not
  conflict with the project's synthetic-patient-data safety rule.
- Chunked/streaming CSV transformation, validation, and idempotent
  PostgreSQL upserts; a CLI import command; validated only against a
  small representative fixture, not the full national dataset.
- Full national NPPES import, Qdrant/vector search, and symptom-to-
  specialty routing are explicitly out of scope here (Phase 1B+).

## Phase 1B — Deterministic Provider Discovery (current)

- All 15 NPPES taxonomy slots parsed (up from 3 in Phase 1A).
- A small, transparent specialty catalog mapping authoritative NUCC/CMS
  taxonomy codes to MedRoute specialties (`specialties`,
  `specialty_taxonomy_mappings`), seeded via a repeatable command.
- Deterministic (no-LLM) provider search — `GET /api/v1/specialties` and
  `GET /api/v1/providers/search` — with explainable ranking, pagination,
  and a disclaimer that NPPES does not validate licensing, credentials,
  quality, or appointment availability.
- Symptom-to-specialty inference, Qdrant/vector search, and appointment
  booking remain explicitly out of scope.

## Phase 1C — Symptom Intake Endpoint (planned)

- Intake endpoint accepting `SymptomIntake`, no routing yet — the
  `SymptomIntake` bullet originally grouped into Phase 1B, split out
  once Phase 1B's scope narrowed to provider discovery only.

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
