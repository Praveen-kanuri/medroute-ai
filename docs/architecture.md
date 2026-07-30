# Architecture

## Overview

MedRoute AI is a backend-first system built around a FastAPI service and a
future LangGraph routing graph. It is designed so LLM-backed intelligence
(routing suggestions) is strictly separated from deterministic data
(doctor directory, appointment slots, bookings).

## Layers

- **api/** — HTTP surface. Thin FastAPI routers, no business logic.
- **schemas/** — Pydantic v2 models shared across layers. The contract
  between API, graph, and services.
- **graph/** — LangGraph state machine that will orchestrate intake →
  routing → doctor search → booking simulation. Empty in Phase 0.
- **providers/** — Abstract interfaces for external capabilities (text
  LLM, speech-to-text, text-to-speech), each with a fake implementation
  for tests and local development, isolating vendor-specific code.
- **db/** — Async SQLAlchemy 2 foundation: declarative `Base`, a lazily
  created engine/session factory, a FastAPI session dependency, a
  `SELECT 1` readiness check (Phase 0.2), and the NPPES provider/location/
  taxonomy/ingestion-run models (Phase 1A). Alembic (`backend/alembic/`),
  not the app, owns all schema creation and changes.
- **ingestion/** — NPPES CSV column mapping (`nppes_mapping.py`), pure
  row transformation/validation (`nppes_transform.py`), chunked streaming
  ingestion with idempotent upserts (`nppes_service.py`), and a CLI
  (`nppes.py`, run via `python -m app.ingestion.nppes`). Validated against
  a small fixture (`backend/tests/fixtures/nppes_sample.csv`) — not the
  full national dataset, which is later phase work.
- **safety/** — Emergency escalation detection and disclaimers. Placeholder
  in Phase 0; real logic is a dedicated future milestone.
- **services/** — Business logic orchestrating schemas, providers, and the
  graph. Empty in Phase 0.
- **tools/** — LangGraph tool functions (e.g., doctor search). Empty in
  Phase 0.
- **config/** — Environment-driven settings via pydantic-settings.

## Data flow (target, future milestones)

1. User provides symptoms/preferences (text or voice) → `SymptomIntake`.
2. LangGraph graph produces a `RoutingDecision` (LLM-derived, advisory).
3. Deterministic doctor search returns `Doctor` and `AppointmentSlot`
   records, filtered by the routed specialty.
4. User selects a slot → `BookingRequest` → simulated `BookingConfirmation`.

At every step, LLM output only ever populates advisory fields
(`RoutingDecision`); doctor and appointment data always comes from
deterministic sources, never from generated text.

## Target conversation flow (Phase 1+, not yet implemented)

```mermaid
flowchart TD
    A[Patient describes symptoms] --> B[Extract symptoms and preferences]
    B --> C{Urgent warning signs?}
    C -->|Yes| D[Provide emergency escalation guidance]
    C -->|No| E[Route to medical specialty]
    E --> F[Search doctors and availability]
    F --> G[Rank suitable options]
    G --> H[Present doctors and time slots]
    H --> I[Confirm and book appointment]
```

This mirrors the LangGraph node sequence: safety check → symptom
extraction → specialty routing → doctor search → availability lookup →
recommendation → booking confirmation. The emergency-escalation branch
(`C` → `D`) always takes priority over routine routing, per the
[safety boundary](safety-design.md). Phase 0 ships none of this logic —
only the schemas and package skeleton that will eventually host it.

## Phase 0 / 0.2 / 1A scope

The package skeleton, abstract provider interfaces with fake
implementations, domain schemas, configuration, and engineering tooling
exist (Phase 0), plus an async SQLAlchemy/Alembic persistence foundation
and a `GET /api/v1/health/readiness` endpoint (Phase 0.2), plus a
normalized NPPES provider-directory schema and chunked, idempotent CSV
ingestion validated against a small fixture (Phase 1A). No graph wiring,
no real LLM/STT/TTS provider calls, no full national NPPES import, and no
symptom-to-specialty routing yet — those are Phase 1B+.
