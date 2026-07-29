# Roadmap

## Phase 0 — Repository & Engineering Foundation (current)

- Backend package skeleton (api, graph, providers, safety, schemas,
  services, tools, config).
- Abstract provider interfaces with fake implementations.
- Core Pydantic domain schemas.
- `/health` and `/api/v1/system/info` endpoints.
- Environment-based configuration.
- Docker, Docker Compose, CI workflow, docs scaffolding.

## Phase 1 — Symptom Intake & Deterministic Doctor Data (planned)

- Static/synthetic doctor and specialty dataset.
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

- Persistence layer (database-backed doctor/appointment data).
- Authentication.
- Deployment/hosting.
