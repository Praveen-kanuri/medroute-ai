# MedRoute AI — Project Wiki

**An Open-Model Multimodal Healthcare Appointment Assistant**

This document is the single-page reference for the MedRoute AI project: what
it is, why it's built the way it is, the current state of the code, and
where it's headed. It consolidates the contents of `CLAUDE.md`, `README.md`,
and everything under `docs/` into one place. For the authoritative,
narrower sources, see the links in [Source documents](#source-documents).

---

## Table of contents

- [1. Overview](#1-overview)
- [2. Safety boundary (non-negotiable)](#2-safety-boundary-non-negotiable)
- [3. Architecture](#3-architecture)
- [4. Data flow](#4-data-flow-target-future-milestones)
- [5. Domain schemas](#5-domain-schemas)
- [6. Providers](#6-providers)
- [7. API surface](#7-api-surface)
- [8. Configuration](#8-configuration)
- [9. Engineering & tooling](#9-engineering--tooling)
- [10. Repository layout](#10-repository-layout)
- [11. Getting started](#11-getting-started)
- [12. Validation commands](#12-validation-commands)
- [13. Roadmap](#13-roadmap)
- [14. Coding standards](#14-coding-standards)
- [15. Glossary](#15-glossary)
- [Source documents](#source-documents)

---

## 1. Overview

MedRoute AI is an **open-model-first, multimodal healthcare appointment
navigation assistant**. It collects a patient's self-reported symptoms and
preferences, suggests an appropriate medical specialty, searches available
doctors, and simulates appointment booking.

It is explicitly **backend-first**: the current build is a FastAPI service
with a package skeleton for a future LangGraph orchestration graph. A
React/TypeScript frontend is planned but not yet started (Phase 5).

**Stack:** Python 3.11 · uv · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) ·
asyncpg · Alembic · LangGraph (future wiring) · pytest · Ruff · mypy ·
Docker · GitHub Actions.

## 2. Safety boundary (non-negotiable)

This is the most important section in the project. Every design decision
defers to it.

- MedRoute AI **is not** a diagnostic or treatment system.
- It must **never** diagnose conditions, prescribe medication, or claim to
  replace a medical professional.
- Any output suggesting a specialty, urgency level, or next step is
  **advisory only** and must be clearly labeled as LLM-derived, not a
  medical determination.
- **Emergency escalation** must exist as a first-class path, separate from
  routine specialty routing — directing users to emergency services always
  takes priority.
- **LLM-generated content must be structurally and visibly separated from
  deterministic data.** Routing suggestions and rationale text (LLM-derived)
  must never overwrite or masquerade as deterministic fields (doctor
  records, appointment slots, booking confirmations).
- All patient data used in development and tests is **synthetic**. Real
  patient data must never be used in this repository.

See [docs/safety-design.md](docs/safety-design.md) for the full design,
including how the separation is enforced in the schema layer.

## 3. Architecture

MedRoute AI is built around a FastAPI service and a future LangGraph
routing graph, designed so that LLM-backed intelligence (routing
suggestions) is strictly separated from deterministic data (doctor
directory, appointment slots, bookings).

| Layer | Purpose | Status |
|---|---|---|
| `api/` | HTTP surface — thin FastAPI routers, no business logic | 3 endpoints live |
| `schemas/` | Pydantic v2 models — the contract between API, graph, and services | Fully defined |
| `db/` | Async SQLAlchemy 2 foundation: declarative `Base`, engine/session lifecycle, readiness check, NPPES provider/location/taxonomy/ingestion-run models | Persistence foundation + NPPES schema |
| `ingestion/` | NPPES CSV mapping, transformation/validation, chunked streaming upsert service, CLI (`python -m app.ingestion.nppes`) | Validated against a small fixture only — not the national dataset |
| `graph/` | LangGraph state machine: intake → routing → doctor search → booking | Empty package, future milestone |
| `providers/` | Abstract interfaces for external capabilities (LLM, STT, TTS), each with a fake implementation | Interfaces + fakes done, real integrations future |
| `safety/` | Emergency escalation detection and disclaimers | Placeholder constants only |
| `services/` | Business logic orchestrating schemas, providers, and the graph | Empty package, future milestone |
| `tools/` | LangGraph tool functions (e.g., doctor search) | Empty package, future milestone |
| `config/` | Environment-driven settings via pydantic-settings | Done |

Schema migrations for `db/` live outside `app/`, under `backend/alembic/` —
Alembic is the only mechanism that creates or changes tables; the app never
calls `Base.metadata.create_all()`.

NPPES is a public government provider-directory registry — real
doctor/organization identities, not patient data — so ingesting it does not
conflict with the synthetic-patient-data safety rule below.

## 4. Data flow (target, future milestones)

1. User provides symptoms/preferences (text or voice) → `SymptomIntake`.
2. LangGraph graph produces a `RoutingDecision` (LLM-derived, advisory).
3. Deterministic doctor search returns `Doctor` and `AppointmentSlot`
   records, filtered by the routed specialty.
4. User selects a slot → `BookingRequest` → simulated `BookingConfirmation`.

At every step, LLM output only ever populates advisory fields
(`RoutingDecision`); doctor and appointment data always comes from
deterministic sources, never from generated text.

## 5. Domain schemas

All schemas live in `backend/app/schemas/`, are Pydantic v2, and carry no
business logic — validation rules only.

| Model | File | Key fields & rules |
|---|---|---|
| `SymptomIntake` | `intake.py` | `symptoms: list[str]` (min length 1 — empty lists rejected); `duration_days: int` (≥ 0); `severity: Severity` enum (`mild`/`moderate`/`severe`); `notes`, `preferred_language` |
| `RoutingDecision` | `routing.py` | **LLM-derived, advisory only, never a diagnosis.** `suggested_specialty`; `confidence: float` (0.0–1.0); `rationale: str`; `is_emergency: bool` — the schema hook for future emergency escalation |
| `Doctor` | `doctor.py` | Deterministic, never LLM-generated. `id`, `name`, `specialty`, `location`; `rating: float` (0.0–5.0) |
| `AppointmentSlot` | `doctor.py` | Deterministic, never LLM-generated. `slot_id` (unique identifier, referenced by `BookingRequest.slot_id`); `doctor_id`; `start_time` / `end_time` (must be **timezone-aware**; `end_time` must be later than `start_time`); `is_available: bool` |
| `BookingRequest` | `booking.py` | Synthetic, simulated booking only. `patient_name`; `slot_id` (references `AppointmentSlot.slot_id`); `contact_email` (optional) |
| `BookingConfirmation` | `booking.py` | Deterministic/simulated, never LLM-generated. `booking_id`; `status: BookingStatus` enum (`confirmed`/`failed`/`cancelled`); `slot_id` |

**Safety-relevant design note:** `RoutingDecision` is the only model that
holds LLM-generated text. Every other model in this list is deterministic
by contract — this is how the safety boundary's "structural separation"
rule is enforced in code, not just in documentation.

## 6. Providers

Abstract, async interfaces in `backend/app/providers/<kind>/base.py`, each
with a deterministic fake implementation in `<kind>/fake.py` for tests and
local development without needing any API keys.

| Provider | Method | Fake behavior |
|---|---|---|
| `TextLLMProvider` | `async generate(prompt: str) -> str` | Echoes the prompt: `"[fake-llm-response] echo: {prompt}"` |
| `SpeechToTextProvider` | `async transcribe(audio_bytes: bytes) -> str` | Reports byte count: `"[fake-transcript] {n} bytes received"` |
| `TextToSpeechProvider` | `async synthesize(text: str) -> bytes` | Returns `f"[fake-audio]{text}".encode()` |

Design intent: callers depend only on the abstract base class, so real
integrations (Groq for LLM, Deepgram for speech) can be dropped in later
without changing calling code. Contracts are deliberately minimal — no
undocumented provider-specific `**kwargs`, no generic provider framework.

## 7. API surface

Versioned under `/api/v1`, aggregated in `backend/app/api/router.py`.

| Method & path | Purpose | Notes |
|---|---|---|
| `GET /health` | Liveness — is the process running? | Returns `{"status": "ok"}`; unaffected by database state |
| `GET /api/v1/system/info` | App metadata | Returns `app_name`, `app_env`, `log_level` — **never** returns API keys or secrets |
| `GET /api/v1/health/readiness` | Readiness — can this instance serve database-backed requests? | `SELECT 1` against PostgreSQL; 200 `{"status": "ready"}` or 503 `{"status": "unavailable"}` — **never** returns connection details |

## 8. Configuration

Environment-driven via `pydantic-settings` (`backend/app/config/settings.py`),
loaded from a `.env` file (see `.env.example` at the repo root — never
commit a real `.env`).

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | |
| `LOG_LEVEL` | `INFO` | |
| `PROVIDER_MODE` | `fake` | Fake providers require no credentials; this is the only supported mode in Phase 0 |
| `GROQ_API_KEY` | `None` (optional) | Not required while `PROVIDER_MODE=fake` |
| `DEEPGRAM_API_KEY` | `None` (optional) | Not required while `PROVIDER_MODE=fake` |
| `DATABASE_URL` | `None` (optional) | `SecretStr`; async (`postgresql+asyncpg://…`); required only for persistence and `/api/v1/health/readiness` |
| `DATABASE_ECHO` | `false` | SQLAlchemy engine SQL echo, local debugging only |
| `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | — | Consumed by Docker Compose's `postgres` service; must match `DATABASE_URL`'s credentials |

The app starts and serves requests with **no credentials set at all**, as
long as it stays in fake mode — this is covered by tests
(`test_config.py`, `test_health.py`). `DATABASE_URL` and `DATABASE_ECHO`
are never exposed via `repr()`, serialization, endpoints, or exceptions —
see `test_config.py`'s secret-masking tests.

## 9. Engineering & tooling

- **Package manager:** `uv` (dependencies + dev deps in `backend/pyproject.toml`)
- **Lint:** Ruff (`ruff check`, plus `ruff format --check` for formatting)
- **Types:** mypy, strict mode
- **Tests:** pytest (`pytest-asyncio`, `httpx` for `TestClient`)
- **Containerization:** `backend/Dockerfile` (slim Python 3.11, `uv sync --no-dev`,
  uvicorn entrypoint) + root `docker-compose.yml` (`backend` service, plus a
  `postgres` service — `postgres:17-alpine`, named volume
  `medroute_postgres_data`, `pg_isready` health check)
- **Migrations:** Alembic, async-compatible `backend/alembic/env.py` — reads
  `DATABASE_URL` from `Settings` at runtime (never hardcoded in `alembic.ini`)
- **CI:** `.github/workflows/ci.yml` — on push to `main` and on PRs, runs a
  PostgreSQL 17 service container, then from the `backend/` working directory:
  `uv sync --frozen` → `alembic upgrade head` → `ruff check .` →
  `ruff format --check .` → `mypy app` → `pytest`

## 10. Repository layout

```
MedRoute-AI/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app factory (disposes DB engine on shutdown)
│   │   ├── config/settings.py       # pydantic-settings config
│   │   ├── api/
│   │   │   ├── router.py            # aggregates v1 routes
│   │   │   └── v1/{health,readiness,system}.py
│   │   ├── schemas/                 # intake, routing, doctor, booking
│   │   ├── db/                      # base, session, models (health check + NPPES schema)
│   │   ├── ingestion/                # nppes_mapping, nppes_transform, nppes_service, nppes CLI
│   │   ├── providers/{llm,speech_to_text,text_to_speech}/{base,fake}.py
│   │   ├── graph/                   # placeholder — future LangGraph graph
│   │   ├── safety/constants.py      # disclaimer text, no logic yet
│   │   ├── services/                # placeholder — future business logic
│   │   └── tools/                   # placeholder — future LangGraph tools
│   ├── alembic/                     # async env.py + versions/ (Alembic-owned schema)
│   ├── alembic.ini                  # no embedded credentials — URL set by env.py
│   ├── tests/                       # health, config, db, nppes_transform, providers, schemas
│   │   ├── fixtures/                # small synthetic NPPES CSV fixture
│   │   └── integration/             # requires a real PostgreSQL, self-skips otherwise
│   ├── pyproject.toml
│   └── Dockerfile
├── docker-compose.yml                # backend + postgres services
├── .env.example
├── CLAUDE.md                        # instructions for Claude Code on this repo
├── README.md
├── PROJECT_WIKI.md                  # this file
└── docs/
    ├── architecture.md
    ├── roadmap.md
    ├── model-experiments.md
    ├── safety-design.md
    └── prompts-used.md
```

Note: the repository root is tracked by Git (active branch `R1`, remote
`origin`); `main` is the stable branch.

## 11. Getting started

**bash / macOS / Linux:**

```bash
cd backend
uv sync --all-extras --dev
cp ../.env.example ../.env   # fill in local values, never commit .env
uv run uvicorn app.main:app --reload
```

**Windows PowerShell:**

```powershell
cd backend
uv sync --all-extras --dev
Copy-Item ..\.env.example ..\.env
uv run uvicorn app.main:app --reload
```

No API keys are needed — the app defaults to `PROVIDER_MODE=fake`. Visit
`http://localhost:8000/health` and `http://localhost:8000/api/v1/system/info`.

## 12. Validation commands

Run from `backend/` (identical on bash and PowerShell). All must pass
before a change is considered complete:

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
```

## 13. Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 0** | Repository & engineering foundation: package skeleton, provider interfaces + fakes, domain schemas, `/health` + `/system/info`, config, Docker/CI/docs | Complete |
| **Phase 0.2** | PostgreSQL persistence foundation: async SQLAlchemy engine/session, Alembic migrations, `/api/v1/health/readiness`, Compose `postgres` service — infrastructure only, no business tables | Complete |
| **Phase 1A** | NPPES provider-directory ingestion foundation: `providers`/`provider_locations`/`provider_taxonomies`/`ingestion_runs` schema, chunked streaming CSV ingestion, idempotent upserts, CLI import — validated against a small fixture, not the national dataset | **Current** |
| **Phase 1B** | Symptom intake & doctor search: doctor search service/endpoint reading the Phase 1A schema, intake endpoint (no routing yet) | Planned |
| **Phase 2** | LLM-backed routing: real `TextLLMProvider` (Groq), LangGraph intake→routing graph, safety subsystem (emergency keyword/escalation detection) | Planned |
| **Phase 3** | Booking simulation: `BookingRequest` → `BookingConfirmation` end-to-end (still simulated, not real scheduling) | Planned |
| **Phase 4** | Multimodal input: real `SpeechToTextProvider` (Deepgram) + `TextToSpeechProvider`, voice-based intake | Planned |
| **Phase 5** | Frontend: React/TypeScript client consuming the FastAPI backend | Planned |
| Later (unscheduled) | Authentication, deployment/hosting | Unscheduled |

## 14. Coding standards

- Python 3.11, full type hints throughout.
- Pydantic v2 for all data models — no bare dicts crossing module boundaries.
- Provider-specific logic stays isolated inside `providers/<kind>/`; callers
  depend only on the abstract base class.
- No premature abstraction — no config, flags, or layers for hypothetical
  future needs.
- No authentication, cloud deployment, or real external API calls until a
  milestone explicitly calls for them.

## 15. Glossary

- **Advisory** — Output that suggests but does not decide; always
  LLM-derived and labeled as non-authoritative (applies to `RoutingDecision`).
- **Deterministic data** — Data from a fixed, non-generative source (doctor
  records, slots, confirmations); never produced by an LLM.
- **Fake provider** — A deterministic stand-in for a real external API
  (LLM/STT/TTS), used so the app runs fully offline with no credentials.
- **PROVIDER_MODE** — Config flag controlling whether fake or real
  providers are wired up; only `fake` exists in Phase 0.
- **Emergency escalation** — The (future) safety path that directs a user
  to emergency services instead of routine specialty routing.
- **Liveness vs. readiness** — Liveness (`GET /health`) asks "is the
  process running?" and ignores database state. Readiness
  (`GET /api/v1/health/readiness`) asks "can this instance serve
  database-backed requests right now?" and reflects PostgreSQL reachability.
- **Alembic** — The migration tool that exclusively owns schema creation
  and changes; the app itself never calls `Base.metadata.create_all()`.
- **NPPES** — The National Plan and Provider Enumeration System, a public
  U.S. government registry of healthcare provider identities (real
  doctors/organizations, not patient data). Phase 1A ingests a small sample
  of this format; the full national file is a later phase.
- **Idempotent upsert** — Re-running the same ingestion file never creates
  duplicate rows; matching records (by NPI, or by a natural key for
  locations/taxonomies) are updated in place instead.

---

## Source documents

This wiki summarizes and links the following canonical files — when in
doubt, they win:

- [CLAUDE.md](CLAUDE.md) — instructions for Claude Code working in this repo
- [README.md](README.md) — quick start
- [docs/architecture.md](docs/architecture.md)
- [docs/roadmap.md](docs/roadmap.md)
- [docs/safety-design.md](docs/safety-design.md)
- [docs/model-experiments.md](docs/model-experiments.md)
- [docs/prompts-used.md](docs/prompts-used.md)
