# Prompts Used

No LLM prompts exist yet — Phase 0 has no real LLM integration, only fake
providers. This document will record, per graph node/service, the exact
system and user prompt templates used in production, along with the
rationale for their design and any safety-relevant constraints they
enforce (e.g., instructions never to diagnose).

## Engineering prompts (Claude Code)

These are development-tooling prompts given to Claude Code to build and
harden the repository scaffold. They are not LLM prompts used by the
running application.

### Phase 0 hardening review

> Perform a final Phase 0 hardening review of the current MedRoute AI
> repository. Do not introduce Phase 1 functionality, external APIs,
> database logic, LangGraph workflows, frontend code, or additional
> architecture. Review and fix only: appointment schema consistency
> (unique `slot_id`, timezone-aware datetimes, `end_time` > `start_time`);
> schema validation bounds (doctor rating, routing confidence, severity
> enum, non-empty symptom lists) plus tests; fake-provider configuration
> so the app starts without `GROQ_API_KEY`/`DEEPGRAM_API_KEY` (add
> `PROVIDER_MODE=fake`, never leak keys from `/api/v1/system/info`);
> provider interface contracts (no undocumented `**kwargs`, concise
> docstrings); CI running `uv sync --frozen`, `ruff check`, `ruff format
> --check`, `mypy`, `pytest` from `backend/`; and documentation updates
> (this entry, Windows PowerShell-compatible README commands, preserved
> non-diagnostic safety boundary).

### Phase 0.2 — PostgreSQL persistence foundation

> Implement MedRoute AI Phase 0.2: the PostgreSQL persistence foundation
> using SQLAlchemy 2.x, asyncpg, Alembic, FastAPI dependency injection,
> database readiness checking, and automated persistence tests. Do not
> implement NPPES ingestion, provider search, Qdrant, embeddings, routing
> intelligence, frontend functionality, authentication, or production
> deployment. Treat the repository documentation and existing tests as
> authoritative; do not invent business-domain tables (doctors,
> specialties, locations, appointments) if the phase only specifies
> infrastructure — use the smallest migration-supported persistence
> object that validates the stack end-to-end instead. Extend the existing
> Pydantic settings (`DATABASE_URL` as `SecretStr`, `DATABASE_ECHO`,
> `database_configured`) rather than a parallel config system; never log
> or expose the connection URL. Build an async SQLAlchemy foundation
> (`DeclarativeBase`, naming convention, lazily created `AsyncEngine`,
> `async_sessionmaker`, a FastAPI session dependency, engine disposal on
> shutdown, a `SELECT 1` connectivity check) with Alembic as the only
> mechanism that creates or changes tables — no `metadata.create_all()`
> at runtime. Add `GET /api/v1/health/readiness` distinct from the
> existing liveness `GET /health`: a database outage must 503 readiness
> without failing liveness. Add unit tests (settings masking, session
> dependency behavior, readiness unavailable when unconfigured/
> unreachable) and integration tests requiring a real PostgreSQL
> (connectivity, ORM persist/retrieve with rollback, migrated tables
> exist, readiness 200) that skip gracefully rather than fail when no
> database is reachable. Update Docker Compose with a `postgres` service,
> CI with a PostgreSQL service container plus `alembic upgrade head`, and
> documentation for starting Postgres, running migrations, and the
> liveness/readiness distinction — without claiming any later-phase
> functionality (NPPES, Qdrant, routing) is implemented.
>
> This prompt surfaced a real conflict with `docs/roadmap.md`, which had
> no "Phase 0.2" and deferred the persistence layer to "Later
> (unscheduled)" after Phase 5. Per this task's own instruction to stop
> on internal documentation conflicts, the roadmap was updated first
> (inserting Phase 0.2 ahead of Phase 1, infrastructure-only, persistence
> line removed from "Later") before any code was written.
