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

### Phase 1A — NPPES provider ingestion foundation

> Implement MedRoute AI Phase 1A: the structured NPPES provider-data
> persistence schema and a reliable, chunked CSV ingestion foundation.
> This phase must establish the normalized PostgreSQL schema, Alembic
> migration, NPPES transformation logic, idempotent batch upserts,
> ingestion tracking, CLI execution, and automated tests. Validate the
> pipeline using a small representative test fixture. Do not download or
> import the complete national NPPES dataset. Do not implement Qdrant,
> embeddings, routing, chatbot integration, scheduling, authentication, or
> frontend work. Design four tables: `providers` (NPI unique + 10-digit
> format check, entity-type check for individual vs. organization),
> `provider_locations` (mailing/practice addresses, natural key on
> provider + address purpose), `provider_taxonomies` (specialty/license
> assignments, natural key on provider + taxonomy code, supporting
> multiple slots), and `ingestion_runs` (status, started/completed
> timestamps, row counts, a bounded sanitized error summary — never a
> full row dump or a private file path). Stream the CSV with
> `csv.DictReader` in configurable chunks, one transaction per chunk, using
> PostgreSQL-native `INSERT ... ON CONFLICT DO UPDATE` for idempotent
> upserts. Never log or persist a full absolute input path — only the
> basename. Never read, print, or touch the real root `.env` file.
>
> This prompt surfaced a conflict with `docs/roadmap.md`, whose Phase 1
> entry described a "static/synthetic doctor and specialty dataset" with
> no "Phase 1A"/"Phase 1B" split, while this task wanted real NPPES
> ingestion positioned as the immediate next step. Per the same
> stop-on-conflict rule applied in the Phase 0.2 prompt, the roadmap was
> updated first — splitting Phase 1 into 1A (this ingestion foundation)
> and 1B (doctor search/intake endpoints built on top) — before any code
> was written. NPPES is a public government provider-directory registry
> (real doctor/organization identities, not patient data), so ingesting it
> does not conflict with the project's synthetic-patient-data safety rule.
>
> A design issue surfaced during test-writing and was fixed before
> committing: the fixture's intentional "repeated provider row" (same NPI
> appearing twice, to test upsert behavior) landed in the same processing
> chunk as its original under the default chunk size, so in-memory
> same-chunk de-duplication silently discarded the original row's second
> taxonomy before it ever reached the database — correct behavior for a
> true same-chunk duplicate, but it also revealed that insert/update
> counts were computed only from de-duplicated NPIs, undercounting a
> chunk's true row-for-row accounting. Fixed by counting insert-vs-update
> per original row (before de-duplication) so `rows_inserted + rows_updated
> == valid rows processed` always holds, and by using a smaller chunk size
> in the relevant tests so the repeated row exercises a genuine cross-chunk
> database update instead of an in-memory same-chunk overwrite.
