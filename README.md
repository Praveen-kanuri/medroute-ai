# MedRoute AI

An open-model-first, multimodal healthcare appointment navigation
assistant. It collects symptoms and preferences, routes users to an
appropriate medical specialty, searches available doctors, and simulates
appointment booking.

**MedRoute AI is not a diagnostic or treatment system.** It does not
diagnose conditions, prescribe medication, or replace medical
professionals. See [docs/safety-design.md](docs/safety-design.md).

## Status

Phase 1B: deterministic provider discovery (all 15 NPPES taxonomy slots, a
small transparent specialty catalog, a no-LLM provider search API with
explainable ranking and pagination — validated against a small fixture and
a bounded sample of a real official file, not the national dataset). No
LLM integration, voice/image processing, Qdrant/vector search,
symptom-to-specialty inference, or real appointment logic yet.

## Stack

Python 3.11 · uv · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · asyncpg ·
Alembic · LangGraph · pytest · Ruff · mypy · Docker · GitHub Actions.
Future: React/TypeScript frontend.

## Getting started

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
Copy-Item ..\.env.example ..\.env   # fill in local values, never commit .env
uv run uvicorn app.main:app --reload
```

No API keys are required to run locally — the app defaults to
`PROVIDER_MODE=fake` and uses deterministic fake providers. PostgreSQL is
optional too: the app and `GET /health` work with no `DATABASE_URL` set at
all (see [Database](#database) below for what needs it).

Visit `http://localhost:8000/health` and
`http://localhost:8000/api/v1/system/info`.

## Database

PostgreSQL is optional for basic app startup, but required for
`GET /api/v1/health/readiness` and for anything touching persistence.

**Start PostgreSQL** (from the repository root, same commands on bash and
PowerShell):

```bash
docker compose up -d postgres
```

**Check its health:**

```bash
docker compose ps
```

Wait until `postgres` shows `healthy` before running migrations.

**Apply migrations** (from `backend/`):

```bash
uv run alembic upgrade head
```

**Check the current migration:**

```bash
uv run alembic current
```

**Start FastAPI** (from `backend/`, once migrations are applied):

```bash
uv run uvicorn app.main:app --reload
```

**Test readiness:**

```bash
curl http://localhost:8000/api/v1/health/readiness
```

Returns HTTP 200 `{"status": "ready"}` when PostgreSQL is reachable, or
HTTP 503 `{"status": "unavailable"}` otherwise. `GET /health` (liveness —
"is the process running?") stays 200 regardless of database state;
readiness ("can this instance serve database-backed requests?") is the
one that reflects PostgreSQL's availability.

**Stop PostgreSQL without deleting its data:**

```bash
docker compose stop postgres
```

(Avoid `docker compose down -v`, which deletes the `medroute_postgres_data`
volume.)

### Tests: unit vs. integration

```bash
# Fast unit tests — no PostgreSQL required (DB-dependent tests skip themselves):
cd backend
uv run pytest -m "not integration"

# Database integration tests — require `docker compose up -d postgres` first:
uv run pytest -m integration

# Everything (unit tests + integration tests, auto-skipping if PostgreSQL is down):
uv run pytest
```

## Deterministic provider search (Phase 1B)

`GET /api/v1/specialties` lists the small, curated specialty catalog
(sourced from the NUCC taxonomy code set — see
[docs/prompts-used.md](docs/prompts-used.md) for the exact version/access
date). `GET /api/v1/providers/search` searches ingested NPPES providers by
specialty slug, taxonomy code, state, city, postal code, entity type, and
name, with pagination (`limit`, `offset`) and a stable, explainable
ranking (no LLM, no randomness): exact specialty match, primary taxonomy
first, exact postal/city/state match, active providers first, then a
stable name/NPI tie-break.

**Seed the specialty catalog** (idempotent — safe to run repeatedly, from
`backend/`, with PostgreSQL migrated):

```bash
uv run python -m app.catalog.seed_specialties
```

**Example request** (synthetic data only):

```bash
curl "http://localhost:8000/api/v1/providers/search?specialty=cardiology&state=TX&city=Dallas&limit=20"
```

Every response includes a `disclaimer` field: NPPES inclusion does not
verify licensing, credentials, quality of care, or appointment
availability.

## NPPES provider ingestion (Phase 1A)

All 15 official NPPES taxonomy slots are parsed as of Phase 1B (Phase 1A
originally read only 3, as a documented, easily-extendable starting
point).

NPPES (the National Plan and Provider Enumeration System) is a public U.S.
government registry of healthcare provider identities — real
doctors/organizations, not patient data, so ingesting it does not conflict
with this project's synthetic-patient-data rule.

**Why chunked:** the national NPPES file has millions of rows. The importer
streams the CSV with `csv.DictReader` (never loading the whole file into
memory) and writes one bounded chunk per database transaction, so a bad
chunk can't corrupt rows already committed by earlier chunks and no single
transaction spans the whole file.

**Why idempotent:** the same file (or an updated monthly file with
overlapping providers) may be imported more than once. Providers are matched
by NPI; locations and taxonomies are matched by natural keys
(`provider_id` + address purpose / taxonomy code). Re-running never creates
duplicates — matching rows are updated instead.

**Run the small fixture import** (from `backend/`, with PostgreSQL migrated):

```bash
uv run python -m app.ingestion.nppes --file tests/fixtures/nppes_sample.csv --chunk-size 500
```

Prints only row counts and a status (`completed` / `partially_completed` /
`failed`) — never a full file path, database URL, or credential. Re-running
it is safe: counts will show updates instead of new inserts.

**Run NPPES unit tests only** (fast, no PostgreSQL required):

```bash
uv run pytest tests/test_nppes_transform.py
```

**Run NPPES and provider-search integration tests** (requires
`docker compose up -d postgres`):

```bash
uv run pytest tests/integration/test_nppes_ingestion.py tests/integration/test_provider_search.py
```

### Real weekly-file pilot (manual, not part of the automated test suite)

Phase 1B was additionally validated against a bounded sample (first 10,000
rows) of a real official CMS weekly incremental NPPES V.2 file, run against
a separate throwaway PostgreSQL database (created and dropped for the
pilot only — never the app's own dev database). This is a manual
validation step, not something CI or `pytest` runs automatically, and it
never downloads the full national file or writes real provider data to
any persistent database. See docs/prompts-used.md for the exact source
file, results, and a compatibility fix it surfaced (a real column-name
difference from what Phase 1A had assumed).

### What's deferred

No full national NPPES import (monthly or weekly-cumulative) and no
automated/scheduled ingestion — only the small fixture and a manual,
bounded real-file pilot. Qdrant/vector search and symptom-to-specialty
inference are later phases still. NPPES also does not provide live
appointment availability — that remains a separate, future concern.

## Validation

Run from `backend/` (same commands on bash and PowerShell):

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
```

## Docs

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Model experiments](docs/model-experiments.md)
- [Safety design](docs/safety-design.md)
- [Prompts used](docs/prompts-used.md)
