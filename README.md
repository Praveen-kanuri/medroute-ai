# MedRoute AI

An open-model-first, multimodal healthcare appointment navigation
assistant. It collects symptoms and preferences, routes users to an
appropriate medical specialty, searches available doctors, and simulates
appointment booking.

**MedRoute AI is not a diagnostic or treatment system.** It does not
diagnose conditions, prescribe medication, or replace medical
professionals. See [docs/safety-design.md](docs/safety-design.md).

## Status

Phase 1A: NPPES provider-directory ingestion foundation (normalized
provider/location/taxonomy schema, chunked CSV ingestion, idempotent
upserts, CLI import — validated against a small fixture only, not the
national dataset). No LLM integration, voice/image processing, Qdrant/vector
search, symptom-to-specialty routing, or real appointment logic yet.

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

## NPPES provider ingestion (Phase 1A)

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

**Run NPPES integration tests** (requires `docker compose up -d postgres`):

```bash
uv run pytest tests/integration/test_nppes_ingestion.py
```

### What's deferred

Phase 1A ships the ingestion *foundation*, validated only against the small
fixture above — **not** the full national NPPES dataset. Downloading and
importing the complete national file is Phase 1B+ work. Qdrant/vector
search and symptom-to-specialty routing are later phases still. NPPES also
does not provide live appointment availability — that remains a separate,
future concern.

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
