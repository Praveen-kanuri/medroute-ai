# MedRoute AI

An open-model-first, multimodal healthcare appointment navigation
assistant. It collects symptoms and preferences, routes users to an
appropriate medical specialty, searches available doctors, and simulates
appointment booking.

**MedRoute AI is not a diagnostic or treatment system.** It does not
diagnose conditions, prescribe medication, or replace medical
professionals. See [docs/safety-design.md](docs/safety-design.md).

## Status

Phase 0: repository and engineering foundation. No LLM integration, voice
processing, image processing, database, or real appointment logic yet.

## Stack

Python 3.11 · uv · FastAPI · Pydantic v2 · LangGraph · pytest · Ruff ·
mypy · Docker · GitHub Actions. Future: React/TypeScript frontend.

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
`PROVIDER_MODE=fake` and uses deterministic fake providers.

Visit `http://localhost:8000/health` and
`http://localhost:8000/api/v1/system/info`.

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
