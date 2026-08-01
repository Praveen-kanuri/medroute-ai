# MedRoute AI

An open-model-first, multimodal healthcare appointment navigation
assistant. It collects symptoms and preferences, routes users to an
appropriate medical specialty, searches available doctors, and simulates
appointment booking.

**MedRoute AI is not a diagnostic or treatment system.** It does not
diagnose conditions, prescribe medication, or replace medical
professionals. See [docs/safety-design.md](docs/safety-design.md).

## Status

Phase 2A: real voice intake with Groq Whisper.
`POST /api/v1/voice/transcribe` accepts one uploaded audio file and returns
a raw speech-to-text transcript — never persisted, never auto-submitted.
The Streamlit demo UI requires the user to review and explicitly confirm
the transcript before it flows through the existing Phase 1C `voice_input`
contract into Phase 1D's `POST /api/v1/navigate` (intake → specialty
routing → provider search). See [Voice intake](#voice-intake-phase-2a)
below.

Phase 1D: controlled specialty routing & navigation demo.
`POST /api/v1/navigate` composes the Phase 1C intake contract, a new
deterministic (no-LLM-by-default) specialty router that only ever selects
from the Phase 1B specialty catalog, and Phase 1B's provider search into
one end-to-end flow — preserving emergency/clarification precedence
exactly. A lightweight Streamlit demo UI calls this endpoint. See
[Navigation demo](#navigation-demo--specialty-routing-phase-1d) below.

## Stack

Python 3.11 · uv · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · asyncpg ·
Alembic · Streamlit (demo UI) · LangGraph · pytest · Ruff · mypy · Docker ·
GitHub Actions. Future: React/TypeScript **production** frontend (Phase 5,
separate from and not replaced by the Streamlit demo).

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

## Multimodal intake (Phase 1C)

**MedRoute AI does not diagnose conditions, recommend treatment, or
autonomously determine emergencies.** `POST /api/v1/intake/validate` is a
stateless validation/normalization contract only:

- **Media URLs are references only.** Image and video URLs are validated
  as strings (HTTPS, no embedded credentials, no localhost/private-IP-
  literal hosts, no fragment, no query string) but are **never fetched,
  opened, or analyzed** in Phase 1C. This is contract-level hardening, not
  a complete SSRF defense — a future media-processing service must
  independently revalidate destinations at fetch time.
- **The voice transcript must already exist.** `voice_input.transcript` is
  text the *caller* generated; this endpoint performs no speech-to-text
  and does not verify the transcript's accuracy.
- **No interpretation occurs in Phase 1C.** No LLM call, no vision-model
  call, no symptom-to-specialty inference, no urgency scoring.
- **Emergency indicators are user-declared only** (`emergency_concern`,
  `emergency_signals`) — never inferred from text, transcript, or media.
  A declared emergency always takes precedence and directs U.S. users to
  call 911 ([911.gov](https://www.911.gov/calling-911)); **MedRoute AI
  does not provide emergency assistance itself.**
- **Stateless and non-persistent.** No database table, no cache, nothing
  saved. The `intake_id` is generated per request and never stored.
- **Privacy-conscious logging.** Only safe operational metadata is logged
  (intake_id, status, per-modality booleans/counts, whether
  duration/location were supplied, processing time) — never symptom text,
  main_concern, transcripts, emergency-signal values, or media URLs. No
  HIPAA-compliance claim is made; see the [HHS minimum-necessary
  guidance](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/)
  this logging approach follows in spirit.
- **Do not submit identifying information** (name, date of birth, phone,
  email, SSN, insurance ID, medical-record ID, payment details, or other
  government identifiers) — this endpoint does not attempt to detect or
  redact such data.

**Example request** (synthetic data only):

```bash
curl -X POST http://localhost:8000/api/v1/intake/validate \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["persistent knee discomfort", "swelling"],
    "main_concern": "Discomfort after routine exercise",
    "duration": {"value": 3, "unit": "days"},
    "location": {"city": "Dallas", "state": "TX", "postal_code": "75201"},
    "vision_inputs": {"image_urls": ["https://media.example.org/intake/example-image.jpg"]}
  }'
```

**Run intake tests:**

```bash
uv run pytest tests/test_multimodal_intake_schema.py tests/test_multimodal_intake_service.py tests/test_multimodal_intake_api.py
```

Phase 1D builds directly on this contract — see the next section. Real
speech-to-text/text-to-speech (Phase 4) and the production frontend
(Phase 5) remain later milestones. See the [FDA's clinical decision
support guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
for the kind of software category MedRoute AI is deliberately staying out
of.

## Navigation demo & specialty routing (Phase 1D)

`POST /api/v1/navigate` composes three already-existing pieces without
duplicating their logic: Phase 1C's intake evaluation, a new specialty
router, and Phase 1B's provider search.

- **Emergency and clarification precedence are unchanged.** Routing and
  provider search only run once intake status is
  `ready_for_multimodal_processing`.
- **Routing is controlled.** It only ever selects a specialty already in
  the Phase 1B catalog (`GET /api/v1/specialties`) — never an invented
  one. By default this is deterministic keyword matching (no model
  call). An explicit `preferred_specialty` always bypasses matching,
  after catalog validation.
- **Never a diagnosis.** No treatment advice, urgency score, or medical-
  certainty claim, ever.
- **Media is still not processed.** If `vision_inputs` was supplied, the
  response includes a plain `media_note` saying image/video analysis
  isn't available in this demo — text/voice-transcript routing still
  proceeds normally.
- **Optional Groq-backed routing:** set `ROUTING_MODE=groq` and a real
  `GROQ_API_KEY` to try a structured, catalog-validated Groq completion
  first (a single JSON `specialty_slug` field only — no chain-of-thought
  is ever requested or shown). Any failure (no key, network error,
  malformed response, or a slug outside the catalog) falls back to
  deterministic matching silently. Off by default; never used in tests.

**Example request** (synthetic data only):

```bash
curl -X POST http://localhost:8000/api/v1/navigate \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["chest pain", "heart palpitations"],
    "duration": {"value": 2, "unit": "days"},
    "location": {"city": "Dallas", "state": "TX"}
  }'
```

**Run navigation/routing tests:**

```bash
uv run pytest tests/test_specialty_routing_service.py tests/test_navigation_api.py
```

### Demo data: a confirmed working example

The NPPES fixture (`backend/tests/fixtures/nppes_sample.csv`) is small and
synthetic, but two rows in it map to specialties in the Phase 1B catalog,
so a search/navigation request against them returns a real result once
the fixture is loaded (see [Run the small fixture
import](#nppes-provider-ingestion-phase-1a) above — idempotent, safe to
re-run):

| Specialty | Location | Matching provider (synthetic) |
|---|---|---|
| `family-medicine` | city=`Springfield`, state=`CA` | "Jane Q Smith, MD, FACP" (individual, primary taxonomy `207Q00000X`) |
| `general-surgery` | city=`Holtsville`, state=`NY` | "Springfield Clinic LLC" (organization, taxonomy `208600000X`) |

Verified directly against both `GET /api/v1/providers/search` and
`POST /api/v1/navigate`:

```bash
curl "http://localhost:8000/api/v1/providers/search?specialty=family-medicine&city=Springfield&state=CA"

curl -X POST http://localhost:8000/api/v1/navigate \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["annual checkup"],
    "duration": {"value": 1, "unit": "days"},
    "location": {"city": "Springfield", "state": "CA"}
  }'
```

The second request deliberately uses `"annual checkup"` rather than
`preferred_specialty` — `"checkup"` is one of `family-medicine`'s
deterministic routing keywords (see
`app/catalog/nucc_specialties.py`), so this also exercises real
keyword-based routing end to end, not just a direct specialty pick.

### Streamlit demo UI

A lightweight demo frontend (`backend/streamlit_app/`) calls
`/api/v1/navigate` and `/api/v1/specialties` only — it contains no
routing, ranking, or search logic of its own, and never hardcodes a
provider result. This is a **demo UI, not the planned production
frontend** (React remains Phase 5, a separate later milestone).

Run (from `backend/`, with the API already running at `localhost:8000`):

```bash
uv run streamlit run streamlit_app/app.py
```

- A permanent warning is always shown: if you believe you may be
  experiencing a medical emergency, don't use this demo — call 911 (U.S.).
- The Backend URL lives in a collapsed **Developer settings** section in
  the sidebar; most users never need to touch it.
- **Preferred specialty** is a dropdown populated live from
  `GET /api/v1/specialties` (falls back to "no specialty selected" if the
  backend is unreachable) — no free-text slug entry.
- **Try demo example** pre-fills the form with the confirmed
  `family-medicine` / Springfield, CA combination above — it only fills in
  inputs; the result shown always comes from a real call to the backend.
- The routing result and the provider-search results are shown in clearly
  separate sections. When routing succeeds but nothing matches, the UI
  says so explicitly ("Specialty routing succeeded, but no matching
  providers are currently loaded for this location.") rather than a bare
  "not found."
- Supports text symptoms, main concern, duration, location, and a
  user-declared emergency checkbox; displays the clarification/emergency/
  media-unavailable/routed/provider-result states and the non-diagnostic
  and emergency disclaimers. It does not persist or log anything itself —
  each submission is a single forwarded request.
- A **Voice intake (optional)** section (Phase 2A) lets you upload an
  audio file and transcribe it via Groq Whisper — see [Voice
  intake](#voice-intake-phase-2a) below for the full workflow and its
  confirmation requirement.

**Run the Streamlit API client tests** (payload construction + the HTTP
calls, against a mock transport — no real network):

```bash
uv run pytest tests/test_streamlit_api_client.py
```

### What's deferred

Real text/vision-model understanding (rather than deterministic keyword
matching), a full LLM-provider abstraction, non-diagnostic visual
description, and model-evaluation fixtures remain future work — see
[docs/architecture.md](docs/architecture.md)'s "Beyond Phase 1D" notes.
Appointment scheduling/booking, authentication, and the production
frontend are separate, later milestones.

## Voice intake (Phase 2A)

`POST /api/v1/voice/transcribe` accepts one multipart audio upload and
returns a raw speech-to-text transcript using Groq's hosted Whisper models.
This endpoint performs transcription only — it never interprets, diagnoses,
or scores urgency, and the returned transcript is not verified medical
information.

- **Supported formats:** mp3, wav, m4a, flac, webm.
- **Size limit:** `VOICE_MAX_UPLOAD_BYTES` (default 10,000,000 bytes / 10 MB).
- **Requires `GROQ_API_KEY`** (see `.env.example`). Returns `503` if not
  configured, or if the Groq call fails/times out — it never fabricates a
  transcript on failure.
- **Model:** `GROQ_STT_MODEL` (default `whisper-large-v3`); timeout via
  `GROQ_STT_TIMEOUT_SECONDS` (default 30s).
- **No remote audio URLs** — the file must be uploaded directly.
- **Nothing is persisted.** Audio and transcripts are never written to the
  database or retained after the request; only safe operational metadata
  (transcription id, model, status, language, byte count, timing) is
  logged — never filenames, audio bytes, transcripts, or symptoms.
- **Speech-to-text only.** No text-to-speech in Phase 2A (deferred, see
  `docs/roadmap.md`'s Phase 4).

**Example request** (using the repo's local `audio.mp3`, if present and not
committed — never stage or commit real or test audio files):

```bash
curl -X POST http://localhost:8000/api/v1/voice/transcribe \
  -F "file=@audio.mp3"
```

**Streamlit confirmation workflow:** upload an audio file, click
**"Transcribe audio"**, then review the editable transcript. The UI always
shows: *"Please review and correct the transcript before continuing.
Speech recognition may contain errors."* The transcript is only included in
the `/api/v1/navigate` request after you check **"I have reviewed this
transcript and confirm it is ready to use"** — an unconfirmed or
unreviewed transcript is never auto-submitted. Selecting a new audio file
always resets any prior transcript and confirmation. Confirmed transcripts
are sent via the existing Phase 1C `voice_input` field alongside (not
replacing) the text symptom/main-concern fields, which remain independently
usable.

**Run voice intake tests** (a fake/mocked provider only — these never call
Groq):

```bash
uv run pytest tests/test_voice_transcription_service.py tests/test_voice_api.py
```

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
