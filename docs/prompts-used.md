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

### Phase 1B — Deterministic provider discovery and NPPES compatibility hardening

> Implement MedRoute AI Phase 1B: deterministic provider discovery and
> NPPES compatibility hardening. Expand NPPES taxonomy parsing from 3 to
> all 15 official slots. Add a controlled specialty catalog mapped to
> authoritative taxonomy codes. Implement asynchronous provider-search
> repository and service layers. Expose deterministic specialty and
> provider-search APIs. Rank results deterministically without an LLM.
> Validate ingestion using fixtures and, when safely available, a bounded
> sample from an official CMS weekly incremental NPPES file. Do not
> implement full national ingestion, automated scheduling, Qdrant,
> embeddings, symptom-to-specialty inference, LLM/Groq/LangChain/LangGraph
> integration, chatbot behavior, appointment booking, voice/multimodal,
> authentication, or frontend work.
>
> This prompt again surfaced a roadmap gap: `docs/roadmap.md`'s existing
> Phase 1B bundled two deliverables — doctor search and a `SymptomIntake`
> intake endpoint — but this task's scope only covered the first. Unlike
> the two prior prompts, this one explicitly said not to silently rewrite
> the roadmap again, so the gap was reported and the user was asked before
> touching it; the resolution was to split Phase 1B (this task, provider
> discovery only) from a new Phase 1C (the intake endpoint, deferred).
>
> **NUCC taxonomy source:** National Uniform Claim Committee (NUCC) Health
> Care Provider Taxonomy Code Set, version 26.1 (effective 2026-07-01).
> Ten codes were verified via
> [nucc.org's v20.0 PDF](https://www.nucc.org/images/stories/PDF/Taxonomy_20_0.pdf),
> [findacode.com's taxonomy list](https://www.findacode.com/tools/taxonomy-codes.html)
> (citing "NUCC Provider Taxonomy version 26.1 7/1/2026"), and
> [npiprofile.com](https://npiprofile.com/taxonomy/code/208600000X) for
> General Surgery and Diagnostic Radiology specifically. Accessed
> 2026-07-29. See `backend/app/catalog/nucc_specialties.py` for the full
> cited list.
>
> **Real-file compatibility pilot:** the official CMS NPI Files page
> (`https://download.cms.gov/nppes/NPI_Files.html`) was reachable, so the
> latest weekly incremental NPPES V.2 file was downloaded
> (`NPPES_Data_Dissemination_072026_072626_Weekly_V2.zip`, week
> 2026-07-20–2026-07-26, ~6.9 MB) into the session's temporary scratchpad
> directory — never the repository. The ZIP was validated
> (`zipfile.is_zipfile`, `testzip()`, and a path-safety check rejecting any
> member path containing `..` or an absolute path) before extracting only
> the main `npidata_pfile_*.csv` member. A bounded sample (the header plus
> the first 10,000 data rows) was ingested into a separate, throwaway
> PostgreSQL database (`medroute_pilot_tmp`, created and dropped on the
> same local Postgres server — never the app's own `medroute` dev
> database) via the existing chunked ingestion service. Result: 10,000 rows
> read, 9,950 inserted, 0 updated, 50 rejected (all for a blank Entity Type
> Code — consistent with deactivated/legacy NPPES records, handled safely
> rather than crashing the run), runtime ~196 seconds, ~26.7 MB peak
> Python-tracked memory (`tracemalloc`). The pilot database was dropped and
> the downloaded ZIP/CSV/sample were deleted from the scratchpad afterward;
> no real provider data was persisted anywhere durable or committed.
>
> **Real compatibility bug the pilot caught:** the actual current NPPES
> V.2 file header uses `"Provider Sex Code"`, not `"Provider Gender Code"`
> as Phase 1A had assumed (verified by diffing the real file's header
> against every mapped column name). Fixed in
> `app/ingestion/nppes_mapping.py`; the synthetic fixture's header was
> updated to match, and all existing tests continued to pass unchanged
> since they reference the mapping module's named constant, not the
> literal string.

### Phase 1C — Multimodal intake foundation

> Implement MedRoute AI Phase 1C: multimodal medical-intake foundation.
> Create a stateless backend intake API that accepts text, voice-
> transcript, image-reference, and video-reference inputs; validates and
> normalizes them; records which modalities were supplied; handles
> explicitly user-declared emergency concerns; produces a typed, routing-
> ready contract for Phase 1D; and does not process, interpret, download,
> or analyze media, diagnose conditions, recommend treatment, or
> autonomously assess urgency. The product owner explicitly changed
> direction: MedRoute AI is multimodal, not a text-only symptom checker.
>
> This prompt anticipated the resulting documentation conflict itself:
> `docs/roadmap.md`'s Phase 1C still described the old "Symptom Intake
> Endpoint" (`SymptomIntake`, no routing) plan. Unlike the two prior
> conflicts, this task explicitly said the conflict was already resolved
> by product direction and instructed updating the milestone documentation
> and continuing automatically without asking again — so the roadmap was
> corrected (Phase 1C redefined as the multimodal intake foundation) before
> any code was written, with no further confirmation requested.
>
> Request contract (`app/schemas/multimodal_intake.py`, all models
> `extra="forbid"`): `symptoms` (max 10, trimmed/collapsed whitespace,
> blank entries rejected, case-insensitive dedup preserving order),
> `main_concern`, `duration` (positive bounded int + closed unit enum),
> `location` (state/country uppercased, postal code never coerced to a
> number), `preferred_specialty` (slug format-validated only — no database
> lookup, keeping intake decoupled from the Phase 1B provider-search
> database), `emergency_concern`/`emergency_signals` (user-declared,
> closed enum, deduplicated), `voice_input` (pre-generated transcript +
> conservative language-tag validation — no STT), and `vision_inputs`
> (image/video URL references, HTTPS-only, credentials/localhost/private-
> IP-literal/fragment/query-string rejected, deduplicated, never fetched —
> documented explicitly as contract hardening, not a complete SSRF
> defense). Pure service (`multimodal_intake_service.py`) evaluates
> `emergency` / `needs_clarification` / `ready_for_multimodal_processing`
> with emergency always taking precedence. Endpoint
> (`api/v1/intake.py`) logs only safe operational metadata — counts,
> booleans, the generated intake_id — never symptom/transcript/URL
> content; log-capture tests with unique synthetic markers prove this.
>
> No database migration was created (Alembic head stayed at
> `7b8a34b61996`); 106 new tests added (64 schema, 22 service, 20 HTTP),
> for 236 total. One test-design issue was caught and fixed: patching the
> raw `socket.socket` constructor to prove "no network calls" broke
> Windows' own asyncio event-loop bootstrap (it uses raw sockets
> internally for an unrelated self-pipe), so the check was narrowed to
> `socket.create_connection` — the higher-level primitive HTTP clients
> actually use to reach a remote host.

### Phase 1D — controlled specialty routing, navigation demo, Streamlit UI

> Implement Phase 1D controlled specialty routing (consuming the existing
> validated Phase 1C intake, preserving emergency/clarification
> precedence, routing only to specialties already in the curated
> catalog, model output strictly structured and validated, never a
> diagnosis/treatment/urgency score, no autonomous emergency inference,
> no chain-of-thought exposure, deterministic provider discovery kept
> separate from AI routing, media not fetched/interpreted yet), one
> end-to-end navigation API (intake → safety/clarification gate →
> specialty routing → provider search; an explicitly selected specialty
> bypasses AI selection after catalog validation; deterministic local
> routing by default, Groq optional via configuration, never called in
> tests), a lightweight Streamlit portfolio UI inside this repository
> calling the navigation API only (no duplicated backend logic, no
> React/auth/booking/STT/TTS/vision, no logging of sensitive intake
> content), and tests for all of it. No database migration unless
> genuinely required — this milestone stays stateless.
>
> Specialty routing (`app/services/specialty_routing_service.py`) is
> deterministic keyword-overlap matching by default, using a new
> `keywords` tuple added to each `SpecialtySeed`
> (`app/catalog/nucc_specialties.py`) — Python-only, no schema change, no
> migration. An explicit `preferred_specialty` bypasses matching entirely
> once validated against the catalog. The optional Groq path
> (`ROUTING_MODE=groq` + a real key) requests a single
> `{"specialty_slug": ...}` JSON field only, never free-form reasoning,
> and is re-validated against the catalog before use; any failure falls
> back to deterministic silently. `POST /api/v1/navigate`
> (`app/api/v1/navigation.py`) composes `evaluate_intake()` (Phase 1C),
> `route_to_specialty()`, and `search_providers_page()` (Phase 1B)
> without duplicating any of their logic; routing/search only execute
> when intake status is `ready_for_multimodal_processing`; a `media_note`
> is added whenever vision input was supplied, since media is still
> never fetched or analyzed.
>
> The Streamlit demo (`backend/streamlit_app/`) is a separate `app.py` +
> `api_client.py` pair — `api_client.py` has no `streamlit` import at all
> so its payload-building and HTTP-call logic is unit-testable in
> isolation, using `httpx.MockTransport` (never a real socket).
>
> This prompt surfaced no new roadmap conflict (docs/roadmap.md's earlier
> Phase 1C entry already pointed forward to "Phase 1D adds actual
> model-based interpretation," and this milestone's scope — controlled,
> catalog-bound routing, not free-form model interpretation — fit within
> that framing), so the roadmap/CLAUDE.md/PROJECT_WIKI.md were updated to
> mark Phase 1D current (and Phase 0.2/1C complete, which had been left
> stale) without needing to ask again.
>
> No Alembic migration was created (head stayed at `7b8a34b61996`); 42
> new tests added (11 routing service — including three that monkeypatch
> `groq.AsyncGroq` to prove the fallback path without any real network
> call, 16 navigation HTTP, 15 Streamlit API client), for 278 total.

### Phase 1D demo polish — Streamlit UI, confirmed demo data

> Polish the Phase 1D Streamlit demo: replace the free-text preferred-
> specialty field with a dropdown populated from the catalog; move the
> Backend URL into a collapsed "Developer settings" section; add a
> permanent visible emergency warning; replace "No matching providers
> found" with clearer wording distinguishing successful routing from an
> empty result; visually separate the routing result from provider-search
> results; preserve the non-diagnostic disclaimer. Separately, inspect the
> existing NPPES fixture and local database data (no external downloads,
> no invented provider identities) to identify and document one
> specialty/location combination that actually returns a result, loading
> the fixture via the existing ingestion command if the database was
> empty, and add a "Try demo example" button only if it can safely reuse
> that existing fixture data — never hardcoding a provider result in the
> UI itself.
>
> The database already had the Phase 1A/1B fixture and specialty catalog
> loaded from earlier manual testing, so no re-ingestion was needed this
> time — confirmed live against both `GET /api/v1/providers/search` and
> `POST /api/v1/navigate`. Two fixture rows map to catalog specialties:
> `family-medicine` in Springfield, CA (an individual provider, primary
> taxonomy) and `general-surgery` in Holtsville, NY (an organization,
> non-primary taxonomy). The demo button uses the first combination via
> `"annual checkup"` (a real `family-medicine` routing keyword) rather
> than a direct specialty pick, so it also exercises genuine keyword-based
> routing end to end, not just a bypass. Added `list_specialties()` to
> `streamlit_app/api_client.py` (same `httpx.MockTransport`-based test
> pattern as `call_navigate()` — no real network in tests) and used
> Streamlit's `key=`-plus-`session_state` pattern in an `on_click`
> callback so the demo button can pre-fill widget values without
> triggering Streamlit's "widget already instantiated" state-mutation
> error. Manually verified end-to-end by running both the real FastAPI
> server and `streamlit run` locally and confirming no exceptions and a
> successful live call to `/api/v1/specialties`.
>
> No backend/API/service code changed — only `streamlit_app/{app.py,
> api_client.py}`, one new test file addition, and documentation. No
> database migration, no new dependency. Total tests: 280 (278 + 2 new
> `list_specialties` client tests).
