# Roadmap

## Phase 0 — Repository & Engineering Foundation (complete)

- Backend package skeleton (api, graph, providers, safety, schemas,
  services, tools, config).
- Abstract provider interfaces with fake implementations.
- Core Pydantic domain schemas.
- `/health` and `/api/v1/system/info` endpoints.
- Environment-based configuration.
- Docker, Docker Compose, CI workflow, docs scaffolding.

## Phase 0.2 — PostgreSQL Persistence Foundation (complete)

- Async SQLAlchemy 2.x engine, session factory, and FastAPI session
  dependency; Alembic owns all schema changes (no `metadata.create_all()`
  at runtime).
- Minimal infrastructure-only schema used to validate the persistence
  stack end-to-end — no business-domain tables yet.
- `GET /api/v1/health/readiness` distinguishing process liveness from
  database-backed readiness.
- Docker Compose PostgreSQL service for local development.
- Doctor/specialty/appointment persistence tables are explicitly out of
  scope here; they arrive in Phase 1 once that domain schema is defined.

## Phase 1A — NPPES Provider-Directory Ingestion Foundation (complete)

- Normalized PostgreSQL schema for provider identity, locations, and
  taxonomies (`providers`, `provider_locations`, `provider_taxonomies`),
  plus `ingestion_runs` tracking — built on the Phase 0.2 async
  SQLAlchemy/Alembic foundation.
- NPPES is a public government provider-directory registry (real
  doctor/organization identities, not patient data), so this does not
  conflict with the project's synthetic-patient-data safety rule.
- Chunked/streaming CSV transformation, validation, and idempotent
  PostgreSQL upserts; a CLI import command; validated only against a
  small representative fixture, not the full national dataset.
- Full national NPPES import, Qdrant/vector search, and symptom-to-
  specialty routing are explicitly out of scope here (Phase 1B+).

## Phase 1B — Deterministic Provider Discovery (complete)

- All 15 NPPES taxonomy slots parsed (up from 3 in Phase 1A).
- A small, transparent specialty catalog mapping authoritative NUCC/CMS
  taxonomy codes to MedRoute specialties (`specialties`,
  `specialty_taxonomy_mappings`), seeded via a repeatable command.
- Deterministic (no-LLM) provider search — `GET /api/v1/specialties` and
  `GET /api/v1/providers/search` — with explainable ranking, pagination,
  and a disclaimer that NPPES does not validate licensing, credentials,
  quality, or appointment availability.
- Symptom-to-specialty inference, Qdrant/vector search, and appointment
  booking remain explicitly out of scope.

## Phase 1C — Multimodal Intake Foundation (complete)

- Product direction corrected: MedRoute AI is a multimodal assistant, not
  a text-only symptom checker. This supersedes the earlier "Symptom Intake
  Endpoint" plan for Phase 1C.
- `POST /api/v1/intake/validate` — a stateless, non-diagnostic intake
  contract accepting text (symptoms, main concern), a pre-generated voice
  transcript, and image/video URL references, plus user-declared
  emergency signals and an optional preferred-specialty slug.
- Validates and normalizes input, records which modalities were supplied,
  and evaluates a deterministic state (`emergency` /
  `needs_clarification` / `ready_for_multimodal_processing`) with
  user-declared emergency taking precedence over everything else.
- Does not interpret media, transcribe audio, synthesize speech, call any
  model/LLM/vision provider, infer a specialty, or persist anything —
  those are Phase 1D+ (see docs/architecture.md's Phase 1D handoff notes).

## Phase 1D — Controlled Specialty Routing & Navigation Demo (complete)

- `POST /api/v1/navigate` composes the existing Phase 1C intake
  evaluation, a new deterministic specialty router, and the existing
  Phase 1B provider search into one end-to-end demo endpoint — none of
  their logic is duplicated.
- Emergency and clarification precedence from Phase 1C are unchanged:
  routing and provider search only run once intake is
  `ready_for_multimodal_processing`.
- Specialty routing is controlled — it only ever selects a specialty
  already present in the Phase 1B catalog. Default: deterministic
  keyword matching (no model call). An explicit `preferred_specialty`
  always bypasses matching, after catalog validation. An optional
  Groq-backed structured router (`ROUTING_MODE=groq` + a real API key)
  may propose a slug, but it is always validated against the same
  catalog before use and falls back to the deterministic path on any
  failure; never invoked in tests.
- Never returns a diagnosis, treatment advice, urgency score, or medical
  certainty claim; never infers an emergency autonomously; never fetches
  or interprets image/video URLs — the response clearly states media
  processing is not available in this demo.
- A lightweight Streamlit UI (`backend/streamlit_app/`) is the **current
  demo frontend** — it calls `/api/v1/navigate` only and contains no
  routing/ranking logic of its own. React remains the planned
  **production** frontend (Phase 5); Streamlit does not replace that
  milestone.
- Stateless: no new database table, no Alembic migration.

## Phase 2A — Real Voice Intake with Groq Whisper (complete)

- `POST /api/v1/voice/transcribe` — a stateless endpoint accepting one
  multipart audio upload (mp3, wav, m4a, flac, webm), enforcing a
  conservative configurable size limit, and returning a strictly typed
  transcript response (`transcription_id`, `transcript`, `language`,
  `model`, `status`). No remote audio URLs are accepted.
- A small `SpeechToTextProvider` implementation using Groq's hosted
  Whisper models (`GROQ_STT_MODEL`, default `whisper-large-v3`), isolated
  from the API route behind the same provider-abstraction pattern used for
  every other external integration in this project. Provider failures and
  empty results always raise rather than fabricating a transcript.
- Streamlit UI: audio upload, an explicit "Transcribe audio" action, and a
  mandatory review-and-confirm step (editable transcript + confirmation
  checkbox) before the transcript is forwarded — unconfirmed, un-transcribed,
  or edited-but-unconfirmed audio is never submitted to `/api/v1/navigate`.
  Selecting a new audio file always resets any prior transcript/confirmation.
- The confirmed transcript flows through the existing Phase 1C `voice_input`
  contract unchanged; text intake (symptoms/main concern) remains available
  and independent of voice.
- Speech-to-text only: no diagnosis, treatment, medical certainty, or
  urgency scoring; no autonomous emergency detection from the transcript
  (user-declared emergency signals are unchanged); no image/video
  interpretation; no text-to-speech (delivered in Phase 2B below); no
  audio or transcript persistence anywhere in the stack; only safe
  operational metadata (counts, model, status, timing) is ever logged.

## Phase 2B — LangGraph Conversational Orchestration with Spoken Output (complete)

- `POST /api/v1/converse` — a compiled LangGraph graph (`app/graph/`)
  replaces manual step-by-step orchestration for this endpoint, composing
  every existing service as a controlled tool rather than rewriting any of
  them: `normalize_intake` (Phase 1C `evaluate_intake`) -> `safety_gate`
  (conditional edge: emergency short-circuits straight to
  `response_composition`) -> `clarification` (conditional edge: pauses via
  LangGraph's `interrupt()` when a required field is missing, reporting
  the same typed `missing_fields` Phase 1C already returns, resumes and
  loops back to `normalize_intake` when answered) -> `specialty_routing`
  (Phase 1D, conditional edge on match/no-match) -> `provider_search`
  (Phase 1B) -> `response_composition` -> `text_to_speech`.
- A `thread_id` plus an in-process `InMemorySaver` checkpointer let a
  caller resume a paused clarification turn with a typed answer
  (`{"duration": {...}}` and/or `{"main_concern": "..."}`) via
  `Command(resume=...)` — the confirmed voice transcript and every other
  intake field already in the conversation state are preserved
  automatically; no re-upload or re-transcription is ever needed. Multiple
  missing fields are asked for one at a time, in successive turns.
- Conservative, deterministic duration extraction (`for 3 days`, `for the
  past three days`, `since yesterday`) from a confirmed voice transcript
  can satisfy the "duration" requirement on its own — ambiguous phrasing
  is never guessed at and still falls back to asking the user.
- A concise, non-diagnostic response summarizing the specialty decision,
  the "navigation guidance, not a diagnosis" disclaimer, and whether
  matching providers were found — deterministic templating by default;
  an optional Groq-backed rephrasing (`RESPONSE_MODE=groq`) is validated
  (bounded length, no forbidden medical-claim words) before use and falls
  back to the deterministic text on any failure.
- Optional Deepgram Aura text-to-speech over that same response text —
  configurable model/voice, timeout, and max text length. Best-effort
  only: any missing configuration, oversized text, or provider failure
  returns no audio but always still returns the text response. Audio is
  never persisted; only safe operational metadata is ever logged.
- `POST /api/v1/navigate` (Phase 1D) is unchanged — same contract, same
  tests, still available for a simple one-shot (non-conversational,
  non-spoken) request.
- Streamlit: the Submit/Continue flow now calls `/api/v1/converse`
  instead of `/api/v1/navigate`, carrying the `thread_id` to resume a
  clarification turn, plus a "Generate spoken response" checkbox that
  plays returned audio via `st.audio()`. All prior functional elements
  (text intake, audio upload, transcript review/confirmation, emergency
  declaration, specialty selection, provider results) are unchanged.
- A typed `vision_observations` placeholder already exists in the graph's
  `ConversationState` for Phase 2C — always empty in Phase 2B, never
  populated with a fabricated result.
- Out of scope: autonomous emergency detection from text/transcript
  (emergency remains strictly user-declared), diagnosis, treatment advice,
  urgency scoring, image/video interpretation, chain-of-thought or raw
  graph-state exposure.

## Phase 2C — Controlled Image & Video Understanding (complete)

- `POST /api/v1/media/analyze` — accepts one directly-uploaded image
  (JPEG/PNG/WebP) or short video (MP4/MOV/WebM), never a remote URL or
  filesystem path. Format is determined by sniffing the file signature
  (`app/services/media_validation_service.py`), never by trusting the
  filename extension or client-declared Content-Type. Empty, oversized,
  malformed, or excessive-dimension/duration uploads are rejected with
  typed errors (422/413) before any vision-model call is ever made.
- A new `vision_analysis` node runs first in the LangGraph graph (only
  when validated media was uploaded this turn — otherwise the graph is
  unchanged from Phase 2B), converting sampled still frames into a list
  of controlled `VisionObservation` objects (`app/schemas/vision.py`):
  `observation_type`, optional `body_area`, `visual_description`,
  `visible_attributes`, a coarse `confidence` category (never a numeric
  score), `source_type`, and `frame_timestamp_seconds` (video only) —
  never a diagnosis, disease name, treatment suggestion, urgency score,
  or emergency classification. Schema validation rejects any model output
  that violates this (including the required "not a diagnosis" disclaimer
  phrase, which is scrubbed before the check, exactly like Phase 2B's
  response rephrasing does).
- Vision analysis is optional and best-effort, mirroring every other
  external-provider integration in this project (Groq STT/routing/
  response, Deepgram TTS): a new `vision_mode` setting (default
  `"deterministic"`) means a real Groq API key alone is never enough to
  trigger a live call — `vision_mode=groq` is required too. Any missing
  configuration, timeout, malformed output, or provider failure yields an
  empty observation list plus a safe status note; it never fabricates an
  observation or blocks the rest of the turn.
- Image workflow: Pillow validates structure, corrects EXIF orientation,
  strips metadata (re-encoded with no `exif=` payload), and enforces
  configurable per-side and total-pixel-count limits (a decompression-
  bomb guard) before any model call.
- Video workflow: OpenCV decodes into a temporary file (always removed,
  including on every failure path), validates duration/dimensions/decode
  success, and deterministically samples a bounded, configurable number
  of frames evenly across the video — the raw video is never sent to a
  model. Frame timestamps are preserved; substantially repeated
  observations across frames are deduplicated by description text.
- Validated visual-observation text is folded into the same deterministic
  specialty-routing keyword match as symptoms/main_concern/confirmed voice
  transcript (Phase 1D/2A/2B) — never into emergency detection, which
  remains strictly user-declared. An image/video upload alone (no typed
  text or voice) can now also satisfy intake's "concern" requirement.
- The final response text names that media was processed, a short
  concrete summary of what was visually observed, and that this is a
  visual review only, not a diagnosis — composed the same
  deterministic-by-default/optional-Groq-rephrasing way as Phase 2B.
- Streamlit: an "Image/video intake" section with upload, local preview
  (`st.image`/`st.video`), and an explicit "Analyze media" action that
  runs the same full conversation turn as "Submit" (via
  `POST /api/v1/media/analyze`); a newly selected file always invalidates
  any prior media-analysis result. Displays the returned controlled
  visual observations. All Phase 1D/2A/2B functional elements are
  unchanged.
- Uploaded media, extracted frames, and temporary files are never
  persisted or logged — only safe operational metadata (media kind, frame
  count, status, timing) is ever logged.

## Phase 2 (superseded) — LLM-Backed Routing

- Originally scoped as "real `TextLLMProvider` + LangGraph graph +
  emergency-detection safety subsystem." The LangGraph graph and
  Groq-backed rephrasing/routing paths were delivered in Phase 1D/2B
  instead (reusing the existing deterministic services as controlled
  tools, not a general-purpose `TextLLMProvider` abstraction). The
  remaining, unimplemented piece — a safety subsystem that could
  conservatively surface likely-emergency language in free text for the
  *user's own* confirmation, never as an autonomous override — stays
  unscheduled pending a dedicated safety-design review; emergency status
  remains strictly user-declared until then.

## Phase 3A — Clinical-Context Intake & Deterministic Red-Flag Safety Gate (complete)

- A protocol-agnostic clinical-context schema
  (`app/schemas/clinical_context.py`): `ClinicalFact` (value + confidence
  `explicit`/`derived`/`unknown` + source `text`/`voice`/`vision`/
  `clarification_answer`), `ClinicalContext` (primary concern, onset,
  duration, location, laterality, progression, severity, associated
  features, risk factors, explicit negative findings, and which protocol
  questions remain unanswered), and `ClinicalNavigationSummary`
  (`diagnosis`/`treatment_recommendation` typed as always-`None`). Nothing
  is ever invented — a field the user never addressed stays unset.
- A single, protocol-agnostic graph step, `clinical_intake_agent`
  (`app/graph/nodes.py`), added between the existing `clarification` and
  `specialty_routing_agent` nodes: a complete no-op (falls straight
  through, exactly as before) for any concern matching no known protocol —
  see `app/services/clinical_intake_service.py`'s `ClinicalProtocol`
  registry, currently one entry: unilateral/one-sided leg swelling. For a
  matched protocol, it asks its own ordered questions one at a time
  (self-looping via the same LangGraph `interrupt()`/`Command(resume=...)`
  mechanism `clarification` already uses), skipping any question the
  original message already answered. Typed and voice-originated answers
  both work unchanged: the graph reports the same fixed
  `missing_fields=["clinical_answer"]` shape `clarification` already
  produces, and accepts either a typed `clarification_answer.
  clinical_answer_text` or a spoken `clarification_answer.voice_transcript`
  (`app/schemas/conversation.py`).
- A dedicated, deterministic red-flag safety gate,
  `clinical_red_flag_gate` (`app/graph/nodes.py` +
  `app/safety/red_flag_rules.py`), runs immediately after the leg-swelling
  protocol's red-flag question is answered — always before specialty/
  provider routing. Uses a centralized, versionable phrase catalog and a
  reusable negation-aware phrase classifier
  (`app/services/mention_classification_service.py`) to distinguish
  affirmed ("I have chest pain") from negated ("no chest pain" / "I don't
  have trouble breathing") from unknown ("I'm not sure" — never treated as
  affirmed). No model call anywhere in this path. An affirmed flag reuses
  the existing `is_emergency`/`EMERGENCY_SAFETY_MESSAGE` path unchanged —
  it never identifies a disease and never bypasses the existing
  user-declared emergency precedence (checked first, unconditionally).
- Once a protocol's questions are all answered with no red flag affirmed,
  a bounded `ClinicalNavigationSummary` is produced from a small,
  reviewable per-protocol definition (never RAG, never scraped content):
  a plain-language recap of what was reported, a handful of broad
  non-diagnostic possible-explanation categories with supporting evidence
  drawn only from the user's own answers, a routine/prompt care-level
  recommendation, and specialty candidates drawn only from the existing
  Phase 1B/1D catalog (`app/catalog/nucc_specialties.py`, extended with a
  few leg-swelling-relevant keywords on Internal Medicine — no new
  specialty, no bypassed routing contract). Folded into
  `POST /api/v1/converse`'s existing response text
  (`app/services/response_composition_service.py`) and exposed as a new,
  optional `ConversationResponse.clinical_navigation` field.
- Streamlit: the existing typed follow-up form renders a plain free-text
  answer box for a pending protocol question (`missing_fields=
  ["clinical_answer"]`) instead of the duration/main-concern controls
  meant for Phase 1C's generic fields, and the chat feed renders the
  clinical-navigation summary (possible categories, unknowns, recommended
  care level) once produced. No changes were needed for voice-answered
  protocol questions — the existing voice-clarification-resume path
  already carries a spoken answer through unchanged.
- Out of scope (unchanged from earlier phases): diagnosis, treatment
  recommendations, autonomous emergency detection from ordinary text
  (still strictly user-declared, checked first), hybrid RAG, and a medical
  knowledge graph — the possible-explanation content is a small, reviewable,
  hand-written protocol definition only, explicitly meant to be later
  replaced or grounded by hybrid RAG rather than being it.

## Phase 3 — Booking Simulation (planned)

- Simulated appointment booking flow using deterministic slot data.
- `BookingRequest` → `BookingConfirmation` end-to-end (still simulated).

## Phase 5 — Production Frontend (planned)

- React/TypeScript client consuming the FastAPI backend. This is the
  planned production UI; it is separate from and does not replace the
  Streamlit demo UI introduced in Phase 1D.

## Later (unscheduled)

- Authentication.
- Deployment/hosting.
