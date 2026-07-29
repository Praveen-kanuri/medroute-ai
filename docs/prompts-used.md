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
