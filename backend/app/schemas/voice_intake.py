"""Phase 2A: typed contract for POST /api/v1/voice/transcribe.

The transcript is raw speech-to-text output only — not verified medical
information, and never interpreted here. Once the caller reviews and
confirms it, it flows into the existing Phase 1C `VoiceInput` contract
(app/schemas/multimodal_intake.py) and on to POST /api/v1/navigate.
"""

from enum import StrEnum

from pydantic import BaseModel


class TranscriptionStatus(StrEnum):
    COMPLETED = "completed"


class TranscriptionResponse(BaseModel):
    transcription_id: str
    transcript: str
    language: str | None = None
    model: str
    status: TranscriptionStatus
