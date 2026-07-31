"""Phase 1D: the end-to-end navigation demo response contract.

Wraps the existing Phase 1C intake response and Phase 1B provider-search
response unchanged — this module adds no new fields to either of them, only
composes them alongside the new routing outcome.
"""

from pydantic import BaseModel

from app.schemas.multimodal_intake import MultimodalIntakeResponse
from app.schemas.provider_search import ProviderSearchResponse


class SpecialtyRoutingOut(BaseModel):
    """Which catalog specialty (if any) applies, and a transparent, non-medical
    explanation of why. Never a diagnosis, never a confidence score."""

    specialty_slug: str | None
    specialty_display_name: str | None
    method: str
    note: str


class NavigationResponse(BaseModel):
    intake: MultimodalIntakeResponse
    media_note: str | None = None
    routing: SpecialtyRoutingOut | None = None
    provider_search: ProviderSearchResponse | None = None
