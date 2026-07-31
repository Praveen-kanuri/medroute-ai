from datetime import date

from pydantic import BaseModel

NPPES_DISCLAIMER = (
    "Sourced from NPPES (the National Plan and Provider Enumeration System). "
    "Inclusion in NPPES does not verify licensing, credentials, quality of "
    "care, or current appointment availability."
)


class SpecialtyOut(BaseModel):
    """A specialty from MedRoute AI's small, curated catalog."""

    slug: str
    display_name: str
    description: str | None = None


class ProviderLocationOut(BaseModel):
    """A provider's displayed location (practice address preferred over mailing)."""

    address_line_1: str | None = None
    address_line_2: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    telephone_number: str | None = None


class ProviderResultOut(BaseModel):
    """One deterministic, non-ranked-by-quality provider search result."""

    npi: str
    display_name: str
    entity_type_code: str
    specialty: SpecialtyOut | None = None
    taxonomy_code: str
    is_primary_taxonomy: bool
    practice_location: ProviderLocationOut | None = None
    last_update_date: date | None = None


class AppliedFilters(BaseModel):
    """Echoes back the filters actually applied to this search."""

    specialty: str | None = None
    taxonomy_code: str | None = None
    state: str | None = None
    city: str | None = None
    postal_code: str | None = None
    entity_type: str | None = None
    name: str | None = None


class PaginationOut(BaseModel):
    limit: int
    offset: int
    has_more: bool


class ProviderSearchResponse(BaseModel):
    filters: AppliedFilters
    pagination: PaginationOut
    results: list[ProviderResultOut]
    disclaimer: str = NPPES_DISCLAIMER
