"""Orchestrates the provider repository and deterministic ranking, and
applies pagination. No LLM calls anywhere in this module."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.provider_repository import ProviderSearchRow, search_providers
from app.schemas.provider_search import (
    NPPES_DISCLAIMER,
    AppliedFilters,
    PaginationOut,
    ProviderLocationOut,
    ProviderResultOut,
    ProviderSearchResponse,
    SpecialtyOut,
)
from app.services.provider_ranking import provider_display_name, rank_provider_results

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


@dataclass
class ProviderSearchPage:
    rows: list[ProviderSearchRow]
    limit: int
    offset: int
    has_more: bool


def clamp_limit(limit: int) -> int:
    """Enforce a safe maximum page size, independent of the API layer's own validation."""
    return min(max(limit, 1), MAX_LIMIT)


def clamp_offset(offset: int) -> int:
    return max(offset, 0)


async def search_providers_page(
    session: AsyncSession,
    *,
    specialty_slug: str | None = None,
    taxonomy_code: str | None = None,
    state: str | None = None,
    city: str | None = None,
    postal_code: str | None = None,
    entity_type: str | None = None,
    name: str | None = None,
    active_only: bool = True,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> ProviderSearchPage:
    """Fetch, rank, and paginate matching providers. Never raises for zero results."""
    limit = clamp_limit(limit)
    offset = clamp_offset(offset)

    rows = await search_providers(
        session,
        specialty_slug=specialty_slug,
        taxonomy_code=taxonomy_code,
        state=state,
        city=city,
        postal_code=postal_code,
        entity_type=entity_type,
        name=name,
        active_only=active_only,
    )
    ranked = rank_provider_results(
        rows, specialty_slug=specialty_slug, postal_code=postal_code, city=city, state=state
    )
    page = ranked[offset : offset + limit]
    has_more = len(ranked) > offset + limit
    return ProviderSearchPage(rows=page, limit=limit, offset=offset, has_more=has_more)


def build_provider_search_response(
    page: ProviderSearchPage, *, specialty_slug: str
) -> ProviderSearchResponse:
    """Map a provider-search page to the same response shape GET
    /api/v1/providers/search uses. Shared by POST /api/v1/navigate and the
    Phase 2B LangGraph provider_search node so neither duplicates this
    mapping."""
    results = [
        ProviderResultOut(
            npi=row.npi,
            display_name=provider_display_name(row),
            entity_type_code=row.entity_type_code,
            specialty=(
                SpecialtyOut(
                    slug=row.specialty_slug,
                    display_name=row.specialty_display_name or "",
                    description=row.specialty_description,
                )
                if row.specialty_slug is not None
                else None
            ),
            taxonomy_code=row.taxonomy_code,
            is_primary_taxonomy=row.is_primary_taxonomy,
            practice_location=(
                ProviderLocationOut(
                    address_line_1=row.location_address_line_1,
                    address_line_2=row.location_address_line_2,
                    city=row.location_city,
                    state=row.location_state,
                    postal_code=row.location_postal_code,
                    telephone_number=row.location_telephone_number,
                )
                if row.location_address_purpose is not None
                else None
            ),
            last_update_date=row.last_update_date,
        )
        for row in page.rows
    ]
    return ProviderSearchResponse(
        filters=AppliedFilters(specialty=specialty_slug),
        pagination=PaginationOut(limit=page.limit, offset=page.offset, has_more=page.has_more),
        results=results,
        disclaimer=NPPES_DISCLAIMER,
    )
