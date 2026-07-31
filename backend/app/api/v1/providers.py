from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.schemas.provider_search import (
    NPPES_DISCLAIMER,
    AppliedFilters,
    PaginationOut,
    ProviderLocationOut,
    ProviderResultOut,
    ProviderSearchResponse,
    SpecialtyOut,
)
from app.services.provider_ranking import provider_display_name
from app.services.provider_search_service import DEFAULT_LIMIT, MAX_LIMIT, search_providers_page

router = APIRouter()


@router.get("/providers/search")
async def search_providers_endpoint(
    specialty: str | None = Query(default=None, description="Specialty slug, e.g. cardiology"),
    taxonomy_code: str | None = Query(default=None, description="Exact NUCC taxonomy code"),
    state: str | None = Query(default=None, min_length=2, max_length=2),
    city: str | None = Query(default=None),
    postal_code: str | None = Query(default=None),
    entity_type: str | None = Query(default=None, pattern="^[12]$"),
    name: str | None = Query(default=None, description="Substring match on provider/org name"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> ProviderSearchResponse:
    """Deterministic (no-LLM) provider search over the ingested NPPES data.

    Returns zero results, not an error, when nothing matches. NPPES
    inclusion does not verify licensing, credentials, quality, or
    appointment availability — see `disclaimer` in the response.
    """
    normalized_state = state.upper() if state else None
    try:
        page = await search_providers_page(
            session,
            specialty_slug=specialty,
            taxonomy_code=taxonomy_code,
            state=normalized_state,
            city=city,
            postal_code=postal_code,
            entity_type=entity_type,
            name=name,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        # Never leak raw database errors, connection details, or credentials.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider search is temporarily unavailable.",
        ) from exc

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
        filters=AppliedFilters(
            specialty=specialty,
            taxonomy_code=taxonomy_code,
            state=normalized_state,
            city=city,
            postal_code=postal_code,
            entity_type=entity_type,
            name=name,
        ),
        pagination=PaginationOut(limit=page.limit, offset=page.offset, has_more=page.has_more),
        results=results,
        disclaimer=NPPES_DISCLAIMER,
    )
