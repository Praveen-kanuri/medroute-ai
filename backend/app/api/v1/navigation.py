"""Phase 1D: end-to-end navigation demo endpoint.

Composes the existing Phase 1C intake evaluation, Phase 1D specialty
routing, and Phase 1B provider search — it duplicates none of their logic.
Emergency and clarification precedence from Phase 1C are preserved exactly:
routing and provider search only run when intake status is
ready_for_multimodal_processing. Image/video URLs are still never fetched
or analyzed — this endpoint only adds a note when vision input was
supplied, exactly like Phase 1C's own multimodal_status already reports.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.db.session import get_db_session
from app.schemas.multimodal_intake import IntakeStatus, MultimodalIntakeRequest
from app.schemas.navigation import NavigationResponse, SpecialtyRoutingOut
from app.schemas.provider_search import (
    NPPES_DISCLAIMER,
    AppliedFilters,
    PaginationOut,
    ProviderLocationOut,
    ProviderResultOut,
    ProviderSearchResponse,
    SpecialtyOut,
)
from app.services.multimodal_intake_service import evaluate_intake
from app.services.provider_ranking import provider_display_name
from app.services.provider_search_service import (
    DEFAULT_LIMIT,
    ProviderSearchPage,
    search_providers_page,
)
from app.services.specialty_routing_service import route_to_specialty

logger = logging.getLogger(__name__)

router = APIRouter()

MEDIA_UNAVAILABLE_NOTE = (
    "Image/video analysis is not available in this demo. Only text and voice-transcript "
    "content were used for routing."
)


def _build_provider_search_response(
    page: ProviderSearchPage, *, specialty_slug: str
) -> ProviderSearchResponse:
    """Map a provider-search page to the same response shape GET
    /api/v1/providers/search uses. Kept local to avoid coupling this new
    endpoint's response format to unrelated changes in that endpoint."""
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


@router.post(
    "/navigate",
    summary="End-to-end demo: intake -> safety gate -> specialty routing -> provider search",
    description=(
        "Demonstrates the full navigation flow on top of the existing Phase 1C intake "
        "contract. Emergency and clarification precedence are unchanged — routing and "
        "provider search only run when the intake is structurally ready. Specialty "
        "routing is controlled: it only ever selects a specialty already present in the "
        "supported catalog (`GET /api/v1/specialties`), defaults to deterministic "
        "keyword matching with no model call, and never returns a diagnosis, treatment "
        "advice, or urgency judgment. Image/video URLs are still never fetched or "
        "analyzed in this demo."
    ),
)
async def navigate(
    payload: MultimodalIntakeRequest,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> NavigationResponse:
    intake_response = evaluate_intake(payload)

    media_note = (
        MEDIA_UNAVAILABLE_NOTE if intake_response.multimodal_status.vision_received else None
    )

    routing_out: SpecialtyRoutingOut | None = None
    provider_search_out: ProviderSearchResponse | None = None

    if intake_response.status == IntakeStatus.READY_FOR_MULTIMODAL_PROCESSING:
        routing_result = await route_to_specialty(
            preferred_specialty=payload.preferred_specialty,
            symptoms=payload.symptoms,
            main_concern=payload.main_concern,
            settings=settings,
        )
        routing_out = SpecialtyRoutingOut(
            specialty_slug=routing_result.specialty_slug,
            specialty_display_name=routing_result.specialty_display_name,
            method=routing_result.method.value,
            note=routing_result.note,
        )

        if routing_result.specialty_slug is not None:
            page = await search_providers_page(
                session, specialty_slug=routing_result.specialty_slug, limit=DEFAULT_LIMIT
            )
            provider_search_out = _build_provider_search_response(
                page, specialty_slug=routing_result.specialty_slug
            )

    logger.info(
        "navigation_evaluated intake_id=%s intake_status=%s routing_method=%s "
        "specialty_matched=%s provider_result_count=%d",
        intake_response.intake_id,
        intake_response.status.value,
        routing_out.method if routing_out else "not_attempted",
        routing_out.specialty_slug is not None if routing_out else False,
        len(provider_search_out.results) if provider_search_out else 0,
    )

    return NavigationResponse(
        intake=intake_response,
        media_note=media_note,
        routing=routing_out,
        provider_search=provider_search_out,
    )
