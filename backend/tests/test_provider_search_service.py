"""Unit tests for pagination bounds and response serialization.

Pure Python/Pydantic — no database, no FastAPI app, no model API.
"""

from app.schemas.provider_search import (
    NPPES_DISCLAIMER,
    AppliedFilters,
    PaginationOut,
    ProviderResultOut,
    ProviderSearchResponse,
    SpecialtyOut,
)
from app.services.provider_search_service import MAX_LIMIT, clamp_limit, clamp_offset


def test_clamp_limit_within_bounds_is_unchanged() -> None:
    assert clamp_limit(20) == 20


def test_clamp_limit_rejects_too_high() -> None:
    assert clamp_limit(10_000) == MAX_LIMIT


def test_clamp_limit_rejects_zero_or_negative() -> None:
    assert clamp_limit(0) == 1
    assert clamp_limit(-5) == 1


def test_clamp_offset_rejects_negative() -> None:
    assert clamp_offset(-1) == 0


def test_clamp_offset_allows_zero_and_positive() -> None:
    assert clamp_offset(0) == 0
    assert clamp_offset(100) == 100


def test_response_serializes_with_zero_results() -> None:
    response = ProviderSearchResponse(
        filters=AppliedFilters(state="ZZ"),
        pagination=PaginationOut(limit=20, offset=0, has_more=False),
        results=[],
    )
    dumped = response.model_dump()
    assert dumped["results"] == []
    assert dumped["disclaimer"] == NPPES_DISCLAIMER


def test_response_serializes_result_with_no_matched_specialty() -> None:
    result = ProviderResultOut(
        npi="1000000003",
        display_name="Sam Lee",
        entity_type_code="1",
        specialty=None,
        taxonomy_code="999999999X",
        is_primary_taxonomy=False,
        practice_location=None,
        last_update_date=None,
    )
    response = ProviderSearchResponse(
        filters=AppliedFilters(),
        pagination=PaginationOut(limit=20, offset=0, has_more=False),
        results=[result],
    )
    dumped = response.model_dump()
    assert dumped["results"][0]["specialty"] is None
    assert dumped["results"][0]["practice_location"] is None


def test_response_always_includes_disclaimer() -> None:
    response = ProviderSearchResponse(
        filters=AppliedFilters(),
        pagination=PaginationOut(limit=20, offset=0, has_more=False),
        results=[],
    )
    assert "does not verify" in response.disclaimer


def test_specialty_out_serializes_without_description() -> None:
    specialty = SpecialtyOut(slug="cardiology", display_name="Cardiology")
    assert specialty.model_dump()["description"] is None
