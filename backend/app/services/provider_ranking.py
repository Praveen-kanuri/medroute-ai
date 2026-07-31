"""Deterministic, explainable provider-search ranking.

No LLM, no randomness, no medical-quality or appointment-availability
claims — just a stable, testable sort over already-fetched rows. Pure
Python: takes and returns ProviderSearchRow objects, so it is fully
unit-testable without a database or FastAPI.

Ranking order (lower sorts first):
1. Exact specialty match (neutral/no-op when no specialty was requested)
2. Primary taxonomy before non-primary
3. Exact postal-code match (neutral when no postal_code was requested)
4. Exact city/state match (neutral when neither was requested)
5. Active provider before inactive
6. Stable tie-break: display name, then NPI
"""

from collections.abc import Sequence

from app.repositories.provider_repository import ProviderSearchRow


def provider_display_name(row: ProviderSearchRow) -> str:
    """The name shown for a search result: organization name, or a formatted
    person name for individuals."""
    if row.organization_name:
        return row.organization_name
    parts = [row.name_prefix, row.first_name, row.middle_name, row.last_name, row.name_suffix]
    name = " ".join(part for part in parts if part)
    if row.credential:
        name = f"{name}, {row.credential}"
    return name


def _specialty_match_key(row: ProviderSearchRow, specialty_slug: str | None) -> int:
    if specialty_slug is None:
        return 0
    return 0 if row.specialty_slug == specialty_slug else 1


def _postal_match_key(row: ProviderSearchRow, postal_code: str | None) -> int:
    if postal_code is None:
        return 0
    return 0 if row.location_postal_code == postal_code else 1


def _city_state_match_key(row: ProviderSearchRow, city: str | None, state: str | None) -> int:
    if city is None and state is None:
        return 0
    city_ok = city is None or (
        row.location_city is not None and row.location_city.lower() == city.lower()
    )
    state_ok = state is None or row.location_state == state
    return 0 if (city_ok and state_ok) else 1


def rank_provider_results(
    rows: Sequence[ProviderSearchRow],
    *,
    specialty_slug: str | None = None,
    postal_code: str | None = None,
    city: str | None = None,
    state: str | None = None,
) -> list[ProviderSearchRow]:
    """Sort rows deterministically. Never mutates the input sequence."""

    def sort_key(row: ProviderSearchRow) -> tuple[int, int, int, int, int, str, str]:
        return (
            _specialty_match_key(row, specialty_slug),
            0 if row.is_primary_taxonomy else 1,
            _postal_match_key(row, postal_code),
            _city_state_match_key(row, city, state),
            0 if row.is_active else 1,
            provider_display_name(row).lower(),
            row.npi,
        )

    return sorted(rows, key=sort_key)
