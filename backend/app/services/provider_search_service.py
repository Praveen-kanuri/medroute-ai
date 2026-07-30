"""Orchestrates the provider repository and deterministic ranking, and
applies pagination. No LLM calls anywhere in this module."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.provider_repository import ProviderSearchRow, search_providers
from app.services.provider_ranking import rank_provider_results

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
