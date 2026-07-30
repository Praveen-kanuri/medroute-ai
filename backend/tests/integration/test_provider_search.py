"""Integration tests for deterministic provider search (Phase 1B), requiring
a real reachable PostgreSQL. Skips gracefully (like the other integration
files) when no database is configured/reachable.
"""

import asyncio
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, inspect, select, update

from app.catalog.seed_specialties import seed_specialties
from app.config.settings import get_settings
from app.db.nppes import IngestionRun, Provider, ProviderLocation, ProviderTaxonomy
from app.db.session import check_database_connection, dispose_engine, get_engine, get_sessionmaker
from app.db.specialty import Specialty, SpecialtyTaxonomyMapping
from app.ingestion.nppes_service import ingest_nppes_file
from app.main import app
from app.repositories.provider_repository import search_providers
from app.services.provider_search_service import search_providers_page

pytestmark = pytest.mark.integration

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "nppes_sample.csv"
BACKEND_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
async def _require_reachable_database() -> AsyncGenerator[None, None]:
    settings = get_settings()
    if not settings.database_configured or not await check_database_connection(settings):
        pytest.skip(
            "PostgreSQL is not configured/reachable; skipping integration tests. "
            "Start it with `docker compose up -d postgres` (see README)."
        )
    yield


@pytest.fixture(autouse=True)
async def _clean_state_seed_and_ingest() -> AsyncGenerator[None, None]:
    """Each test starts from a clean, freshly-seeded-and-ingested slate."""
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        await session.execute(delete(ProviderTaxonomy))
        await session.execute(delete(ProviderLocation))
        await session.execute(delete(Provider))
        await session.execute(delete(IngestionRun))
        await session.execute(delete(SpecialtyTaxonomyMapping))
        await session.execute(delete(Specialty))
        await session.commit()

    await seed_specialties()
    await ingest_nppes_file(FIXTURE_PATH, chunk_size=2)
    yield
    await dispose_engine()


@pytest.mark.asyncio
async def test_alembic_created_specialty_tables_exist() -> None:
    engine = get_engine()
    async with engine.connect() as connection:
        table_names = await connection.run_sync(lambda c: inspect(c).get_table_names())
    assert "specialties" in table_names
    assert "specialty_taxonomy_mappings" in table_names


@pytest.mark.asyncio
async def test_seed_specialties_is_idempotent() -> None:
    # Already seeded once by the autouse fixture; seeding again must not duplicate.
    result = await seed_specialties()
    assert result.specialties_inserted == 0
    assert result.specialties_updated == 10
    assert result.mappings_inserted == 0
    assert result.mappings_updated == 10

    session_factory = get_sessionmaker()
    async with session_factory() as session:
        specialty_count = await session.scalar(select(func.count()).select_from(Specialty))
        mapping_count = await session.scalar(
            select(func.count()).select_from(SpecialtyTaxonomyMapping)
        )
    assert specialty_count == 10
    assert mapping_count == 10


@pytest.mark.asyncio
async def test_search_by_specialty_slug_maps_taxonomy_to_specialty() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, specialty_slug="family-medicine")
    assert len(rows) == 1
    assert rows[0].npi == "1000000001"
    assert rows[0].specialty_slug == "family-medicine"
    assert rows[0].specialty_display_name == "Family Medicine"


@pytest.mark.asyncio
async def test_search_by_taxonomy_code() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, taxonomy_code="208600000X")
    assert {r.npi for r in rows} == {"1000000002"}
    assert rows[0].specialty_slug == "general-surgery"
    assert rows[0].is_primary_taxonomy is False  # slot 15, primary switch N in the fixture


@pytest.mark.asyncio
async def test_search_by_state() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, state="NY")
    assert {r.npi for r in rows} == {"1000000002"}


@pytest.mark.asyncio
async def test_search_by_city() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, city="Springfield")
    assert {r.npi for r in rows} == {"1000000001"}


@pytest.mark.asyncio
async def test_search_by_postal_code_preserves_leading_zero() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, postal_code="00501")
    assert {r.npi for r in rows} == {"1000000002"}


@pytest.mark.asyncio
async def test_search_returns_both_individual_and_organization_entity_types() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        individuals = await search_providers(session, entity_type="1")
        organizations = await search_providers(session, entity_type="2")
    assert {r.npi for r in individuals} == {"1000000001"}
    assert {r.npi for r in organizations} == {"1000000002"}


@pytest.mark.asyncio
async def test_inactive_providers_excluded_by_default() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        await session.execute(
            update(Provider).where(Provider.npi == "1000000001").values(is_active=False)
        )
        await session.commit()

        active_only = await search_providers(session, taxonomy_code="207Q00000X")
        including_inactive = await search_providers(
            session, taxonomy_code="207Q00000X", active_only=False
        )
    assert active_only == []
    assert len(including_inactive) == 1


@pytest.mark.asyncio
async def test_no_duplicate_providers_from_joins() -> None:
    # Provider 1000000001 has two taxonomies and two locations — a naive
    # join without the ranking subqueries would duplicate it.
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, city="Springfield")
    npis = [r.npi for r in rows]
    assert len(npis) == len(set(npis))


@pytest.mark.asyncio
async def test_primary_taxonomy_preferred_when_no_specialty_filter() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, city="Springfield")
    row = next(r for r in rows if r.npi == "1000000001")
    # Provider has 207Q00000X (primary) and 208D00000X (non-primary); the
    # ranked-taxonomy window function must select the primary one.
    assert row.taxonomy_code == "207Q00000X"
    assert row.is_primary_taxonomy is True


@pytest.mark.asyncio
async def test_practice_location_preferred_over_mailing() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, specialty_slug="family-medicine")
    assert rows[0].location_address_purpose == "practice"
    assert rows[0].location_address_line_1 == "456 Oak Ave"


@pytest.mark.asyncio
async def test_pagination_limits_and_reports_has_more() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        page_one = await search_providers_page(session, limit=1, offset=0)
        page_two = await search_providers_page(session, limit=1, offset=1)
    # Two searchable providers overall (1000000003 has no taxonomy at all,
    # so it is not searchable — see provider_repository's module docstring).
    assert len(page_one.rows) == 1
    assert page_one.has_more is True
    assert len(page_two.rows) == 1
    assert page_two.has_more is False
    assert page_one.rows[0].npi != page_two.rows[0].npi


@pytest.mark.asyncio
async def test_search_returns_empty_list_when_nothing_matches() -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        rows = await search_providers(session, state="ZZ")
    assert rows == []


@pytest.mark.asyncio
async def test_migration_downgrade_and_reupgrade() -> None:
    downgrade = subprocess.run(
        ["uv", "run", "alembic", "downgrade", "8199fd7b743c"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    assert downgrade.returncode == 0, downgrade.stderr

    engine = get_engine()
    async with engine.connect() as connection:
        tables_after_downgrade = await connection.run_sync(lambda c: inspect(c).get_table_names())
    assert "specialties" not in tables_after_downgrade
    assert "specialty_taxonomy_mappings" not in tables_after_downgrade
    assert "providers" in tables_after_downgrade  # untouched by this migration's downgrade

    upgrade = subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    assert upgrade.returncode == 0, upgrade.stderr

    async with engine.connect() as connection:
        tables_after_upgrade = await connection.run_sync(lambda c: inspect(c).get_table_names())
    assert "specialties" in tables_after_upgrade
    assert "specialty_taxonomy_mappings" in tables_after_upgrade


def _fresh_test_client() -> TestClient:
    """A TestClient whose first DB access starts a brand-new engine bound to
    *this* client's own event loop.

    The autouse async fixture's setup (seeding/ingesting) runs under a
    separate temporary event loop that pytest-asyncio spins up just for
    that fixture, since these tests are plain sync functions. Disposing
    the engine here (a no-op on the already-committed Postgres data, which
    persists independently of any app-side engine object) prevents the
    sync TestClient below from ever reusing a connection pool bound to a
    now-closed foreign loop.
    """
    asyncio.run(dispose_engine())
    return TestClient(app)


def test_specialties_endpoint_returns_active_catalog() -> None:
    with _fresh_test_client() as client:
        response = client.get("/api/v1/specialties")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 10
    assert "family-medicine" in {s["slug"] for s in body}


def test_providers_search_endpoint_returns_disclaimer_and_applied_filters() -> None:
    with _fresh_test_client() as client:
        response = client.get("/api/v1/providers/search", params={"specialty": "family-medicine"})
    assert response.status_code == 200
    body = response.json()
    assert body["filters"]["specialty"] == "family-medicine"
    assert "does not verify" in body["disclaimer"]
    assert len(body["results"]) == 1
    assert body["results"][0]["npi"] == "1000000001"


def test_providers_search_endpoint_returns_200_with_empty_results_for_no_match() -> None:
    with _fresh_test_client() as client:
        response = client.get("/api/v1/providers/search", params={"state": "ZZ"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_providers_search_endpoint_rejects_invalid_state_length() -> None:
    with _fresh_test_client() as client:
        response = client.get("/api/v1/providers/search", params={"state": "California"})
    assert response.status_code == 422


def test_providers_search_endpoint_rejects_invalid_entity_type() -> None:
    with _fresh_test_client() as client:
        response = client.get("/api/v1/providers/search", params={"entity_type": "9"})
    assert response.status_code == 422


def test_providers_search_endpoint_rejects_limit_over_maximum() -> None:
    with _fresh_test_client() as client:
        response = client.get("/api/v1/providers/search", params={"limit": 1000})
    assert response.status_code == 422


def test_providers_search_endpoint_rejects_negative_offset() -> None:
    with _fresh_test_client() as client:
        response = client.get("/api/v1/providers/search", params={"offset": -1})
    assert response.status_code == 422


def test_providers_search_endpoint_never_exposes_database_credentials() -> None:
    with _fresh_test_client() as client:
        response = client.get("/api/v1/providers/search", params={"specialty": "family-medicine"})
    assert "postgresql" not in response.text
    assert "medroute_dev_only" not in response.text
