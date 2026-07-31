"""Integration tests for NPPES ingestion, requiring a real reachable PostgreSQL.

Skips gracefully (like test_database_integration.py) when no database is
configured/reachable, so `uv run pytest` stays safe without Docker.

Fixture (backend/tests/fixtures/nppes_sample.csv) has 5 raw rows: one
individual (NPI 1000000001), one organization (1000000002), one minimal
individual with all optional fields blank (1000000003), one row with an
invalid NPI (rejected), and a repeat of the first individual's NPI with
updated fields (tests idempotent upsert). So each run processes 4 valid rows
across 3 unique providers: 1 rejected + 3 inserted-or-updated == rows_read.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy import delete, inspect, select

from app.config.settings import get_settings
from app.db.nppes import IngestionRun, IngestionStatus, Provider, ProviderLocation, ProviderTaxonomy
from app.db.session import check_database_connection, dispose_engine, get_engine, get_sessionmaker
from app.ingestion.nppes_service import ingest_nppes_file

pytestmark = pytest.mark.integration

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "nppes_sample.csv"

# chunk_size=2 deliberately puts the fixture's repeated-NPI rows (1st and
# 5th of 5) in different chunks/transactions, so the second occurrence
# exercises a genuine cross-chunk database UPDATE rather than being silently
# collapsed by in-memory same-chunk de-duplication before ever reaching the
# database.
_TEST_CHUNK_SIZE = 2


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
async def _clean_nppes_tables() -> AsyncGenerator[None, None]:
    """Each test starts from an empty slate for the four Phase 1A tables.

    The ingestion service manages its own per-chunk commits, so a rollback-
    only transaction wrapper (as used for the simple Phase 0.2 tests) does
    not apply here; explicit cleanup between tests is the isolation strategy.
    """
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        await session.execute(delete(ProviderTaxonomy))
        await session.execute(delete(ProviderLocation))
        await session.execute(delete(Provider))
        await session.execute(delete(IngestionRun))
        await session.commit()
    yield
    await dispose_engine()


@pytest.mark.asyncio
async def test_alembic_created_nppes_tables_exist() -> None:
    engine = get_engine()
    async with engine.connect() as connection:
        table_names = await connection.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
    for table in ("providers", "provider_locations", "provider_taxonomies", "ingestion_runs"):
        assert table in table_names


@pytest.mark.asyncio
async def test_ingest_fixture_persists_providers_locations_taxonomies() -> None:
    result = await ingest_nppes_file(FIXTURE_PATH, chunk_size=_TEST_CHUNK_SIZE)

    assert result.rows_read == 5
    assert result.rows_rejected == 1  # the invalid-NPI row
    assert result.rows_inserted == 3  # three unique providers, first time seen
    assert result.rows_updated == 1  # the repeated row for NPI 1000000001
    assert result.status in (IngestionStatus.COMPLETED, IngestionStatus.PARTIALLY_COMPLETED)

    session_factory = get_sessionmaker()
    async with session_factory() as session:
        providers = (await session.scalars(select(Provider))).all()
        assert len(providers) == 3

        individual = (
            await session.scalars(select(Provider).where(Provider.npi == "1000000001"))
        ).one()
        assert individual.entity_type_code == "1"
        assert individual.last_name == "Smith"

        organization = (
            await session.scalars(select(Provider).where(Provider.npi == "1000000002"))
        ).one()
        assert organization.entity_type_code == "2"
        assert organization.organization_name == "Springfield Clinic LLC"

        locations = (
            await session.scalars(
                select(ProviderLocation).where(ProviderLocation.provider_id == individual.id)
            )
        ).all()
        assert {loc.address_purpose for loc in locations} == {"mailing", "practice"}

        org_locations = (
            await session.scalars(
                select(ProviderLocation).where(ProviderLocation.provider_id == organization.id)
            )
        ).all()
        assert len(org_locations) == 1
        assert org_locations[0].postal_code == "00501"  # leading zero preserved

        taxonomies = (
            await session.scalars(
                select(ProviderTaxonomy).where(ProviderTaxonomy.provider_id == individual.id)
            )
        ).all()
        assert {t.taxonomy_code for t in taxonomies} == {"207Q00000X", "208D00000X"}


@pytest.mark.asyncio
async def test_ingest_is_idempotent_on_rerun() -> None:
    await ingest_nppes_file(FIXTURE_PATH, chunk_size=_TEST_CHUNK_SIZE)
    result_second = await ingest_nppes_file(FIXTURE_PATH, chunk_size=_TEST_CHUNK_SIZE)

    assert result_second.rows_inserted == 0
    assert result_second.rows_updated == 4  # every valid row now updates an existing provider

    session_factory = get_sessionmaker()
    async with session_factory() as session:
        providers = (await session.scalars(select(Provider))).all()
        assert len(providers) == 3  # no duplicates created by the rerun

        individual = (
            await session.scalars(select(Provider).where(Provider.npi == "1000000001"))
        ).one()
        locations = (
            await session.scalars(
                select(ProviderLocation).where(ProviderLocation.provider_id == individual.id)
            )
        ).all()
        assert len(locations) == 2  # still exactly mailing + practice, no duplicates

        taxonomies = (
            await session.scalars(
                select(ProviderTaxonomy).where(ProviderTaxonomy.provider_id == individual.id)
            )
        ).all()
        assert len(taxonomies) == 2  # no duplicate taxonomy rows


@pytest.mark.asyncio
async def test_ingest_updates_existing_provider_fields() -> None:
    await ingest_nppes_file(FIXTURE_PATH, chunk_size=_TEST_CHUNK_SIZE)

    session_factory = get_sessionmaker()
    async with session_factory() as session:
        individual = (
            await session.scalars(select(Provider).where(Provider.npi == "1000000001"))
        ).one()
        # Row 5 (same NPI, later in the file) has the updated values — the
        # last occurrence in the file wins.
        assert individual.credential == "MD, FACP"

        mailing = (
            await session.scalars(
                select(ProviderLocation).where(
                    ProviderLocation.provider_id == individual.id,
                    ProviderLocation.address_purpose == "mailing",
                )
            )
        ).one()
        assert mailing.telephone_number == "5551110000"

        taxonomy = (
            await session.scalars(
                select(ProviderTaxonomy).where(
                    ProviderTaxonomy.provider_id == individual.id,
                    ProviderTaxonomy.taxonomy_code == "207Q00000X",
                )
            )
        ).one()
        assert taxonomy.license_number == "99999"


@pytest.mark.asyncio
async def test_ingestion_run_tracks_completion_counts() -> None:
    result = await ingest_nppes_file(FIXTURE_PATH, chunk_size=_TEST_CHUNK_SIZE)

    session_factory = get_sessionmaker()
    async with session_factory() as session:
        run = await session.get(IngestionRun, result.run_id)
        assert run is not None
        assert run.source_name == "nppes"
        assert run.source_filename == "nppes_sample.csv"  # basename only
        assert run.rows_read == 5
        assert run.rows_rejected == 1
        assert run.rows_inserted == 3
        assert run.rows_updated == 1
        assert run.status in (
            IngestionStatus.COMPLETED.value,
            IngestionStatus.PARTIALLY_COMPLETED.value,
        )
        assert run.completed_at is not None


@pytest.mark.asyncio
async def test_ingestion_run_marks_failed_status_when_file_missing() -> None:
    missing_path = FIXTURE_PATH.parent / "does_not_exist.csv"
    result = await ingest_nppes_file(missing_path)
    assert result.status == IngestionStatus.FAILED

    session_factory = get_sessionmaker()
    async with session_factory() as session:
        run = await session.get(IngestionRun, result.run_id)
        assert run is not None
        assert run.status == IngestionStatus.FAILED.value
        assert run.source_filename == "does_not_exist.csv"
        # Never a full/private path in the persisted error summary.
        assert str(missing_path.parent) not in (run.error_summary or "")
