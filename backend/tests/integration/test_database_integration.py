"""Integration tests requiring a real, reachable PostgreSQL instance.

Run with `docker compose up -d postgres` (see README) and a DATABASE_URL that
matches the running container. These tests skip themselves gracefully — rather
than failing — when no reachable database is configured, so the default
`uv run pytest` invocation stays safe to run without Docker.
"""

from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from app.config.settings import get_settings
from app.db.models import HealthCheckRecord
from app.db.session import check_database_connection, dispose_engine, get_engine, get_sessionmaker
from app.main import app

pytestmark = pytest.mark.integration


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
async def _fresh_engine_per_test() -> AsyncGenerator[None, None]:
    # pytest-asyncio gives each test its own event loop; the shared engine's
    # connection pool must not survive across loops, or cleanup crashes.
    yield
    await dispose_engine()


@pytest.mark.asyncio
async def test_select_1_succeeds_against_real_database() -> None:
    assert await check_database_connection() is True


@pytest.mark.asyncio
async def test_alembic_created_tables_exist() -> None:
    engine = get_engine()
    async with engine.connect() as connection:
        table_names = await connection.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
    assert "health_check_records" in table_names
    assert "alembic_version" in table_names


@pytest.mark.asyncio
async def test_orm_persist_and_retrieve_then_rollback() -> None:
    session_factory = get_sessionmaker()

    async with session_factory() as session:
        record = HealthCheckRecord()
        session.add(record)
        await session.flush()
        record_id = record.id

        fetched = await session.get(HealthCheckRecord, record_id)
        assert fetched is not None
        assert fetched.id == record_id

        await session.rollback()

    # A fresh session must not see the rolled-back row: proves transaction
    # isolation between the write above and this read.
    async with session_factory() as verify_session:
        assert await verify_session.get(HealthCheckRecord, record_id) is None


def test_readiness_returns_200_when_database_reachable() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health/readiness")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
