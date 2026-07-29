import pytest

from app.config.settings import Settings
from app.db import session as db_session


@pytest.fixture(autouse=True)
def _reset_engine_state():
    """Isolate the module-level lazy engine/sessionmaker across tests."""
    db_session._engine = None
    db_session._sessionmaker = None
    yield
    db_session._engine = None
    db_session._sessionmaker = None


def test_get_engine_raises_when_database_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        db_session, "get_settings", lambda: Settings(_env_file=None, database_url=None)
    )
    with pytest.raises(RuntimeError):
        db_session.get_engine()


@pytest.mark.asyncio
async def test_check_database_connection_false_when_not_configured() -> None:
    not_configured = Settings(_env_file=None, database_url=None)
    assert await db_session.check_database_connection(not_configured) is False


@pytest.mark.asyncio
async def test_check_database_connection_false_when_unreachable() -> None:
    unreachable = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:1/nonexistent",
    )
    assert await db_session.check_database_connection(unreachable) is False


@pytest.mark.asyncio
async def test_dispose_engine_is_safe_when_never_created() -> None:
    await db_session.dispose_engine()
