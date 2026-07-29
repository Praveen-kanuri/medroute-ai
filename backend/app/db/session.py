from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import Settings, get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use.

    Raises:
        RuntimeError: if DATABASE_URL is not configured.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        if not settings.database_configured:
            raise RuntimeError("DATABASE_URL is not configured.")
        assert settings.database_url is not None
        _engine = create_async_engine(
            settings.database_url.get_secret_value(),
            echo=settings.database_echo,
            pool_pre_ping=True,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory, creating it on first use."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a transactional AsyncSession per request."""
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the engine's connection pool. Call during application shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


async def check_database_connection(settings: Settings | None = None) -> bool:
    """Execute SELECT 1 against PostgreSQL. Returns False on any failure.

    Uses a short-lived engine scoped to this call rather than the shared
    connection-pool engine, so a readiness probe never reports a stale result
    from a pool established against different settings, and never keeps a
    pool alive purely for health checks. Never raises, and never includes
    connection details in its result, so it is safe to call directly from a
    readiness endpoint.
    """
    settings = settings or get_settings()
    if not settings.database_configured:
        return False
    assert settings.database_url is not None
    probe_engine = create_async_engine(settings.database_url.get_secret_value(), pool_pre_ping=True)
    try:
        async with probe_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except (SQLAlchemyError, OSError):
        return False
    finally:
        await probe_engine.dispose()
