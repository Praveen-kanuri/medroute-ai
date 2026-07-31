from fastapi import APIRouter, Depends, Response, status

from app.config.settings import Settings, get_settings
from app.db.session import check_database_connection

router = APIRouter()


@router.get("/health/readiness")
async def readiness(
    response: Response, settings: Settings = Depends(get_settings)
) -> dict[str, str]:
    """Report whether this instance can serve database-backed requests.

    Distinct from GET /health (process liveness): a database outage must not
    fail liveness, only readiness.
    """
    if not settings.database_configured or not await check_database_connection(settings):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable"}
    return {"status": "ready"}
