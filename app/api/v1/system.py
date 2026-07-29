from fastapi import APIRouter, Depends

from app.config.settings import Settings, get_settings

router = APIRouter()


@router.get("/system/info")
async def system_info(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {
        "app_name": "MedRoute AI",
        "app_env": settings.app_env,
        "log_level": settings.log_level,
    }
