from fastapi import APIRouter

from app.api.v1 import health, readiness, system

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(system.router, prefix="/api/v1")
api_router.include_router(readiness.router, prefix="/api/v1")
