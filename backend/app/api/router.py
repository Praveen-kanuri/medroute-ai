from fastapi import APIRouter

from app.api.v1 import health, providers, readiness, specialties, system

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(system.router, prefix="/api/v1")
api_router.include_router(readiness.router, prefix="/api/v1")
api_router.include_router(specialties.router, prefix="/api/v1")
api_router.include_router(providers.router, prefix="/api/v1")
