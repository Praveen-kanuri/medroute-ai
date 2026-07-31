from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.db.session import dispose_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(title="MedRoute AI", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()
