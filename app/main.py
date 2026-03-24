from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import configure_services, router
from app.core.config import get_settings
from app.core.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_services(settings)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description=(
        'FastAPI backend for the NeuroScan AI final project. '
        'The service focuses on backend structure, image analysis orchestration, '
        'report generation, and honest model-availability handling.'
    ),
    lifespan=lifespan,
)
app.include_router(router)
