from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import configure_logging


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

configure_logging()


# ------------------------------------------------------------
# Application
# ------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Trustlens backend for explainable "
        "video deepfake forensic analysis."
    ),
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------
#
# This allows our React frontend to communicate with
# the FastAPI backend during development.
#

app.add_middleware(
    CORSMiddleware,

    allow_origins=settings.CORS_ORIGINS,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ------------------------------------------------------------
# API routes
# ------------------------------------------------------------

app.include_router(
    api_router,
    prefix="/api",
)


# ------------------------------------------------------------
# Startup
# ------------------------------------------------------------

@app.on_event("startup")
def startup_event() -> None:
    """
    Initialize directories and database
    when the backend starts.
    """

    settings.ensure_directories()

    init_db()


# ------------------------------------------------------------
# Root health endpoint
# ------------------------------------------------------------

@app.get(
    "/health",
    tags=["system"],
)
def health() -> dict:
    """
    Basic backend health check.
    """

    return {
        "status": "ok",
        "service": "trustlens-backend",
        "version": settings.APP_VERSION,
    }