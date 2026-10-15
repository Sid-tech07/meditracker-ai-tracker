"""FastAPI entry point for MediTrack.

Run:  uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.config import Settings, get_settings
from app.container import bootstrap, build_services
from ml.train import MODEL_PATH

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None, model_path: Path = MODEL_PATH, auto_bootstrap: bool = True) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if auto_bootstrap:
            bootstrap(settings, model_path)
        services = build_services(settings, model_path)
        app.state.services = services
        yield
        services.conn.close()

    app = FastAPI(
        title="MediTrack",
        description="Receptionist assistant: appointment availability chatbot and no-show risk.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    # Mounted last so /api/* routes win; serves index.html at "/".
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
