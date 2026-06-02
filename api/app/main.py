import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import ENABLE_RAG, LLM_PROVIDER, sync_legacy_llm_env
from .routers.chat import router as chat_router
from .routers.health import router as health_router
from .services import legacy

logger = logging.getLogger("GamatrainAPI")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting Gamatrain AI Server (Modular App)...")
    logger.info("Using provider: %s", LLM_PROVIDER)
    sync_legacy_llm_env()
    if ENABLE_RAG:
        legacy.initialize()
    else:
        logger.warning("ENABLE_RAG=false; skipping RAG initialization (no embedding downloads).")
    logger.info("Server ready!")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gamatrain AI API",
        description="RAG-powered educational AI (Production, modularized)",
        version="2.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()

