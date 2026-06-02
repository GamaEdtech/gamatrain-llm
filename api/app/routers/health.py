from fastapi import APIRouter

from ..core.config import ENABLE_RAG, LLM_PROVIDER, MODEL
from ..services import legacy

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    return {
        "status": "online",
        "service": "Gamatrain AI (Production)",
        "provider": LLM_PROVIDER,
        "model": MODEL,
        "rag_enabled": ENABLE_RAG,
    }


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "provider": LLM_PROVIDER,
        "model": MODEL,
        "rag_enabled": ENABLE_RAG,
        "rag_ready": legacy.rag_ready() if ENABLE_RAG else False,
    }

