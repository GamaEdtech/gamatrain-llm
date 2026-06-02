import os

# =============================================================================
# RAG
# =============================================================================
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").strip().lower() not in {"0", "false", "no", "off"}

# =============================================================================
# LLM provider (ONE active per deployment)
# =============================================================================
# Prefer LLM_PROVIDER; PROVIDER is kept for backward compatibility.
LLM_PROVIDER = (
    os.getenv("LLM_PROVIDER") or os.getenv("PROVIDER", "ollama")
).strip().lower()

_PROVIDER_DEFAULT_MODELS = {
    "ollama": "gamatrain-qwen",
    "groq": "llama-3.1-8b-instant",
    "openrouter": "meta-llama/llama-3.1-8b-instruct:free",
}


def _resolve_model() -> str:
    explicit = os.getenv("MODEL", "").strip()
    if explicit:
        return explicit
    if LLM_PROVIDER == "ollama":
        return os.getenv("OLLAMA_MODEL", _PROVIDER_DEFAULT_MODELS["ollama"])
    if LLM_PROVIDER == "groq":
        return os.getenv("GROQ_MODEL", _PROVIDER_DEFAULT_MODELS["groq"])
    if LLM_PROVIDER == "openrouter":
        return os.getenv("OPENROUTER_MODEL", _PROVIDER_DEFAULT_MODELS["openrouter"])
    return _PROVIDER_DEFAULT_MODELS.get(LLM_PROVIDER, _PROVIDER_DEFAULT_MODELS["ollama"])


def _resolve_api_key() -> str:
    explicit = os.getenv("API_KEY", "").strip()
    if explicit:
        return explicit
    if LLM_PROVIDER == "groq":
        return os.getenv("GROQ_API_KEY", "")
    if LLM_PROVIDER == "openrouter":
        return os.getenv("OPENROUTER_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "")


MODEL = _resolve_model()
API_KEY = _resolve_api_key()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# Deprecated aliases (use LLM_PROVIDER / MODEL / API_KEY in new deployments)
PROVIDER = LLM_PROVIDER


def sync_legacy_llm_env() -> None:
    """Map unified settings to env names used by llm_server_production."""
    os.environ["PROVIDER"] = LLM_PROVIDER
    os.environ["LLM_PROVIDER"] = LLM_PROVIDER
    if LLM_PROVIDER == "ollama":
        os.environ["OLLAMA_MODEL"] = MODEL
    elif LLM_PROVIDER == "groq":
        os.environ["GROQ_MODEL"] = MODEL
        if API_KEY:
            os.environ["GROQ_API_KEY"] = API_KEY
    elif LLM_PROVIDER == "openrouter":
        os.environ["OPENROUTER_MODEL"] = MODEL
        if API_KEY:
            os.environ["OPENROUTER_API_KEY"] = API_KEY
    os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
