# Gamatrain AI Server

FastAPI server with RAG, conversation memory, and multi-provider LLM support.

## Features

- **RAG (Retrieval-Augmented Generation)**: Smart search across 2000+ blogs
- **Conversation Memory**: Remembers last 5 messages per session
- **Follow-up Detection**: Handles "tell me more" style questions
- **Multi-Provider**: Supports Ollama (local), Groq (cloud), OpenRouter
- **Streaming**: Real-time token streaming for better UX

## Quick Start

### Development (with Ollama)

```bash
pip install -r requirements.txt
python llm_server.py
# Server runs on http://localhost:8000
```

### Production (with Groq - Free)

```bash
cp .env.production.example .env
# Edit .env and add your GROQ_API_KEY

pip install -r requirements-production.txt
python llm_server_production.py
# Server runs on http://localhost:8002
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/query` | Query with RAG and streaming |
| `POST` | `/v1/chat/completions` | OpenAI-compatible endpoint |
| `POST` | `/v1/refresh` | Refresh RAG index |
| `DELETE` | `/v1/session/{id}` | Clear session memory |
| `GET` | `/health` | Health check |
| `GET` | `/v1/debug/search?q=...` | Debug: search with scores |

## Example Request

```bash
curl -X POST "http://localhost:8002/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "session_id": "user123"}'
```

## Environment Variables

See `.env.production.example` for all available options.

Key variables:
- `PROVIDER`: `ollama`, `groq`, or `openrouter`
- `GROQ_API_KEY`: Get free key from https://console.groq.com
- `SIMILARITY_THRESHOLD`: RAG confidence threshold (default: 0.45)

## Files

- `llm_server.py` - Development server (uses local Ollama)
- `llm_server_production.py` - Production server (multi-provider)
- `requirements.txt` - Development dependencies
- `requirements-production.txt` - Production dependencies
