# Gamatrain AI API - Production Server

Production-ready API server with RAG (Retrieval-Augmented Generation) using HuggingFace embeddings.

## Features

- ✅ **Free embeddings** - Uses HuggingFace instead of OpenAI (no API key needed)
- ✅ **Multilingual support** - Excellent for Persian/Farsi and 100+ languages
- ✅ **Multiple LLM providers** - Ollama (local), Groq (cloud-free), OpenRouter
- ✅ **RAG with conversation memory** - Context-aware responses
- ✅ **Streaming responses** - Real-time token streaming
- ✅ **Docker ready** - Easy deployment

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements-production.txt

# Set environment variables
cp .env.production.example .env
nano .env

# Run server
python llm_server_production.py
```

### Docker Deployment

See [DOCKER_DEPLOYMENT.md](../DOCKER_DEPLOYMENT.md) for detailed instructions.

```bash
# Quick start
docker-compose -f docker-compose.production.yml up -d
```

## Configuration

### Environment Variables

```env
# Provider (ollama, groq, or openrouter)
PROVIDER=ollama

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gamatrain-qwen2.5

# Groq settings (free!)
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Storage
STORAGE_DIR=./storage
SIMILARITY_THRESHOLD=0.45
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Query
```bash
POST /query
{
  "query": "What is Gamatrain?",
  "session_id": "user123",
  "use_rag": true
}
```

### Stream Query
```bash
POST /stream
{
  "query": "Explain machine learning",
  "session_id": "user123"
}
```

### Regenerate Response
```bash
POST /v1/regenerate
{
  "session_id": "user123",
  "use_rag": true,
  "stream": false
}
```

### Clear Session
```bash
DELETE /v1/session/{session_id}
```

## Embedding Model

Uses `intfloat/multilingual-e5-large`:
- Size: ~2GB
- Dimensions: 1024
- Languages: 100+ including Persian/Farsi
- Quality: Excellent for RAG

**Note:** Model downloads on first run (~5-10 minutes).

## Troubleshooting

### "Could not load OpenAI embedding model"

Old index was built with OpenAI. Clean storage:
```bash
rm -rf storage/*
```

### "Connection refused" (Ollama)

Make sure Ollama is running:
```bash
ollama list
```

Or use Groq instead:
```env
PROVIDER=groq
GROQ_API_KEY=your_key_here
```

### Model download takes long

First run downloads ~2GB embedding model. This is normal and only happens once.

## Requirements

- Python 3.11+
- 4GB+ RAM (8GB recommended)
- 5GB+ disk space

## Files

- `llm_server_production.py` - Production server
- `requirements-production.txt` - Production dependencies
- `.env.production.example` - Example configuration

## License

See [LICENSE](../LICENSE) file.
