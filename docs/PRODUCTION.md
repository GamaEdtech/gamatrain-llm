# Production Deployment Guide 🚀

This guide explains how to deploy the Gamatrain AI model in production environments.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│  Production Server   │────▶│   LLM Provider  │
│   (Nuxt/React)  │     │  (FastAPI + RAG)     │     │  (Groq/Ollama)  │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                  │
                                  ▼
                        ┌──────────────────────┐
                        │   Gamatrain API      │
                        │   (Blogs, Schools)   │
                        └──────────────────────┘
```

## Deployment Options

### Option 1: Cloud LLM (Recommended for Production)

Uses Groq API - **free tier available**, very fast inference.

**Pros:**
- No GPU required
- Fast response times (~1-2s)
- Free tier: 30 requests/minute
- Easy to scale

**Cons:**
- Requires internet connection
- Rate limits on free tier

#### Setup

1. Get a free API key from [Groq Console](https://console.groq.com)

2. Configure environment:
```bash
cd api/
cp .env.production.example .env
```

3. Edit `.env`:
```env
PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
PORT=8002
SIMILARITY_THRESHOLD=0.45
```

4. Run:
```bash
pip install -r requirements-production.txt
python llm_server_production.py
```

### Option 2: Local Ollama (Self-Hosted)

Uses your fine-tuned model locally.

**Pros:**
- Full control over the model
- No external dependencies
- No rate limits
- Uses your custom fine-tuned model

**Cons:**
- Requires server with decent CPU/RAM
- Slower inference without GPU

#### Setup

1. Install Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Import the model:
```bash
cd model/
# Download qwen2-gamatrain.gguf (see model/README.md)
ollama create gamatrain-qwen -f Modelfile
```

3. Configure environment:
```env
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gamatrain-qwen
```

4. Run:
```bash
python llm_server_production.py
```

### Option 3: Docker Deployment

```bash
# Build and run
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Stop
docker-compose -f docker-compose.production.yml down
```

## Production Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROVIDER` | Yes | `ollama` | `ollama`, `groq`, or `openrouter` |
| `GROQ_API_KEY` | If using Groq | - | Get from console.groq.com |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model name |
| `OLLAMA_BASE_URL` | If using Ollama | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | If using Ollama | `gamatrain-qwen` | Model name in Ollama |
| `PORT` | No | `8001` | Server port |
| `HOST` | No | `0.0.0.0` | Server host |
| `SIMILARITY_THRESHOLD` | No | `0.45` | RAG confidence threshold |
| `MAX_TOKENS` | No | `1024` | Max response tokens |
| `GAMATRAIN_API_URL` | No | `https://...` | Gamatrain API for fetching content |

### Recommended Production Settings

```env
# Production .env
PROVIDER=groq
GROQ_API_KEY=your_production_key
GROQ_MODEL=llama-3.1-8b-instant

HOST=0.0.0.0
PORT=8002

SIMILARITY_THRESHOLD=0.45
MAX_TOKENS=1024

GAMATRAIN_API_URL=https://your-api-url/api/v1
```

## RAG Index Management

### Initial Setup

On first startup, the server:
1. Fetches all blogs from Gamatrain API
2. Loads custom documents from `data/custom_docs.json`
3. Builds vector index using `intfloat/multilingual-e5-large` embeddings
4. Saves index to `storage/` directory

### Refreshing the Index

When new content is added to Gamatrain:

```bash
curl -X POST http://localhost:8002/v1/refresh
```

This will:
- Fetch latest blogs from API
- Rebuild the vector index
- Save to storage

### Adding Custom Documents

Edit `data/custom_docs.json`:

```json
{
  "documents": [
    {
      "text": "Your custom content here...",
      "type": "faq",
      "id": "faq_001"
    }
  ]
}
```

Then refresh: `curl -X POST http://localhost:8002/v1/refresh`

## Nginx Reverse Proxy

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name ai.gamatrain.com;

    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_cache_bypass $http_upgrade;
        
        # For streaming responses
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
```

## Systemd Service

Create `/etc/systemd/system/gamatrain-ai.service`:

```ini
[Unit]
Description=Gamatrain AI Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/gamatrain-ai-research/api
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python llm_server_production.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable gamatrain-ai
sudo systemctl start gamatrain-ai
sudo systemctl status gamatrain-ai
```

## Monitoring

### Health Check

```bash
curl http://localhost:8002/health
```

Response:
```json
{
  "status": "healthy",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "rag_ready": true
}
```

### Debug Endpoints

```bash
# Search RAG with scores
curl "http://localhost:8002/v1/debug/search?q=machine+learning"

# Find specific blog
curl "http://localhost:8002/v1/debug/find-blog?title=physics"

# List all blogs
curl "http://localhost:8002/v1/debug/list-blogs?search=math"
```

## Troubleshooting

### RAG not finding content

1. Check similarity threshold (lower = more results):
```env
SIMILARITY_THRESHOLD=0.35
```

2. Refresh the index:
```bash
curl -X POST http://localhost:8002/v1/refresh
```

3. Debug search results:
```bash
curl "http://localhost:8002/v1/debug/search?q=your+query"
```

### Slow responses

1. If using Ollama, ensure adequate RAM (8GB+ recommended)
2. Consider switching to Groq for faster inference
3. Reduce `MAX_TOKENS` if responses are too long

### Rate limiting (Groq)

Free tier: 30 requests/minute. For higher limits:
- Upgrade to paid tier
- Implement request queuing
- Use multiple API keys with rotation

## Security Considerations

1. **API Keys**: Never commit `.env` files. Use environment variables or secrets management.

2. **CORS**: Configure allowed origins in production:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gamatrain.com"],  # Specific origins
    ...
)
```

3. **Rate Limiting**: Implement rate limiting for public endpoints.

4. **Input Validation**: The server validates input, but consider additional sanitization.

## Performance Tuning

### Embedding Model

The default `intfloat/multilingual-e5-large` is good for multilingual content. First load downloads ~2GB.

For faster startup, pre-download:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')
```

### Index Persistence

Index is saved to `storage/` directory. On restart, it loads from disk instead of rebuilding.

To force rebuild:
```bash
rm -rf storage/
python llm_server_production.py
```
