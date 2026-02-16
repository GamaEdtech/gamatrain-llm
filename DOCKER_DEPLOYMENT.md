# Docker Deployment Guide - Gamatrain AI

## ✅ What's New

This version uses **HuggingFace embeddings**:
- ✅ Free and unlimited
- ✅ No OpenAI API key required
- ✅ Multilingual support (including Persian/Farsi)
- ✅ Excellent quality for RAG

## 🚀 Quick Start

### 1. Setup Environment Variables

```bash
# Copy example file
cp .env.docker.example .env.docker

# Edit and configure
nano .env.docker
```

### 2. Clean Old Storage (Important!)

```bash
# If you have old storage built with OpenAI, remove it
rm -rf storage/*
```

### 3. Build and Run Docker

```bash
# Build image
docker-compose -f docker-compose.production.yml build

# Run
docker-compose -f docker-compose.production.yml up -d
```

### 4. Check Logs

```bash
docker-compose -f docker-compose.production.yml logs -f
```

## 📝 Expected Logs

You should see these logs:

```
INFO:GamatrainAPI:Starting Gamatrain AI Server (Production)...
INFO:GamatrainAPI:Using provider: ollama
INFO:GamatrainAPI:Setting up embedding model...
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: intfloat/multilingual-e5-large
INFO:GamatrainAPI:Embedding model ready
INFO:GamatrainAPI:Building index with XXXX documents...
INFO:GamatrainAPI:Using HuggingFace embedding model (intfloat/multilingual-e5-large)
Generating embeddings: 100%|██████████| XX/XX
INFO:GamatrainAPI:Index built and saved successfully
INFO:GamatrainAPI:Server ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## ⚙️ Provider Configuration

### Option 1: Ollama (Local - Recommended)

```env
PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=gamatrain-qwen2.5
```

**Note:** Ollama must be running on the host machine.

### Option 2: Groq (Cloud - Free!)

```env
PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Get API key: https://console.groq.com

### Option 3: OpenRouter (Cloud)

```env
PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

## 🔧 Troubleshooting

### Error: "Could not load OpenAI embedding model"

**Cause:** Old index was built with OpenAI.

**Solution:**
```bash
# Stop container
docker-compose -f docker-compose.production.yml down

# Clean storage
rm -rf storage/*

# Restart
docker-compose -f docker-compose.production.yml up -d
```

### Error: "Connection refused" (Ollama)

**Cause:** Docker cannot access Ollama on host.

**Solution:**
```bash
# Make sure Ollama is running on host
ollama list

# Or use Groq instead
PROVIDER=groq
```

### Error: "No module named..."

**Cause:** Dependencies not installed.

**Solution:**
```bash
# Rebuild image
docker-compose -f docker-compose.production.yml build --no-cache
docker-compose -f docker-compose.production.yml up -d
```

### Embedding model download takes long

**Cause:** Model (~2GB) is downloading.

**Solution:** Wait. This only happens once.

## 📊 Monitoring

### Check Status

```bash
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Gamatrain?", "session_id": "test"}'
```

### View Logs

```bash
# Live logs
docker-compose -f docker-compose.production.yml logs -f

# Last 100 lines
docker-compose -f docker-compose.production.yml logs --tail=100
```

### Check Resources

```bash
# CPU/Memory usage
docker stats gamatrain-ai-research-gamatrain-ai-1
```

## 🔄 Updates

```bash
# Pull latest changes
git pull

# Rebuild
docker-compose -f docker-compose.production.yml build

# Restart
docker-compose -f docker-compose.production.yml up -d
```

## 🛑 Stop and Remove

```bash
# Stop
docker-compose -f docker-compose.production.yml down

# Stop and remove volumes
docker-compose -f docker-compose.production.yml down -v

# Remove image
docker rmi gamatrain-ai-research-gamatrain-ai
```

## 📦 Storage and Resources

### Required Storage:
- Docker Image: ~2GB
- Embedding Model: ~2GB
- Index Storage: ~500MB-1GB
- Total: ~5GB

### Recommended Resources:
- CPU: 2+ cores
- RAM: 4GB+ (8GB recommended)
- Disk: 10GB+ free space

## 🌐 Production Deployment

### With Docker Swarm

```bash
docker stack deploy -c docker-compose.production.yml gamatrain
```

### With Kubernetes

```bash
# Convert docker-compose to kubernetes manifests
kompose convert -f docker-compose.production.yml

# Apply
kubectl apply -f .
```

## 🔐 Security

### Security Recommendations:

1. **Keep API keys in environment variables**
2. **Add .env.docker to .gitignore**
3. **Use reverse proxy (nginx)**
4. **Enable SSL/TLS**
5. **Add rate limiting**

### Nginx Example:

```nginx
server {
    listen 80;
    server_name api.gamatrain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📞 Support

If you encounter issues:
1. Check Docker logs
2. Review this documentation
3. Open an issue on GitHub

## ✨ Important Notes

1. **First run:** Embedding model download takes time (~2GB)
2. **Storage:** Make sure to clean old storage
3. **Provider:** Ollama for local, Groq for cloud recommended
4. **No OpenAI:** No external API key needed!

Good luck! 🚀
