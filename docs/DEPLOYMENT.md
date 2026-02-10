# Deployment Guide 🚀

## Option 1: Ollama (Recommended)

### Install Ollama
```bash
# Linux/WSL
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama
```

### Import Model
```bash
cd model/
# Place qwen2-gamatrain.gguf here
ollama create gamatrain-qwen -f Modelfile
```

### Run Model
```bash
# Interactive mode
ollama run gamatrain-qwen

# Single query
ollama run gamatrain-qwen "What is 2 + 2?"
```

## Option 2: API Server

### Setup
```bash
cd api/
pip install -r requirements.txt
```

### Run Server
```bash
python llm_server.py
# Server runs on http://localhost:8000
```

### API Usage
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about Ohm'\''s Law"}'
```

### OpenAI-Compatible Endpoint
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Production Deployment

For production, consider:

1. **Reverse Proxy**: Use Nginx for SSL/load balancing
2. **Process Manager**: Use systemd or PM2
3. **Rate Limiting**: Add rate limiting to API
4. **Monitoring**: Add logging and health checks

### Example systemd Service
```ini
[Unit]
Description=Gamatrain LLM API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/api
ExecStart=/usr/bin/python3 llm_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```
