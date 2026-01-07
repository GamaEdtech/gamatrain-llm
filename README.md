# Gamatrain AI - Educational LLM with RAG 🤖

Fine-tuned LLM (Qwen2-1.5B) with RAG-powered API for Gamatrain's educational platform.

## 🎯 Overview

An AI assistant that:
- Answers questions about Gamatrain's educational content (courses, tests, blogs)
- Uses RAG (Retrieval-Augmented Generation) for accurate, context-aware responses
- Maintains conversation memory for follow-up questions
- Prevents hallucination with similarity threshold checks

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Fine-tuned LLM** | Qwen2-1.5B trained on Gamatrain content |
| **RAG Integration** | LlamaIndex-powered retrieval from 2000+ blogs |
| **Anti-Hallucination** | Similarity threshold + entity verification |
| **Conversation Memory** | Remembers context for follow-up questions |
| **OpenAI-Compatible API** | Drop-in replacement for OpenAI endpoints |
| **Multi-Provider** | Supports Ollama (local), Groq, OpenRouter |

## 📊 Model Stats

| Metric | Value |
|--------|-------|
| Base Model | Qwen2-1.5B-Instruct |
| Training Dataset | 2,614 samples |
| Domain Data | 2,422 (Gamatrain blogs, tests, courses) |
| General Data | 192 (math, logic, chat - weighted 4x) |
| Output Format | GGUF (4-bit quantized) |
| RAG Test Pass Rate | 92.9% |

## 🗂️ Project Structure

```
gamatrain-ai-research/
├── api/
│   ├── llm_server.py              # Development server (Ollama)
│   ├── llm_server_production.py   # Production server (Groq/OpenRouter)
│   ├── requirements.txt
│   └── requirements-production.txt
├── data/
│   ├── custom_docs.json           # Custom RAG documents
│   ├── gamatrain_final_dataset.jsonl
│   └── scripts/                   # Data extraction scripts
├── model/
│   ├── Modelfile                  # Ollama configuration
│   └── README.md                  # Model download instructions
├── scripts/
│   ├── test_model_and_rag.py      # Main test suite
│   └── test_random_blogs.py       # Random blog RAG tests
├── notebooks/
│   └── fine-tuning-complete.ipynb # Training notebook (Colab)
├── docs/
│   ├── DEPLOYMENT.md
│   ├── RESEARCH.md
│   └── TRAINING.md
├── docker-compose.production.yml
└── Dockerfile.production
```


## 🚀 Quick Start

### Option 1: Local Development (with Ollama)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Import the fine-tuned model
cd model/
# Place qwen2-gamatrain.gguf here (see model/README.md)
ollama create gamatrain-qwen -f Modelfile

# 3. Start the API server
cd api/
pip install -r requirements.txt
python llm_server.py
# Server runs on http://localhost:8000
```

### Option 2: Production (No GPU Required)

Uses cloud LLM providers (Groq is free and fast).

```bash
# 1. Setup environment
cd api/
cp .env.production.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)

# 2. Install and run
pip install -r requirements-production.txt
python llm_server_production.py
# Server runs on http://localhost:8001
```

### Option 3: Docker

```bash
docker-compose -f docker-compose.production.yml up -d
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/query` | POST | RAG query with streaming |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/refresh` | POST | Refresh RAG index |
| `/v1/session/{id}` | DELETE | Clear conversation memory |
| `/health` | GET | Health check |

### Example Requests

```bash
# Simple query
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Gamatrain?", "session_id": "user1"}'

# Follow-up question (uses conversation memory)
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me more about that", "session_id": "user1"}'

# Refresh RAG index (after new content is added)
curl -X POST http://localhost:8000/v1/refresh
```

### Response Format

```json
{
  "query": "What is Gamatrain?",
  "response": "Gamatrain is an educational technology company...",
  "confidence": "high",
  "similarity_score": 0.897,
  "session_id": "user1"
}
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVIDER` | ollama | LLM provider: `ollama`, `groq`, `openrouter` |
| `GROQ_API_KEY` | - | Groq API key (free tier available) |
| `GROQ_MODEL` | llama-3.1-8b-instant | Model to use with Groq |
| `OLLAMA_MODEL` | gamatrain-qwen | Local Ollama model name |
| `SIMILARITY_THRESHOLD` | 0.45 | RAG confidence threshold |
| `MAX_TOKENS` | 1024 | Maximum response tokens |
| `PORT` | 8000/8001 | Server port |

## 🛡️ Anti-Hallucination

The system prevents made-up responses through:

1. **Similarity Threshold** - Low-confidence queries return "I don't know"
2. **Entity Verification** - Checks if mentioned entities exist in context
3. **Strict Prompting** - Instructs model to only use provided context

## 🧪 Running Tests

```bash
# Main test suite
python scripts/test_model_and_rag.py

# Random blog RAG tests
python scripts/test_random_blogs.py
```

## 📚 Documentation

- [PRODUCTION.md](docs/PRODUCTION.md) - **Production deployment guide** (recommended)
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Basic deployment guide
- [TRAINING.md](docs/TRAINING.md) - Fine-tuning guide
- [RESEARCH.md](docs/RESEARCH.md) - Research findings

## ⚠️ Key Learning: Catastrophic Forgetting

Fine-tuning only on domain data caused the model to "forget" basic abilities.

**Solution:** Mix domain data with general knowledge samples (weighted 4x).

| Before | After |
|--------|-------|
| `2 + 2 = 0` ❌ | `2 + 2 = 4` ✅ |

## 📄 License

MIT License
