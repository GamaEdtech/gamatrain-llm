# Gamatrain AI Research 🤖

Fine-tuned LLM (Qwen2-1.5B) with RAG-powered API for Gamatrain's educational platform.

## 🎯 Project Goal

Create an AI assistant that can:
- Answer questions about Gamatrain's educational content (courses, tests, blogs)
- Use RAG (Retrieval-Augmented Generation) for accurate, context-aware responses
- Maintain conversation memory for follow-up questions
- Prevent hallucination with similarity threshold checks
- Be deployed locally using Ollama

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Fine-tuned LLM** | Qwen2-1.5B trained on Gamatrain content |
| **RAG Integration** | LlamaIndex-powered retrieval from blogs, schools, FAQs |
| **Anti-Hallucination** | Similarity threshold + entity verification |
| **Conversation Memory** | Remembers context for follow-up questions |
| **OpenAI-Compatible API** | Drop-in replacement for OpenAI endpoints |
| **Auto-sync** | Fetches latest content from Gamatrain API |

## 📊 Model Stats

| Metric | Value |
|--------|-------|
| Base Model | Qwen2-1.5B-Instruct |
| Training Dataset | 2,614 samples |
| Domain Data | 2,422 (Gamatrain blogs, tests, courses) |
| General Data | 192 (math, logic, chat - weighted 4x) |
| Output Format | GGUF (4-bit quantized) |
| RAG Test Pass Rate | 92.9% |

## 🗂️ Repository Structure

```
gamatrain-ai-research/
├── api/
│   ├── llm_server.py          # FastAPI + RAG server
│   └── requirements.txt
├── data/
│   ├── custom_docs.json       # Custom documents for RAG
│   ├── gamatrain_final_dataset.jsonl
│   └── scripts/               # Data extraction scripts
├── model/
│   ├── Modelfile              # Ollama configuration
│   ├── qwen2-gamatrain.gguf   # Fine-tuned model
│   └── README.md
├── scripts/
│   └── test_model_and_rag.py  # Test suite
├── storage/                   # RAG vector store
├── notebooks/
│   └── fine-tuning-demo.ipynb
└── docs/
    ├── RESEARCH.md
    ├── TRAINING.md
    └── DEPLOYMENT.md
```

## 🚀 Quick Start

### 1. Setup Model
```bash
# Download model (see model/README.md)
cd model/
ollama create gamatrain-qwen -f Modelfile
```

### 2. Start API Server
```bash
cd api/
pip install -r requirements.txt
python llm_server.py
```

### 3. Test Endpoints
```bash
# Simple query
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Gamatrain?"}'

# With conversation memory
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Oil blog about?", "session_id": "user1"}'

curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me more about that", "session_id": "user1"}'
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/query` | POST | RAG query with confidence score |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/documents/add` | POST | Add document to RAG |
| `/v1/documents/count` | GET | Get document count |
| `/v1/refresh` | POST | Refresh RAG index |
| `/v1/session/{id}` | DELETE | Clear conversation memory |
| `/health` | GET | Health check |

## 📝 Query Response Example

```json
{
  "query": "What is Gamatrain?",
  "response": "Gamatrain is an educational technology company (EdTech) that provides AI-powered learning tools.",
  "confidence": "high",
  "similarity_score": 0.897,
  "session_id": "default",
  "source": "rag"
}
```

## 🛡️ Anti-Hallucination

The system prevents made-up responses through:

1. **Similarity Threshold (0.75)** - Low-confidence queries return "I don't know"
2. **Entity Verification** - Checks if mentioned names exist in context
3. **Strict Prompting** - Instructs model to only use provided context

```bash
# Example: Asking about non-existent school
curl -X POST http://localhost:8000/v1/query \
  -d '{"query": "Tell me about XYZ123 school"}'

# Response:
{"response": "I don't have specific information about XYZ123 in my knowledge base."}
```

## 🔄 Adding Custom Data

### Option 1: JSON File
Edit `data/custom_docs.json`:
```json
{
  "documents": [
    {"text": "Your content here...", "type": "faq", "id": "faq_001"}
  ]
}
```
Then refresh: `curl -X POST http://localhost:8000/v1/refresh -d '{"force": true}'`

### Option 2: API
```bash
curl -X POST http://localhost:8000/v1/documents/add \
  -H "Content-Type: application/json" \
  -d '{"text": "New document content", "doc_type": "faq"}'
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | gamatrain-qwen | Ollama model name |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama API URL |
| `PORT` | 8000 | API server port |
| `SIMILARITY_THRESHOLD` | 0.75 | RAG confidence threshold |
| `STORAGE_DIR` | ./storage | Vector store location |

## 🧪 Running Tests

```bash
python scripts/test_model_and_rag.py
```

Test categories:
- Identity (Who are you?)
- RAG Retrieval
- Hallucination Prevention
- Educational Knowledge
- Response Quality

## ⚠️ Key Learning: Catastrophic Forgetting

Fine-tuning only on domain data caused the model to "forget" basic abilities.

**Solution:** Mix domain data with general knowledge samples (weighted 4x).

| Before | After |
|--------|-------|
| `2 + 2 = 0` ❌ | `2 + 2 = 4` ✅ |

## 📚 Documentation

- [RESEARCH.md](docs/RESEARCH.md) - Problem statement & findings
- [TRAINING.md](docs/TRAINING.md) - Fine-tuning guide
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment guide

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read the documentation first.
