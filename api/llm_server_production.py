"""
Gamatrain LLM API Server - Production Version (HuggingFace + RAG)
================================================================

This version runs WITHOUT local GPU:
- RAG, conversation memory, follow-up detection run on your server
- Model inference uses HuggingFace Inference API

Requirements:
    pip install fastapi uvicorn httpx llama-index llama-index-embeddings-huggingface huggingface_hub python-dotenv
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import uvicorn
import httpx
import logging
import json
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# =============================================================================
# Configuration
# =============================================================================
# Model Provider Settings
# Option 1: OpenRouter (free tier available)
# Option 2: Together AI
# Option 3: Groq (fast & free)
PROVIDER = os.getenv("PROVIDER", "groq")  # groq, openrouter, together

# Groq Settings (FREE and FAST!)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# OpenRouter Settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
CUSTOM_DOCS_PATH = os.getenv("CUSTOM_DOCS_PATH", "../data/custom_docs.json")

# Gamatrain API for fetching documents
API_BASE_URL = os.getenv("GAMATRAIN_API_URL", "https://185.204.170.142/api/v1")
AUTH_TOKEN = os.getenv("GAMATRAIN_AUTH_TOKEN", "")

# RAG Settings
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))  # Lowered for better recall
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GamatrainAPI")

# Global components
query_engine = None
index_store = None

# Conversation memory
conversation_memory = defaultdict(list)
MAX_MEMORY_TURNS = 5


# =============================================================================
# LLM Inference (Multiple Providers)
# =============================================================================
async def call_llm_api(prompt: str, max_tokens: int = 1024):
    """Call LLM API based on configured provider."""
    
    if PROVIDER == "groq":
        return await call_groq_api(prompt, max_tokens)
    elif PROVIDER == "openrouter":
        return await call_openrouter_api(prompt, max_tokens)
    else:
        return "Error: No valid provider configured"


async def call_groq_api(prompt: str, max_tokens: int = 1024):
    """Call Groq API (FREE and very fast!)"""
    
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are Gamatrain AI, an educational assistant. Be helpful and concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"Error: {str(e)}"


async def call_openrouter_api(prompt: str, max_tokens: int = 1024):
    """Call OpenRouter API (has free models)"""
    
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not set"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are Gamatrain AI, an educational assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        return f"Error: {str(e)}"


async def stream_huggingface_api(prompt: str, max_tokens: int = 1024):
    """Stream response with realistic typing effect."""
    import asyncio
    
    try:
        full_response = await call_llm_api(prompt, max_tokens)
        
        # Simulate streaming by yielding chunks with delay
        words = full_response.split()
        for i, word in enumerate(words):
            token = word + " " if i < len(words) - 1 else word
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            # Add small delay for typing effect (30ms per word)
            await asyncio.sleep(0.08)
        
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


# =============================================================================
# RAG Setup (same as before, but without local LLM)
# =============================================================================
def setup_embeddings():
    """Initialize embedding model (runs on CPU)."""
    logger.info("Setting up embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large"
    )
    logger.info("Embedding model ready")


def fetch_documents():
    """Fetch documents from Gamatrain API and custom docs file."""
    documents = []
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
    
    # Add Gamatrain company info
    gamatrain_info = """
    Gamatrain is an educational technology company (EdTech) that provides AI-powered learning tools.
    Gamatrain AI is an intelligent educational assistant created by Gamatrain's development team.
    Gamatrain helps students learn through personalized education and smart tutoring.
    The Gamatrain app is available on both iOS and Android and can be downloaded from the App Store or Google Play.
    """
    documents.append(Document(text=gamatrain_info, metadata={"type": "company", "id": "gamatrain"}))
    
    # Load custom documents
    if os.path.exists(CUSTOM_DOCS_PATH):
        try:
            with open(CUSTOM_DOCS_PATH, 'r', encoding='utf-8') as f:
                custom_data = json.load(f)
                for doc in custom_data.get("documents", []):
                    documents.append(Document(
                        text=doc["text"],
                        metadata={"type": doc.get("type", "custom"), "id": doc.get("id", "")}
                    ))
                logger.info(f"Loaded {len(custom_data.get('documents', []))} custom documents")
        except Exception as e:
            logger.warning(f"Could not load custom docs: {e}")
    
    # Fetch blogs from API
    try:
        with httpx.Client(verify=False, timeout=120) as client:
            resp = client.get(
                f"{API_BASE_URL}/blogs/posts",
                params={"PagingDto.PageFilter.Size": 2000, "PagingDto.PageFilter.Skip": 0},
                headers=headers
            )
            if resp.status_code == 200:
                blogs = resp.json().get("data", {}).get("list", [])
                for post in blogs:
                    title = post.get("title", "")
                    summary = post.get("summary", "")
                    content = post.get("content", "")
                    
                    if title:
                        import re
                        clean_content = re.sub(r'<[^>]+>', '', content) if content else ""
                        blog_text = f"Blog Title: {title}\n"
                        if summary:
                            blog_text += f"Summary: {summary}\n"
                        if clean_content:
                            blog_text += f"Content: {clean_content[:1000]}\n"
                        
                        documents.append(Document(
                            text=blog_text,
                            metadata={"type": "blog", "id": str(post.get("id"))}
                        ))
                logger.info(f"Fetched {len(blogs)} blogs")
    except Exception as e:
        logger.warning(f"Could not fetch blogs: {e}")
    
    # Fetch schools
    try:
        with httpx.Client(verify=False, timeout=30) as client:
            resp = client.get(
                f"{API_BASE_URL}/schools",
                params={"PagingDto.PageFilter.Size": 100, "PagingDto.PageFilter.Skip": 0},
                headers=headers
            )
            if resp.status_code == 200:
                schools = resp.json().get("data", {}).get("list", [])
                for school in schools:
                    name = school.get("name", "")
                    if name and "gamatrain" not in name.lower():
                        documents.append(Document(
                            text=f"School: {name}\nCity: {school.get('cityTitle', '')}\nCountry: {school.get('countryTitle', '')}",
                            metadata={"type": "school", "id": str(school.get("id"))}
                        ))
                logger.info(f"Fetched {len(schools)} schools")
    except Exception as e:
        logger.warning(f"Could not fetch schools: {e}")
    
    return documents


def build_index(documents: List[Document]):
    """Build or load RAG index."""
    global query_engine, index_store
    
    # Try to load existing index
    if os.path.exists(os.path.join(STORAGE_DIR, "docstore.json")):
        try:
            logger.info("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            index_store = load_index_from_storage(storage_context)
            query_engine = index_store.as_retriever(similarity_top_k=3)
            logger.info("Index loaded")
            return
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
    
    # Build new index
    logger.info(f"Building index with {len(documents)} documents...")
    index_store = VectorStoreIndex.from_documents(documents)
    index_store.storage_context.persist(persist_dir=STORAGE_DIR)
    query_engine = index_store.as_retriever(similarity_top_k=3)
    logger.info("Index built and saved")


# =============================================================================
# Query Processing (with RAG + Memory + Follow-up)
# =============================================================================
async def process_query(query_text: str, session_id: str = "default", use_rag: bool = True):
    """Process query with RAG, memory, and follow-up detection."""
    global index_store, conversation_memory
    
    query_lower = query_text.lower().strip()
    history = conversation_memory[session_id]
    
    # Detect follow-up questions
    follow_up_words = ["that", "this", "it", "those", "these", "more", "explain", "elaborate", 
                       "details", "different", "same", "similar", "compare"]
    follow_up_phrases = ["tell me more", "explain more", "can you explain", "what about", 
                         "how about", "more details", "how is it", "different from"]
    
    is_follow_up = history and (
        any(word in query_lower.split() for word in follow_up_words) or
        any(phrase in query_lower for phrase in follow_up_phrases)
    )
    
    # Detect general greetings
    general_patterns = ['hi', 'hello', 'hey', 'good morning', 'how are you', 
                        'what can you do', 'who are you', 'thanks', 'bye']
    is_general = any(query_lower == p or query_lower.startswith(p + ' ') for p in general_patterns)
    
    # Build prompt based on context
    if is_general and not is_follow_up:
        prompt = f"You are Gamatrain AI, a friendly educational assistant. Respond briefly: {query_text}"
        return prompt, None
    
    # Handle follow-up questions
    if is_follow_up and history:
        last_entry = history[-1]
        last_response = last_entry.get("response", "")[:600]
        
        prompt = f"""Based on this context: {last_response}

{query_text}"""
        return prompt, last_entry.get("topic", "")
    
    # Use RAG for new questions
    if use_rag and index_store:
        retriever = index_store.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(query_text)
        
        if nodes and max([n.score for n in nodes]) >= SIMILARITY_THRESHOLD:
            context = "\n".join([n.text for n in nodes[:3]])
            
            prompt = f"""Context:
{context}

Question: {query_text}

Answer based on the context above. If the answer is not in the context, say so."""
            
            # Extract topic
            topic = ""
            best_node = max(nodes, key=lambda n: n.score)
            if "Blog Title:" in best_node.text:
                topic = best_node.text.split("Blog Title:")[1].split("\n")[0].strip()
            
            return prompt, topic
    
    # Fallback to direct question
    prompt = f"You are Gamatrain AI, an educational assistant. Answer: {query_text}"
    return prompt, None


async def stream_query(query_text: str, session_id: str = "default", use_rag: bool = True):
    """Stream response with RAG and memory."""
    global conversation_memory
    
    # Process query
    prompt, topic = await process_query(query_text, session_id, use_rag)
    
    # Stream from HuggingFace
    full_response = ""
    async for chunk in stream_huggingface_api(prompt, MAX_TOKENS):
        # Parse the chunk to extract token
        try:
            data = json.loads(chunk.replace("data: ", "").strip())
            if not data.get("done"):
                full_response += data.get("token", "")
        except:
            pass
        yield chunk
    
    # Save to memory
    conversation_memory[session_id].append({
        "query": query_text,
        "response": full_response,
        "topic": topic or query_text
    })
    
    if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS:]


# =============================================================================
# FastAPI App
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    logger.info("Starting Gamatrain AI Server (Production)...")
    logger.info(f"Using provider: {PROVIDER}")
    
    if PROVIDER == "groq" and not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set!")
    elif PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set!")
    
    setup_embeddings()
    documents = fetch_documents()
    build_index(documents)
    
    logger.info("Server ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Gamatrain AI API",
    description="RAG-powered educational AI (Production)",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Data Models
# =============================================================================
class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True
    session_id: str = "default"
    stream: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True
    session_id: str = "default"
    use_rag: bool = True


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Gamatrain AI (Production)",
        "model": GROQ_MODEL if PROVIDER == "groq" else OPENROUTER_MODEL,
        "rag_enabled": index_store is not None
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": GROQ_MODEL if PROVIDER == "groq" else OPENROUTER_MODEL,
        "rag_ready": index_store is not None
    }


@app.post("/v1/query")
async def query(request: QueryRequest):
    """Main query endpoint with streaming."""
    if not request.query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    logger.info(f"Query: {request.query[:50]}... (session: {request.session_id})")
    
    if request.stream:
        return StreamingResponse(
            stream_query(request.query, request.session_id, request.use_rag),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    prompt, topic = await process_query(request.query, request.session_id, request.use_rag)
    response_text = await call_llm_api(prompt, MAX_TOKENS)
    
    # Save to memory
    conversation_memory[request.session_id].append({
        "query": request.query,
        "response": response_text,
        "topic": topic or request.query
    })
    
    return {
        "query": request.query,
        "response": response_text,
        "session_id": request.session_id
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat endpoint."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    last_message = request.messages[-1].content
    
    if request.stream:
        return StreamingResponse(
            stream_query(last_message, request.session_id, request.use_rag),
            media_type="text/event-stream"
        )
    
    prompt, topic = await process_query(last_message, request.session_id, request.use_rag)
    response_text = await call_llm_api(prompt, MAX_TOKENS)
    
    return {
        "id": "chatcmpl-gamatrain",
        "object": "chat.completion",
        "model": GROQ_MODEL if PROVIDER == "groq" else OPENROUTER_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }]
    }


@app.delete("/v1/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
        return {"status": "success", "message": f"Session {session_id} cleared"}
    return {"status": "not_found"}


@app.post("/v1/refresh")
async def refresh_index():
    """Refresh RAG index."""
    documents = fetch_documents()
    build_index(documents)
    return {"status": "success", "documents_count": len(documents)}


@app.get("/v1/debug/search")
async def debug_search(q: str):
    """Debug endpoint to see RAG search results with scores."""
    global index_store
    
    if not index_store:
        return {"error": "Index not ready"}
    
    retriever = index_store.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(q)
    
    results = []
    for node in nodes:
        results.append({
            "score": round(node.score, 4),
            "text_preview": node.text[:300],
            "metadata": node.metadata,
            "passes_threshold": node.score >= SIMILARITY_THRESHOLD
        })
    
    return {
        "query": q,
        "threshold": SIMILARITY_THRESHOLD,
        "results_count": len(results),
        "results": results
    }


@app.get("/v1/debug/find-blog")
async def find_blog(title: str):
    """Search for a specific blog by title keyword."""
    global index_store
    
    if not index_store:
        return {"error": "Index not ready"}
    
    # Search with the title
    retriever = index_store.as_retriever(similarity_top_k=20)
    nodes = retriever.retrieve(title)
    
    # Filter to only blogs
    blog_results = []
    for node in nodes:
        if node.metadata.get("type") == "blog":
            blog_results.append({
                "score": round(node.score, 4),
                "text_preview": node.text[:400],
                "id": node.metadata.get("id")
            })
    
    return {
        "search_title": title,
        "blogs_found": len(blog_results),
        "results": blog_results[:10]
    }


@app.get("/v1/debug/list-blogs")
async def list_blogs(search: str = ""):
    """List all blog titles in the index."""
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
    
    try:
        with httpx.Client(verify=False, timeout=120) as client:
            resp = client.get(
                f"{API_BASE_URL}/blogs/posts",
                params={"PagingDto.PageFilter.Size": 2000, "PagingDto.PageFilter.Skip": 0},
                headers=headers
            )
            if resp.status_code == 200:
                blogs = resp.json().get("data", {}).get("list", [])
                
                # Filter by search term if provided
                if search:
                    blogs = [b for b in blogs if search.lower() in b.get("title", "").lower()]
                
                titles = [{"id": b.get("id"), "title": b.get("title")} for b in blogs[:100]]
                
                return {
                    "total_blogs": len(resp.json().get("data", {}).get("list", [])),
                    "filtered_count": len(titles),
                    "search_term": search,
                    "blogs": titles
                }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    uvicorn.run("llm_server_production:app", host=HOST, port=PORT, reload=False)
