"""
Gamatrain LLM API Server with RAG (FastAPI + Ollama + LlamaIndex)
=================================================================

Features:
- RAG-powered responses using LlamaIndex
- Direct LLM chat without RAG
- OpenAI-compatible API endpoints
- Auto-loads documents from Gamatrain API

Requirements:
    pip install fastapi uvicorn httpx llama-index llama-index-llms-ollama llama-index-embeddings-huggingface
"""

import os
import uvicorn
import httpx
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "gamatrain-qwen")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
CUSTOM_DOCS_PATH = os.getenv("CUSTOM_DOCS_PATH", "../data/custom_docs.json")

# Gamatrain API
API_BASE_URL = os.getenv("GAMATRAIN_API_URL", "https://185.204.170.142/api/v1")
AUTH_TOKEN = os.getenv("GAMATRAIN_AUTH_TOKEN", "")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GamatrainAPI")

# Global RAG components
query_engine = None
index_store = None
llm = None

# Conversation memory - stores recent context per session
from collections import defaultdict
conversation_memory = defaultdict(list)
MAX_MEMORY_TURNS = 5  # Keep last 5 Q&A pairs


# =============================================================================
# RAG Setup
# =============================================================================
def setup_llm():
    """Initialize LLM and embedding model."""
    global llm
    logger.info(f"Setting up LLM: {MODEL_NAME}")
    
    llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
    
    return llm


def fetch_documents():
    """Fetch documents from Gamatrain API and custom docs file."""
    documents = []
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
    
    # Add Gamatrain company info
    gamatrain_info = """
    Gamatrain is an educational technology company (EdTech) that provides AI-powered learning tools.
    Gamatrain AI is an intelligent educational assistant created by Gamatrain's development team.
    Gamatrain helps students learn through personalized education and smart tutoring.
    """
    documents.append(Document(text=gamatrain_info, metadata={"type": "company", "id": "gamatrain"}))
    
    # Load custom documents from JSON file
    if os.path.exists(CUSTOM_DOCS_PATH):
        try:
            import json
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
    
    # Fetch ALL blogs with full content
    try:
        with httpx.Client(verify=False, timeout=120) as client:
            resp = client.get(
                f"{API_BASE_URL}/blogs/posts",
                params={
                    "PagingDto.PageFilter.Size": 2000,  # Get all blogs (1826+)
                    "PagingDto.PageFilter.Skip": 0,
                    "PagingDto.PageFilter.ReturnTotalRecordsCount": "true"
                },
                headers=headers
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                blogs = data.get("list", [])
                total = data.get("totalRecordsCount", len(blogs))
                
                for post in blogs:
                    title = post.get("title", "")
                    summary = post.get("summary", "")
                    slug = post.get("slug", "")
                    content = post.get("content", "")  # Full content if available
                    
                    # Build comprehensive blog text
                    blog_text = f"Blog Title: {title}\n"
                    if summary:
                        blog_text += f"Summary: {summary}\n"
                    if content:
                        # Strip HTML tags if present
                        import re
                        clean_content = re.sub(r'<[^>]+>', '', content)
                        blog_text += f"Content: {clean_content}\n"
                    if slug:
                        blog_text += f"URL: /blog/{slug}"
                    
                    if title:
                        documents.append(Document(
                            text=blog_text,
                            metadata={
                                "type": "blog",
                                "id": str(post.get("id")),
                                "slug": slug
                            }
                        ))
                logger.info(f"Fetched {len(blogs)}/{total} blogs")
    except Exception as e:
        logger.warning(f"Could not fetch blogs: {e}")
    
    # Fetch schools
    try:
        with httpx.Client(verify=False, timeout=30) as client:
            resp = client.get(
                f"{API_BASE_URL}/schools",
                params={"PagingDto.PageFilter.Size": 50, "PagingDto.PageFilter.Skip": 0},
                headers=headers
            )
            if resp.status_code == 200:
                schools = resp.json().get("data", {}).get("list", [])
                for school in schools[:30]:
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
    
    # Custom QA prompt to reduce hallucination
    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "IMPORTANT: Answer the question ONLY using the context above. "
        "If the answer is NOT in the context, say 'I don't have information about that in my knowledge base.' "
        "Do NOT make up or invent any information.\n\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    
    # Try to load existing index
    if os.path.exists(os.path.join(STORAGE_DIR, "docstore.json")):
        try:
            logger.info("Loading existing index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            index_store = load_index_from_storage(storage_context)
            query_engine = index_store.as_query_engine(
                similarity_top_k=3,
                response_mode="compact",
                text_qa_template=qa_prompt,
            )
            logger.info("Index loaded successfully")
            return query_engine
        except Exception as e:
            logger.warning(f"Could not load index: {e}, rebuilding...")
    
    # Build new index
    logger.info(f"Building new index with {len(documents)} documents...")
    index_store = VectorStoreIndex.from_documents(documents)
    
    # Persist index
    index_store.storage_context.persist(persist_dir=STORAGE_DIR)
    logger.info(f"Index saved to {STORAGE_DIR}")
    
    query_engine = index_store.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        text_qa_template=qa_prompt,
    )
    
    return query_engine


# Similarity threshold for RAG
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))


async def stream_query(query_text: str, session_id: str = "default", use_rag: bool = True):
    """Stream response token by token using Server-Sent Events."""
    import json
    import asyncio
    
    try:
        if use_rag and index_store:
            # Get context from RAG
            history = conversation_memory[session_id]
            enhanced_query = query_text
            
            # Handle follow-up
            follow_up_words = ["that", "this", "it", "those", "these"]
            query_lower = query_text.lower()
            is_follow_up = history and any(word in query_lower.split() for word in follow_up_words)
            
            if is_follow_up:
                last_topic = history[-1].get("topic", "")
                if last_topic:
                    enhanced_query = query_lower
                    for word in follow_up_words:
                        enhanced_query = enhanced_query.replace(f" {word} ", f" {last_topic} ")
            
            # Retrieve context
            retriever = index_store.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(enhanced_query)
            
            if not nodes or max([n.score for n in nodes]) < SIMILARITY_THRESHOLD:
                no_info_msg = "I don't have information about that in my knowledge base."
                yield f"data: {json.dumps({'token': no_info_msg, 'done': True})}\n\n"
                return
            
            # Build context for streaming
            context = "\n".join([n.text for n in nodes[:3]])
            
            prompt = f"""Context information is below.
---------------------
{context}
---------------------
IMPORTANT: Answer the question ONLY using the context above.
If the answer is NOT in the context, say 'I don't have information about that in my knowledge base.'
Do NOT make up or invent any information.

Question: {enhanced_query}
Answer: """
            
        else:
            prompt = query_text
        
        # Stream from Ollama
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": True
                }
            ) as response:
                full_response = ""
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            done = data.get("done", False)
                            full_response += token
                            
                            yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
                            
                            if done:
                                # Save to memory
                                topic = ""
                                if use_rag and nodes:
                                    best_node = max(nodes, key=lambda n: n.score)
                                    if "Blog Title:" in best_node.text:
                                        topic = best_node.text.split("Blog Title:")[1].split("\n")[0].strip()
                                
                                conversation_memory[session_id].append({
                                    "query": query_text,
                                    "response": full_response,
                                    "topic": topic
                                })
                                
                                if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS:
                                    conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS:]
                        except json.JSONDecodeError:
                            continue
                            
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

def query_with_threshold(query_text: str, session_id: str = "default"):
    """Query with similarity threshold check, content verification, and conversation memory."""
    global index_store, llm, query_engine, conversation_memory
    
    # Build context from conversation history
    history = conversation_memory[session_id]
    
    # Detect follow-up questions
    enhanced_query = query_text
    follow_up_words = ["that", "this", "it", "those", "these"]
    query_lower = query_text.lower()
    
    is_follow_up = history and any(word in query_lower.split() for word in follow_up_words)
    
    if is_follow_up:
        last_topic = history[-1].get("topic", "")
        if last_topic:
            # Replace pronouns with actual topic
            enhanced_query = query_lower
            for word in follow_up_words:
                # Replace whole word only
                enhanced_query = enhanced_query.replace(f" {word} ", f" {last_topic} ")
                enhanced_query = enhanced_query.replace(f" {word}?", f" {last_topic}?")
                enhanced_query = enhanced_query.replace(f" {word}.", f" {last_topic}.")
                if enhanced_query.endswith(f" {word}"):
                    enhanced_query = enhanced_query[:-len(word)-1] + f" {last_topic}"
            
            logger.info(f"Follow-up detected. Topic: '{last_topic}', Enhanced: '{enhanced_query}'")
    
    # Get retriever to check similarity scores
    retriever = index_store.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(enhanced_query)
    
    if not nodes:
        return {
            "response": "I don't have information about that in my knowledge base.",
            "confidence": "low",
            "max_score": 0
        }
    
    max_score = max([n.score for n in nodes])
    
    # Check if score meets threshold
    if max_score < SIMILARITY_THRESHOLD:
        logger.info(f"Low similarity score ({max_score:.2f}), returning 'don't know'")
        return {
            "response": "I don't have information about that in my knowledge base.",
            "confidence": "low",
            "max_score": max_score
        }
    
    # Extract key terms from original query - look for specific entity names
    import re
    common_words = {'tell', 'me', 'about', 'what', 'is', 'the', 'who', 'where', 'how', 'can', 'you', 'please', 'school', 'city', 'country', 'that', 'this', 'more', 'also', 'common', 'mistakes', 'questions', 'important', 'are'}
    specific_terms = re.findall(r'\b[A-Z][A-Za-z0-9]+\b', query_text)
    specific_terms = [t for t in specific_terms if t.lower() not in common_words]
    
    if specific_terms:
        context_text = " ".join([n.text.lower() for n in nodes])
        missing_terms = [t for t in specific_terms if t.lower() not in context_text]
        
        if missing_terms:
            logger.info(f"Query mentions '{missing_terms}' not found in context")
            return {
                "response": f"I don't have specific information about {', '.join(missing_terms)} in my knowledge base.",
                "confidence": "low",
                "max_score": max_score
            }
    
    # Good match - use RAG with enhanced query
    response = query_engine.query(enhanced_query)
    response_text = str(response)
    
    # Extract topic for future reference
    topic = ""
    if nodes:
        best_node = max(nodes, key=lambda n: n.score)
        if "Blog Title:" in best_node.text:
            topic = best_node.text.split("Blog Title:")[1].split("\n")[0].strip()
    
    # Save to conversation memory
    conversation_memory[session_id].append({
        "query": query_text,
        "enhanced_query": enhanced_query,
        "response": response_text,
        "topic": topic
    })
    
    # Trim memory if too long
    if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS:]
    
    return {
        "response": response_text,
        "confidence": "high" if max_score > 0.85 else "medium",
        "max_score": max_score
    }


# =============================================================================
# FastAPI App
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG on startup."""
    global query_engine, llm
    
    logger.info("Starting Gamatrain AI Server...")
    llm = setup_llm()
    documents = fetch_documents()
    query_engine = build_index(documents)
    logger.info("Server ready!")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="Gamatrain AI API",
    description="RAG-powered educational AI assistant",
    version="2.0",
    lifespan=lifespan
)

# CORS
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
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7
    use_rag: bool = True
    session_id: str = "default"  # For conversation memory

class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True
    session_id: str = "default"
    stream: bool = False  # Enable streaming

class RefreshRequest(BaseModel):
    force: bool = False

class AddDocumentRequest(BaseModel):
    text: str
    doc_type: str = "custom"  # blog, school, faq, custom
    metadata: dict = {}


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Gamatrain AI Gateway",
        "model": MODEL_NAME,
        "rag_enabled": query_engine is not None
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "rag_ready": query_engine is not None,
        "llm_ready": llm is not None
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat endpoint with optional RAG.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    last_message = request.messages[-1].content
    logger.info(f"Chat request: {last_message[:50]}... (RAG: {request.use_rag})")
    
    try:
        if request.use_rag and query_engine:
            # Use RAG with threshold
            result = query_with_threshold(last_message)
            content = result["response"]
            confidence = result["confidence"]
        else:
            # Direct LLM call
            response = llm.complete(last_message)
            content = str(response)
            confidence = "direct"
        
        return {
            "id": "chatcmpl-gamatrain",
            "object": "chat.completion",
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "confidence": confidence,
            "usage": {
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(last_message.split()) + len(content.split())
            }
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/query")
async def query(request: QueryRequest):
    """
    Simple query endpoint for RAG with confidence score and conversation memory.
    Supports streaming responses.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    logger.info(f"Query: {request.query[:50]}... (session: {request.session_id}, stream: {request.stream})")
    
    # Streaming response
    if request.stream:
        return StreamingResponse(
            stream_query(request.query, request.session_id, request.use_rag),
            media_type="text/event-stream"
        )
    
    # Normal response
    try:
        if request.use_rag and query_engine:
            result = query_with_threshold(request.query, request.session_id)
            return {
                "query": request.query,
                "response": result["response"],
                "confidence": result["confidence"],
                "similarity_score": result["max_score"],
                "session_id": request.session_id,
                "source": "rag"
            }
        else:
            response = llm.complete(request.query)
            return {
                "query": request.query,
                "response": str(response),
                "confidence": "direct",
                "source": "llm"
            }
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/refresh")
async def refresh_index(request: RefreshRequest):
    """
    Refresh RAG index with latest data from API.
    """
    global query_engine, index_store
    
    logger.info("Refreshing index...")
    
    try:
        documents = fetch_documents()
        
        if request.force:
            # Delete existing storage
            import shutil
            if os.path.exists(STORAGE_DIR):
                shutil.rmtree(STORAGE_DIR)
                os.makedirs(STORAGE_DIR)
        
        query_engine = build_index(documents)
        
        return {
            "status": "success",
            "documents_count": len(documents),
            "message": "Index refreshed successfully"
        }
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/documents/add")
async def add_document(request: AddDocumentRequest):
    """
    Add a new document to RAG index.
    """
    global index_store, query_engine
    
    if not request.text or len(request.text) < 10:
        raise HTTPException(status_code=400, detail="Document text too short")
    
    try:
        # Create document
        metadata = {"type": request.doc_type, **request.metadata}
        doc = Document(text=request.text, metadata=metadata)
        
        # Insert into existing index
        index_store.insert(doc)
        
        # Persist
        index_store.storage_context.persist(persist_dir=STORAGE_DIR)
        
        logger.info(f"Added document: {request.text[:50]}...")
        
        return {
            "status": "success",
            "message": "Document added successfully",
            "doc_type": request.doc_type
        }
    except Exception as e:
        logger.error(f"Add document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents/count")
async def get_document_count():
    """Get total document count in index."""
    try:
        count = len(index_store.docstore.docs)
        return {"count": count}
    except:
        return {"count": 0}


@app.delete("/v1/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    global conversation_memory
    if session_id in conversation_memory:
        del conversation_memory[session_id]
        return {"status": "success", "message": f"Session {session_id} cleared"}
    return {"status": "not_found", "message": f"Session {session_id} not found"}


@app.get("/v1/stream")
async def stream_get(query: str, session_id: str = "default"):
    """
    Streaming endpoint (GET) - easier to test in browser.
    Usage: /v1/stream?query=What is Gamatrain?
    """
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    logger.info(f"Stream query: {query[:50]}...")
    
    return StreamingResponse(
        stream_query(query, session_id, use_rag=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    uvicorn.run("llm_server:app", host=HOST, port=PORT, reload=True)
