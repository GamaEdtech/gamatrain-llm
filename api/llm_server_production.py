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
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import uvicorn
import httpx
import logging
import json
import asyncio
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
# Option 1: ollama (local - recommended for development)
# Option 2: groq (cloud - FREE and fast)
# Option 3: openrouter (cloud - some free models)
PROVIDER = os.getenv("PROVIDER", "ollama")  # ollama, groq, openrouter

# Ollama Settings (LOCAL - no internet needed)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gamatrain-qwen")  
# Groq Settings (FREE and FAST!)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# OpenRouter Settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
CUSTOM_DOCS_PATH = os.getenv("CUSTOM_DOCS_PATH", "../data/custom_docs.json")

# Gamatrain API for fetching documents
API_BASE_URL = os.getenv("GAMATRAIN_API_URL", "https://api.gamaedtech.com/api/v1")
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
    
    if PROVIDER == "ollama":
        return await call_ollama_api(prompt, max_tokens)
    elif PROVIDER == "groq":
        return await call_groq_api(prompt, max_tokens)
    elif PROVIDER == "openrouter":
        return await call_openrouter_api(prompt, max_tokens)
    else:
        return "Error: No valid provider configured"


async def call_ollama_api(prompt: str, max_tokens: int = 1024):
    """Call local Ollama API."""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return f"Error: {str(e)}"


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
        # For Ollama, use native streaming
        if PROVIDER == "ollama":
            async for chunk in stream_ollama_api(prompt, max_tokens):
                yield chunk
            return
        
        # For cloud APIs (Groq, OpenRouter), simulate streaming
        full_response = await call_llm_api(prompt, max_tokens)
        
        # Simulate streaming by yielding chunks with delay
        words = full_response.split()
        for i, word in enumerate(words):
            token = word + " " if i < len(words) - 1 else word
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            # Add small delay for typing effect (80ms per word)
            await asyncio.sleep(0.08)
        
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


async def stream_ollama_api(prompt: str, max_tokens: int = 1024):
    """Native streaming from Ollama."""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            done = data.get("done", False)
                            yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        logger.error(f"Ollama stream error: {e}")
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
                    slug = post.get("slug", "")  # Extract slug from post
                    
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
                            metadata={
                                "type": "blog",
                                "id": str(post.get("id")),
                                "slug": slug
                            }
                        ))
                logger.info(f"Fetched {len(blogs)} blogs")
    except Exception as e:
        logger.warning(f"Could not fetch blogs: {e}")
    
    # Fetch schools (get 10000 schools with pagination)
    try:
        all_schools = []
        batch_size = 1000
        max_schools = 10000
        
        with httpx.Client(verify=False, timeout=120) as client:
            for skip in range(0, max_schools, batch_size):
                resp = client.get(
                    f"{API_BASE_URL}/schools",
                    params={"PagingDto.PageFilter.Size": batch_size, "PagingDto.PageFilter.Skip": skip},
                    headers=headers
                )
                if resp.status_code == 200:
                    schools = resp.json().get("data", {}).get("list", [])
                    if not schools:
                        break
                    all_schools.extend(schools)
                    logger.info(f"Fetched {len(all_schools)} schools...")
                else:
                    break
        
        for school in all_schools:
            name = school.get("name", "")
            if name and "gamatrain" not in name.lower():
                # Build comprehensive school text
                school_text = f"School Name: {name}"
                if school.get("cityTitle"):
                    school_text += f"\nCity: {school['cityTitle']}"
                if school.get("stateTitle"):
                    school_text += f"\nState/Province: {school['stateTitle']}"
                if school.get("countryTitle"):
                    school_text += f"\nCountry: {school['countryTitle']}"
                if school.get("score"):
                    school_text += f"\nRating Score: {school['score']} out of 5 stars"
                if school.get("slug"):
                    school_text += f"\nURL: /schools/{school['slug']}"
                
                documents.append(Document(
                    text=school_text,
                    metadata={
                        "type": "school",
                        "id": str(school.get("id")),
                        "slug": school.get("slug", "")
                    }
                ))
        logger.info(f"Total schools indexed: {len(all_schools)}")
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


async def rewrite_query_with_context(query: str, history: list) -> tuple:
    """Rewrite follow-up questions to include context. Returns (rewritten_query, is_follow_up)."""
    if not history:
        return query, False
    
    query_lower = query.lower().strip()
    
    # Check for explicit follow-up phrases first
    explicit_followup_phrases = ["tell me more", "explain more", "can you explain", "more details", 
                                  "more information", "go on", "continue", "elaborate"]
    is_explicit = any(phrase in query_lower for phrase in explicit_followup_phrases)
    
    if is_explicit:
        # For explicit follow-ups, use the last topic directly
        last_entry = history[-1]
        last_topic = last_entry.get("topic", last_entry.get("query", ""))
        logger.info(f"Explicit follow-up detected about: {last_topic[:50]}...")
        return f"Explain more about {last_topic}", True
    
    # Check for follow-up indicators (pronouns)
    follow_up_words = ["it", "its", "this", "that", "they", "their", "there", "here"]
    
    # Check if query contains follow-up words as separate words
    query_words = query_lower.replace("?", " ").replace(".", " ").split()
    has_follow_up = any(word in query_words for word in follow_up_words)
    
    if not has_follow_up:
        return query, False
    
    # Get the last topic
    last_entry = history[-1]
    last_topic = last_entry.get("topic", "")
    
    if not last_topic:
        # Try to extract from last query
        last_query = last_entry.get("query", "")
        # Simple extraction: look for capitalized words
        words = last_query.split()
        topic_words = [w for w in words if w[0].isupper() and len(w) > 2] if words else []
        if topic_words:
            last_topic = " ".join(topic_words[:4])  # Max 4 words
    
    if not last_topic:
        return query, False
    
    # Rewrite the query by replacing pronouns with the topic
    rewritten = query
    replacements = [
        (" it ", f" {last_topic} "),
        (" it?", f" {last_topic}?"),
        (" its ", f" {last_topic}'s "),
        ("does it ", f"does {last_topic} "),
        ("is it ", f"is {last_topic} "),
        ("what is it", f"what is {last_topic}"),
    ]
    
    for old, new in replacements:
        if old in query_lower:
            rewritten = query_lower.replace(old, new)
            break
    
    logger.info(f"Query rewritten: '{query}' -> '{rewritten}' (topic: {last_topic})")
    return rewritten, True


def filter_external_links(text: str) -> str:
    """Remove external links from response, keep only gamatrain.com links."""
    import re
    
    # Pattern 1: Match full URLs (http://... or https://...)
    # But NOT gamatrain.com
    external_url_pattern1 = r'https?://(?!(?:www\.)?gamatrain\.com)[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}[^\s\)]*'
    
    # Pattern 2: Match www.example.com (without http)
    external_url_pattern2 = r'www\.(?!gamatrain\.com)[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}[^\s\)]*'
    
    # Remove external URLs (replace with empty string, preserving spaces)
    cleaned_text = re.sub(external_url_pattern1, '', text)
    cleaned_text = re.sub(external_url_pattern2, '', cleaned_text)
    
    # DON'T clean up spaces - that was causing the problem!
    # Just return as-is
    return cleaned_text


def extract_source_links(nodes, base_url: str = "https://gamatrain.com"):
    """Extract blog/school links from RAG nodes."""
    sources = []
    for node in nodes[:3]:  # Top 3 sources
        metadata = node.metadata
        doc_type = metadata.get("type", "")
        
        if doc_type == "blog":
            slug = metadata.get("slug", "")
            blog_id = metadata.get("id", "")
            
            # CRITICAL: Only add source if BOTH slug and blog_id exist
            if slug and blog_id:
                # Extract title from text
                text = node.text
                title = ""
                if "Blog Title:" in text:
                    title = text.split("Blog Title:")[1].split("\n")[0].strip()
                
                # Correct format: /blog/ID/slug
                sources.append({
                    "type": "blog",
                    "title": title or "Blog Post",
                    "url": f"{base_url}/blog/{blog_id}/{slug}",
                    "score": round(node.score, 3)
                })
            else:
                # Log missing metadata to help debug
                logger.warning(f"Blog node missing slug or id: slug={slug}, id={blog_id}")
                
        elif doc_type == "school":
            slug = metadata.get("slug", "")
            
            # CRITICAL: Only add source if slug exists
            if slug:
                # Extract school name
                text = node.text
                name = ""
                if "School Name:" in text:
                    name = text.split("School Name:")[1].split("\n")[0].strip()
                
                sources.append({
                    "type": "school",
                    "title": name or "School",
                    "url": f"{base_url}/schools/{slug}",
                    "score": round(node.score, 3)
                })
            else:
                logger.warning(f"School node missing slug")
    
    return sources


def format_sources_text(sources):
    """Don't format sources as text - frontend will handle all source display."""
    # Return empty string - frontend will use the sources metadata
    # to display the blue-bordered "Related Sources" section
    return ""


# =============================================================================
# Query Processing (with RAG + Memory + Follow-up)
# =============================================================================
async def process_query(query_text: str, session_id: str = "default", use_rag: bool = True):
    """Process query with RAG, memory, and follow-up detection."""
    global index_store, conversation_memory
    
    query_lower = query_text.lower().strip()
    query_normalized = re.sub(r"[^\w\s]", " ", query_lower)
    query_normalized = " ".join(query_normalized.split())
    
    history = conversation_memory[session_id]

    # Detect general greetings (English only)
    general_patterns = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you',
                        'what can you do', 'who are you', 'help', 'thanks', 'thank you',
                        'bye', 'goodbye', 'ok', 'okay', 'yes', 'no', 'sure', "i'm not sure"]
    is_general = any(query_normalized == p or query_normalized.startswith(p + ' ') for p in general_patterns)

    if is_general:
        prompt = f"You are Gamatrain AI, a friendly educational assistant. Respond briefly: {query_text}"
        return prompt, None

    # Check for explicit follow-up phrases
    follow_up_words = ["that", "this", "it", "those", "these", "more", "explain", "elaborate", "details", "different", "same", "similar", "compare", "versus", "vs"]
    follow_up_phrases = ["tell me more", "explain more", "can you explain", "what about", "how about", "also", "continue", "go on"]
    is_follow_up_check = history and (any(word in query_normalized.split() for word in follow_up_words) or any(phrase in query_lower for phrase in follow_up_phrases))
    
    link_keywords = ["link", "links", "source", "sources", "reference", "references"]
    request_keywords = ["send", "share", "give", "provide", "show", "please", "about"]
    explicit_link_request = any(k in query_normalized for k in link_keywords) and any(r in query_normalized for r in request_keywords)
    
    allow_sources = explicit_link_request or (not is_general and not is_follow_up_check)
    
    # Check for explicit follow-up phrases
    explicit_followup_phrases = ["tell me more", "explain more", "can you explain", "more details", 
                                  "more information", "go on", "continue", "elaborate"]
    is_explicit_followup = any(phrase in query_lower for phrase in explicit_followup_phrases)
    
    # For explicit follow-ups, use conversation history directly
    if is_explicit_followup and history:
        last_entry = history[-1]
        last_query = last_entry.get("query", "")
        last_response = last_entry.get("response", "")[:800]
        last_topic = last_entry.get("topic", last_query)
        
        logger.info(f"Explicit follow-up about: {last_topic[:50]}...")
        
        prompt = f"""You are Gamatrain AI, an educational assistant.

The user previously asked: "{last_query}"

You answered: {last_response}

Now the user wants more information about this same topic ({last_topic}).

User's follow-up: {query_text}

Provide additional details and explanations:"""
        
        return prompt, last_topic
    
    # Use LLM to rewrite query if there's conversation history
    search_query = query_text
    is_follow_up = False
    if history:
        search_query, is_follow_up = await rewrite_query_with_context(query_text, history)
    
    logger.info(f"Query: {query_text}, Rewritten: {search_query}, Follow-up: {is_follow_up}")
    
    # Use RAG with the rewritten query
    if allow_sources and use_rag and index_store:
        retriever = index_store.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(search_query)  # Use rewritten query for search
        
        if nodes and max([n.score for n in nodes]) >= SIMILARITY_THRESHOLD:
            context = "\n".join([n.text for n in nodes[:3]])
            
            prompt = f"""Context:
{context}

Question: {query_text}

Answer based on the context above. Be specific and include exact numbers if available (like scores, ratings, etc.). If the answer is not in the context, say so."""
            
            # Extract topic (for blogs and schools)
            topic = ""
            best_node = max(nodes, key=lambda n: n.score)
            if "Blog Title:" in best_node.text:
                topic = best_node.text.split("Blog Title:")[1].split("\n")[0].strip()
            elif "School Name:" in best_node.text:
                topic = best_node.text.split("School Name:")[1].split("\n")[0].strip()
            
            return prompt, topic
    
    # Fallback to direct question
    prompt = f"You are Gamatrain AI, an educational assistant. Answer: {query_text}"
    return prompt, None


async def stream_query(query_text: str, session_id: str = "default", use_rag: bool = True):
    """Stream response with RAG, memory, and source citations."""
    global conversation_memory
    
    # Process query
    prompt, topic = await process_query(query_text, session_id, use_rag)
    
    # Detect if sources should be allowed
    q_low = query_text.lower().strip()
    q_norm = re.sub(r"[^\w\s]", " ", q_low)
    q_norm = " ".join(q_norm.split())
    
    hist = conversation_memory[session_id]
    
    gen_pats = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you',
                'what can you do', 'who are you', 'help', 'thanks', 'thank you',
                'bye', 'goodbye', 'ok', 'okay', 'yes', 'no', 'sure', "i'm not sure"]
    is_gen = any(q_norm == p or q_norm.startswith(p + ' ') for p in gen_pats)
    
    f_words = ["that", "this", "it", "those", "these", "more", "explain", "elaborate", "details", "different", "same", "similar", "compare", "versus", "vs"]
    f_phrases = ["tell me more", "explain more", "can you explain", "what about", "how about", "also", "continue", "go on"]
    is_foll = hist and (any(word in q_norm.split() for word in f_words) or any(phrase in q_low for phrase in f_phrases))
    
    l_kws = ["link", "links", "source", "sources", "reference", "references"]
    r_kws = ["send", "share", "give", "provide", "show", "please", "about"]
    expl_req = any(k in q_norm for k in l_kws) and any(r in q_norm for r in r_kws)
    
    allow_sources = expl_req or (not is_gen and not is_foll)
    
    # Get sources if RAG was used and allowed
    sources = []
    if allow_sources and use_rag and index_store:
        try:
            retriever = index_store.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(query_text)
            if nodes and max([n.score for n in nodes]) >= SIMILARITY_THRESHOLD:
                # Extract sources, but filter out irrelevant ones
                all_sources = extract_source_links(nodes)
                
                # Smart filtering: prioritize blogs over schools for educational content
                blog_sources = [s for s in all_sources if s["type"] == "blog"]
                school_sources = [s for s in all_sources if s["type"] == "school"]
                
                # For "what is gamatrain" type questions, only show company/blog sources
                query_about_gamatrain = "gamatrain" in query_text.lower() and any(
                    word in query_text.lower() for word in ["what", "who", "tell", "about", "explain"]
                )
                
                # For school search queries (looking for specific schools)
                school_search_keywords = ["school", "university", "college", "academy", "institute"]
                is_school_search = any(keyword in query_text.lower() for keyword in school_search_keywords)
                
                if query_about_gamatrain:
                    # Only blogs for gamatrain questions
                    sources = blog_sources
                    logger.info(f"Gamatrain query: showing {len(sources)} blog sources")
                elif is_school_search:
                    # User is looking for schools, show school sources
                    sources = school_sources
                    logger.info(f"School search: showing {len(sources)} school sources")
                elif blog_sources:
                    # For educational content, prefer blogs over schools
                    sources = blog_sources
                    logger.info(f"Educational query: showing {len(sources)} blog sources (filtered out {len(school_sources)} schools)")
                else:
                    # No blogs found, show schools as fallback
                    sources = school_sources
                    logger.info(f"No blogs found, showing {len(sources)} school sources")
        except Exception as e:
            logger.error(f"Error extracting sources: {e}")
            sources = []
    
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
    
    # Add sources at the end if available
    if sources:
        sources_text = format_sources_text(sources)
        # Stream the sources
        for char in sources_text:
            yield f"data: {json.dumps({'token': char, 'done': False})}\n\n"
            await asyncio.sleep(0.01)
        
        full_response += sources_text
        yield f"data: {json.dumps({'token': '', 'done': True, 'sources': sources})}\n\n"
    else:
        # Filter out external links if no sources found
        full_response = filter_external_links(full_response)
    
    # Save to memory
    conversation_memory[session_id].append({
        "query": query_text,
        "response": full_response,
        "topic": topic or query_text,
        "sources": sources if sources else []
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


class RegenerateRequest(BaseModel):
    session_id: str = "default"
    use_rag: bool = True
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
    model_name = OLLAMA_MODEL if PROVIDER == "ollama" else (GROQ_MODEL if PROVIDER == "groq" else OPENROUTER_MODEL)
    return {
        "status": "online",
        "service": "Gamatrain AI (Production)",
        "provider": PROVIDER,
        "model": model_name,
        "rag_enabled": index_store is not None
    }


@app.get("/health")
async def health():
    model_name = OLLAMA_MODEL if PROVIDER == "ollama" else (GROQ_MODEL if PROVIDER == "groq" else OPENROUTER_MODEL)
    return {
        "status": "healthy",
        "provider": PROVIDER,
        "model": model_name,
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


@app.post("/v1/regenerate")
async def regenerate_response(request: RegenerateRequest):
    """Regenerate the last response for a session."""
    global conversation_memory
    
    session_id = request.session_id
    
    # Check if session exists and has history
    if session_id not in conversation_memory or not conversation_memory[session_id]:
        raise HTTPException(status_code=404, detail="No conversation history found for this session")
    
    # Get the last user query
    last_entry = conversation_memory[session_id][-1]
    last_query = last_entry.get("query", "")
    
    if not last_query:
        raise HTTPException(status_code=400, detail="No query found in conversation history")
    
    # Remove the last response from memory
    conversation_memory[session_id].pop()
    
    # Generate new response using the same query
    if request.stream:
        return StreamingResponse(
            stream_query(last_query, session_id, request.use_rag),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    prompt, topic = await process_query(last_query, session_id, request.use_rag)
    response_text = await call_llm_api(prompt, MAX_TOKENS)
    
    # Save to memory
    conversation_memory[session_id].append({
        "query": last_query,
        "response": response_text,
        "topic": topic or last_query,
        "sources": []
    })
    
    return {
        "id": "chatcmpl-gamatrain-regenerate",
        "object": "chat.completion",
        "model": GROQ_MODEL if PROVIDER == "groq" else OPENROUTER_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }]
    }


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


@app.get("/v1/search/blogs")
async def search_blogs(q: str, limit: int = 5):
    """
    Search for blogs related to a query and return links.
    Usage: /v1/search/blogs?q=photosynthesis&limit=5
    """
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query too short")
    
    if not index_store:
        raise HTTPException(status_code=503, detail="Index not ready")
    
    try:
        # Search in RAG index
        retriever = index_store.as_retriever(similarity_top_k=limit * 2)
        nodes = retriever.retrieve(q)
        
        # Filter only blogs
        blog_results = []
        for node in nodes:
            if node.metadata.get("type") == "blog":
                text = node.text
                title = ""
                slug = node.metadata.get("slug", "")
                
                if "Blog Title:" in text:
                    title = text.split("Blog Title:")[1].split("\n")[0].strip()
                
                if slug and title:
                    blog_results.append({
                        "title": title,
                        "url": f"https://gamatrain.com/blog/{slug}",
                        "slug": slug,
                        "relevance_score": round(node.score, 3),
                        "preview": text[:200].replace("Blog Title:", "").strip()
                    })
                
                if len(blog_results) >= limit:
                    break
        
        return {
            "query": q,
            "results_count": len(blog_results),
            "blogs": blog_results
        }
    except Exception as e:
        logger.error(f"Blog search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/search/schools")
async def search_schools(q: str, limit: int = 5):
    """
    Search for schools related to a query and return links.
    Usage: /v1/search/schools?q=MIT&limit=5
    """
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query too short")
    
    if not index_store:
        raise HTTPException(status_code=503, detail="Index not ready")
    
    try:
        # Search in RAG index
        retriever = index_store.as_retriever(similarity_top_k=limit * 2)
        nodes = retriever.retrieve(q)
        
        # Filter only schools
        school_results = []
        for node in nodes:
            if node.metadata.get("type") == "school":
                text = node.text
                name = ""
                slug = node.metadata.get("slug", "")
                
                if "School Name:" in text:
                    name = text.split("School Name:")[1].split("\n")[0].strip()
                
                if slug and name:
                    school_results.append({
                        "name": name,
                        "url": f"https://gamatrain.com/schools/{slug}",
                        "slug": slug,
                        "relevance_score": round(node.score, 3),
                        "info": text[:200].replace("School Name:", "").strip()
                    })
                
                if len(school_results) >= limit:
                    break
        
        return {
            "query": q,
            "results_count": len(school_results),
            "schools": school_results
        }
    except Exception as e:
        logger.error(f"School search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    uvicorn.run("llm_server_production:app", host=HOST, port=PORT, reload=False)
