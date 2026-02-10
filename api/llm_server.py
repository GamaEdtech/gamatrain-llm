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
API_BASE_URL = os.getenv("GAMATRAIN_API_URL", "https://api.gamaedtech.com/api/v1")
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
    
    # Fetch schools (get more schools with detailed info)
    try:
        with httpx.Client(verify=False, timeout=60) as client:
            resp = client.get(
                f"{API_BASE_URL}/schools",
                params={"PagingDto.PageFilter.Size": 1000, "PagingDto.PageFilter.Skip": 0},
                headers=headers
            )
            if resp.status_code == 200:
                schools = resp.json().get("data", {}).get("list", [])
                for school in schools:
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
                            school_text += f"\nRating: {school['score']}/5"
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
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))  # Lowered from 0.75


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
        elif doc_type == "school":
            slug = metadata.get("slug", "")
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
    
    return sources


def format_sources_text(sources):
    """Format sources as readable text for LLM response."""
    if not sources:
        return ""
    
    text = "\n\n" + "="*60 + "\n"
    text += "📚Related Sources\n"
    text += "="*60 + "\n\n"
    
    for i, source in enumerate(sources, 1):
        if source["type"] == "blog":
            text += f"{i}. 📝 {source['title']}\n"
            text += f"   🔗 {source['url']}\n\n"
        elif source["type"] == "school":
            text += f"{i}. 🏫 {source['title']}\n"
            text += f"   🔗 {source['url']}\n\n"
    
    return text


async def stream_query(query_text: str, session_id: str = "default", use_rag: bool = True):
    """Stream response token by token using Server-Sent Events with source citations."""
    import json
    import asyncio
    import re

    query_lower = query_text.lower().strip()
    query_normalized = re.sub(r"[^\w\s]", " ", query_lower)
    query_normalized = " ".join(query_normalized.split())
    nodes = []  # Initialize nodes to avoid reference error
    sources = []  # Store source links
    
    # Detect general/greeting queries that don't need RAG
    general_patterns = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you',
                        'what can you do', 'who are you', 'help', 'thanks', 'thank you',
                        'bye', 'goodbye', 'ok', 'okay', 'yes', 'no', 'sure', "i'm not sure",
                        'سلام', 'درود', 'صبح بخیر', 'عصر بخیر', 'شب بخیر', 'چطوری', 'حالت چطوره',
                        'خسته نباشی', 'مرسی', 'ممنون', 'خداحافظ']
    
    is_general = any(query_normalized == p or query_normalized.startswith(p + ' ')
                     for p in general_patterns)
    
    # Get conversation history
    history = conversation_memory[session_id]
    
    # Detect follow-up questions - expanded patterns
    follow_up_words = ["that", "this", "it", "those", "these", "more", "explain", "elaborate", "details", "different", "same", "similar", "compare", "versus", "vs"]
    follow_up_phrases = ["tell me more", "explain more", "can you explain", "what about", "how about", 
                         "and what", "also", "continue", "go on", "more details", "more information",
                         "how is it", "what is the difference", "is it the same", "compared to",
                         "how does it", "why is it", "when is it", "where is it", "different from"]
    
    is_follow_up = history and (
        any(word in query_normalized.split() for word in follow_up_words) or
        any(phrase in query_lower for phrase in follow_up_phrases)
    )

    link_keywords = ["link", "links", "source", "sources", "reference", "references", "منبع", "منابع", "لینک", "لینکها", "لینکها", "آدرس", "رفرنس"]
    request_keywords = ["send", "share", "give", "provide", "show", "please", "about", "درباره", "بده", "ارسال", "بفرست", "میخوام", "میخوام", "میخواهم", "میخواهم", "لطفا"]
    explicit_link_request = any(k in query_normalized for k in link_keywords) and any(r in query_normalized for r in request_keywords)
    allow_sources = explicit_link_request or (not is_general and not is_follow_up)
    
    try:
        # First, check if the answer might be in recent conversation history
        context_from_history = ""
        if history:
            # Search for relevant info in previous responses
            query_keywords = set(query_lower.replace("?", "").replace(".", "").split())
            query_keywords -= {"is", "are", "the", "a", "an", "does", "do", "has", "have", "what", "how", "can", "you", "tell", "me", "about", "more", "please"}
            
            for entry in history[-3:]:  # Check last 3 turns
                prev_response = entry.get("response", "").lower()
                # If query keywords appear in previous response, use it as context
                matches = sum(1 for kw in query_keywords if kw in prev_response and len(kw) > 2)
                if matches >= 1:
                    context_from_history = entry.get("response", "")[:800]
                    logger.info(f"Found relevant context in history: {matches} keyword matches")
                    break
        
        # Check for follow-up phrases FIRST before general check
        follow_up_phrases_check = ["tell me more", "explain more", "can you explain", "more details", 
                                   "more information", "go on", "continue", "elaborate"]
        is_explicit_followup = any(phrase in query_lower for phrase in follow_up_phrases_check)
        
        if is_explicit_followup and history:
            # This is definitely a follow-up, use last conversation directly
            last_entry = history[-1]
            last_query = last_entry.get("query", "")
            last_response = last_entry.get("response", "")[:800]
            last_topic = last_entry.get("topic", last_query)
            
            logger.info(f"Explicit follow-up detected. Topic: '{last_topic[:50]}...', Last query: '{last_query[:50]}...'")
            
            # Build a clear prompt that maintains context
            prompt = f"""You are Gamatrain AI, an educational assistant.

The user previously asked: "{last_query}"

You answered: {last_response}

Now the user wants to know more about this same topic. Provide additional details, examples, or explanations about {last_topic if last_topic else "the topic"}.

User's follow-up question: {query_text}

Continue explaining in detail:"""
            
            # Stream directly without RAG
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "num_predict": 1024,
                            "temperature": 0.7
                        }
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
                                    conversation_memory[session_id].append({
                                        "query": query_text,
                                        "response": full_response,
                                        "topic": last_topic
                                    })
                                    if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS:
                                        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS:]
                            except json.JSONDecodeError:
                                continue
            return
        
        if is_general and not is_follow_up:
            # Use direct LLM for greetings/general chat
            prompt = f"You are Gamatrain AI, a friendly educational assistant. Respond briefly and helpfully to: {query_text}\n\nIMPORTANT: Do NOT include any external URLs or links."
        elif context_from_history and not is_general:
            # Answer based on conversation history
            logger.info("Using conversation history to answer")
            prompt = f"""You are Gamatrain AI, an educational assistant.

Based on our previous conversation, here is relevant information:
{context_from_history}

Now answer this question using the information above: {query_text}

If the information above doesn't contain the answer, say so honestly."""
        elif use_rag and index_store:
            enhanced_query = query_text
            
            # Handle follow-up with conversation context
            if is_follow_up and history:
                last_entry = history[-1]
                last_topic = last_entry.get("topic", "")
                last_query = last_entry.get("query", "")
                last_response = last_entry.get("response", "")[:600]  # Limit context size
                
                # If no topic saved, use the last query as context
                if not last_topic:
                    last_topic = last_query
                
                logger.info(f"Follow-up detected. Topic: '{last_topic}', Last query: '{last_query}'")
                
                # For ALL follow-up questions, use conversation context directly
                # Use simple prompt format to avoid model echoing the prompt
                prompt = f"""Based on this context: {last_response}

{query_text}"""
                
                # Skip RAG retrieval, go directly to LLM
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": MODEL_NAME,
                            "prompt": prompt,
                            "stream": True,
                            "options": {
                                "num_predict": 1024
                            }
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
                                        # Preserve topic from previous turn
                                        conversation_memory[session_id].append({
                                            "query": query_text,
                                            "response": full_response,
                                            "topic": last_topic
                                        })
                                        
                                        if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS:
                                            conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS:]
                                except json.JSONDecodeError:
                                    continue
                return  # Exit after handling follow-up
            
            # Retrieve context
            retriever = index_store.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(enhanced_query)
            
            # Extract source links from nodes
            if nodes and max([n.score for n in nodes]) >= SIMILARITY_THRESHOLD:
                sources = extract_source_links(nodes)
                if not allow_sources:
                    sources = []
            
            if not nodes or max([n.score for n in nodes]) < SIMILARITY_THRESHOLD:
                # For follow-ups with no RAG match, use conversation history
                if is_follow_up and history:
                    last_entry = history[-1]
                    prompt = f"""You are Gamatrain AI, an educational assistant.

Previous conversation:
User asked: {last_entry.get('query', '')}
You answered: {last_entry.get('response', '')[:500]}

Now the user asks: {query_text}

Please continue the conversation and provide more details based on your previous answer."""
                else:
                    prompt = f"You are Gamatrain AI, an educational assistant. Answer this question: {query_text}\n\nIMPORTANT: Do NOT include any external URLs or links unless they are from gamatrain.com."
            else:
                # Build context for streaming
                context = "\n".join([n.text for n in nodes[:3]])
                
                # Include conversation history in prompt for follow-ups
                history_context = ""
                if is_follow_up and history:
                    last_entry = history[-1]
                    history_context = f"""
Previous conversation:
User asked: {last_entry.get('query', '')}
You answered: {last_entry.get('response', '')[:300]}

"""
                
                prompt = f"""Context information is below.
---------------------
{context}
---------------------
{history_context}IMPORTANT: Answer the question using the context above and conversation history.
If the answer is NOT in the context, say 'I don't have information about that in my knowledge base.'
Do NOT make up or invent any information.
Do NOT include external links or URLs unless they are from gamatrain.com.

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
                    "stream": True,
                    "options": {
                        "num_predict": 1024
                    }
                }
            ) as response:
                full_response = ""
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            done = data.get("done", False)
                            
                            # Filter external links from token
                            token = filter_external_links(token)
                            
                            full_response += token
                            
                            yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
                            
                            if done:
                                # Add source links at the end if available
                                if sources:
                                    sources_text = format_sources_text(sources)
                                    # Stream the sources as additional tokens
                                    for char in sources_text:
                                        yield f"data: {json.dumps({'token': char, 'done': False})}\n\n"
                                        await asyncio.sleep(0.01)  # Small delay for smooth streaming
                                    
                                    full_response += sources_text
                                    
                                    # Send final message with sources metadata
                                    yield f"data: {json.dumps({
                                        'token': '', 
                                        'done': True, 
                                        'sources': sources,
                                        'has_sources': True
                                    })}\n\n"
                                else:
                                    # Filter out external links from response
                                    full_response = filter_external_links(full_response)
                                    yield f"data: {json.dumps({'token': '', 'done': True, 'has_sources': False})}\n\n"
                                
                                # Save to memory
                                topic = ""
                                if use_rag and nodes:
                                    best_node = max(nodes, key=lambda n: n.score)
                                    if "Blog Title:" in best_node.text:
                                        topic = best_node.text.split("Blog Title:")[1].split("\n")[0].strip()
                                elif is_follow_up and history:
                                    # Preserve topic from previous turn
                                    topic = history[-1].get("topic", "")
                                
                                conversation_memory[session_id].append({
                                    "query": query_text,
                                    "response": full_response,
                                    "topic": topic,
                                    "sources": sources if sources else []
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
    
    # Detect general/greeting queries that don't need RAG
    general_patterns = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you',
                        'what can you do', 'who are you', 'help', 'thanks', 'thank you',
                        'bye', 'goodbye', 'ok', 'okay', 'yes', 'no', 'sure', "i'm not sure",
                        'سلام', 'درود', 'صبح بخیر', 'عصر بخیر', 'شب بخیر', 'چطوری', 'حالت چطوره',
                        'خسته نباشی', 'مرسی', 'ممنون', 'خداحافظ']
    query_lower = query_text.lower().strip()
    query_normalized = re.sub(r"[^\w\s]", " ", query_lower)
    query_normalized = " ".join(query_normalized.split())
    
    # Build context from conversation history
    history = conversation_memory[session_id]
    
    # Detect follow-up questions - expanded patterns
    follow_up_words = ["that", "this", "it", "those", "these", "more", "explain", "elaborate", "details", "different", "same", "similar", "compare", "versus", "vs"]
    follow_up_phrases = ["tell me more", "explain more", "can you explain", "what about", "how about", 
                         "and what", "also", "continue", "go on", "more details", "more information",
                         "how is it", "what is the difference", "is it the same", "compared to",
                         "how does it", "why is it", "when is it", "where is it", "different from"]
    
    is_follow_up = history and (
        any(word in query_normalized.split() for word in follow_up_words) or
        any(phrase in query_lower for phrase in follow_up_phrases)
    )
    
    is_general = any(query_normalized == p or query_normalized.startswith(p + ' ') 
                     for p in general_patterns)
    
    if is_general and not is_follow_up:
        # Use direct LLM for greetings/general chat
        logger.info(f"General query detected, using direct LLM")
        response = llm.complete(f"You are Gamatrain AI, a friendly educational assistant. Respond briefly and helpfully to: {query_text}")
        return {
            "response": str(response),
            "confidence": "direct",
            "max_score": 1.0
        }
    
    # Check if the answer might be in recent conversation history
    context_from_history = ""
    if history:
        query_keywords = set(query_lower.replace("?", "").replace(".", "").split())
        query_keywords -= {"is", "are", "the", "a", "an", "does", "do", "has", "have", "what", "how", "can", "you", "tell", "me", "about"}
        
        for entry in history[-3:]:
            prev_response = entry.get("response", "").lower()
            matches = sum(1 for kw in query_keywords if kw in prev_response and len(kw) > 2)
            if matches >= 1:
                context_from_history = entry.get("response", "")[:800]
                logger.info(f"Found relevant context in history: {matches} keyword matches")
                break
    
    if context_from_history and not is_general:
        logger.info("Using conversation history to answer")
        prompt = f"""You are Gamatrain AI, an educational assistant.

Based on our previous conversation, here is relevant information:
{context_from_history}

Now answer this question using the information above: {query_text}

If the information above doesn't contain the answer, say so honestly."""
        response = llm.complete(prompt)
        
        # Save to memory
        conversation_memory[session_id].append({
            "query": query_text,
            "response": str(response),
            "topic": history[-1].get("topic", "") if history else ""
        })
        if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS:
            conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS:]
        
        return {
            "response": str(response),
            "confidence": "from_history",
            "max_score": 1.0
        }
    
    # Detect follow-up questions
    enhanced_query = query_text
    
    if is_follow_up and history:
        last_entry = history[-1]
        last_topic = last_entry.get("topic", "")
        last_query = last_entry.get("query", "")
        last_response = last_entry.get("response", "")[:500]
        
        # If no topic saved, use the last query as context
        if not last_topic:
            last_topic = last_query
        
        logger.info(f"Follow-up detected. Topic: '{last_topic}', Last query: '{last_query}'")
        
        # For "explain more" type queries, use conversation context directly
        if any(phrase in query_lower for phrase in ["explain more", "tell me more", "more details", "can you explain"]):
            prompt = f"""You are Gamatrain AI, an educational assistant.

Previous conversation:
User asked: {last_query}
You answered: {last_response}

Now the user asks: {query_text}

Please provide more details and expand on your previous answer. Be helpful and informative."""
            response = llm.complete(prompt)
            
            # Save to conversation memory
            conversation_memory[session_id].append({
                "query": query_text,
                "enhanced_query": query_text,
                "response": str(response),
                "topic": last_topic
            })
            
            if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS:
                conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS:]
            
            return {
                "response": str(response),
                "confidence": "follow_up",
                "max_score": 1.0
            }
        
        # For other follow-ups, try to enhance the query
        enhanced_query = query_lower
        for word in ["that", "this", "it", "those", "these"]:
            enhanced_query = enhanced_query.replace(f" {word} ", f" {last_topic} ")
            enhanced_query = enhanced_query.replace(f" {word}?", f" {last_topic}?")
            enhanced_query = enhanced_query.replace(f" {word}.", f" {last_topic}.")
            if enhanced_query.endswith(f" {word}"):
                enhanced_query = enhanced_query[:-len(word)-1] + f" {last_topic}"
        
        logger.info(f"Enhanced query: '{enhanced_query}'")
    
    # Get retriever to check similarity scores
    retriever = index_store.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(enhanced_query)
    
    if not nodes:
        # Fallback to direct LLM
        response = llm.complete(query_text)
        return {
            "response": str(response),
            "confidence": "direct",
            "max_score": 0
        }
    
    max_score = max([n.score for n in nodes])
    
    # Check if score meets threshold
    if max_score < SIMILARITY_THRESHOLD:
        logger.info(f"Low similarity score ({max_score:.2f}), falling back to direct LLM")
        # For follow-ups with no RAG match, use conversation history
        if is_follow_up and history:
            last_entry = history[-1]
            prompt = f"""You are Gamatrain AI, an educational assistant.

Previous conversation:
User asked: {last_entry.get('query', '')}
You answered: {last_entry.get('response', '')[:500]}

Now the user asks: {query_text}

Please continue the conversation and provide more details based on your previous answer."""
            response = llm.complete(prompt)
        else:
            response = llm.complete(f"You are Gamatrain AI, an educational assistant. Answer this question: {query_text}")
        return {
            "response": str(response),
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
    
    # Preserve topic from previous turn if no new topic found
    if not topic and is_follow_up and history:
        topic = history[-1].get("topic", "")
    
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

class RegenerateRequest(BaseModel):
    session_id: str = "default"
    use_rag: bool = True
    stream: bool = False

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
    
    # Non-streaming response (fallback)
    result = query_with_threshold(last_query, session_id)
    
    return {
        "response": result["response"],
        "sources": result.get("sources", []),
        "session_id": session_id
    }


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

@app.post("/v1/query/html")
async def query_html(request: QueryRequest):
    """
    Query endpoint that returns HTML-formatted response with clickable links.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    logger.info(f"HTML Query: {request.query[:50]}...")
    
    try:
        if request.use_rag and query_engine:
            result = query_with_threshold(request.query, request.session_id)
            response_text = result["response"]
            
            # Get sources with suppression logic
            sources = []
            
            # Detect if sources should be allowed
            q_lower = request.query.lower().strip()
            q_norm = re.sub(r"[^\w\s]", " ", q_lower)
            q_norm = " ".join(q_norm.split())
            
            gen_patterns = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you',
                            'what can you do', 'who are you', 'help', 'thanks', 'thank you',
                            'bye', 'goodbye', 'ok', 'okay', 'yes', 'no', 'sure', "i'm not sure",
                            'سلام', 'درود', 'صبح بخیر', 'عصر بخیر', 'شب بخیر', 'چطوری', 'حالت چطوره',
                            'خسته نباشی', 'مرسی', 'ممنون', 'خداحافظ']
            
            is_gen = any(q_norm == p or q_norm.startswith(p + ' ') for p in gen_patterns)
            
            # Check for follow-up indicators
            hist = conversation_memory[request.session_id]
            follow_words = ["that", "this", "it", "those", "these", "more", "explain", "elaborate", "details", "different", "same", "similar", "compare", "versus", "vs"]
            follow_phrases = ["tell me more", "explain more", "can you explain", "what about", "how about", "also", "continue", "go on"]
            is_foll = hist and (any(word in q_lower.split() for word in follow_words) or any(phrase in q_lower for phrase in follow_phrases))
            
            link_kws = ["link", "links", "source", "sources", "reference", "references", "منبع", "منابع", "لینک", "لینکها", "آدرس", "رفرنس"]
            req_kws = ["send", "share", "give", "provide", "show", "please", "about", "درباره", "بده", "ارسال", "بفرست", "میخوام", "لطفا"]
            expl_link_req = any(k in q_norm for k in link_kws) and any(r in q_norm for r in req_kws)
            
            allow_srcs = expl_link_req or (not is_gen and not is_foll)

            if allow_srcs and index_store:
                retriever = index_store.as_retriever(similarity_top_k=3)
                nodes = retriever.retrieve(request.query)
                if nodes and max([n.score for n in nodes]) >= SIMILARITY_THRESHOLD:
                    sources = extract_source_links(nodes)
            
            # Format as HTML
            html_response = f'<div class="response-text">{response_text}</div>'
            
            return {
                "query": request.query,
                "response": response_text,
                "html": html_response,
                "sources": sources,
                "confidence": result["confidence"],
                "session_id": request.session_id
            }
        else:
            response = llm.complete(request.query)
            return {
                "query": request.query,
                "response": str(response),
                "html": f'<div class="response-text">{str(response)}</div>',
                "sources": [],
                "confidence": "direct"
            }
    except Exception as e:
        logger.error(f"HTML Query error: {e}")
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
