from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas.chat import ChatRequest, QueryRequest, RegenerateRequest
from ..services import legacy

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/query")
async def query(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="No query provided")

    if request.use_rag and not legacy.rag_ready():
        raise HTTPException(
            status_code=503,
            detail="RAG is disabled or not initialized. Set ENABLE_RAG=true and restart, or call /v1/refresh.",
        )

    if request.stream:
        return StreamingResponse(
            legacy.stream_query(request.query, request.session_id, request.use_rag),
            media_type="text/event-stream",
        )

    prompt, topic = await legacy.process_query(request.query, request.session_id, request.use_rag)
    response_text = await legacy.call_llm(prompt)
    memory = legacy.conversation_memory()
    memory[request.session_id].append({"query": request.query, "response": response_text, "topic": topic or request.query})
    return {"query": request.query, "response": response_text, "session_id": request.session_id}


@router.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = request.messages[-1].content
    if request.use_rag and not legacy.rag_ready():
        raise HTTPException(
            status_code=503,
            detail="RAG is disabled or not initialized. Set ENABLE_RAG=true and restart, or call /v1/refresh.",
        )
    if request.stream:
        return StreamingResponse(
            legacy.stream_query(last_message, request.session_id, request.use_rag),
            media_type="text/event-stream",
        )

    prompt, _topic = await legacy.process_query(last_message, request.session_id, request.use_rag)
    response_text = await legacy.call_llm(prompt)
    return {
        "id": "chatcmpl-gamatrain",
        "object": "chat.completion",
        "model": legacy.model_name(),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
    }


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    memory = legacy.conversation_memory()
    if session_id in memory:
        del memory[session_id]
        return {"status": "success", "message": f"Session {session_id} cleared"}
    return {"status": "not_found"}


@router.post("/regenerate")
async def regenerate_response(request: RegenerateRequest):
    memory = legacy.conversation_memory()
    session_id = request.session_id
    if session_id not in memory or not memory[session_id]:
        raise HTTPException(status_code=404, detail="No conversation history found for this session")

    last_entry = memory[session_id][-1]
    last_query = last_entry.get("query", "")
    if not last_query:
        raise HTTPException(status_code=400, detail="No query found in conversation history")
    memory[session_id].pop()

    if request.stream:
        return StreamingResponse(
            legacy.stream_query(last_query, session_id, request.use_rag),
            media_type="text/event-stream",
        )

    prompt, topic = await legacy.process_query(last_query, session_id, request.use_rag)
    response_text = await legacy.call_llm(prompt)
    memory[session_id].append({"query": last_query, "response": response_text, "topic": topic or last_query, "sources": []})
    return {
        "id": "chatcmpl-gamatrain-regenerate",
        "object": "chat.completion",
        "model": legacy.model_name(),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
    }


@router.post("/refresh")
async def refresh_index():
    count = legacy.refresh_index()
    return {"status": "success", "documents_count": count}


@router.get("/debug/search")
async def debug_search(q: str):
    return legacy.debug_search(q)


@router.get("/debug/find-blog")
async def find_blog(title: str):
    return legacy.find_blog(title)


@router.get("/search/blogs")
async def search_blogs(q: str, limit: int = 5):
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query too short")
    try:
        return legacy.search_blogs(q, limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/search/schools")
async def search_schools(q: str, limit: int = 5):
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query too short")
    try:
        return legacy.search_schools(q, limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

