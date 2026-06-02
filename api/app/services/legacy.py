import importlib
from typing import Any, Dict, Tuple

from ..core.config import MAX_TOKENS, MODEL


def _legacy() -> Any:
    """Load the existing monolith module regardless of run context."""
    try:
        return importlib.import_module("api.llm_server_production")
    except ModuleNotFoundError:
        return importlib.import_module("llm_server_production")


def model_name() -> str:
    return MODEL


def rag_ready() -> bool:
    legacy = _legacy()
    return legacy.index_store is not None


def initialize() -> None:
    legacy = _legacy()
    legacy.setup_embeddings()
    documents = legacy.fetch_documents()
    legacy.build_index(documents)


async def process_query(query: str, session_id: str, use_rag: bool) -> Tuple[str, str]:
    legacy = _legacy()
    return await legacy.process_query(query, session_id, use_rag)


async def call_llm(prompt: str) -> str:
    legacy = _legacy()
    return await legacy.call_llm_api(prompt, MAX_TOKENS)


def stream_query(query: str, session_id: str, use_rag: bool):
    legacy = _legacy()
    return legacy.stream_query(query, session_id, use_rag)


def conversation_memory():
    legacy = _legacy()
    return legacy.conversation_memory


def refresh_index() -> int:
    legacy = _legacy()
    documents = legacy.fetch_documents()
    legacy.build_index(documents)
    return len(documents)


def debug_search(query: str) -> Dict[str, Any]:
    legacy = _legacy()
    if not legacy.index_store:
        return {"error": "Index not ready"}

    retriever = legacy.index_store.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(query)
    results = [
        {
            "score": round(node.score, 4),
            "text_preview": node.text[:300],
            "metadata": node.metadata,
            "passes_threshold": node.score >= legacy.SIMILARITY_THRESHOLD,
        }
        for node in nodes
    ]
    return {
        "query": query,
        "threshold": legacy.SIMILARITY_THRESHOLD,
        "results_count": len(results),
        "results": results,
    }


def find_blog(title: str) -> Dict[str, Any]:
    legacy = _legacy()
    if not legacy.index_store:
        return {"error": "Index not ready"}

    retriever = legacy.index_store.as_retriever(similarity_top_k=20)
    nodes = retriever.retrieve(title)
    blog_results = []
    for node in nodes:
        if node.metadata.get("type") == "blog":
            blog_results.append(
                {
                    "score": round(node.score, 4),
                    "text_preview": node.text[:400],
                    "id": node.metadata.get("id"),
                }
            )
    return {"search_title": title, "blogs_found": len(blog_results), "results": blog_results[:10]}


def search_blogs(query: str, limit: int) -> Dict[str, Any]:
    legacy = _legacy()
    if not legacy.index_store:
        raise RuntimeError("Index not ready")

    retriever = legacy.index_store.as_retriever(similarity_top_k=limit * 2)
    nodes = retriever.retrieve(query)
    blog_results = []
    for node in nodes:
        if node.metadata.get("type") != "blog":
            continue
        text = node.text
        title = ""
        slug = node.metadata.get("slug", "")
        if "Blog Title:" in text:
            title = text.split("Blog Title:")[1].split("\n")[0].strip()
        if slug and title:
            blog_results.append(
                {
                    "title": title,
                    "url": f"https://gamatrain.com/blog/{slug}",
                    "slug": slug,
                    "relevance_score": round(node.score, 3),
                    "preview": text[:200].replace("Blog Title:", "").strip(),
                }
            )
        if len(blog_results) >= limit:
            break
    return {"query": query, "results_count": len(blog_results), "blogs": blog_results}


def search_schools(query: str, limit: int) -> Dict[str, Any]:
    legacy = _legacy()
    if not legacy.index_store:
        raise RuntimeError("Index not ready")

    retriever = legacy.index_store.as_retriever(similarity_top_k=limit * 2)
    nodes = retriever.retrieve(query)
    school_results = []
    for node in nodes:
        if node.metadata.get("type") != "school":
            continue
        text = node.text
        name = ""
        slug = node.metadata.get("slug", "")
        if "School Name:" in text:
            name = text.split("School Name:")[1].split("\n")[0].strip()
        if slug and name:
            school_results.append(
                {
                    "name": name,
                    "url": f"https://gamatrain.com/schools/{slug}",
                    "slug": slug,
                    "relevance_score": round(node.score, 3),
                    "info": text[:200].replace("School Name:", "").strip(),
                }
            )
        if len(school_results) >= limit:
            break
    return {"query": query, "results_count": len(school_results), "schools": school_results}

