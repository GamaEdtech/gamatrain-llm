from typing import List

from pydantic import BaseModel


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

