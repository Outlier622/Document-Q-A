"""Resolve legacy query payloads against server-owned conversation state."""
from fastapi import HTTPException

from app.schemas.rag_schema import AssistantQuerySchema, ChatHistoryItem
from app.services.session_service import get_active_session


def resolve_query_context(request: AssistantQuerySchema) -> AssistantQuerySchema:
    session = get_active_session(request.session_id)
    if session is None:
        raise HTTPException(409, "The conversation has ended or does not exist.")
    document_id = session["document_id"]
    if request.document_id is not None and request.document_id != document_id:
        raise HTTPException(404, "Document is not the current document for this session.")
    return request.model_copy(update={
        "document_id": document_id,
        "chat_history": [ChatHistoryItem(**item) for item in session["chat_history"]],
        "web_search_enabled": request.web_search_enabled and request.assistant_mode != "strict",
    })
