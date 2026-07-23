from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


AssistantMode = Literal["assistant", "strict"]


class QueryCategory(str, Enum):
    GENERAL = "GENERAL"
    WEB = "WEB"
    DOCUMENT = "DOCUMENT"
    FOLLOW_UP_DOCUMENT = "FOLLOW_UP_DOCUMENT"
    CONVERSATION_HISTORY = "CONVERSATION_HISTORY"
    HYBRID = "HYBRID"
    DOCUMENT_AND_WEB = "DOCUMENT_AND_WEB"
    DOCUMENT_UNAVAILABLE = "DOCUMENT_UNAVAILABLE"


class WebSource(BaseModel):
    title: str
    url: str
    cited_text: str | None = None


class QueryOnlySchema(BaseModel):
    query: str


class QueryWithReferenceSchema(BaseModel):
    query: str
    expected_answer: str | None = None


class ChatHistoryItem(BaseModel):
    query: str
    answer: str | None = None
    document_id: str | None = None
    query_category: str | None = None
    source_type: str | None = None
    web_search_used: bool = False
    web_sources: list[WebSource] = Field(default_factory=list)


class AssistantQuerySchema(BaseModel):
    session_id: str
    query: str
    document_id: str | None = None
    assistant_mode: AssistantMode = "assistant"
    web_search_enabled: bool = True
    chat_history: list[ChatHistoryItem] = Field(default_factory=list)


# Backward-compatible alias for existing imports and evaluation code.
QueryWithDocumentIdSchema = AssistantQuerySchema


class StartOrResumeSessionSchema(BaseModel):
    client_id: str


class EndSessionSchema(BaseModel):
    client_id: str