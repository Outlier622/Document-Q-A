from pydantic import BaseModel, Field


class QueryOnlySchema(BaseModel):
    query: str


class QueryWithReferenceSchema(BaseModel):
    query: str
    expected_answer: str | None = None


class ChatHistoryItem(BaseModel):
    query: str
    answer: str | None = None
    document_id: str | None = None


class QueryWithDocumentIdSchema(BaseModel):
    session_id: str
    query: str
    document_id: str
    chat_history: list[ChatHistoryItem] = Field(default_factory=list)


class StartOrResumeSessionSchema(BaseModel):
    client_id: str


class EndSessionSchema(BaseModel):
    client_id: str
