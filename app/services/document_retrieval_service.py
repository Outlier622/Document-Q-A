"""Session-bound retrieval interface for the existing pipeline and future tools."""
from pathlib import Path
from uuid import UUID

from app.services.session_service import get_active_session


def load_session_vector_store(session_id: str, document_id: str | None = None):
    # Validate again at the retrieval boundary, including future direct tool calls.
    session = get_active_session(session_id)
    if session is None:
        raise ValueError("Active session not found")
    current_id = session["document_id"]
    if not current_id or (document_id is not None and document_id != current_id):
        raise FileNotFoundError("Document is not the current document for this session")
    # Existing uploads use UUIDs. Never construct a path from arbitrary tool input.
    UUID(session_id)
    UUID(current_id)
    path = Path("app/data/sessions") / session_id / "vectorstores" / f"faiss_index_{current_id}"
    from app.services.storage_service import StorageService
    from app.processing.generate_vector_db import load_vector_store
    StorageService().ensure_vector_store_local(session_id, current_id, str(path))
    return load_vector_store(str(path))


def retrieve_chunks(vector_store, query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("Retrieval query cannot be empty")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    return [
        {"excerpt_id": index, "text": str(doc.page_content).strip(),
         "metadata": dict(doc.metadata or {})}
        for index, doc in enumerate(vector_store.similarity_search(query, k=top_k), 1)
        if str(doc.page_content).strip()
    ]


def retrieve_document(session_id: str, query: str, top_k: int = 5,
                      document_id: str | None = None) -> list[dict]:
    """Return excerpts and existing metadata; legacy indexes may have no page number."""
    return retrieve_chunks(load_session_vector_store(session_id, document_id), query, top_k)


def format_document_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Document excerpt {chunk['excerpt_id']}]\n{chunk['text']}" for chunk in chunks
    )
