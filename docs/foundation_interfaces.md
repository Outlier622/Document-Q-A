# Foundation interfaces for the local assistant

This step preserves the HTTP routes and current routing behavior. It does not
introduce an Agent, LangGraph, new dependencies, or database schema changes.

## Server-owned conversation context

`session_service.get_active_session(session_id)` reads the active document and
ordered persisted messages without creating or updating a session.
`query_context_service.resolve_query_context(request)` resolves the query against
that snapshot before classification or model calls:

- An absent or ended session returns HTTP 409.
- An omitted/null document ID resolves to the current document, if any.
- An explicitly supplied non-current document ID returns HTTP 404, even for
  general queries. Clients holding an old ID should resume the session to refresh it.
- The legacy `chat_history` field remains accepted but is ignored. History comes
  from SQLite/PostgreSQL, including persisted web sources.
- Strict mode disables web search and still requires a current document.
- Saving a document-linked answer after replacement or session termination fails
  with HTTP 409 rather than inserting a stale answer.

These checks are session scoping, not user authentication. Requests still identify
anonymous sessions by session ID.

## Reusable document retrieval

`document_retrieval_service.retrieve_document(session_id, query, top_k=5,
document_id=None)` verifies the current document before opening any index and
returns a list of `{excerpt_id, text, metadata}`. IDs are validated as UUIDs
before constructing paths. Existing upload IDs already use UUIDs.

`load_session_vector_store` is shared by the existing document QA chain.
`retrieve_chunks` and `format_document_context` are used by the mixed-source
answer paths. Excerpt IDs identify positions within this retrieval result, not
stable chunk IDs. Existing metadata is preserved; no page numbers are invented
for old indexes. Future tools should bind the session ID in application code.

## Reusable web search

`web_search_service.search_web(prompt)` returns `{answer, web_search_used,
web_sources}` using the existing Gemini grounding integration. Provider exceptions
propagate to the caller; the current answer functions keep their existing error
responses. Search permission is enforced by the caller, not this low-level adapter.

Legacy imports of `invoke_web_search` and `extract_grounded_response` from
`rag_service` remain available for the existing smoke script.

## Verification

Run in the project environment:

```bash
python -B -m unittest discover -s tests -v
```

The foundation tests use a disposable SQLite database and mock model/storage
operations. They cover server history, current-document scoping, replacement,
strict mode, grounding extraction, and both HTTP query routes. They do not
measure live model quality, real FAISS retrieval quality, or cloud access.
