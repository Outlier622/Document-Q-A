"""Reusable Gemini grounding adapter, independent of HTTP and sessions."""
from typing import Any

from app.processing.generate_rag_chain import initialize_llm


def _as_dict(value: Any) -> dict:
    """Convert dictionaries and proto-like objects into plain dictionaries."""
    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}

    if hasattr(value, "__dict__"):
        return dict(vars(value))

    return {}


def _first_present(mapping: dict, *keys: str):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def invoke_web_search(prompt: str):
    """Invoke native Google Search with the current Google Gen AI integration."""
    llm = initialize_llm()
    return llm.invoke(prompt, tools=[{"google_search": {}}])


def _extract_old_grounding_metadata(response) -> dict:
    """Read grounding metadata while tolerating both protobuf naming styles."""
    response_metadata = getattr(response, "response_metadata", {}) or {}
    response_metadata = _as_dict(response_metadata)

    grounding_metadata = _first_present(
        response_metadata,
        "grounding_metadata",
        "groundingMetadata",
    )
    return _as_dict(grounding_metadata)


def _extract_cited_text_by_chunk(
    answer_text: str,
    grounding_metadata: dict,
) -> dict[int, str]:
    """Map each grounding chunk index to the answer segment it supports."""
    cited_text_by_chunk: dict[int, str] = {}
    supports = _first_present(
        grounding_metadata,
        "grounding_supports",
        "groundingSupports",
    ) or []

    for support in supports:
        support_dict = _as_dict(support)
        segment = _as_dict(support_dict.get("segment"))
        start_index = _first_present(segment, "start_index", "startIndex")
        end_index = _first_present(segment, "end_index", "endIndex")
        segment_text = _first_present(segment, "text")

        if not segment_text and isinstance(start_index, int) and isinstance(
            end_index, int
        ):
            segment_text = answer_text[start_index:end_index]

        indices = _first_present(
            support_dict,
            "grounding_chunk_indices",
            "groundingChunkIndices",
        ) or []

        for index in indices:
            if isinstance(index, int) and segment_text:
                cited_text_by_chunk.setdefault(index, str(segment_text).strip())

    return cited_text_by_chunk


def extract_grounded_response(response) -> dict:
    """Extract answer and provider-returned Google Search sources."""
    raw_content = getattr(response, "content", "")
    if isinstance(raw_content, str):
        answer = raw_content.strip()
    elif isinstance(raw_content, list):
        text_parts = []
        for block in raw_content:
            block_dict = _as_dict(block)
            text_value = block_dict.get("text")
            if text_value:
                text_parts.append(str(text_value))
            elif isinstance(block, str):
                text_parts.append(block)
        answer = "\n".join(text_parts).strip()
    else:
        answer = str(raw_content or "").strip()

    grounding_metadata = _extract_old_grounding_metadata(response)
    search_queries = _first_present(
        grounding_metadata,
        "web_search_queries",
        "webSearchQueries",
    ) or []
    chunks = _first_present(
        grounding_metadata,
        "grounding_chunks",
        "groundingChunks",
    ) or []
    supports = _first_present(
        grounding_metadata,
        "grounding_supports",
        "groundingSupports",
    ) or []

    web_search_used = bool(grounding_metadata or search_queries or chunks or supports)
    cited_text_by_chunk = _extract_cited_text_by_chunk(
        answer_text=answer,
        grounding_metadata=grounding_metadata,
    )

    sources = []
    seen_urls = set()

    for index, chunk in enumerate(chunks):
        chunk_dict = _as_dict(chunk)
        web = _as_dict(_first_present(chunk_dict, "web"))
        if not web:
            continue

        url = _first_present(web, "uri", "url")
        if not url:
            continue

        url = str(url)
        if url in seen_urls:
            continue

        title = _first_present(web, "title") or url
        sources.append(
            {
                "title": str(title),
                "url": url,
                "cited_text": cited_text_by_chunk.get(index, ""),
            }
        )
        seen_urls.add(url)

    return {
        "answer": answer,
        "web_search_used": web_search_used,
        "web_sources": sources,
    }


def search_web(prompt: str) -> dict:
    """Return answer, verified search status, and sources; propagate failures."""
    if not prompt.strip():
        raise ValueError("Search prompt cannot be empty")
    return extract_grounded_response(invoke_web_search(prompt))
