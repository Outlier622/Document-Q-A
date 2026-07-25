import os
import re
from datetime import datetime, timezone
from typing import Any

from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.processing.evaluate_rag import evaluate_rag_with_reference
from app.processing.generate_embeddings import get_embeddings
from app.processing.generate_rag_chain import create_rag_chain, initialize_llm
from app.processing.generate_text_chunks import generate_text_chunks_from_pdf
from app.processing.generate_vector_db import create_vector_store, load_vector_store
from app.processing.single_query_inference import run_inference
from app.schemas.rag_schema import (
    AssistantQuerySchema,
    QueryOnlySchema,
    QueryWithReferenceSchema,
)
from app.services.storage_service import StorageService
from app.services.queue_service import QueueService
from app.services.session_service import (
    create_processing_job,
    is_session_active,
    mark_processing_job_failed,
    save_message,
    set_session_document,
)


config = Config()
logger = configure_logging("RAG_SERVICE")
storage_service = StorageService(config)
queue_service = QueueService(config)

_default_embeddings = None
_default_rag_chain = None


def _get_default_embeddings():
    global _default_embeddings
    if _default_embeddings is None:
        _default_embeddings = get_embeddings()
    return _default_embeddings


def _get_default_rag_chain():
    global _default_rag_chain
    if _default_rag_chain is None:
        default_vector_store = load_vector_store(config.VECTOR_STORE_PATH)
        _default_rag_chain = create_rag_chain(default_vector_store)
    return _default_rag_chain


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
    """Invoke Google Search using langchain-google-genai 2.1.8.

    Version 2.1.8 uses google-ai-generativelanguage and expects the native
    GenAITool protobuf object. Grounding details are returned in the
    AIMessage.response_metadata['grounding_metadata'] field.
    """
    llm = initialize_llm()
    search_tool = GenAITool(google_search={})
    return llm.invoke(prompt, tools=[search_tool])


def _extract_old_grounding_metadata(response) -> dict:
    """Read grounding metadata exposed by langchain-google-genai 2.1.8."""
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
    """Extract answer and Google Search sources for dependency version 2.1.8."""
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


async def query_rag_without_reference(request: QueryOnlySchema):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {request.query}")
        answer = run_inference(
            rag_chain=_get_default_rag_chain(),
            query=request.query,
        )

        return JSONResponse(
            content={
                "query": request.query,
                "answer": answer,
            }
        )

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Query processing error: {error}")
        raise HTTPException(status_code=500, detail=str(error)) from error


async def query_rag_with_reference(request: QueryWithReferenceSchema):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {request.query}")

        if request.expected_answer:
            result = evaluate_rag_with_reference(
                query=request.query,
                expected_answer=request.expected_answer,
                rag_chain=_get_default_rag_chain(),
                embeddings=_get_default_embeddings(),
            )

            return JSONResponse(
                content={
                    "query": result.get("query"),
                    "expected_answer": result.get("expected_answer"),
                    "actual": result.get("actual"),
                    "cosine_similarity": result.get("cosine_similarity"),
                    "context": result.get("context"),
                }
            )

        answer = run_inference(
            rag_chain=_get_default_rag_chain(),
            query=request.query,
        )

        return JSONResponse(
            content={
                "query": request.query,
                "expected_answer": "N/A",
                "actual": answer,
                "cosine_similarity": "N/A",
                "context": ["N/A"],
            }
        )

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Query processing error: {error}")
        raise HTTPException(status_code=500, detail=str(error)) from error


async def generate_vector_store_for_pdf(
    pdf_file: UploadFile,
    session_id: str,
    document_id: str,
    saved_pdf_path: str,
    output_text_file_path: str,
    saved_vector_store_path: str,
):
    try:
        if not is_session_active(session_id):
            raise HTTPException(
                status_code=409,
                detail="The conversation has ended or does not exist.",
            )

        os.makedirs(os.path.dirname(output_text_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(saved_vector_store_path), exist_ok=True)

        uploaded_filename = pdf_file.filename or "uploaded.pdf"

        storage_service.save_uploaded_pdf(
            upload_file=pdf_file,
            local_path=saved_pdf_path,
            session_id=session_id,
            document_id=document_id,
        )

        chunks = generate_text_chunks_from_pdf(
            saved_pdf_path,
            output_text_file_path,
        )

        create_vector_store(chunks, saved_vector_store_path)

        storage_service.sync_processed_artifacts(
            session_id=session_id,
            document_id=document_id,
            text_path=output_text_file_path,
            vector_store_path=saved_vector_store_path,
        )

        set_session_document(
            session_id=session_id,
            document_id=document_id,
            uploaded_filename=uploaded_filename,
        )

        logger.info(
            f"PDF uploaded and processed successfully. "
            f"session_id={session_id}, document_id={document_id}"
        )

        return JSONResponse(
            content={
                "session_id": session_id,
                "document_id": document_id,
                "uploaded_filename": uploaded_filename,
                "storage_backend": config.STORAGE_BACKEND,
                "message": "PDF uploaded and vector store created.",
            }
        )

    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error in PDF upload and vector processing: {error}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the PDF: {error}",
        ) from error


async def enqueue_pdf_processing(
    pdf_file: UploadFile,
    session_id: str,
    document_id: str,
    job_id: str,
    saved_pdf_path: str,
):
    """Persist an upload and dispatch its heavy processing through SQS."""
    if config.STORAGE_BACKEND != "s3":
        raise HTTPException(
            status_code=500,
            detail=(
                "Asynchronous SQS processing requires STORAGE_BACKEND=s3."
            ),
        )

    if not is_session_active(session_id):
        raise HTTPException(
            status_code=409,
            detail="The conversation has ended or does not exist.",
        )

    uploaded_filename = pdf_file.filename or "uploaded.pdf"

    try:
        create_processing_job(
            job_id=job_id,
            session_id=session_id,
            document_id=document_id,
            uploaded_filename=uploaded_filename,
        )

        storage_service.save_uploaded_pdf(
            upload_file=pdf_file,
            local_path=saved_pdf_path,
            session_id=session_id,
            document_id=document_id,
        )

        message_id = queue_service.send_document_job(
            job_id=job_id,
            session_id=session_id,
            document_id=document_id,
            uploaded_filename=uploaded_filename,
        )

        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "PENDING",
                "session_id": session_id,
                "document_id": document_id,
                "uploaded_filename": uploaded_filename,
                "storage_backend": config.STORAGE_BACKEND,
                "processing_mode": config.DOCUMENT_PROCESSING_MODE,
                "sqs_message_id": message_id,
                "message": "PDF accepted and queued for processing.",
            },
        )

    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        try:
            mark_processing_job_failed(
                job_id=job_id,
                attempt_count=0,
                error_message=str(error),
            )
        except Exception:
            logger.exception("Could not mark enqueue failure in processing_jobs")
        logger.exception("Failed to queue PDF processing job")
        raise HTTPException(
            status_code=500,
            detail=f"Could not queue PDF processing: {error}",
        ) from error


def _get_history_value(item, key: str, default=None):
    if isinstance(item, dict):
        value = item.get(key, default)
    else:
        value = getattr(item, key, default)

    if value is None:
        return default

    return value


def _select_history(
    chat_history,
    current_document_id: str | None,
    scope: str,
):
    if not chat_history:
        return []

    if scope == "all":
        return list(chat_history)

    if scope != "document":
        raise ValueError(f"Unsupported history scope: {scope}")

    if not current_document_id:
        return []

    return [
        item
        for item in chat_history
        if _get_history_value(item, "document_id") == current_document_id
    ]


def format_chat_history(
    chat_history,
    current_document_id: str | None = None,
    max_turns: int | None = None,
    scope: str = "all",
):
    selected_history = _select_history(
        chat_history=chat_history,
        current_document_id=current_document_id,
        scope=scope,
    )

    if max_turns is not None:
        selected_history = selected_history[-max_turns:]

    lines = []

    for index, item in enumerate(selected_history, start=1):
        user_query = _get_history_value(item, "query", "")
        assistant_answer = _get_history_value(item, "answer", "")
        query_category = _get_history_value(item, "query_category", "")

        category_suffix = f" [{query_category}]" if query_category else ""

        if user_query:
            lines.append(f"Turn {index} User{category_suffix}: {user_query}")

        if assistant_answer:
            lines.append(f"Turn {index} Assistant: {assistant_answer}")

    return "\n".join(lines)


def _invoke_llm(prompt: str) -> str:
    llm = initialize_llm()
    response = llm.invoke(prompt)
    raw_content = getattr(response, "content", str(response))

    if isinstance(raw_content, str):
        return raw_content.strip()

    text_parts = []
    for block in raw_content or []:
        block_dict = _as_dict(block)
        text = block_dict.get("text")
        if text:
            text_parts.append(str(text))

    return "\n".join(text_parts).strip() or str(raw_content).strip()


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _mentions_current_document(query: str) -> bool:
    normalized = query.lower()
    phrases = (
        "this document",
        "the uploaded document",
        "uploaded document",
        "this pdf",
        "the pdf",
        "uploaded pdf",
        "this file",
        "the uploaded file",
        "according to the document",
        "based on the document",
        "这份文档",
        "这个文档",
        "上传的文档",
        "文档中",
        "根据文档",
        "结合文档",
        "这份文件",
        "这个文件",
        "这份pdf",
        "这个pdf",
    )
    return any(phrase in normalized for phrase in phrases)


def _explicitly_requests_web(query: str) -> bool:
    normalized = query.lower()
    phrases = (
        "search the web",
        "search online",
        "browse the web",
        "look it up online",
        "google search",
        "web search",
        "online sources",
        "current sources",
        "联网",
        "上网",
        "网上搜索",
        "搜索网页",
        "搜索网络",
        "查一下最新",
        "搜一下最新",
        "联网核实",
        "联网查询",
    )
    return any(phrase in normalized for phrase in phrases)


def _likely_time_sensitive(query: str) -> bool:
    normalized = query.lower()
    phrases = (
        "latest",
        "today",
        "currently",
        "right now",
        "as of today",
        "up to date",
        "current price",
        "current policy",
        "current rule",
        "current version",
        "current rate limit",
        "最近",
        "最新",
        "今天",
        "目前",
        "截至今天",
        "现在的价格",
        "当前价格",
        "当前政策",
        "当前规定",
        "当前版本",
        "当前额度",
    )
    return any(phrase in normalized for phrase in phrases)


def _recent_history_uses_document(
    chat_history,
    document_id: str | None,
    max_turns: int = 3,
) -> bool:
    if not document_id:
        return False

    document_categories = {
        "DOCUMENT",
        "FOLLOW_UP_DOCUMENT",
        "HYBRID",
        "DOCUMENT_AND_WEB",
    }

    for item in list(chat_history or [])[-max_turns:]:
        item_document_id = _get_history_value(item, "document_id")
        item_category = _get_history_value(item, "query_category", "")
        if (
            item_document_id == document_id
            and item_category in document_categories
        ):
            return True

    return False


def _missing_document_answer(query: str) -> str:
    if _contains_chinese(query):
        return "当前没有上传文档。你仍然可以询问通用问题，或先上传 PDF 后再询问文档内容。"
    return (
        "No document is currently uploaded. You can still ask a general "
        "question, or upload a PDF before asking about document content."
    )


def _web_failure_answer(query: str) -> str:
    if _contains_chinese(query):
        return "联网搜索未能完成，因此我无法可靠核实当前信息。请检查后端日志、模型支持情况和 API 配额后重试。"
    return (
        "The web search could not be completed, so I cannot reliably verify "
        "current information. Check the backend logs, model support, and API "
        "quota, then try again."
    )


def classify_query(
    query: str,
    chat_history,
    document_id: str | None,
    assistant_mode: str = "assistant",
    web_search_enabled: bool = True,
):
    document_available = bool(document_id)
    recent_history_text = format_chat_history(
        chat_history=chat_history,
        current_document_id=document_id,
        max_turns=6,
        scope="all",
    )

    if assistant_mode == "strict":
        valid_categories = (
            "CONVERSATION_HISTORY",
            "FOLLOW_UP_DOCUMENT",
            "DOCUMENT",
        )
        category_definitions = """
CONVERSATION_HISTORY:
The user asks about earlier questions, answers, or the conversation itself.

FOLLOW_UP_DOCUMENT:
The question is about the uploaded document and depends on an earlier turn.

DOCUMENT:
The question should be answered only from the uploaded document. In strict
mode, general questions are also sent to document retrieval so unsupported
information is not supplied from model or web knowledge.
"""
        fallback_category = "FOLLOW_UP_DOCUMENT" if chat_history else "DOCUMENT"
    else:
        if web_search_enabled and (
            _explicitly_requests_web(query) or _likely_time_sensitive(query)
        ):
            needs_document = document_available and (
                _mentions_current_document(query)
                or _recent_history_uses_document(chat_history, document_id)
            )
            deterministic_category = (
                "DOCUMENT_AND_WEB" if needs_document else "WEB"
            )
            logger.info(
                "Deterministic web routing selected category: %s",
                deterministic_category,
            )
            return deterministic_category

        if document_available:
            categories = [
                "CONVERSATION_HISTORY",
                "FOLLOW_UP_DOCUMENT",
            ]
            if web_search_enabled:
                categories.extend(["DOCUMENT_AND_WEB", "WEB"])
            categories.extend(["HYBRID", "DOCUMENT", "GENERAL"])
            valid_categories = tuple(categories)

            web_definitions = """
WEB:
The user needs current, recent, changing, externally verifiable, or explicitly
requested online information, but the answer does not require the uploaded
document.

DOCUMENT_AND_WEB:
The request requires both evidence from the uploaded document and current web
information, such as updating, verifying, or comparing document claims.
""" if web_search_enabled else ""

            category_definitions = f"""
CONVERSATION_HISTORY:
The user asks about earlier questions, answers, conversation order, or a
summary or comparison of prior turns.

FOLLOW_UP_DOCUMENT:
The user is asking about the uploaded document, and the current question needs
an earlier turn to become a complete document-retrieval query.

DOCUMENT:
The user asks for facts, extraction, explanation, or summarization that should
come from the uploaded document only.

GENERAL:
The user asks for stable ordinary knowledge, writing, reasoning, or advice that
does not require the uploaded document or current web information.

HYBRID:
The user asks for a recommendation, evaluation, comparison, implication, or
advice that requires both facts from the uploaded document and stable general
knowledge. Do not use HYBRID for current or explicitly online information.
{web_definitions}
"""
            fallback_category = "DOCUMENT" if not chat_history else "GENERAL"
        else:
            categories = ["CONVERSATION_HISTORY"]
            if web_search_enabled:
                categories.append("WEB")
            categories.append("GENERAL")
            valid_categories = tuple(categories)

            web_definition = """
WEB:
The user needs current, recent, changing, externally verifiable, or explicitly
requested online information.
""" if web_search_enabled else ""

            category_definitions = f"""
CONVERSATION_HISTORY:
The user asks about earlier questions, answers, conversation order, or a
summary or comparison of prior turns.

GENERAL:
Any normal assistant request that can be answered from stable general
knowledge, including explanations, writing, reasoning, advice, and follow-up
discussion. No uploaded document is available.
{web_definition}
"""
            fallback_category = "GENERAL"

    labels = "\n".join(valid_categories)
    history_display = recent_history_text or "No previous conversation."

    classification_prompt = f"""
Classify the current request for a document-aware conversational assistant.

Assistant mode: {assistant_mode}
Uploaded document available: {document_available}
Web search enabled: {web_search_enabled}

Return exactly one of these labels:
{labels}

Definitions:
{category_definitions}

Recent conversation:
{history_display}

Current request:
{query}

Return only the label:
"""

    try:
        raw_category = _invoke_llm(classification_prompt).upper().strip()

        for category in valid_categories:
            if raw_category == category or re.search(
                rf"\b{re.escape(category)}\b",
                raw_category,
            ):
                logger.info(f"Query category: {category}")
                return category

        logger.warning(
            f"Unexpected query category '{raw_category}'. "
            f"Falling back to {fallback_category}."
        )
        return fallback_category

    except Exception as error:
        logger.error(f"Query classification failed: {error}")
        return fallback_category


def answer_from_chat_history(
    query: str,
    chat_history,
    document_id: str | None = None,
):
    full_history_text = format_chat_history(
        chat_history=chat_history,
        current_document_id=document_id,
        max_turns=None,
        scope="all",
    )

    if not full_history_text:
        return "There is no previous conversation history."

    history_prompt = f"""
You answer questions about a previous conversation.

Use only the conversation history below.
Do not retrieve from a document or the web.
Do not add outside knowledge.
Do not invent missing questions or answers.
Answer in the same language as the user's current question.

Conversation history:
{full_history_text}

Current question about the conversation:
{query}

Answer:
"""

    try:
        answer = _invoke_llm(history_prompt)
        if not answer:
            return "I could not find that information in the conversation history."
        return answer

    except Exception as error:
        logger.error(f"History question answering failed: {error}")
        return "I could not answer the question from the conversation history."


def rewrite_query_with_history(
    query: str,
    chat_history,
    document_id: str | None,
):
    history_text = format_chat_history(
        chat_history=chat_history,
        current_document_id=document_id,
        max_turns=4,
        scope="all",
    )

    logger.info(f"Original query: {query}")
    logger.info(f"History text used for rewrite:\n{history_text}")

    if not history_text:
        return query

    rewrite_prompt = f"""
Rewrite the current follow-up question into a complete standalone question for
retrieval from the currently uploaded document.

Rules:
- Do not answer the question.
- Resolve omitted subjects and references from the recent conversation.
- Do not mention the conversation history.
- Do not add unsupported facts.
- Preserve the user's language and intent.
- Return only the standalone question.

Recent conversation:
{history_text}

Current follow-up question:
{query}

Standalone question:
"""

    try:
        rewritten_query = _invoke_llm(rewrite_prompt)
        if not rewritten_query:
            return query
        logger.info(f"Rewritten query: {rewritten_query}")
        return rewritten_query

    except Exception as error:
        logger.error(f"Question rewriting failed: {error}")
        return query


def answer_general_question(query: str, chat_history) -> str:
    recent_history_text = format_chat_history(
        chat_history=chat_history,
        max_turns=8,
        scope="all",
    )
    history_display = recent_history_text or "No previous conversation."

    prompt = f"""
You are a capable general conversational assistant.

Answer the user's request directly and in the same language as the user.
Use the recent conversation when relevant to resolve follow-up context.
Do not claim that general knowledge came from an uploaded document.
Do not claim to have searched the web. For rapidly changing information, say
that current verification requires web search.

Recent conversation:
{history_display}

Current request:
{query}

Answer:
"""

    try:
        answer = _invoke_llm(prompt)
        return answer or "I could not generate an answer."
    except Exception as error:
        logger.error(f"General question answering failed: {error}")
        return "I could not answer the general question."


def answer_web_question(query: str, chat_history) -> tuple[str, list[dict], bool, str | None]:
    recent_history_text = format_chat_history(
        chat_history=chat_history,
        max_turns=6,
        scope="all",
    )
    current_date = datetime.now(timezone.utc).date().isoformat()

    prompt = f"""
Use Google Search to answer the user's request with current, externally
verifiable information.

Requirements:
- The current UTC date is {current_date}.
- Actually use Google Search before answering.
- Prefer official, primary, and recently updated sources.
- Distinguish established facts from uncertainty or conflicting reports.
- Answer in the same language as the user.
- Do not claim that web facts came from an uploaded document.
- Do not invent URLs or citations; citation metadata will be rendered by the UI.

Recent conversation:
{recent_history_text or "No previous conversation."}

Current request:
{query}

Answer:
"""

    try:
        response = invoke_web_search(prompt)
        grounded = extract_grounded_response(response)
        answer = grounded["answer"] or _web_failure_answer(query)
        return (
            answer,
            grounded["web_sources"],
            grounded["web_search_used"],
            None,
        )
    except Exception as error:
        logger.exception("Web search question answering failed")
        return _web_failure_answer(query), [], False, str(error)


def _retrieve_document_context(vector_store, query: str, top_k: int = 5) -> str:
    documents = vector_store.similarity_search(query, k=top_k)
    excerpts = []

    for index, document in enumerate(documents, start=1):
        page_content = str(getattr(document, "page_content", "")).strip()
        if page_content:
            excerpts.append(f"[Document excerpt {index}]\n{page_content}")

    return "\n\n".join(excerpts)


def answer_hybrid_question(
    query: str,
    chat_history,
    document_id: str,
    current_vector_store,
) -> tuple[str, str]:
    standalone_query = rewrite_query_with_history(
        query=query,
        chat_history=chat_history,
        document_id=document_id,
    )
    document_context = _retrieve_document_context(
        vector_store=current_vector_store,
        query=standalone_query,
    )
    recent_history_text = format_chat_history(
        chat_history=chat_history,
        current_document_id=document_id,
        max_turns=6,
        scope="all",
    )

    prompt = f"""
You are a document-aware assistant answering a mixed-source question.

Use the document excerpts for claims about the uploaded document. You may use
stable general knowledge only for explanation, comparison, implications, or
advice. Clearly distinguish document-grounded facts from general guidance.
Never imply that general knowledge appears in the document. If the excerpts do
not support a requested document fact, say that it was not found in the
provided excerpts. Do not claim to have searched the web. Answer in the same
language as the user's question.

Recent conversation:
{recent_history_text or "No previous conversation."}

Document excerpts:
{document_context or "No relevant document excerpts were retrieved."}

Current request:
{query}

Answer:
"""

    try:
        answer = _invoke_llm(prompt)
        return (
            answer or "I could not generate a mixed-source answer.",
            standalone_query,
        )
    except Exception as error:
        logger.error(f"Hybrid question answering failed: {error}")
        return (
            "I could not combine the document with general guidance.",
            standalone_query,
        )


def answer_document_and_web(
    query: str,
    chat_history,
    document_id: str,
    current_vector_store,
) -> tuple[str, str, list[dict], bool, str | None]:
    standalone_query = rewrite_query_with_history(
        query=query,
        chat_history=chat_history,
        document_id=document_id,
    )
    document_context = _retrieve_document_context(
        vector_store=current_vector_store,
        query=standalone_query,
    )
    recent_history_text = format_chat_history(
        chat_history=chat_history,
        current_document_id=document_id,
        max_turns=6,
        scope="all",
    )
    current_date = datetime.now(timezone.utc).date().isoformat()

    prompt = f"""
Answer using two explicitly separated evidence sources: the uploaded document
excerpts below and current Google Search results.

Requirements:
- The current UTC date is {current_date}.
- Actually use Google Search to obtain current external information.
- Prefer official, primary, and recently updated web sources.
- State what the uploaded document says before adding web information.
- Never attribute web facts to the uploaded document.
- Identify conflicts, outdated claims, or unresolved uncertainty.
- Give recommendations only after presenting the evidence.
- Answer in the same language as the user's question.
- Do not invent URLs; citation metadata will be rendered by the UI.

Recent conversation:
{recent_history_text or "No previous conversation."}

Uploaded document excerpts:
{document_context or "No relevant document excerpts were retrieved."}

Current request:
{query}

Answer:
"""

    try:
        response = invoke_web_search(prompt)
        grounded = extract_grounded_response(response)
        answer = grounded["answer"] or _web_failure_answer(query)
        return (
            answer,
            standalone_query,
            grounded["web_sources"],
            grounded["web_search_used"],
            None,
        )
    except Exception as error:
        logger.exception("Document-and-web question answering failed")
        return (
            _web_failure_answer(query),
            standalone_query,
            [],
            False,
            str(error),
        )


def _build_response(
    request: AssistantQuerySchema,
    query_category: str,
    answer: str,
    standalone_query: str,
    source_type: str,
    used_document_retrieval: bool,
    web_search_used: bool = False,
    web_sources: list[dict] | None = None,
    web_search_error: str | None = None,
):
    return JSONResponse(
        content={
            "query": request.query,
            "query_category": query_category,
            "standalone_query": standalone_query,
            "answer": answer,
            "assistant_mode": request.assistant_mode,
            "document_id": request.document_id,
            "source_type": source_type,
            "used_document_retrieval": used_document_retrieval,
            "web_search_enabled": request.web_search_enabled,
            "web_search_used": web_search_used,
            "web_sources": web_sources or [],
            "web_search_error": web_search_error,
        }
    )


def _save_answer(
    request: AssistantQuerySchema,
    answer: str,
    query_category: str,
    source_type: str,
    web_search_used: bool = False,
    web_sources: list[dict] | None = None,
) -> None:
    # General and web-only turns are independent of the currently attached PDF.
    stored_document_id = (
        None
        if query_category in {"GENERAL", "WEB", "DOCUMENT_UNAVAILABLE"}
        else request.document_id
    )

    save_message(
        session_id=request.session_id,
        document_id=stored_document_id,
        query=request.query,
        answer=answer,
        query_category=query_category,
        source_type=source_type,
        web_search_used=web_search_used,
        web_sources=web_sources or [],
    )


async def query_assistant(request: AssistantQuerySchema):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not is_session_active(request.session_id):
            raise HTTPException(
                status_code=409,
                detail="The conversation has ended or does not exist.",
            )

        if request.assistant_mode == "strict" and not request.document_id:
            raise HTTPException(
                status_code=400,
                detail="Strict Document mode requires an uploaded document.",
            )

        logger.info(f"Received assistant query: {request.query}")
        logger.info(
            f"document_id={request.document_id}, "
            f"assistant_mode={request.assistant_mode}, "
            f"web_search_enabled={request.web_search_enabled}, "
            f"chat_history_length={len(request.chat_history)}"
        )

        if not request.document_id and _mentions_current_document(request.query):
            query_category = "DOCUMENT_UNAVAILABLE"
            answer = _missing_document_answer(request.query)
            _save_answer(request, answer, query_category, "no_document")
            return _build_response(
                request=request,
                query_category=query_category,
                answer=answer,
                standalone_query=request.query,
                source_type="no_document",
                used_document_retrieval=False,
            )

        query_category = classify_query(
            query=request.query,
            chat_history=request.chat_history,
            document_id=request.document_id,
            assistant_mode=request.assistant_mode,
            web_search_enabled=request.web_search_enabled,
        )

        if query_category == "CONVERSATION_HISTORY":
            answer = answer_from_chat_history(
                query=request.query,
                chat_history=request.chat_history,
                document_id=request.document_id,
            )
            _save_answer(
                request,
                answer,
                query_category,
                "conversation_history",
            )
            return _build_response(
                request=request,
                query_category=query_category,
                answer=answer,
                standalone_query=request.query,
                source_type="conversation_history",
                used_document_retrieval=False,
            )

        if query_category == "GENERAL":
            answer = answer_general_question(
                query=request.query,
                chat_history=request.chat_history,
            )
            _save_answer(request, answer, query_category, "general_knowledge")
            return _build_response(
                request=request,
                query_category=query_category,
                answer=answer,
                standalone_query=request.query,
                source_type="general_knowledge",
                used_document_retrieval=False,
            )

        if query_category == "WEB":
            answer, web_sources, web_search_used, web_search_error = (
                answer_web_question(
                    query=request.query,
                    chat_history=request.chat_history,
                )
            )
            source_type = (
                "web_search" if web_search_used else "web_search_unavailable"
            )
            _save_answer(
                request,
                answer,
                query_category,
                source_type,
                web_search_used=web_search_used,
                web_sources=web_sources,
            )
            return _build_response(
                request=request,
                query_category=query_category,
                answer=answer,
                standalone_query=request.query,
                source_type=source_type,
                used_document_retrieval=False,
                web_search_used=web_search_used,
                web_sources=web_sources,
                web_search_error=web_search_error,
            )

        if not request.document_id:
            answer = _missing_document_answer(request.query)
            _save_answer(
                request,
                answer,
                "DOCUMENT_UNAVAILABLE",
                "no_document",
            )
            return _build_response(
                request=request,
                query_category="DOCUMENT_UNAVAILABLE",
                answer=answer,
                standalone_query=request.query,
                source_type="no_document",
                used_document_retrieval=False,
            )

        saved_vector_store_path = (
            f"app/data/sessions/{request.session_id}/"
            f"vectorstores/faiss_index_{request.document_id}"
        )

        try:
            storage_service.ensure_vector_store_local(
                session_id=request.session_id,
                document_id=request.document_id,
                local_vector_store_path=saved_vector_store_path,
            )
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail="Document ID not found for this session.",
            ) from error

        current_vector_store = load_vector_store(saved_vector_store_path)

        if query_category == "DOCUMENT_AND_WEB":
            (
                answer,
                standalone_query,
                web_sources,
                web_search_used,
                web_search_error,
            ) = answer_document_and_web(
                query=request.query,
                chat_history=request.chat_history,
                document_id=request.document_id,
                current_vector_store=current_vector_store,
            )
            source_type = (
                "document_and_web"
                if web_search_used
                else "document_and_web_search_unavailable"
            )
            _save_answer(
                request,
                answer,
                query_category,
                source_type,
                web_search_used=web_search_used,
                web_sources=web_sources,
            )
            return _build_response(
                request=request,
                query_category=query_category,
                answer=answer,
                standalone_query=standalone_query,
                source_type=source_type,
                used_document_retrieval=True,
                web_search_used=web_search_used,
                web_sources=web_sources,
                web_search_error=web_search_error,
            )

        if query_category == "HYBRID":
            answer, standalone_query = answer_hybrid_question(
                query=request.query,
                chat_history=request.chat_history,
                document_id=request.document_id,
                current_vector_store=current_vector_store,
            )
            _save_answer(
                request,
                answer,
                query_category,
                "document_and_general_knowledge",
            )
            return _build_response(
                request=request,
                query_category=query_category,
                answer=answer,
                standalone_query=standalone_query,
                source_type="document_and_general_knowledge",
                used_document_retrieval=True,
            )

        current_rag_chain = create_rag_chain(current_vector_store)

        if query_category == "FOLLOW_UP_DOCUMENT":
            standalone_query = rewrite_query_with_history(
                query=request.query,
                chat_history=request.chat_history,
                document_id=request.document_id,
            )
        else:
            standalone_query = request.query

        answer = run_inference(
            rag_chain=current_rag_chain,
            query=standalone_query,
        )
        _save_answer(request, answer, query_category, "uploaded_document")

        return _build_response(
            request=request,
            query_category=query_category,
            answer=answer,
            standalone_query=standalone_query,
            source_type="uploaded_document",
            used_document_retrieval=True,
        )

    except HTTPException:
        raise
    except Exception as error:
        logger.error(
            f"Error processing assistant query. "
            f"document_id={request.document_id}: {error}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the query: {error}",
        ) from error


# Backward-compatible service name used by the previous route implementation.
async def query_rag_by_document(request: AssistantQuerySchema):
    return await query_assistant(request)