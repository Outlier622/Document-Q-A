import os
import shutil

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
    QueryOnlySchema,
    QueryWithDocumentIdSchema,
    QueryWithReferenceSchema,
)
from app.services.session_service import (
    is_session_active,
    save_message,
    set_session_document,
)


config = Config()
logger = configure_logging("RAG_SERVICE")

embeddings = get_embeddings()
vector_store = load_vector_store(config.VECTOR_STORE_PATH)
rag_chain = create_rag_chain(vector_store)


async def query_rag_without_reference(request: QueryOnlySchema):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {request.query}")
        answer = run_inference(rag_chain=rag_chain, query=request.query)

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
                rag_chain=rag_chain,
                embeddings=embeddings,
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

        answer = run_inference(rag_chain=rag_chain, query=request.query)

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

        os.makedirs(os.path.dirname(saved_pdf_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_text_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(saved_vector_store_path), exist_ok=True)

        uploaded_filename = pdf_file.filename or "uploaded.pdf"

        with open(saved_pdf_path, "wb") as output_file:
            shutil.copyfileobj(pdf_file.file, output_file)

        chunks = generate_text_chunks_from_pdf(
            saved_pdf_path,
            output_text_file_path,
        )

        create_vector_store(chunks, saved_vector_store_path)

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


def _get_history_value(item, key: str, default: str = ""):
    if isinstance(item, dict):
        value = item.get(key, default)
    else:
        value = getattr(item, key, default)

    if value is None:
        return default

    return value


def format_chat_history(
    chat_history,
    current_document_id: str,
    max_turns: int | None = None,
):
    if not chat_history:
        return ""

    same_document_history = [
        item
        for item in chat_history
        if _get_history_value(item, "document_id", current_document_id)
        == current_document_id
    ]

    if max_turns is not None:
        selected_history = same_document_history[-max_turns:]
    else:
        selected_history = same_document_history

    lines = []

    for index, item in enumerate(selected_history, start=1):
        user_query = _get_history_value(item, "query", "")
        assistant_answer = _get_history_value(item, "answer", "")

        if user_query:
            lines.append(f"Turn {index} User: {user_query}")

        if assistant_answer:
            lines.append(f"Turn {index} Assistant: {assistant_answer}")

    return "\n".join(lines)


def classify_query(query: str, chat_history, document_id: str):
    if not chat_history:
        return "DOCUMENT"

    recent_history_text = format_chat_history(
        chat_history=chat_history,
        current_document_id=document_id,
        max_turns=5,
    )

    classification_prompt = f"""
Classify the current user question for a document question-answering system.

Return exactly one of these labels:

CONVERSATION_HISTORY
FOLLOW_UP_DOCUMENT
DOCUMENT

CONVERSATION_HISTORY:
The user is asking about previous questions, previous answers,
conversation order, or a summary or comparison of earlier turns.

FOLLOW_UP_DOCUMENT:
The user is asking about the uploaded document, but the current
question depends on earlier conversation context.

DOCUMENT:
The user is asking a complete standalone question about the
uploaded document.

Recent conversation:
{recent_history_text}

Current question:
{query}

Return only the label:
"""

    try:
        llm = initialize_llm()
        response = llm.invoke(classification_prompt)

        raw_category = getattr(response, "content", str(response))
        raw_category = raw_category.strip().upper()

        valid_categories = (
            "CONVERSATION_HISTORY",
            "FOLLOW_UP_DOCUMENT",
            "DOCUMENT",
        )

        for category in valid_categories:
            if category in raw_category:
                logger.info(f"Query category: {category}")
                return category

        logger.warning(
            f"Unexpected query category '{raw_category}'. "
            f"Falling back to FOLLOW_UP_DOCUMENT."
        )
        return "FOLLOW_UP_DOCUMENT"

    except Exception as error:
        logger.error(f"Query classification failed: {error}")
        return "FOLLOW_UP_DOCUMENT"


def answer_from_chat_history(query: str, chat_history, document_id: str):
    full_history_text = format_chat_history(
        chat_history=chat_history,
        current_document_id=document_id,
        max_turns=None,
    )

    if not full_history_text:
        return "There is no previous conversation history for this document."

    history_prompt = f"""
You answer questions about a previous conversation.

Use only the conversation history below.
Do not use the uploaded document.
Do not invent missing questions or answers.
Answer in the same language as the user's current question.

Conversation history:
{full_history_text}

Current question about the conversation:
{query}

Answer:
"""

    try:
        llm = initialize_llm()
        response = llm.invoke(history_prompt)
        answer = getattr(response, "content", str(response)).strip()

        if not answer:
            return "I could not find that information in the conversation history."

        return answer

    except Exception as error:
        logger.error(f"History question answering failed: {error}")
        return "I could not answer the question from the conversation history."


def rewrite_query_with_history(query: str, chat_history, document_id: str):
    history_text = format_chat_history(
        chat_history=chat_history,
        current_document_id=document_id,
        max_turns=3,
    )

    logger.info(f"Original query: {query}")
    logger.info(f"History text used for rewrite:\n{history_text}")

    if not history_text:
        logger.info(f"Rewritten query: {query}")
        return query

    rewrite_prompt = f"""
Rewrite the current follow-up question into a complete standalone
question for document retrieval.

Use the recent conversation to recover the subject, entity,
requirement, comparison target, or reference omitted from the
current question.

Rules:
- Do not answer the question.
- Do not mention the conversation history.
- Do not add facts unsupported by the conversation.
- Preserve the user's original intent.
- Return only the standalone question.

Recent conversation:
{history_text}

Current follow-up question:
{query}

Standalone question:
"""

    try:
        llm = initialize_llm()
        response = llm.invoke(rewrite_prompt)
        rewritten_query = getattr(response, "content", str(response)).strip()

        if not rewritten_query:
            logger.info(f"Rewritten query: {query}")
            return query

        logger.info(f"Rewritten query: {rewritten_query}")
        return rewritten_query

    except Exception as error:
        logger.error(f"Question rewriting failed: {error}")
        logger.info(f"Rewritten query: {query}")
        return query


async def query_rag_by_document(request: QueryWithDocumentIdSchema):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not is_session_active(request.session_id):
            raise HTTPException(
                status_code=409,
                detail="The conversation has ended or does not exist.",
            )

        saved_vector_store_path = (
            f"app/data/sessions/{request.session_id}/"
            f"vectorstores/faiss_index_{request.document_id}"
        )

        if not os.path.exists(saved_vector_store_path):
            raise HTTPException(
                status_code=404,
                detail="Document ID not found for this session.",
            )

        logger.info(f"Received query: {request.query}")
        logger.info(
            f"Received chat history length: {len(request.chat_history)}"
        )

        query_category = classify_query(
            query=request.query,
            chat_history=request.chat_history,
            document_id=request.document_id,
        )

        if query_category == "CONVERSATION_HISTORY":
            answer = answer_from_chat_history(
                query=request.query,
                chat_history=request.chat_history,
                document_id=request.document_id,
            )

            save_message(
                session_id=request.session_id,
                document_id=request.document_id,
                query=request.query,
                answer=answer,
            )

            return JSONResponse(
                content={
                    "query": request.query,
                    "query_category": query_category,
                    "standalone_query": request.query,
                    "answer": answer,
                }
            )

        current_vector_store = load_vector_store(saved_vector_store_path)
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

        save_message(
            session_id=request.session_id,
            document_id=request.document_id,
            query=request.query,
            answer=answer,
        )

        return JSONResponse(
            content={
                "query": request.query,
                "query_category": query_category,
                "standalone_query": standalone_query,
                "answer": answer,
            }
        )

    except HTTPException:
        raise
    except Exception as error:
        logger.error(
            f"Error querying document with ID {request.document_id}: {error}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the query: {error}",
        ) from error
