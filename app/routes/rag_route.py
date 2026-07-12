import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.logger import configure_logging
from app.schemas.rag_schema import (
    EndSessionSchema,
    QueryOnlySchema,
    QueryWithDocumentIdSchema,
    QueryWithReferenceSchema,
    StartOrResumeSessionSchema,
)
from app.services.rag_service import (
    generate_vector_store_for_pdf,
    query_rag_by_document,
    query_rag_with_reference,
    query_rag_without_reference,
)
from app.services.session_service import (
    end_session,
    is_session_active,
    start_or_resume_session,
)


router = APIRouter()
logger = configure_logging("RAG_ROUTE")


@router.get("/")
async def get_index():
    return {"message": "Welcome to the Document Question Answering API!"}


@router.post("/query")
async def query_rag(request: QueryOnlySchema):
    return await query_rag_without_reference(request)


@router.post("/query-with-reference")
async def query_rag_reference(request: QueryWithReferenceSchema):
    return await query_rag_with_reference(request)


@router.post("/sessions/start-or-resume")
async def start_or_resume_anonymous_session(
    request: StartOrResumeSessionSchema,
):
    try:
        session_data = start_or_resume_session(request.client_id)
        return JSONResponse(content=session_data)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.post("/sessions/{session_id}/end")
async def end_anonymous_session(
    session_id: str,
    request: EndSessionSchema,
):
    try:
        result = end_session(
            client_id=request.client_id,
            session_id=session_id,
        )

        return JSONResponse(content=result)

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.post("/upload-document-pdf")
async def upload_pdf(
    pdf_file: UploadFile = File(...),
    session_id: str = Form(...),
):
    if not is_session_active(session_id):
        raise HTTPException(
            status_code=409,
            detail="The conversation has ended or does not exist.",
        )

    document_id = str(uuid.uuid4())
    session_base_path = f"app/data/sessions/{session_id}"

    saved_pdf_path = f"{session_base_path}/pdfs/{document_id}.pdf"
    output_text_file_path = f"{session_base_path}/texts/{document_id}.txt"
    saved_vector_store_path = (
        f"{session_base_path}/vectorstores/faiss_index_{document_id}"
    )

    logger.info(
        f"Uploading PDF. session_id={session_id}, "
        f"document_id={document_id}, "
        f"vector_store_path={saved_vector_store_path}"
    )

    return await generate_vector_store_for_pdf(
        pdf_file=pdf_file,
        session_id=session_id,
        document_id=document_id,
        saved_pdf_path=saved_pdf_path,
        output_text_file_path=output_text_file_path,
        saved_vector_store_path=saved_vector_store_path,
    )


@router.post("/query-by-document")
async def query_by_document(request: QueryWithDocumentIdSchema):
    logger.info(
        f"Route received query-by-document request. "
        f"session_id={request.session_id}, "
        f"document_id={request.document_id}, "
        f"query={request.query}, "
        f"chat_history_length={len(request.chat_history)}"
    )

    return await query_rag_by_document(request)
