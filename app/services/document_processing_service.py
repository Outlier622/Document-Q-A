from pathlib import Path

from app.config.configuration import Config
from app.core.logger import configure_logging
from app.processing.generate_text_chunks import generate_text_chunks_from_pdf
from app.processing.generate_vector_db import create_vector_store
from app.services.session_service import is_session_active, set_session_document
from app.services.storage_service import StorageService


logger = configure_logging("DOCUMENT_PROCESSING_SERVICE")
config = Config()
storage_service = StorageService(config)


def get_document_paths(session_id: str, document_id: str) -> dict[str, str]:
    session_base = Path("app/data/sessions") / session_id
    return {
        "pdf": str(session_base / "pdfs" / f"{document_id}.pdf"),
        "text": str(session_base / "texts" / f"{document_id}.txt"),
        "vector": str(
            session_base / "vectorstores" / f"faiss_index_{document_id}"
        ),
    }


def process_document_job(
    session_id: str,
    document_id: str,
    uploaded_filename: str,
) -> None:
    if not is_session_active(session_id):
        raise RuntimeError("The session ended before document processing started.")

    paths = get_document_paths(session_id, document_id)

    storage_service.ensure_pdf_local(
        session_id=session_id,
        document_id=document_id,
        local_pdf_path=paths["pdf"],
    )

    Path(paths["text"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["vector"]).parent.mkdir(parents=True, exist_ok=True)

    chunks = generate_text_chunks_from_pdf(paths["pdf"], paths["text"])
    create_vector_store(chunks, paths["vector"])

    storage_service.sync_processed_artifacts(
        session_id=session_id,
        document_id=document_id,
        text_path=paths["text"],
        vector_store_path=paths["vector"],
    )

    # The new document becomes active only after all artifacts are complete.
    set_session_document(
        session_id=session_id,
        document_id=document_id,
        uploaded_filename=uploaded_filename,
    )

    logger.info(
        "Document processing completed. session_id=%s document_id=%s",
        session_id,
        document_id,
    )