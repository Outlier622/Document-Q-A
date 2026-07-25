import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update

from app.database.database import SessionLocal, initialize_database
from app.database.models import (
    MessageRecord,
    ProcessingJobRecord,
    SessionRecord,
)


ACTIVE_JOB_STATUSES = (
    "PENDING",
    "PROCESSING",
    "RETRYING",
)


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_web_sources(raw_value: str | None) -> list[dict]:
    """Deserialize persisted web-source metadata safely."""
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return []

    return parsed if isinstance(parsed, list) else []


def _job_to_dict(
    job: ProcessingJobRecord | None,
) -> dict | None:
    """Convert a processing-job ORM record into the existing API shape."""
    if job is None:
        return None

    return {
        "job_id": job.job_id,
        "session_id": job.session_id,
        "document_id": job.document_id,
        "uploaded_filename": job.uploaded_filename,
        "status": job.status,
        "attempt_count": job.attempt_count,
        "last_error": job.last_error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "updated_at": job.updated_at,
    }


def _build_session_payload(
    db,
    session_record: SessionRecord,
) -> dict:
    """Build the same session payload previously returned by SQLite code."""
    message_rows = db.scalars(
        select(MessageRecord)
        .where(
            MessageRecord.session_id == session_record.session_id,
        )
        .order_by(
            MessageRecord.id.asc(),
        )
    ).all()

    chat_history = []

    for row in message_rows:
        history_item = {
            "document_id": row.document_id,
            "query": row.query,
            "answer": row.answer,
            "web_search_used": bool(row.web_search_used),
            "web_sources": _parse_web_sources(
                row.web_sources_json,
            ),
        }

        if row.query_category is not None:
            history_item["query_category"] = row.query_category

        if row.source_type is not None:
            history_item["source_type"] = row.source_type

        chat_history.append(history_item)

    return {
        "session_id": session_record.session_id,
        "client_id": session_record.client_id,
        "status": session_record.status,
        "document_id": session_record.document_id,
        "uploaded_filename": session_record.uploaded_filename,
        "created_at": session_record.created_at,
        "updated_at": session_record.updated_at,
        "chat_history": chat_history,
    }


def start_or_resume_session(client_id: str) -> dict:
    """Resume the latest active session for a client or create a new one."""
    client_id = client_id.strip()

    if not client_id:
        raise ValueError("client_id cannot be empty")

    now = _utc_now()

    with SessionLocal.begin() as db:
        session_record = db.scalar(
            select(SessionRecord)
            .where(
                SessionRecord.client_id == client_id,
                SessionRecord.status == "ACTIVE",
            )
            .order_by(
                SessionRecord.updated_at.desc(),
            )
            .limit(1)
        )

        if session_record is None:
            session_record = SessionRecord(
                session_id=str(uuid.uuid4()),
                client_id=client_id,
                status="ACTIVE",
                document_id=None,
                uploaded_filename=None,
                created_at=now,
                updated_at=now,
            )

            db.add(session_record)
            db.flush()

        else:
            session_record.updated_at = now
            db.flush()

        return _build_session_payload(
            db=db,
            session_record=session_record,
        )


def is_session_active(session_id: str) -> bool:
    """Return True only when the requested session exists and is active."""
    with SessionLocal() as db:
        active_session_id = db.scalar(
            select(SessionRecord.session_id).where(
                SessionRecord.session_id == session_id,
                SessionRecord.status == "ACTIVE",
            )
        )

    return active_session_id is not None


def set_session_document(
    session_id: str,
    document_id: str,
    uploaded_filename: str,
) -> None:
    """Attach a processed PDF as the active document for a session.

    The application keeps one active PDF per conversation. When a replacement
    PDF becomes active, document-linked turns from the previous PDF are removed.
    General-knowledge and web-only turns have document_id=None and are preserved.
    """
    now = _utc_now()

    with SessionLocal.begin() as db:
        session_record = db.scalar(
            select(SessionRecord)
            .where(
                SessionRecord.session_id == session_id,
                SessionRecord.status == "ACTIVE",
            )
            .with_for_update()
        )

        if session_record is None:
            raise ValueError("Active session not found")

        session_record.document_id = document_id
        session_record.uploaded_filename = uploaded_filename
        session_record.updated_at = now

        db.execute(
            delete(MessageRecord).where(
                MessageRecord.session_id == session_id,
                MessageRecord.document_id.is_not(None),
            )
        )


def save_message(
    session_id: str,
    document_id: str | None,
    query: str,
    answer: str,
    query_category: str | None = None,
    source_type: str | None = None,
    web_search_used: bool = False,
    web_sources: list[dict] | None = None,
) -> None:
    """Persist one assistant turn and its routing/source metadata."""
    now = _utc_now()

    serialized_sources = json.dumps(
        web_sources or [],
        ensure_ascii=False,
    )

    with SessionLocal.begin() as db:
        session_record = db.scalar(
            select(SessionRecord)
            .where(
                SessionRecord.session_id == session_id,
                SessionRecord.status == "ACTIVE",
            )
            .with_for_update()
        )

        if session_record is None:
            raise ValueError("Active session not found")

        message_record = MessageRecord(
            session_id=session_id,
            document_id=document_id,
            query=query,
            answer=answer,
            query_category=query_category,
            source_type=source_type,
            web_search_used=int(web_search_used),
            web_sources_json=serialized_sources,
            created_at=now,
        )

        db.add(message_record)
        session_record.updated_at = now


def create_processing_job(
    job_id: str,
    session_id: str,
    document_id: str,
    uploaded_filename: str,
) -> dict:
    """Create one asynchronous document-processing job for a session.

    The parent session row is locked while checking for active jobs. PostgreSQL
    therefore serializes competing uploads for the same session. SQLite ignores
    SELECT ... FOR UPDATE, but the same service interface remains usable for
    local regression testing.
    """
    now = _utc_now()

    with SessionLocal.begin() as db:
        session_record = db.scalar(
            select(SessionRecord)
            .where(
                SessionRecord.session_id == session_id,
                SessionRecord.status == "ACTIVE",
            )
            .with_for_update()
        )

        if session_record is None:
            raise ValueError("Active session not found")

        existing_job = db.scalar(
            select(ProcessingJobRecord)
            .where(
                ProcessingJobRecord.session_id == session_id,
                ProcessingJobRecord.status.in_(
                    ACTIVE_JOB_STATUSES,
                ),
            )
            .order_by(
                ProcessingJobRecord.created_at.desc(),
            )
            .limit(1)
        )

        if existing_job is not None:
            raise ValueError(
                "A document is already being processed for this session."
            )

        job_record = ProcessingJobRecord(
            job_id=job_id,
            session_id=session_id,
            document_id=document_id,
            uploaded_filename=uploaded_filename,
            status="PENDING",
            attempt_count=0,
            last_error=None,
            created_at=now,
            started_at=None,
            completed_at=None,
            updated_at=now,
        )

        db.add(job_record)
        db.flush()

        return _job_to_dict(job_record)


def get_processing_job(job_id: str) -> dict | None:
    """Return the current persisted state of one processing job."""
    with SessionLocal() as db:
        job_record = db.get(
            ProcessingJobRecord,
            job_id,
        )

        return _job_to_dict(job_record)


def mark_processing_job_processing(
    job_id: str,
    attempt_count: int,
) -> None:
    """Mark a job as actively being processed by a worker."""
    now = _utc_now()

    with SessionLocal.begin() as db:
        job_record = db.get(
            ProcessingJobRecord,
            job_id,
        )

        if job_record is None:
            raise ValueError("Processing job not found")

        job_record.status = "PROCESSING"
        job_record.attempt_count = attempt_count

        if job_record.started_at is None:
            job_record.started_at = now

        job_record.last_error = None
        job_record.updated_at = now


def mark_processing_job_retrying(
    job_id: str,
    attempt_count: int,
    error_message: str,
) -> None:
    """Persist a recoverable processing failure before SQS redelivery."""
    now = _utc_now()

    with SessionLocal.begin() as db:
        job_record = db.get(
            ProcessingJobRecord,
            job_id,
        )

        if job_record is None:
            raise ValueError("Processing job not found")

        job_record.status = "RETRYING"
        job_record.attempt_count = attempt_count
        job_record.last_error = error_message
        job_record.updated_at = now


def mark_processing_job_completed(job_id: str) -> None:
    """Mark a document-processing job as successfully completed."""
    now = _utc_now()

    with SessionLocal.begin() as db:
        job_record = db.get(
            ProcessingJobRecord,
            job_id,
        )

        if job_record is None:
            raise ValueError("Processing job not found")

        job_record.status = "COMPLETED"
        job_record.last_error = None
        job_record.completed_at = now
        job_record.updated_at = now


def mark_processing_job_failed(
    job_id: str,
    attempt_count: int,
    error_message: str,
) -> None:
    """Mark a processing job as permanently failed."""
    now = _utc_now()

    with SessionLocal.begin() as db:
        job_record = db.get(
            ProcessingJobRecord,
            job_id,
        )

        if job_record is None:
            raise ValueError("Processing job not found")

        job_record.status = "FAILED"
        job_record.attempt_count = attempt_count
        job_record.last_error = error_message
        job_record.completed_at = now
        job_record.updated_at = now


def mark_processing_job_cancelled(
    job_id: str,
    reason: str = "Session ended before processing completed.",
) -> None:
    """Cancel a processing job when its parent conversation has ended."""
    now = _utc_now()

    with SessionLocal.begin() as db:
        job_record = db.get(
            ProcessingJobRecord,
            job_id,
        )

        if job_record is None:
            return

        job_record.status = "CANCELLED"
        job_record.last_error = reason
        job_record.completed_at = now
        job_record.updated_at = now


def end_session(
    client_id: str,
    session_id: str,
) -> dict:
    """End a conversation and cancel its unfinished document jobs."""
    now = _utc_now()

    with SessionLocal.begin() as db:
        session_record = db.scalar(
            select(SessionRecord)
            .where(
                SessionRecord.session_id == session_id,
                SessionRecord.client_id == client_id,
                SessionRecord.status == "ACTIVE",
            )
            .with_for_update()
        )

        if session_record is None:
            raise ValueError(
                "Active session not found for this client"
            )

        session_record.status = "ENDED"
        session_record.updated_at = now

        db.execute(
            update(ProcessingJobRecord)
            .where(
                ProcessingJobRecord.session_id == session_id,
                ProcessingJobRecord.status.in_(
                    ACTIVE_JOB_STATUSES,
                ),
            )
            .values(
                status="CANCELLED",
                last_error=(
                    "Session ended before processing completed."
                ),
                completed_at=now,
                updated_at=now,
            )
        )

    return {
        "session_id": session_id,
        "status": "ENDED",
    }


# Create missing tables on application startup. Existing tables are left intact.
initialize_database()