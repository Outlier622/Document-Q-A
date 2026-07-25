from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class SessionRecord(Base):
    __tablename__ = "sessions"
    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    client_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'ENDED')", name="ck_sessions_status"),
        Index("idx_sessions_client_status", "client_id", "status", "updated_at"),
    )


class MessageRecord(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    query_category: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    web_search_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    web_sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (Index("idx_messages_session", "session_id", "id"),)


class ProcessingJobRecord(Base):
    __tablename__ = "processing_jobs"
    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_filename: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (
        CheckConstraint("status IN ('PENDING', 'PROCESSING', 'RETRYING', 'COMPLETED', 'FAILED', 'CANCELLED')", name="ck_processing_jobs_status"),
        Index("idx_processing_jobs_session_status", "session_id", "status", "updated_at"),
    )