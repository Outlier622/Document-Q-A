import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path("app/data/app.db")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")

    return connection


def _ensure_message_metadata_columns(
    connection: sqlite3.Connection,
) -> None:
    """Add assistant metadata columns to an existing messages table."""
    existing_columns = {
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(messages)"
        ).fetchall()
    }

    column_statements = {
        "query_category": "ALTER TABLE messages ADD COLUMN query_category TEXT",
        "source_type": "ALTER TABLE messages ADD COLUMN source_type TEXT",
        "web_search_used": (
            "ALTER TABLE messages ADD COLUMN web_search_used "
            "INTEGER NOT NULL DEFAULT 0"
        ),
        "web_sources_json": (
            "ALTER TABLE messages ADD COLUMN web_sources_json TEXT"
        ),
    }

    for column_name, statement in column_statements.items():
        if column_name not in existing_columns:
            connection.execute(statement)


def initialize_session_database() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('ACTIVE', 'ENDED')),
                document_id TEXT,
                uploaded_filename TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                document_id TEXT,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                query_category TEXT,
                source_type TEXT,
                web_search_used INTEGER NOT NULL DEFAULT 0,
                web_sources_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_client_status
                ON sessions(client_id, status, updated_at);

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);
            """
        )

        # Existing app.db files were created before these nullable metadata
        # fields existed. Add them without deleting or rebuilding the database.
        _ensure_message_metadata_columns(connection)


def _parse_web_sources(raw_value: str | None) -> list[dict]:
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return []

    return parsed if isinstance(parsed, list) else []


def _build_session_payload(
    connection: sqlite3.Connection,
    session_row: sqlite3.Row,
) -> dict:
    message_rows = connection.execute(
        """
        SELECT
            document_id,
            query,
            answer,
            query_category,
            source_type,
            web_search_used,
            web_sources_json,
            created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_row["session_id"],),
    ).fetchall()

    chat_history = []

    for row in message_rows:
        history_item = {
            "document_id": row["document_id"],
            "query": row["query"],
            "answer": row["answer"],
            "web_search_used": bool(row["web_search_used"]),
            "web_sources": _parse_web_sources(row["web_sources_json"]),
        }

        if row["query_category"] is not None:
            history_item["query_category"] = row["query_category"]

        if row["source_type"] is not None:
            history_item["source_type"] = row["source_type"]

        chat_history.append(history_item)

    return {
        "session_id": session_row["session_id"],
        "client_id": session_row["client_id"],
        "status": session_row["status"],
        "document_id": session_row["document_id"],
        "uploaded_filename": session_row["uploaded_filename"],
        "created_at": session_row["created_at"],
        "updated_at": session_row["updated_at"],
        "chat_history": chat_history,
    }


def start_or_resume_session(client_id: str) -> dict:
    client_id = client_id.strip()

    if not client_id:
        raise ValueError("client_id cannot be empty")

    now = _utc_now()

    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        session_row = connection.execute(
            """
            SELECT *
            FROM sessions
            WHERE client_id = ? AND status = 'ACTIVE'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (client_id,),
        ).fetchone()

        if session_row is None:
            session_id = str(uuid.uuid4())

            connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    client_id,
                    status,
                    document_id,
                    uploaded_filename,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, 'ACTIVE', NULL, NULL, ?, ?)
                """,
                (session_id, client_id, now, now),
            )

            session_row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            connection.execute(
                """
                UPDATE sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (now, session_row["session_id"]),
            )

            session_row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_row["session_id"],),
            ).fetchone()

        return _build_session_payload(connection, session_row)


def is_session_active(session_id: str) -> bool:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM sessions
            WHERE session_id = ? AND status = 'ACTIVE'
            """,
            (session_id,),
        ).fetchone()

    return row is not None


def set_session_document(
    session_id: str,
    document_id: str,
    uploaded_filename: str,
) -> None:
    """Attach a new active document while preserving independent turns.

    The application supports one active PDF per conversation. Messages tied to
    a previous PDF are removed so they cannot affect the new document. General
    and web-only assistant turns have document_id NULL and remain available.
    """
    now = _utc_now()

    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE sessions
            SET document_id = ?,
                uploaded_filename = ?,
                updated_at = ?
            WHERE session_id = ? AND status = 'ACTIVE'
            """,
            (
                document_id,
                uploaded_filename,
                now,
                session_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError("Active session not found")

        connection.execute(
            """
            DELETE FROM messages
            WHERE session_id = ?
              AND document_id IS NOT NULL
            """,
            (session_id,),
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
    now = _utc_now()
    serialized_sources = json.dumps(
        web_sources or [],
        ensure_ascii=False,
    )

    with _connect() as connection:
        active_row = connection.execute(
            """
            SELECT 1
            FROM sessions
            WHERE session_id = ? AND status = 'ACTIVE'
            """,
            (session_id,),
        ).fetchone()

        if active_row is None:
            raise ValueError("Active session not found")

        connection.execute(
            """
            INSERT INTO messages (
                session_id,
                document_id,
                query,
                answer,
                query_category,
                source_type,
                web_search_used,
                web_sources_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                document_id,
                query,
                answer,
                query_category,
                source_type,
                int(web_search_used),
                serialized_sources,
                now,
            ),
        )

        connection.execute(
            """
            UPDATE sessions
            SET updated_at = ?
            WHERE session_id = ?
            """,
            (now, session_id),
        )


def end_session(client_id: str, session_id: str) -> dict:
    now = _utc_now()

    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE sessions
            SET status = 'ENDED',
                updated_at = ?
            WHERE session_id = ?
              AND client_id = ?
              AND status = 'ACTIVE'
            """,
            (
                now,
                session_id,
                client_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError("Active session not found for this client")

    return {
        "session_id": session_id,
        "status": "ENDED",
    }


initialize_session_database()