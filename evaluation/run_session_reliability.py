from __future__ import annotations

import argparse
import concurrent.futures
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import mean, percentile  # noqa: E402
from app.services import session_service  # noqa: E402


def configure_db(db_path: Path) -> None:
    session_service.DB_PATH = db_path.resolve()
    session_service.initialize_session_database()


def timed_call(latencies: list[float], function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    latencies.append((time.perf_counter() - started) * 1000.0)
    return result


def expected_messages(client_index: int, count: int, document_id: str) -> list[dict[str, str]]:
    return [
        {
            "document_id": document_id,
            "query": f"client-{client_index}-query-{message_index}",
            "answer": f"client-{client_index}-answer-{message_index}",
        }
        for message_index in range(count)
    ]


def create_client(
    client_index: int,
    messages_per_session: int,
    operation_latencies: list[float],
) -> dict[str, Any]:
    client_id = f"reliability-client-{client_index:04d}"
    session = timed_call(
        operation_latencies, session_service.start_or_resume_session, client_id
    )
    document_id = f"reliability-document-{client_index:04d}"
    timed_call(
        operation_latencies,
        session_service.set_session_document,
        session["session_id"],
        document_id,
        f"document-{client_index:04d}.pdf",
    )
    messages = expected_messages(client_index, messages_per_session, document_id)
    for message in messages:
        timed_call(
            operation_latencies,
            session_service.save_message,
            session["session_id"],
            document_id,
            message["query"],
            message["answer"],
        )
    return {
        "client_id": client_id,
        "session_id": session["session_id"],
        "document_id": document_id,
        "messages": messages,
    }


def verify_expected(db_path: Path, expected_path: Path) -> dict[str, Any]:
    configure_db(db_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    recovered_sessions = 0
    recovered_messages = 0
    lost_messages = 0
    duplicate_messages = 0
    ordering_errors = 0
    wrong_document_sessions = 0

    for item in expected:
        payload = session_service.start_or_resume_session(item["client_id"])
        if payload["session_id"] == item["session_id"]:
            recovered_sessions += 1
        if payload.get("document_id") != item["document_id"]:
            wrong_document_sessions += 1

        actual_messages = payload.get("chat_history") or []
        expected_items = item["messages"]
        recovered_messages += sum(
            actual == expected_message
            for actual, expected_message in zip(actual_messages, expected_items)
        )
        lost_messages += max(0, len(expected_items) - len(actual_messages))
        duplicate_messages += max(0, len(actual_messages) - len(expected_items))
        if actual_messages != expected_items:
            ordering_errors += 1

    return {
        "recovered_sessions": recovered_sessions,
        "total_sessions": len(expected),
        "recovered_messages": recovered_messages,
        "total_messages": sum(len(item["messages"]) for item in expected),
        "lost_messages": lost_messages,
        "duplicate_messages": duplicate_messages,
        "sessions_with_ordering_or_content_error": ordering_errors,
        "wrong_document_sessions": wrong_document_sessions,
    }


def run_child_verification(script_path: Path, db_path: Path, expected_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--verify-only",
            "--db-path",
            str(db_path),
            "--expected-path",
            str(expected_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process-restart and concurrency benchmark for session_service.py."
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path("evaluation/results/session/session_test.db")
    )
    parser.add_argument(
        "--expected-path",
        type=Path,
        default=Path("evaluation/results/session/expected_sessions.json"),
    )
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--messages-per-session", type=int, default=5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--restart-cycles", type=int, default=10)
    parser.add_argument("--reset-sessions", type=int, default=10)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/results/session/summary.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        print(json.dumps(verify_expected(args.db_path, args.expected_path)))
        return

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    args.expected_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.db_path.exists() and not args.keep_db:
        args.db_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(args.db_path) + suffix)
        if sidecar.exists() and not args.keep_db:
            sidecar.unlink()

    configure_db(args.db_path)
    operation_latencies: list[float] = []
    creation_errors: list[str] = []
    expected: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                create_client,
                index,
                args.messages_per_session,
                operation_latencies,
            ): index
            for index in range(args.sessions)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                expected.append(future.result())
            except Exception as error:  # benchmark must record, not hide, failures
                creation_errors.append(f"client-{index}: {type(error).__name__}: {error}")

    expected.sort(key=lambda item: item["client_id"])
    args.expected_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    restart_results = [
        run_child_verification(Path(__file__).resolve(), args.db_path, args.expected_path)
        for _ in range(args.restart_cycles)
    ]

    reset_count = min(args.reset_sessions, len(expected))
    reset_success = 0
    for item in expected[:reset_count]:
        new_document_id = f"{item['document_id']}-replacement"
        session_service.set_session_document(
            item["session_id"], new_document_id, "replacement.pdf"
        )
        payload = session_service.start_or_resume_session(item["client_id"])
        if payload.get("document_id") == new_document_id and payload.get("chat_history") == []:
            reset_success += 1

    ended_success = 0
    for item in expected:
        try:
            session_service.end_session(item["client_id"], item["session_id"])
            ended_success += 1
        except ValueError:
            pass

    ended_resurrections = 0
    fresh_sessions_after_end = 0
    for item in expected:
        payload = session_service.start_or_resume_session(item["client_id"])
        if payload["session_id"] == item["session_id"]:
            ended_resurrections += 1
        else:
            fresh_sessions_after_end += 1

    with sqlite3.connect(args.db_path) as connection:
        original_ended_rows = connection.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'ENDED'"
        ).fetchone()[0]

    total_expected_messages = len(expected) * args.messages_per_session
    aggregate_recovered_messages = sum(
        result["recovered_messages"] for result in restart_results
    )
    aggregate_total_messages = sum(result["total_messages"] for result in restart_results)

    summary = {
        "configuration": {
            "db_path": str(args.db_path),
            "sessions_requested": args.sessions,
            "messages_per_session": args.messages_per_session,
            "workers": args.workers,
            "restart_cycles": args.restart_cycles,
            "reset_sessions": reset_count,
        },
        "creation": {
            "sessions_created": len(expected),
            "sessions_requested": args.sessions,
            "success_rate": len(expected) / args.sessions if args.sessions else None,
            "errors": creation_errors,
            "messages_written": total_expected_messages,
        },
        "restart_recovery": {
            "cycles": restart_results,
            "session_recovery_rate": mean(
                result["recovered_sessions"] / result["total_sessions"]
                for result in restart_results
                if result["total_sessions"]
            ),
            "message_recovery_rate": (
                aggregate_recovered_messages / aggregate_total_messages
                if aggregate_total_messages
                else None
            ),
            "lost_messages": sum(result["lost_messages"] for result in restart_results),
            "duplicate_messages": sum(
                result["duplicate_messages"] for result in restart_results
            ),
            "sessions_with_ordering_or_content_error": sum(
                result["sessions_with_ordering_or_content_error"]
                for result in restart_results
            ),
        },
        "document_reset": {
            "tested": reset_count,
            "successful": reset_success,
            "success_rate": reset_success / reset_count if reset_count else None,
        },
        "ended_session_behavior": {
            "end_calls_succeeded": ended_success,
            "original_ended_rows": original_ended_rows,
            "ended_session_resurrections": ended_resurrections,
            "fresh_sessions_created_after_end": fresh_sessions_after_end,
        },
        "operation_latency_ms": {
            "count": len(operation_latencies),
            "mean": mean(operation_latencies),
            "p50": percentile(operation_latencies, 50),
            "p95": percentile(operation_latencies, 95),
            "p99": percentile(operation_latencies, 99),
        },
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
