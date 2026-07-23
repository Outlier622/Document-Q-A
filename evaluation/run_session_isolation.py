from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import shutil
import time
import uuid
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fitz
import requests

from evaluation.metrics import mean, percentile


def create_pdf(path: Path, marker: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        f"Session isolation test document. Unique marker: {marker}",
        fontsize=12,
    )
    document.save(path)
    document.close()


def request_json(method: str, url: str, **kwargs) -> tuple[requests.Response, float]:
    started = time.perf_counter()
    response = requests.request(method, url, **kwargs)
    return response, (time.perf_counter() - started) * 1000.0


def create_and_upload(base_url: str, pdf_dir: Path, index: int, timeout: float) -> dict[str, Any]:
    client_id = f"isolation-client-{index:04d}-{uuid.uuid4()}"
    marker = f"SESSION-{index:04d}-{uuid.uuid4().hex[:12].upper()}"
    pdf_path = pdf_dir / f"session_{index:04d}.pdf"
    create_pdf(pdf_path, marker)

    session_response, start_latency = request_json(
        "POST",
        f"{base_url}/rag/sessions/start-or-resume",
        json={"client_id": client_id},
        timeout=timeout,
    )
    session_response.raise_for_status()
    session_id = session_response.json()["session_id"]

    with pdf_path.open("rb") as handle:
        upload_response, upload_latency = request_json(
            "POST",
            f"{base_url}/rag/upload-document-pdf",
            files={"pdf_file": (pdf_path.name, handle, "application/pdf")},
            data={"session_id": session_id},
            timeout=timeout,
        )
    upload_response.raise_for_status()
    payload = upload_response.json()
    return {
        "client_id": client_id,
        "session_id": session_id,
        "document_id": payload["document_id"],
        "marker": marker,
        "pdf_path": str(pdf_path),
        "start_latency_ms": start_latency,
        "upload_latency_ms": upload_latency,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black-box API test for the repository's current session-scoped storage boundary."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--run-own-query", action="store_true")
    parser.add_argument(
        "--work-dir", type=Path, default=Path("evaluation/results/isolation/generated_pdfs")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/results/isolation")
    )
    parser.add_argument("--keep-pdfs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sessions: list[dict[str, Any]] = []
    setup_errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                create_and_upload, base_url, args.work_dir, index, args.timeout
            ): index
            for index in range(args.sessions)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                sessions.append(future.result())
            except Exception as error:
                setup_errors.append(f"session-{index}: {type(error).__name__}: {error}")
    sessions.sort(key=lambda item: item["client_id"])

    attempts: list[dict[str, Any]] = []
    cross_rejected = 0
    cross_leaks = 0
    cross_latencies: list[float] = []
    for attacker in sessions:
        for victim in sessions:
            if attacker["session_id"] == victim["session_id"]:
                continue
            response, latency = request_json(
                "POST",
                f"{base_url}/rag/query-by-document",
                json={
                    "session_id": attacker["session_id"],
                    "document_id": victim["document_id"],
                    "query": f"What is the unique marker {victim['marker']}?",
                    "chat_history": [],
                },
                timeout=args.timeout,
            )
            cross_latencies.append(latency)
            rejected = response.status_code in {403, 404}
            leaked = victim["marker"] in response.text
            cross_rejected += int(rejected)
            cross_leaks += int(leaked)
            attempts.append(
                {
                    "type": "cross_session_query",
                    "attacker_session": attacker["session_id"],
                    "victim_document": victim["document_id"],
                    "status_code": response.status_code,
                    "rejected": rejected,
                    "marker_leaked": leaked,
                    "latency_ms": latency,
                    "response_excerpt": response.text[:300],
                }
            )

    own_query_success = 0
    own_query_latencies: list[float] = []
    if args.run_own_query:
        for item in sessions:
            response, latency = request_json(
                "POST",
                f"{base_url}/rag/query-by-document",
                json={
                    "session_id": item["session_id"],
                    "document_id": item["document_id"],
                    "query": "What is the unique marker?",
                    "chat_history": [],
                },
                timeout=args.timeout,
            )
            own_query_latencies.append(latency)
            succeeded = response.status_code == 200 and item["marker"] in response.text
            own_query_success += int(succeeded)
            attempts.append(
                {
                    "type": "own_session_query",
                    "attacker_session": item["session_id"],
                    "victim_document": item["document_id"],
                    "status_code": response.status_code,
                    "rejected": False,
                    "marker_leaked": False,
                    "latency_ms": latency,
                    "response_excerpt": response.text[:300],
                }
            )

    ended_query_rejections = 0
    ended_upload_rejections = 0
    for item in sessions:
        end_response, end_latency = request_json(
            "POST",
            f"{base_url}/rag/sessions/{item['session_id']}/end",
            json={"client_id": item["client_id"]},
            timeout=args.timeout,
        )
        attempts.append(
            {
                "type": "end_session",
                "attacker_session": item["session_id"],
                "victim_document": item["document_id"],
                "status_code": end_response.status_code,
                "rejected": False,
                "marker_leaked": False,
                "latency_ms": end_latency,
                "response_excerpt": end_response.text[:300],
            }
        )

        query_response, query_latency = request_json(
            "POST",
            f"{base_url}/rag/query-by-document",
            json={
                "session_id": item["session_id"],
                "document_id": item["document_id"],
                "query": "What is the unique marker?",
                "chat_history": [],
            },
            timeout=args.timeout,
        )
        ended_query_rejections += int(query_response.status_code == 409)
        attempts.append(
            {
                "type": "ended_session_query",
                "attacker_session": item["session_id"],
                "victim_document": item["document_id"],
                "status_code": query_response.status_code,
                "rejected": query_response.status_code == 409,
                "marker_leaked": item["marker"] in query_response.text,
                "latency_ms": query_latency,
                "response_excerpt": query_response.text[:300],
            }
        )

        pdf_path = Path(item["pdf_path"])
        with pdf_path.open("rb") as handle:
            upload_response, upload_latency = request_json(
                "POST",
                f"{base_url}/rag/upload-document-pdf",
                files={"pdf_file": (pdf_path.name, handle, "application/pdf")},
                data={"session_id": item["session_id"]},
                timeout=args.timeout,
            )
        ended_upload_rejections += int(upload_response.status_code == 409)
        attempts.append(
            {
                "type": "ended_session_upload",
                "attacker_session": item["session_id"],
                "victim_document": item["document_id"],
                "status_code": upload_response.status_code,
                "rejected": upload_response.status_code == 409,
                "marker_leaked": False,
                "latency_ms": upload_latency,
                "response_excerpt": upload_response.text[:300],
            }
        )

    removed_endpoint_results = []
    for endpoint in ("/rag/list-vector-stores", "/rag/pdf/nonexistent-document"):
        response, latency = request_json(
            "GET", f"{base_url}{endpoint}", timeout=args.timeout
        )
        removed_endpoint_results.append(
            {
                "endpoint": endpoint,
                "status_code": response.status_code,
                "inaccessible": response.status_code in {404, 405},
                "latency_ms": latency,
            }
        )

    cross_attempts = len(sessions) * max(0, len(sessions) - 1)
    upload_latencies = [item["upload_latency_ms"] for item in sessions]
    summary = {
        "scope_warning": (
            "This validates session-scoped path isolation in the current repository. "
            "It does not prove authenticated ownership because upload/query requests "
            "currently accept session_id without a server-issued ownership token."
        ),
        "configuration": {
            "base_url": base_url,
            "sessions_requested": args.sessions,
            "workers": args.workers,
            "run_own_query": args.run_own_query,
        },
        "setup": {
            "sessions_uploaded": len(sessions),
            "sessions_requested": args.sessions,
            "success_rate": len(sessions) / args.sessions if args.sessions else None,
            "errors": setup_errors,
            "upload_latency_ms": {
                "mean": mean(upload_latencies),
                "p50": percentile(upload_latencies, 50),
                "p95": percentile(upload_latencies, 95),
            },
        },
        "cross_session": {
            "attempts": cross_attempts,
            "rejected": cross_rejected,
            "rejection_rate": cross_rejected / cross_attempts if cross_attempts else None,
            "marker_leaks": cross_leaks,
            "latency_ms": {
                "mean": mean(cross_latencies),
                "p50": percentile(cross_latencies, 50),
                "p95": percentile(cross_latencies, 95),
            },
        },
        "own_session_queries": {
            "tested": len(sessions) if args.run_own_query else 0,
            "successful_with_marker": own_query_success,
            "latency_ms": {
                "mean": mean(own_query_latencies),
                "p50": percentile(own_query_latencies, 50),
                "p95": percentile(own_query_latencies, 95),
            },
        },
        "ended_session": {
            "query_tests": len(sessions),
            "queries_rejected_with_409": ended_query_rejections,
            "upload_tests": len(sessions),
            "uploads_rejected_with_409": ended_upload_rejections,
        },
        "removed_endpoints": removed_endpoint_results,
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "attempts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = list(attempts[0].keys()) if attempts else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attempts)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.keep_pdfs:
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
