from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Run from repository root so imports resolve against the existing app package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import (  # noqa: E402
    classification_report,
    fact_accuracy,
    grouped_mean,
    mean,
    percentile,
    retrieval_scores,
)
from app.processing.generate_rag_chain import create_rag_chain, initialize_llm  # noqa: E402
from app.processing.generate_vector_db import load_vector_store  # noqa: E402

LABELS = ["DOCUMENT", "FOLLOW_UP_DOCUMENT", "CONVERSATION_HISTORY"]
VALID_LABELS = frozenset(LABELS)
ROUTER_LABEL_PATTERN = re.compile(
    r"(?<![A-Z0-9_])"
    r"(CONVERSATION_HISTORY|FOLLOW_UP_DOCUMENT|DOCUMENT)"
    r"(?![A-Z0-9_])"
)
MODES = ["baseline", "router_only", "full"]


@dataclass
class ApiRateLimiter:
    min_interval_seconds: float = 13.0
    _last_call_started: float | None = None
    total_wait_seconds: float = 0.0

    def sleep(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        if seconds:
            time.sleep(seconds)
            self.total_wait_seconds += seconds

    def wait(self) -> None:
        if self._last_call_started is not None:
            elapsed = time.monotonic() - self._last_call_started
            remaining = self.min_interval_seconds - elapsed
            if remaining > 0:
                print(f"Rate-limit pause: {remaining:.1f}s", flush=True)
                self.sleep(remaining)
        self._last_call_started = time.monotonic()


def _retry_delay_seconds(error: Exception, fallback: float) -> float:
    message = str(error)
    match = re.search(r"Please retry in\s+([0-9.]+)s", message, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0
    return fallback


def call_gemini(
    operation: Any,
    limiter: ApiRateLimiter,
    max_retries: int,
    retry_base_seconds: float,
) -> Any:
    for attempt in range(max_retries + 1):
        limiter.wait()
        try:
            return operation()
        except Exception as error:
            is_rate_limit = "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error).upper()
            if not is_rate_limit or attempt >= max_retries:
                raise
            wait_seconds = _retry_delay_seconds(
                error, fallback=retry_base_seconds * (2 ** min(attempt, 4))
            )
            print(
                f"Gemini rate limit reached; retrying in {wait_seconds:.1f}s "
                f"({attempt + 1}/{max_retries}).",
                flush=True,
            )
            limiter.sleep(wait_seconds)
            # The explicit retry wait already exceeds the normal request interval.
            limiter._last_call_started = None
    raise RuntimeError("Unreachable retry state")


@dataclass
class DocumentRuntime:
    vector_store: Any
    rag_chain: Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


def format_chat_history(history: Iterable[dict[str, Any]], document_id: str, max_turns: int | None) -> str:
    selected = [
        item
        for item in history
        if item.get("document_id", document_id) == document_id
    ]
    if max_turns is not None:
        selected = selected[-max_turns:]
    lines: list[str] = []
    for index, item in enumerate(selected, start=1):
        query = str(item.get("query") or "")
        answer = str(item.get("answer") or "")
        if query:
            lines.append(f"Turn {index} User: {query}")
        if answer:
            lines.append(f"Turn {index} Assistant: {answer}")
    return "\n".join(lines)


def invoke_text(
    llm: Any,
    prompt: str,
    limiter: ApiRateLimiter,
    max_retries: int,
    retry_base_seconds: float,
) -> str:
    response = call_gemini(
        lambda: llm.invoke(prompt),
        limiter=limiter,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
    )
    return str(getattr(response, "content", response)).strip()


def parse_query_label(raw: str) -> str:
    """Parse one router label without substring collisions.

    In particular, ``DOCUMENT`` must not match the suffix of
    ``FOLLOW_UP_DOCUMENT``.
    """
    cleaned = str(raw).strip().upper()
    cleaned = re.sub(r"```(?:TEXT|PLAIN|PLAINTEXT)?", "", cleaned)
    cleaned = cleaned.replace("```", "").strip()

    # Handle the expected output and common wrappers such as
    # ``Label: FOLLOW_UP_DOCUMENT``.
    for line in cleaned.splitlines():
        candidate = line.strip().lstrip("-*• ").strip()
        candidate = re.sub(
            r"^(?:LABEL|CLASSIFICATION|CATEGORY)\s*:\s*",
            "",
            candidate,
        )
        candidate = candidate.strip(" `.:;,[](){}")
        if candidate in VALID_LABELS:
            return candidate

    # Tolerate a short explanation only when it contains exactly one distinct
    # complete label. Custom token boundaries prevent DOCUMENT from matching
    # inside FOLLOW_UP_DOCUMENT.
    matches = ROUTER_LABEL_PATTERN.findall(cleaned)
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]

    print(
        f"[router warning] Could not parse one unambiguous label from {raw!r}; "
        "falling back to FOLLOW_UP_DOCUMENT.",
        flush=True,
    )
    # classify_query is called only when history exists. Treating an ambiguous
    # output as a follow-up is safer than silently discarding that context.
    return "FOLLOW_UP_DOCUMENT"


def classify_query(
    llm: Any,
    query: str,
    history: list[dict[str, Any]],
    document_id: str,
    limiter: ApiRateLimiter,
    max_retries: int,
    retry_base_seconds: float,
) -> str:
    if not history:
        return "DOCUMENT"

    recent_history = format_chat_history(
        history,
        document_id,
        max_turns=5,
    )

    prompt = f"""
Classify the CURRENT user question for a document question-answering system.

Return exactly one of these labels:
CONVERSATION_HISTORY
FOLLOW_UP_DOCUMENT
DOCUMENT

Definitions:

CONVERSATION_HISTORY:
The user is asking about previous questions, previous answers, conversation
order, or a summary or comparison of earlier conversation turns.

FOLLOW_UP_DOCUMENT:
The user is asking about information in the uploaded document, but the current
question depends on an entity, subject, event, or detail mentioned in an earlier
conversation turn.

Examples:
- "What time does it depart?"
- "And how much was the tax?"
- "Which terminal does it arrive at?"
- "Does it arrive the same day?"
- "What about the return flight?"

DOCUMENT:
The current question is a complete standalone question that can be understood
without using earlier conversation turns.

Important distinctions:

Previous question:
"Which flight travels from Beijing to Dubai?"
Current question:
"What time does it depart?"
Label:
FOLLOW_UP_DOCUMENT

Current question:
"What time does flight EK307 depart?"
Label:
DOCUMENT

Current question:
"What did I previously ask about flight EK307?"
Label:
CONVERSATION_HISTORY

Recent conversation:
{recent_history}

Current question:
{query}

Return only one label:
"""

    raw = invoke_text(
        llm,
        prompt,
        limiter,
        max_retries,
        retry_base_seconds,
    )

    label = parse_query_label(raw)

    print(
        f"[router] query={query!r} "
        f"raw={raw!r} "
        f"parsed={label}"
    )

    return label


def rewrite_query(
    llm: Any,
    query: str,
    history: list[dict[str, Any]],
    document_id: str,
    limiter: ApiRateLimiter,
    max_retries: int,
    retry_base_seconds: float,
) -> str:
    recent_history = format_chat_history(history, document_id, max_turns=3)
    if not recent_history:
        return query
    prompt = f"""
Rewrite the current follow-up question into a complete standalone question for document retrieval.

Use the recent conversation to recover the subject, entity, requirement,
comparison target, or reference omitted from the current question.

Rules:
- Do not answer the question.
- Do not mention the conversation history.
- Do not add facts unsupported by the conversation.
- Preserve the user's original intent.
- Return only the standalone question.

Recent conversation:
{recent_history}

Current follow-up question:
{query}

Standalone question:
"""
    rewritten = invoke_text(llm, prompt, limiter, max_retries, retry_base_seconds)
    rewritten = rewritten.strip() if rewritten else ""
    if rewritten:
        print(
            f"[rewrite] query={query!r} rewritten={rewritten!r}",
            flush=True,
        )
    return rewritten or query


def answer_history(
    llm: Any,
    query: str,
    history: list[dict[str, Any]],
    document_id: str,
    limiter: ApiRateLimiter,
    max_retries: int,
    retry_base_seconds: float,
) -> str:
    full_history = format_chat_history(history, document_id, max_turns=None)
    if not full_history:
        return "There is no previous conversation history for this document."
    prompt = f"""
You answer questions about a previous conversation.

Use only the conversation history below.
Do not use the uploaded document.
Do not invent missing questions or answers.
Answer in the same language as the user's current question.

Conversation history:
{full_history}

Current question about the conversation:
{query}

Answer:
"""
    answer = invoke_text(llm, prompt, limiter, max_retries, retry_base_seconds)
    return answer or "I could not find that information in the conversation history."


def run_rag(
    chain: Any,
    query: str,
    limiter: ApiRateLimiter,
    max_retries: int,
    retry_base_seconds: float,
) -> str:
    result = call_gemini(
        lambda: chain.invoke({"query": query}),
        limiter=limiter,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
    )
    return str(result.get("result", "")).strip()


def resolve_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(manifest_path)
    documents = payload.get("documents", payload)
    if not isinstance(documents, dict):
        raise ValueError("Manifest must contain an object named 'documents'.")
    return documents


def load_document_runtime(document_id: str, manifest: dict[str, dict[str, Any]]) -> DocumentRuntime:
    try:
        config = manifest[document_id]
    except KeyError as error:
        raise KeyError(f"No manifest entry for document_id={document_id!r}") from error
    vector_store_path = Path(str(config["vector_store_path"]))
    if not vector_store_path.is_absolute():
        vector_store_path = REPO_ROOT / vector_store_path
    vector_store = load_vector_store(str(vector_store_path))
    return DocumentRuntime(vector_store=vector_store, rag_chain=create_rag_chain(vector_store))


def retrieve_chunks(runtime: DocumentRuntime, query: str, top_k: int) -> list[str]:
    docs = runtime.vector_store.similarity_search(query, k=top_k)
    return [str(doc.page_content) for doc in docs]


def run_case(
    case: dict[str, Any],
    mode: str,
    runtime: DocumentRuntime,
    llm: Any,
    top_k: int,
    evidence_threshold: float,
    limiter: ApiRateLimiter,
    max_retries: int,
    retry_base_seconds: float,
) -> dict[str, Any]:
    query = str(case["query"])
    document_id = str(case["document_id"])
    history = list(case.get("history") or [])
    expected_category = str(case["expected_category"]).upper()

    predicted_category = "DOCUMENT"
    standalone_query = query
    retrieval_bypassed = False

    throttle_wait_before = limiter.total_wait_seconds
    started = time.perf_counter()
    if mode == "baseline":
        answer = run_rag(runtime.rag_chain, query, limiter, max_retries, retry_base_seconds)
    else:
        predicted_category = classify_query(
            llm, query, history, document_id, limiter, max_retries, retry_base_seconds
        )
        if predicted_category == "CONVERSATION_HISTORY":
            retrieval_bypassed = True
            answer = answer_history(
                llm, query, history, document_id, limiter, max_retries, retry_base_seconds
            )
        else:
            if mode == "full" and predicted_category == "FOLLOW_UP_DOCUMENT":
                standalone_query = rewrite_query(
                    llm, query, history, document_id, limiter, max_retries, retry_base_seconds
                )
            answer = run_rag(
                runtime.rag_chain, standalone_query, limiter, max_retries, retry_base_seconds
            )
    elapsed_seconds = time.perf_counter() - started
    throttle_wait_seconds = limiter.total_wait_seconds - throttle_wait_before
    latency_ms = max(0.0, elapsed_seconds - throttle_wait_seconds) * 1000.0

    retrieved_chunks: list[str] = []
    if not retrieval_bypassed:
        # Retrieval for scoring is intentionally outside the latency timer so the
        # benchmark does not count the same retrieval twice.
        retrieved_chunks = retrieve_chunks(runtime, standalone_query, top_k)

    retrieval = retrieval_scores(
        retrieved_chunks,
        [str(item) for item in case.get("gold_evidence") or []],
        token_recall_threshold=evidence_threshold,
    )
    facts = fact_accuracy(answer, case.get("required_facts") or [])

    return {
        "case_id": case["case_id"],
        "mode": mode,
        "document_id": document_id,
        "expected_category": expected_category,
        "predicted_category": predicted_category,
        "route_correct": expected_category == predicted_category,
        "query": query,
        "standalone_query": standalone_query,
        "answer": answer,
        "latency_ms": latency_ms,
        "throttle_wait_ms": throttle_wait_seconds * 1000.0,
        "retrieval_bypassed": retrieval_bypassed,
        "retrieved_chunk_count": len(retrieved_chunks),
        "context_recall": retrieval["context_recall"],
        "context_precision": retrieval["context_precision"],
        "gold_evidence_hits": retrieval["gold_evidence_hits"],
        "relevant_chunks": retrieval["relevant_chunks"],
        "fact_accuracy": facts["fact_accuracy"],
        "facts_matched": facts["facts_matched"],
        "facts_total": facts["facts_total"],
        "fact_details": facts["fact_details"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, Any] = {}
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        route_report = classification_report(
            [row["expected_category"] for row in subset],
            [row["predicted_category"] for row in subset],
            LABELS,
        )
        latencies = [float(row["latency_ms"]) for row in subset]
        by_mode[mode] = {
            "cases": len(subset),
            "routing": route_report,
            "mean_context_recall": mean(
                row["context_recall"] for row in subset if row["context_recall"] is not None
            ),
            "mean_context_precision": mean(
                row["context_precision"]
                for row in subset
                if row["context_precision"] is not None
            ),
            "mean_fact_accuracy": mean(
                row["fact_accuracy"] for row in subset if row["fact_accuracy"] is not None
            ),
            "retrieval_bypass_rate": mean(
                1.0 if row["retrieval_bypassed"] else 0.0 for row in subset
            ),
            "latency_ms": {
                "mean": mean(latencies),
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
            },
            "fact_accuracy_by_category": grouped_mean(
                subset, "expected_category", "fact_accuracy"
            ),
            "context_recall_by_category": grouped_mean(
                subset, "expected_category", "context_recall"
            ),
            "latency_by_category_ms": grouped_mean(
                subset, "expected_category", "latency_ms"
            ),
        }

    pairwise: dict[str, Any] = {}
    index = {(row["case_id"], row["mode"]): row for row in rows}
    if all(mode in by_mode for mode in ("baseline", "full")):
        wins = ties = losses = 0
        for case_id in sorted({row["case_id"] for row in rows}):
            baseline = index.get((case_id, "baseline"))
            full = index.get((case_id, "full"))
            if not baseline or not full:
                continue
            baseline_score = baseline.get("fact_accuracy")
            full_score = full.get("fact_accuracy")
            if baseline_score is None or full_score is None:
                continue
            if full_score > baseline_score:
                wins += 1
            elif full_score < baseline_score:
                losses += 1
            else:
                ties += 1
        denominator = wins + losses + ties
        pairwise["full_vs_baseline_fact_accuracy"] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate_excluding_ties": wins / (wins + losses) if wins + losses else None,
            "win_rate_including_ties_as_half": (
                (wins + 0.5 * ties) / denominator if denominator else None
            ),
        }

    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
        "environment": {
            "llm_model": os.getenv("LLM_MODEL"),
            "chunk_size": os.getenv("CHUNK_SIZE"),
            "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
            "embedding_model": os.getenv("HUGGINGFACE_EMBEDDING_MODEL"),
        },
        "by_mode": by_mode,
        "pairwise": pairwise,
    }


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        item = dict(row)
        item["fact_details"] = json.dumps(item["fact_details"], ensure_ascii=False)
        serializable.append(item)
    fieldnames = list(serializable[0].keys()) if serializable else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable)


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid checkpoint JSONL at {path}:{line_number}: {error}"
                ) from error
    return rows


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation benchmark for the current conversational Document Q&A repository."
    )
    parser.add_argument("--cases", type=Path, required=True, help="Frozen JSONL cases.")
    parser.add_argument("--manifest", type=Path, required=True, help="Document to FAISS path mapping.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
        help="Systems to evaluate.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--evidence-token-recall", type=float, default=0.70)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=13.0,
        help="Minimum seconds between Gemini API calls. Use at least 12.5 for a 5 RPM limit.",
    )
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=15.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed case/mode/repeat rows from progress.jsonl.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/results/conversation")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.cases)
    if not cases:
        raise SystemExit("No evaluation cases were loaded.")
    manifest = resolve_manifest(args.manifest)
    llm = initialize_llm()
    runtimes: dict[str, DocumentRuntime] = {}
    limiter = ApiRateLimiter(min_interval_seconds=max(0.0, args.request_interval))
    checkpoint_path = args.output_dir / "progress.jsonl"
    if args.resume:
        rows = load_checkpoint(checkpoint_path)
        print(f"Loaded {len(rows)} completed rows from {checkpoint_path}", flush=True)
    else:
        rows = []
        checkpoint_path.unlink(missing_ok=True)
    completed = {
        (str(row["case_id"]), str(row["mode"]), int(row.get("repeat", 1)))
        for row in rows
    }

    for case in cases:
        document_id = str(case["document_id"])
        if document_id not in runtimes:
            runtimes[document_id] = load_document_runtime(document_id, manifest)
        for repeat in range(1, args.repeats + 1):
            for mode in args.modes:
                key = (str(case["case_id"]), mode, repeat)
                if key in completed:
                    print(f"[resume] skipping {key}", flush=True)
                    continue
                row = run_case(
                    case=case,
                    mode=mode,
                    runtime=runtimes[document_id],
                    llm=llm,
                    top_k=args.top_k,
                    evidence_threshold=args.evidence_token_recall,
                    limiter=limiter,
                    max_retries=args.max_retries,
                    retry_base_seconds=args.retry_base_seconds,
                )
                row["repeat"] = repeat
                rows.append(row)
                append_checkpoint(checkpoint_path, row)
                completed.add(key)
                print(
                    f"[{mode}] {case['case_id']} "
                    f"expected={row['expected_category']} "
                    f"route={row['predicted_category']} "
                    f"fact={row['fact_accuracy']} recall={row['context_recall']} "
                    f"latency_ms={row['latency_ms']:.1f}",
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "per_case_results.csv")
    summary = summarize(rows)
    summary["configuration"] = {
        "cases_path": str(args.cases),
        "manifest_path": str(args.manifest),
        "modes": args.modes,
        "top_k": args.top_k,
        "repeats": args.repeats,
        "evidence_token_recall": args.evidence_token_recall,
        "request_interval": args.request_interval,
        "max_retries": args.max_retries,
        "retry_base_seconds": args.retry_base_seconds,
        "checkpoint_path": str(checkpoint_path),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()