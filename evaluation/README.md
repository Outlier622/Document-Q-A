# Evaluation package for `Outlier622/Document-Q-A`

This package is tailored to the repository's current implementation:

- three query labels: `DOCUMENT`, `FOLLOW_UP_DOCUMENT`, and `CONVERSATION_HISTORY`
- recent five turns for routing
- recent three turns for query rewriting
- FAISS similarity retrieval with `k=5`
- SQLite sessions with `ACTIVE` and `ENDED` states
- session-scoped paths under `app/data/sessions/{session_id}`

Copy the `evaluation/` folder into the repository root. Run every command from the repository root.

## 1. Conversational RAG ablation benchmark

This compares three systems on the same frozen cases:

- `baseline`: every query is sent directly to document RAG
- `router_only`: history queries are routed, but follow-ups are not rewritten
- `full`: current routing plus recent-turn rewriting

It produces:

- route accuracy and Macro F1
- Context Recall@5
- Context Precision@5
- deterministic fact accuracy
- retrieval bypass rate
- p50 and p95 end-to-end latency
- full-system versus baseline pairwise win rate

### Prepare the data

1. Upload PDFs through the existing application so each PDF has a FAISS index.
2. Copy `documents_manifest.example.json` to `documents_manifest.json` and map each logical document ID to its real FAISS path.
3. Copy `conversation_cases.example.jsonl` to `conversation_cases.jsonl`.
4. Use exact evidence sentences from the PDF for `gold_evidence`.
5. For every required fact, list acceptable literal patterns. This makes scoring deterministic and auditable.
6. Freeze the final test file before tuning prompts.

### Run

```bash
python evaluation/run_conversational_eval.py \
  --cases evaluation/datasets/conversation_cases.jsonl \
  --manifest evaluation/datasets/documents_manifest.json \
  --modes baseline router_only full \
  --top-k 5 \
  --repeats 1
```

Results:

```text
evaluation/results/conversation/per_case_results.csv
evaluation/results/conversation/summary.json
```

For a defensible resume result, use at least 90 frozen test queries. A practical split is 30 development cases and 90 frozen test cases. Do not tune prompts after examining frozen-test failures.

## 2. SQLite session reliability benchmark

This uses a separate benchmark database by default, so it does not alter `app/data/app.db`. It creates sessions concurrently, writes messages, launches fresh Python processes to simulate application restarts, validates order and content, tests document replacement, ends sessions, and verifies that ended sessions are not resumed.

```bash
python evaluation/run_session_reliability.py \
  --sessions 100 \
  --messages-per-session 5 \
  --workers 10 \
  --restart-cycles 10 \
  --reset-sessions 10
```

It produces measurable values such as:

- `N/N` sessions recovered
- `M/M` messages recovered
- lost, duplicated, or reordered messages
- p50, p95, and p99 SQLite operation latency
- ended-session resurrection count
- document-reset success rate

Result:

```text
evaluation/results/session/summary.json
```

## 3. API session isolation benchmark

Start the existing FastAPI backend first:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then run:

```bash
python evaluation/run_session_isolation.py \
  --base-url http://localhost:8000 \
  --sessions 20 \
  --workers 2
```

Twenty sessions create `20 × 19 = 380` cross-session document access attempts. Those requests should fail before Gemini inference because the victim document ID does not exist inside the attacker's session directory.

Add `--run-own-query` only when you also want to pay the Gemini cost and verify that each session can retrieve its own marker:

```bash
python evaluation/run_session_isolation.py \
  --sessions 20 \
  --workers 2 \
  --run-own-query
```

Results:

```text
evaluation/results/isolation/summary.json
evaluation/results/isolation/attempts.csv
```

### Security wording limitation

The current repository checks whether a `session_id` is active and scopes storage paths by that ID. Upload and query requests do not bind the session to a server-issued ownership token. Therefore, this benchmark supports the claim **session-scoped document isolation**, not **authenticated multi-user authorization**.

Do not write `token-bound ownership`, `authenticated tenant isolation`, or equivalent wording until the application adds a server-issued secret or authenticated user identity and verifies it on every upload, query, history, and end operation.

## Data that can become resume bullets

Only use values actually present in the generated JSON files. Examples of valid structures are:

```text
Evaluated query routing and recent-turn rewriting across N frozen queries from M PDFs, improving follow-up Recall@5 from A% to B% and fact accuracy from C% to D%, while bypassing document retrieval for E% of conversation-history requests.
```

```text
Validated SQLite-backed recovery across N sessions, M persisted messages, and R fresh-process restart cycles, restoring X/M messages with zero lost, duplicated, or reordered records and p95 session-operation latency of Y ms.
```

```text
Validated session-scoped storage across N anonymous sessions, rejecting A/A cross-session document queries and all ended-session operations with zero unique-marker leakage.
```

Do not fill these templates until the scripts have generated the actual values.
