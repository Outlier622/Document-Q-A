# Changes

This file records the main changes made while setting up and improving:

```text
https://github.com/FaisalAhmedBijoy/Document-QA-RAG-System-FastAPI
```

## Modified Files

```text
requirements.txt
streamlit.py
app/schemas/rag_schema.py
app/routes/rag_route.py
app/services/rag_service.py
```

## Added Files

```text
app/services/session_service.py
```

The project also creates or uses the following local/runtime files and folders:

```text
.env
app/data/app.db
app/data/sessions/
```

`app/data/app.db` is generated at runtime by the SQL-based session service.
`app/data/sessions/` stores per-session PDFs, extracted text files, and FAISS vector stores.

---

## 1. Environment and Dependency Fixes

Added missing environment configuration and runtime dependencies required for local setup.

Configured required variables such as:

```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=app/data/vectorstores/faiss_index_<document_id>
GROQ_API_KEY=<your_groq_api_key>
```

Updated `requirements.txt` with missing packages, including:

```txt
PyMuPDF
pdfplumber==0.11.4
Pillow<12
bangla_pdf_ocr
python-dotenv
```

These changes make the project easier to run locally and reduce setup friction.

---

## 2. Streamlit PDF Upload Flow

Updated `streamlit.py` to provide a complete frontend document QA workflow.

The frontend now supports:

```text
Upload PDF
→ Send PDF to FastAPI backend
→ Receive document_id
→ Ask questions about that document
→ Receive document-grounded answers
```

The Streamlit app now calls:

```text
POST /rag/upload-document-pdf
POST /rag/query-by-document
```

and warns users if they try to ask questions before uploading a PDF.

---

## 3. LLM Backend Update

Switched the LLM backend from Groq to Gemini and increased the maximum output token limit.

This was done to reduce answer truncation and improve response completeness during document QA.

---

## 4. Document-Specific Querying

Updated:

```text
app/routes/rag_route.py
app/services/rag_service.py
app/schemas/rag_schema.py
```

to support querying a specific uploaded document by `document_id`.

The `/rag/query-by-document` endpoint now loads the FAISS vector store associated with the selected document instead of relying only on a default vector store.

---

## 5. Follow-Up Question Support

Added chat history support for follow-up questions.

The frontend now sends previous query-answer turns to the backend. The backend uses recent conversation history to rewrite follow-up questions into standalone document queries before retrieval.

Example:

```text
User: What vaccines are required?
User: What about students over 22?
```

The second question can be rewritten into a standalone query about age-related vaccine exemptions.

---

## 6. Conversation History Question Answering

Added query classification in `app/services/rag_service.py`.

Queries are now classified as:

```text
DOCUMENT
FOLLOW_UP_DOCUMENT
CONVERSATION_HISTORY
```

Routing behavior:

```text
DOCUMENT
→ Direct RAG retrieval

FOLLOW_UP_DOCUMENT
→ Rewrite using recent history
→ RAG retrieval

CONVERSATION_HISTORY
→ Answer from chat history
→ Do not search the PDF
```

This allows the system to answer questions such as:

```text
What was my first question?
What did I ask before the TB screening question?
Which previous answer mentioned age 22?
Summarize what we discussed.
```

---

## 7. Full Chat History Sent to Backend

Updated `streamlit.py` so the frontend sends the full chat history for the current document.

The backend then uses:

```text
Recent history
→ Follow-up query rewriting

Full history
→ Conversation-history question answering
```

This allows the system to answer history-related questions even after longer conversations.

---

## 8. Anonymous Multi-User Session Isolation

Added anonymous session isolation using UUID-based `session_id`.

The frontend sends `session_id` with upload and query requests.

Backend storage was changed from shared paths:

```text
app/data/pdfs/
app/data/texts/
app/data/vectorstores/
```

to session-scoped paths:

```text
app/data/sessions/{session_id}/pdfs/
app/data/sessions/{session_id}/texts/
app/data/sessions/{session_id}/vectorstores/
```

Each anonymous session now has its own PDF, extracted text, and FAISS vector store.

---

## 9. Removed Unsafe Public Endpoints

Removed or disabled public utility endpoints that could break session isolation:

```text
GET /rag/list-vector-stores
GET /rag/pdf/{document_id}
```

The main user workflow now depends only on upload and document-query endpoints.

This avoids exposing global document IDs, vector stores, or PDFs across sessions.

---

## 10. Resumable Anonymous Conversations

Added persistent anonymous conversation recovery.

New file:

```text
app/services/session_service.py
```

The SQL-backed session service stores:

```text
sessions
messages
```

Each anonymous browser receives a persistent `client_id`.
Each conversation receives a `session_id` with one of two states:

```text
ACTIVE
ENDED
```

The frontend starts by calling:

```text
POST /rag/sessions/start-or-resume
```

If the `client_id` has an `ACTIVE` session, the backend restores:

```text
session_id
document_id
uploaded_filename
chat_history
```

If no active session exists, the backend creates a new one.

The frontend also adds an explicit:

```text
End Conversation
```

button.

Clicking it calls:

```text
POST /rag/sessions/{session_id}/end
```

and changes the session status from `ACTIVE` to `ENDED`.

---

## 11. Persistent Message Storage

The backend now stores each query-answer pair in a SQL local database after successful responses.

Stored message fields include:

```text
session_id
document_id
query
answer
created_at
```

When an active session is resumed, saved chat history is restored to the frontend.

---

## 12. One Active Document Per Conversation

The current application supports one active document per conversation.

When a new PDF is uploaded, the session’s active document is updated and the previous chat history for that session is cleared.

This keeps the app simple and prevents cross-document contamination.

---

## Final Workflow

The current application supports:

```text
Open Streamlit app
→ Start or resume anonymous active session
→ Upload one PDF
→ Ask document-grounded questions
→ Ask follow-up questions
→ Ask questions about conversation history
→ Close or restart the app
→ Resume unfinished conversation
→ End Conversation when finished
```

After clicking `End Conversation`, the old conversation will not be restored on the next run.

---

## Current Limitations

```text
Only one active document is supported per conversation.
There is no formal user login system.
Anonymous identity depends on the browser client_id.
Removing client_id from the URL creates a new anonymous client.
There is no automatic cleanup for old ended sessions yet.
There is no multi-document library UI yet.
There is no background job queue for large PDF processing yet.
The app is still designed mainly for local or self-hosted use.
```

---

## Summary

The project was improved from a basic backend RAG demo into a usable single-document QA application with:

```text
Reproducible local setup
Frontend PDF upload flow
Gemini-based document QA
Document-specific querying
Follow-up question rewriting
Conversation-history question answering
Anonymous multi-user session isolation
Session-scoped PDF/text/FAISS storage
Removed unsafe public document endpoints
SQL-backed resumable conversations
Explicit End Conversation control
```
