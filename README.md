Document Q&A System

The Document Q&A System is a document-centered conversational assistant for users who need to upload a PDF, ask grounded questions about its contents, continue with context-dependent follow-up questions, review earlier conversation history, and optionally supplement document analysis with current web information.

The project extends a basic single-turn PDF RAG backend into a multi-source assistant. It can operate as a strict document question-answering system or as a broader assistant that chooses among the uploaded document, conversation history, general model knowledge, and Google Search grounding.

Original project:

https://github.com/FaisalAhmedBijoy/Document-QA-RAG-System-FastAPI

What This Project Is For

The system supports workflows that require more than one-off PDF retrieval:

Ask factual questions whose answers must come from an uploaded PDF.

Ask follow-up questions that depend on earlier conversation context.

Review, summarize, or compare previous questions and answers.

Ask general questions without forcing every request through document retrieval.

Combine document evidence with general explanation or advice.

Verify or update document content with current web information.

Resume an unfinished anonymous conversation after restarting the application.

The system keeps document facts, conversation history, general knowledge, and web sources separate so users can understand where an answer came from.

Key Features

Document-grounded Q&A

Users can upload one PDF per active conversation. The backend extracts text, creates a session-scoped FAISS vector store, retrieves relevant chunks, and generates an answer grounded in the uploaded document.

Follow-up question rewriting

Recent conversation turns are used to rewrite incomplete follow-up questions into standalone document-retrieval queries.

User: What insurance coverage does the lease require?
User: What about the second option?

The second question is rewritten into a complete document query before retrieval.

Conversation-history Q&A

The system can answer questions about earlier turns without searching the PDF.

What was my first question?
Which previous answer mentioned the insurance amount?
Summarize what we discussed.

General assistant responses

In Assistant mode, users can ask general questions even when no PDF is uploaded. These questions bypass FAISS retrieval and are answered using the language model and relevant conversation context.

Document and general-knowledge synthesis

For recommendations, evaluations, comparisons, or implications, the system can combine document evidence with general explanation or advice while keeping the two source types separate.

Google Search grounding

When web search is enabled, current or explicitly online questions can use Gemini Google Search grounding. The frontend displays:

Whether web search was used.

Source titles.

Grounding redirect URLs.

The answer text supported by each source.

The system also supports document-plus-web questions, such as checking whether information in an uploaded PDF is still current.

Resumable anonymous conversations

Each browser receives a persistent client_id, and each active conversation receives a UUID-based session_id. PostgreSQL persistence restores:

Active session ID.

Current document ID.

Uploaded filename.

Chat history.

Query categories and source types.

Web search status and citations.

Session isolation

Each session stores its resources under a dedicated path:

app/data/sessions/{session_id}/pdfs/
app/data/sessions/{session_id}/texts/
app/data/sessions/{session_id}/vectorstores/

This prevents PDFs, extracted text, FAISS indexes, chat histories, and web citations from being mixed across sessions.

Response Modes

Assistant

Assistant mode can use:

Document retrieval
Conversation history
General model knowledge
Document + general knowledge
Google Search
Document + Google Search

A PDF is optional in this mode.

Strict Document

Strict Document mode is limited to:

Uploaded document
Conversation history

A PDF is required. General knowledge and web search are not used to fill gaps in the document.

Query Routing

Route

Behavior

DOCUMENT

Retrieves from the uploaded PDF and answers from document evidence.

FOLLOW_UP_DOCUMENT

Rewrites the follow-up using recent history, then retrieves from the PDF.

CONVERSATION_HISTORY

Answers from prior turns and bypasses document retrieval.

GENERAL

Uses general model knowledge and relevant conversation context.

HYBRID

Combines uploaded-document evidence with general explanation or advice.

WEB

Uses Gemini Google Search grounding and returns web citations.

DOCUMENT_AND_WEB

Combines retrieved document evidence with current grounded web information.

If a user asks about an uploaded document when no PDF is available, the system returns a document-unavailable response instead of inventing content.

End-to-End Workflow

Open Streamlit frontend
→ Start or resume an anonymous active session
→ Ask a general question or upload a PDF
→ Classify the request by required source
→ Retrieve from the PDF, history, model knowledge, web, or a combination
→ Generate a source-aware answer
→ Save the answer and metadata to database
→ Restart the application and resume the active conversation
→ End the conversation explicitly when finished

Architecture

Streamlit frontend
        |
        v
FastAPI routes
        |
        v
Source-aware query router
        |
        +--> Conversation history
        +--> General Gemini response
        +--> FAISS document retrieval
        +--> Gemini Google Search grounding
        +--> Document + general or document + web synthesis
        |
        v
PostgreSQL session and message persistence

Main technologies:

Python
FastAPI
Streamlit
Gemini API
Gemini Google Search grounding
FAISS
all-MiniLM-L6-v2 embeddings
PyMuPDF
PostgreSQL
LangChain

Main Files

frontend.py
app/main.py
app/routes/rag_route.py
app/schemas/rag_schema.py
app/services/rag_service.py
app/services/session_service.py
app/processing/generate_rag_chain.py
app/processing/generate_vector_db.py
app/processing/generate_text_chunks.py
app/processing/single_query_inference.py

Runtime data:

app/data/app.db
app/data/sessions/

API Endpoints

POST /rag/sessions/start-or-resume
POST /rag/sessions/{session_id}/end
POST /rag/upload-document-pdf
POST /rag/assistant/query
POST /rag/query-by-document

/rag/query-by-document remains available for backward compatibility.

Local Setup

1. Create and activate the environment

conda create -n rag_llm python=3.11
conda activate rag_llm

2. Install dependencies

pip install -r requirements.txt

The current Google Search grounding implementation is compatible with:

langchain-google-genai==2.1.8
google-ai-generativelanguage==0.6.18

3. Configure environment variables

Create a .env file using the variable names expected by app/config/configuration.py.

GOOGLE_API_KEY=<your_google_api_key>
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=app/data/vectorstores/faiss_index

Do not commit real API keys.

4. Start the FastAPI backend

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

5. Start the Streamlit frontend

In a second terminal:

streamlit run frontend.py

The frontend normally opens at:

http://localhost:8501

Google Search Grounding Test

The standalone script can test search grounding without starting FastAPI or Streamlit:

python test_google_search.py

A successful response should include:

WEB SEARCH USED: True

and at least one entry under SOURCES.

Grounding may return redirect URLs such as:

https://vertexaisearch.cloud.google.com/grounding-api-redirect/...

These links redirect to the underlying source pages.

Evaluation

Conversational routing ablation

A 33-run ablation across 11 development cases compared the original direct-RAG behavior with the full conversational pipeline.

End-to-end fact accuracy: 54.5% → 100%
Correctly routed development cases: 11/11
Conversation-history requests that bypassed document retrieval: 3/3

End-to-end functional testing

The current system also completed 30 end-to-end functional tests covering:

General conversation without a PDF.

Standalone document questions.

Context-dependent document follow-ups.

Conversation-history questions.

Hybrid document and general answers.

Strict Document mode.

General-query retrieval bypass.

Google Search routing and citation extraction.

Document-plus-web answers.

Session restart and conversation recovery.

PDF replacement behavior.

Persistence and restoration of web citations.

Explicit conversation termination.

Result:

30/30 tests completed successfully

This result describes tested local functionality. It does not imply production-scale load, security, or reliability validation.

Session and Storage Behavior

The application supports one active PDF per conversation.

When a new PDF is uploaded:

The session's active document is replaced.

Messages tied to the previous PDF are removed.

Document-independent general and web conversations are preserved.

New document questions use only the new session-scoped FAISS index.

PostgreSQL stores session and message metadata, including:

session_id
document_id
query
answer
query_category
source_type
web_search_used
web_sources
created_at

Security and Isolation Improvements

Public utility endpoints that could expose shared document resources were removed or disabled, including:

GET /rag/list-vector-stores
GET /rag/pdf/{document_id}

The main workflow now uses session-scoped upload and query endpoints.

The system provides application-level anonymous session isolation, but it is not a replacement for authentication, authorization, encryption, or production security controls.

Current Limitations

Only one active PDF is supported per conversation.

There is no formal user login system.

Anonymous identity depends on the browser client_id.

Removing the client_id from the URL creates a new anonymous client.

Ended sessions are not automatically deleted.

There is no multi-document library interface.

PDF processing is synchronous and has no background job queue.

Search quality depends on Gemini Google Search grounding.

Free Gemini API quotas can limit testing volume.

The project is designed mainly for local or self-hosted use.

The 30 successful tests are functional tests, not production load tests.

Summary

The project has been extended from a basic single-turn PDF RAG demo into a Document Q&A System with:

General conversation without a PDF
Strict document-only question answering
Document-scoped FAISS retrieval
Follow-up query rewriting
Conversation-history answering
Document and general-knowledge synthesis
Gemini Google Search grounding
Document and web synthesis
Citation extraction and persistence
Anonymous multi-user session isolation
Resumable conversations
Session-scoped PDF, text, and vector-store storage
30/30 successful end-to-end functional tests