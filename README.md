Document Q&A System

The Document Q&A System is a document-centered conversational assistant for users who need to upload a PDF, ask grounded questions about its contents, continue with context-dependent follow-up questions, review earlier conversation history, and optionally supplement document analysis with current web information.

The project extends a basic single-turn PDF RAG backend into a multi-source conversational system with persistent sessions, asynchronous document processing, cloud-backed storage, and an ECS-deployed backend. It can operate as a strict document question-answering system or as a broader assistant that chooses among the uploaded document, conversation history, general model knowledge, and Google Search grounding.

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

Process uploaded documents asynchronously instead of blocking the API request.

The system keeps document facts, conversation history, general knowledge, and web sources separate so users can understand where an answer came from.

Key Features

Document-grounded Q&A

Users can upload one PDF per active conversation. The backend extracts text, creates a session-scoped FAISS vector store, retrieves relevant chunks, and generates an answer grounded in the uploaded document.

Follow-up question rewriting

Recent conversation turns are used to rewrite incomplete follow-up questions into standalone document-retrieval queries.

User: What insurance coverage does the lease require?User: What about the second option?

The second question is rewritten into a complete document query before retrieval.

Conversation-history Q&A

The system can answer questions about earlier turns without searching the PDF.

What was my first question?Which previous answer mentioned the insurance amount?Summarize what we discussed.

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

Each conversation keeps its document, extracted text, vector store, and message history isolated from other sessions.

For local storage, session resources use:

app/data/sessions/{session_id}/

For S3-backed storage, document artifacts are stored under:

document-qa/sessions/{session_id}/documents/{document_id}/original.pdfextracted.txtfaiss/index.faissfaiss/index.pkl

This prevents PDFs, extracted text, FAISS indexes, chat histories, and web citations from being mixed across sessions.

Asynchronous document processing

PDF upload no longer requires the API request to perform the complete parsing, chunking, embedding, and indexing workflow synchronously.

FastAPI upload request→ Upload original PDF to Amazon S3→ Create processing-job metadata in PostgreSQL→ Send a job to Amazon SQS→ ECS Fargate worker downloads the PDF→ Extract text and split document into chunks→ Generate embeddings and build the FAISS index→ Upload extracted text and FAISS artifacts to S3→ Update processing-job status in PostgreSQL

This separates API responsiveness from document-processing work and allows the worker to run independently from the web service.

Response Modes

Assistant

Assistant mode can use:

Document retrievalConversation historyGeneral model knowledgeDocument + general knowledgeGoogle SearchDocument + Google Search

A PDF is optional in this mode.

Strict Document

Strict Document mode is limited to:

Uploaded documentConversation history

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

Open Streamlit frontend→ Start or resume an anonymous active session→ Ask a general question or upload a PDF→ FastAPI stores the original PDF in S3→ PostgreSQL records a processing job→ SQS dispatches the job to the ECS worker→ Worker parses, chunks, embeds, builds FAISS, and stores artifacts in S3→ Classify later questions by required source→ Retrieve from the PDF, history, model knowledge, web, or a combination→ Generate a source-aware answer→ Save the answer and metadata to PostgreSQL→ Restart the application and resume the active conversation→ End the conversation explicitly when finished

Architecture

Streamlit frontend|vECS Fargate FastAPI service|+--> Amazon RDS PostgreSQL|       - sessions|       - messages|       - processing jobs|+--> Amazon S3|       - original PDFs|       - extracted text|       - FAISS artifacts|+--> Amazon SQS|vECS Fargate document worker|+--> PDF parsing+--> chunking+--> embeddings+--> FAISS indexing+--> S3 artifact upload

FastAPI query pipeline|+--> Conversation history+--> General Gemini response+--> FAISS document retrieval+--> Gemini Google Search grounding+--> Document + general synthesis+--> Document + web synthesis

AWS deployment components:

Amazon S3Amazon SQSAmazon ECS FargateAmazon RDS for PostgreSQLAmazon ECRAWS Secrets ManagerIAM task rolesCloudWatch Logs

Main technologies:

PythonFastAPIStreamlitGemini APIGemini Google Search groundingFAISSall-MiniLM-L6-v2 embeddingsPyMuPDFPostgreSQLLangChainDockerAmazon S3Amazon SQSAmazon ECS FargateAmazon RDSAmazon ECRAWS Secrets Managerboto3

Main Files

frontend.pyapp/main.pyapp/routes/rag_route.pyapp/schemas/rag_schema.pyapp/services/rag_service.pyapp/services/session_service.pyapp/services/storage_service.pyapp/services/queue_service.pyapp/services/document_processing_service.pyapp/workers/document_worker.pyapp/database/app/processing/generate_rag_chain.pyapp/processing/generate_vector_db.pyapp/processing/generate_text_chunks.pyapp/processing/single_query_inference.pyDockerfile.ecsrequirements.txtrequirements.ecs.txt.dockerignore

Local runtime data may still exist under:

app/data/

Runtime data, local databases, session artifacts, and .env files should not be committed to Git.

API Endpoints

POST /rag/sessions/start-or-resumePOST /rag/sessions/{session_id}/endPOST /rag/upload-document-pdfPOST /rag/assistant/queryPOST /rag/query-by-document

/rag/query-by-document remains available for backward compatibility.

Local Setup

Create and activate the environment

conda create -n rag_llm python=3.11conda activate rag_llm

Install dependencies

pip install -r requirements.txt

The current Google Search grounding implementation is compatible with:

langchain-google-genai==2.1.8google-ai-generativelanguage==0.6.18

Configure environment variables

Create a .env file using the variable names expected by app/config/configuration.py.

GOOGLE_API_KEY=<your_google_api_key>CHUNK_SIZE=1000CHUNK_OVERLAP=200HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2VECTOR_STORE_PATH=app/data/vectorstores/faiss_index

DATABASE_URL=<your_database_url>STORAGE_BACKEND=localS3_BUCKET_NAME=<your_bucket_name>S3_PREFIX=document-qaSQS_QUEUE_URL=<your_queue_url>SQS_VISIBILITY_TIMEOUT=120

Do not commit real API keys, passwords, database URLs, AWS credentials, or .env files.

Start the FastAPI backend

uvicorn app.main --reload --host 0.0.0.0 --port 8000

Start the Streamlit frontend

In a second terminal:

streamlit run frontend.py

The frontend normally opens at:

http://localhost:8501

When the frontend connects to an ECS-hosted API instead of a local FastAPI process, set:

$env="http://<ecs-api-address>:8000"

Then start Streamlit normally.

ECS Container Setup

The ECS image uses:

Dockerfile.ecsrequirements.ecs.txt

Build locally:

docker build --platform linux/amd64 -f Dockerfile.ecs -t document-qa .

Before publishing an image, verify that .env is not included:

docker run --rm document-qa sh -c "if [ -f /app/.env ]; then echo ENV_PRESENT; else echo ENV_NOT_PRESENT; fi"

Expected result:

ENV_NOT_PRESENT

The .dockerignore file excludes local secrets and runtime artifacts from the Docker build context.

The API container starts FastAPI with Uvicorn. The worker service uses the same image but overrides the container command to:

python -m app.workers.document_worker

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

End-to-end fact accuracy: 54.5% → 100%Correctly routed development cases: 11/11Conversation-history requests that bypassed document retrieval: 3/3

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

This result describes tested local functionality. It does not imply production-scale load, security, reliability, or cloud-performance validation.

Session and Storage Behavior

The application supports one active PDF per conversation.

When a new PDF is uploaded:

The session's active document is replaced.

Messages tied to the previous PDF are removed.

Document-independent general and web conversations are preserved.

New document questions use only the new session-scoped FAISS index.

PostgreSQL stores session and message metadata, including:

session_iddocument_idqueryanswerquery_categorysource_typeweb_search_usedweb_sourcescreated_at

PostgreSQL also stores document-processing job state so the FastAPI service and ECS worker can coordinate asynchronous processing.

Security and Isolation Improvements

Public utility endpoints that could expose shared document resources were removed or disabled, including:

GET /rag/list-vector-storesGET /rag/pdf/{document_id}

The main workflow now uses session-scoped upload and query endpoints.

For ECS deployment, AWS access is provided through IAM task roles rather than local AWS profiles. Sensitive runtime configuration is injected through AWS Secrets Manager, and .dockerignore prevents .env files from being copied into container images.

The system provides application-level anonymous session isolation, but it is not a replacement for authentication, authorization, encryption, or production security controls.

Current Limitations

Only one active PDF is supported per conversation.

There is no formal user login system.

Anonymous identity depends on the browser client_id.

Removing the client_id from the URL creates a new anonymous client.

Ended sessions are not automatically deleted.

There is no multi-document library interface.

Search quality depends on Gemini Google Search grounding.

Gemini API quotas can limit testing volume.

The current ECS deployment does not include a load balancer or stable public domain.

A Fargate task public IP can change after redeployment, so a locally hosted frontend must update API_BASE_URL when using the task's public IP directly.

ECS document-processing latency is currently higher than local processing for some PDFs and can exceed the frontend polling timeout.

The AWS deployment has not been validated under production-scale concurrency or load.

The 30 successful tests are functional tests, not production load tests.

Summary

The project has been extended from a basic single-turn PDF RAG demo into a Document Q&A System with:

General conversation without a PDFStrict document-only question answeringDocument-scoped FAISS retrievalFollow-up query rewritingConversation-history answeringDocument and general-knowledge synthesisGemini Google Search groundingDocument and web synthesisCitation extraction and persistenceAnonymous multi-user session isolationResumable conversationsPostgreSQL persistenceS3-backed PDF, text, and FAISS artifact storageSQS-based asynchronous document processingECS Fargate deployment for the FastAPI service and document workerECR-hosted container imagesIAM task-role access to AWS resourcesSecrets Manager-based runtime secret injection30/30 successful local end-to-end functional tests

The current cloud deployment demonstrates the transition from a local conversational RAG application to a cloud-backed asynchronous document-processing architecture. ECS services are deployed successfully, while document-processing latency on ECS remains a known optimization issue for some uploads.