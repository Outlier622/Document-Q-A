"""Streamlit frontend for the document-aware conversational assistant.

This version supports:
1. General conversation without an uploaded document.
2. Optional PDF upload for document-grounded questions.
3. Assistant and Strict Document modes.
4. Automatic routing across history, documents, general knowledge, web search,
   and mixed document/web answers.
5. Resumable anonymous conversations with persisted web citations.
"""

import uuid

import requests
import streamlit as st

from app.core.logger import configure_logging


logger = configure_logging("STREAMLIT_APP")
API_BASE_URL = "http://localhost:8000"

SOURCE_LABELS = {
    "conversation_history": "Conversation history",
    "general_knowledge": "General model knowledge",
    "uploaded_document": "Uploaded document",
    "document_and_general_knowledge": "Uploaded document + general knowledge",
    "web_search": "Google Search",
    "web_search_unavailable": "Web search unavailable",
    "document_and_web": "Uploaded document + Google Search",
    "document_and_web_search_unavailable": (
        "Uploaded document + unavailable web search"
    ),
    "no_document": "No document available",
}

MODE_LABELS = {
    "assistant": "Assistant",
    "strict": "Strict Document",
}


def _get_query_parameter(name: str):
    value = st.query_params.get(name)

    if isinstance(value, list):
        return value[0] if value else None

    return value


def get_or_create_client_id() -> str:
    client_id = _get_query_parameter("client_id")

    if not client_id:
        client_id = str(uuid.uuid4())
        st.query_params["client_id"] = client_id

    return client_id


def start_or_resume_backend_session(client_id: str):
    response = requests.post(
        f"{API_BASE_URL}/rag/sessions/start-or-resume",
        json={"client_id": client_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def end_backend_session(client_id: str, session_id: str):
    response = requests.post(
        f"{API_BASE_URL}/rag/sessions/{session_id}/end",
        json={"client_id": client_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def upload_pdf_to_backend(uploaded_file, session_id: str):
    files = {
        "pdf_file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/rag/upload-document-pdf",
        files=files,
        data={"session_id": session_id},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def query_assistant(
    session_id: str,
    query: str,
    document_id: str | None,
    assistant_mode: str,
    web_search_enabled: bool,
    chat_history=None,
):
    payload = {
        "session_id": session_id,
        "query": query,
        "document_id": document_id,
        "assistant_mode": assistant_mode,
        "web_search_enabled": web_search_enabled,
        "chat_history": chat_history or [],
    }

    response = requests.post(
        f"{API_BASE_URL}/rag/assistant/query",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def get_full_chat_history():
    return [
        {
            "query": item.get("query", ""),
            "answer": item.get("answer", ""),
            "document_id": item.get("document_id"),
            "query_category": item.get("query_category"),
            "source_type": item.get("source_type"),
            "web_search_used": item.get("web_search_used", False),
            "web_sources": item.get("web_sources", []),
        }
        for item in st.session_state.chat_history
    ]


def restore_session_state(session_data: dict):
    st.session_state.session_id = session_data.get("session_id")
    st.session_state.document_id = session_data.get("document_id")
    st.session_state.uploaded_filename = session_data.get("uploaded_filename")
    st.session_state.chat_history = session_data.get("chat_history", [])
    st.session_state.session_loaded = True

    if "assistant_mode" not in st.session_state:
        st.session_state.assistant_mode = "assistant"

    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True


def initialize_conversation():
    if "client_id" not in st.session_state:
        st.session_state.client_id = get_or_create_client_id()

    if not st.session_state.get("session_loaded", False):
        session_data = start_or_resume_backend_session(
            st.session_state.client_id
        )
        restore_session_state(session_data)


def reset_frontend_after_end():
    for key in (
        "session_id",
        "document_id",
        "uploaded_filename",
        "chat_history",
        "session_loaded",
        "assistant_mode",
        "web_search_enabled",
    ):
        st.session_state.pop(key, None)


def _render_web_sources(web_sources: list[dict], heading: str = "Web Sources"):
    if not web_sources:
        return

    st.markdown(f"**{heading}**")
    for index, source in enumerate(web_sources, start=1):
        title = str(source.get("title") or "Source").replace("]", "\\]")
        url = source.get("url", "")
        cited_text = source.get("cited_text", "")

        if url:
            st.markdown(f"{index}. [{title}]({url})")
        else:
            st.markdown(f"{index}. {title}")

        if cited_text:
            st.caption(f"Cited text: {cited_text}")


def _render_history_entry(entry: dict, history_number: int):
    query_text = entry.get("query", "")
    category = entry.get("query_category")
    source_type = entry.get("source_type")
    web_search_used = entry.get("web_search_used", False)
    web_sources = entry.get("web_sources", [])

    with st.expander(f"Query {history_number}: {query_text[:50]}..."):
        st.write(f"**Query**: {query_text}")
        st.write(f"**Answer**: {entry.get('answer', '')}")

        metadata = []
        if category:
            metadata.append(f"Route: {category}")
        if source_type:
            metadata.append(
                f"Source: {SOURCE_LABELS.get(source_type, source_type)}"
            )
        if category in {"WEB", "DOCUMENT_AND_WEB"}:
            metadata.append(
                "Web search: " + ("used" if web_search_used else "not verified")
            )
        if metadata:
            st.caption(" | ".join(metadata))

        _render_web_sources(web_sources)


def main():
    st.set_page_config(
        page_title="Document-Aware AI Assistant",
        page_icon="💬",
        layout="wide",
    )

    try:
        initialize_conversation()
    except requests.exceptions.RequestException as error:
        st.error(f"Could not start or resume the conversation: {error}")
        st.stop()

    with st.sidebar:
        st.header("Conversation")
        st.caption(f"Session: {st.session_state.session_id}")

        selected_mode = st.radio(
            "Response mode",
            options=["assistant", "strict"],
            format_func=lambda value: MODE_LABELS[value],
            index=0 if st.session_state.assistant_mode == "assistant" else 1,
            help=(
                "Assistant can use general knowledge, Google Search, and the "
                "document when relevant. Strict Document answers only from "
                "the PDF and conversation history."
            ),
        )
        st.session_state.assistant_mode = selected_mode

        web_search_enabled = st.checkbox(
            "Allow Google Search",
            value=st.session_state.web_search_enabled,
            disabled=selected_mode == "strict",
            help=(
                "When enabled, current or explicitly online questions can use "
                "Gemini Google Search grounding and return source links."
            ),
        )
        st.session_state.web_search_enabled = web_search_enabled

        if selected_mode == "assistant":
            st.caption(
                "The assistant automatically chooses general knowledge, the "
                "uploaded PDF, Google Search, or a combination."
            )
        else:
            st.caption(
                "Answers are limited to the uploaded document and history. "
                "Google Search is disabled."
            )

        if st.button("End Conversation", type="secondary"):
            try:
                end_backend_session(
                    client_id=st.session_state.client_id,
                    session_id=st.session_state.session_id,
                )
                reset_frontend_after_end()
                st.rerun()

            except requests.exceptions.RequestException as error:
                st.error(f"Failed to end conversation: {error}")

        st.divider()
        st.header("Optional Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            st.write(f"Selected file: `{uploaded_file.name}`")

            if st.button("Upload and Process PDF"):
                with st.spinner("Uploading and processing PDF..."):
                    try:
                        result = upload_pdf_to_backend(
                            uploaded_file=uploaded_file,
                            session_id=st.session_state.session_id,
                        )

                        st.session_state.document_id = result.get("document_id")
                        st.session_state.uploaded_filename = result.get(
                            "uploaded_filename",
                            uploaded_file.name,
                        )

                        # Keep document-independent general and web-only turns,
                        # but remove history tied to the previously active PDF.
                        st.session_state.chat_history = [
                            item
                            for item in st.session_state.chat_history
                            if item.get("document_id") is None
                        ]

                        st.success("PDF uploaded and attached to this conversation.")

                    except requests.exceptions.RequestException as error:
                        st.error(f"Failed to upload PDF: {error}")

        st.divider()
        st.header("Current Document")

        if st.session_state.document_id:
            st.write(f"File: `{st.session_state.uploaded_filename}`")
            st.write(f"Document ID: `{st.session_state.document_id}`")
        else:
            st.info("No PDF attached. Assistant mode remains available.")

        st.divider()
        st.header("Chat History")

        if st.session_state.chat_history:
            history_length = len(st.session_state.chat_history)
            for index, entry in enumerate(
                reversed(st.session_state.chat_history)
            ):
                _render_history_entry(
                    entry=entry,
                    history_number=history_length - index,
                )
        else:
            st.write("No queries yet.")

    st.title("💬 Document-Aware AI Assistant")
    st.markdown(
        "Ask general questions, search current web information, or upload a "
        "PDF for document-grounded answers and document/web comparison."
    )

    if (
        st.session_state.assistant_mode == "strict"
        and st.session_state.document_id is None
    ):
        st.warning("Strict Document mode requires a PDF upload.")
    elif st.session_state.document_id:
        st.info(
            "A document is attached. Assistant mode will use it only when the "
            "question requires document evidence."
        )

    query = st.text_input(
        "Enter your question:",
        placeholder=(
            "Ask a general question, request current web information, or ask "
            "about the uploaded document..."
        ),
    )

    if st.button("Submit Query", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        if (
            st.session_state.assistant_mode == "strict"
            and st.session_state.document_id is None
        ):
            st.warning("Upload a PDF before using Strict Document mode.")
            return

        effective_web_search_enabled = (
            st.session_state.web_search_enabled
            and st.session_state.assistant_mode == "assistant"
        )

        with st.spinner("Processing your query..."):
            try:
                result = query_assistant(
                    session_id=st.session_state.session_id,
                    document_id=st.session_state.document_id,
                    query=query,
                    assistant_mode=st.session_state.assistant_mode,
                    web_search_enabled=effective_web_search_enabled,
                    chat_history=get_full_chat_history(),
                )

                answer_text = result.get("answer", "")
                standalone_query = result.get("standalone_query", query)
                query_category = result.get("query_category", "")
                source_type = result.get("source_type", "")
                used_document_retrieval = result.get(
                    "used_document_retrieval",
                    False,
                )
                web_search_used = result.get("web_search_used", False)
                web_sources = result.get("web_sources", [])
                web_search_error = result.get("web_search_error")

                st.subheader("Result")
                st.write(f"**Query**: {query}")

                metadata_parts = []
                if query_category:
                    metadata_parts.append(f"Route: {query_category}")
                if source_type:
                    metadata_parts.append(
                        f"Source: {SOURCE_LABELS.get(source_type, source_type)}"
                    )
                metadata_parts.append(
                    "Document retrieval: "
                    + ("used" if used_document_retrieval else "bypassed")
                )
                if query_category in {"WEB", "DOCUMENT_AND_WEB"}:
                    metadata_parts.append(
                        "Web search: "
                        + ("used" if web_search_used else "not verified")
                    )
                st.caption(" | ".join(metadata_parts))

                if standalone_query != query:
                    st.write(f"**Rewritten Query**: {standalone_query}")

                st.write(f"**Answer**: {answer_text}")

                if web_search_error:
                    st.error(
                        "Google Search grounding failed. Check the FastAPI "
                        "terminal for the complete error."
                    )

                _render_web_sources(web_sources)

                history_document_id = (
                    None
                    if query_category
                    in {"GENERAL", "WEB", "DOCUMENT_UNAVAILABLE"}
                    else st.session_state.document_id
                )

                st.session_state.chat_history.append(
                    {
                        "query": query,
                        "answer": answer_text,
                        "document_id": history_document_id,
                        "query_category": query_category,
                        "source_type": source_type,
                        "web_search_used": web_search_used,
                        "web_sources": web_sources,
                    }
                )

            except requests.exceptions.HTTPError as error:
                detail = ""
                try:
                    detail = error.response.json().get("detail", "")
                except ValueError:
                    detail = ""
                st.error(detail or f"Backend request failed: {error}")

            except requests.exceptions.RequestException as error:
                st.error(f"Error querying backend: {error}")


if __name__ == "__main__":
    main()