"""
Streamlit frontend for the Document Q&A System.

This version supports:
1. Anonymous browser clients.
2. Resumable active conversations.
3. One active PDF per conversation.
4. Persistent chat history through SQLite.
5. Explicit conversation termination.
"""

import uuid

import requests
import streamlit as st

from app.core.logger import configure_logging


logger = configure_logging("STREAMLIT_APP")
API_BASE_URL = "http://localhost:8000"


def _get_query_parameter(name: str):
    value = st.query_params.get(name)

    if isinstance(value, list):
        return value[0] if value else None

    return value


def get_or_create_client_id() -> str:
    """
    Store the anonymous browser identifier in the page URL.

    The same URL can resume the same browser client's active conversation
    after the frontend and backend are restarted.
    """
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


def query_document(
    session_id: str,
    document_id: str,
    query: str,
    chat_history=None,
):
    payload = {
        "session_id": session_id,
        "query": query,
        "document_id": document_id,
        "chat_history": chat_history or [],
    }

    response = requests.post(
        f"{API_BASE_URL}/rag/query-by-document",
        json=payload,
        timeout=300,
    )

    response.raise_for_status()
    return response.json()


def get_full_history_for_current_document(document_id: str):
    return [
        {
            "query": item.get("query", ""),
            "answer": item.get("answer", ""),
            "document_id": item.get("document_id", ""),
        }
        for item in st.session_state.chat_history
        if item.get("document_id") == document_id
    ]


def restore_session_state(session_data: dict):
    st.session_state.session_id = session_data.get("session_id")
    st.session_state.document_id = session_data.get("document_id")
    st.session_state.uploaded_filename = session_data.get("uploaded_filename")
    st.session_state.chat_history = session_data.get("chat_history", [])
    st.session_state.session_loaded = True


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
    ):
        st.session_state.pop(key, None)


def main():
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide",
    )

    try:
        initialize_conversation()
    except requests.exceptions.RequestException as error:
        st.error(
            f"Could not start or resume the conversation: {error}"
        )
        st.stop()

    with st.sidebar:
        st.header("Conversation")
        st.caption(f"Session: {st.session_state.session_id}")

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
        st.header("Upload Document")

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

                        # One active document per conversation.
                        st.session_state.chat_history = []

                        st.success(
                            "PDF uploaded and processed successfully."
                        )

                    except requests.exceptions.RequestException as error:
                        st.error(f"Failed to upload PDF: {error}")

        st.divider()
        st.header("Current Document")

        if st.session_state.document_id:
            st.write(
                f"File: `{st.session_state.uploaded_filename}`"
            )
            st.write(
                f"Document ID: `{st.session_state.document_id}`"
            )
        else:
            st.info("No PDF uploaded yet.")

        st.divider()
        st.header("Chat History")

        if st.session_state.chat_history:
            for index, entry in enumerate(
                reversed(st.session_state.chat_history)
            ):
                history_number = (
                    len(st.session_state.chat_history) - index
                )

                with st.expander(
                    f"Query {history_number}: "
                    f"{entry.get('query', '')[:50]}..."
                ):
                    st.write(f"**Query**: {entry.get('query', '')}")
                    st.write(f"**Answer**: {entry.get('answer', '')}")
        else:
            st.write("No queries yet.")

    st.title("📚 Document Q&A System")
    st.markdown(
        "Upload a PDF document first, then ask questions about that document."
    )

    if st.session_state.document_id is None:
        st.warning(
            "Please upload and process a PDF document before asking questions."
        )

    query = st.text_input(
        "Enter your question:",
        placeholder="Example: What is this document about?",
    )

    if st.button("Submit Query"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        if st.session_state.document_id is None:
            st.warning(
                "Please upload and process a PDF document before asking a question."
            )
            return

        with st.spinner("Processing your query..."):
            try:
                full_history = get_full_history_for_current_document(
                    st.session_state.document_id
                )

                result = query_document(
                    session_id=st.session_state.session_id,
                    document_id=st.session_state.document_id,
                    query=query,
                    chat_history=full_history,
                )

                answer_text = result.get("answer", "")
                standalone_query = result.get("standalone_query", query)
                query_category = result.get("query_category", "")

                st.subheader("Result")
                st.write(f"**Query**: {query}")

                if query_category:
                    st.caption(f"Query category: {query_category}")

                if standalone_query != query:
                    st.write(f"**Rewritten Query**: {standalone_query}")

                st.write(f"**Answer**: {answer_text}")

                st.session_state.chat_history.append(
                    {
                        "query": query,
                        "answer": answer_text,
                        "document_id": st.session_state.document_id,
                    }
                )

            except requests.exceptions.RequestException as error:
                st.error(f"Error querying backend: {error}")


if __name__ == "__main__":
    main()
