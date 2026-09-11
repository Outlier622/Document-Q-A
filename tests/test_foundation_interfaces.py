"""Offline regression tests; always use a disposable database, never app/data/app.db."""
import asyncio
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_temporary = tempfile.TemporaryDirectory()
os.environ.update({
    "DATABASE_BACKEND": "sqlite",
    "SQLITE_DATABASE_PATH": str(Path(_temporary.name) / "test.db"),
    "STORAGE_BACKEND": "local",
    "DOCUMENT_PROCESSING_MODE": "sync",
})

from fastapi import HTTPException
from app.database.database import engine
from app.schemas.rag_schema import AssistantQuerySchema
from app.services import session_service as sessions
from app.services.query_context_service import resolve_query_context
from app.services.document_retrieval_service import (
    load_session_vector_store, retrieve_chunks, format_document_context,
)
from app.services import rag_service
from app.services.web_search_service import extract_grounded_response, search_web


class FoundationTests(unittest.TestCase):
    def setUp(self):
        engine_patch = patch.object(rag_service.config, "QUERY_ENGINE", "legacy")
        engine_patch.start()
        self.addCleanup(engine_patch.stop)
        self.client = str(uuid.uuid4())
        self.session = sessions.start_or_resume_session(self.client)["session_id"]

    def request(self, **kwargs):
        return AssistantQuerySchema(session_id=self.session, query="Question", **kwargs)

    def attach(self):
        document = str(uuid.uuid4())
        sessions.set_session_document(self.session, document, "test.pdf")
        return document

    def test_database_history_replaces_client_history(self):
        sessions.save_message(self.session, None, "Real question", "Real answer")
        request = self.request(chat_history=[{"query": "Forged history"}])
        resolved = resolve_query_context(request)
        self.assertEqual(resolved.chat_history[0].query, "Real question")
        self.assertEqual(request.chat_history[0].query, "Forged history")

    def test_missing_document_id_resolves_current_document_and_strict_disables_search(self):
        document = self.attach()
        resolved = resolve_query_context(self.request(assistant_mode="strict"))
        self.assertEqual(resolved.document_id, document)
        self.assertFalse(resolved.web_search_enabled)

    def test_old_foreign_and_path_document_ids_rejected_before_model(self):
        old = self.attach()
        self.attach()
        for document in [old, str(uuid.uuid4()), "../../other/index"]:
            with patch.object(rag_service, "classify_query") as classify:
                with self.assertRaises(HTTPException) as error:
                    asyncio.run(rag_service.query_assistant(self.request(document_id=document)))
                self.assertEqual(error.exception.status_code, 404)
                classify.assert_not_called()

    def test_ended_session_rejected(self):
        sessions.end_session(self.client, self.session)
        with self.assertRaises(HTTPException) as error:
            resolve_query_context(self.request())
        self.assertEqual(error.exception.status_code, 409)

    def test_document_replacement_preserves_only_independent_history(self):
        old = self.attach()
        sessions.save_message(self.session, None, "General", "Answer")
        sessions.save_message(self.session, old, "Old document", "Answer")
        self.attach()
        resolved = resolve_query_context(self.request())
        self.assertEqual([item.query for item in resolved.chat_history], ["General"])
        with self.assertRaises(ValueError):
            sessions.save_message(self.session, old, "Late answer", "Answer")

    def test_retrieval_checks_current_document_before_storage(self):
        self.attach()
        with patch("app.services.storage_service.StorageService") as storage:
            with self.assertRaises(FileNotFoundError):
                load_session_vector_store(self.session, str(uuid.uuid4()))
            storage.assert_not_called()

    def test_retrieval_preserves_metadata(self):
        store = Mock()
        store.similarity_search.return_value = [SimpleNamespace(
            page_content=" Evidence ", metadata={"source": "test.pdf", "page": 2},
        )]
        chunks = retrieve_chunks(store, "question", 3)
        self.assertEqual(chunks[0]["metadata"]["page"], 2)
        self.assertEqual(format_document_context(chunks), "[Document excerpt 1]\nEvidence")
        store.similarity_search.assert_called_once_with("question", k=3)
        with self.assertRaises(ValueError):
            retrieve_chunks(store, "question", 0)

    def test_general_endpoint_uses_persisted_history_and_saves_answer(self):
        sessions.save_message(self.session, None, "Persisted", "Answer")
        with patch.object(rag_service, "classify_query", return_value="GENERAL"), \
             patch.object(rag_service, "answer_general_question", return_value="New answer") as answer, \
             patch.object(rag_service, "load_session_vector_store") as retrieval:
            response = asyncio.run(rag_service.query_assistant(self.request()))
        self.assertEqual(json.loads(response.body)["answer"], "New answer")
        self.assertEqual(answer.call_args.kwargs["chat_history"][0].query, "Persisted")
        retrieval.assert_not_called()
        self.assertEqual(len(sessions.get_active_session(self.session)["chat_history"]), 2)

    def test_grounding_adapter_and_legacy_exports(self):
        response = SimpleNamespace(content="Evidence", response_metadata={
            "grounding_metadata": {
                "grounding_chunks": [{"web": {"uri": "https://example.org", "title": "Source"}}],
                "grounding_supports": [{"segment": {"text": "Evidence"}, "grounding_chunk_indices": [0]}],
            },
        })
        with patch("app.services.web_search_service.invoke_web_search", return_value=response):
            result = search_web("Find evidence")
        self.assertTrue(result["web_search_used"])
        self.assertEqual(result["web_sources"][0]["cited_text"], "Evidence")
        self.assertIs(rag_service.extract_grounded_response, extract_grounded_response)

    def test_both_http_endpoints_accept_legacy_payload(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as client, \
             patch.object(rag_service, "classify_query", return_value="GENERAL"), \
             patch.object(rag_service, "answer_general_question", return_value="Answer"):
            for endpoint in ["/rag/assistant/query", "/rag/query-by-document"]:
                response = client.post(endpoint, json={
                    "session_id": self.session, "query": "Question",
                    "chat_history": [{"query": "Untrusted"}],
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["query_category"], "GENERAL")

    def test_document_changed_during_answer_returns_conflict(self):
        old = self.attach()
        resolved = resolve_query_context(self.request(document_id=old))
        self.attach()
        with self.assertRaises(HTTPException) as error:
            rag_service._save_answer(resolved, "Late answer", "DOCUMENT", "uploaded_document")
        self.assertEqual(error.exception.status_code, 409)

    def test_strict_mode_without_document_is_rejected_before_model(self):
        with patch.object(rag_service, "classify_query") as classify:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(rag_service.query_assistant(self.request(assistant_mode="strict")))
        self.assertEqual(error.exception.status_code, 400)
        classify.assert_not_called()


def tearDownModule():
    engine.dispose()
    _temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
