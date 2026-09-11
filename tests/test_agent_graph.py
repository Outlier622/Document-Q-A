"""Exercise the real graph and HTTP integration without paid model calls."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch
import unittest

# Reuse the disposable database setup; no real application DB is opened.
from test_foundation_interfaces import sessions, rag_service
from app.agents.graph import run_agent
from app.agents.tools import build_tools
from app.schemas.rag_schema import AssistantQuerySchema, ChatHistoryItem
from langchain_core.messages import AIMessage, ToolMessage


def call(name, query="question", call_id="call-1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": {} if name == "conversation_history" else {"query": query},
        "id": call_id,
    }])


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.inputs.append(list(messages))
        return next(self.responses)


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = SimpleNamespace(AGENT_TOP_K=5, AGENT_MAX_ROUNDS=4,
                                      AGENT_MAX_TOOL_CALLS=6, AGENT_TIMEOUT_SECONDS=10)
        self.request = AssistantQuerySchema(session_id="test-session", query="Question")

    async def test_general_answer_without_tools(self):
        model = ScriptedModel([AIMessage(content="General answer")])
        result = await run_agent(self.request, self.config, model)
        self.assertEqual(result["answer"], "General answer")
        self.assertEqual(result["agent_steps"], [])
        self.assertEqual(result["execution_engine"], "langgraph")

    async def test_document_then_web_then_synthesis(self):
        self.request.document_id = "document"
        model = ScriptedModel([call("document_search"), call("web_search", call_id="call-2"),
                               AIMessage(content="Document and current web comparison")])
        with patch("app.agents.tools.retrieve_document", return_value=[
            {"excerpt_id": 1, "text": "Document evidence", "metadata": {"page": 2}},
        ]) as retrieve, patch("app.agents.tools.search_web", return_value={
            "answer": "Current evidence", "web_search_used": True,
            "web_sources": [{"title": "Official", "url": "https://example.org", "cited_text": "Intermediate answer"}],
        }):
            result = await run_agent(self.request, self.config, model)
        retrieve.assert_called_once_with("test-session", "question", 5, document_id="document")
        self.assertEqual(result["query_category"], "DOCUMENT_AND_WEB")
        self.assertTrue(result["used_document_retrieval"])
        self.assertEqual(result["web_sources"][0]["cited_text"], "")
        self.assertEqual(result["document_sources"][0]["metadata"]["page"], 2)
        self.assertTrue(any(isinstance(message, ToolMessage) for message in model.inputs[-1]))

    async def test_strict_mode_denies_hallucinated_web_tool(self):
        self.request.document_id = "document"
        self.request.assistant_mode = "strict"
        model = ScriptedModel([call("web_search"), AIMessage(content="Unsupported claim")])
        with patch("app.agents.tools.search_web") as search:
            result = await run_agent(self.request, self.config, model)
        search.assert_not_called()
        self.assertNotIn("web_search", [tool.name for tool in model.tools])
        self.assertEqual(result["agent_steps"][0]["status"], "denied")
        self.assertNotEqual(result["answer"], "Unsupported claim")

    async def test_search_toggle_removes_web_tool(self):
        self.request.web_search_enabled = False
        self.assertNotIn("web_search", build_tools(self.request, 5))

    async def test_empty_history_does_not_authorize_strict_answer(self):
        self.request.document_id = "document"
        self.request.assistant_mode = "strict"
        model = ScriptedModel([call("conversation_history"), AIMessage(content="Unsupported claim")])
        result = await run_agent(self.request, self.config, model)
        self.assertNotEqual(result["answer"], "Unsupported claim")

    async def test_history_tool_only_uses_server_snapshot(self):
        self.request.chat_history = [ChatHistoryItem(query="Past question", answer="Past answer")]
        model = ScriptedModel([call("conversation_history"), AIMessage(content="Past question")])
        result = await run_agent(self.request, self.config, model)
        self.assertEqual(result["query_category"], "CONVERSATION_HISTORY")
        self.assertIn("Past question", model.inputs[-1][-1].content)

    async def test_strict_history_excludes_web_and_general_answers(self):
        self.request.document_id = "document"
        self.request.assistant_mode = "strict"
        self.request.chat_history = [
            ChatHistoryItem(query="Web fact", document_id="document", source_type="document_and_web"),
            ChatHistoryItem(query="PDF fact", document_id="document", source_type="uploaded_document"),
        ]
        result = await build_tools(self.request, 5)["conversation_history"].ainvoke({})
        self.assertEqual([item["query"] for item in result["turns"]], ["PDF fact"])

    async def test_tool_budget_stops_multiple_calls_in_one_response(self):
        self.config.AGENT_MAX_TOOL_CALLS = 1
        response = call("conversation_history")
        response.tool_calls.append({"name": "conversation_history", "args": {}, "id": "call-2"})
        model = ScriptedModel([response, AIMessage(content="Final answer")])
        result = await run_agent(self.request, self.config, model)
        self.assertTrue(result["agent_limit_reached"])
        self.assertEqual([step["status"] for step in result["agent_steps"]], ["completed", "limit_reached"])

    async def test_round_budget_stops_repeated_tool_calls(self):
        self.config.AGENT_MAX_ROUNDS = 1
        model = ScriptedModel([call("conversation_history"), AIMessage(content="Final")])
        result = await run_agent(self.request, self.config, model)
        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(result["agent_limit_reached"])

    async def test_failed_tool_is_reported_without_leaking_exception(self):
        model = ScriptedModel([call("web_search"), AIMessage(content="Cannot verify")])
        with patch("app.agents.tools.search_web", side_effect=RuntimeError("secret-token")):
            result = await run_agent(self.request, self.config, model)
        self.assertFalse(result["web_search_used"])
        self.assertEqual(result["agent_steps"][0]["status"], "failed")
        self.assertNotIn("secret-token", model.inputs[-1][-1].content)

    async def test_timeout_cancels_graph(self):
        class SlowModel(ScriptedModel):
            async def ainvoke(self, messages):
                await asyncio.sleep(1)
        self.config.AGENT_TIMEOUT_SECONDS = 0.01
        with self.assertRaises(asyncio.TimeoutError):
            await run_agent(self.request, self.config, SlowModel([]))

    async def test_request_cannot_override_bound_document(self):
        self.request.document_id = "document"
        tool = build_tools(self.request, 5)["document_search"]
        with patch("app.agents.tools.retrieve_document") as retrieve:
            with self.assertRaises(ValueError):
                await tool.ainvoke({"query": "Question", "document_id": "foreign"})
        retrieve.assert_not_called()

    async def test_agent_http_path_persists_answer(self):
        import uuid
        import json
        session = sessions.start_or_resume_session(str(uuid.uuid4()))
        request = AssistantQuerySchema(session_id=session["session_id"], query="Hello")
        model = ScriptedModel([AIMessage(content="Hello back")])
        with patch.object(rag_service.config, "QUERY_ENGINE", "agent"), \
             patch("app.agents.graph.initialize_llm", return_value=model):
            response = await rag_service.query_assistant(request)
        self.assertEqual(json.loads(response.body)["execution_engine"], "langgraph")
        self.assertEqual(sessions.get_active_session(session["session_id"])["chat_history"][-1]["answer"], "Hello back")


if __name__ == "__main__":
    unittest.main()
