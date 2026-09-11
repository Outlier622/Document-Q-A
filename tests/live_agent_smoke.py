"""Opt-in Gemini smoke test using synthetic history and a temporary database.

Run: python -B tests/live_agent_smoke.py
This makes paid/quota-counted Gemini calls; it does not read user documents.
"""
import asyncio
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory() as directory:
        os.environ.update({"DATABASE_BACKEND": "sqlite", "STORAGE_BACKEND": "local",
                           "DOCUMENT_PROCESSING_MODE": "sync",
                           "SQLITE_DATABASE_PATH": str(Path(directory) / "smoke.db")})
        from app.agents.graph import run_agent
        from app.config.configuration import Config
        from app.database.database import engine
        from app.schemas.rag_schema import AssistantQuerySchema, ChatHistoryItem
        request = AssistantQuerySchema(
            session_id="synthetic-smoke", web_search_enabled=False,
            query="Use the conversation_history tool to tell me the project codename I chose earlier.",
            chat_history=[ChatHistoryItem(query="The project codename is MAPLE-742.", answer="Understood.")],
        )
        config = Config()
        config.AGENT_TIMEOUT_SECONDS = 45
        try:
            result = asyncio.run(run_agent(request, config))
            assert "MAPLE-742" in result["answer"], "Synthetic history was not recovered"
            assert any(step == {"tool": "conversation_history", "status": "completed"}
                       for step in result["agent_steps"]), "Model did not execute the history tool"
            print("Live Gemini + LangGraph tool loop: PASS")
            print("Tools:", result["agent_steps"])
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
