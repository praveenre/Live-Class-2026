"""Tests for the local/cloud toggle, the local RAG pipeline and the agent middleware.

Everything runs offline: embeddings and the chat model are replaced by fakes.
"""
import pytest
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent.middleware import build_agent_middleware
from api.main import app
from config.settings import Settings, settings
from exceptions import ConfigurationError
from rag import retrieval, vector_store
from rag.data_ingestion import ingest_pdf, ingest_stored_pdfs

client = TestClient(app)


class FakeToolModel(GenericFakeChatModel):
    """A scripted chat model that 'supports' tools, so create_agent accepts it."""

    def bind_tools(self, tools, **kwargs):
        return self


# ---------------------------------------------------------------------------
# The toggle
# ---------------------------------------------------------------------------

def test_default_mode_is_local(monkeypatch):
    monkeypatch.delenv("APP_MODE", raising=False)
    assert Settings(_env_file=None).app_mode == "local"


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_MODE", "hybrid")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_cloud_mode_lists_missing_settings(monkeypatch):
    monkeypatch.setattr(settings, "app_mode", "cloud")
    vector_store.get_vector_store.cache_clear()
    try:
        with pytest.raises(ConfigurationError, match="VECTOR_SEARCH_INDEX_ID"):
            vector_store.get_vector_store()
    finally:
        vector_store.get_vector_store.cache_clear()


def test_missing_api_key_gives_503_not_500(monkeypatch):
    from rag import llm

    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    llm.get_llm.cache_clear()
    try:
        res = client.post("/api/rag/ask", json={"query": "hi"})
        assert res.status_code == 503
        assert "GOOGLE_API_KEY" in res.json()["detail"]
    finally:
        llm.get_llm.cache_clear()


# ---------------------------------------------------------------------------
# Local RAG end to end: PDF -> Chroma -> retrieval -> answer
# ---------------------------------------------------------------------------

@pytest.fixture
def offline_local_store(monkeypatch):
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: DeterministicFakeEmbedding(size=768))
    vector_store.get_vector_store.cache_clear()
    yield vector_store.get_vector_store()
    vector_store.get_vector_store.cache_clear()


def test_local_ingest_and_ask(offline_local_store, monkeypatch, tmp_path):
    import glob
    import os
    import shutil

    pdf = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "sample_docs", "*.pdf")))[0]
    summary = ingest_pdf(pdf, "local://sample.pdf")
    assert summary["chunks"] > 0

    # Re-ingesting the same file must not duplicate chunks (stable IDs)
    before = offline_local_store._collection.count()
    ingest_pdf(pdf, "local://sample.pdf")
    assert offline_local_store._collection.count() == before

    # The "Index stored documents" button reads from the local upload folder
    os.makedirs(settings.local_upload_dir, exist_ok=True)
    shutil.copy(pdf, settings.local_upload_dir)
    assert ingest_stored_pdfs()["files"] == 1

    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="EUR 2.84 billion")]))
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)
    assert retrieval.ask_question("What is the revenue?") == "EUR 2.84 billion"


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

def test_middleware_stack_has_the_expected_layers():
    names = [type(m).__name__ for m in build_agent_middleware()]
    for expected in ["PIIMiddleware", "ModelCallLimitMiddleware", "ToolCallLimitMiddleware",
                     "ModelRetryMiddleware", "ToolRetryMiddleware"]:
        assert expected in names


def test_agent_runs_tools_through_middleware_and_redacts_email():
    calls = []

    @tool
    def lookup_vendor(vendor_name: str) -> str:
        """Look a vendor up."""
        calls.append(vendor_name)
        return "CLEARED"

    model = FakeToolModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "lookup_vendor", "args": {"vendor_name": "Takumi"}, "id": "1"}]),
        AIMessage(content="SANCTIONS CHECK: CLEARED"),
    ]))
    agent = create_agent(model=model, tools=[lookup_vendor], middleware=build_agent_middleware())
    result = agent.invoke({"messages": [HumanMessage("Screen Takumi, contact buyer@aldermoor.example")]})

    assert calls == ["Takumi"]
    assert result["messages"][-1].text == "SANCTIONS CHECK: CLEARED"
    assert "buyer@aldermoor.example" not in result["messages"][0].text


def test_supervisor_builds_three_agents_with_middleware():
    from agent.agents import ProcurementSupervisor

    supervisor = ProcurementSupervisor()  # no network call happens at build time
    assert supervisor.risk_agent and supervisor.tax_agent and supervisor.control_agent
