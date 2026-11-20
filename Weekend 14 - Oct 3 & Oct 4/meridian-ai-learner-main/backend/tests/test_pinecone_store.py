"""Tests for Pinecone vector store configuration and FloatEmbeddingsWrapper."""
import pytest
import numpy as np

from config.settings import settings
from exceptions import ConfigurationError
from rag import vector_store
from rag.vector_store import FloatEmbeddingsWrapper


def test_pinecone_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.setattr(settings, "vector_backend", "pinecone")
    monkeypatch.setattr(settings, "pinecone_api_key", "")
    vector_store.get_vector_store.cache_clear()
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            vector_store.get_vector_store()
        assert "PINECONE_API_KEY" in str(exc_info.value)
    finally:
        vector_store.get_vector_store.cache_clear()


def test_float_embeddings_wrapper_converts_numpy_floats():
    class DummyEmbeddings:
        def embed_documents(self, texts):
            return [np.array([1.0, 2.5, 3.7], dtype=np.float64) for _ in texts]

        def embed_query(self, text):
            return np.array([4.2, 5.1, 6.0], dtype=np.float64)

    wrapper = FloatEmbeddingsWrapper(DummyEmbeddings())

    docs_res = wrapper.embed_documents(["test text"])
    assert len(docs_res) == 1
    assert all(type(x) is float for x in docs_res[0])
    assert docs_res[0] == [1.0, 2.5, 3.7]

    query_res = wrapper.embed_query("query")
    assert all(type(x) is float for x in query_res)
    assert query_res == [4.2, 5.1, 6.0]
