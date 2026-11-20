"""The vector database that stores our embeddings.

    pinecone -> Pinecone Serverless vector database in the cloud
    chroma   -> Chroma, an open-source vector database saved to disk (.meridian/chroma)

Both are LangChain `VectorStore`s, so the rest of the code (ingestion, retrieval)
calls the same methods (`add_documents`, `similarity_search`, `as_retriever`).
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config.settings import settings
from exceptions import ConfigurationError
from logger import GLOBAL_LOGGER as log
from rag.embeddings import get_embeddings


class FloatEmbeddingsWrapper(Embeddings):
    """Ensures all embedding values are pure Python floats for Pinecone's strict type validator."""

    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.base_embeddings.embed_documents(texts)
        return [[float(x) for x in vec] for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        vec = self.base_embeddings.embed_query(text)
        return [float(x) for x in vec]


def _pinecone_store() -> VectorStore:
    missing = settings.missing_pinecone_settings()
    if missing:
        raise ConfigurationError(f"Pinecone vector store selected, but these settings are empty: {', '.join(missing)}")

    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    log.info("Using Pinecone vector store", index_name=settings.pinecone_index_name)
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    return PineconeVectorStore(index=index, embedding=FloatEmbeddingsWrapper(get_embeddings()))


def _local_store() -> VectorStore:
    from langchain_chroma import Chroma

    log.info("Using local Chroma vector store", path=settings.chroma_dir)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Connect on first use, so the API can start before anything is configured."""
    if settings.is_pinecone:
        return _pinecone_store()
    return _local_store()
