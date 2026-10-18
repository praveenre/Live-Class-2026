"""The vector database that stores our embeddings. Which one depends on APP_MODE.

    local  -> Chroma, a vector database that saves to a folder on your disk
    cloud  -> Vertex AI Vector Search, Google's managed vector database

Both are LangChain `VectorStore`s, so the rest of the code (ingestion, retrieval)
calls the same methods (`add_documents`, `as_retriever`) and never knows which one it got.
"""
from functools import lru_cache

from langchain_core.vectorstores import VectorStore

from config.settings import settings
from exceptions import ConfigurationError
from logger import GLOBAL_LOGGER as log
from rag.embeddings import get_embeddings


def _local_store() -> VectorStore:
    from langchain_chroma import Chroma  # imported here so cloud images do not need it loaded

    log.info("Using local Chroma vector store", path=settings.chroma_dir)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def _cloud_store() -> VectorStore:
    missing = settings.missing_cloud_settings()
    if missing:
        raise ConfigurationError(f"APP_MODE=cloud but these settings are empty: {', '.join(missing)}")

    from google.cloud import aiplatform
    from langchain_google_vertexai import VectorSearchVectorStore

    log.info("Using Vertex AI Vector Search", index_id=settings.vector_search_index_id)
    aiplatform.init(project=settings.GCP_PROJECT, location=settings.GCP_REGION)
    return VectorSearchVectorStore.from_components(
        project_id=settings.GCP_PROJECT,
        region=settings.GCP_REGION,
        embedding=get_embeddings(),  # NOTE: singular "embedding", not "embeddings"
        index_id=settings.vector_search_index_id,
        endpoint_id=settings.vector_search_index_endpoint_id,
        gcs_bucket_name=settings.GCS_BUCKET_NAME,
        stream_update=True,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Connect on first use, so the API can start (e.g. /api/health) before anything is configured."""
    return _cloud_store() if settings.is_cloud else _local_store()
