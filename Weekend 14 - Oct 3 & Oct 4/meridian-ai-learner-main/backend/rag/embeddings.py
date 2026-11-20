"""Text -> vector conversion, used both when indexing documents and when searching.

Embeddings route through:
  * Mesh API / OpenAI-compatible endpoint (when MESH_API_KEY is configured)
  * Google AI Studio direct key (when GOOGLE_API_KEY is configured)

All models output 768-dimensional vectors matching the vector database index.
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config.settings import settings
from exceptions import ConfigurationError


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if settings.mesh_api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.mesh_embedding_model,
            api_key=settings.mesh_api_key,
            base_url=settings.mesh_base_url,
            check_embedding_ctx_length=False,
        )

    if not settings.GOOGLE_API_KEY:
        raise ConfigurationError("Embeddings require MESH_API_KEY or GOOGLE_API_KEY in .env.")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.local_embedding_model_name,
        google_api_key=settings.GOOGLE_API_KEY,
        output_dimensionality=settings.embedding_dimensions,
    )
