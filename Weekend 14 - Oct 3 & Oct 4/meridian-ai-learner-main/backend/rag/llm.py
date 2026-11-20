"""The LLM used by both the RAG chain and the agents.

Models are accessed via Mesh API (OpenAI-compatible gateway) or Google AI Studio (API key).
Embeddings (rag/embeddings.py) use Mesh API or Gemini API.
"""
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings
from exceptions import ConfigurationError


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Create the chat model on first use (not at import time) and reuse it afterwards."""
    if settings.mesh_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.mesh_model,
            api_key=settings.mesh_api_key,
            base_url=settings.mesh_base_url,
            temperature=settings.llm_temperature,
        )

    if not settings.GOOGLE_API_KEY:
        raise ConfigurationError("Missing MESH_API_KEY or GOOGLE_API_KEY in .env.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm_model_name,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.llm_temperature,
    )


def extract_text(content) -> str:
    """Gemini may return a list of content blocks, e.g. [{'type': 'text', 'text': '...'}]; join the text."""
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content
