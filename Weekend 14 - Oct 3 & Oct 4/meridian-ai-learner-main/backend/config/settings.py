"""All configuration in one place, read from environment variables / .env.

THE LOCAL <-> PINECONE / CLOUD TOGGLE
--------------------------------------
    APP_MODE=local   (default)  Runs on your laptop with local storage.
                                Vectors  -> Chroma, saved in ./.meridian/chroma
                                Files    -> ./.meridian/uploads
                                Embeddings -> Mesh API or Google AI Studio (API key)

    APP_MODE=cloud              Runs with Pinecone cloud vector database.
                                Vectors  -> Pinecone Serverless
                                Files    -> ./.meridian/uploads
                                Embeddings -> Mesh API (google/text-embedding-005) or Google AI Studio
"""
import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels above backend/config/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "Meridian AI - Procurement Intelligence"
    app_env: str = Field("local", validation_alias="ENVIRONMENT")
    app_version: str = "1.1.0"

    # --- The toggle ---
    app_mode: Literal["local", "cloud"] = Field("local", validation_alias="APP_MODE")

    # --- Vector backend (pinecone, chroma) ---
    vector_backend: Literal["pinecone", "chroma"] = Field("pinecone", validation_alias="VECTOR_BACKEND")
    pinecone_api_key: str = Field("", validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("meridian-docs", validation_alias="PINECONE_INDEX_NAME")

    # --- Mesh API Gateway (OpenAI-compatible) ---
    mesh_api_key: str = Field("", validation_alias="MESH_API_KEY")
    mesh_base_url: str = Field("https://api.meshapi.ai/v1", validation_alias="MESH_BASE_URL")
    mesh_model: str = Field("google/gemini-2.5-flash-lite", validation_alias="MESH_MODEL")
    mesh_embedding_model: str = Field("google/text-embedding-005", validation_alias="MESH_EMBEDDING_MODEL")

    # --- Gemini (used in BOTH modes for the chat model fallback) ---
    GOOGLE_API_KEY: str = Field("", validation_alias="GOOGLE_API_KEY")
    llm_model_name: str = Field("gemini-2.5-flash-lite", validation_alias="LLM_MODEL_NAME")
    # 0 = most repeatable answers. If the model loops or repeats itself, try 1.0
    llm_temperature: float = Field(0.0, validation_alias="LLM_TEMPERATURE")

    # --- Embeddings ---
    embedding_model_name: str = Field("text-embedding-005", validation_alias="EMBEDDING_MODEL_NAME")
    local_embedding_model_name: str = Field("gemini-embedding-001", validation_alias="LOCAL_EMBEDDING_MODEL_NAME")
    embedding_dimensions: int = 768

    # --- Local mode storage ---
    local_data_dir: str = Field(str(PROJECT_ROOT / ".meridian"), validation_alias="LOCAL_DATA_DIR")
    chroma_collection: str = Field("meridian_docs", validation_alias="CHROMA_COLLECTION")

    # --- Agent middleware limits ---
    agent_max_model_calls: int = Field(8, validation_alias="AGENT_MAX_MODEL_CALLS")
    agent_max_tool_calls: int = Field(6, validation_alias="AGENT_MAX_TOOL_CALLS")
    agent_max_retries: int = Field(2, validation_alias="AGENT_MAX_RETRIES")

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),  # found no matter which folder you start from
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Convenience ---
    @property
    def is_pinecone(self) -> bool:
        if self.vector_backend == "pinecone":
            return True
        if self.vector_backend == "chroma":
            return False
        return bool(self.pinecone_api_key)

    @property
    def is_cloud(self) -> bool:
        return self.app_mode == "cloud" or self.is_pinecone

    @property
    def chroma_dir(self) -> str:
        return os.path.join(self.local_data_dir, "chroma")

    @property
    def local_upload_dir(self) -> str:
        return os.path.join(self.local_data_dir, "uploads")

    def missing_pinecone_settings(self) -> list[str]:
        """Names of Pinecone settings that are still empty."""
        required = {
            "PINECONE_API_KEY": self.pinecone_api_key,
            "PINECONE_INDEX_NAME": self.pinecone_index_name,
        }
        return [name for name, value in required.items() if not value]


settings = Settings()
