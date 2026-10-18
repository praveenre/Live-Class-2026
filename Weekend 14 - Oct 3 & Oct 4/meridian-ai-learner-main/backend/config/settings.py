"""All configuration in one place, read from environment variables / .env.

THE LOCAL <-> CLOUD TOGGLE
--------------------------
    APP_MODE=local   (default)  Runs on your laptop with ONLY a Gemini API key.
                                Vectors  -> Chroma, saved in ./.meridian/chroma
                                Files    -> ./.meridian/uploads
                                Embeddings -> Gemini API (API key)
                                No Google Cloud project, bucket or service account needed.

    APP_MODE=cloud              Runs against Google Cloud (what Cloud Run uses).
                                Vectors  -> Vertex AI Vector Search
                                Files    -> Cloud Storage bucket
                                Embeddings -> Vertex AI (Google Cloud credentials)

Every module that touches storage asks `settings.is_cloud` instead of guessing,
so switching mode is one line in .env.
"""
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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

    # --- Gemini (used in BOTH modes for the chat model) ---
    GOOGLE_API_KEY: str = Field("", validation_alias="GOOGLE_API_KEY")
    llm_model_name: str = Field("gemini-3.8-flash", validation_alias="VERTEX_LLM_MODEL_NAME")
    # 0 = most repeatable answers. If the model loops or repeats itself, try 1.0 (Gemini 3 default)
    llm_temperature: float = Field(0.0, validation_alias="LLM_TEMPERATURE")

    # --- Embeddings ---
    # Cloud: Vertex AI model. Must output 768 dims to match the Vector Search index.
    embedding_model_name: str = Field("text-embedding-005", validation_alias="VERTEX_EMBEDDING_MODEL_NAME")
    # Local: Gemini API model (API key). Chroma has no fixed size, 768 keeps both modes alike.
    local_embedding_model_name: str = Field("gemini-embedding-001", validation_alias="LOCAL_EMBEDDING_MODEL_NAME")
    embedding_dimensions: int = 768

    # --- Local mode storage ---
    local_data_dir: str = Field(str(PROJECT_ROOT / ".meridian"), validation_alias="LOCAL_DATA_DIR")
    chroma_collection: str = Field("meridian_docs", validation_alias="CHROMA_COLLECTION")

    # --- Cloud mode (GCP) ---
    GCP_PROJECT: str = Field("", validation_alias="GCP_PROJECT_ID")
    GCP_REGION: str = Field("us-central1", validation_alias="GCP_REGION")
    GCS_BUCKET_NAME: str = Field("", validation_alias="GCS_BUCKET_NAME")
    GCS_PREFIX: str = Field("uploads/", validation_alias="GCS_PREFIX")
    gcp_service_account_path: str = Field("", validation_alias="GCP_SERVICE_ACCOUNT_PATH")
    vector_search_index_id: str = Field("", validation_alias="VECTOR_SEARCH_INDEX_ID")
    vector_search_index_endpoint_id: str = Field("", validation_alias="VECTOR_SEARCH_INDEX_ENDPOINT_ID")

    # --- Agent middleware limits ---
    agent_max_model_calls: int = Field(8, validation_alias="AGENT_MAX_MODEL_CALLS")
    agent_max_tool_calls: int = Field(6, validation_alias="AGENT_MAX_TOOL_CALLS")
    agent_max_retries: int = Field(2, validation_alias="AGENT_MAX_RETRIES")

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),  # found no matter which folder you start from
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _normalise(self) -> "Settings":
        # Vertex AI Vector Search expects the prefix to end with "/"
        if self.GCS_PREFIX and not self.GCS_PREFIX.endswith("/"):
            self.GCS_PREFIX += "/"
        return self

    # --- Convenience ---
    @property
    def is_cloud(self) -> bool:
        return self.app_mode == "cloud"

    @property
    def chroma_dir(self) -> str:
        return os.path.join(self.local_data_dir, "chroma")

    @property
    def local_upload_dir(self) -> str:
        return os.path.join(self.local_data_dir, "uploads")

    def missing_cloud_settings(self) -> list[str]:
        """Names of cloud settings that are still empty (used for clear error messages)."""
        required = {
            "GCP_PROJECT_ID": self.GCP_PROJECT,
            "GCS_BUCKET_NAME": self.GCS_BUCKET_NAME,
            "VECTOR_SEARCH_INDEX_ID": self.vector_search_index_id,
            "VECTOR_SEARCH_INDEX_ENDPOINT_ID": self.vector_search_index_endpoint_id,
        }
        return [name for name, value in required.items() if not value]


settings = Settings()


# Cloud mode on a laptop: point Google's client libraries at the service-account key.
# (On Cloud Run nothing is set here; the service's own identity is used.)
if settings.is_cloud and settings.gcp_service_account_path:
    key_path = Path(settings.gcp_service_account_path)
    if not key_path.is_absolute():
        key_path = PROJECT_ROOT / key_path
    if key_path.exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(key_path))
