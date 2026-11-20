"""How the project reads its configuration.

Run:    python 06_settings_demo.py
Then:   LLM_TEMPERATURE=0.7 python 06_settings_demo.py

Order of priority (highest first): environment variable -> .env file -> default in the code.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DemoSettings(BaseSettings):
    backend: str = Field("pinecone", validation_alias="VECTOR_BACKEND")
    model: str = Field("gemini-2.5-flash-lite", validation_alias="LLM_MODEL_NAME")
    temperature: float = Field(0.0, validation_alias="LLM_TEMPERATURE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = DemoSettings()
print("backend    :", settings.backend)
print("model      :", settings.model)
print("temperature:", settings.temperature, f"(type: {type(settings.temperature).__name__})")
print("\nNote that the text '0.0' from an environment variable became a real float.")
