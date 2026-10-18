"""Test setup: fake config so the tests run without Google Cloud, a Gemini key or the internet."""
import os
import sys
import tempfile

# Make `import api`, `import rag`, ... work when pytest is run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Environment variables win over .env, so a developer's real .env never leaks into tests
os.environ["APP_MODE"] = "local"
os.environ["LOCAL_DATA_DIR"] = tempfile.mkdtemp(prefix="meridian-test-")
os.environ["GOOGLE_API_KEY"] = "test-key"
os.environ["GCP_PROJECT_ID"] = "test-project"
os.environ["GCP_REGION"] = "us-central1"
# Empty on purpose: nothing may try to reach Cloud Storage
os.environ["GCS_BUCKET_NAME"] = ""
os.environ["GCS_PREFIX"] = "uploads/"
os.environ["VECTOR_SEARCH_INDEX_ID"] = ""
os.environ["VECTOR_SEARCH_INDEX_ENDPOINT_ID"] = ""
