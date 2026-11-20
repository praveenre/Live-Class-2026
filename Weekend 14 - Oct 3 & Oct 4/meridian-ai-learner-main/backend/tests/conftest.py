"""Test setup: fake config so the tests run offline without external services."""
import os
import sys
import tempfile

# Make `import api`, `import rag`, ... work when pytest is run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Environment variables win over .env, so a developer's real .env never leaks into tests
os.environ["APP_MODE"] = "local"
os.environ["LOCAL_DATA_DIR"] = tempfile.mkdtemp(prefix="meridian-test-")
os.environ["GOOGLE_API_KEY"] = "test-key"
os.environ["VECTOR_BACKEND"] = "chroma"
os.environ["PINECONE_API_KEY"] = ""
os.environ["PINECONE_INDEX_NAME"] = ""
os.environ["MESH_API_KEY"] = ""
