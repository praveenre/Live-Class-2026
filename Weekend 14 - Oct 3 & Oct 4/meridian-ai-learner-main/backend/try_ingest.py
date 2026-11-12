"""Index one PDF twice and count what is stored (run from backend/):  uv run python try_ingest.py"""
from rag.data_ingestion import ingest_pdf
from rag.vector_store import get_vector_store

PDF = "../sample_docs/aldermoor_procurement_policy.pdf"
SOURCE = "local://aldermoor_procurement_policy.pdf"

print("first run :", ingest_pdf(PDF, SOURCE))
print("stored    :", len(get_vector_store().get()["ids"]))



print("second run:", ingest_pdf(PDF, SOURCE))
print("stored    :", len(get_vector_store().get()["ids"]))
