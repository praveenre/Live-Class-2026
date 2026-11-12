"""Keep a copy of a PDF and list what is stored (run from backend/):  uv run python try_documents.py"""
from rag import document_store

print("location:", document_store.location())
print("saved at:", document_store.save("../sample_docs/aldermoor_procurement_policy.pdf", "aldermoor_procurement_policy.pdf"))
with document_store.pdfs() as files:
    for local_path, source in files:
        print("stored:", source)
