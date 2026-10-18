"""Getting documents INTO the vector database.

    PDF file -> pages -> chunks (~1000 characters) -> embeddings -> vector store

Two entry points share the same steps:
  * ingest_pdf()          - one PDF on disk (used by the upload endpoint)
  * ingest_stored_pdfs()  - every PDF in the document store
                            (local: .meridian/uploads/, cloud: the GCS bucket)
"""
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from exceptions import IngestionError
from logger import GLOBAL_LOGGER as log
from rag import document_store
from rag.vector_store import get_vector_store

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100  # neighbouring chunks share 100 characters so sentences are not cut off


def split_into_chunks(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(pages)


def _chunk_id(chunk: Document, position: int) -> str:
    """Stable ID per chunk: re-indexing the same file overwrites instead of duplicating."""
    key = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page')}|{position}"
    return hashlib.sha1(key.encode()).hexdigest()


def ingest_pdf(local_path: str, source: str) -> dict:
    """Load one PDF, chunk it and store it. Returns {"pages": n, "chunks": n}.

    `source` is saved in each chunk's metadata so answers can be traced back to a file.
    """
    try:
        pages = PyPDFLoader(local_path).load()
    except Exception as exc:  # pypdf raises many different error types
        raise IngestionError(f"Could not read PDF '{source}': {exc}") from exc

    for page in pages:
        page.metadata["source"] = source

    chunks = split_into_chunks(pages)
    if chunks:
        log.info("Embedding chunks and adding to vector store", source=source, chunks=len(chunks))
        ids = [_chunk_id(c, i) for i, c in enumerate(chunks)]
        get_vector_store().add_documents(chunks, ids=ids)
    return {"pages": len(pages), "chunks": len(chunks)}


def ingest_stored_pdfs() -> dict:
    """Index every PDF already in the document store."""
    log.info("Indexing stored documents", location=document_store.location())
    totals = {"files": 0, "pages": 0, "chunks": 0}

    with document_store.pdfs() as files:
        for local_path, source in files:
            result = ingest_pdf(local_path, source=source)
            log.info("Ingested stored PDF", source=source, **result)
            totals["files"] += 1
            totals["pages"] += result["pages"]
            totals["chunks"] += result["chunks"]

    if totals["files"] == 0:
        log.warning("No PDF documents found in the document store", location=document_store.location())
    log.info("Ingestion complete", **totals)
    return totals
