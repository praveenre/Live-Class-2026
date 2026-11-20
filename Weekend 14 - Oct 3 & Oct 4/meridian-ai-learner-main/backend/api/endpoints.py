"""REST endpoints. Each route is thin: validate input, call the real logic, return JSON.

    /api/health        - is the server up?
    /api/status        - configuration summary shown on the System Status tab
    /api/agent/audit   - run the multi-agent procurement audit        (agent/agents.py)
    /api/rag/ask       - ask a question about the ingested documents  (rag/retrieval.py)
    /api/rag/upload    - upload PDFs and index them                   (rag/data_ingestion.py)
    /api/rag/ingest    - index every PDF already in the document store (.meridian/uploads/)

Errors: our own exceptions (exceptions/) are turned into clear HTTP answers in api/main.py,
so the routes only catch what they can explain better themselves.
"""
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from agent.agents import ProcurementSupervisor
from api.schemas import AuditRequest, AuditResponse, QueryRequest, QueryResponse
from config.settings import settings
from exceptions import MeridianError
from logger import GLOBAL_LOGGER as log
from rag import document_store
from rag.data_ingestion import ingest_pdf, ingest_stored_pdfs
from rag.retrieval import ask_question

health_router = APIRouter(prefix="/api", tags=["Health"])
status_router = APIRouter(prefix="/api", tags=["Status"])
agent_router = APIRouter(prefix="/api/agent", tags=["Agent"])
rag_router = APIRouter(prefix="/api/rag", tags=["RAG"])

# In-memory upload tracker (reset on restart)
_upload_history: list[dict] = []
_server_start_time = datetime.now(timezone.utc)


@lru_cache(maxsize=1)
def get_supervisor() -> ProcurementSupervisor:
    """Build the agents on the first audit request, not when the server starts."""
    return ProcurementSupervisor()


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------

@health_router.get("/health")
def health():
    return {"status": "ok", "mode": settings.app_mode}


@status_router.get("/status")
def system_status():
    """Return system configuration and health information (makes no calls to Google)."""
    uptime_seconds = int((datetime.now(timezone.utc) - _server_start_time).total_seconds())
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return {
        "mode": settings.app_mode,
        "backend": {
            "status": "healthy",
            "uptime": f"{hours:02d}h {minutes:02d}m {seconds:02d}s",
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
        },
        "storage": {
            "location": document_store.location(),
        },
        "vector_search": {
            "backend": "Pinecone (Serverless)" if settings.is_pinecone else "Chroma (local disk)",
            "index_id": settings.pinecone_index_name if settings.is_pinecone else settings.chroma_dir,
            "endpoint_id": "Pinecone Cloud" if settings.is_pinecone else "-",
            "dimension": settings.embedding_dimensions,
        },
        "models": {
            "embedding": (
                settings.mesh_embedding_model if settings.mesh_api_key
                else settings.local_embedding_model_name
            ),
            "llm": settings.mesh_model if settings.mesh_api_key else settings.llm_model_name,
            "llm_framework": (
                f"Mesh API Gateway ({settings.mesh_model}) via LangChain v1" if settings.mesh_api_key
                else "Google Gemini API via LangChain v1"
            ),
        },
        "ingestion": {
            "uploads_this_session": len(_upload_history),
        },
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@agent_router.post("/audit", response_model=AuditResponse)
def run_audit(payload: AuditRequest):
    """Run the full multi-agent procurement audit and return all phase results."""
    try:
        return AuditResponse(**get_supervisor().run_audit(payload.request_text))
    except MeridianError:
        raise  # handled in api/main.py
    except Exception as e:
        log.exception("Audit failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

@rag_router.post("/ask", response_model=QueryResponse)
def rag_query(payload: QueryRequest):
    """Ask a question; the answer comes only from the indexed documents."""
    try:
        return QueryResponse(answer=ask_question(payload.query, payload.retriever_type))
    except MeridianError:
        raise
    except Exception as e:
        log.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Accept one or more PDFs, keep a copy in the document store, then chunk, embed and index them."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    tmp_dir = tempfile.mkdtemp()
    try:
        for upload in files:
            filename = os.path.basename(upload.filename or "")
            if not filename.lower().endswith(".pdf"):
                results.append({"filename": filename, "status": "skipped",
                                "reason": "Only PDF files are supported."})
                continue

            # Save to a temp file (the async UploadFile must be awaited to read it)
            local_path = os.path.join(tmp_dir, filename)
            with open(local_path, "wb") as f:
                f.write(await upload.read())

            # Keep the original PDF in document store. Indexing still works if this fails.
            try:
                await run_in_threadpool(document_store.save, local_path, filename)
            except Exception as store_err:
                log.warning("Could not store original file", filename=filename, error=str(store_err))

            # ingest_pdf is slow and blocking, so run it in a worker thread to keep the server responsive
            summary = await run_in_threadpool(ingest_pdf, local_path, f"upload://{filename}")

            if summary["chunks"] == 0:
                log.warning("No chunks extracted, skipping indexing", filename=filename)
                results.append({"filename": filename, "status": "skipped",
                                "reason": "No extractable text found in PDF (may be a scanned/image-based PDF)."})
                continue

            log.info("Successfully ingested chunks", filename=filename, **summary)
            results.append({"filename": filename, "status": "ingested", **summary})
    except MeridianError:
        raise
    except Exception as e:
        log.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _upload_history.extend(results)
    return {"results": results}


@rag_router.get("/uploads")
def list_uploads():
    """Return the history of uploaded documents (in-memory, resets on restart)."""
    return {"uploads": _upload_history}


@rag_router.post("/ingest")
@rag_router.post("/ingest-stored", include_in_schema=False)
def trigger_store_ingestion():
    """Index every PDF already in the document store (.meridian/uploads/)."""
    try:
        totals = ingest_stored_pdfs()
        return {"status": f"Ingestion complete ({settings.app_mode} mode)", **totals}
    except MeridianError:
        raise
    except Exception as e:
        log.exception("Stored-document ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))
