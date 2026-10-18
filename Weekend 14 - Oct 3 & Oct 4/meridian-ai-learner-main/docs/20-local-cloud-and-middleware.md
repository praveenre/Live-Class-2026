# 20. Local vs cloud mode, and agent middleware (LangChain v1)

## The toggle: `APP_MODE`

One line in `.env` decides where the app keeps its data:

```
APP_MODE=local   # laptop: Gemini API key only, Chroma on disk
APP_MODE=cloud   # Google Cloud: Vertex AI Vector Search + Cloud Storage
```

| Concern | File | local | cloud |
|---|---|---|---|
| Read the toggle | `backend/config/settings.py` | `settings.is_cloud == False` | `True` |
| Embeddings | `backend/rag/embeddings.py` | `gemini-embedding-001` (API key) | `text-embedding-005` (Vertex AI) |
| Vector database | `backend/rag/vector_store.py` | Chroma in `.meridian/chroma` | Vertex AI Vector Search |
| Original PDFs | `backend/rag/document_store.py` | `.meridian/uploads/` | `gs://<bucket>/<prefix>` |
| Log upload | `backend/logger/custom_logger.py` | skipped | copied to the bucket on shutdown |

**Why this design?** Every place that needs storage asks *one* question (`settings.is_cloud`)
and gets back a LangChain `VectorStore`. Ingestion and retrieval call `add_documents` and
`similarity_search` and never know which database is behind them. That is the
*strategy pattern*: same interface, swappable implementation.

**Both modes use 768-dimensional vectors**, but the two databases are separate. Documents
indexed locally are not in the cloud index; upload them again after switching.

**Errors are explicit.** If `APP_MODE=cloud` but `VECTOR_SEARCH_INDEX_ID` is empty, the API
answers `503` with the list of missing settings (see `backend/exceptions/`), instead of a
stack trace.

## LangChain v1 changes in this project

| Before | Now | Why |
|---|---|---|
| `create_retrieval_chain`, `MultiQueryRetriever`, `ContextualCompressionRetriever` from `langchain-classic` | A plain LCEL pipeline in `rag/retrieval.py`: `{context, input} \| prompt \| llm \| StrOutputParser()` | v1 moved these helpers to the legacy package; LCEL is the v1 way and is easier to read |
| `VertexAIEmbeddings` | `GoogleGenerativeAIEmbeddings` (with `vertexai=True` in cloud) | `VertexAIEmbeddings` is deprecated and will be removed |
| `create_agent(model, tools, system_prompt)` | `create_agent(..., middleware=build_agent_middleware())` | v1's main new feature for agents |

## Middleware (`backend/agent/middleware.py`)

Middleware wraps the agent loop without changing prompts or tools. The order is the order
in the list:

| # | Middleware | What it does here |
|---|---|---|
| 1 | `PIIMiddleware("email")`, `PIIMiddleware("credit_card")` | Redacts e-mails and masks card numbers in the purchase request before Gemini sees them |
| 2 | `ModelCallLimitMiddleware` | Ends a phase after `AGENT_MAX_MODEL_CALLS` model calls (cost guard against loops) |
| 3 | `ToolCallLimitMiddleware` | Caps tool calls per phase at `AGENT_MAX_TOOL_CALLS` |
| 4 | `ModelRetryMiddleware` | Retries Gemini on transient errors (429/503) with exponential backoff |
| 5 | `ToolRetryMiddleware` | Retries flaky tools such as the live FX-rate API |
| 6 | `log_tool_calls` (custom) | One JSON log line per tool call: name, arguments, duration, result preview |

The custom one is ten lines with the `@wrap_tool_call` decorator. It receives the tool request
and a `handler`; it calls `handler(request)` to run the tool, and can do anything before
and after. Writing your own middleware is the same shape.

## Exercises

1. Set `AGENT_MAX_TOOL_CALLS=1` and run an audit. Which report changes, and why?
2. Add `PIIMiddleware("url", strategy="redact")` and paste a request containing a link.
3. Write a `@before_model` middleware that logs how many messages are in the conversation.
4. Switch to `APP_MODE=cloud` with an empty index ID and call `/api/rag/ask`. Read the 503 message.
