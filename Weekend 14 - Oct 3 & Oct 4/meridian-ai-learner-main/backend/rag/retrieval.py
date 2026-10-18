"""RAG question answering: retrieve relevant chunks, then let Gemini answer from them.

    question -> retriever (top matching chunks) -> prompt (chunks + question) -> Gemini -> answer

LangChain v1 style: the chain is built with LCEL (the `|` pipe operator) from
langchain-core pieces. The old `create_retrieval_chain` / `MultiQueryRetriever` /
`ContextualCompressionRetriever` helpers now live in the legacy `langchain-classic`
package, so the two advanced strategies are written out below in a few lines each,
which also makes them easier to read and explain.
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from logger import GLOBAL_LOGGER as log
from rag.llm import get_llm
from rag.vector_store import get_vector_store

SYSTEM_PROMPT = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer based on the context, say that you don't know. "
    "Keep the answer concise and accurate."
    "\n\n"
    "Context: {context}"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])

EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "Copy, word for word, only the sentences from the context that help answer the question. "
    "If nothing is relevant, reply exactly NO_OUTPUT.\n\n"
    "Question: {question}\n\nContext:\n{context}"
)


class AlternativeQuestions(BaseModel):
    """Different phrasings of the user's question, used to widen the search."""

    questions: list[str] = Field(description="3 alternative phrasings of the question")


# ---------------------------------------------------------------------------
# Retrieval strategies: each one is a function  question -> list[Document]
# ---------------------------------------------------------------------------

def _similarity(question: str, k: int = 3) -> list[Document]:
    """The k chunks closest in meaning to the question (fast, default)."""
    return get_vector_store().similarity_search(question, k=k)


def _multiquery(question: str) -> list[Document]:
    """Gemini rewrites the question 3 ways; we search with all 4 and merge the results."""
    rewriter = get_llm().with_structured_output(AlternativeQuestions)
    rewrites = rewriter.invoke(f"Write 3 alternative phrasings of this question: {question}")
    queries = [question, *rewrites.questions]
    log.info("MultiQuery rewrites", queries=queries)

    seen, merged = set(), []
    for query in queries:
        for doc in _similarity(query):
            if doc.page_content not in seen:  # drop duplicates found by several queries
                seen.add(doc.page_content)
                merged.append(doc)
    return merged


def _contextual(question: str) -> list[Document]:
    """Fetch 10 chunks, then Gemini trims each one to the relevant sentences (drops the rest)."""
    docs = _similarity(question, k=10)
    extractor = EXTRACT_PROMPT | get_llm() | StrOutputParser()
    extracts = extractor.batch([{"question": question, "context": d.page_content} for d in docs])
    return [
        Document(page_content=text.strip(), metadata=doc.metadata)
        for doc, text in zip(docs, extracts)
        if text.strip() and "NO_OUTPUT" not in text
    ]


STRATEGIES = {"similarity": _similarity, "multiquery": _multiquery, "contextual": _contextual}


def build_retriever(retriever_type: str = "similarity") -> RunnableLambda:
    if retriever_type not in STRATEGIES:
        log.warning("Unknown retriever_type, using similarity", retriever_type=retriever_type)
        retriever_type = "similarity"
    return RunnableLambda(STRATEGIES[retriever_type], name=f"{retriever_type}_retriever")


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs)


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

def build_rag_chain(retriever_type: str = "similarity"):
    """question (str) -> answer (str), written as one LCEL pipeline."""
    return (
        {"context": build_retriever(retriever_type) | format_docs, "input": RunnablePassthrough()}
        | ANSWER_PROMPT
        | get_llm()
        | StrOutputParser()
    )


def ask_question(query: str, retriever_type: str = "similarity") -> str:
    """Answer a question using the documents stored in the vector store."""
    log.info("RAG query received", query=query, retriever_type=retriever_type)
    answer = build_rag_chain(retriever_type).invoke(query)
    log.info("RAG answer generated", answer=answer)
    return answer
