"""Ask one question with each search mode (run from backend/):  uv run python try_modes.py"""
import time

from rag.retrieval import ask_question, build_retriever

QUESTION = "What payment terms do servo motor suppliers get?"

for mode in ("similarity", "multiquery", "contextual"):
    chunks = build_retriever(mode).invoke(QUESTION)
    print(f"\n=== {mode}: {len(chunks)} chunk(s) go into the prompt")
    for chunk in chunks:
        print(f"  {chunk.metadata['source']}  ({len(chunk.page_content)} characters)")

    started = time.time()
    answer = ask_question(QUESTION, mode)
    print(f"  one full answer took {time.time() - started:.1f} s")
    print("  " + answer.replace("\n", "\n  "))
