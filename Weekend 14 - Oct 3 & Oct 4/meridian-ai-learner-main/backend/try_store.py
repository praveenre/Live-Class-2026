"""Store three sentences, search by meaning, then clean up (run from backend/):  uv run python try_store.py"""
from rag.vector_store import get_vector_store

store = get_vector_store()
store.add_texts(
    ["Servo motors are paid on Net-45 terms",
     "Filling line equipment is depreciated over 10 years",
     "Revenue in FY2024 was EUR 2.84 billion"],
    ids=["demo-1", "demo-2", "demo-3"],
)
for doc in store.similarity_search("When do we pay servo motor suppliers?", k=2):
    print(doc.page_content)
# store.delete(ids=["demo-1", "demo-2", "demo-3"])
print("stored now:", len(store.get()["ids"]))
