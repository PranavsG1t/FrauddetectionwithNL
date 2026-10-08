"""Day 4a - Load the saved FAISS index and expose it as a LangChain retriever."""
from .embeddings import INDEX_DIR, get_embeddings_model
from langchain_community.vectorstores import FAISS


def load_retriever(k: int = 2):
    """Load the persisted fraud-pattern index and return it as a retriever.

    k=2 keeps retrieval tight - we want the 1-2 most relevant patterns fed
    to the LLM, not a broad dump that dilutes grounding.
    """
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {INDEX_DIR}. Run embeddings.py first "
            "to build and save it."
        )
    embeddings = get_embeddings_model()
    index = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    return index.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    retriever = load_retriever()
    results = retriever.invoke("card used in two cities within an hour")
    for doc in results:
        print(f"- {doc.metadata['title']}: {doc.page_content[:80]}...")
