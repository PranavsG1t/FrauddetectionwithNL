"""Day 4a - Build the fraud-pattern knowledge base and embed it with LangChain +
Gemini embeddings, indexed with FAISS.

This is the retrieval half of RAG: turn a handful of written fraud-pattern
descriptions into a searchable vector index so alert_generator.py can pull
the most relevant pattern for a given SHAP/CV verdict.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

INDEX_DIR = Path(__file__).resolve().parents[2] / "outputs" / "rag_index"

# Hand-written fraud-pattern descriptions. Keep these short and specific -
# they're what the LLM grounds its explanation in, so vague text here means
# vague (or hallucinated) alerts later. Covers both fraud vectors (CNP
# transactions + document tampering) since both feed the same alert layer.
FRAUD_PATTERNS = [
    {
        "id": "impossible_travel",
        "title": "Impossible travel",
        "text": (
            "Two transactions from the same card occur in geographically "
            "distant locations within a time window too short for physical "
            "travel between them (e.g. two cities 500+ km apart within an "
            "hour). Strong indicator of card cloning or credential theft."
        ),
    },
    {
        "id": "card_testing",
        "title": "Card testing",
        "text": (
            "A burst of small, low-value transactions in quick succession, "
            "often at different merchants, used by fraudsters to verify a "
            "stolen card number is still active before attempting a larger "
            "purchase."
        ),
    },
    {
        "id": "high_velocity",
        "title": "High transaction velocity",
        "text": (
            "An unusually high number of transactions on one card within a "
            "short time-since-last-transaction window, well above that "
            "cardholder's historical average gap between purchases."
        ),
    },
    {
        "id": "amount_outlier",
        "title": "Amount anomaly",
        "text": (
            "A transaction amount that is a large z-score outlier relative "
            "to the cardholder's historical mean and standard deviation of "
            "spend, especially combined with an unfamiliar merchant category."
        ),
    },
    {
        "id": "category_risk",
        "title": "High-risk merchant category",
        "text": (
            "The transaction falls in a merchant category with a "
            "historically elevated fraud rate (e.g. electronics, gift cards, "
            "cash-like categories), which raises baseline risk even before "
            "other signals are considered."
        ),
    },
    {
        "id": "odd_hour",
        "title": "Off-hours transaction",
        "text": (
            "The transaction occurs at an hour of day or day of week that is "
            "atypical for this cardholder's usual spending pattern, which can "
            "indicate the card is being used without the owner's knowledge."
        ),
    },
    {
        "id": "copy_move_tamper",
        "title": "Copy-move document tampering",
        "text": (
            "A region of a document image has been copy-pasted from another "
            "location in the same image, typically to alter a name, photo, "
            "or ID number. Detectable as a duplicated patch with a subtly "
            "different blur or compression signature at its boundary."
        ),
    },
    {
        "id": "splice_tamper",
        "title": "Spliced document tampering",
        "text": (
            "Content from a different source document has been inserted "
            "into this one, often visible as mismatched lighting, resolution, "
            "or font rendering between the spliced region and the rest of "
            "the document."
        ),
    },
    {
        "id": "genuine_variation",
        "title": "Genuine document, normal capture noise",
        "text": (
            "Blur, glare, or skew consistent with an ordinary phone-camera "
            "or scanner capture, not tampering. Genuine documents can look "
            "imperfect without being forged."
        ),
    },
]


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_index(save: bool = True) -> FAISS:
    """Embed FRAUD_PATTERNS and build a FAISS index over them."""
    docs = [
        Document(page_content=p["text"], metadata={"id": p["id"], "title": p["title"]})
        for p in FRAUD_PATTERNS
    ]
    embeddings = get_embeddings_model()
    index = FAISS.from_documents(docs, embeddings)

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index.save_local(str(INDEX_DIR))
        print(f"Saved FAISS index ({len(docs)} patterns) to {INDEX_DIR}")

    return index


if __name__ == "__main__":
    build_index()

