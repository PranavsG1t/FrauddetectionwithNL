"""Day 4b - Tie retriever + prompt + Gemini into one LCEL chain that turns
SHAP output (transaction model) or a CV verdict (document model) into a
grounded natural-language alert.

Two input formats are handled here since the project has two fraud vectors
(Day 2's LightGBM+SHAP and Day 3's CV+Grad-CAM) - both route through the
same chain, which is the pattern Day 5 will formalize into a shared schema.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from ..schema.fraud_event import FraudEvent

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2, max_retries=10)
from .prompt_templates import ALERT_PROMPT
from .retriever import load_retriever

_FEATURE_PHRASES = {
    "amt": "unusually large transaction amount",
    "geo_distance_km": "merchant far from the cardholder's home location",
    "amt_zscore": "transaction amount far above the cardholder's usual spending",
    "category_fraud_rate": "transaction in a high-risk merchant category",
}


def build_transaction_query(shap_output: dict, top_k: int = 3) -> str:
    """Natural-language rendering of the top SHAP features, used only for
    retrieval - the raw facts string doesn't share enough vocabulary with
    the hand-written KB entries to retrieve well as-is."""
    top = sorted(shap_output.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    phrases = [_FEATURE_PHRASES[f] for f, v in top if f in _FEATURE_PHRASES and v > 0]
    if not phrases:
        return "suspicious card-not-present transaction"
    return "Suspicious card-not-present transaction: " + "; ".join(phrases)

def format_transaction_facts(shap_output: dict) -> str:
    """shap_output: {feature_name: shap_value, ...} - top contributing
    features for a single flagged transaction, from shap_explainer.py."""
    top_features = sorted(shap_output.items(), key=lambda kv: abs(kv[1]), reverse=True)
    lines = [f"- {name}: SHAP contribution {value:+.3f}" for name, value in top_features[:5]]
    return "Transaction flagged by LightGBM. Top contributing features:\n" + "\n".join(lines)


def format_document_facts(cv_verdict: dict) -> str:
    """cv_verdict: {"prediction": "tampered"|"genuine", "confidence": float,
    "gradcam_region": str (optional)} - from 03b/03c output."""
    facts = (
        f"Document classifier verdict: {cv_verdict['prediction']} "
        f"(confidence {cv_verdict['confidence']:.2f})."
    )
    if cv_verdict.get("gradcam_region"):
        facts += f" Grad-CAM highlighted region: {cv_verdict['gradcam_region']}."
    return facts


def _retrieve_context(query: str, retriever) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(f"[{d.metadata['title']}] {d.page_content}" for d in docs)


def build_alert_chain(retriever=None):
    retriever = retriever or load_retriever()
    chain = (
        RunnableParallel(
            context=RunnableLambda(lambda x: _retrieve_context(x["query"], retriever)),
            facts=RunnableLambda(lambda x: x["facts"]),
        )
        | ALERT_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def generate_transaction_alert(shap_output: dict, risk_score: float, chain=None) -> FraudEvent:
    chain = chain or build_alert_chain()
    facts = format_transaction_facts(shap_output)
    query = build_transaction_query(shap_output)
    explanation_text = chain.invoke({"facts": facts, "query": query})
    return FraudEvent.from_transaction(shap_output, risk_score=risk_score, explanation_text=explanation_text)
def generate_document_alert(cv_verdict: dict, chain=None) -> FraudEvent:
    chain = chain or build_alert_chain()
    facts = format_document_facts(cv_verdict)
    explanation_text = chain.invoke({"facts": facts, "query": facts})
    return FraudEvent.from_document(cv_verdict, explanation_text)


if __name__ == "__main__":
    chain = build_alert_chain()

    example_shap = {"amt_zscore": 3.4, "geo_velocity_kmh": 890.0, "category_risk": 0.71}
    print("--- Transaction alert ---")
    print(generate_transaction_alert(example_shap,risk_score= 0.87, chain=chain))

    example_cv = {"prediction": "tampered", "confidence": 0.93, "gradcam_region": "photo area, top-left"}
    print("\n--- Document alert ---")
    print(generate_document_alert(example_cv, chain))

