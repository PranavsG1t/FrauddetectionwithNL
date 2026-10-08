"""Turns (shap_output, risk_score) into a FraudEvent with a Gemini-written
explanation. Wraps src/rag/alert_generator.py without modifying it.

IMPORTANT: src.rag.alert_generator creates the Gemini client at *import*
time. That import therefore lives inside build(), never at module top, so
it can only happen after the LightGBM/SHAP models are loaded (the load-order
rule from the blueprint's Day 6 notes).
"""
import logging
from typing import Callable

from src.schema.fraud_event import FraudEvent

logger = logging.getLogger(__name__)

GenerateFn = Callable[[dict, float], FraudEvent]


def fallback_explanation(shap_output: dict, top_k: int = 3) -> str:
    """Used when Gemini is down/slow: still tells the analyst *why*, straight
    from SHAP, with no LLM involved."""
    top = sorted(shap_output.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    factors = ", ".join(f"{name} ({value:+.2f})" for name, value in top)
    return f"Automated explanation unavailable. Top contributing factors: {factors}."


class AlertService:
    def __init__(self, generate: GenerateFn):
        # `generate` is injected so tests can pass a fake - no network needed.
        self._generate = generate

    @classmethod
    def build(cls) -> "AlertService":
        from src.rag.alert_generator import build_alert_chain, generate_transaction_alert

        chain = build_alert_chain()  # loads retriever + builds the LCEL chain once
        return cls(lambda shap, risk: generate_transaction_alert(shap, risk_score=risk, chain=chain))

    def explain_transaction(self, shap_output: dict, risk_score: float) -> FraudEvent:
        try:
            return self._generate(shap_output, risk_score)
        except Exception:
            # Graceful degradation: the score + SHAP are already computed and
            # valid, so a Gemini failure must not turn into a 500.
            logger.exception("Gemini alert generation failed; using SHAP fallback text")
            return FraudEvent.from_transaction(
                shap_output=shap_output,
                risk_score=risk_score,
                explanation_text=fallback_explanation(shap_output),
            )