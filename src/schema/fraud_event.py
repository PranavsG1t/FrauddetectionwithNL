"""Day 5 - Shared output schema for both fraud vectors.

Both the transaction pipeline (LightGBM+SHAP) and the document pipeline
(ResNet+GradCAM) produce a FraudEvent with the same six fields, so
everything downstream (API, dashboard) never needs to branch on which
model produced the alert.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


EventType = Literal["transaction_fraud", "document_forgery"]


@dataclass #decorators used
class FraudEvent:
    event_type: EventType
    risk_score: float                      # 0.0-1.0
    source_features: dict                  # SHAP dict or cv_verdict dict, as-is
    explanation_text: str                  # alert_generator.py's output
    recommended_action: str                # "block" | "manual_review" | "monitor"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_transaction(cls, shap_output: dict, risk_score: float, explanation_text: str) -> "FraudEvent":
        return cls(
            event_type="transaction_fraud",
            risk_score=risk_score,
            source_features=shap_output,
            explanation_text=explanation_text,
            recommended_action=_recommend_action(risk_score),
        )

    @classmethod
    def from_document(cls, cv_verdict: dict, explanation_text: str) -> "FraudEvent":
        return cls(
            event_type="document_forgery",
            risk_score=cv_verdict["confidence"],
            source_features=cv_verdict,
            explanation_text=explanation_text,
            recommended_action=_recommend_action(cv_verdict["confidence"]),
        )


def _recommend_action(risk_score: float) -> str:
    """Placeholder thresholds - tune these once you have real score
    distributions from Day 2's eval metrics."""
    if risk_score >= 0.989:
        return "block"
    if risk_score >= 0.747:
        return "manual_review"
    return "monitor"