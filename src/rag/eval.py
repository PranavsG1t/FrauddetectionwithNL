"""Day 4c - Basic groundedness check for generated alerts.

Not a full RAGAS-style eval - just a fast, dependency-light tripwire: does
the generated alert actually reference the facts we gave it, or has the
LLM wandered off and invented something? Catches the most common and most
dangerous failure mode (hallucinated fraud reasons) without needing an
eval framework. Good enough to log a groundedness score per alert and flag
low-scoring ones for review.
"""
import re
from dataclasses import dataclass, field


@dataclass
class GroundednessResult:
    score: float                       # fraction of key terms from facts found in the alert
    matched_terms: list = field(default_factory=list)
    missing_terms: list = field(default_factory=list)
    flagged: bool = False              # True if score is below threshold - needs review


def _extract_key_terms(facts: str) -> list:
    """Pull out the terms that matter from the facts string: feature names,
    numbers, and the prediction/verdict word. Deliberately simple - this is
    a tripwire, not NLP."""
    numbers = re.findall(r"-?\d+\.\d+|\d+%?", facts)
    feature_names = re.findall(r"\b[a-z][a-z_]{3,}\b", facts.lower())
    verdicts = [w for w in ["tampered", "genuine", "fraud"] if w in facts.lower()]
    return list(set(numbers + feature_names + verdicts))


def check_groundedness(facts: str, alert_text: str, threshold: float = 0.3) -> GroundednessResult:
    key_terms = _extract_key_terms(facts)
    if not key_terms:
        return GroundednessResult(score=1.0)

    alert_lower = alert_text.lower()
    matched = [t for t in key_terms if t in alert_lower]
    missing = [t for t in key_terms if t not in alert_lower]
    score = len(matched) / len(key_terms)

    return GroundednessResult(
        score=score,
        matched_terms=matched,
        missing_terms=missing,
        flagged=score < threshold,
    )


if __name__ == "__main__":
    facts = (
        "Transaction flagged by LightGBM. Top contributing features:\n"
        "- amt_zscore: SHAP contribution +3.400\n"
        "- geo_velocity_kmh: SHAP contribution +890.000"
    )
    good_alert = (
        "This transaction was flagged due to an unusually high amount "
        "z-score and a geo velocity of nearly 890 km/h, consistent with "
        "impossible travel."
    )
    bad_alert = (
        "This transaction looks suspicious because the cardholder recently "
        "changed their phone number."
    )

    for label, alert in [("good", good_alert), ("bad", bad_alert)]:
        result = check_groundedness(facts, alert)
        print(f"{label}: score={result.score:.2f} flagged={result.flagged} missing={result.missing_terms}")
