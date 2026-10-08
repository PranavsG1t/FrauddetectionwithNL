"""Day 4a - Prompt template for grounded NL alert generation.

Key design choice: separate RETRIEVED CONTEXT (verified fraud-pattern text)
from CASE FACTS (the specific SHAP/CV output for this event). This is what
turns "explain this fraud case" (open-ended, hallucination-prone) into a
structured prompt the LLM can't wander away from - Day 4's theory note.
"""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are a fraud-analyst assistant that writes short, plain-English "
    "alerts for a fraud-review dashboard. You must base your explanation "
    "ONLY on the retrieved fraud-pattern context and the case facts given "
    "to you. Do not invent details, numbers, or patterns that are not "
    "present in the context or facts. If the facts are ambiguous or don't "
    "clearly match a known pattern, say so plainly instead of guessing. "
    "Keep the alert to 2-3 sentences."
)

HUMAN_TEMPLATE = """Retrieved fraud-pattern context:
{context}

Case facts (from the model, not to be altered or exaggerated):
{facts}

Write the alert."""

ALERT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ]
)
