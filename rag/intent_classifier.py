"""
CARA - Greenfield Medical Centre AI Receptionist
intent_classifier.py - Classifies patient query into one of 5 intents
using Claude Haiku for fast cheap classification
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# VALID INTENTS
# ─────────────────────────────────────────
VALID_INTENTS = {
    "appointments",
    "triage",
    "prescription",
    "hours",
    "general"
}

# ─────────────────────────────────────────
# INIT ANTHROPIC CLIENT
# ─────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────
def classify_intent(combined_query: str) -> str:
    """
    Classifies patient query into one of 5 intents.

    Args:
        combined_query: rolling summary + last 6 turns + current message

    Returns:
        One of: appointments / triage / prescription / hours / general
    """

    system_prompt = """You are an intent classifier for a GP surgery AI receptionist called CARA.

Classify the patient query into EXACTLY ONE of these categories:

appointments - patient wants to book, cancel, reschedule an appointment, 
               ask about doctor availability, ask which doctor to see,
               ask about specific dates or times for appointments,
               ask about doctor languages spoken, ask for doctor who speaks a specific language,
               ask about doctor specialisms or which doctor is best for their condition

triage - patient describes symptoms, pain, illness, medical condition,
         asks what to do about a health problem, mentions feeling unwell,
         asks about urgency of their condition, mentions emergency symptoms

prescription - patient asks about medication, repeat prescription, 
               collecting prescription, prescription charges,
               running out of medication, new medication request

hours - patient asks about opening times, when surgery is open or closed,
        what to do when surgery is closed, out of hours advice,
        asks about registration, sick notes, referrals, test results,
        home visits, complaints, interpreters, translator, language support,
        any general admin query

general - anything that does not clearly fit the above categories

CRITICAL RULES:
- If patient mentions ANY symptom or health problem alongside appointment request,
  classify as TRIAGE not appointments. Safety first.
- If patient mentions chest pain, breathing difficulty, stroke, bleeding,
  unconscious, overdose — ALWAYS classify as triage regardless of other words.
- Reply with ONE WORD ONLY from the list above.
- No explanation. No punctuation. Just the single category word."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify this patient query:\n\n{combined_query}"
                }
            ]
        )

        intent = response.content[0].text.strip().lower()

        # Validate — if unexpected response fallback to general
        if intent not in VALID_INTENTS:
            print(f"  [IntentClassifier] Unexpected response: '{intent}' — falling back to 'general'")
            return "general"

        return intent

    except Exception as e:
        print(f"  [IntentClassifier] Error: {e} — falling back to 'general'")
        return "general"


# ─────────────────────────────────────────
# TEST — run directly to verify
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "I need to book an appointment with Dr Patel tomorrow morning",
        "I have been having chest pains since this morning",
        "I need my repeat prescription for blood pressure medication",
        "What time does the surgery close on Saturday",
        "I need to register as a new patient",
        "I have angina, can I see a doctor today",
        "My breathing is very bad",
        "I want to cancel my appointment next Tuesday",
        "How long does a repeat prescription take",
        "I am not feeling well, can I come in today",
    ]

    print("Testing IntentClassifier...\n")
    for query in test_queries:
        intent = classify_intent(query)
        print(f"Query : {query}")
        print(f"Intent: {intent}")
        print()