"""
CARA - Greenfield Medical Centre AI Receptionist
main.py - FastAPI server orchestrating the entire pipeline

Architecture:
- Slot filling for booking (no rigid state machine during collection)
- Strict YES/NO gate only at PENDING_CONFIRMATION
- Strict state machine only for cancellation (security flow)

Run with:
    python main.py
Then open: http://localhost:8000
"""

import os
import re
import uuid
import json
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime, timedelta

# RAG pipeline imports
from rag.intent_classifier import classify_intent
from rag.retriever import get_retriever
from rag.relevance_checker import check_relevance
from rag.conversation_manager import get_session, clear_session

# Booking imports
from bookings.booking_manager import (
    init_db, BookingState,
    get_pending_booking, upsert_pending_booking, clear_pending_booking,
    confirm_booking, verify_cancellation, cancel_booking,
    get_booked_slots, filter_booked_slots_from_chunk
)

load_dotenv()

# ─────────────────────────────────────────
# INIT
# ─────────────────────────────────────────
app = FastAPI(title="CARA - Greenfield Medical Centre AI Receptionist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("Initialising CARA pipeline...")
retriever = get_retriever()
init_db()
print("CARA pipeline ready.\n")


# ─────────────────────────────────────────
# REQUIRED BOOKING SLOTS
# ─────────────────────────────────────────
REQUIRED_SLOTS = ["clinician", "day", "date", "slot", "patient_name", "patient_dob"]


def is_booking_complete(pending: dict) -> bool:
    return all(pending.get(f) for f in REQUIRED_SLOTS)


# ─────────────────────────────────────────
# YES / NO DETECTION
# ─────────────────────────────────────────
YES_PHRASES = {
    "yes", "yeah", "yep", "yup", "yes please", "confirm", "confirmed",
    "go ahead", "that's correct", "that's right", "correct", "right",
    "sounds good", "please do", "please confirm", "ok", "okay",
    "fine", "sure", "absolutely", "definitely", "proceed", "book it",
    "book that", "yes confirm", "i confirm", "please book"
}

NO_PHRASES = {
    "no", "nope", "no thank you", "no thanks", "cancel", "don't book",
    "do not book", "stop", "actually no", "changed my mind", "never mind",
    "forget it", "don't confirm", "do not confirm", "incorrect", "wrong"
}

def detect_yes_no(message: str) -> str:
    msg = message.lower().strip()
    msg = re.sub(r'[^\w\s]', '', msg)
    for phrase in YES_PHRASES:
        if phrase in msg:
            return "yes"
    for phrase in NO_PHRASES:
        if phrase in msg:
            return "no"
    return "unclear"


# ─────────────────────────────────────────
# DATE RESOLUTION — PYTHON ONLY
# ─────────────────────────────────────────
def resolve_date_intent(date_intent: str) -> tuple:
    """
    Resolves date expressions to actual date and day.
    Python calculates — never LLM. Zero arithmetic errors.
    Returns (date_str, day_str) e.g. ("15 June 2026", "Monday")
    """
    if not date_intent:
        return "", ""

    intent = date_intent.lower().strip()
    today = datetime.now()

    def next_weekday(weekday_num):
        days_ahead = weekday_num - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)

    def fmt(dt):
        day_num = dt.day
        return f"{day_num} {dt.strftime('%B %Y')}", dt.strftime("%A")

    if "today" in intent:
        return fmt(today)
    if "tomorrow" in intent:
        return fmt(today + timedelta(days=1))
    if "day after tomorrow" in intent:
        return fmt(today + timedelta(days=2))

    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }

    for day_name, day_num in weekday_map.items():
        if day_name in intent:
            return fmt(next_weekday(day_num))

    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }

    for month_name, month_num in month_map.items():
        if month_name in intent:
            day_match = re.search(r'(\d{1,2})', intent)
            if day_match:
                d = int(day_match.group(1))
                year_match = re.search(r'(\d{4})', intent)
                year = int(year_match.group(1)) if year_match else today.year
                try:
                    dt = datetime(year, month_num, d)
                    if dt.date() < today.date():
                        dt = datetime(year + 1, month_num, d)
                    return fmt(dt)
                except ValueError:
                    pass

    return "", ""


# ─────────────────────────────────────────
# SLOT FILLING — SINGLE HAIKU CALL
# ─────────────────────────────────────────
def extract_slots(patient_message: str, claude_response: str, existing: dict) -> dict:
    """
    Extracts booking fields using single Claude Haiku call.
    Haiku extracts date INTENT as raw text.
    Python resolves intent to actual date — no LLM arithmetic errors.
    """

    today_str = datetime.now().strftime("%A %d %B %Y")

    existing_summary = []
    for k, v in existing.items():
        if v and k not in ["state", "idempotency_key", "cancel_ref", "cancel_attempts", "created_at", "updated_at", "session_id"]:
            existing_summary.append(f"{k}: {v}")

    existing_text = ", ".join(existing_summary) if existing_summary else "nothing collected yet"

    prompt = f"""You are extracting booking details from a GP surgery patient conversation.

TODAY: {today_str}

Our clinicians (use EXACT names from this list only):
- Dr. Margaret Collins (specialises in diabetes, elderly care)
- Dr. James Whitfield (specialises in mental health, skin conditions)
- Dr. Priya Patel (specialises in womens health, only female GP)
- Dr. Oliver Bennett (specialises in children, respiratory)
- Nurse Sarah Williams (blood pressure, vaccinations, blood tests, dressings)

Already collected so far: {existing_text}

Patient just said: "{patient_message}"

RULES:
1. Extract ONLY from patient message
2. clinician: match to exact name from list. "dr oliver"=Dr. Oliver Bennett. "whitfield"=Dr. James Whitfield. "the lady"=Dr. Priya Patel. "nurse"=Nurse Sarah Williams
3. date_intent: extract RAW date expression patient used. "tomorrow", "next Monday", "15th June", "15th June 2026". Do NOT calculate dates.
4. patient_dob: date of birth — always a past date before today
5. slot: time patient wants. "around 10"=10:00am. "half past nine"=9:30am. "morning" alone is NOT a slot
6. patient_name: full name of patient
7. reason: reason for appointment — only if explicitly mentioned
8. If field not mentioned → empty string ""

Return ONLY JSON:
{{"clinician": "", "date_intent": "", "slot": "", "patient_name": "", "patient_dob": "", "reason": ""}}

Pure JSON only, no explanation."""

    try:
        resp = claude_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        text = re.sub(r'```json|```', '', text).strip()
        extracted = json.loads(text)

        fields = {}
        for k, v in extracted.items():
            if not v or not str(v).strip() or str(v).strip() == '""':
                continue
            v = str(v).strip()

            if k == "date_intent":
                # Python resolves — never LLM arithmetic
                date_str, day_str = resolve_date_intent(v)
                if date_str:
                    fields["date"] = date_str
                    fields["day"] = day_str
                    print(f"  [SlotFiller] Resolved '{v}' → {date_str} ({day_str})")
            else:
                fields[k] = v

        print(f"  [SlotFiller] Haiku extracted: {fields}")
        return fields

    except Exception as e:
        print(f"  [SlotFiller] Haiku error: {e}")
        return {}



# ─────────────────────────────────────────
# BUILD CONFIRMATION READBACK
# ─────────────────────────────────────────
def build_confirmation_readback(pending: dict) -> str:
    """
    Builds the YES/NO confirmation message directly from SQLite data.
    Never relies on Claude — zero hallucination risk.
    """
    return (
        f"Just to confirm your booking:\n"
        f"- Doctor: {pending.get('clinician')}\n"
        f"- Date: {pending.get('date')} ({pending.get('day')})\n"
        f"- Time: {pending.get('slot')}\n"
        f"- Name: {pending.get('patient_name')}\n"
        f"- Date of birth: {pending.get('patient_dob')}\n"
        f"- Reason: {pending.get('reason') or 'not provided'}\n\n"
        f"Please say YES to confirm this booking or NO to cancel.\n"
        f"Please keep your date of birth safe — you will need it if you wish to cancel."
    )


# ─────────────────────────────────────────
# FILTER BOOKED SLOTS FROM CHUNKS
# ─────────────────────────────────────────
def filter_appointment_chunks(chunks: list[dict]) -> list[dict]:
    filtered = []
    for chunk in chunks:
        if chunk.get("collection") == "appointments":
            text = chunk["text"]
            clinician_match = re.search(
                r'(Dr\. [A-Za-z]+ [A-Za-z]+|Nurse Sarah Williams)', text
            )
            day_match = re.search(
                r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)', text
            )
            if clinician_match and day_match:
                clinician = clinician_match.group(0)
                day = day_match.group(0)
                filtered_text = filter_booked_slots_from_chunk(text, clinician, day)
                chunk = dict(chunk)
                chunk["text"] = filtered_text
        filtered.append(chunk)
    return filtered


# ─────────────────────────────────────────
# DETECT CANCELLATION INTENT
# ─────────────────────────────────────────
def is_cancellation_intent(message: str) -> bool:
    keywords = ["cancel", "cancellation", "cancel my appointment", "cancel appointment"]
    return any(kw in message.lower() for kw in keywords)


# ─────────────────────────────────────────
# CARA SYSTEM PROMPT
# ─────────────────────────────────────────
CARA_BASE_PROMPT = """You are CARA, the AI receptionist for Greenfield Medical Centre, an NHS GP surgery in London.

YOUR ROLE:
- Answer patient calls warmly and professionally
- Help patients book, cancel, or reschedule appointments
- Provide accurate information about the surgery using ONLY the context provided
- Triage patient symptoms and direct them to appropriate care
- Handle prescription, registration, and admin queries

YOUR PERSONALITY:
- Warm, calm, clear and patient
- Speak in plain English — no medical jargon
- Concise responses — this is a phone call not a letter
- Never rush a patient

CLINICAL SAFETY RULES — ALWAYS FOLLOW:
- Chest pain, difficulty breathing, stroke symptoms, severe bleeding, loss of consciousness, overdose → advise 999 immediately
- Angina, heart attack symptoms, anaphylaxis → advise 999 immediately
- Self harm or suicidal thoughts → same day urgent appointment + Samaritans 116 123
- Never diagnose. Never recommend medications.

APPOINTMENT HANDLING:
- Use the weekly schedule in context to give accurate slot information
- Slots shown are already filtered — only available slots listed. Do not offer others.
- Check leave notices — absent doctors are unavailable that day
- When a doctor is on leave offer other available doctors with their slots
- All 4 doctors are GPs who handle any general condition
- Dr Margaret Collins — diabetes, elderly care, hypertension
- Dr James Whitfield — mental health, skin conditions
- Dr Priya Patel — womens health, family planning, cervical screening. Only female GP
- Dr Oliver Bennett — children, respiratory, asthma
- Nurse Sarah Williams — blood pressure, vaccinations, blood tests, dressings, smear tests, ECG, ear syringing
- If patient asks for specialist not at surgery → explain we are a GP surgery, any GP can assess and refer. List all 4 GPs by name.

BOOKING FLOW:
- Collect naturally: clinician, date, time slot, patient full name, date of birth, reason (optional)
- Ask only for what is missing — patient may provide details in any order
- Once all details collected the system will handle confirmation automatically
- Do NOT say "booking confirmed" or give reference numbers — the system handles this

RESPONSE FORMAT:
- Introduce yourself only at the very start
- Keep responses under 100 words
- Never repeat confirmed information unnecessarily
- This is a VOICE call — do NOT use markdown formatting, asterisks, bullet points, or bold text
- Speak naturally as if on a phone call
- Use plain conversational English only

SURGERY DETAILS:
- Name: Greenfield Medical Centre, London
- Hours: Monday-Friday 8am-6:30pm, Saturday 8am-12pm, Sunday closed
- Emergency: 999 | Out of hours: 111"""


# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    chunks: list[str]
    top_score: float
    threshold_passed: bool
    booking_state: str
    booking_reference: str
    pending_booking: dict


# ─────────────────────────────────────────
# MAIN CHAT ENDPOINT
# ─────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    patient_message = request.message.strip()

    if not patient_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    print(f"\n{'='*60}")
    print(f"Session : {session_id}")
    print(f"Patient : {patient_message}")

    session = get_session(session_id)
    pending = get_pending_booking(session_id)
    current_state = pending.get("state") if pending else BookingState.IDLE
    booking_reference = ""
    intent = "general"
    chunk_texts = []
    top_score = 0.0
    relevant = False

    print(f"Booking state: {current_state}")

    # ── GREETING ────────────────────────────────────────────────
    greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hiya"}
    if patient_message.lower().strip() in greetings and len(session.history) == 0:
        clear_pending_booking(session_id)  # clear any stale pending booking from previous call
        reply = "Good morning, Greenfield Medical Centre, CARA speaking. How can I help you today?"
        session.add_turn("user", patient_message)
        session.add_turn("assistant", reply)
        return ChatResponse(
            reply=reply, session_id=session_id, intent="general",
            chunks=[], top_score=1.0, threshold_passed=True,
            booking_state=BookingState.IDLE, booking_reference="",
            pending_booking={}
        )

    # ── PENDING CONFIRMATION — YES/NO GATE ──────────────────────
    if current_state == BookingState.PENDING_CONFIRMATION:
        yes_no = detect_yes_no(patient_message)

        if yes_no == "yes":
            ref = confirm_booking(session_id)
            if ref:
                booking_reference = ref
                reply = (
                    f"Your appointment is confirmed!\n"
                    f"- Doctor: {pending.get('clinician')}\n"
                    f"- Date: {pending.get('date')} ({pending.get('day')})\n"
                    f"- Time: {pending.get('slot')}\n"
                    f"- Name: {pending.get('patient_name')}\n"
                    f"- Reference number: {ref}\n\n"
                    f"Please keep your reference number safe — you will need it to cancel or reschedule. "
                    f"Is there anything else I can help you with?"
                )
            else:
                reply = "I'm sorry, there was an issue confirming your booking. Please try again or call reception directly."
            session.add_turn("user", patient_message)
            session.add_turn("assistant", reply)
            return ChatResponse(
                reply=reply, session_id=session_id, intent="appointments",
                chunks=[], top_score=1.0, threshold_passed=True,
                booking_state=BookingState.CONFIRMED if ref else BookingState.IDLE,
                booking_reference=booking_reference, pending_booking={}
            )

        elif yes_no == "no":
            clear_pending_booking(session_id)
            reply = "No problem — I have cancelled the booking process. Is there anything else I can help you with?"
            session.add_turn("user", patient_message)
            session.add_turn("assistant", reply)
            return ChatResponse(
                reply=reply, session_id=session_id, intent="appointments",
                chunks=[], top_score=1.0, threshold_passed=True,
                booking_state=BookingState.IDLE, booking_reference="",
                pending_booking={}
            )

        else:
            # Unclear — re-ask
            reply = build_confirmation_readback(pending)
            session.add_turn("user", patient_message)
            session.add_turn("assistant", reply)
            return ChatResponse(
                reply=reply, session_id=session_id, intent="appointments",
                chunks=[], top_score=1.0, threshold_passed=True,
                booking_state=BookingState.PENDING_CONFIRMATION,
                booking_reference="", pending_booking=dict(pending)
            )

    # ── CANCELLATION FLOW ────────────────────────────────────────
    if current_state == BookingState.CANCELLATION_REFERENCE:
        ref_match = re.search(r'GMC-\d{4}-\d{4}', patient_message.upper())
        if ref_match:
            upsert_pending_booking(session_id, {
                "state": BookingState.CANCELLATION_VERIFICATION,
                "cancel_ref": ref_match.group(0)
            })
            reply = "Thank you. For security, could I verify your date of birth please?"
        else:
            reply = "Could I have your reference number please? It should be in the format GMC-2026-0001."
        session.add_turn("user", patient_message)
        session.add_turn("assistant", reply)
        pending = get_pending_booking(session_id)
        return ChatResponse(
            reply=reply, session_id=session_id, intent="appointments",
            chunks=[], top_score=1.0, threshold_passed=True,
            booking_state=pending.get("state"), booking_reference="",
            pending_booking=dict(pending)
        )

    if current_state == BookingState.CANCELLATION_VERIFICATION:
        cancel_ref = pending.get("cancel_ref", "")
        booking = verify_cancellation(cancel_ref, patient_message.strip())
        if booking:
            upsert_pending_booking(session_id, {"state": BookingState.CANCELLATION_PENDING})
            reply = (
                f"Identity verified. I found your appointment:\n"
                f"- Doctor: {booking['clinician']}\n"
                f"- Date: {booking['date']} ({booking['day']})\n"
                f"- Time: {booking['slot']}\n\n"
                f"Please say YES to confirm cancellation or NO to keep this appointment."
            )
        else:
            attempts = (pending.get("cancel_attempts") or 0) + 1
            upsert_pending_booking(session_id, {"cancel_attempts": attempts})
            if attempts >= 3:
                clear_pending_booking(session_id)
                reply = "I cannot verify your identity after 3 attempts. Please call reception directly for assistance."
            else:
                reply = f"I cannot find a booking with that reference and date of birth. Please check and try again. ({attempts}/3 attempts)"
        session.add_turn("user", patient_message)
        session.add_turn("assistant", reply)
        pending = get_pending_booking(session_id)
        return ChatResponse(
            reply=reply, session_id=session_id, intent="appointments",
            chunks=[], top_score=1.0, threshold_passed=True,
            booking_state=pending.get("state") if pending else BookingState.IDLE,
            booking_reference="", pending_booking=dict(pending) if pending else {}
        )

    if current_state == BookingState.CANCELLATION_PENDING:
        yes_no = detect_yes_no(patient_message)
        if yes_no == "yes":
            success = cancel_booking(pending.get("cancel_ref", ""))
            clear_pending_booking(session_id)
            reply = f"Your appointment {pending.get('cancel_ref')} has been cancelled. The slot has been released. Is there anything else I can help you with?" if success else "There was an issue cancelling. Please call reception directly."
        elif yes_no == "no":
            clear_pending_booking(session_id)
            reply = "No problem — your appointment has been kept. Is there anything else I can help you with?"
        else:
            reply = "Please say YES to confirm cancellation or NO to keep your appointment."
        session.add_turn("user", patient_message)
        session.add_turn("assistant", reply)
        return ChatResponse(
            reply=reply, session_id=session_id, intent="appointments",
            chunks=[], top_score=1.0, threshold_passed=True,
            booking_state=BookingState.CANCELLED if yes_no == "yes" else BookingState.IDLE,
            booking_reference="", pending_booking={}
        )

    # ── DETECT NEW CANCELLATION INTENT ───────────────────────────
    if is_cancellation_intent(patient_message) and current_state == BookingState.IDLE:
        upsert_pending_booking(session_id, {
            "state": BookingState.CANCELLATION_REFERENCE,
            "idempotency_key": f"{session_id}-cancel-{uuid.uuid4().hex[:8]}"
        })
        reply = "I can help you cancel your appointment. Could I have your reference number please? It will be in the format GMC-2026-0001."
        session.add_turn("user", patient_message)
        session.add_turn("assistant", reply)
        return ChatResponse(
            reply=reply, session_id=session_id, intent="appointments",
            chunks=[], top_score=1.0, threshold_passed=True,
            booking_state=BookingState.CANCELLATION_REFERENCE,
            booking_reference="", pending_booking={}
        )

    # ── RAG PIPELINE ─────────────────────────────────────────────
    today_dt = datetime.now()
    tomorrow = today_dt + timedelta(days=1)
    date_context = f"Today is {today_dt.strftime('%A %d %B %Y')}. Tomorrow is {tomorrow.strftime('%A %d %B %Y')}."
    retrieval_query = date_context + " " + session.build_retrieval_query(patient_message)

    intent = classify_intent(retrieval_query)
    print(f"Intent: {intent}")

    chunks = retriever.retrieve(retrieval_query, intent)

    if intent == "appointments":
        chunks = filter_appointment_chunks(chunks)

    relevant, top_score = check_relevance(chunks)
    chunk_texts = [c["text"] for c in chunks]

    if not relevant:
        reply = "I want to make sure I help you correctly — could you tell me a little more about what you need today?"
    else:
        context_text = "\n\n".join(chunk_texts)
        today_str = today_dt.strftime("%A %d %B %Y")

        # Build what slots are still needed for booking context
        booking_context = ""
        if pending and current_state not in [BookingState.PENDING_CONFIRMATION]:
            missing = [f for f in REQUIRED_SLOTS if not pending.get(f)]
            if missing:
                booking_context = f"\nACTIVE BOOKING — collected so far: {dict({k: pending.get(k) for k in REQUIRED_SLOTS if pending.get(k)})}\nStill needed: {missing}\nAsk naturally for missing fields only."

        dynamic_system = f"""{CARA_BASE_PROMPT}

TODAY'S DATE: {today_str}
When confirming appointment details always use the EXACT date from the booking context provided below — do NOT recalculate dates independently.
{booking_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELEVANT SURGERY INFORMATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        try:
            response = claude_client.messages.create(
                model="claude-opus-4-6",
                max_tokens=400,
                system=dynamic_system,
                messages=session.get_claude_history() + [{"role": "user", "content": patient_message}]
            )
            reply = response.content[0].text.strip()
            print(f"Response: {reply[:100]}...")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── SLOT FILLING ─────────────────────────────────────────────
    # Run on EVERY turn — no condition
    # Slot filler only extracts if fields are actually present in message
    # Zero harm if nothing found — just returns empty dict
    if True:
        new_fields = extract_slots(patient_message, reply, pending or {})

        if new_fields:
            print(f"  [SlotFiller] Extracted: {new_fields}")

            if not pending:
                new_fields["idempotency_key"] = f"{session_id}-{uuid.uuid4().hex[:8]}"
                new_fields["state"] = BookingState.IDLE

            upsert_pending_booking(session_id, new_fields)
            pending = get_pending_booking(session_id)

        # Check if all required slots now filled
        if pending and is_booking_complete(pending):
            upsert_pending_booking(session_id, {"state": BookingState.PENDING_CONFIRMATION})
            # Override Claude response with hardcoded confirmation readback
            reply = build_confirmation_readback(pending)
            print("  [SlotFiller] All slots filled → PENDING_CONFIRMATION → confirmation readback")

    # ── UPDATE HISTORY ───────────────────────────────────────────
    session.add_turn("user", patient_message)
    session.add_turn("assistant", reply)

    final_pending = get_pending_booking(session_id)
    final_state = final_pending.get("state") if final_pending else BookingState.IDLE

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        intent=intent,
        chunks=chunk_texts,
        top_score=round(top_score, 3),
        threshold_passed=relevant,
        booking_state=final_state,
        booking_reference=booking_reference,
        pending_booking=dict(final_pending) if final_pending else {}
    )


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────
@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    clear_session(session_id)
    clear_pending_booking(session_id)
    return {"status": "cleared"}

@app.get("/health")
async def health():
    return {"status": "CARA is running"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
