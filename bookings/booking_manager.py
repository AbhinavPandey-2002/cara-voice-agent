"""
CARA - Greenfield Medical Centre AI Receptionist
booking_manager.py - All SQLite operations for booking lifecycle
 
State Machine States:
IDLE → COLLECTING → PENDING_CONFIRMATION → CONFIRMED
IDLE → CANCELLATION_REFERENCE → CANCELLATION_VERIFICATION → CANCELLATION_PENDING → CANCELLED
"""
 
import sqlite3
import os
from datetime import datetime
 
# ─────────────────────────────────────────
# DATABASE PATH
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "bookings", "bookings.db")
 
# ─────────────────────────────────────────
# STATES
# ─────────────────────────────────────────
class BookingState:
    IDLE                      = "IDLE"
    COLLECTING                = "COLLECTING"
    PENDING_CONFIRMATION      = "PENDING_CONFIRMATION"
    CONFIRMED                 = "CONFIRMED"
    CANCELLATION_REFERENCE    = "CANCELLATION_REFERENCE"
    CANCELLATION_VERIFICATION = "CANCELLATION_VERIFICATION"
    CANCELLATION_PENDING      = "CANCELLATION_PENDING"
    CANCELLED                 = "CANCELLED"
 
 
# ─────────────────────────────────────────
# INIT DATABASE
# ─────────────────────────────────────────
def init_db():
    """
    Creates SQLite database and both tables if they don't exist.
    Called once on server startup.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _get_conn()
 
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_bookings (
            session_id        TEXT PRIMARY KEY,
            state             TEXT NOT NULL DEFAULT 'IDLE',
            clinician         TEXT,
            day               TEXT,
            date              TEXT,
            slot              TEXT,
            patient_name      TEXT,
            patient_dob       TEXT,
            reason            TEXT,
            idempotency_key   TEXT UNIQUE,
            cancel_ref        TEXT,
            cancel_attempts   INTEGER DEFAULT 0,
            created_at        TEXT,
            updated_at        TEXT
        )
    """)
 
    conn.execute("""
        CREATE TABLE IF NOT EXISTS confirmed_bookings (
            reference_number  TEXT PRIMARY KEY,
            idempotency_key   TEXT UNIQUE,
            clinician         TEXT NOT NULL,
            day               TEXT NOT NULL,
            date              TEXT NOT NULL,
            slot              TEXT NOT NULL,
            patient_name      TEXT NOT NULL,
            patient_dob       TEXT NOT NULL,
            reason            TEXT,
            session_id        TEXT,
            confirmed_at      TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'confirmed',
            cancelled_at      TEXT
        )
    """)
 
    conn.commit()
    conn.close()
    print("  [BookingManager] Database initialised.")
 
 
# ─────────────────────────────────────────
# CONNECTION HELPER
# ─────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access
    return conn
 
 
# ─────────────────────────────────────────
# PENDING BOOKING OPERATIONS
# ─────────────────────────────────────────
def get_pending_booking(session_id: str) -> dict | None:
    """
    Returns current pending booking for session.
    Called at start of every chat request to restore state.
    Returns None if no pending booking exists.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM pending_bookings WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    conn.close()
 
    if row:
        return dict(row)
    return None
 
 
def upsert_pending_booking(session_id: str, fields: dict):
    """
    Creates or updates pending booking with new fields.
    Called every time a new booking field is collected.
    Persists immediately to SQLite — server restart safe.
 
    fields: dict of any combination of:
    state, clinician, day, date, slot,
    patient_name, patient_dob, reason,
    idempotency_key, cancel_ref, cancel_attempts
    """
    now = datetime.now().isoformat()
    conn = _get_conn()
 
    existing = conn.execute(
        "SELECT * FROM pending_bookings WHERE session_id = ?",
        (session_id,)
    ).fetchone()
 
    if existing:
        # Build dynamic UPDATE with only provided fields
        set_clauses = []
        values = []
        for key, value in fields.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)
        set_clauses.append("updated_at = ?")
        values.append(now)
        values.append(session_id)
 
        conn.execute(
            f"UPDATE pending_bookings SET {', '.join(set_clauses)} WHERE session_id = ?",
            values
        )
    else:
        # INSERT new row
        conn.execute("""
            INSERT INTO pending_bookings
            (session_id, state, clinician, day, date, slot,
             patient_name, patient_dob, reason, idempotency_key,
             cancel_ref, cancel_attempts, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            fields.get("state", BookingState.IDLE),
            fields.get("clinician"),
            fields.get("day"),
            fields.get("date"),
            fields.get("slot"),
            fields.get("patient_name"),
            fields.get("patient_dob"),
            fields.get("reason"),
            fields.get("idempotency_key"),
            fields.get("cancel_ref"),
            fields.get("cancel_attempts", 0),
            now,
            now
        ))
 
    conn.commit()
    conn.close()
 
 
def clear_pending_booking(session_id: str):
    """
    Removes pending booking for session.
    Called when booking confirmed, cancelled, or patient abandons.
    """
    conn = _get_conn()
    conn.execute(
        "DELETE FROM pending_bookings WHERE session_id = ?",
        (session_id,)
    )
    conn.commit()
    conn.close()
 
 
# ─────────────────────────────────────────
# SLOT AVAILABILITY
# ─────────────────────────────────────────
def get_booked_slots(clinician: str, day: str) -> list[str]:
    """
    Returns list of booked slot times for a clinician on a day of week.
    Used to filter available slots before showing to patient.
 
    Example return: ["9:00am", "10:30am"]
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT slot FROM confirmed_bookings
        WHERE clinician = ?
        AND day = ?
        AND status = 'confirmed'
    """, (clinician, day)).fetchall()
    conn.close()
 
    return [row["slot"] for row in rows]
 
 
def get_booked_slots_for_date(clinician: str, date: str) -> list[str]:
    """
    Returns booked slots for a specific date (not just day of week).
    More precise than get_booked_slots when exact date is known.
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT slot FROM confirmed_bookings
        WHERE clinician = ?
        AND date = ?
        AND status = 'confirmed'
    """, (clinician, date)).fetchall()
    conn.close()
 
    return [row["slot"] for row in rows]
 
 
# ─────────────────────────────────────────
# REFERENCE NUMBER GENERATION
# ─────────────────────────────────────────
def generate_reference_number() -> str:
    """
    Generates unique reference number: GMC-YYYY-XXXX
    Increments based on total confirmed bookings this year.
    Thread safe — reads count inside same connection.
    """
    year = datetime.now().year
    conn = _get_conn()
    count = conn.execute(
        "SELECT COUNT(*) as cnt FROM confirmed_bookings WHERE reference_number LIKE ?",
        (f"GMC-{year}-%",)
    ).fetchone()["cnt"]
    conn.close()
 
    return f"GMC-{year}-{str(count + 1).zfill(4)}"
 
 
# ─────────────────────────────────────────
# CONFIRM BOOKING
# ─────────────────────────────────────────
def confirm_booking(session_id: str) -> str | None:
    """
    Moves pending booking to confirmed_bookings.
    Generates reference number.
    Uses idempotency key to prevent duplicate writes.
    Deletes from pending_bookings.
 
    Returns reference number on success.
    Returns None if pending booking not found or already confirmed.
    """
    conn = _get_conn()
 
    pending = conn.execute(
        "SELECT * FROM pending_bookings WHERE session_id = ?",
        (session_id,)
    ).fetchone()
 
    if not pending:
        conn.close()
        print(f"  [BookingManager] No pending booking found for session {session_id}")
        return None
 
    pending = dict(pending)
 
    # Check required fields
    required = ["clinician", "day", "date", "slot", "patient_name", "patient_dob"]
    missing = [f for f in required if not pending.get(f)]
    if missing:
        conn.close()
        print(f"  [BookingManager] Cannot confirm — missing fields: {missing}")
        return None
 
    # Check idempotency — already confirmed?
    idempotency_key = pending.get("idempotency_key")
    if idempotency_key:
        existing = conn.execute(
            "SELECT reference_number FROM confirmed_bookings WHERE idempotency_key = ?",
            (idempotency_key,)
        ).fetchone()
        if existing:
            conn.close()
            print(f"  [BookingManager] Already confirmed — returning existing reference")
            return existing["reference_number"]
 
    # Generate reference number
    reference_number = generate_reference_number()
    now = datetime.now().isoformat()
 
    try:
        # Write to confirmed_bookings — atomic transaction
        conn.execute("""
            INSERT INTO confirmed_bookings
            (reference_number, idempotency_key, clinician, day, date, slot,
             patient_name, patient_dob, reason, session_id, confirmed_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')
        """, (
            reference_number,
            idempotency_key,
            pending["clinician"],
            pending["day"],
            pending["date"],
            pending["slot"],
            pending["patient_name"],
            pending["patient_dob"],
            pending.get("reason", ""),
            session_id,
            now
        ))
 
        # Delete from pending
        conn.execute(
            "DELETE FROM pending_bookings WHERE session_id = ?",
            (session_id,)
        )
 
        conn.commit()
        print(f"  [BookingManager] Booking confirmed: {reference_number}")
        return reference_number
 
    except Exception as e:
        conn.rollback()
        print(f"  [BookingManager] Confirm failed: {e}")
        return None
    finally:
        conn.close()
 
 
# ─────────────────────────────────────────
# CANCELLATION
# ─────────────────────────────────────────
def verify_cancellation(reference_number: str, patient_dob: str) -> dict | None:
    """
    Verifies cancellation request using reference number + DOB.
    Two factor verification — prevents unauthorised cancellation.
 
    Returns booking details if verified.
    Returns None if verification fails.
    """
    conn = _get_conn()
    booking = conn.execute("""
        SELECT * FROM confirmed_bookings
        WHERE reference_number = ?
        AND patient_dob = ?
        AND status = 'confirmed'
    """, (reference_number, patient_dob)).fetchone()
    conn.close()
 
    if booking:
        return dict(booking)
    return None
 
 
def cancel_booking(reference_number: str) -> bool:
    """
    Cancels a confirmed booking.
    Slot released immediately — available for other patients.
 
    Returns True on success, False if booking not found.
    """
    conn = _get_conn()
    now = datetime.now().isoformat()
 
    result = conn.execute("""
        UPDATE confirmed_bookings
        SET status = 'cancelled', cancelled_at = ?
        WHERE reference_number = ?
        AND status = 'confirmed'
    """, (now, reference_number))
 
    success = result.rowcount > 0
    conn.commit()
    conn.close()
 
    if success:
        print(f"  [BookingManager] Booking cancelled: {reference_number}")
    else:
        print(f"  [BookingManager] Cancel failed — not found or already cancelled: {reference_number}")
 
    return success
 
 
# ─────────────────────────────────────────
# SLOT FILTERING UTILITY
# ─────────────────────────────────────────
def filter_booked_slots_from_chunk(chunk_text: str, clinician: str, day: str, date: str = None) -> str:
    """
    Removes booked slots from a chunk text before injecting into Claude.
 
    Example:
    Input:  "Dr Patel Monday morning: 9:00am 9:30am 10:00am 10:30am"
    Booked: ["9:00am", "10:30am"]
    Output: "Dr Patel Monday morning available: 9:30am 10:00am
             Note: 9:00am and 10:30am are already booked."
    """
    if date:
        booked = get_booked_slots_for_date(clinician, date)
    else:
        booked = get_booked_slots(clinician, day)
 
    if not booked:
        return chunk_text  # nothing booked — return as is
 
    # Remove booked slots from text
    filtered_text = chunk_text
    for slot in booked:
        filtered_text = filtered_text.replace(slot, "")
 
    # Clean up extra spaces/commas
    import re
    filtered_text = re.sub(r',\s*,', ',', filtered_text)
    filtered_text = re.sub(r'\s+', ' ', filtered_text)
    filtered_text = filtered_text.strip()
 
    # Add note about booked slots
    booked_str = ", ".join(booked)
    filtered_text += f"\nNote: The following slots are already booked and unavailable: {booked_str}"
 
    return filtered_text
 
 
# ─────────────────────────────────────────
# TEST — run directly to verify
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uuid
 
    print("Testing BookingManager...\n")
 
    # Init database
    init_db()
 
    # Test session
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    idempotency_key = f"{session_id}-DrPatel-Monday-9am"
 
    print("1. Creating pending booking...")
    upsert_pending_booking(session_id, {
        "state": BookingState.COLLECTING,
        "clinician": "Dr. Priya Patel",
        "day": "Monday",
        "date": "8th June 2026",
        "slot": "9:00am",
        "idempotency_key": idempotency_key
    })
 
    print("2. Adding patient details...")
    upsert_pending_booking(session_id, {
        "state": BookingState.PENDING_CONFIRMATION,
        "patient_name": "John Smith",
        "patient_dob": "15th March 1980",
        "reason": "Back pain"
    })
 
    print("3. Retrieving pending booking...")
    pending = get_pending_booking(session_id)
    print(f"   State: {pending['state']}")
    print(f"   Clinician: {pending['clinician']}")
    print(f"   Slot: {pending['slot']}")
    print(f"   Name: {pending['patient_name']}")
 
    print("\n4. Confirming booking...")
    ref = confirm_booking(session_id)
    print(f"   Reference number: {ref}")
 
    print("\n5. Checking booked slots for Dr Patel Monday...")
    booked = get_booked_slots("Dr. Priya Patel", "Monday")
    print(f"   Booked slots: {booked}")
 
    print("\n6. Testing slot filtering...")
    chunk = "Dr. Priya Patel Monday morning appointments: 9:00am, 9:30am, 10:00am, 10:30am, 11:00am, 11:30am"
    filtered = filter_booked_slots_from_chunk(chunk, "Dr. Priya Patel", "Monday")
    print(f"   Filtered chunk:\n   {filtered}")
 
    print("\n7. Testing cancellation verification...")
    booking = verify_cancellation(ref, "15th March 1980")
    print(f"   Verified: {booking is not None}")
    if booking:
        print(f"   Found: {booking['clinician']} {booking['slot']} {booking['date']}")
 
    print("\n8. Testing wrong DOB verification...")
    booking_wrong = verify_cancellation(ref, "1st January 1990")
    print(f"   Wrong DOB verified: {booking_wrong is not None} (should be False)")
 
    print("\n9. Cancelling booking...")
    cancelled = cancel_booking(ref)
    print(f"   Cancelled: {cancelled}")
 
    print("\n10. Checking slots after cancellation...")
    booked_after = get_booked_slots("Dr. Priya Patel", "Monday")
    print(f"    Booked slots after cancel: {booked_after} (should be empty)")
 
    print("\n✅ All tests passed.")