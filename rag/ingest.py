"""
CARA - Greenfield Medical Centre AI Receptionist
ingest.py - Run once to load all knowledge base data into ChromaDB

Run from CARA/ folder:
    python rag/ingest.py
"""

import json
import os
import sys

# Add parent directory to path so we can run from CARA/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
CHROMA_DIR = os.path.join(BASE_DIR, "rag", "chroma_db")

# ─────────────────────────────────────────
# INIT EMBEDDING MODEL
# ─────────────────────────────────────────
print("Loading embedding model: all-MiniLM-L6-v2 ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.\n")

# ─────────────────────────────────────────
# INIT CHROMADB — 4 SEPARATE COLLECTIONS
# ─────────────────────────────────────────
print("Connecting to ChromaDB ...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete existing collections so we can re-ingest cleanly
for name in ["appointments", "triage", "prescription", "hours"]:
    try:
        client.delete_collection(name)
        print(f"  Deleted existing collection: {name}")
    except Exception:
        pass

# Create fresh collections with cosine similarity
appointments_col = client.create_collection(
    name="appointments",
    metadata={"hnsw:space": "cosine"}
)
triage_col = client.create_collection(
    name="triage",
    metadata={"hnsw:space": "cosine"}
)
prescription_col = client.create_collection(
    name="prescription",
    metadata={"hnsw:space": "cosine"}
)
hours_col = client.create_collection(
    name="hours",
    metadata={"hnsw:space": "cosine"}
)
print("ChromaDB collections created.\n")


# ─────────────────────────────────────────
# HELPER — EMBED AND STORE CHUNKS
# ─────────────────────────────────────────
def store_chunks(collection, chunks: list[dict]):
    """
    chunks: list of {"text": str, "metadata": dict}
    Embeds each chunk and stores in ChromaDB collection.
    """
    if not chunks:
        print(f"  WARNING: No chunks to store in {collection.name}")
        return

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{collection.name}_chunk_{i}" for i in range(len(chunks))]

    print(f"  Embedding {len(chunks)} chunks for [{collection.name}] ...")
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"  Stored {len(chunks)} chunks in [{collection.name}]\n")


# ─────────────────────────────────────────
# STEP 1 — DOCTORS + APPOINTMENTS → appointments collection
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 1: Processing doctors.json + appointments.json")
print("=" * 50)

appointment_chunks = []

# --- Doctors ---
doctors_path = os.path.join(KB_DIR, "doctors.json")
with open(doctors_path, encoding="utf-8") as f:
    doctors = json.load(f)

for doc in doctors:
    # Build natural language chunk for each clinician
    specialisms = ", ".join(doc["specialisms"])
    available_days = ", ".join(doc["available_days"])
    languages = ", ".join(doc["languages"])
    leave_dates = ", ".join(doc["not_available_days"]) if doc["not_available_days"] else "None"

    chunk_text = f"""Clinician: {doc['name']}
Role: {doc['role']}
Gender: {doc['gender']}
Specialises in: {specialisms}
Available days of week: {available_days}
Leave dates: {leave_dates}
Languages spoken: {languages}
Notes: {doc['notes']}"""

    appointment_chunks.append({
        "text": chunk_text.strip(),
        "metadata": {
            "source": "doctors",
            "clinician": doc["name"],
            "role": doc["role"]
        }
    })

print(f"  Loaded {len(doctors)} clinicians from doctors.json")

# --- Appointments weekly schedule ---
appointments_path = os.path.join(KB_DIR, "appointments.json")
with open(appointments_path, encoding="utf-8") as f:
    appointments_data = json.load(f)

weekly_schedule = appointments_data["weekly_schedule"]
leaves = appointments_data["leaves"]
instructions = appointments_data["instructions_for_cara"]

# One chunk per clinician per day
for entry in weekly_schedule:
    clinician = entry["clinician"]
    day = entry["day"]
    morning = ", ".join(entry["morning_slots"]) if entry["morning_slots"] else "No morning slots"
    afternoon = ", ".join(entry["afternoon_slots"]) if entry["afternoon_slots"] else "No afternoon slots"
    notes = entry.get("notes", "")

    if day == "Sunday":
        chunk_text = f"""Day: Sunday
All clinicians: Surgery is closed on Sundays.
For urgent advice call 111. For emergencies call 999."""
        # Only add Sunday chunk once
        if not any(c["metadata"].get("day") == "Sunday" for c in appointment_chunks):
            appointment_chunks.append({
                "text": chunk_text.strip(),
                "metadata": {"source": "schedule", "day": "Sunday", "clinician": "all"}
            })
        continue

    chunk_text = f"""{clinician} {day} appointments:
Role: {entry['role']}
{clinician} is available on {day}.
{day} morning appointments: {morning}
{day} afternoon appointments: {afternoon}"""

    if notes:
        chunk_text += f"\nNotes: {notes}"

    appointment_chunks.append({
        "text": chunk_text.strip(),
        "metadata": {
            "source": "schedule",
            "clinician": clinician,
            "day": day,
            "role": entry["role"]
        }
    })

# Leave dates as separate chunks
for leave in leaves:
    chunk_text = f"""Leave notice: {leave['clinician']} is absent on {leave['date']} ({leave['day']}).
Note: {leave['note']}"""
    appointment_chunks.append({
        "text": chunk_text.strip(),
        "metadata": {
            "source": "leave",
            "clinician": leave["clinician"],
            "date": leave["date"]
        }
    })

# Instructions for CARA as a chunk
appointment_chunks.append({
    "text": f"Instructions for handling appointment queries: {instructions}",
    "metadata": {"source": "instructions", "clinician": "all"}
})

print(f"  Loaded {len(weekly_schedule)} schedule entries")
print(f"  Loaded {len(leaves)} leave entries")
print(f"  Total appointment chunks: {len(appointment_chunks)}")

store_chunks(appointments_col, appointment_chunks)


# ─────────────────────────────────────────
# STEP 2 — TRIAGE POLICIES → triage collection
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 2: Processing triage_policies.txt")
print("=" * 50)

triage_chunks = []
triage_path = os.path.join(KB_DIR, "triage_policies.txt")

with open(triage_path, encoding="utf-8") as f:
    content = f.read()

# Split by blank line — each policy rule = one chunk
raw_policies = [p.strip() for p in content.split("\n\n") if p.strip()]

for i, policy in enumerate(raw_policies):
    # Determine urgency level from first line
    first_line = policy.split("\n")[0].upper()
    if "999" in first_line or "EMERGENCY" in first_line:
        urgency = "emergency_999"
    elif "SAME DAY" in first_line or "URGENT" in first_line:
        urgency = "urgent_same_day"
    elif "ROUTINE" in first_line:
        urgency = "routine_gp"
    elif "NURSE" in first_line:
        urgency = "nurse_appropriate"
    else:
        urgency = "self_care"

    triage_chunks.append({
        "text": policy,
        "metadata": {
            "source": "triage_policies",
            "urgency": urgency
        }
    })

print(f"  Loaded {len(triage_chunks)} triage policy chunks")
store_chunks(triage_col, triage_chunks)


# ─────────────────────────────────────────
# STEP 3 — PRESCRIPTION POLICIES → prescription collection
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 3: Processing prescription_policies.txt")
print("=" * 50)

prescription_chunks = []
prescription_path = os.path.join(KB_DIR, "prescription_policies.txt")

with open(prescription_path, encoding="utf-8") as f:
    content = f.read()

raw_policies = [p.strip() for p in content.split("\n\n") if p.strip()]

for policy in raw_policies:
    prescription_chunks.append({
        "text": policy,
        "metadata": {"source": "prescription_policies"}
    })

print(f"  Loaded {len(prescription_chunks)} prescription policy chunks")
store_chunks(prescription_col, prescription_chunks)


# ─────────────────────────────────────────
# STEP 4 — OPENING HOURS + FAQS → hours collection
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 4: Processing opening_hours.txt + faqs.txt")
print("=" * 50)

hours_chunks = []

# Opening hours
hours_path = os.path.join(KB_DIR, "opening_hours.txt")
with open(hours_path, encoding="utf-8") as f:
    content = f.read()

raw_sections = [s.strip() for s in content.split("\n\n") if s.strip()]
for section in raw_sections:
    hours_chunks.append({
        "text": section,
        "metadata": {"source": "opening_hours"}
    })
print(f"  Loaded {len(raw_sections)} opening hours chunks")

# FAQs
faqs_path = os.path.join(KB_DIR, "faqs.txt")
with open(faqs_path, encoding="utf-8") as f:
    content = f.read()

raw_faqs = [f.strip() for f in content.split("\n\n") if f.strip()]
for faq in raw_faqs:
    hours_chunks.append({
        "text": faq,
        "metadata": {"source": "faqs"}
    })
print(f"  Loaded {len(raw_faqs)} FAQ chunks")
print(f"  Total hours collection chunks: {len(hours_chunks)}")

store_chunks(hours_col, hours_chunks)


# ─────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────
print("=" * 50)
print("VERIFICATION — chunks stored per collection:")
print("=" * 50)
print(f"  appointments : {appointments_col.count()} chunks")
print(f"  triage       : {triage_col.count()} chunks")
print(f"  prescription : {prescription_col.count()} chunks")
print(f"  hours        : {hours_col.count()} chunks")
print(f"  TOTAL        : {appointments_col.count() + triage_col.count() + prescription_col.count() + hours_col.count()} chunks")
print()
print("=" * 50)
print("INGESTION COMPLETE — ChromaDB is ready.")
print("Next step: run  python main.py  to start CARA")
print("=" * 50)