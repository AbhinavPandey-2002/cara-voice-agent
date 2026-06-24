"""
CARA - Greenfield Medical Centre AI Receptionist
conversation_manager.py - Manages conversation history,
rolling summary every 4 turns, builds retrieval query
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
SUMMARY_EVERY_N_TURNS = 4      # update summary every 4 turns
MAX_TURNS_FOR_CLAUDE = 30      # last 30 turns sent to Claude for response
TURNS_FOR_RETRIEVAL = 6        # last 6 raw turns combined for retrieval query

# ─────────────────────────────────────────
# INIT CLIENT
# ─────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class ConversationManager:
    """
    Manages conversation state for one patient session.
    
    Responsibilities:
    - Store full conversation history
    - Generate rolling summary every 4 turns
    - Build combined retrieval query: summary + last 6 turns + current message
    - Return last 30 turns for Claude response generation
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []           # full conversation history
        self.summary = ""           # rolling summary of conversation so far
        self.turn_counter = 0       # counts every message added

    # ─────────────────────────────────────
    # ADD TURN
    # ─────────────────────────────────────
    def add_turn(self, role: str, content: str):
        """
        Adds one turn to history.
        role: "user" or "assistant"
        content: message text
        """
        self.history.append({
            "role": role,
            "content": content
        })
        self.turn_counter += 1

        # Update rolling summary every 4 turns
        if self.turn_counter % SUMMARY_EVERY_N_TURNS == 0:
            self._update_summary()

    # ─────────────────────────────────────
    # BUILD RETRIEVAL QUERY
    # ─────────────────────────────────────
    def build_retrieval_query(self, current_message: str) -> str:
        """
        Builds combined retrieval query:
        rolling summary + last 6 raw turns + current message

        This gives retriever:
        - Long term context (summary)
        - Recent specific context (last 6 turns)
        - Current need (current message)
        """
        parts = []

        # Add rolling summary if exists
        if self.summary:
            parts.append(f"Conversation summary: {self.summary}")

        # Add last 6 raw turns
        last_turns = self.history[-TURNS_FOR_RETRIEVAL:]
        if last_turns:
            turns_text = " ".join([t["content"] for t in last_turns])
            parts.append(f"Recent conversation: {turns_text}")

        # Add current message
        parts.append(f"Current query: {current_message}")

        combined = " ".join(parts)
        return combined

    # ─────────────────────────────────────
    # GET CLAUDE HISTORY
    # ─────────────────────────────────────
    def get_claude_history(self) -> list[dict]:
        """
        Returns last 30 turns for Claude response generation.
        """
        return self.history[-MAX_TURNS_FOR_CLAUDE:]

    # ─────────────────────────────────────
    # UPDATE ROLLING SUMMARY
    # ─────────────────────────────────────
    def _update_summary(self):
        """
        Sends last 4 turns to Claude Haiku to generate
        a clean rolling summary of the conversation.
        """
        last_4_turns = self.history[-SUMMARY_EVERY_N_TURNS:]

        if not last_4_turns:
            return

        # Build conversation text for summarisation
        conversation_text = ""
        for turn in last_4_turns:
            role = "Patient" if turn["role"] == "user" else "CARA"
            conversation_text += f"{role}: {turn['content']}\n"

        # Include previous summary if exists
        previous_summary = ""
        if self.summary:
            previous_summary = f"Previous summary: {self.summary}\n\n"

        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=150,
                system="""You are summarising a GP surgery receptionist conversation.
Write a 2-3 sentence summary covering:
- What the patient needs
- What has been resolved or confirmed
- What is still pending or unclear
Be concise and factual. No preamble.""",
                messages=[
                    {
                        "role": "user",
                        "content": f"{previous_summary}Latest conversation:\n{conversation_text}\n\nProvide updated summary:"
                    }
                ]
            )
            self.summary = response.content[0].text.strip()
            print(f"  [ConversationManager] Summary updated: {self.summary[:100]}...")

        except Exception as e:
            print(f"  [ConversationManager] Summary update failed: {e}")
            # Keep previous summary if update fails

    # ─────────────────────────────────────
    # CLEAR SESSION
    # ─────────────────────────────────────
    def clear(self):
        """Resets conversation state."""
        self.history = []
        self.summary = ""
        self.turn_counter = 0


# ─────────────────────────────────────────
# SESSION STORE — one manager per session
# ─────────────────────────────────────────
_sessions: dict[str, ConversationManager] = {}


def get_session(session_id: str) -> ConversationManager:
    """Returns existing session or creates new one."""
    if session_id not in _sessions:
        _sessions[session_id] = ConversationManager(session_id)
        print(f"  [ConversationManager] New session created: {session_id}")
    return _sessions[session_id]


def clear_session(session_id: str):
    """Clears and removes a session."""
    if session_id in _sessions:
        _sessions[session_id].clear()
        del _sessions[session_id]
        print(f"  [ConversationManager] Session cleared: {session_id}")


# ─────────────────────────────────────────
# TEST — run directly to verify
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ConversationManager...\n")

    # Simulate a multi-turn conversation
    session = ConversationManager("test-session-001")

    turns = [
        ("user", "Hello I need to book an appointment"),
        ("assistant", "Good morning, Greenfield Medical Centre, CARA speaking. I'd be happy to help you book an appointment. Could you tell me which doctor you'd like to see?"),
        ("user", "I would like to see Dr Patel please"),
        ("assistant", "Of course. Dr Patel is available Monday through Friday. What day works best for you?"),
        ("user", "Can I come in tomorrow morning"),
        ("assistant", "Dr Patel has slots available tomorrow morning at 9:00am, 9:30am, 10:00am, 10:30am, 11:00am and 11:30am. Which would you prefer?"),
        ("user", "9am please"),
        ("assistant", "I have booked you in with Dr Patel tomorrow at 9:00am. Could I take your full name and date of birth to confirm the booking?"),
    ]

    print("Adding turns to conversation...\n")
    for role, content in turns:
        session.add_turn(role, content)
        print(f"  Turn {session.turn_counter}: [{role}] {content[:60]}...")

    print(f"\nFinal summary: {session.summary}")
    print(f"\nTotal turns in history: {len(session.history)}")
    print(f"Turns sent to Claude: {len(session.get_claude_history())}")

    print("\nRetrieval query for 'What about a prescription as well':")
    retrieval_query = session.build_retrieval_query("What about a prescription as well")
    print(f"\n{retrieval_query}")