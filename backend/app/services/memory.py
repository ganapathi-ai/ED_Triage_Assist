"""
Conversation Memory Module
Manages multi-turn conversation context for the RAG chatbot
"""
import logging
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    sources: List[dict] = field(default_factory=list)  # RAG sources for this turn

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class ConversationSession:
    """A conversation session with memory."""
    session_id: str
    messages: deque = field(default_factory=lambda: deque(maxlen=50))
    summary: str = ""
    context_window: List[dict] = field(default_factory=list)
    user_preferences: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, metadata: dict = None, sources: List[dict] = None):
        """Add a message to the conversation."""
        msg = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
            sources=sources or [],
        )
        self.messages.append(msg)
        self.updated_at = time.time()

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get the n most recent messages."""
        return list(self.messages)[-n:]

    def get_context_for_llm(self, max_turns: int = 6) -> List[dict]:
        """Get formatted conversation context for LLM."""
        recent = self.get_recent_messages(max_turns)
        return [{"role": m.role, "content": m.content} for m in recent]

    def get_last_user_query(self) -> Optional[str]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
            "context_window": self.context_window,
            "user_preferences": self.user_preferences,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict):
        session = cls(session_id=data["session_id"])
        session.messages = deque(
            [Message.from_dict(m) for m in data.get("messages", [])],
            maxlen=50
        )
        session.summary = data.get("summary", "")
        session.context_window = data.get("context_window", [])
        session.user_preferences = data.get("user_preferences", {})
        session.created_at = data.get("created_at", time.time())
        session.updated_at = data.get("updated_at", time.time())
        return session


class ConversationMemory:
    """
    Manages all active conversation sessions.
    Provides context-aware retrieval and conversation management.
    """

    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout  # seconds

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]

    def add_user_message(self, session_id: str, content: str, metadata: dict = None):
        """Add a user message to the session."""
        session = self.get_or_create_session(session_id)
        session.add_message("user", content, metadata=metadata)

    def add_assistant_message(self, session_id: str, content: str, sources: List[dict] = None, metadata: dict = None):
        """Add an assistant message to the session."""
        session = self.get_or_create_session(session_id)
        session.add_message("assistant", content, metadata=metadata, sources=sources)

    def get_conversation_context(self, session_id: str, max_turns: int = 6) -> List[dict]:
        """Get conversation context for RAG."""
        session = self.get_or_create_session(session_id)
        return session.get_context_for_llm(max_turns)

    def get_session_summary(self, session_id: str) -> str:
        """Get a summary of the conversation session."""
        session = self.get_or_create_session(session_id)
        if not session.messages:
            return "No conversation yet."

        # Generate summary from recent messages
        recent = session.get_recent_messages(10)
        summary_parts = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            summary_parts.append(f"{role_label}: {msg.content[:100]}")

        return " | ".join(summary_parts)

    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.updated_at > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def get_stats(self) -> dict:
        """Get memory statistics."""
        total_messages = sum(len(s.messages) for s in self.sessions.values())
        return {
            "active_sessions": len(self.sessions),
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / max(len(self.sessions), 1),
        }


class ContextualRetriever:
    """
    Enhances retrieval by incorporating conversation context.
    Detects follow-up questions and adds context to the query.
    """

    def __init__(self, memory: ConversationMemory):
        self.memory = memory

    def get_contextual_query(self, session_id: str, current_query: str) -> str:
        """
        Enhance the current query with conversation context.
        Detects if this is a follow-up question and adds context.
        """
        session = self.memory.get_or_create_session(session_id)

        if not session.messages:
            return current_query

        last_user_query = session.get_last_user_query()

        # Check if this looks like a follow-up
        followup_indicators = [
            "it", "that", "this", "also", "additionally", "furthermore",
            "what about", "and", "moreover", "besides", "in addition",
            "what if", "how about", "tell me more", "elaborate",
        ]

        is_followup = any(indicator in current_query.lower() for indicator in followup_indicators)

        if is_followup and last_user_query and last_user_query != current_query:
            # Combine with previous context
            contextual_query = f"Context: {last_user_query}. Current question: {current_query}"
            return contextual_query

        return current_query

    def build_conversation_summary(self, session_id: str) -> str:
        """Build a summary of the conversation for context."""
        session = self.memory.get_or_create_session(session_id)
        messages = session.get_recent_messages(10)

        if not messages:
            return ""

        summary_parts = []
        for msg in messages:
            prefix = "User asked" if msg.role == "user" else "Assistant explained"
            summary_parts.append(f"{prefix}: {msg.content[:150]}")

        return "\n".join(summary_parts)


# Global memory instance
conversation_memory = ConversationMemory()
