"""Persist every chatbot interaction for later analysis."""

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger("meditrack.chat")


@dataclass(frozen=True)
class ChatLogEntry:
    query: str
    intent: str | None
    source: str
    cached: bool
    latency_ms: float
    error: str | None = None


class ChatLogger:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.Lock()  # one shared connection across worker threads

    def record(self, entry: ChatLogEntry) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO chat_logs (timestamp, query, intent, source, cached, latency_ms, error)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, entry.query, entry.intent, entry.source, int(entry.cached),
                     round(entry.latency_ms, 1), entry.error),
                )
                self._conn.commit()
        except sqlite3.Error:
            # Logging must never break the chat itself.
            log.exception("Failed to write chat log")
        log.info("chat intent=%s source=%s cached=%s latency=%.1fms", entry.intent, entry.source,
                 entry.cached, entry.latency_ms)

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, query, intent, source, cached, latency_ms, error"
                " FROM chat_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) | {"cached": bool(r["cached"])} for r in rows]
