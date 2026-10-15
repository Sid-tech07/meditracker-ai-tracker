"""Free-slot lookup against the appointments table, with an in-memory TTL cache.

A slot is free when the doctor works that weekday, the time is inside clinic
hours (lunch excluded), no 'booked' appointment holds it and it hasn't
already started.
"""

import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from cachetools import TTLCache

from app.schedule import LUNCH_END, slot_times, works_on

TimeOfDay = Literal["any", "morning", "afternoon"]


@dataclass(frozen=True)
class AvailabilityQuery:
    day: date
    department: str | None = None
    doctor: str | None = None
    time_of_day: TimeOfDay = "any"

    def cache_key(self) -> tuple:
        return (
            self.day.isoformat(),
            (self.department or "").strip().lower(),
            (self.doctor or "").strip().lower(),
            self.time_of_day,
        )


@dataclass(frozen=True)
class DoctorSlots:
    doctor: str
    department: str
    slots: tuple[str, ...]  # "HH:MM"


def _matches_time_of_day(slot: datetime, time_of_day: TimeOfDay) -> bool:
    if time_of_day == "morning":
        return slot.hour < 13
    if time_of_day == "afternoon":
        return slot.time() >= LUNCH_END
    return True


def find_available_slots(
    conn: sqlite3.Connection, query: AvailabilityQuery, now: datetime | None = None
) -> list[DoctorSlots]:
    now = now or datetime.now()
    sql = "SELECT id, name, department, working_days FROM doctors WHERE 1 = 1"
    params: list[str] = []
    if query.department:
        sql += " AND LOWER(department) LIKE ?"
        params.append(f"%{query.department.strip().lower()}%")
    if query.doctor:
        # tolerate "Dr. Rao", "rao", "Ananya Rao"
        name = query.doctor.strip().lower().removeprefix("dr.").removeprefix("dr ").strip()
        sql += " AND LOWER(name) LIKE ?"
        params.append(f"%{name}%")
    doctors = conn.execute(sql + " ORDER BY department, name", params).fetchall()

    results = []
    for doc in doctors:
        if not works_on(doc["working_days"], query.day):
            continue
        booked = {
            row["start_time"]
            for row in conn.execute(
                "SELECT start_time FROM appointments WHERE doctor_id = ? AND status = 'booked'"
                " AND start_time LIKE ?",
                (doc["id"], f"{query.day.isoformat()}%"),
            )
        }
        free = tuple(
            s.strftime("%H:%M")
            for s in slot_times(query.day)
            if s > now
            and _matches_time_of_day(s, query.time_of_day)
            and s.isoformat(timespec="minutes") not in booked
        )
        if free:
            results.append(DoctorSlots(doc["name"], doc["department"], free))
    return results


class AvailabilityService:
    """Wraps the lookup with a thread-safe TTL cache.

    The mock data is read-only, so a time-based expiry is enough. A real
    system would also call `invalidate()` whenever a booking changes.
    """

    def __init__(self, conn: sqlite3.Connection, ttl_seconds: int = 300, max_entries: int = 256,
                 clock: Callable[[], datetime] = datetime.now):
        self._conn = conn
        self._clock = clock
        self._cache: TTLCache = TTLCache(maxsize=max_entries, ttl=ttl_seconds)
        self._lock = threading.Lock()

    def get(self, query: AvailabilityQuery) -> tuple[list[DoctorSlots], bool]:
        """Return (slots, served_from_cache)."""
        key = query.cache_key()
        with self._lock:
            if key in self._cache:
                return self._cache[key], True
        slots = find_available_slots(self._conn, query, now=self._clock())
        with self._lock:
            self._cache[key] = slots
        return slots, False

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def list_departments(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT DISTINCT department FROM doctors ORDER BY 1")]

    def list_doctors(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute("SELECT name, department FROM doctors ORDER BY department, name")]
