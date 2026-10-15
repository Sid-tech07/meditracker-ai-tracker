"""Chatbot orchestration: parse -> look up real data -> write a grounded reply -> log.

Every number and time in a reply comes from the database, never from the LLM.
"""

import logging
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta

from cachetools import TTLCache

from app.features import load_booked
from app.schedule import CLOSE_TIME, LUNCH_END, LUNCH_START, OPEN_TIME
from app.schemas import AppointmentRisk, ChatResponse, SlotGroup
from app.services.availability import AvailabilityQuery, AvailabilityService, DoctorSlots
from app.services.chat_log import ChatLogEntry, ChatLogger
from app.services.intent import IntentParser, ParseContext, ParsedQuery, RuleBasedParser
from app.services.predictor import NoShowPredictor

log = logging.getLogger("meditrack.chat")

BOOKING_HORIZON_DAYS = 14
MAX_SLOTS_IN_TEXT = 6
HELP = ("I can check free appointment slots and flag likely no-shows. Try: "
        "\"Any cardiology slots tomorrow morning?\", \"When is Dr. Rao free on Monday?\" "
        "or \"Which appointments tomorrow are high risk?\"")


def _pretty(day: date) -> str:
    return day.strftime("%a %d %b")


class ChatService:
    def __init__(
        self,
        availability: AvailabilityService,
        predictor: NoShowPredictor,
        chat_logger: ChatLogger,
        conn,
        llm_parser: IntentParser | None = None,
        today: Callable[[], date] = date.today,
        cache_ttl_seconds: int = 300,
    ):
        self._availability = availability
        self._predictor = predictor
        self._logger = chat_logger
        self._conn = conn
        self._llm = llm_parser
        self._rules = RuleBasedParser()
        self._today = today
        # Same question, same day -> skip the LLM round trip entirely.
        self._intent_cache: TTLCache = TTLCache(maxsize=512, ttl=cache_ttl_seconds)

    # ------------------------------------------------------------------ public

    def handle(self, message: str) -> ChatResponse:
        started = time.perf_counter()
        message = message.strip()
        parsed, source, error = self._parse(message)
        try:
            response = self._answer(parsed, source)
        except Exception as exc:  # noqa: BLE001 - last line of defence, logged below
            log.exception("Failed to answer chat message")
            error = f"{error + '; ' if error else ''}answer failed: {exc}"
            response = ChatResponse(reply="Sorry, something went wrong looking that up. Please try again.",
                                    intent="error", source=source, cached=False, latency_ms=0)
        response.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self._logger.record(ChatLogEntry(message, response.intent, response.source, response.cached,
                                         response.latency_ms, error))
        return response

    # ----------------------------------------------------------------- parsing

    def _context(self) -> ParseContext:
        doctors = self._availability.list_doctors()
        return ParseContext(today=self._today(), departments=self._availability.list_departments(),
                            doctors=[d["name"] for d in doctors])

    def _parse(self, message: str) -> tuple[ParsedQuery, str, str | None]:
        """Returns (query, source, error). Never raises."""
        key = (re.sub(r"\s+", " ", message.lower()), self._today().isoformat())
        if key in self._intent_cache:
            return self._intent_cache[key], "cache", None

        ctx = self._context()
        error = None
        if self._llm is not None:
            try:
                parsed = self._llm.parse(message, ctx)
                self._intent_cache[key] = parsed
                return parsed, self._llm.name, None
            except Exception as exc:  # noqa: BLE001 - any LLM failure -> rules
                error = f"llm failed, used fallback: {type(exc).__name__}: {exc}"[:300]
                log.warning(error)
        parsed = self._rules.parse(message, ctx)
        self._intent_cache[key] = parsed
        return parsed, self._rules.name, error

    # --------------------------------------------------------------- answering

    def _answer(self, q: ParsedQuery, source: str) -> ChatResponse:
        if q.intent == "availability":
            return self._availability_answer(q, source)
        if q.intent == "no_show_risk":
            return self._risk_answer(q, source)
        if q.intent == "clinic_info":
            reply = (f"Clinics are open Monday to Saturday, {OPEN_TIME:%H:%M}-{CLOSE_TIME:%H:%M}, closed for lunch "
                     f"{LUNCH_START:%H:%M}-{LUNCH_END:%H:%M} and on Sundays. Departments: "
                     f"{', '.join(self._availability.list_departments())}. Appointments are 30 minutes.")
            return ChatResponse(reply=reply, intent=q.intent, source=source, cached=source == "cache", latency_ms=0)
        return ChatResponse(reply=HELP, intent=q.intent, source=source, cached=source == "cache", latency_ms=0)

    def _check_date(self, day: date) -> str | None:
        today = self._today()
        if day < today:
            return f"{_pretty(day)} is in the past. Which upcoming date should I check?"
        if day > today + timedelta(days=BOOKING_HORIZON_DAYS):
            return f"I can only see the schedule {BOOKING_HORIZON_DAYS} days ahead (up to {_pretty(today + timedelta(days=BOOKING_HORIZON_DAYS))})."
        return None

    def _availability_answer(self, q: ParsedQuery, source: str) -> ChatResponse:
        day = q.date or self._today()
        problem = self._check_date(day)
        if problem:
            return ChatResponse(reply=problem, intent=q.intent, date=day.isoformat(), source=source,
                                cached=False, latency_ms=0)

        query = AvailabilityQuery(day=day, department=q.department, doctor=q.doctor, time_of_day=q.time_of_day)
        groups, data_cached = self._availability.get(query)
        scope_parts = [q.doctor or q.department, q.time_of_day if q.time_of_day != "any" else None]
        scope = ", ".join(p for p in scope_parts if p)
        scope_text = f" ({scope})" if scope else ""

        if groups:
            total = sum(len(g.slots) for g in groups)
            lines = [f"{total} free slot{'s' if total != 1 else ''} on {_pretty(day)}{scope_text}:"]
            for g in groups:
                shown = ", ".join(g.slots[:MAX_SLOTS_IN_TEXT])
                more = f" (+{len(g.slots) - MAX_SLOTS_IN_TEXT} more)" if len(g.slots) > MAX_SLOTS_IN_TEXT else ""
                lines.append(f"• {g.doctor}, {g.department}: {shown}{more}")
            reply = "\n".join(lines)
        else:
            nxt = self._next_available(query)
            reply = f"No free slots on {_pretty(day)}{scope_text}."
            if nxt:
                reply += f" The next opening is {_pretty(nxt[0])}: {nxt[1].doctor} at {nxt[1].slots[0]}."

        return ChatResponse(
            reply=reply, headline=reply.split("\n", 1)[0], intent=q.intent, date=day.isoformat(),
            results=[SlotGroup(doctor=g.doctor, department=g.department, slots=list(g.slots)) for g in groups],
            source=source, cached=source == "cache" and data_cached, latency_ms=0,
        )

    def _next_available(self, query: AvailabilityQuery) -> tuple[date, DoctorSlots] | None:
        last = self._today() + timedelta(days=BOOKING_HORIZON_DAYS)
        day = query.day + timedelta(days=1)
        while day <= last:
            groups, _ = self._availability.get(AvailabilityQuery(day, query.department, query.doctor, query.time_of_day))
            if groups:
                return day, groups[0]
            day += timedelta(days=1)
        return None

    def _risk_answer(self, q: ParsedQuery, source: str) -> ChatResponse:
        day = q.date or self._today()
        problem = self._check_date(day)
        if problem:
            return ChatResponse(reply=problem, intent=q.intent, date=day.isoformat(), source=source,
                                cached=False, latency_ms=0)
        df = load_booked(self._conn, day=day.isoformat())
        if not df.empty and q.department:
            df = df[df["department"] == q.department]
        if not df.empty and q.doctor:
            df = df[df["doctor"] == q.doctor]
        risks = []
        if not df.empty:
            preds = self._predictor.predict(df)
            for row, p in zip(df.itertuples(), preds, strict=True):
                if p.risk == "high":
                    risks.append(AppointmentRisk(
                        appointment_id=int(row.appointment_id), start_time=row.start_time,
                        patient_name=row.patient_name, doctor=row.doctor, department=row.department,
                        sms_reminder=bool(row.sms_reminder), no_show_probability=p.probability,
                        risk=p.risk, reasons=list(p.reasons)))
        risks.sort(key=lambda r: r.no_show_probability, reverse=True)

        booked = 0 if df.empty else len(df)
        if risks:
            headline = (f"{len(risks)} of {booked} appointments on {_pretty(day)} are high risk of a no-show. "
                        "A reminder call could help.")
            top = ", ".join(f"{datetime.fromisoformat(r.start_time):%H:%M} {r.patient_name} ({r.no_show_probability:.0%})"
                            for r in risks[:5])
            reply = f"{headline} Highest: {top}."
        else:
            headline = reply = f"None of the {booked} appointments on {_pretty(day)} are high risk."
        return ChatResponse(reply=reply, headline=headline, intent=q.intent, date=day.isoformat(), risks=risks,
                            source=source, cached=False, latency_ms=0)
