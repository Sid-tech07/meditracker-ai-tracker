from datetime import date, datetime

import pytest

from app.services.availability import AvailabilityService
from app.services.chat_log import ChatLogger
from app.services.chatbot import ChatService
from app.services.intent import ParsedQuery
from app.services.predictor import NoShowPredictor
from tests.conftest import ANCHOR, MONDAY

EARLY = datetime(2026, 10, 16, 8, 0)


class FakeLLM:
    name = "llm"

    def __init__(self, result: ParsedQuery | None = None, error: Exception | None = None):
        self.result, self.error, self.calls = result, error, 0

    def parse(self, message, ctx):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def make_chat(conn, model_path):
    def _make(llm=None):
        availability = AvailabilityService(conn, clock=lambda: EARLY)
        return ChatService(availability, NoShowPredictor(model_path), ChatLogger(conn), conn,
                           llm_parser=llm, today=lambda: ANCHOR)
    return _make


def _last_log(conn):
    return conn.execute("SELECT * FROM chat_logs ORDER BY id DESC LIMIT 1").fetchone()


def test_availability_reply_is_grounded_in_data(make_chat, conn):
    llm = FakeLLM(ParsedQuery(intent="availability", date=MONDAY, department="Cardiology"))
    res = make_chat(llm).handle("cardiology on monday?")
    assert res.source == "llm" and res.intent == "availability"
    assert res.results and all(g.department == "Cardiology" for g in res.results)
    # every slot in the reply text is a real free slot from the results
    shown = {s for g in res.results for s in g.slots}
    for token in res.reply.replace(",", " ").split():
        if len(token) == 5 and token[2] == ":":
            assert token in shown


def test_llm_failure_falls_back_to_rules_and_logs_error(make_chat, conn):
    llm = FakeLLM(error=TimeoutError("deadline exceeded"))
    res = make_chat(llm).handle("any pediatrics slots on monday")
    assert res.source == "fallback"
    assert res.intent == "availability" and res.date == MONDAY.isoformat()
    row = _last_log(conn)
    assert "TimeoutError" in row["error"] and row["source"] == "fallback"


def test_no_api_key_uses_rules(make_chat):
    res = make_chat(llm=None).handle("free slots tomorrow morning")
    assert res.source == "fallback" and res.date == "2026-10-17"
    assert all(s < "13:00" for g in res.results for s in g.slots)


def test_every_interaction_is_logged_with_timestamp_and_query(make_chat, conn):
    before = conn.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0]
    make_chat().handle("tell me a joke")
    row = _last_log(conn)
    assert conn.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0] == before + 1
    assert row["query"] == "tell me a joke"
    assert datetime.fromisoformat(row["timestamp"])  # valid ISO timestamp
    assert row["intent"] == "unsupported"


def test_repeat_question_hits_cache_and_skips_llm(make_chat):
    llm = FakeLLM(ParsedQuery(intent="availability", date=MONDAY))
    chat = make_chat(llm)
    first = chat.handle("Slots on Monday?")
    second = chat.handle("  slots   on monday? ")
    assert llm.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.source == "cache" and second.results == first.results


def test_past_and_far_future_dates_are_refused(make_chat):
    chat = make_chat(FakeLLM(ParsedQuery(intent="availability", date=date(2026, 10, 1))))
    assert "in the past" in chat.handle("slots on 1 oct").reply
    chat = make_chat(FakeLLM(ParsedQuery(intent="availability", date=date(2026, 12, 25))))
    assert "14 days ahead" in chat.handle("slots on christmas").reply


def test_no_slots_suggests_next_opening(make_chat):
    sunday = date(2026, 10, 18)
    res = make_chat(FakeLLM(ParsedQuery(intent="availability", date=sunday))).handle("sunday?")
    assert res.results == []
    assert "No free slots" in res.reply and "next opening is Mon 19 Oct" in res.reply


def test_risk_question_lists_high_risk_appointments(make_chat):
    res = make_chat().handle("which appointments on monday are high risk of no-show")
    assert res.intent == "no_show_risk"
    assert all(r.risk == "high" for r in res.risks)
    probs = [r.no_show_probability for r in res.risks]
    assert probs == sorted(probs, reverse=True)


def test_unexpected_error_returns_friendly_reply_and_is_logged(make_chat, conn, monkeypatch):
    chat = make_chat(FakeLLM(ParsedQuery(intent="availability", date=MONDAY)))
    monkeypatch.setattr(chat, "_availability_answer", lambda *a: (_ for _ in ()).throw(RuntimeError("db gone")))
    res = chat.handle("slots monday")
    assert res.intent == "error" and "something went wrong" in res.reply
    assert "db gone" in _last_log(conn)["error"]


# --- API --------------------------------------------------------------------


def test_chat_endpoint_and_logs(client):
    res = client.post("/api/chat", json={"message": "what are your clinic hours?"})
    assert res.status_code == 200
    assert res.json()["intent"] == "clinic_info"
    logs = client.get("/api/chat/logs", params={"limit": 1}).json()
    assert logs[0]["query"] == "what are your clinic hours?"


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "x" * 501}, {"message": 42}])
def test_chat_endpoint_validates_input(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422
