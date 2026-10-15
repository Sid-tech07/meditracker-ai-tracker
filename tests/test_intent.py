from datetime import date

import pytest

from app.services.intent import GeminiIntentParser, ParseContext, RuleBasedParser
from tests.conftest import ANCHOR  # Friday 16 Oct 2026

CTX = ParseContext(
    today=ANCHOR,
    departments=["Cardiology", "Dermatology", "General Medicine", "Orthopedics", "Pediatrics"],
    doctors=["Dr. Ananya Rao", "Dr. Rohan Kulkarni", "Dr. Priya Nair", "Dr. Sara Thomas"],
)
rules = RuleBasedParser()


@pytest.mark.parametrize("text,expected", [
    ("any slots today?", date(2026, 10, 16)),
    ("free slots tomorrow", date(2026, 10, 17)),
    ("day after tomorrow please", date(2026, 10, 18)),
    ("availability on monday", date(2026, 10, 19)),
    ("is there anything on wed", date(2026, 10, 21)),
    ("slots on 2026-10-22", date(2026, 10, 22)),
    ("slots on 23 oct", date(2026, 10, 23)),
    ("slots on October 24th", date(2026, 10, 24)),
    ("slots on 26/10", date(2026, 10, 26)),
    ("next friday", date(2026, 10, 23)),
    ("any free slots", None),
])
def test_dates(text, expected):
    assert rules.parse(text, CTX).date == expected


@pytest.mark.parametrize("text,intent", [
    ("Any cardiology slots tomorrow?", "availability"),
    ("When can Dr. Rao see a patient?", "availability"),
    ("my kid has a rash", "availability"),
    ("which appointments are high risk tomorrow", "no_show_risk"),
    ("who might not show up on monday", "no_show_risk"),
    ("what are your clinic hours", "clinic_info"),
    ("are you closed on sunday", "clinic_info"),
    ("tell me a joke", "unsupported"),
])
def test_intents(text, intent):
    assert rules.parse(text, CTX).intent == intent


def test_department_synonyms_and_doctor():
    assert rules.parse("heart specialist on monday", CTX).department == "Cardiology"
    assert rules.parse("skin doctor", CTX).department == "Dermatology"
    assert rules.parse("is kulkarni free", CTX).doctor == "Dr. Rohan Kulkarni"


def test_time_of_day():
    assert rules.parse("slots tomorrow morning", CTX).time_of_day == "morning"
    assert rules.parse("anything in the evening", CTX).time_of_day == "afternoon"
    assert rules.parse("slots tomorrow", CTX).time_of_day == "any"


# --- LLM output validation (no network) -------------------------------------


def test_llm_json_is_validated_and_normalised():
    q = GeminiIntentParser._to_query(
        '{"intent": "availability", "date": "2026-10-19", "department": "cardio", '
        '"doctor": "Dr Rao", "time_of_day": "morning"}', CTX)
    assert q.date == date(2026, 10, 19)
    assert q.department == "Cardiology"
    assert q.doctor == "Dr. Ananya Rao"


def test_llm_unknown_names_are_dropped_not_trusted():
    q = GeminiIntentParser._to_query('{"intent": "availability", "department": "Neurology", "doctor": "Dr. Who"}', CTX)
    assert q.department is None and q.doctor is None


@pytest.mark.parametrize("raw", [
    "not json", "", None, '{"intent": "book_flight"}', '{"intent": "availability", "date": "next tuesday"}',
])
def test_llm_bad_output_raises(raw):
    with pytest.raises(ValueError):
        GeminiIntentParser._to_query(raw, CTX)
