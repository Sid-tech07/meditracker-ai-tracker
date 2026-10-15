from datetime import date, datetime, timedelta

from app.services.availability import (
    AvailabilityQuery,
    AvailabilityService,
    find_available_slots,
)

MONDAY = date(2026, 10, 19)
BEFORE_DATA = datetime(2026, 10, 1, 8, 0)  # "now" earlier than every queried slot


def test_free_slots_exclude_booked_appointments(conn):
    results = find_available_slots(conn, AvailabilityQuery(day=MONDAY), now=BEFORE_DATA)
    assert results, "expected at least one doctor with free slots"
    booked = {
        r["start_time"][11:]
        for r in conn.execute(
            "SELECT a.start_time FROM appointments a JOIN doctors d ON d.id = a.doctor_id"
            " WHERE d.name = ? AND a.status = 'booked' AND a.start_time LIKE ?",
            (results[0].doctor, f"{MONDAY.isoformat()}%"),
        )
    }
    assert not booked & set(results[0].slots)


def test_department_filter_and_working_days(conn):
    # Cardiology: Dr. Kulkarni works Mon/Wed/Fri, Dr. Sen Tue/Thu/Sat
    results = find_available_slots(conn, AvailabilityQuery(day=MONDAY, department="cardio"), now=BEFORE_DATA)
    assert {r.doctor for r in results} <= {"Dr. Rohan Kulkarni"}
    assert all(r.department == "Cardiology" for r in results)


def test_doctor_name_is_matched_loosely(conn):
    for name in ("Dr. Ananya Rao", "rao", "Dr Rao"):
        results = find_available_slots(conn, AvailabilityQuery(day=MONDAY, doctor=name), now=BEFORE_DATA)
        assert [r.doctor for r in results] == ["Dr. Ananya Rao"]


def test_no_slots_on_sunday(conn):
    sunday = date(2026, 10, 18)
    assert find_available_slots(conn, AvailabilityQuery(day=sunday), now=BEFORE_DATA) == []


def test_lunch_and_time_of_day(conn):
    morning = find_available_slots(conn, AvailabilityQuery(day=MONDAY, time_of_day="morning"), now=BEFORE_DATA)
    afternoon = find_available_slots(conn, AvailabilityQuery(day=MONDAY, time_of_day="afternoon"), now=BEFORE_DATA)
    all_morning = [s for r in morning for s in r.slots]
    all_afternoon = [s for r in afternoon for s in r.slots]
    assert all(s < "13:00" for s in all_morning)
    assert all(s >= "14:00" for s in all_afternoon)
    assert not any(s.startswith("13:") for s in all_morning + all_afternoon)


def test_past_slots_are_hidden(conn):
    now = datetime.combine(MONDAY, datetime.min.time()) + timedelta(hours=15)
    results = find_available_slots(conn, AvailabilityQuery(day=MONDAY), now=now)
    assert all(s > "15:00" for r in results for s in r.slots)


def test_unknown_department_returns_empty(conn):
    assert find_available_slots(conn, AvailabilityQuery(day=MONDAY, department="'; DROP TABLE doctors;--"),
                                now=BEFORE_DATA) == []
    # table still there: parameterised query, no injection
    assert conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] == 8


def test_cache_hit_on_repeat_and_normalised_key(conn, monkeypatch):
    import app.services.availability as availability

    calls = []
    real = availability.find_available_slots

    def counting(c, q, now=None):
        calls.append(q)
        return real(c, q, now=BEFORE_DATA)

    monkeypatch.setattr(availability, "find_available_slots", counting)
    service = AvailabilityService(conn, ttl_seconds=60)

    first, cached1 = service.get(AvailabilityQuery(day=MONDAY, department="Pediatrics"))
    second, cached2 = service.get(AvailabilityQuery(day=MONDAY, department="  pediatrics "))
    assert (cached1, cached2) == (False, True)
    assert first == second
    assert len(calls) == 1

    service.invalidate()
    _, cached3 = service.get(AvailabilityQuery(day=MONDAY, department="Pediatrics"))
    assert cached3 is False
