"""Clinic schedule rules shared by the data generator and the availability service."""

from datetime import date, datetime, time, timedelta

SLOT_MINUTES = 30
OPEN_TIME = time(9, 0)
CLOSE_TIME = time(17, 0)
LUNCH_START = time(13, 0)
LUNCH_END = time(14, 0)


def slot_times(day: date) -> list[datetime]:
    """All bookable slot start times for one doctor-day (lunch excluded)."""
    slots = []
    current = datetime.combine(day, OPEN_TIME)
    end = datetime.combine(day, CLOSE_TIME)
    while current < end:
        if not (LUNCH_START <= current.time() < LUNCH_END):
            slots.append(current)
        current += timedelta(minutes=SLOT_MINUTES)
    return slots


def works_on(working_days: str, day: date) -> bool:
    return str(day.weekday()) in working_days.split(",")
