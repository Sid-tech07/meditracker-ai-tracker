"""Generate the simulated MediTrack dataset into SQLite.

Creates doctors, patients, ~6 months of historical appointments labelled
completed / no_show, and bookings for the next 14 days (status 'booked').

No-show labels are NOT random: they come from a known logistic process so the
model has a real signal to learn and we can reason about what it should find.
Risk goes up with long booking lead time, a patient's hidden "unreliability",
distance, younger age, Mondays and early/late slots; an SMS reminder lowers it.

Usage:
    python -m scripts.generate_data                 # anchored on today
    python -m scripts.generate_data --anchor 2026-10-16 --seed 7
"""

import argparse
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from app.config import get_settings
from app.db import connect, init_schema
from app.schedule import slot_times, works_on

DOCTORS = [
    # name, department, working days (Mon=0)
    ("Dr. Ananya Rao", "General Medicine", "0,1,2,3,4"),
    ("Dr. Vikram Mehta", "General Medicine", "1,2,3,4,5"),
    ("Dr. Priya Nair", "Pediatrics", "0,1,2,3,4"),
    ("Dr. Rohan Kulkarni", "Cardiology", "0,2,4"),
    ("Dr. Sara Thomas", "Dermatology", "1,3,5"),
    ("Dr. Arjun Iyer", "Orthopedics", "0,1,3,4"),
    ("Dr. Meera Joshi", "Pediatrics", "2,3,4,5"),
    ("Dr. Kabir Sen", "Cardiology", "1,3,5"),
]

FIRST_NAMES = [
    "Aarav", "Diya", "Ishaan", "Kavya", "Reyansh", "Saanvi", "Vihaan", "Anika",
    "Aditya", "Myra", "Arnav", "Riya", "Krishna", "Tara", "Dev", "Nisha",
    "Kunal", "Pooja", "Rahul", "Sneha", "Yash", "Zara", "Omkar", "Leela",
]
LAST_NAMES = [
    "Sharma", "Patil", "Deshmukh", "Gupta", "Reddy", "Das", "Pillai", "Khan",
    "Bose", "Chopra", "Jain", "Menon", "Shetty", "Verma", "Naidu", "Ghosh",
]

HISTORY_DAYS = 180
FUTURE_DAYS = 14
N_PATIENTS = 900
HISTORY_FILL = 0.70
FUTURE_FILL = 0.55


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def no_show_probability(
    lead_days: int, unreliability: float, age: int, distance_km: float,
    sms: bool, start: datetime,
) -> float:
    """The 'true' process used to label history (unknown to the model)."""
    logit = -2.4
    logit += 0.045 * min(lead_days, 45)
    logit += 0.9 * unreliability
    logit += 0.015 * max(0, 35 - age)
    logit += 0.035 * distance_km
    logit -= 0.7 if sms else 0.0
    logit += 0.25 if start.weekday() == 0 else 0.0
    logit += 0.2 if start.hour == 9 or start.hour >= 16 else 0.0
    return _sigmoid(logit)


def _lead_days(rng: random.Random) -> int:
    return min(int(rng.expovariate(1 / 9)), 60)


def generate(db_path: Path, anchor: date, seed: int = 42) -> dict:
    rng = random.Random(seed)
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    init_schema(conn)

    conn.executemany(
        "INSERT INTO doctors (id, name, department, working_days) VALUES (?, ?, ?, ?)",
        [(i + 1, *d) for i, d in enumerate(DOCTORS)],
    )

    patients = []
    for pid in range(1, N_PATIENTS + 1):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        age = max(1, min(90, int(rng.gauss(38, 18))))
        distance = round(min(rng.expovariate(1 / 6), 40), 1)
        unreliability = rng.gauss(0, 1)  # hidden trait, never stored
        patients.append((pid, name, age, distance, unreliability))
    conn.executemany(
        "INSERT INTO patients (id, name, age, distance_km) VALUES (?, ?, ?, ?)",
        [p[:4] for p in patients],
    )

    rows = []
    start_day = anchor - timedelta(days=HISTORY_DAYS)
    for offset in range(HISTORY_DAYS + FUTURE_DAYS):
        day = start_day + timedelta(days=offset)
        is_future = day >= anchor
        fill = FUTURE_FILL if is_future else HISTORY_FILL
        for doc_id, (_, _, working_days) in enumerate(DOCTORS, start=1):
            if not works_on(working_days, day):
                continue
            for slot in slot_times(day):
                if rng.random() > fill:
                    continue
                pid, _, age, distance, unreliability = rng.choice(patients)
                lead = _lead_days(rng)
                if is_future:
                    # booked on or before today: lead = days until the visit + days since booking
                    lead = (day - anchor).days + lead // 2
                booked_at = slot - timedelta(days=lead, hours=rng.randint(1, 8))
                sms = lead >= 2 and rng.random() < 0.6
                if is_future:
                    status = "booked"
                else:
                    p = no_show_probability(lead, unreliability, age, distance, sms, slot)
                    status = "no_show" if rng.random() < p else "completed"
                rows.append((pid, doc_id, slot.isoformat(timespec="minutes"),
                             booked_at.isoformat(timespec="minutes"), int(sms), status))

    conn.executemany(
        "INSERT INTO appointments (patient_id, doctor_id, start_time, booked_at, sms_reminder, status)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    counts = {
        status: conn.execute("SELECT COUNT(*) FROM appointments WHERE status = ?", (status,)).fetchone()[0]
        for status in ("completed", "no_show", "booked")
    }
    conn.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--anchor", type=date.fromisoformat, default=date.today(),
                        help="'today' for the dataset (default: real today)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=Path, default=get_settings().database_path)
    args = parser.parse_args()

    counts = generate(args.db, args.anchor, args.seed)
    history = counts["completed"] + counts["no_show"]
    print(f"Wrote {args.db}")
    print(f"  history: {history} appointments, {counts['no_show']} no-shows "
          f"({counts['no_show'] / history:.1%})")
    print(f"  upcoming bookings: {counts['booked']}")


if __name__ == "__main__":
    main()
