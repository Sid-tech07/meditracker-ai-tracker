"""JSON API routes."""

from datetime import date
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.container import Services, get_services
from app.features import load_booked
from app.schemas import AppointmentRisk, ChatRequest, ChatResponse, PredictionOut, PredictRequest, RiskBand
from app.services.predictor import NoShowPredictor

router = APIRouter(prefix="/api")
ServicesDep = Annotated[Services, Depends(get_services)]


def score_booked(services: Services, df: pd.DataFrame) -> list[AppointmentRisk]:
    predictions = services.predictor.predict(df)
    return [
        AppointmentRisk(
            appointment_id=int(row.appointment_id),
            start_time=row.start_time,
            patient_name=row.patient_name,
            doctor=row.doctor,
            department=row.department,
            sms_reminder=bool(row.sms_reminder),
            no_show_probability=p.probability,
            risk=p.risk,
            reasons=list(p.reasons),
        )
        for row, p in zip(df.itertuples(), predictions, strict=True)
    ]


@router.get("/health")
def health(services: ServicesDep) -> dict:
    return {"status": "ok", "model": services.predictor.name, "llm_enabled": services.settings.llm_enabled}


@router.get("/appointments/{appointment_id}/no-show-risk", response_model=AppointmentRisk)
def appointment_risk(appointment_id: int, services: ServicesDep) -> AppointmentRisk:
    df = load_booked(services.conn, appointment_ids=[appointment_id])
    if df.empty:
        raise HTTPException(404, f"No upcoming (booked) appointment with id {appointment_id}")
    return score_booked(services, df)[0]


@router.get("/appointments/upcoming", response_model=list[AppointmentRisk])
def upcoming(
    services: ServicesDep,
    day: Annotated[date | None, Query(alias="date", description="YYYY-MM-DD, defaults to today")] = None,
    risk: RiskBand | None = None,
) -> list[AppointmentRisk]:
    df = load_booked(services.conn, day=(day or date.today()).isoformat())
    scored = score_booked(services, df)
    if risk:
        scored = [a for a in scored if a.risk == risk]
    return scored


@router.post("/predict", response_model=PredictionOut)
def predict(body: PredictRequest, services: ServicesDep) -> PredictionOut:
    departments = services.availability.list_departments()
    if body.department not in departments:
        raise HTTPException(422, f"Unknown department. Use one of: {', '.join(departments)}")
    row = body.model_dump()
    row["sms_reminder"] = int(row["sms_reminder"])
    row["weekday"], row["hour"] = str(row["weekday"]), str(row["hour"])
    row["is_first_visit"] = int(row["prior_visits"] == 0)
    row["prior_no_show_rate"] = row["prior_no_shows"] / row["prior_visits"] if row["prior_visits"] else 0.0
    prediction = services.predictor.predict(pd.DataFrame([row]))[0]
    return PredictionOut(
        no_show_probability=prediction.probability,
        risk=prediction.risk,
        reasons=list(prediction.reasons),
        model=services.predictor.name,
    )


@router.get("/model")
def model_info(services: ServicesDep) -> dict:
    """Which model is serving and how it scored on the held-out period."""
    predictor: NoShowPredictor = services.predictor
    return predictor.metrics


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, services: ServicesDep) -> ChatResponse:
    """Answer an availability / no-show question. Sync on purpose: FastAPI runs it in
    a worker thread, so the blocking LLM call doesn't stall the event loop."""
    return services.chat.handle(body.message)


@router.get("/chat/logs")
def chat_logs(services: ServicesDep, limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[dict]:
    return services.chat_logger.recent(limit)
