"""Request/response models. Pydantic validates every input at the API edge."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

RiskBand = Literal["low", "medium", "high"]


class PredictRequest(BaseModel):
    """Score a hypothetical appointment (what-if), without it being in the database."""

    lead_time_days: float = Field(ge=0, le=365, description="Days between booking and appointment")
    age: int = Field(ge=0, le=120)
    distance_km: float = Field(ge=0, le=500)
    prior_visits: int = Field(default=0, ge=0, le=1000)
    prior_no_shows: int = Field(default=0, ge=0, le=1000)
    sms_reminder: bool = False
    weekday: int = Field(ge=0, le=6, description="Mon=0 ... Sun=6")
    hour: int = Field(ge=0, le=23)
    department: str = Field(min_length=2, max_length=50)

    @model_validator(mode="after")
    def no_shows_within_visits(self):
        if self.prior_no_shows > self.prior_visits:
            raise ValueError("prior_no_shows cannot exceed prior_visits")
        return self


class PredictionOut(BaseModel):
    no_show_probability: float = Field(ge=0, le=1)
    risk: RiskBand
    reasons: list[str]
    model: str


class AppointmentRisk(BaseModel):
    appointment_id: int
    start_time: str
    patient_name: str
    doctor: str
    department: str
    sms_reminder: bool
    no_show_probability: float
    risk: RiskBand
    reasons: list[str]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class SlotGroup(BaseModel):
    doctor: str
    department: str
    slots: list[str]


class ChatResponse(BaseModel):
    reply: str
    intent: str
    date: str | None = None
    results: list[SlotGroup] = []
    risks: list[AppointmentRisk] = []
    source: Literal["llm", "fallback", "cache"]
    cached: bool
    latency_ms: float
