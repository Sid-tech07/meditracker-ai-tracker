"""FastAPI entry point for MediTrack."""

from fastapi import FastAPI

app = FastAPI(
    title="MediTrack",
    description="Receptionist assistant: appointment availability chatbot and no-show risk.",
    version="0.1.0",
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
