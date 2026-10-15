"""Builds the long-lived services once at startup and hands them to routes."""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fastapi import Request

from app.config import Settings
from app.db import connect, init_schema
from app.services.availability import AvailabilityService
from app.services.predictor import NoShowPredictor

log = logging.getLogger("meditrack")


@dataclass
class Services:
    settings: Settings
    conn: sqlite3.Connection
    availability: AvailabilityService
    predictor: NoShowPredictor


def bootstrap(settings: Settings, model_path: Path) -> None:
    """First run convenience: create the dataset and model if they're missing."""
    if not Path(settings.database_path).exists():
        from scripts.generate_data import generate

        log.info("No database found, generating mock data at %s", settings.database_path)
        generate(settings.database_path, date.today())
    if not model_path.exists():
        from ml.train import train

        log.info("No trained model found, training one")
        train(settings.database_path, model_path=model_path, metrics_path=model_path.with_name("metrics.json"))


def build_services(settings: Settings, model_path: Path) -> Services:
    conn = connect(settings.database_path)
    init_schema(conn)  # idempotent; makes sure chat_logs exists
    return Services(
        settings=settings,
        conn=conn,
        availability=AvailabilityService(conn, settings.cache_ttl_seconds, settings.cache_max_entries),
        predictor=NoShowPredictor(model_path),
    )


def get_services(request: Request) -> Services:
    return request.app.state.services
