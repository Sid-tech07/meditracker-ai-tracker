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
from app.services.chat_log import ChatLogger
from app.services.chatbot import ChatService
from app.services.intent import GeminiIntentParser, IntentParser
from app.services.predictor import NoShowPredictor

log = logging.getLogger("meditrack")


@dataclass
class Services:
    settings: Settings
    conn: sqlite3.Connection
    availability: AvailabilityService
    predictor: NoShowPredictor
    chat_logger: ChatLogger
    chat: ChatService


def bootstrap(settings: Settings, model_path: Path) -> None:
    """First run convenience: create the dataset and model if they're missing."""
    if not Path(settings.database_path).exists():
        from scripts.generate_data import generate

        log.info("No database found, generating mock data at %s", settings.database_path)
        generate(settings.database_path, date.today())
    else:
        _warn_if_stale(settings.database_path)
    if not model_path.exists():
        from ml.train import train

        log.info("No trained model found, training one")
        train(settings.database_path, model_path=model_path, metrics_path=model_path.with_name("metrics.json"))


def _warn_if_stale(db_path: Path) -> None:
    """The mock schedule is anchored on the day it was generated; warn once it runs out."""
    with sqlite3.connect(db_path) as conn:
        last = conn.execute("SELECT MAX(start_time) FROM appointments WHERE status = 'booked'").fetchone()[0]
    if last is None or last[:10] < date.today().isoformat():
        log.warning("Mock schedule ends %s, before today. Regenerate it: python -m scripts.generate_data "
                    "&& python -m ml.train", last)


def build_llm_parser(settings: Settings) -> IntentParser | None:
    if not settings.llm_enabled:
        log.warning("GEMINI_API_KEY not set: chatbot will use the rule-based parser only")
        return None
    log.info("Chatbot using Gemini model %s", settings.gemini_model)
    return GeminiIntentParser(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds)


def build_services(settings: Settings, model_path: Path, llm_parser: IntentParser | None = None) -> Services:
    conn = connect(settings.database_path)
    init_schema(conn)  # idempotent; makes sure chat_logs exists
    availability = AvailabilityService(conn, settings.cache_ttl_seconds, settings.cache_max_entries)
    predictor = NoShowPredictor(model_path)
    chat_logger = ChatLogger(conn)
    parser = llm_parser if llm_parser is not None else build_llm_parser(settings)
    chat = ChatService(availability, predictor, chat_logger, conn, llm_parser=parser,
                       cache_ttl_seconds=settings.cache_ttl_seconds)
    return Services(settings, conn, availability, predictor, chat_logger, chat)


def get_services(request: Request) -> Services:
    return request.app.state.services
