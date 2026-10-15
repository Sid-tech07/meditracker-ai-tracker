"""Shared fixtures: a small deterministic database, a trained model and an API client."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import connect
from app.main import create_app
from ml.train import train
from scripts.generate_data import generate

ANCHOR = date(2026, 10, 16)  # a Friday
MONDAY = date(2026, 10, 19)


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test.db"
    generate(path, ANCHOR, seed=1)
    return path


@pytest.fixture(scope="session")
def model_path(db_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("ml")
    path = out / "model.joblib"
    train(db_path, model_path=path, metrics_path=out / "metrics.json")
    return path


@pytest.fixture
def conn(db_path):
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture
def settings(db_path):
    return Settings(database_path=db_path, gemini_api_key="", _env_file=None)


@pytest.fixture
def client(settings, model_path):
    app = create_app(settings, model_path=model_path, auto_bootstrap=False)
    with TestClient(app) as c:
        yield c
