"""Shared fixtures: a small deterministic database per test session."""

from datetime import date

import pytest

from app.db import connect
from scripts.generate_data import generate

ANCHOR = date(2026, 10, 16)  # a Friday


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test.db"
    generate(path, ANCHOR, seed=1)
    return path


@pytest.fixture
def conn(db_path):
    c = connect(db_path)
    yield c
    c.close()
