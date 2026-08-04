"""Shared test fixtures."""

import shutil
from pathlib import Path

import pytest

DATA_DIR = Path("data")


@pytest.fixture(autouse=True)
def _clean_data_dir():
    """Isolate the on-disk stores (conversations + knowledge DB) between tests."""
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    yield
    shutil.rmtree(DATA_DIR, ignore_errors=True)
