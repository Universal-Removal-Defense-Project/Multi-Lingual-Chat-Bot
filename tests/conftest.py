"""Shared test fixtures.

Tests must never touch a developer's real data or call the real model API, so
this module (which pytest imports before any test module) redirects the app's
data directory to a temporary one and blocks network access by default.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Must be set before the app modules are imported, since they read it once.
DATA_DIR = Path(tempfile.mkdtemp(prefix="urdp-test-data-"))
os.environ["URDP_DATA_DIR"] = str(DATA_DIR)


@pytest.fixture(autouse=True)
def _clean_data_dir():
    """Give every test an empty data directory (conversations, settings, KB)."""
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    yield
    shutil.rmtree(DATA_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Make any unmocked model call fail fast instead of reaching the Groq API.

    Tests that need a model response patch ``backend.get_client`` (or the
    function that uses it) themselves, which overrides this default.
    """
    import backend

    def offline_client(api_key=None):
        raise backend.BackendError("Network access is disabled in tests")

    monkeypatch.setattr(backend, "get_client", offline_client)
