"""Shared pytest fixtures and configuration."""

import pytest


# Suppress ResourceWarnings from unclosed SQLite connections in test teardown.
# Fixtures yield GraphVec instances and close them after each test.
# This prevents false positives from Python's gc-based ResourceWarning.
@pytest.fixture(autouse=True)
def _suppress_resource_warnings():
    """Silence ResourceWarning for unclosed sqlite3.Connection objects in tests."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        yield
