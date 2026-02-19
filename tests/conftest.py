"""Shared test fixtures and real data helpers.

Real Data Tests
===============
Some tests use actual provider export files. Set the ``PAM_TEST_DATA_DIR``
environment variable to a directory with the following structure::

    $PAM_TEST_DATA_DIR/
    ├── chatgpt/
    │   └── conversations.json          # ChatGPT export (required)
    │   └── user.json                   # (optional) user profile
    │   └── shared_conversations.json   # (optional) shared links
    ├── claude/
    │   └── conversations.json          # Claude export (required)
    │   └── memories.json               # (optional) memories
    │   └── users.json                  # (optional) user profile
    │   └── projects.json               # (optional) project docs
    ├── grok/
    │   └── <user>/export_data/<id>/
    │       └── prod-grok-backend.json  # Grok backend export (required)
    │       └── prod-mc-auth-mgmt-api.json  # (optional) auth/profile
    ├── copilot/
    │   └── *.csv                       # Copilot CSV exports (at least one)
    │       Supported formats:
    │       - Format A: Conversation, Time, Author, Message
    │       - Format B: CreatedAt, MessageContent, Author, ChatName
    │       - Format C: Timestamp, ClientApp, Prompt (Windows apps)
    └── gemini/
        └── *.html                      # Gemini Takeout HTML (at least one)

If ``PAM_TEST_DATA_DIR`` is set but a provider subdirectory is missing,
a warning is printed at collection time so missing data doesn't go unnoticed.

Tests that require real data are decorated with ``@skip_no_real_data`` and
will be skipped entirely when the env var is not set.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.factories import make_memory, make_store

if TYPE_CHECKING:
    from portable_ai_memory.models.memory_store import MemoryObject, MemoryStore

# Expected provider subdirectories and the file/pattern that confirms data exists
_EXPECTED_PROVIDERS: dict[str, str] = {
    "chatgpt": "conversations.json",
    "claude": "conversations.json",
    "grok": "**/prod-grok-backend.json",
    "copilot": "*.csv",
    "gemini": "*.html",
}


def _real_data_dir() -> Path | None:
    """Return the real test data directory from env var, or None."""
    val = os.environ.get("PAM_TEST_DATA_DIR")
    if val:
        p = Path(val)
        if p.is_dir():
            return p
    return None


REAL_DATA_DIR: Path | None = _real_data_dir()


def _warn_missing_providers() -> None:
    """Emit warnings for expected provider subdirectories that are missing or empty."""
    if REAL_DATA_DIR is None:
        return
    for provider, pattern in _EXPECTED_PROVIDERS.items():
        subdir = REAL_DATA_DIR / provider
        if not subdir.exists():
            warnings.warn(
                f"PAM_TEST_DATA_DIR is set but '{provider}/' subdirectory is missing. "
                f"Tests for {provider} will be skipped.",
                stacklevel=1,
            )
        elif not list(subdir.glob(pattern)):
            warnings.warn(
                f"PAM_TEST_DATA_DIR/{provider}/ exists but no '{pattern}' found. "
                f"Tests for {provider} will be skipped.",
                stacklevel=1,
            )


_warn_missing_providers()


def real_data_path(subdir: str) -> Path:
    """Return path to a provider subdirectory inside the real data dir."""
    assert REAL_DATA_DIR is not None
    return REAL_DATA_DIR / subdir


skip_no_real_data = pytest.mark.skipif(
    REAL_DATA_DIR is None,
    reason="PAM_TEST_DATA_DIR not set or directory not found",
)


@pytest.fixture
def minimal_memory() -> MemoryObject:
    """A minimal valid memory object."""
    return make_memory(
        id="mem-001",
        content="User prefers dark mode in all applications.",
        provenance={"platform": "claude"},
    )


@pytest.fixture
def minimal_store(minimal_memory: MemoryObject) -> MemoryStore:
    """A minimal valid memory store."""
    return make_store(memories=[minimal_memory], owner={"id": "user-test"})
