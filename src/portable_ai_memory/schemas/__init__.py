"""Bundled PAM JSON Schemas for validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent

MEMORY_STORE_SCHEMA_PATH = _DIR / "portable-ai-memory.schema.json"
CONVERSATION_SCHEMA_PATH = _DIR / "portable-ai-memory-conversation.schema.json"
EMBEDDINGS_SCHEMA_PATH = _DIR / "portable-ai-memory-embeddings.schema.json"


def load_schema(name: str) -> dict[str, Any]:
    """Load a bundled JSON schema by document type name.

    Args:
        name: One of 'memory-store', 'conversation', 'embeddings'.
    """
    paths = {
        "memory-store": MEMORY_STORE_SCHEMA_PATH,
        "conversation": CONVERSATION_SCHEMA_PATH,
        "embeddings": EMBEDDINGS_SCHEMA_PATH,
    }
    path = paths.get(name)
    if path is None:
        msg = f"Unknown schema: '{name}'. Expected one of: {list(paths.keys())}"
        raise ValueError(msg)
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
