"""Read and write PAM files.

Provides type-safe loading and serialization of PAM documents.
Auto-detects document type from the 'schema' field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError as _PydanticValidationError

from portable_ai_memory.exceptions import PAMSchemaError, PAMValidationError
from portable_ai_memory.models.conversation import Conversation
from portable_ai_memory.models.embeddings import EmbeddingsFile
from portable_ai_memory.models.memory_store import MemoryStore

PAMDocument = MemoryStore | Conversation | EmbeddingsFile

_SCHEMA_MAP: dict[str, type[PAMDocument]] = {
    "portable-ai-memory": MemoryStore,
    "portable-ai-memory-conversation": Conversation,
    "portable-ai-memory-embeddings": EmbeddingsFile,
}


def load(path: str | Path) -> PAMDocument:
    """Load a PAM file, auto-detecting the document type.

    Args:
        path: Path to a PAM JSON file.

    Returns:
        A typed PAM document (MemoryStore, Conversation, or EmbeddingsFile).

    Raises:
        FileNotFoundError: If the file does not exist.
        PAMSchemaError: If the schema type cannot be detected.
        PAMValidationError: If the file does not conform to the schema.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {path}: {e}"
        raise PAMSchemaError(msg) from e
    return load_dict(raw)


def load_dict(data: dict[str, Any]) -> PAMDocument:
    """Load a PAM document from a dict, auto-detecting the document type.

    Args:
        data: Parsed JSON dict with a 'schema' field.

    Returns:
        A typed PAM document.

    Raises:
        PAMSchemaError: If the schema type cannot be detected.
        PAMValidationError: If the data does not conform.
    """
    schema_id = data.get("schema")
    if schema_id is None:
        msg = "Missing 'schema' field — cannot determine document type"
        raise PAMSchemaError(msg)

    model_class = _SCHEMA_MAP.get(schema_id)
    if model_class is None:
        msg = f"Unknown schema: '{schema_id}'. Expected one of: {list(_SCHEMA_MAP.keys())}"
        raise PAMSchemaError(msg)

    try:
        return model_class.model_validate(data)
    except _PydanticValidationError as e:
        raise PAMValidationError(str(e), pydantic_error=e) from e


def load_memory_store(path: str | Path) -> MemoryStore:
    """Load a PAM Memory Store file.

    Args:
        path: Path to a memory store JSON file.

    Returns:
        A validated MemoryStore instance.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return MemoryStore.model_validate(raw)


def load_conversation(path: str | Path) -> Conversation:
    """Load a PAM Conversation file.

    Args:
        path: Path to a conversation JSON file.

    Returns:
        A validated Conversation instance.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Conversation.model_validate(raw)


def load_embeddings(path: str | Path) -> EmbeddingsFile:
    """Load a PAM Embeddings file.

    Args:
        path: Path to an embeddings JSON file.

    Returns:
        A validated EmbeddingsFile instance.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EmbeddingsFile.model_validate(raw)


def save(document: PAMDocument, path: str | Path, *, indent: int = 2) -> Path:
    """Save a PAM document to a JSON file.

    Args:
        document: A PAM document instance.
        path: Output file path.
        indent: JSON indentation (default: 2).

    Returns:
        The Path where the file was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = document.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    path.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def to_dict(document: PAMDocument) -> dict[str, Any]:
    """Serialize a PAM document to a JSON-compatible dict.

    Uses camelCase aliases and excludes None values, matching the PAM JSON Schema.

    Args:
        document: A PAM document instance.

    Returns:
        A dict ready for ``json.dumps()``.
    """
    return document.model_dump(mode="json", by_alias=True, exclude_none=True)
