"""Pydantic models for the PAM Embeddings schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from portable_ai_memory.models._config import STRICT
from portable_ai_memory.models.enums import StorageType


class EmbeddingStorage(BaseModel):
    """Reference to where an embedding vector is stored externally.

    Attributes:
        type: Storage type (file, url, inline).
        ref: Storage reference (file path or URL).
    """

    model_config = STRICT

    type: StorageType
    ref: str = Field(min_length=1)


class EmbeddingObject(BaseModel):
    """A single embedding vector associated with a memory."""

    model_config = STRICT

    id: str = Field(min_length=1)
    memory_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    created_at: datetime
    vector: list[float] | None = None
    storage: EmbeddingStorage | None = None


class EmbeddingsFile(BaseModel):
    """Root PAM Embeddings document."""

    model_config = STRICT

    schema_: str = Field("portable-ai-memory-embeddings", alias="schema")
    schema_version: str = Field(pattern=r"^[0-9]+\.[0-9]+(-(rc|alpha|beta)[0-9]*)?$")
    embeddings: list[EmbeddingObject]
