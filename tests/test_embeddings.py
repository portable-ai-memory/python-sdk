"""Tests for embeddings model and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from portable_ai_memory.core.integrity import compute_content_hash
from portable_ai_memory.core.io import load_embeddings, load_memory_store
from portable_ai_memory.core.validator import validate_embeddings
from portable_ai_memory.models.embeddings import EmbeddingObject, EmbeddingsFile
from portable_ai_memory.models.memory_store import (
    MemoryObject,
    MemoryStore,
    Owner,
    ProvenanceBlock,
    TemporalBlock,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestEmbeddingsModel:
    def test_minimal_valid(self) -> None:
        emb = EmbeddingObject(
            id="emb-001",
            memory_id="mem-001",
            model="text-embedding-3-small",
            dimensions=3,
            created_at="2026-01-01T00:00:00Z",
            vector=[0.1, 0.2, 0.3],
        )
        assert emb.dimensions == 3

    def test_without_vector(self) -> None:
        emb = EmbeddingObject(
            id="emb-001",
            memory_id="mem-001",
            model="text-embedding-3-small",
            dimensions=1536,
            created_at="2026-01-01T00:00:00Z",
        )
        assert emb.vector is None

    def test_dimensions_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingObject(
                id="emb-001",
                memory_id="mem-001",
                model="test",
                dimensions=0,
                created_at="2026-01-01T00:00:00Z",
            )

    def test_embeddings_file(self) -> None:
        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-001",
                    memory_id="mem-001",
                    model="test",
                    dimensions=2,
                    created_at="2026-01-01T00:00:00Z",
                    vector=[0.1, 0.2],
                )
            ],
        )
        assert len(ef.embeddings) == 1


class TestValidateEmbeddings:
    def _make_memory(self, id: str, embedding_ref: str | None = None) -> MemoryObject:
        content = f"content for {id}"
        return MemoryObject(
            id=id,
            type="fact",
            content=content,
            content_hash=compute_content_hash(content),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude"),
            embedding_ref=embedding_ref,
        )

    def test_valid_standalone(self) -> None:
        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-001",
                    memory_id="mem-001",
                    model="test",
                    dimensions=2,
                    created_at="2026-01-01T00:00:00Z",
                    vector=[0.1, 0.2],
                )
            ],
        )
        result = validate_embeddings(ef)
        assert result.is_valid

    def test_dimensions_mismatch(self) -> None:
        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-001",
                    memory_id="mem-001",
                    model="test",
                    dimensions=3,
                    created_at="2026-01-01T00:00:00Z",
                    vector=[0.1, 0.2],  # only 2 elements
                )
            ],
        )
        result = validate_embeddings(ef)
        assert not result.is_valid
        assert any("dimensions-mismatch" in e.check for e in result.errors)

    def test_duplicate_memory_id(self) -> None:
        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-001",
                    memory_id="mem-001",
                    model="test",
                    dimensions=2,
                    created_at="2026-01-01T00:00:00Z",
                    vector=[0.1, 0.2],
                ),
                EmbeddingObject(
                    id="emb-002",
                    memory_id="mem-001",
                    model="test",
                    dimensions=2,
                    created_at="2026-01-01T00:00:00Z",
                    vector=[0.3, 0.4],
                ),
            ],
        )
        result = validate_embeddings(ef)
        assert not result.is_valid
        assert any("memory-id-unique" in e.check for e in result.errors)

    def test_xref_memory_not_in_store(self) -> None:
        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-001",
                    memory_id="mem-ghost",
                    model="test",
                    dimensions=2,
                    created_at="2026-01-01T00:00:00Z",
                    vector=[0.1, 0.2],
                )
            ],
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[self._make_memory("mem-001")],
        )
        result = validate_embeddings(ef, memory_store=store)
        assert not result.is_valid
        assert any("xref-memory" in e.check for e in result.errors)

    def test_xref_embedding_not_found(self) -> None:
        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-001",
                    memory_id="mem-001",
                    model="test",
                    dimensions=2,
                    created_at="2026-01-01T00:00:00Z",
                    vector=[0.1, 0.2],
                )
            ],
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[self._make_memory("mem-001", embedding_ref="emb-ghost")],
        )
        result = validate_embeddings(ef, memory_store=store)
        assert not result.is_valid
        assert any("xref-embedding" in e.check for e in result.errors)

    def test_example_files_cross_validate(self) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        emb = load_embeddings(FIXTURES / "example-embeddings.json")
        result = validate_embeddings(emb, memory_store=store)
        assert result.is_valid
