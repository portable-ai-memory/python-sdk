"""Tests for PAM models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from portable_ai_memory.core.integrity import compute_content_hash
from portable_ai_memory.models.memory_store import (
    MemoryObject,
    MemoryStore,
    Owner,
    ProvenanceBlock,
    TemporalBlock,
)


class TestMemoryObject:
    def test_minimal_valid(self, minimal_memory: MemoryObject) -> None:
        assert minimal_memory.id == "mem-001"
        assert minimal_memory.type == "preference"
        assert minimal_memory.status == "active"

    def test_custom_type_required(self) -> None:
        content = "Custom content"
        with pytest.raises(ValidationError, match="custom_type"):
            MemoryObject(
                id="mem-err",
                type="custom",
                content=content,
                content_hash=compute_content_hash(content),
                temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
                provenance=ProvenanceBlock(platform="claude"),
            )

    def test_custom_type_must_be_null_for_standard_types(self) -> None:
        content = "Some fact"
        with pytest.raises(ValidationError, match="custom_type must be null"):
            MemoryObject(
                id="mem-err",
                type="fact",
                custom_type="not-allowed",
                content=content,
                content_hash=compute_content_hash(content),
                temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
                provenance=ProvenanceBlock(platform="claude"),
            )

    def test_invalid_tag_format(self) -> None:
        content = "Some content"
        with pytest.raises(ValidationError, match="Invalid tag"):
            MemoryObject(
                id="mem-err",
                type="fact",
                content=content,
                content_hash=compute_content_hash(content),
                tags=["INVALID-UPPERCASE"],
                temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
                provenance=ProvenanceBlock(platform="claude"),
            )


class TestMemoryStore:
    def test_minimal_valid(self, minimal_store: MemoryStore) -> None:
        assert minimal_store.schema_ == "portable-ai-memory"
        assert len(minimal_store.memories) == 1

    def test_serialization_roundtrip(self, minimal_store: MemoryStore) -> None:
        data = minimal_store.model_dump(mode="json", by_alias=True)
        restored = MemoryStore.model_validate(data)
        assert restored.memories[0].id == minimal_store.memories[0].id
        assert data["schema"] == "portable-ai-memory"

    def test_signature_requires_export_fields(self) -> None:
        with pytest.raises(ValidationError, match="export_id"):
            MemoryStore(
                schema_version="1.0",
                owner=Owner(id="user-test"),
                memories=[],
                signature={
                    "algorithm": "Ed25519",
                    "public_key": "z6Mk...",
                    "value": "base64...",
                    "signed_at": "2026-01-01T00:00:00Z",
                },
            )

    def test_get_memory_by_id(self, minimal_store: MemoryStore) -> None:
        found = minimal_store.get_memory_by_id("mem-001")
        assert found is not None
        assert found.id == "mem-001"
        assert minimal_store.get_memory_by_id("nonexistent") is None

    def test_get_memories_by_type(self, minimal_store: MemoryStore) -> None:
        from portable_ai_memory.models.enums import MemoryType

        results = minimal_store.get_memories_by_type(MemoryType.PREFERENCE)
        assert len(results) == 1
        assert results[0].id == "mem-001"
        assert minimal_store.get_memories_by_type(MemoryType.FACT) == []

    def test_get_memories_with_tag(self, minimal_store: MemoryStore) -> None:
        # minimal_store memory has no tags by default
        assert minimal_store.get_memories_with_tag("food") == []


class TestMemoryObjectCreate:
    def test_create_basic(self) -> None:
        from portable_ai_memory.models.enums import MemoryType

        mem = MemoryObject.create(
            id="test-001",
            type=MemoryType.FACT,
            content="The sky is blue",
            platform="test",
        )
        assert mem.id == "test-001"
        assert mem.type == MemoryType.FACT
        assert mem.content == "The sky is blue"
        assert mem.content_hash.startswith("sha256:")
        assert mem.temporal.created_at is not None
        assert mem.provenance.platform == "test"

    def test_create_with_tags(self) -> None:
        from portable_ai_memory.models.enums import MemoryType

        mem = MemoryObject.create(
            id="test-002",
            type=MemoryType.PREFERENCE,
            content="I like Python",
            platform="claude",
            tags=["coding", "language"],
            summary="Language preference",
        )
        assert mem.tags == ["coding", "language"]
        assert mem.summary == "Language preference"

    def test_create_custom_type(self) -> None:
        from portable_ai_memory.models.enums import MemoryType

        mem = MemoryObject.create(
            id="test-003",
            type=MemoryType.CUSTOM,
            content="Custom data",
            platform="test",
            custom_type="my_type",
        )
        assert mem.type == MemoryType.CUSTOM
        assert mem.custom_type == "my_type"

    def test_create_custom_type_required(self) -> None:
        from portable_ai_memory.models.enums import MemoryType

        with pytest.raises(ValidationError, match="custom_type"):
            MemoryObject.create(
                id="test-004",
                type=MemoryType.CUSTOM,
                content="Missing custom_type",
                platform="test",
            )
