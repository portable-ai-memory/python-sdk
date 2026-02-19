"""Tests validating the spec example files load and pass deep validation."""

from __future__ import annotations

from pathlib import Path

from portable_ai_memory.core.io import load, load_conversation, load_embeddings, load_memory_store
from portable_ai_memory.core.validator import (
    validate_conversation,
    validate_embeddings,
    validate_memory_store,
)
from portable_ai_memory.models.conversation import Conversation
from portable_ai_memory.models.embeddings import EmbeddingsFile
from portable_ai_memory.models.memory_store import MemoryStore

FIXTURES = Path(__file__).parent / "fixtures"


class TestExampleMemoryStore:
    def test_loads_successfully(self) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        assert isinstance(store, MemoryStore)
        assert store.schema_version == "1.0"
        assert len(store.memories) == 5
        assert len(store.relations) == 3
        assert len(store.conversations_index) == 3

    def test_auto_detect(self) -> None:
        doc = load(FIXTURES / "example-memory-store.json")
        assert isinstance(doc, MemoryStore)

    def test_deep_validation_passes(self) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        result = validate_memory_store(store)
        # Allow warnings but no errors
        assert result.is_valid, str(result)

    def test_owner(self) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        assert store.owner.id == "550e8400-e29b-41d4-a716-446655440000"
        assert store.owner.did is not None

    def test_integrity_block(self) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        assert store.integrity is not None
        assert store.integrity.total_memories == 5

    def test_conversations_index_uses_tags(self) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        conv = store.conversations_index[0]
        assert hasattr(conv, "tags")
        assert len(conv.tags) > 0


class TestExampleConversation:
    def test_loads_successfully(self) -> None:
        conv = load_conversation(FIXTURES / "example-conversation.json")
        assert isinstance(conv, Conversation)
        assert conv.schema_version == "1.0"
        assert len(conv.messages) == 2

    def test_auto_detect(self) -> None:
        doc = load(FIXTURES / "example-conversation.json")
        assert isinstance(doc, Conversation)

    def test_deep_validation_passes(self) -> None:
        conv = load_conversation(FIXTURES / "example-conversation.json")
        result = validate_conversation(conv)
        assert result.is_valid, str(result)

    def test_temporal_required(self) -> None:
        conv = load_conversation(FIXTURES / "example-conversation.json")
        assert conv.temporal is not None
        assert conv.temporal.created_at is not None

    def test_import_metadata(self) -> None:
        conv = load_conversation(FIXTURES / "example-conversation.json")
        assert conv.import_metadata is not None
        assert conv.import_metadata.importer == "gines/0.5.0"


class TestExampleEmbeddings:
    def test_loads_successfully(self) -> None:
        emb = load_embeddings(FIXTURES / "example-embeddings.json")
        assert isinstance(emb, EmbeddingsFile)
        assert len(emb.embeddings) == 5

    def test_auto_detect(self) -> None:
        doc = load(FIXTURES / "example-embeddings.json")
        assert isinstance(doc, EmbeddingsFile)

    def test_deep_validation_passes(self) -> None:
        emb = load_embeddings(FIXTURES / "example-embeddings.json")
        result = validate_embeddings(emb)
        assert result.is_valid, str(result)

    def test_vectors_match_dimensions(self) -> None:
        emb = load_embeddings(FIXTURES / "example-embeddings.json")
        for e in emb.embeddings:
            if e.vector is not None:
                assert len(e.vector) == e.dimensions
