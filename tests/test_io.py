"""Tests for PAM I/O module — load/save roundtrip for all document types."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from portable_ai_memory.core.io import (
    load,
    load_conversation,
    load_dict,
    load_embeddings,
    load_memory_store,
    save,
    to_dict,
)
from portable_ai_memory.exceptions import PAMSchemaError
from portable_ai_memory.models.conversation import Conversation
from portable_ai_memory.models.embeddings import EmbeddingsFile
from portable_ai_memory.models.memory_store import MemoryStore

FIXTURES = Path(__file__).parent / "fixtures"


class TestLoadAutoDetect:
    def test_load_memory_store(self) -> None:
        doc = load(FIXTURES / "example-memory-store.json")
        assert isinstance(doc, MemoryStore)
        assert doc.schema_ == "portable-ai-memory"

    def test_load_conversation(self) -> None:
        doc = load(FIXTURES / "example-conversation.json")
        assert isinstance(doc, Conversation)
        assert doc.schema_ == "portable-ai-memory-conversation"

    def test_load_embeddings(self) -> None:
        doc = load(FIXTURES / "example-embeddings.json")
        assert isinstance(doc, EmbeddingsFile)
        assert doc.schema_ == "portable-ai-memory-embeddings"


class TestTypedLoaders:
    def test_load_memory_store(self) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        assert len(store.memories) == 5

    def test_load_conversation(self) -> None:
        conv = load_conversation(FIXTURES / "example-conversation.json")
        assert len(conv.messages) == 2

    def test_load_embeddings(self) -> None:
        emb = load_embeddings(FIXTURES / "example-embeddings.json")
        assert len(emb.embeddings) == 5


class TestSaveRoundtrip:
    def test_memory_store_roundtrip(self, tmp_path: Path) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        out = save(store, tmp_path / "out.json")
        restored = load_memory_store(out)
        assert len(restored.memories) == len(store.memories)
        assert restored.owner.id == store.owner.id

    def test_conversation_roundtrip(self, tmp_path: Path) -> None:
        conv = load_conversation(FIXTURES / "example-conversation.json")
        out = save(conv, tmp_path / "out.json")
        restored = load_conversation(out)
        assert len(restored.messages) == len(conv.messages)
        assert restored.id == conv.id

    def test_embeddings_roundtrip(self, tmp_path: Path) -> None:
        emb = load_embeddings(FIXTURES / "example-embeddings.json")
        out = save(emb, tmp_path / "out.json")
        restored = load_embeddings(out)
        assert len(restored.embeddings) == len(emb.embeddings)


class TestLoadDictErrors:
    def test_missing_schema_field(self) -> None:
        with pytest.raises(PAMSchemaError, match="Missing 'schema' field"):
            load_dict({"version": "1.0"})

    def test_unknown_schema(self) -> None:
        with pytest.raises(PAMSchemaError, match="Unknown schema"):
            load_dict({"schema": "not-a-real-schema"})


class TestMalformedJson:
    def test_load_malformed_json_raises_pam_schema_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(PAMSchemaError, match="Invalid JSON"):
            load(bad)


class TestToDict:
    def test_uses_aliases(self) -> None:
        store = load_memory_store(FIXTURES / "example-memory-store.json")
        d = to_dict(store)
        assert "schema" in d
        assert d["schema"] == "portable-ai-memory"
        # Verify JSON-serializable
        json.dumps(d)
