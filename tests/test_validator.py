"""Tests for integrity utilities and deep validator."""

from __future__ import annotations

from portable_ai_memory.core.integrity import (
    compute_content_hash,
    normalize_content,
    verify_content_hash,
)
from portable_ai_memory.core.validator import validate_memory_store
from portable_ai_memory.models.memory_store import (
    MemoryObject,
    MemoryStore,
    Owner,
    ProvenanceBlock,
    RelationObject,
    TemporalBlock,
)


class TestNormalization:
    def test_basic_normalization(self) -> None:
        assert normalize_content("  Hello  World  ") == "hello world"

    def test_unicode_nfc(self) -> None:
        # é as two codepoints (e + combining accent) vs single codepoint
        decomposed = "caf\u0065\u0301"
        composed = "caf\u00e9"
        assert normalize_content(decomposed) == normalize_content(composed)

    def test_collapse_whitespace(self) -> None:
        assert normalize_content("a\t\nb") == "a b"


class TestContentHash:
    def test_deterministic(self) -> None:
        h1 = compute_content_hash("Hello World")
        h2 = compute_content_hash("Hello World")
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_verify(self) -> None:
        content = "Some content"
        h = compute_content_hash(content)
        assert verify_content_hash(content, h)
        assert not verify_content_hash("Different", h)


class TestDeepValidation:
    def _make_memory(self, id: str, content: str = "test") -> MemoryObject:
        return MemoryObject(
            id=id,
            type="fact",
            content=content,
            content_hash=compute_content_hash(content),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude"),
        )

    def test_valid_store(self, minimal_store: MemoryStore) -> None:
        result = validate_memory_store(minimal_store)
        assert result.is_valid

    def test_content_hash_mismatch(self) -> None:
        mem = MemoryObject(
            id="mem-bad",
            type="fact",
            content="Actual content",
            content_hash="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude"),
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        result = validate_memory_store(store)
        assert not result.is_valid
        assert any("content-hash" in e.check for e in result.errors)

    def test_relation_to_missing_memory(self) -> None:
        mem = self._make_memory("mem-001")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            relations=[
                RelationObject(
                    id="rel-001",
                    **{"from": "mem-001"},
                    to="mem-GHOST",
                    type="supports",
                    created_at="2026-01-01T00:00:00Z",
                )
            ],
        )
        result = validate_memory_store(store)
        assert not result.is_valid
        assert any("xref-relation" in e.check for e in result.errors)

    def test_temporal_updated_before_created(self) -> None:
        mem = MemoryObject(
            id="mem-time",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(
                created_at="2026-06-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
            ),
            provenance=ProvenanceBlock(platform="claude"),
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        result = validate_memory_store(store)
        assert not result.is_valid
        assert any("temporal" in e.check for e in result.errors)

    def test_duplicate_memory_ids(self) -> None:
        mem1 = self._make_memory("mem-dup", "content a")
        mem2 = self._make_memory("mem-dup", "content b")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem1, mem2],
        )
        result = validate_memory_store(store)
        assert not result.is_valid
        assert any("id-unique" in e.check for e in result.errors)

    def test_duplicate_relation_ids(self) -> None:
        mem1 = self._make_memory("mem-a")
        mem2 = self._make_memory("mem-b", "other")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem1, mem2],
            relations=[
                RelationObject(
                    id="rel-1",
                    **{"from": "mem-a"},
                    to="mem-b",
                    type="supports",
                    created_at="2026-01-01T00:00:00Z",
                ),
                RelationObject(
                    id="rel-1",
                    **{"from": "mem-b"},
                    to="mem-a",
                    type="supports",
                    created_at="2026-01-01T00:00:00Z",
                ),
            ],
        )
        result = validate_memory_store(store)
        assert any(i.check == "id-unique" and "relation" in i.message for i in result.errors)

    def test_duplicate_conversation_ids(self) -> None:
        from portable_ai_memory.models.memory_store import ConversationIndexEntry, StorageReference
        from portable_ai_memory.models.memory_store import ConversationTemporal as CIT

        mem = self._make_memory("mem-1")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            conversations_index=[
                ConversationIndexEntry(
                    id="conv-1",
                    platform="test",
                    temporal=CIT(created_at="2026-01-01T00:00:00Z"),
                    storage=StorageReference(type="file", ref="c1.json"),
                ),
                ConversationIndexEntry(
                    id="conv-1",
                    platform="test",
                    temporal=CIT(created_at="2026-01-01T00:00:00Z"),
                    storage=StorageReference(type="file", ref="c2.json"),
                ),
            ],
        )
        result = validate_memory_store(store)
        assert any(
            i.check == "id-unique" and "conversation" in i.message.lower() for i in result.errors
        )

    def test_self_referencing_relation(self) -> None:
        mem = self._make_memory("mem-1")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            relations=[
                RelationObject(
                    id="rel-1",
                    **{"from": "mem-1"},
                    to="mem-1",
                    type="supports",
                    created_at="2026-01-01T00:00:00Z",
                ),
            ],
        )
        result = validate_memory_store(store)
        assert any(
            i.check == "xref-relation" and "self-referencing" in i.message for i in result.warnings
        )

    def test_relation_from_unknown(self) -> None:
        mem = self._make_memory("mem-1")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            relations=[
                RelationObject(
                    id="rel-1",
                    **{"from": "mem-GHOST"},
                    to="mem-1",
                    type="supports",
                    created_at="2026-01-01T00:00:00Z",
                ),
            ],
        )
        result = validate_memory_store(store)
        assert any(i.check == "xref-relation" and "from" in i.message for i in result.errors)

    def test_conversation_ref_not_in_index(self) -> None:
        from portable_ai_memory.models.memory_store import ConversationIndexEntry, StorageReference
        from portable_ai_memory.models.memory_store import ConversationTemporal as CIT

        mem = MemoryObject(
            id="mem-1",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude", conversation_ref="conv-GHOST"),
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            conversations_index=[
                ConversationIndexEntry(
                    id="conv-1",
                    platform="test",
                    temporal=CIT(created_at="2026-01-01T00:00:00Z"),
                    storage=StorageReference(type="file", ref="c.json"),
                ),
            ],
        )
        result = validate_memory_store(store)
        assert any(i.check == "xref-conversation" for i in result.errors)

    def test_superseded_by_not_found(self) -> None:
        mem = MemoryObject(
            id="mem-1",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z", superseded_by="mem-GHOST"),
            provenance=ProvenanceBlock(platform="claude"),
            status="superseded",
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        result = validate_memory_store(store)
        assert any(i.check == "xref-superseded" for i in result.errors)

    def test_derived_memories_unknown(self) -> None:
        from portable_ai_memory.models.memory_store import ConversationIndexEntry, StorageReference
        from portable_ai_memory.models.memory_store import ConversationTemporal as CIT

        mem = self._make_memory("mem-1")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            conversations_index=[
                ConversationIndexEntry(
                    id="conv-1",
                    platform="test",
                    temporal=CIT(created_at="2026-01-01T00:00:00Z"),
                    storage=StorageReference(type="file", ref="c.json"),
                    derived_memories=["mem-GHOST"],
                ),
            ],
        )
        result = validate_memory_store(store)
        assert any(i.check == "xref-derived" for i in result.errors)

    def test_bidirectional_conversation_ref_warning(self) -> None:
        from portable_ai_memory.models.memory_store import ConversationIndexEntry, StorageReference
        from portable_ai_memory.models.memory_store import ConversationTemporal as CIT

        mem = MemoryObject(
            id="mem-1",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude", conversation_ref="conv-1"),
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            conversations_index=[
                ConversationIndexEntry(
                    id="conv-1",
                    platform="test",
                    temporal=CIT(created_at="2026-01-01T00:00:00Z"),
                    storage=StorageReference(type="file", ref="c.json"),
                    derived_memories=[],  # mem-1 NOT listed
                ),
            ],
        )
        result = validate_memory_store(store)
        assert any(i.check == "xref-bidirectional" for i in result.warnings)

    def test_valid_until_before_valid_from(self) -> None:
        mem = MemoryObject(
            id="mem-1",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(
                created_at="2026-01-01T00:00:00Z",
                valid_from="2026-06-01T00:00:00Z",
                valid_until="2026-01-01T00:00:00Z",
            ),
            provenance=ProvenanceBlock(platform="claude"),
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        result = validate_memory_store(store)
        assert any(i.check == "temporal" and "valid_until" in i.message for i in result.errors)

    def test_last_reinforced_before_created(self) -> None:
        from portable_ai_memory.models.memory_store import ConfidenceBlock

        mem = MemoryObject(
            id="mem-1",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(created_at="2026-06-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude"),
            confidence=ConfidenceBlock(last_reinforced="2025-01-01T00:00:00Z"),
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        result = validate_memory_store(store)
        assert any(
            i.check == "temporal" and "last_reinforced" in i.message for i in result.warnings
        )

    def test_conversation_temporal_updated_before_created(self) -> None:
        from portable_ai_memory.models.memory_store import ConversationIndexEntry, StorageReference
        from portable_ai_memory.models.memory_store import ConversationTemporal as CIT

        mem = self._make_memory("mem-1")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            conversations_index=[
                ConversationIndexEntry(
                    id="conv-1",
                    platform="test",
                    temporal=CIT(
                        created_at="2026-06-01T00:00:00Z",
                        updated_at="2025-01-01T00:00:00Z",
                    ),
                    storage=StorageReference(type="file", ref="c.json"),
                ),
            ],
        )
        result = validate_memory_store(store)
        assert any(i.check == "temporal" and "Conversation" in i.message for i in result.errors)

    def test_custom_type_missing(self) -> None:
        """type='custom' but no custom_type set — caught by model validator, not deep validator."""
        import pytest as _pytest

        with _pytest.raises(ValueError):
            MemoryObject(
                id="mem-1",
                type="custom",
                content="test",
                content_hash=compute_content_hash("test"),
                temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
                provenance=ProvenanceBlock(platform="claude"),
            )

    def test_custom_type_on_non_custom_type_deep_validation(self) -> None:
        """Deep validator catches custom_type set on non-custom type (bypassing model validator)."""
        mem = self._make_memory("mem-1")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        # Bypass Pydantic validator by mutating after store construction
        object.__setattr__(store.memories[0], "custom_type", "my-custom")
        result = validate_memory_store(store)
        assert any(i.check == "custom-type" and "custom_type=" in i.message for i in result.errors)

    def test_custom_type_missing_deep_validation(self) -> None:
        """Deep validator catches type='custom' without custom_type (bypassing model validator)."""
        mem = self._make_memory("mem-1")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        object.__setattr__(store.memories[0], "type", "custom")
        object.__setattr__(store.memories[0], "custom_type", None)
        result = validate_memory_store(store)
        assert any(
            i.check == "custom-type" and "no custom_type" in i.message for i in result.errors
        )

    def test_status_superseded_without_superseded_by(self) -> None:
        mem = MemoryObject(
            id="mem-1",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude"),
            status="superseded",
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        result = validate_memory_store(store)
        assert any(
            i.check == "status" and "superseded_by not set" in i.message for i in result.warnings
        )

    def test_status_superseded_by_set_but_not_superseded(self) -> None:
        mem1 = self._make_memory("mem-1")
        mem2 = MemoryObject(
            id="mem-2",
            type="fact",
            content="old",
            content_hash=compute_content_hash("old"),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z", superseded_by="mem-1"),
            provenance=ProvenanceBlock(platform="claude"),
            status="active",  # should be superseded
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem1, mem2],
        )
        result = validate_memory_store(store)
        assert any(
            i.check == "status" and "superseded_by set" in i.message for i in result.warnings
        )

    def test_integrity_total_mismatch(self) -> None:
        from portable_ai_memory.core.integrity import compute_integrity_checksum
        from portable_ai_memory.core.io import to_dict
        from portable_ai_memory.models.memory_store import IntegrityBlock

        mem = self._make_memory("mem-1")
        memories_dicts = [to_dict(mem)]
        checksum = compute_integrity_checksum(memories_dicts)
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            integrity=IntegrityBlock(total_memories=99, checksum=checksum),
        )
        result = validate_memory_store(store)
        assert any(i.check == "integrity-total" for i in result.errors)

    def test_integrity_checksum_mismatch(self) -> None:
        from portable_ai_memory.models.memory_store import IntegrityBlock

        mem = self._make_memory("mem-1")
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
            integrity=IntegrityBlock(total_memories=1, checksum="sha256:" + "0" * 64),
        )
        result = validate_memory_store(store)
        assert any(i.check == "integrity-checksum" for i in result.errors)


class TestIssueStr:
    def test_error_icon(self) -> None:
        from portable_ai_memory.core.validator import Issue

        i = Issue(level="error", check="test", message="bad")
        assert str(i).startswith("✗")

    def test_warning_icon(self) -> None:
        from portable_ai_memory.core.validator import Issue

        i = Issue(level="warning", check="test", message="meh")
        assert str(i).startswith("⚠")


class TestValidationResultStr:
    def test_all_passed(self) -> None:
        from portable_ai_memory.core.validator import ValidationResult

        r = ValidationResult()
        assert "All checks passed" in str(r)

    def test_with_issues(self) -> None:
        from portable_ai_memory.core.validator import ValidationResult

        r = ValidationResult()
        r.error("test", "bad thing")
        r.warning("test", "meh thing")
        s = str(r)
        assert "bad thing" in s
        assert "meh thing" in s

    def test_warnings_property(self) -> None:
        from portable_ai_memory.core.validator import ValidationResult

        r = ValidationResult()
        r.warning("w", "warn")
        assert len(r.warnings) == 1
        assert r.warnings[0].level == "warning"


class TestValidateConversation:
    def test_missing_parent(self) -> None:
        from portable_ai_memory.core.validator import validate_conversation
        from portable_ai_memory.models.conversation import (
            Conversation,
            ConversationTemporal,
            Message,
            ProviderInfo,
        )
        from portable_ai_memory.models.enums import MessageRole

        conv = Conversation(
            schema_version="1.0",
            id="conv-1",
            provider=ProviderInfo(name="chatgpt"),
            temporal=ConversationTemporal(created_at="2026-01-01T00:00:00Z"),
            messages=[
                Message(
                    id="msg-1",
                    role=MessageRole.USER,
                    created_at="2026-01-01T00:00:00Z",
                    parent_id="msg-GHOST",
                ),
            ],
        )
        result = validate_conversation(conv)
        assert any(i.check == "dag" for i in result.warnings)

    def test_missing_child(self) -> None:
        from portable_ai_memory.core.validator import validate_conversation
        from portable_ai_memory.models.conversation import (
            Conversation,
            ConversationTemporal,
            Message,
            ProviderInfo,
        )
        from portable_ai_memory.models.enums import MessageRole

        conv = Conversation(
            schema_version="1.0",
            id="conv-1",
            provider=ProviderInfo(name="chatgpt"),
            temporal=ConversationTemporal(created_at="2026-01-01T00:00:00Z"),
            messages=[
                Message(
                    id="msg-1",
                    role=MessageRole.USER,
                    created_at="2026-01-01T00:00:00Z",
                    children_ids=["msg-GHOST"],
                ),
            ],
        )
        result = validate_conversation(conv)
        assert any(i.check == "dag" and "child" in i.message for i in result.warnings)

    def test_dag_bidirectional_mismatch(self) -> None:
        from portable_ai_memory.core.validator import validate_conversation
        from portable_ai_memory.models.conversation import (
            Conversation,
            ConversationTemporal,
            Message,
            ProviderInfo,
        )
        from portable_ai_memory.models.enums import MessageRole

        conv = Conversation(
            schema_version="1.0",
            id="conv-1",
            provider=ProviderInfo(name="chatgpt"),
            temporal=ConversationTemporal(created_at="2026-01-01T00:00:00Z"),
            messages=[
                Message(
                    id="msg-1",
                    role=MessageRole.USER,
                    created_at="2026-01-01T00:00:00Z",
                    children_ids=["msg-2"],
                ),
                Message(
                    id="msg-2",
                    role=MessageRole.ASSISTANT,
                    created_at="2026-01-01T00:00:00Z",
                    parent_id="msg-OTHER",
                ),
            ],
        )
        result = validate_conversation(conv)
        assert any(i.check == "dag-bidirectional" for i in result.warnings)


class TestValidateEmbeddings:
    def test_duplicate_embedding_id(self) -> None:
        from datetime import UTC, datetime

        from portable_ai_memory.core.validator import validate_embeddings
        from portable_ai_memory.models.embeddings import EmbeddingObject, EmbeddingsFile

        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-1",
                    memory_id="mem-1",
                    model="m",
                    dimensions=2,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                EmbeddingObject(
                    id="emb-1",
                    memory_id="mem-2",
                    model="m",
                    dimensions=2,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ],
        )
        result = validate_embeddings(ef)
        assert any(i.check == "id-unique" for i in result.errors)

    def test_duplicate_memory_id_in_embeddings(self) -> None:
        from datetime import UTC, datetime

        from portable_ai_memory.core.validator import validate_embeddings
        from portable_ai_memory.models.embeddings import EmbeddingObject, EmbeddingsFile

        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-1",
                    memory_id="mem-1",
                    model="m",
                    dimensions=2,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                EmbeddingObject(
                    id="emb-2",
                    memory_id="mem-1",
                    model="m",
                    dimensions=2,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ],
        )
        result = validate_embeddings(ef)
        assert any(i.check == "memory-id-unique" for i in result.errors)

    def test_dimensions_mismatch(self) -> None:
        from datetime import UTC, datetime

        from portable_ai_memory.core.validator import validate_embeddings
        from portable_ai_memory.models.embeddings import EmbeddingObject, EmbeddingsFile

        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-1",
                    memory_id="mem-1",
                    model="m",
                    dimensions=3,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    vector=[1.0, 2.0],
                ),
            ],
        )
        result = validate_embeddings(ef)
        assert any(i.check == "dimensions-mismatch" for i in result.errors)

    def test_xref_memory_not_in_store(self) -> None:
        from datetime import UTC, datetime

        from portable_ai_memory.core.validator import validate_embeddings
        from portable_ai_memory.models.embeddings import EmbeddingObject, EmbeddingsFile

        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-1",
                    memory_id="mem-GHOST",
                    model="m",
                    dimensions=2,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ],
        )
        store = MemoryStore(schema_version="1.0", owner=Owner(id="test"), memories=[])
        result = validate_embeddings(ef, memory_store=store)
        assert any(i.check == "xref-memory" for i in result.errors)

    def test_xref_embedding_ref_not_in_file(self) -> None:
        from datetime import UTC, datetime

        from portable_ai_memory.core.validator import validate_embeddings
        from portable_ai_memory.models.embeddings import EmbeddingObject, EmbeddingsFile

        mem = MemoryObject(
            id="mem-1",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude"),
            embedding_ref="emb-GHOST",
        )
        ef = EmbeddingsFile(
            schema_version="1.0",
            embeddings=[
                EmbeddingObject(
                    id="emb-1",
                    memory_id="mem-1",
                    model="m",
                    dimensions=2,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ],
        )
        store = MemoryStore(schema_version="1.0", owner=Owner(id="test"), memories=[mem])
        result = validate_embeddings(ef, memory_store=store)
        assert any(i.check == "xref-embedding" for i in result.errors)


class TestValidateDispatch:
    """Tests for the unified validate() function."""

    def test_validate_memory_store(self) -> None:
        from portable_ai_memory.core.validator import validate

        mem = MemoryObject(
            id="mem-1",
            type="fact",
            content="test",
            content_hash=compute_content_hash("test"),
            temporal=TemporalBlock(created_at="2026-01-01T00:00:00Z"),
            provenance=ProvenanceBlock(platform="claude"),
        )
        store = MemoryStore(
            schema_version="1.0",
            owner=Owner(id="test"),
            memories=[mem],
        )
        result = validate(store)
        assert result.is_valid

    def test_validate_conversation(self) -> None:
        from portable_ai_memory.core.validator import validate
        from portable_ai_memory.models.conversation import (
            Conversation,
            ConversationTemporal,
            Message,
            ProviderInfo,
        )
        from portable_ai_memory.models.enums import MessageRole

        conv = Conversation(
            schema_version="1.0",
            id="conv-1",
            provider=ProviderInfo(name="chatgpt"),
            temporal=ConversationTemporal(created_at="2026-01-01T00:00:00Z"),
            messages=[
                Message(
                    id="msg-1",
                    role=MessageRole.USER,
                    created_at="2026-01-01T00:00:00Z",
                ),
            ],
        )
        result = validate(conv)
        assert result.is_valid

    def test_validate_unknown_type_raises(self) -> None:
        import pytest

        from portable_ai_memory.core.validator import validate

        with pytest.raises(TypeError, match="Unknown PAM document type"):
            validate("not a document")  # type: ignore[arg-type]

    def test_validate_exported_from_package(self) -> None:
        from portable_ai_memory import validate

        assert callable(validate)


class TestValidateConverterOutput:
    def test_valid_output_passes(self) -> None:
        from datetime import UTC, datetime

        from portable_ai_memory.converters.base import validate_converter_output
        from portable_ai_memory.models.conversation import (
            Conversation,
            ConversationTemporal,
            Message,
            MessageContent,
            Participant,
            ProviderInfo,
        )
        from portable_ai_memory.models.enums import ContentType, MessageRole

        conv = Conversation(
            schema_version="1.0",
            id="conv-1",
            provider=ProviderInfo(name="test"),
            temporal=ConversationTemporal(created_at=datetime.now(tz=UTC)),
            participants=[Participant(role=MessageRole.USER)],
            messages=[
                Message(
                    id="msg-1",
                    role=MessageRole.USER,
                    content=MessageContent(type=ContentType.TEXT, text="hello"),
                    created_at=datetime.now(tz=UTC),
                ),
            ],
        )
        # Should not raise
        validate_converter_output([conv])
