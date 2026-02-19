"""Deep validation for PAM documents.

Goes beyond JSON Schema validation to check:
- Content hash integrity
- Integrity block consistency
- Cross-references (relations → memories, conversation_ref → conversations_index, etc.)
- Temporal ordering (created_at ≤ updated_at, valid_from ≤ valid_until)
- ID uniqueness
- Custom type rules
- Status ↔ superseded_by consistency
- Conversation message DAG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from portable_ai_memory.core.integrity import compute_content_hash, compute_integrity_checksum

if TYPE_CHECKING:
    from portable_ai_memory.models.conversation import Conversation
    from portable_ai_memory.models.embeddings import EmbeddingsFile
    from portable_ai_memory.models.memory_store import MemoryStore


@dataclass
class Issue:
    """A single validation issue."""

    level: Literal["error", "warning"]
    check: str
    message: str

    def __str__(self) -> str:
        icon = "✗" if self.level == "error" else "⚠"
        return f"{icon} [{self.check}] {self.message}"


@dataclass
class ValidationResult:
    """Result of deep validation.

    Checks performed on MemoryStore:
        content-hash     — content_hash matches computed SHA-256
        integrity-total  — integrity.total_memories matches actual count
        integrity-checksum — integrity.checksum matches recomputed value
        id-unique        — no duplicate memory/relation/conversation IDs
        xref-relation    — relation from/to point to existing memories
        xref-conversation — conversation_ref points to conversations_index
        xref-superseded  — superseded_by points to existing memory
        xref-derived     — derived_memories point to existing memories
        xref-bidirectional — conversation_ref ↔ derived_memories consistency
        temporal         — created_at ≤ updated_at, valid_from ≤ valid_until
        custom-type      — type='custom' requires custom_type and vice versa
        status           — superseded status ↔ superseded_by consistency

    Checks performed on Conversation:
        dag              — parent_id references exist
        dag-bidirectional — parent/children consistency

    Checks performed on EmbeddingsFile:
        id-unique        — no duplicate embedding IDs
        memory-id-unique — no duplicate memory_id in embeddings
        dimensions-mismatch — vector length matches declared dimensions
        xref-memory      — embedding memory_id exists in store (if provided)
        xref-embedding   — memory embedding_ref exists in file (if provided)
    """

    issues: list[Issue] = field(default_factory=list)

    def error(self, check: str, message: str) -> None:
        """Record a validation error (causes is_valid to be False)."""
        self.issues.append(Issue("error", check, message))

    def warning(self, check: str, message: str) -> None:
        """Record a validation warning (does not affect is_valid)."""
        self.issues.append(Issue("warning", check, message))

    @property
    def errors(self) -> list[Issue]:
        """All error-level issues found during validation."""
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        """All warning-level issues found during validation."""
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        """True if no errors were found (warnings are allowed)."""
        return len(self.errors) == 0

    def __str__(self) -> str:
        if self.is_valid and not self.warnings:
            return "✓ All checks passed"
        lines = [str(i) for i in self.issues]
        return "\n".join(lines)


def validate(document: MemoryStore | Conversation | EmbeddingsFile) -> ValidationResult:
    """Validate any PAM document (auto-detects type).

    This is the recommended entry point for validation. It dispatches to
    the appropriate type-specific validator based on the document class.

    Args:
        document: A MemoryStore, Conversation, or EmbeddingsFile instance.

    Returns:
        ValidationResult with any errors and warnings found.

    Raises:
        TypeError: If the document is not a recognized PAM type.
    """
    from portable_ai_memory.models.conversation import Conversation
    from portable_ai_memory.models.embeddings import EmbeddingsFile
    from portable_ai_memory.models.memory_store import MemoryStore

    if isinstance(document, MemoryStore):
        return validate_memory_store(document)
    if isinstance(document, Conversation):
        return validate_conversation(document)
    if isinstance(document, EmbeddingsFile):
        return validate_embeddings(document)
    msg = f"Unknown PAM document type: {type(document).__name__}"
    raise TypeError(msg)


def validate_memory_store(store: MemoryStore) -> ValidationResult:
    """Run all deep validation checks on a MemoryStore.

    Checks content hashes, integrity block, ID uniqueness, cross-references,
    temporal consistency, custom type rules, and status consistency.
    """
    result = ValidationResult()

    _check_content_hashes(store, result)
    _check_integrity_block(store, result)
    _check_id_uniqueness(store, result)
    _check_cross_references(store, result)
    _check_temporal_consistency(store, result)
    _check_custom_type_rules(store, result)
    _check_status_consistency(store, result)

    return result


def validate_conversation(conversation: Conversation) -> ValidationResult:
    """Run deep validation checks on a Conversation.

    Checks message DAG integrity (parent_id references and children consistency).
    """
    result = ValidationResult()
    _check_message_dag(conversation, result)
    return result


def validate_embeddings(
    embeddings: EmbeddingsFile,
    memory_store: MemoryStore | None = None,
) -> ValidationResult:
    """Run deep validation checks on an EmbeddingsFile.

    Checks:
    - Embedding memory_id uniqueness
    - dimensions matches len(vector) when vector is present
    - If memory_store provided, all memory_id exist in store
    - If memory_store provided, all embedding_ref in store point to valid embeddings
    """
    result = ValidationResult()

    # ID uniqueness
    seen_ids: set[str] = set()
    seen_memory_ids: set[str] = set()
    embedding_ids: set[str] = set()
    for emb in embeddings.embeddings:
        if emb.id in seen_ids:
            result.error("id-unique", f"Duplicate embedding ID: '{emb.id}'")
        seen_ids.add(emb.id)
        embedding_ids.add(emb.id)

        if emb.memory_id in seen_memory_ids:
            result.error(
                "memory-id-unique",
                f"Duplicate memory_id in embeddings: '{emb.memory_id}'",
            )
        seen_memory_ids.add(emb.memory_id)

        # dimensions vs vector length
        if emb.vector is not None and len(emb.vector) != emb.dimensions:
            result.error(
                "dimensions-mismatch",
                f"Embedding '{emb.id}': dimensions={emb.dimensions}, "
                f"vector length={len(emb.vector)}",
            )

    if memory_store is not None:
        memory_ids = {m.id for m in memory_store.memories}

        # All embedding memory_id must exist in store
        for emb in embeddings.embeddings:
            if emb.memory_id not in memory_ids:
                result.error(
                    "xref-memory",
                    f"Embedding '{emb.id}': memory_id='{emb.memory_id}' not in store",
                )

        # All embedding_ref in store must point to valid embeddings
        for mem in memory_store.memories:
            if mem.embedding_ref and mem.embedding_ref not in embedding_ids:
                result.error(
                    "xref-embedding",
                    f"Memory '{mem.id}': embedding_ref='{mem.embedding_ref}' "
                    f"not in embeddings file",
                )

    return result


# ── Individual checks ─────────────────────────────────────


def _check_content_hashes(store: MemoryStore, result: ValidationResult) -> None:
    for mem in store.memories:
        computed = compute_content_hash(mem.content)
        if computed != mem.content_hash:
            result.error(
                "content-hash",
                f"Memory '{mem.id}': expected {computed}, declared {mem.content_hash}",
            )


def _check_integrity_block(store: MemoryStore, result: ValidationResult) -> None:
    if store.integrity is None:
        return

    if store.integrity.total_memories != len(store.memories):
        result.error(
            "integrity-total",
            f"integrity.total_memories={store.integrity.total_memories}, "
            f"actual={len(store.memories)}",
        )

    # IMPORTANT: by_alias=True must match how checksums are computed in
    # cli/_convert.py and how files are saved in core/io.py (both use by_alias).
    memories_dicts = [
        m.model_dump(mode="json", by_alias=True, exclude_none=True) for m in store.memories
    ]
    computed = compute_integrity_checksum(memories_dicts)
    if computed != store.integrity.checksum:
        result.error(
            "integrity-checksum",
            f"Declared: {store.integrity.checksum}, computed: {computed}",
        )


def _check_id_uniqueness(store: MemoryStore, result: ValidationResult) -> None:
    seen_mem: set[str] = set()
    for mem in store.memories:
        if mem.id in seen_mem:
            result.error("id-unique", f"Duplicate memory ID: '{mem.id}'")
        seen_mem.add(mem.id)

    seen_rel: set[str] = set()
    for rel in store.relations:
        if rel.id in seen_rel:
            result.error("id-unique", f"Duplicate relation ID: '{rel.id}'")
        seen_rel.add(rel.id)

    seen_conv: set[str] = set()
    for conv in store.conversations_index:
        if conv.id in seen_conv:
            result.error("id-unique", f"Duplicate conversation ID: '{conv.id}'")
        seen_conv.add(conv.id)


def _check_cross_references(store: MemoryStore, result: ValidationResult) -> None:
    memory_ids = {m.id for m in store.memories}
    conv_ids = {c.id for c in store.conversations_index}

    # Relations → memories
    for rel in store.relations:
        if rel.from_ not in memory_ids:
            result.error("xref-relation", f"Relation '{rel.id}': 'from' → unknown '{rel.from_}'")
        if rel.to not in memory_ids:
            result.error("xref-relation", f"Relation '{rel.id}': 'to' → unknown '{rel.to}'")
        if rel.from_ == rel.to:
            result.warning("xref-relation", f"Relation '{rel.id}': self-referencing")

    # Provenance conversation_ref → conversations_index
    for mem in store.memories:
        ref = mem.provenance.conversation_ref
        if ref and conv_ids and ref not in conv_ids:
            result.error(
                "xref-conversation",
                f"Memory '{mem.id}': conversation_ref='{ref}' not in conversations_index",
            )

        # superseded_by → memory
        sup = mem.temporal.superseded_by
        if sup and sup not in memory_ids:
            result.error(
                "xref-superseded",
                f"Memory '{mem.id}': superseded_by='{sup}' not found",
            )

    # derived_memories → memories
    for conv in store.conversations_index:
        for dm_id in conv.derived_memories:
            if dm_id not in memory_ids:
                result.error(
                    "xref-derived",
                    f"Conversation '{conv.id}': derived_memories → unknown '{dm_id}'",
                )

    # Bidirectional: conversation_ref ↔ derived_memories
    conv_derived: dict[str, set[str]] = {
        c.id: set(c.derived_memories) for c in store.conversations_index
    }
    for mem in store.memories:
        ref = mem.provenance.conversation_ref
        if ref and ref in conv_derived and mem.id not in conv_derived[ref]:
            result.warning(
                "xref-bidirectional",
                f"Memory '{mem.id}' refs conversation '{ref}', "
                f"but derived_memories doesn't include it",
            )


def _check_temporal_consistency(store: MemoryStore, result: ValidationResult) -> None:
    for mem in store.memories:
        t = mem.temporal
        if t.updated_at and t.updated_at < t.created_at:
            result.error("temporal", f"Memory '{mem.id}': updated_at < created_at")
        if t.valid_from and t.valid_until and t.valid_until < t.valid_from:
            result.error("temporal", f"Memory '{mem.id}': valid_until < valid_from")

        if (
            mem.confidence
            and mem.confidence.last_reinforced
            and mem.confidence.last_reinforced < t.created_at
        ):
            result.warning("temporal", f"Memory '{mem.id}': last_reinforced < created_at")

    for conv in store.conversations_index:
        ct = conv.temporal
        if ct.updated_at and ct.updated_at < ct.created_at:
            result.error("temporal", f"Conversation '{conv.id}': updated_at < created_at")


def _check_custom_type_rules(store: MemoryStore, result: ValidationResult) -> None:
    for mem in store.memories:
        if mem.type == "custom" and not mem.custom_type:
            result.error("custom-type", f"Memory '{mem.id}': type='custom' but no custom_type")
        if mem.type != "custom" and mem.custom_type is not None:
            result.error(
                "custom-type",
                f"Memory '{mem.id}': type='{mem.type}' but custom_type='{mem.custom_type}'",
            )


def _check_status_consistency(store: MemoryStore, result: ValidationResult) -> None:
    for mem in store.memories:
        sup = mem.temporal.superseded_by
        if mem.status == "superseded" and not sup:
            result.warning(
                "status",
                f"Memory '{mem.id}': status='superseded' but superseded_by not set",
            )
        if sup and mem.status != "superseded":
            result.warning(
                "status",
                f"Memory '{mem.id}': superseded_by set but status='{mem.status}'",
            )


def _check_message_dag(conversation: Conversation, result: ValidationResult) -> None:
    msg_ids = {m.id for m in conversation.messages}
    msg_map = {m.id: m for m in conversation.messages}

    for msg in conversation.messages:
        if msg.parent_id and msg.parent_id not in msg_ids:
            result.warning("dag", f"Message '{msg.id}': parent_id='{msg.parent_id}' not found")

        for child_id in msg.children_ids:
            if child_id not in msg_ids:
                result.warning("dag", f"Message '{msg.id}': child '{child_id}' not found")
            elif msg_map[child_id].parent_id != msg.id:
                result.warning(
                    "dag-bidirectional",
                    f"Message '{msg.id}' → child '{child_id}', but child's parent_id"
                    f"='{msg_map[child_id].parent_id}'",
                )
