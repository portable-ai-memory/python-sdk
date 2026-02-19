"""Pydantic models for the PAM Memory Store schema.

Every model maps 1:1 to a $defs entry in portable-ai-memory.schema.json.
Field names use snake_case (Pydantic default) and alias to camelCase where
the JSON schema uses a different convention.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from portable_ai_memory.models._config import STRICT
from portable_ai_memory.models.enums import (
    Canonicalization,
    DecayModel,
    ExportType,
    ExtractionMethod,
    MemoryStatus,
    MemoryType,
    Permission,
    RelationType,
    SignatureAlgorithm,
    StorageType,
    Visibility,
)

_PLATFORM_RE = re.compile(r"^[a-z0-9_-]{2,32}$")
_HASH_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_DID_RE = re.compile(r"^did:[a-z0-9]+:.+$")
_EXPORTED_BY_RE = re.compile(r"^[a-zA-Z0-9_-]+/[0-9]+\.[0-9]+\.[0-9]+$")
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+(-(rc|alpha|beta)[0-9]*)?$")


# ── Sub-blocks ────────────────────────────────────────────


class Owner(BaseModel):
    """The individual who owns these memories (spec §6).

    Attributes:
        id: Unique owner identifier (e.g., user UUID or email).
        did: Decentralized Identifier (DID) for portable identity.
        created_at: When the owner record was created.
    """

    model_config = STRICT

    id: str = Field(min_length=1)
    did: str | None = Field(default=None, pattern=_DID_RE.pattern)
    created_at: datetime | None = None


class ConfidenceBlock(BaseModel):
    """System-computed confidence scoring (spec §10).

    Attributes:
        initial: Confidence score at creation (0.0-1.0).
        current: Current confidence score after decay/reinforcement (0.0-1.0).
        decay_model: How confidence decays over time (exponential, linear, step, none).
        last_reinforced: When this memory was last confirmed or reinforced.
    """

    model_config = STRICT

    initial: float | None = Field(default=None, ge=0.0, le=1.0)
    current: float | None = Field(default=None, ge=0.0, le=1.0)
    decay_model: DecayModel | None = None
    last_reinforced: datetime | None = None


class TemporalBlock(BaseModel):
    """Temporal metadata for a memory (spec §9).

    Attributes:
        created_at: When the memory was first created.
        updated_at: When the memory was last modified.
        valid_from: Start of the memory's validity window.
        valid_until: End of the memory's validity window.
        superseded_by: ID of the memory that replaces this one.
    """

    model_config = STRICT

    created_at: datetime
    updated_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    superseded_by: str | None = None


class ProvenanceBlock(BaseModel):
    """Origin tracking — where and how a memory was extracted (spec §11).

    Attributes:
        platform: Source platform (e.g., 'chatgpt', 'claude').
        platform_user_id: User ID on the source platform.
        conversation_ref: ID of the conversation this memory came from.
        message_ref: ID of the specific message.
        extraction_method: How the memory was extracted (api_export, user_input, etc.).
        extracted_at: When the extraction happened.
        extractor: Tool that performed extraction (e.g., 'pam-cli/1.0.0').
    """

    model_config = STRICT

    platform: str = Field(min_length=2, max_length=32, pattern=_PLATFORM_RE.pattern)
    platform_user_id: str | None = None
    conversation_ref: str | None = None
    message_ref: str | None = None
    extraction_method: ExtractionMethod | None = None
    extracted_at: datetime | None = None
    extractor: str | None = Field(default=None, pattern=_EXPORTED_BY_RE.pattern)


class AccessGrant(BaseModel):
    """Single access grant entry (spec §12)."""

    model_config = STRICT

    entity: str = Field(min_length=1)
    permissions: list[Permission] = Field(min_length=1)


class AccessBlock(BaseModel):
    """Access control (spec §12)."""

    model_config = STRICT

    visibility: Visibility = Visibility.PRIVATE
    exportable: bool = True
    shared_with: list[AccessGrant] = Field(default_factory=list)


class MetadataBlock(BaseModel):
    """Extensible metadata (spec §13). Allows arbitrary extra fields."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    language: str | None = None
    domain: str | None = None


class StorageReference(BaseModel):
    """Reference to external storage (spec §15)."""

    model_config = STRICT

    type: StorageType
    ref: str = Field(min_length=1)
    format: str | None = None


# ── Core objects ──────────────────────────────────────────


class MemoryObject(BaseModel):
    """A single portable memory unit (spec §7).

    Represents one piece of knowledge about the user (preference, fact, context, etc.).
    Use ``MemoryObject.create()`` for convenient construction with auto-filled fields.

    Attributes:
        id: Unique memory identifier.
        type: Memory type (preference, fact, context, instruction, skill, project, custom).
        custom_type: Required when type is 'custom'.
        status: Memory status (active, archived, superseded, deleted).
        content: The memory text content.
        content_hash: SHA-256 hash of normalized content ('sha256:<hex>').
        summary: Optional short summary.
        tags: User-defined tags for categorization.
        confidence: Confidence scoring and decay model.
        temporal: Created/updated/validity timestamps.
        provenance: Where and how this memory was extracted.
        access: Visibility and sharing permissions.
        embedding_ref: Reference to an embedding in the embeddings file.
        metadata: Extensible metadata (language, domain, custom fields).
    """

    model_config = STRICT

    id: str = Field(min_length=1)
    type: MemoryType
    custom_type: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=_HASH_RE.pattern)
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    confidence: ConfidenceBlock | None = None
    temporal: TemporalBlock
    provenance: ProvenanceBlock
    access: AccessBlock | None = None
    embedding_ref: str | None = None
    metadata: MetadataBlock | None = None

    @model_validator(mode="after")
    def _validate_custom_type(self) -> MemoryObject:
        if self.type == MemoryType.CUSTOM and not self.custom_type:
            msg = "custom_type is required when type is 'custom'"
            raise ValueError(msg)
        if self.type != MemoryType.CUSTOM and self.custom_type is not None:
            msg = f"custom_type must be null when type is '{self.type}'"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_tags(self) -> MemoryObject:
        for tag in self.tags:
            if not _TAG_RE.match(tag):
                msg = f"Invalid tag format: '{tag}'. Must match {_TAG_RE.pattern}"
                raise ValueError(msg)
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        type: MemoryType,
        content: str,
        platform: str,
        extraction_method: ExtractionMethod = ExtractionMethod.API_EXPORT,
        tags: list[str] | None = None,
        summary: str | None = None,
        custom_type: str | None = None,
    ) -> MemoryObject:
        """Factory that auto-fills content_hash, temporal, and provenance.

        Args:
            id: Unique memory identifier.
            type: Memory type (preference, fact, context, custom, etc.).
            content: The memory text content.
            platform: Source platform name (e.g., 'claude', 'chatgpt').
            extraction_method: How the memory was extracted.
            tags: Optional list of tags.
            summary: Optional short summary.
            custom_type: Required when type is CUSTOM, ignored otherwise.
        """
        from portable_ai_memory.core.integrity import compute_content_hash

        now = datetime.now(tz=UTC)
        return cls(
            id=id,
            type=type,
            content=content,
            content_hash=compute_content_hash(content),
            summary=summary,
            tags=tags or [],
            custom_type=custom_type,
            temporal=TemporalBlock(created_at=now),
            provenance=ProvenanceBlock(
                platform=platform,
                extraction_method=extraction_method,
                extracted_at=now,
            ),
        )


class RelationObject(BaseModel):
    """Semantic relationship between two memories (spec §14).

    Attributes:
        id: Unique relation identifier.
        from_: Source memory ID (aliased as 'from' in JSON).
        to: Target memory ID.
        type: Relation type (related_to, derived_from, contradicts, etc.).
        confidence: Relation confidence score (0.0-1.0).
        created_at: When the relation was created.
    """

    model_config = STRICT

    id: str = Field(min_length=1)
    from_: str = Field(min_length=1, alias="from")
    to: str = Field(min_length=1)
    type: RelationType
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime


class ConversationTemporal(BaseModel):
    """Temporal block for conversation index entries."""

    model_config = STRICT

    created_at: datetime
    updated_at: datetime | None = None


class ConversationIndexEntry(BaseModel):
    """Lightweight conversation reference in the memory store (spec §15).

    Points to a full Conversation file stored separately in the bundle.

    Attributes:
        id: Conversation ID (matches the Conversation document's id).
        platform: Source platform (e.g., 'chatgpt', 'claude').
        title: Conversation title.
        message_count: Number of messages in the conversation.
        temporal: Created/updated timestamps.
        tags: User-defined tags.
        derived_memories: IDs of memories extracted from this conversation.
        storage: Reference to the conversation file in the bundle.
    """

    model_config = STRICT

    id: str = Field(min_length=1)
    platform: str = Field(min_length=2, max_length=32, pattern=_PLATFORM_RE.pattern)
    title: str | None = None
    message_count: int | None = Field(default=None, ge=0)
    temporal: ConversationTemporal
    tags: list[str] = Field(default_factory=list)
    derived_memories: list[str] = Field(default_factory=list)
    storage: StorageReference | None = None

    @model_validator(mode="after")
    def _validate_tags(self) -> ConversationIndexEntry:
        for tag in self.tags:
            if not _TAG_RE.match(tag):
                msg = f"Invalid tag format: '{tag}'. Must match {_TAG_RE.pattern}"
                raise ValueError(msg)
        return self


class IntegrityBlock(BaseModel):
    """Integrity verification block (spec §17).

    Contains a checksum over all memories for tamper detection.

    Attributes:
        canonicalization: JSON canonicalization method (default: RFC 8785).
        checksum: SHA-256 checksum of canonicalized memories ('sha256:<hex>').
        total_memories: Expected number of memories (cross-checked during validation).
    """

    model_config = STRICT

    canonicalization: Canonicalization = Canonicalization.RFC8785
    checksum: str = Field(pattern=_HASH_RE.pattern)
    total_memories: int = Field(ge=0)


class SignatureBlock(BaseModel):
    """Cryptographic signature (spec §18)."""

    model_config = STRICT

    algorithm: SignatureAlgorithm
    public_key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    signed_at: datetime
    key_id: str | None = None


# ── Root document ─────────────────────────────────────────


class MemoryStore(BaseModel):
    """Root PAM Memory Store document (spec §5).

    The top-level model for a PAM export — contains memories, relations,
    a conversation index, and optional integrity/signature blocks.

    Use convenience methods for common lookups::

        store = load("memory-store.json")
        mem = store.get_memory_by_id("mem-001")
        prefs = store.get_memories_by_type(MemoryType.PREFERENCE)
        tagged = store.get_memories_with_tag("important")

    Attributes:
        schema_version: PAM spec version (e.g., '1.0').
        owner: Who owns these memories.
        memories: List of memory objects.
        relations: Semantic relationships between memories.
        conversations_index: Lightweight references to conversation files.
        integrity: Checksum block for tamper detection.
        export_type: Full or incremental export.
        signature: Optional cryptographic signature.
    """

    model_config = STRICT

    schema_: str = Field("portable-ai-memory", alias="schema")
    schema_version: str = Field(pattern=_VERSION_RE.pattern)
    spec_uri: str | None = None
    export_id: str | None = None
    exported_by: str | None = Field(default=None, pattern=_EXPORTED_BY_RE.pattern)
    export_date: datetime | None = None
    owner: Owner
    memories: list[MemoryObject]
    relations: list[RelationObject] = Field(default_factory=list)
    conversations_index: list[ConversationIndexEntry] = Field(default_factory=list)
    integrity: IntegrityBlock | None = None
    export_type: ExportType = ExportType.FULL
    base_export_id: str | None = None
    since: datetime | None = None
    type_registry: str | None = None
    signature: SignatureBlock | None = None

    def get_memory_by_id(self, memory_id: str) -> MemoryObject | None:
        """Return a memory by ID, or None if not found."""
        for m in self.memories:
            if m.id == memory_id:
                return m
        return None

    def get_memories_by_type(self, type: MemoryType) -> list[MemoryObject]:
        """Return all memories matching the given type."""
        return [m for m in self.memories if m.type == type]

    def get_memories_with_tag(self, tag: str) -> list[MemoryObject]:
        """Return all memories that have the given tag."""
        return [m for m in self.memories if tag in m.tags]

    @model_validator(mode="after")
    def _validate_signature_requires_export_fields(self) -> MemoryStore:
        if self.signature is not None:
            if not self.export_id:
                msg = "export_id is required when signature is present"
                raise ValueError(msg)
            if not self.export_date:
                msg = "export_date is required when signature is present"
                raise ValueError(msg)
        return self
