"""PAM data models.

All models map 1:1 to the PAM JSON Schemas (Draft 2020-12).
"""

from portable_ai_memory.models.conversation import Conversation, Message, ProviderInfo
from portable_ai_memory.models.embeddings import EmbeddingObject, EmbeddingsFile
from portable_ai_memory.models.enums import (
    DecayModel,
    ExportType,
    ExtractionMethod,
    MemoryStatus,
    MemoryType,
    MessageRole,
    Permission,
    RelationType,
    SignatureAlgorithm,
    StorageType,
    Visibility,
)
from portable_ai_memory.models.memory_store import (
    AccessBlock,
    AccessGrant,
    ConfidenceBlock,
    ConversationIndexEntry,
    IntegrityBlock,
    MemoryObject,
    MemoryStore,
    MetadataBlock,
    Owner,
    ProvenanceBlock,
    RelationObject,
    SignatureBlock,
    StorageReference,
    TemporalBlock,
)

__all__ = [
    # Root documents
    "MemoryStore",
    "Conversation",
    "EmbeddingsFile",
    # Memory objects
    "MemoryObject",
    "RelationObject",
    "ConversationIndexEntry",
    "EmbeddingObject",
    # Blocks
    "Owner",
    "ConfidenceBlock",
    "TemporalBlock",
    "ProvenanceBlock",
    "AccessBlock",
    "AccessGrant",
    "MetadataBlock",
    "StorageReference",
    "IntegrityBlock",
    "SignatureBlock",
    # Conversation
    "Message",
    "ProviderInfo",
    # Enums
    "MemoryType",
    "MemoryStatus",
    "DecayModel",
    "ExtractionMethod",
    "Visibility",
    "Permission",
    "RelationType",
    "StorageType",
    "ExportType",
    "SignatureAlgorithm",
    "MessageRole",
]
