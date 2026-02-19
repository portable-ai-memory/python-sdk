"""Portable AI Memory (PAM) — Python SDK.

A universal interchange format for AI user memories.

Quick start:
    >>> from portable_ai_memory import load, validate_memory_store
    >>> store = load("my-export.json")
    >>> result = validate_memory_store(store)
    >>> print(result)

    >>> from portable_ai_memory.models import MemoryStore, MemoryObject
    >>> from portable_ai_memory.converters import detect_provider
"""

from __future__ import annotations

from importlib.metadata import version as _meta_version

__version__: str = _meta_version("portable-ai-memory")

# Exceptions
# Re-export the primary API
from portable_ai_memory.core.integrity import compute_content_hash, compute_integrity_checksum
from portable_ai_memory.core.io import (
    PAMDocument,
    load,
    load_conversation,
    load_dict,
    load_embeddings,
    load_memory_store,
    save,
    to_dict,
)
from portable_ai_memory.core.validator import (
    ValidationResult,
    validate,
    validate_conversation,
    validate_embeddings,
    validate_memory_store,
)
from portable_ai_memory.exceptions import (
    PAMError,
    PAMSchemaError,
    PAMValidationError,
    ProviderNotDetectedError,
)

# Models — re-export commonly used classes for convenient access
from portable_ai_memory.models.conversation import (
    Attachment,
    Citation,
    Conversation,
    ConversationTemporal,
    ImportMetadata,
    Message,
    MessageContent,
    Participant,
    ProviderInfo,
    ToolCall,
)
from portable_ai_memory.models.embeddings import EmbeddingsFile
from portable_ai_memory.models.enums import (
    AttachmentType,
    ContentType,
    ExtractionMethod,
    MemoryType,
    MessageRole,
    StorageType,
)
from portable_ai_memory.models.memory_store import (
    AccessBlock,
    ConfidenceBlock,
    ConversationIndexEntry,
    IntegrityBlock,
    MemoryObject,
    MemoryStore,
    Owner,
    ProvenanceBlock,
    StorageReference,
    TemporalBlock,
)

__all__ = [
    "__version__",
    # Exceptions
    "PAMError",
    "PAMSchemaError",
    "PAMValidationError",
    "ProviderNotDetectedError",
    # Types
    "PAMDocument",
    # I/O
    "load",
    "load_dict",
    "load_memory_store",
    "load_conversation",
    "load_embeddings",
    "save",
    "to_dict",
    # Validation
    "validate",
    "validate_memory_store",
    "validate_conversation",
    "validate_embeddings",
    "ValidationResult",
    # Integrity
    "compute_content_hash",
    "compute_integrity_checksum",
    # Models — Memory Store
    "MemoryStore",
    "MemoryObject",
    "Owner",
    "TemporalBlock",
    "ProvenanceBlock",
    "AccessBlock",
    "ConfidenceBlock",
    "IntegrityBlock",
    "ConversationIndexEntry",
    "StorageReference",
    # Models — Conversation
    "Conversation",
    "Message",
    "MessageContent",
    "Participant",
    "ProviderInfo",
    "ImportMetadata",
    "ConversationTemporal",
    "Attachment",
    "Citation",
    "ToolCall",
    # Models — Enums
    "MemoryType",
    "MessageRole",
    "ContentType",
    "AttachmentType",
    "ExtractionMethod",
    "StorageType",
    # Models — Embeddings
    "EmbeddingsFile",
]
