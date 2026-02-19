"""Core PAM operations — I/O, integrity, and validation."""

from portable_ai_memory.core.integrity import (
    compute_content_hash,
    compute_integrity_checksum,
    verify_content_hash,
    verify_integrity,
)
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
    validate_conversation,
    validate_embeddings,
    validate_memory_store,
)

__all__ = [
    "PAMDocument",
    "ValidationResult",
    "compute_content_hash",
    "compute_integrity_checksum",
    "load",
    "load_conversation",
    "load_dict",
    "load_embeddings",
    "load_memory_store",
    "save",
    "to_dict",
    "validate_conversation",
    "validate_embeddings",
    "validate_memory_store",
    "verify_content_hash",
    "verify_integrity",
]
