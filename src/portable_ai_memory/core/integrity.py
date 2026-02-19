"""Integrity utilities — hashing, canonicalization, and checksum computation.

Implements the deterministic operations defined in PAM spec §17.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

import rfc8785


def normalize_content(content: str) -> str:
    """Normalize content for hash computation (spec §9.4).

    Steps: trim whitespace, lowercase, NFC unicode normalize, collapse whitespace.
    """
    text = content.strip()
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 of normalized content.

    Content is normalized (lowercased, whitespace collapsed, NFC unicode)
    before hashing.

    Args:
        content: Raw text content of a memory.

    Returns:
        Hash string in the format ``'sha256:<64 hex chars>'``.
    """
    normalized = normalize_content(content)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def canonicalize_jcs(obj: object) -> bytes:
    """RFC 8785 (JCS) canonicalization.

    Uses the rfc8785 library for spec-compliant deterministic JSON serialization.
    """
    return rfc8785.dumps(obj)  # type: ignore[arg-type]


def compute_integrity_checksum(memories: list[dict[str, Any]]) -> str:
    """Compute integrity checksum for a memories array (spec §17).

    Memories are sorted by id ascending, then canonicalized per RFC 8785.
    Returns 'sha256:<hex>'.
    """
    sorted_memories = sorted(memories, key=lambda m: m.get("id", ""))
    canonical = canonicalize_jcs(sorted_memories)
    digest = hashlib.sha256(canonical).hexdigest()
    return f"sha256:{digest}"


def verify_content_hash(content: str, declared_hash: str) -> bool:
    """Verify a content hash matches the actual content.

    Args:
        content: Raw text content to verify.
        declared_hash: The declared hash to check against (``'sha256:<hex>'``).

    Returns:
        True if the computed hash matches the declared hash.
    """
    return compute_content_hash(content) == declared_hash


def verify_integrity(memories: list[dict[str, Any]], declared_checksum: str) -> bool:
    """Verify the integrity checksum of a memories array.

    Args:
        memories: List of memory dicts (JSON-serialized format).
        declared_checksum: The declared checksum to check against.

    Returns:
        True if the computed checksum matches the declared checksum.
    """
    return compute_integrity_checksum(memories) == declared_checksum
