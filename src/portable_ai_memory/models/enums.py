"""Enumerations matching the PAM specification."""

from __future__ import annotations

from enum import StrEnum


class MemoryType(StrEnum):
    """Closed taxonomy of memory types (spec §7)."""

    FACT = "fact"
    PREFERENCE = "preference"
    SKILL = "skill"
    CONTEXT = "context"
    RELATIONSHIP = "relationship"
    GOAL = "goal"
    INSTRUCTION = "instruction"
    IDENTITY = "identity"
    ENVIRONMENT = "environment"
    PROJECT = "project"
    CUSTOM = "custom"


class MemoryStatus(StrEnum):
    """Lifecycle status of a memory (spec §8)."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"
    RETRACTED = "retracted"
    ARCHIVED = "archived"


class DecayModel(StrEnum):
    """Confidence decay models (spec §10)."""

    TIME_LINEAR = "time_linear"
    TIME_EXPONENTIAL = "time_exponential"
    NONE = "none"


class ExtractionMethod(StrEnum):
    """How a memory was extracted from the source (spec §11.4)."""

    LLM_INFERENCE = "llm_inference"
    EXPLICIT_USER_INPUT = "explicit_user_input"
    API_EXPORT = "api_export"
    BROWSER_EXTRACTION = "browser_extraction"
    MANUAL = "manual"


class Visibility(StrEnum):
    """Access control visibility levels (spec §12)."""

    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"


class Permission(StrEnum):
    """Access grant permissions (spec §12)."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"


class RelationType(StrEnum):
    """Semantic relation types between memories (spec §14)."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    SUPERSEDES = "supersedes"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"


class StorageType(StrEnum):
    """External storage backend types (spec §15)."""

    FILE = "file"
    DATABASE = "database"
    OBJECT_STORAGE = "object_storage"
    VECTOR_DB = "vector_db"
    URI = "uri"


class ExportType(StrEnum):
    """Export types (spec §16)."""

    FULL = "full"
    INCREMENTAL = "incremental"


class SignatureAlgorithm(StrEnum):
    """Supported signature algorithms (spec §18)."""

    ED25519 = "Ed25519"
    ES256 = "ES256"
    ES384 = "ES384"
    RS256 = "RS256"
    RS384 = "RS384"
    RS512 = "RS512"


class Canonicalization(StrEnum):
    """Canonicalization methods (spec §17)."""

    RFC8785 = "RFC8785"


# ── Conversation enums ────────────────────────────────────


class MessageRole(StrEnum):
    """Normalized message roles (conversation spec)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(StrEnum):
    """Message content types."""

    TEXT = "text"
    MULTIPART = "multipart"


class ContentPartType(StrEnum):
    """Content part types for multipart messages."""

    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"


class AttachmentType(StrEnum):
    """Attachment types."""

    FILE = "file"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
