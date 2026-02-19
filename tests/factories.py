"""Shared test data factories for PAM test suite."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from portable_ai_memory.core.integrity import compute_content_hash
from portable_ai_memory.models.conversation import (
    Conversation,
    ConversationTemporal,
    Message,
    MessageContent,
    ProviderInfo,
)
from portable_ai_memory.models.enums import ContentType, MemoryType, MessageRole
from portable_ai_memory.models.memory_store import (
    MemoryObject,
    MemoryStore,
    Owner,
    ProvenanceBlock,
    TemporalBlock,
)


def make_memory(
    id: str = "m1",
    content: str = "hello world",
    **overrides: Any,
) -> MemoryObject:
    """Create a MemoryObject with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": id,
        "type": MemoryType.PREFERENCE,
        "content": content,
        "content_hash": compute_content_hash(content),
        "temporal": TemporalBlock(created_at=datetime(2024, 1, 1, tzinfo=UTC)),
        "provenance": ProvenanceBlock(platform="chatgpt"),
    }
    defaults.update(overrides)
    return MemoryObject(**defaults)


def make_store(
    memories: list[MemoryObject] | None = None,
    **overrides: Any,
) -> MemoryStore:
    """Create a MemoryStore with sensible defaults."""
    defaults: dict[str, Any] = {
        "schema_version": "1.0",
        "owner": Owner(id="test"),
        "memories": memories or [],
    }
    defaults.update(overrides)
    return MemoryStore(**defaults)


def make_conversation(
    id: str = "conv-1",
    title: str = "Test",
    **overrides: Any,
) -> Conversation:
    """Create a Conversation with sensible defaults."""
    defaults: dict[str, Any] = {
        "schema_version": "1.0",
        "id": id,
        "provider": ProviderInfo(name="chatgpt"),
        "title": title,
        "temporal": ConversationTemporal(created_at=datetime(2024, 1, 1, tzinfo=UTC)),
        "messages": [],
    }
    defaults.update(overrides)
    return Conversation(**defaults)


def make_message(
    id: str = "m1",
    role: MessageRole = MessageRole.USER,
    text: str = "hi",
    **overrides: Any,
) -> Message:
    """Create a Message with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": id,
        "role": role,
        "content": MessageContent(type=ContentType.TEXT, text=text),
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return Message(**defaults)
