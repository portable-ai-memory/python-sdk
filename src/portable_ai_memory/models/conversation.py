"""Pydantic models for the PAM Conversation schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from portable_ai_memory.models._config import STRICT
from portable_ai_memory.models.enums import (
    AttachmentType,
    ContentPartType,
    ContentType,
    MessageRole,
)
from portable_ai_memory.models.memory_store import (
    ConversationTemporal as ConversationTemporal,
)

# ── Sub-blocks ────────────────────────────────────────────


class ProviderInfo(BaseModel):
    """Source provider information (e.g., 'chatgpt', 'claude').

    Attributes:
        name: Provider identifier (lowercase, 2-32 chars).
        conversation_id: Provider's original conversation ID.
        account_id: Provider's account/user ID.
        export_format_version: Version of the provider's export format.
    """

    model_config = STRICT

    name: str = Field(min_length=2, max_length=32, pattern=r"^[a-z0-9_-]{2,32}$")
    conversation_id: str | None = None
    account_id: str | None = None
    export_format_version: str | None = None


class Participant(BaseModel):
    """A conversation participant (user, assistant, or system).

    Attributes:
        role: The participant's role (user, assistant, system, tool).
        name: Display name (e.g., 'Alice', 'Claude').
        provider_id: Provider's internal ID for this participant.
    """

    model_config = STRICT

    role: MessageRole
    name: str | None = None
    provider_id: str | None = None


class ContentPart(BaseModel):
    """A single part of a multipart message (text, code, image ref, etc.).

    Attributes:
        type: Part type (text, code, image, audio, video, file).
        text: Text content (for text/code parts).
        language: Programming language (for code parts).
        mime_type: MIME type (for binary parts).
        ref: Reference to external content (file path or URL).
    """

    model_config = STRICT

    type: ContentPartType
    text: str | None = None
    language: str | None = None
    mime_type: str | None = None
    ref: str | None = None


class MessageContent(BaseModel):
    """Content of a message — either plain text or multipart.

    Attributes:
        type: Content type ('text' for plain text, 'multipart' for structured).
        text: Plain text content (when type is 'text').
        parts: List of ContentPart elements (when type is 'multipart').
    """

    model_config = STRICT

    type: ContentType
    text: str | None = None
    parts: list[ContentPart] = Field(default_factory=list)


class Attachment(BaseModel):
    """File attached to a message (image, document, etc.).

    Attributes:
        type: Attachment type (image, document, audio, video, file).
        name: Original filename.
        mime_type: MIME type of the file.
        size_bytes: File size in bytes.
        ref: Path to the file in the PAM bundle.
        provider_id: Provider's internal attachment ID.
    """

    model_config = STRICT

    type: AttachmentType
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    ref: str | None = None
    provider_id: str | None = None


class Citation(BaseModel):
    """A web citation referenced in a message.

    Attributes:
        title: Page or document title.
        url: URL of the cited source.
        snippet: Relevant text excerpt from the source.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    url: str | None = None
    snippet: str | None = None


class ToolCall(BaseModel):
    """A function/tool call made by the assistant.

    Attributes:
        id: Tool call ID (provider-assigned).
        name: Function or tool name.
        input: Arguments passed to the tool (dict or raw string).
        output: Tool execution result.
    """

    model_config = STRICT

    id: str | None = None
    name: str = Field(min_length=1)
    input: dict[str, Any] | str | None = None
    output: str | None = None


_IMPORTER_RE = r"^[a-zA-Z0-9_-]+/[0-9]+\.[0-9]+\.[0-9]+$"


class ImportMetadata(BaseModel):
    """Provenance metadata recording how this conversation was imported.

    Attributes:
        importer: Tool that performed the import (e.g., 'pam-cli/1.0.0').
        importer_version: Deprecated — use importer field instead.
        imported_at: Timestamp of the import.
        source_file: Original filename that was converted.
        source_checksum: SHA-256 checksum of the source file.
    """

    model_config = STRICT

    importer: str | None = Field(default=None, pattern=_IMPORTER_RE)
    importer_version: str | None = None
    imported_at: datetime | None = None
    source_file: str | None = None
    source_checksum: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")


# ── Message ───────────────────────────────────────────────


class Message(BaseModel):
    """A single message in a conversation.

    Attributes:
        id: Unique message identifier.
        provider_message_id: Provider's original message ID.
        role: Who sent this message (user, assistant, system, tool).
        content: Message text or multipart content.
        created_at: When the message was created.
        parent_id: ID of the parent message (for branching conversations).
        children_ids: IDs of child messages.
        model: AI model used to generate this message.
        is_thought: True if this is an internal reasoning step (chain-of-thought).
        token_count: Number of tokens in the message.
        attachments: Files attached to this message.
        citations: Web sources cited in this message.
        tool_calls: Tool/function calls made in this message.
        raw_metadata: Provider-specific fields not mapped to PAM.
    """

    model_config = STRICT

    id: str = Field(min_length=1)
    provider_message_id: str | None = None
    role: MessageRole
    content: MessageContent | None = None
    created_at: datetime
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    model: str | None = None
    is_thought: bool = False
    token_count: int | None = Field(default=None, ge=0)
    attachments: list[Attachment] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


# ── Root document ─────────────────────────────────────────


class Conversation(BaseModel):
    """Root PAM Conversation document.

    Represents a full conversation with messages, participants, and metadata.

    Attributes:
        schema_version: PAM spec version (e.g., '1.0').
        id: Unique conversation identifier.
        provider: Source provider info (name, conversation_id).
        title: Conversation title or summary.
        temporal: Created/updated timestamps.
        participants: List of conversation participants.
        messages: Ordered list of messages.
        model: Default AI model for this conversation.
        system_instruction: System prompt used in this conversation.
        is_archived: Whether the conversation is archived.
        tags: User-defined tags.
        raw_metadata: Provider-specific fields not mapped to PAM.
        import_metadata: How this conversation was imported.
    """

    model_config = STRICT

    schema_: str = Field("portable-ai-memory-conversation", alias="schema")
    schema_version: str = Field(pattern=r"^[0-9]+\.[0-9]+(-(rc|alpha|beta)[0-9]*)?$")
    id: str = Field(min_length=1)
    provider: ProviderInfo
    title: str | None = None
    temporal: ConversationTemporal
    participants: list[Participant] = Field(default_factory=list)
    messages: list[Message]
    model: str | None = None
    system_instruction: str | None = None
    is_archived: bool = False
    tags: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    import_metadata: ImportMetadata | None = None
