"""Anthropic (Claude) export converter.

Handles:
- conversations.json (conversation history with chat_messages)
- memories.json (conversations_memory + project_memories)

See importer-mappings.md §2 for field-by-field mapping details.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from portable_ai_memory.converters.base import BaseConverter, parse_iso_or_now, register_converter
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
from portable_ai_memory.models.enums import (
    AttachmentType,
    ContentType,
    MemoryType,
    MessageRole,
)
from portable_ai_memory.models.memory_store import (
    MemoryObject,
    MemoryStore,
    Owner,
)


@register_converter
class AnthropicConverter(BaseConverter):
    """Convert Claude/Anthropic exports to PAM format."""

    @property
    def provider_name(self) -> str:
        return "claude"

    def detect(self, data: dict[str, Any] | list[Any], path: Path | None = None) -> bool:
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict) and "chat_messages" in first:
                return True
        return False

    def convert_conversations(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
        import_metadata: ImportMetadata | None = None,
    ) -> list[Conversation]:
        if not isinstance(data, list):
            return []

        conversations: list[Conversation] = []
        for raw_conv in data:
            conv = self._convert_single_conversation(
                raw_conv, owner_id=owner_id, import_metadata=import_metadata
            )
            if conv is not None:
                conversations.append(conv)
        return conversations

    def convert_memories(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
    ) -> MemoryStore:
        """Convert memories.json to MemoryStore.

        Expects the memories.json format: list with single object containing
        conversations_memory (string) and project_memories (dict).
        """
        memories: list[MemoryObject] = []

        if isinstance(data, list) and len(data) > 0:
            mem_data = data[0]
        elif isinstance(data, dict):
            mem_data = data
        else:
            mem_data = {}

        # conversations_memory: single text block
        conv_memory = mem_data.get("conversations_memory", "")
        if conv_memory and conv_memory.strip():
            memories.append(
                MemoryObject.create(
                    id=f"claude-mem-{uuid4().hex[:8]}",
                    type=MemoryType.CONTEXT,
                    content=conv_memory.strip(),
                    platform="claude",
                )
            )

        # project_memories: dict of uuid → text
        project_mems = mem_data.get("project_memories", {})
        if isinstance(project_mems, dict):
            for proj_id, proj_text in project_mems.items():
                if not proj_text or not str(proj_text).strip():
                    continue
                memories.append(
                    MemoryObject.create(
                        id=f"claude-proj-{proj_id[:8]}",
                        type=MemoryType.PROJECT,
                        content=str(proj_text).strip(),
                        platform="claude",
                    )
                )

        return MemoryStore(
            schema_version="1.0",
            owner=Owner(id=owner_id),
            memories=memories,
        )

    def extract_owner_info(
        self,
        source_dir: Path,
        owner_id: str = "unknown",
    ) -> tuple[str, list[MemoryObject]]:
        """Read users.json, memories.json, projects.json for owner info and memories."""
        import json

        memories: list[MemoryObject] = []

        # users.json → owner_id + profile memory
        users_file = source_dir / "users.json"
        if users_file.exists():
            users_data = json.loads(users_file.read_bytes())
            if isinstance(users_data, list) and users_data:
                user = users_data[0]
                if owner_id == "unknown":
                    uid = user.get("uuid")
                    if uid:
                        owner_id = uid

                full_name = user.get("full_name")
                profile_parts: list[str] = []
                if full_name:
                    profile_parts.append(f"Name: {full_name}")
                email = user.get("email_address")
                if email:
                    profile_parts.append(f"Email: {email}")
                if profile_parts:
                    content = "Owner profile from Claude export.\n" + "\n".join(profile_parts)
                    memories.append(
                        MemoryObject.create(
                            id="claude-user-profile",
                            type=MemoryType.CONTEXT,
                            content=content,
                            platform="claude",
                        )
                    )

        # memories.json → MemoryStore memories
        mem_file = source_dir / "memories.json"
        if mem_file.exists():
            mem_data = json.loads(mem_file.read_bytes())
            store = self.convert_memories(mem_data, owner_id=owner_id)
            memories.extend(store.memories)

        # projects.json → project doc memories
        proj_file = source_dir / "projects.json"
        if proj_file.exists():
            proj_data = json.loads(proj_file.read_bytes())
            if isinstance(proj_data, list):
                proj_mems = self.convert_project_docs(proj_data, owner_id=owner_id)
                memories.extend(proj_mems)

        return owner_id, memories

    def convert_project_docs(
        self,
        data: list[Any],
        *,
        owner_id: str = "unknown",
    ) -> list[MemoryObject]:
        """Convert projects.json docs to MemoryObject list.

        Each project doc with non-empty content becomes a PROJECT memory.
        """
        memories: list[MemoryObject] = []

        for project in data:
            if not isinstance(project, dict):
                continue
            proj_name = project.get("name", "")
            for doc in project.get("docs", []):
                content = doc.get("content", "")
                if not content or not content.strip():
                    continue
                content = content.strip()
                filename = doc.get("filename", "")
                doc_id = doc.get("uuid", uuid4().hex[:8])
                label = (
                    f"{proj_name}: {filename}"
                    if proj_name and filename
                    else (filename or proj_name)
                )
                memories.append(
                    MemoryObject.create(
                        id=f"claude-projdoc-{doc_id[:8]}",
                        type=MemoryType.PROJECT,
                        content=content,
                        summary=label or None,
                        platform="claude",
                    )
                )

        return memories

    def _convert_single_conversation(
        self,
        raw: dict[str, Any],
        *,
        owner_id: str,
        import_metadata: ImportMetadata | None = None,
    ) -> Conversation | None:
        conv_id = raw.get("uuid") or str(uuid4())
        title = raw.get("name")
        created_at = parse_iso_or_now(raw.get("created_at"))
        updated_at_str = raw.get("updated_at")
        updated_at = datetime.fromisoformat(updated_at_str) if updated_at_str else None

        chat_messages = raw.get("chat_messages", [])
        messages = self._convert_messages(chat_messages)
        if not messages:
            return None

        account_id = None
        account = raw.get("account")
        if isinstance(account, dict):
            account_id = account.get("uuid")

        raw_metadata: dict[str, Any] = {}
        summary = raw.get("summary")
        if summary:
            raw_metadata["summary"] = summary

        return Conversation(
            schema_version="1.0",
            id=conv_id,
            provider=ProviderInfo(
                name="claude",
                conversation_id=conv_id,
                account_id=account_id,
            ),
            title=title,
            temporal=ConversationTemporal(
                created_at=created_at,
                updated_at=updated_at,
            ),
            participants=[
                Participant(role=MessageRole.USER),
                Participant(role=MessageRole.ASSISTANT),
            ],
            messages=messages,
            import_metadata=import_metadata,
            raw_metadata=raw_metadata,
        )

    def _convert_messages(self, chat_messages: list[Any]) -> list[Message]:
        messages: list[Message] = []

        for msg_data in chat_messages:
            msg_id = msg_data.get("uuid", str(uuid4()))
            sender = msg_data.get("sender", "human")
            role = self._normalize_role(sender)

            created_at = parse_iso_or_now(msg_data.get("created_at"))

            # Extract text and structured content
            text = msg_data.get("text", "")
            content_parts = msg_data.get("content", [])

            # Process structured content for tool calls, citations, thinking
            tool_calls: list[ToolCall] = []
            citations: list[Citation] = []
            is_thought = False
            raw_metadata: dict[str, Any] = {}

            if isinstance(content_parts, list):
                raw_content_parts: list[dict[str, Any]] = []
                for part in content_parts:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")

                    # Preserve full content part for raw_metadata
                    part_extras = _extract_extras(part, part_type)
                    if part_extras:
                        raw_content_parts.append(part_extras)

                    if part_type == "thinking":
                        is_thought = True
                        if not text:
                            text = part.get("thinking", "")

                    elif part_type == "tool_use":
                        tool_calls.append(
                            ToolCall(
                                id=part.get("id"),
                                name=part.get("name", "unknown"),
                                input=part.get("input"),
                            )
                        )

                    elif part_type == "tool_result":
                        # Extract citations from knowledge content
                        nested_content = part.get("content", [])
                        if isinstance(nested_content, list):
                            for nc in nested_content:
                                if isinstance(nc, dict) and nc.get("type") == "knowledge":
                                    citations.append(
                                        Citation(
                                            title=nc.get("title"),
                                            url=nc.get("url"),
                                        )
                                    )

                    elif part_type == "text":
                        # Extract citations from text parts
                        text_citations = part.get("citations", [])
                        if isinstance(text_citations, list):
                            for tc in text_citations:
                                if isinstance(tc, dict):
                                    citations.append(
                                        Citation(
                                            title=tc.get("title"),
                                            url=tc.get("url"),
                                            snippet=tc.get("snippet"),
                                        )
                                    )

                if raw_content_parts:
                    raw_metadata["content_parts"] = raw_content_parts

            # Extract attachments from files[] and attachments[]
            attachments: list[Attachment] = []
            for f in msg_data.get("files", []):
                fname = f.get("file_name", "")
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                att_type = (
                    AttachmentType.IMAGE
                    if ext in {"png", "jpg", "jpeg", "gif", "webp", "svg", "bmp"}
                    else AttachmentType.FILE
                )
                attachments.append(Attachment(type=att_type, name=fname or None))

            for att in msg_data.get("attachments", []):
                fname = att.get("file_name") or None
                size = att.get("file_size")
                extracted = att.get("extracted_content")
                a = Attachment(
                    type=AttachmentType.DOCUMENT,
                    name=fname if fname else None,
                    size_bytes=size,
                )
                if extracted:
                    extracts = raw_metadata.setdefault("attachment_extracts", [])
                    if isinstance(extracts, list):
                        extracts.append({"file_name": fname, "extracted_content": extracted})
                attachments.append(a)

            updated_at = msg_data.get("updated_at")
            if updated_at:
                raw_metadata["updated_at"] = updated_at

            # NOTE: Messages are appended even when text is empty. Claude exports
            # include file-only uploads (no text, only attachments) and error
            # responses with metadata. Filtering these would lose real data.
            msg = Message(
                id=msg_id,
                provider_message_id=msg_id,
                role=role,
                content=MessageContent(type=ContentType.TEXT, text=text or None),
                created_at=created_at,
                is_thought=is_thought,
                tool_calls=tool_calls,
                citations=citations,
                attachments=attachments,
                raw_metadata=raw_metadata,
            )
            messages.append(msg)

        return messages

    @staticmethod
    def _normalize_role(sender: str) -> MessageRole:
        if sender == "human":
            return MessageRole.USER
        return MessageRole.ASSISTANT


# Keys already mapped to PAM fields per content type — no need to duplicate in raw_metadata
_MAPPED_KEYS: dict[str | None, set[str]] = {
    "text": {"type", "text", "citations"},
    "thinking": {"type", "thinking"},
    "tool_use": {"type", "id", "name", "input"},
    "tool_result": {"type"},
    "token_budget": {"type"},
    None: {"type"},
}


def _extract_extras(
    part: dict[str, Any],
    part_type: str | None,
) -> dict[str, Any]:
    """Extract non-null fields from a content part that aren't already mapped to PAM fields."""
    mapped = _MAPPED_KEYS.get(part_type, _MAPPED_KEYS[None])
    extras: dict[str, Any] = {}
    for k, v in part.items():
        if k in mapped:
            continue
        if v is None:
            continue
        extras[k] = v
    if extras:
        extras["type"] = part_type
    return extras
