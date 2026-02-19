"""OpenAI / ChatGPT export converter.

Handles:
- conversations.json (full conversation history)
- memory prompt (ChatGPT memory entries)

See importer-mappings.md §1 for field-by-field mapping details.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from portable_ai_memory.converters.base import BaseConverter, register_converter
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
from portable_ai_memory.models.enums import AttachmentType, ContentType, MessageRole


@register_converter
class OpenAIConverter(BaseConverter):
    """Convert ChatGPT/OpenAI exports to PAM format."""

    @property
    def provider_name(self) -> str:
        return "chatgpt"

    def detect(self, data: dict[str, Any] | list[Any], path: Path | None = None) -> bool:
        # ChatGPT conversations.json is a list of objects with 'mapping' key
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict) and "mapping" in first:
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

    def extract_owner_info(
        self,
        source_dir: Path,
        owner_id: str = "unknown",
    ) -> tuple[str, list[Any]]:
        """Read user.json and sora.json for owner info."""
        import json

        user_file = source_dir / "user.json"
        if user_file.exists():
            user_data = json.loads(user_file.read_bytes())
            if owner_id == "unknown":
                file_owner_id = user_data.get("id")
                if file_owner_id:
                    owner_id = file_owner_id

        return owner_id, []

    def post_process(
        self,
        conversations: list[Any],
        source_dir: Path,
        output_dir: Path,
    ) -> int:
        """Inject feedback and copy attachments."""
        import json

        from portable_ai_memory.converters.openai_attachments import copy_attachments

        # Inject message_feedback.json
        feedback_file = source_dir / "message_feedback.json"
        if feedback_file.exists():
            feedback_data = json.loads(feedback_file.read_bytes())
            feedback_map: dict[str, dict[str, Any]] = {}
            for fb in feedback_data:
                msg_id = fb.get("message_id")
                if msg_id:
                    feedback_map[msg_id] = {
                        "rating": fb.get("rating"),
                        "tags": fb.get("tags", []),
                    }

            for conv in conversations:
                for msg in conv.messages:
                    if msg.provider_message_id and msg.provider_message_id in feedback_map:
                        msg.raw_metadata["feedback"] = feedback_map[msg.provider_message_id]

        # Copy attachments
        return copy_attachments(conversations, source_dir, output_dir)

    # ── Internal ──────────────────────────────────────────

    def _convert_single_conversation(
        self,
        raw: dict[str, Any],
        *,
        owner_id: str,
        import_metadata: ImportMetadata | None = None,
    ) -> Conversation | None:
        conv_id = raw.get("id") or str(uuid4())
        title = raw.get("title")
        create_time = raw.get("create_time")
        update_time = raw.get("update_time")

        # Parse timestamps (OpenAI uses Unix epoch floats)
        created_at = (
            datetime.fromtimestamp(create_time, tz=UTC) if create_time else datetime.now(tz=UTC)
        )
        updated_at = datetime.fromtimestamp(update_time, tz=UTC) if update_time else None

        # Extract messages from mapping (DAG structure)
        mapping = raw.get("mapping", {})

        # Extract user_editable_context as system_instruction
        system_instruction = None
        for node in mapping.values():
            msg_data = node.get("message")
            if msg_data is None:
                continue
            content_obj = msg_data.get("content", {})
            if content_obj.get("content_type") == "user_editable_context":
                parts = []
                if ui := content_obj.get("user_instructions"):
                    parts.append(f"Custom instructions:\n{ui}")
                if up := content_obj.get("user_profile"):
                    parts.append(f"User profile:\n{up}")
                if parts:
                    system_instruction = "\n\n".join(parts)
                break

        messages = self._convert_messages(mapping)

        # Determine model
        model = raw.get("default_model_slug")
        is_archived = raw.get("is_archived", False)

        return Conversation(
            schema_version="1.0",
            id=conv_id,
            is_archived=is_archived,
            provider=ProviderInfo(
                name="chatgpt",
                conversation_id=conv_id,
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
            model=model,
            import_metadata=import_metadata,
            system_instruction=system_instruction,
        )

    def _convert_messages(self, mapping: dict[str, Any]) -> list[Message]:
        """Convert OpenAI's mapping dict (DAG) to PAM messages."""
        messages: list[Message] = []

        for node_id, node in mapping.items():
            msg_data = node.get("message")
            if msg_data is None:
                continue

            # Skip system-injected context (user profile, custom instructions)
            if msg_data.get("metadata", {}).get("is_visually_hidden_from_conversation", False):
                continue

            author = msg_data.get("author", {})
            role_raw = author.get("role", "user")
            role = self._normalize_role(role_raw)

            # Skip system messages with no content
            content_obj = msg_data.get("content", {})
            content_parts = content_obj.get("parts", [])
            text = ""
            attachments: list[Attachment] = []
            citations: list[Citation] = []
            is_thought = False
            content_type_raw = content_obj.get("content_type", "text")

            if content_type_raw == "thoughts":
                thoughts_list = content_obj.get("thoughts", [])
                text = "\n".join(t.get("content", "") for t in thoughts_list if t.get("content"))
                is_thought = True
            elif content_type_raw == "reasoning_recap":
                text = content_obj.get("content", "")
                is_thought = True
            elif content_parts:
                text_parts: list[str] = []
                for part in content_parts:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        if part.get("content_type") == "image_asset_pointer":
                            attachments.append(
                                Attachment(
                                    type=AttachmentType.IMAGE,
                                    ref=part.get("asset_pointer"),
                                    size_bytes=part.get("size_bytes"),
                                    provider_id=part.get("asset_pointer"),
                                )
                            )
                        elif part.get("content_type") == "sonic_webpage":
                            if page_text := part.get("text"):
                                text_parts.append(page_text)
                            citations.append(
                                Citation(
                                    title=part.get("title"),
                                    url=part.get("url"),
                                    snippet=part.get("snippet"),
                                )
                            )
                        elif part.get("content_type") == "tether_browsing_display":
                            if result := part.get("result"):
                                text_parts.append(result)
                        elif part.get("content_type") == "tether_quote":
                            if quote_text := part.get("text"):
                                text_parts.append(quote_text)
                            citations.append(
                                Citation(
                                    title=part.get("title"),
                                    url=part.get("url"),
                                    snippet=part.get("text"),
                                )
                            )
                text = "\n".join(text_parts)

            # Check metadata for reasoning status
            if msg_data.get("metadata", {}).get("reasoning_status") == "is_reasoning":
                is_thought = True

            # Extract metadata attachments (non-image files)
            for att in msg_data.get("metadata", {}).get("attachments", []):
                attachments.append(
                    Attachment(
                        type=AttachmentType.FILE,
                        name=att.get("name"),
                        mime_type=att.get("mime_type"),
                        size_bytes=att.get("size"),
                        provider_id=att.get("id"),
                    )
                )

            # Extract citations from metadata
            for cit in msg_data.get("metadata", {}).get("citations", []):
                cit_meta = cit.get("metadata", {})
                citations.append(
                    Citation(
                        title=cit_meta.get("title"),
                        url=cit_meta.get("url"),
                        snippet=cit_meta.get("text"),
                    )
                )

            # Extract citations from search result groups
            for group in msg_data.get("metadata", {}).get("search_result_groups", []):
                for entry in group.get("entries", []):
                    citations.append(
                        Citation(
                            title=entry.get("title"),
                            url=entry.get("url"),
                            snippet=entry.get("snippet"),
                        )
                    )

            create_time = msg_data.get("create_time")
            created_at = (
                datetime.fromtimestamp(create_time, tz=UTC) if create_time else datetime.now(tz=UTC)
            )

            # Build tool_calls for assistant→tool or tool response messages
            tool_calls: list[ToolCall] = []
            recipient = msg_data.get("recipient")
            if role == MessageRole.ASSISTANT and recipient and recipient != "all":
                tool_calls.append(
                    ToolCall(
                        name=recipient,
                        input=text or None,
                        id=msg_data.get("id"),
                    )
                )
            elif role == MessageRole.TOOL:
                tool_name = author.get("name", "unknown")
                tool_calls.append(
                    ToolCall(
                        name=tool_name,
                        output=text or None,
                        id=msg_data.get("id"),
                    )
                )

            # Skip truly empty messages (no content of any kind)
            if not text and not attachments and not citations and not tool_calls:
                continue

            msg = Message(
                id=msg_data.get("id", node_id),
                provider_message_id=msg_data.get("id"),
                role=role,
                content=MessageContent(type=ContentType.TEXT, text=text or None),
                created_at=created_at,
                parent_id=node.get("parent"),
                children_ids=node.get("children", []),
                model=msg_data.get("metadata", {}).get("model_slug"),
                attachments=attachments,
                citations=citations,
                tool_calls=tool_calls or [],
                is_thought=is_thought,
            )
            messages.append(msg)

        # Clean up DAG references: remove parent_id/children_ids pointing to
        # nodes that were filtered out (weight=0, hidden, non-message nodes)
        msg_ids = {m.id for m in messages}
        for msg in messages:
            if msg.parent_id and msg.parent_id not in msg_ids:
                msg.parent_id = None
            msg.children_ids = [c for c in msg.children_ids if c in msg_ids]

        return messages

    @staticmethod
    def _normalize_role(role: str) -> MessageRole:
        """Normalize OpenAI role strings to PAM roles."""
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "tool": MessageRole.TOOL,
            "human": MessageRole.USER,
        }
        return role_map.get(role, MessageRole.USER)
