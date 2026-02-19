"""Microsoft (Copilot) CSV export converter.

Handles CSV exports from Microsoft Privacy Dashboard.

See importer-mappings.md §4 for field-by-field mapping details.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from portable_ai_memory.converters.base import BaseConverter, register_converter
from portable_ai_memory.models.conversation import (
    Conversation,
    ConversationTemporal,
    ImportMetadata,
    Message,
    MessageContent,
    Participant,
    ProviderInfo,
)
from portable_ai_memory.models.enums import ContentType, MessageRole

# Author values that map to USER role (case-insensitive)
_USER_AUTHORS = {"user", "human"}


@register_converter
class MicrosoftConverter(BaseConverter):
    """Convert Microsoft Copilot CSV exports to PAM format."""

    @property
    def provider_name(self) -> str:
        return "copilot"

    def detect(self, data: dict[str, Any] | list[Any], path: Path | None = None) -> bool:
        # CSV detection is file-based, not JSON-based
        # This will only match if data was pre-parsed from CSV
        return bool(path and path.suffix.lower() == ".csv")

    def detect_csv(self, header: list[str], path: Path | None = None) -> bool:
        """Detect Copilot CSV by header columns."""
        header_lower = [h.lower().strip().lstrip("\ufeff") for h in header]
        # Format A: Conversation, Time, Author, Message
        if "conversation" in header_lower and "message" in header_lower:
            return True
        # Format B: CreatedAt, MessageContent, Author, ChatName
        if "messagecontent" in header_lower and "chatname" in header_lower:
            return True
        # Format C: Timestamp, ClientApp, Prompt (Windows apps - prompt only)
        return "clientapp" in header_lower and "prompt" in header_lower

    def convert_conversations(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
        import_metadata: ImportMetadata | None = None,
    ) -> list[Conversation]:
        # For CSV, data should be passed as a list of dicts (pre-parsed rows)
        if not isinstance(data, list):
            return []

        return self._convert_rows(data, owner_id=owner_id, import_metadata=import_metadata)

    def convert_conversations_from_csv(
        self,
        csv_text: str,
        *,
        owner_id: str = "unknown",
        import_metadata: ImportMetadata | None = None,
        source_file: str | None = None,
    ) -> list[Conversation]:
        """Convert CSV text directly to PAM conversations."""
        reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
        rows = list(reader)
        return self._convert_rows(
            rows,
            owner_id=owner_id,
            import_metadata=import_metadata,
            source_file=source_file,
        )

    def _convert_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        owner_id: str,
        import_metadata: ImportMetadata | None = None,
        source_file: str | None = None,
    ) -> list[Conversation]:
        if not rows:
            return []

        # Detect format from available keys
        first = rows[0]
        keys = {k.lower().strip().lstrip("\ufeff"): k for k in first}

        is_format_a = "conversation" in keys and "message" in keys
        is_format_b = "chatname" in keys and "messagecontent" in keys
        is_format_c = "clientapp" in keys and "prompt" in keys

        # Format C (Windows apps) has only prompts, no responses — treat as single-message convos
        if is_format_c:
            return self._convert_windows_app_rows(
                rows,
                keys,
                owner_id=owner_id,
                import_metadata=import_metadata,
                source_file=source_file,
            )

        # Group rows by conversation
        groups: dict[str, list[dict[str, Any]]] = {}

        for row in rows:
            if is_format_a:
                conv_name = row.get(keys.get("conversation", "Conversation"), "")
                time_str = row.get(keys.get("time", "Time"), "")
                author = row.get(keys.get("author", "Author"), "")
                message = row.get(keys.get("message", "Message"), "")
            elif is_format_b:
                conv_name = row.get(keys.get("chatname", "ChatName"), "")
                time_str = row.get(keys.get("createdat", "CreatedAt"), "")
                author = row.get(keys.get("author", "Author"), "")
                message = row.get(keys.get("messagecontent", "MessageContent"), "")
            else:
                continue

            if not message or not message.strip():
                continue

            groups.setdefault(conv_name or "untitled", []).append(
                {
                    "time": time_str,
                    "author": author,
                    "message": message,
                }
            )

        conversations: list[Conversation] = []
        for conv_name, conv_rows in groups.items():
            conv = self._build_conversation(
                conv_name,
                conv_rows,
                owner_id=owner_id,
                import_metadata=import_metadata,
                source_file=source_file,
            )
            if conv is not None:
                conversations.append(conv)

        return conversations

    def _convert_windows_app_rows(
        self,
        rows: list[dict[str, Any]],
        keys: dict[str, Any],
        *,
        owner_id: str,
        import_metadata: ImportMetadata | None = None,
        source_file: str | None = None,
    ) -> list[Conversation]:
        """Convert Windows app prompt-only rows to single-message conversations."""
        conversations: list[Conversation] = []

        for row in rows:
            prompt = row.get(keys.get("prompt", "Prompt"), "")
            if not prompt or not prompt.strip():
                continue
            time_str = row.get(keys.get("timestamp", "Timestamp"), "")
            client_app = row.get(keys.get("clientapp", "ClientApp"), "")

            created_at = self._parse_timestamp(time_str)
            conv_id = str(uuid4())

            msg_metadata: dict[str, Any] = {}
            if client_app:
                msg_metadata["client_app"] = client_app
            if source_file:
                msg_metadata["source_file"] = source_file

            msg = Message(
                id=str(uuid4()),
                role=MessageRole.USER,
                content=MessageContent(type=ContentType.TEXT, text=prompt.strip()),
                created_at=created_at,
                raw_metadata=msg_metadata,
            )

            conv_metadata: dict[str, Any] = {}
            if source_file:
                conv_metadata["source_file"] = source_file

            conv = Conversation(
                schema_version="1.0",
                id=conv_id,
                provider=ProviderInfo(name="copilot", conversation_id=conv_id),
                title=f"{client_app} prompt" if client_app else None,
                temporal=ConversationTemporal(created_at=created_at),
                participants=[Participant(role=MessageRole.USER)],
                messages=[msg],
                import_metadata=import_metadata,
                raw_metadata=conv_metadata,
            )
            conversations.append(conv)

        return conversations

    def _build_conversation(
        self,
        conv_name: str,
        rows: list[dict[str, Any]],
        *,
        owner_id: str,
        import_metadata: ImportMetadata | None = None,
        source_file: str | None = None,
    ) -> Conversation | None:
        messages: list[Message] = []
        timestamps: list[datetime] = []

        for row in rows:
            time_str = row["time"]
            author = row["author"]
            text = row["message"]

            created_at = self._parse_timestamp(time_str)
            timestamps.append(created_at)

            role = self._normalize_role(author)

            # source_file is tracked at conversation level (raw_metadata), not per
            # message, since all messages in a conversation come from the same CSV.
            messages.append(
                Message(
                    id=str(uuid4()),
                    role=role,
                    content=MessageContent(type=ContentType.TEXT, text=text),
                    created_at=created_at,
                )
            )

        if not messages:
            return None

        # Sort: by timestamp, then USER before ASSISTANT for identical timestamps
        # (Copilot CSV may store AI response before Human prompt in each pair)
        role_order = {MessageRole.USER: 0, MessageRole.ASSISTANT: 1}
        messages.sort(key=lambda m: (m.created_at, role_order.get(m.role, 2)))

        conv_id = str(uuid4())
        created_at = min(timestamps) if timestamps else datetime.now(tz=UTC)
        updated_at = max(timestamps) if len(timestamps) > 1 else None

        raw_metadata: dict[str, Any] = {}
        if source_file:
            raw_metadata["source_file"] = source_file

        return Conversation(
            schema_version="1.0",
            id=conv_id,
            provider=ProviderInfo(
                name="copilot",
                conversation_id=conv_id,
            ),
            title=conv_name if conv_name != "untitled" else None,
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
            raw_metadata=raw_metadata if raw_metadata else {},
        )

    @staticmethod
    def _normalize_role(author: str) -> MessageRole:
        if author.lower().strip() in _USER_AUTHORS:
            return MessageRole.USER
        return MessageRole.ASSISTANT

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse various timestamp formats from Copilot CSVs.

        All returned datetimes are timezone-aware. Naive datetimes (no offset
        in the source string) are assumed UTC.
        """
        if not ts or not ts.strip():
            return datetime.now(tz=UTC)
        ts = ts.strip()
        try:
            dt = datetime.fromisoformat(ts)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            pass
        # M/D/YYYY H:MM:SS with explicit timezone offset
        try:
            return datetime.strptime(ts, "%m/%d/%Y %H:%M:%S %z")
        except ValueError:
            pass
        # Naive formats — assume UTC
        for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
            try:
                return datetime.strptime(ts, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        return datetime.now(tz=UTC)
