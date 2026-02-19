"""xAI (Grok) export converter.

Handles prod-grok-backend.json from Grok data export.

See importer-mappings.md §5 for field-by-field mapping details.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from portable_ai_memory.converters.base import BaseConverter, parse_iso_or_now, register_converter
from portable_ai_memory.core.integrity import compute_content_hash
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
)
from portable_ai_memory.models.enums import (
    AttachmentType,
    ContentType,
    ExtractionMethod,
    MemoryType,
    MessageRole,
)
from portable_ai_memory.models.memory_store import (
    MemoryObject,
    ProvenanceBlock,
    TemporalBlock,
)

# Response keys already mapped to PAM fields — not duplicated in raw_metadata
_MAPPED_RESP_KEYS = {
    "_id",
    "sender",
    "message",
    "create_time",
    "parent_response_id",
    "model",
    "conversation_id",
    "cited_web_search_results",
    "generated_image_urls",
    "file_attachments",
}


@register_converter
class XAIConverter(BaseConverter):
    """Convert xAI/Grok exports to PAM format."""

    @property
    def provider_name(self) -> str:
        return "grok"

    def detect(self, data: dict[str, Any] | list[Any], path: Path | None = None) -> bool:
        if isinstance(data, dict) and "conversations" in data:
            convs = data["conversations"]
            if isinstance(convs, list) and len(convs) > 0:
                first = convs[0]
                if isinstance(first, dict) and "conversation" in first and "responses" in first:
                    return True
        return False

    @staticmethod
    def find_backend_file(source: Path) -> Path | None:
        """Find prod-grok-backend.json in Grok export directory structure."""
        direct = source / "prod-grok-backend.json"
        if direct.exists():
            return direct

        for pattern in (
            "*/export_data/*/prod-grok-backend.json",
            "*/*/export_data/*/prod-grok-backend.json",
        ):
            matches = list(source.glob(pattern))
            if matches:
                return matches[0]

        return None

    def extract_owner_info(
        self,
        source_dir: Path,
        owner_id: str = "unknown",
    ) -> tuple[str, list[MemoryObject]]:
        """Read auth file and backend data for owner info and memories."""
        import json

        memories: list[MemoryObject] = []

        # prod-mc-auth-mgmt-api.json → owner_id + user profile memory
        auth_file = source_dir / "prod-mc-auth-mgmt-api.json"
        if auth_file.exists():
            auth_data = json.loads(auth_file.read_bytes())
            user = auth_data.get("user", {})
            if isinstance(user, dict):
                uid = user.get("userId")
                if uid and owner_id == "unknown":
                    owner_id = uid

                profile_parts: list[str] = []
                given = user.get("givenName")
                family = user.get("familyName")
                if given or family:
                    name = f"{given or ''} {family or ''}".strip()
                    profile_parts.append(f"Name: {name}")
                email = user.get("email")
                if email:
                    profile_parts.append(f"Email: {email}")

                if profile_parts:
                    content = "Owner profile from Grok export.\n" + "\n".join(profile_parts)
                    memories.append(
                        MemoryObject.create(
                            id="grok-user-profile",
                            type=MemoryType.CONTEXT,
                            content=content,
                            platform="grok",
                        )
                    )

        # Projects and media_posts from the backend data
        backend_file = self.find_backend_file(source_dir)
        if backend_file:
            raw = json.loads(backend_file.read_bytes())
            if isinstance(raw, dict):
                proj_mems = self.convert_projects(raw, owner_id=owner_id)
                memories.extend(proj_mems)

                media_mems = self.convert_media_posts(raw, owner_id=owner_id)
                memories.extend(media_mems)

        return owner_id, memories

    def convert_conversations(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
        import_metadata: ImportMetadata | None = None,
    ) -> list[Conversation]:
        if not isinstance(data, dict):
            return []

        raw_convs = data.get("conversations", [])
        conversations: list[Conversation] = []

        for wrapper in raw_convs:
            if not isinstance(wrapper, dict):
                continue
            conv = self._convert_single(wrapper, owner_id=owner_id, import_metadata=import_metadata)
            if conv is not None:
                conversations.append(conv)

        return conversations

    def convert_projects(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
    ) -> list[MemoryObject]:
        """Convert projects[] from prod-grok-backend.json to MemoryObject list."""
        if not isinstance(data, dict):
            return []

        projects = data.get("projects", [])
        memories: list[MemoryObject] = []

        for proj in projects:
            if not isinstance(proj, dict):
                continue
            name = proj.get("name", "")
            personality = proj.get("custom_personality", "")
            if not personality and not name:
                continue

            parts: list[str] = []
            if name:
                parts.append(f"Project: {name}")
            if personality:
                parts.append(f"Custom personality: {personality}")
            starters = proj.get("conversation_starters", [])
            if starters:
                parts.append(f"Conversation starters: {', '.join(str(s) for s in starters)}")
            model = proj.get("preferred_model")
            if model:
                parts.append(f"Preferred model: {model}")

            wid = proj.get("workspace_id", uuid4().hex[:8])
            memories.append(
                MemoryObject.create(
                    id=f"grok-proj-{wid[:8]}",
                    type=MemoryType.PROJECT,
                    content="\n".join(parts),
                    summary=name or None,
                    platform="grok",
                )
            )

        return memories

    def convert_media_posts(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
    ) -> list[MemoryObject]:
        """Convert media_posts[] from prod-grok-backend.json to MemoryObject list."""
        if not isinstance(data, dict):
            return []

        posts = data.get("media_posts", [])
        memories: list[MemoryObject] = []
        now = datetime.now(tz=UTC)

        for post in posts:
            if not isinstance(post, dict):
                continue
            post_id = post.get("id", uuid4().hex[:8])
            prompt = post.get("original_prompt", "")
            media_type = post.get("media_type", "")
            link = post.get("link", "")
            create_time = post.get("create_time")

            parts: list[str] = []
            if prompt:
                parts.append(f"Prompt: {prompt}")
            if media_type:
                parts.append(f"Type: {media_type}")
            if link:
                parts.append(f"Link: {link}")

            content = "\n".join(parts) if parts else f"Media post {post_id} ({media_type})"
            created = datetime.fromisoformat(create_time) if create_time else now
            memories.append(
                MemoryObject(
                    id=f"grok-media-{post_id[:8]}",
                    type=MemoryType.CONTEXT,
                    content=content,
                    content_hash=compute_content_hash(content),
                    summary=f"Grok {media_type} generation" if media_type else None,
                    temporal=TemporalBlock(created_at=created),
                    provenance=ProvenanceBlock(
                        platform="grok",
                        extraction_method=ExtractionMethod.API_EXPORT,
                        extracted_at=now,
                    ),
                )
            )

        return memories

    def _convert_single(
        self,
        wrapper: dict[str, Any],
        *,
        owner_id: str,
        import_metadata: ImportMetadata | None = None,
    ) -> Conversation | None:
        conv_data = wrapper.get("conversation", {})
        responses = wrapper.get("responses", [])

        conv_id = conv_data.get("id") or str(uuid4())
        title = conv_data.get("title")

        created_at = parse_iso_or_now(conv_data.get("create_time"))
        updated_at_str = conv_data.get("modify_time")
        updated_at = datetime.fromisoformat(updated_at_str) if updated_at_str else None

        messages = self._convert_responses(responses)

        if not messages:
            return None

        # Preserve ALL non-null conversation-level fields
        raw_metadata: dict[str, Any] = {}
        for key in (
            "starred",
            "system_prompt_name",
            "summary",
            "temporary",
            "media_types",
            "system_prompt_id",
            "leaf_response_id",
            "root_asset_id",
            "controller",
            "shared_with_team",
            "shared_with_user_ids",
            "asset_ids",
            "task_result_id",
            "team_id",
            "x_user_id",
            "anon_user_id",
        ):
            val = conv_data.get(key)
            if val is not None:
                raw_metadata[key] = val

        return Conversation(
            schema_version="1.0",
            id=conv_id,
            provider=ProviderInfo(
                name="grok",
                conversation_id=conv_id,
                account_id=conv_data.get("user_id"),
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

    def _convert_responses(self, responses: list[Any]) -> list[Message]:
        messages: list[Message] = []

        # First pass: assign stable IDs and record parent references
        stable_ids: list[str] = []
        original_ids: list[str | None] = []
        parent_refs: list[str | None] = []
        for resp_wrapper in responses:
            if not isinstance(resp_wrapper, dict):
                stable_ids.append("")
                original_ids.append(None)
                parent_refs.append(None)
                continue
            resp = resp_wrapper.get("response", {})
            if not isinstance(resp, dict):
                stable_ids.append("")
                original_ids.append(None)
                parent_refs.append(None)
                continue
            original_id = resp.get("_id")
            msg_id = original_id or str(uuid4())
            stable_ids.append(msg_id)
            original_ids.append(original_id)
            parent_refs.append(resp.get("parent_response_id"))

        # Build mapping from original provider IDs to stable IDs so that
        # parent_response_id references (which use original IDs) can be
        # resolved to our stable IDs even when _id was missing.
        # NOTE: Only responses WITH an _id can be referenced as parents
        # (parent_response_id contains the original _id value), so we only
        # need to map those. Responses without _id get a uuid4() but can
        # never be a parent_response_id target.
        original_to_stable: dict[str, str] = {}
        for idx, orig in enumerate(original_ids):
            if orig:
                original_to_stable[orig] = stable_ids[idx]

        # Build children map using stable IDs for both parent and child
        children_map: dict[str, list[str]] = {}
        for idx, parent_ref in enumerate(parent_refs):
            if parent_ref:
                stable_parent = original_to_stable.get(parent_ref, parent_ref)
                children_map.setdefault(stable_parent, []).append(stable_ids[idx])

        # Second pass: build messages using stable IDs
        for idx, resp_wrapper in enumerate(responses):
            if not isinstance(resp_wrapper, dict):
                continue
            resp = resp_wrapper.get("response", {})
            if not isinstance(resp, dict):
                continue

            msg_id = stable_ids[idx]
            sender = resp.get("sender", "human")
            role = self._normalize_role(sender)
            text = resp.get("message", "")

            created_at = self._parse_bson_timestamp(resp.get("create_time"))

            # Citations
            citations: list[Citation] = []
            cited = resp.get("cited_web_search_results", [])
            if isinstance(cited, list):
                for c in cited:
                    if isinstance(c, dict):
                        citations.append(
                            Citation(
                                title=c.get("title"),
                                url=c.get("url"),
                                snippet=c.get("preview"),
                            )
                        )

            # Attachments from generated images
            attachments: list[Attachment] = []
            image_urls = resp.get("generated_image_urls", [])
            if isinstance(image_urls, list):
                for url in image_urls:
                    if url:
                        attachments.append(Attachment(type=AttachmentType.IMAGE, ref=str(url)))

            # Attachments from file_attachments (asset UUIDs)
            file_atts = resp.get("file_attachments", [])
            if isinstance(file_atts, list):
                for asset_id in file_atts:
                    if asset_id:
                        attachments.append(
                            Attachment(
                                type=AttachmentType.FILE,
                                provider_id=str(asset_id),
                            )
                        )

            # Resolve parent_response_id to stable ID
            raw_parent = parent_refs[idx]
            parent_id = original_to_stable.get(raw_parent, raw_parent) if raw_parent else None
            children_ids = children_map.get(msg_id, [])

            # Preserve ALL non-null response fields not already mapped
            raw_metadata: dict[str, Any] = {}
            for key, val in resp.items():
                if key in _MAPPED_RESP_KEYS:
                    continue
                if val is None:
                    continue
                # Convert BSON timestamps for thinking times
                if key in ("thinking_start_time", "thinking_end_time") and isinstance(val, dict):
                    raw_metadata[key] = self._parse_bson_timestamp(val).isoformat()
                elif key == "metadata":
                    # Rename to avoid ambiguity with PAM metadata (per importer-mappings spec)
                    raw_metadata["grok_metadata"] = val
                else:
                    raw_metadata[key] = val

            # Preserve share_link data from wrapper
            share_link = resp_wrapper.get("share_link")
            if share_link is not None:
                raw_metadata["share_link"] = share_link

            # NOTE: Messages are appended even when text is empty. Grok exports
            # include error responses (with "error" key in raw_metadata) and
            # thinking traces with no visible text. Filtering would lose data.
            msg = Message(
                id=msg_id,
                provider_message_id=msg_id,
                role=role,
                content=MessageContent(type=ContentType.TEXT, text=text or None),
                created_at=created_at,
                parent_id=parent_id,
                children_ids=children_ids,
                model=resp.get("model"),
                citations=citations,
                attachments=attachments,
                raw_metadata=raw_metadata,
            )
            messages.append(msg)

        return messages

    @staticmethod
    def _normalize_role(sender: str) -> MessageRole:
        """Normalize role: case-insensitive, any non-'human' is assistant."""
        if sender.lower() == "human":
            return MessageRole.USER
        return MessageRole.ASSISTANT

    @staticmethod
    def _parse_bson_timestamp(ts: object) -> datetime:
        """Parse BSON timestamp {"$date": {"$numberLong": "<ms>"}} or ISO string."""
        if isinstance(ts, dict):
            date_val = ts.get("$date", {})
            if isinstance(date_val, dict):
                ms = int(date_val.get("$numberLong", "0"))
                return datetime.fromtimestamp(ms / 1000, tz=UTC)
            if isinstance(date_val, str):
                return datetime.fromisoformat(date_val)
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        return datetime.now(tz=UTC)
