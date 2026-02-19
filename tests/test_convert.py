"""Tests for pam export command and OpenAI attachment handling."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from portable_ai_memory.converters.openai import OpenAIConverter
from portable_ai_memory.converters.openai_attachments import copy_attachments, resolve_attachment
from portable_ai_memory.models.conversation import Conversation, ImportMetadata
from portable_ai_memory.models.enums import AttachmentType

from .conftest import real_data_path, skip_no_real_data

# ── Fixtures ──────────────────────────────────────────────


def _make_raw_conversation(
    *,
    parts: list | None = None,  # type: ignore[type-arg]
    metadata_attachments: list | None = None,  # type: ignore[type-arg]
) -> list:  # type: ignore[type-arg]
    """Build a minimal OpenAI conversations.json list with one conversation."""
    msg: dict = {  # type: ignore[type-arg]
        "id": "msg-1",
        "author": {"role": "user"},
        "create_time": 1700000000.0,
        "content": {"content_type": "text", "parts": parts or ["hello"]},
        "metadata": {},
    }
    if metadata_attachments:
        msg["metadata"]["attachments"] = metadata_attachments

    return [
        {
            "id": "conv-1",
            "title": "Test",
            "create_time": 1700000000.0,
            "update_time": 1700001000.0,
            "mapping": {
                "node-1": {
                    "message": msg,
                    "parent": None,
                    "children": [],
                },
            },
        }
    ]


# ── Attachment extraction ─────────────────────────────────


class TestAttachmentExtraction:
    def test_image_asset_pointer(self) -> None:
        raw = _make_raw_conversation(
            parts=[
                "check this image",
                {
                    "content_type": "image_asset_pointer",
                    "asset_pointer": "file-service://file-ABC123",
                    "size_bytes": 12345,
                },
            ]
        )
        converter = OpenAIConverter()
        convs = converter.convert_conversations(raw, owner_id="test")
        assert len(convs) == 1
        msg = convs[0].messages[0]
        assert msg.content is not None
        assert msg.content.text == "check this image"
        assert len(msg.attachments) == 1
        att = msg.attachments[0]
        assert att.type == AttachmentType.IMAGE
        assert att.ref == "file-service://file-ABC123"
        assert att.size_bytes == 12345
        assert att.provider_id == "file-service://file-ABC123"

    def test_metadata_attachments(self) -> None:
        raw = _make_raw_conversation(
            metadata_attachments=[
                {
                    "id": "file-xyz",
                    "name": "report.pdf",
                    "mime_type": "application/pdf",
                    "size": 99000,
                }
            ]
        )
        converter = OpenAIConverter()
        convs = converter.convert_conversations(raw, owner_id="test")
        msg = convs[0].messages[0]
        assert len(msg.attachments) == 1
        att = msg.attachments[0]
        assert att.type == AttachmentType.FILE
        assert att.name == "report.pdf"
        assert att.mime_type == "application/pdf"
        assert att.size_bytes == 99000
        assert att.provider_id == "file-xyz"

    def test_import_metadata_populated(self) -> None:
        raw = _make_raw_conversation()
        converter = OpenAIConverter()
        meta = ImportMetadata(
            importer="pam-cli/0.1.0",
            imported_at=datetime(2024, 1, 1, tzinfo=UTC),
            source_file="conversations.json",
            source_checksum="sha256:" + "a" * 64,
        )
        convs = converter.convert_conversations(raw, owner_id="test", import_metadata=meta)
        assert convs[0].import_metadata is not None
        assert convs[0].import_metadata.importer == "pam-cli/0.1.0"
        assert convs[0].import_metadata.source_file == "conversations.json"


# ── Attachment resolver ───────────────────────────────────


class TestResolveAttachment:
    def test_file_service_uri(self, tmp_path: Path) -> None:
        # Create a fake export file
        (tmp_path / "file-ABC123-something.png").touch()
        result = resolve_attachment("file-service://file-ABC123", tmp_path)
        assert result is not None
        assert result.name == "file-ABC123-something.png"

    def test_sediment_uri_root(self, tmp_path: Path) -> None:
        (tmp_path / "file_deadbeef-sanitized.png").touch()
        result = resolve_attachment("sediment://file_deadbeef", tmp_path)
        assert result is not None
        assert result.name == "file_deadbeef-sanitized.png"

    def test_sediment_uri_user_subdir(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user-NReGOPwX"
        user_dir.mkdir()
        (user_dir / "file_cafebabe-uuid123.png").touch()
        result = resolve_attachment("sediment://file_cafebabe", tmp_path)
        assert result is not None
        assert "file_cafebabe" in result.name

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        result = resolve_attachment("file-service://file-NOTEXIST", tmp_path)
        assert result is None

    def test_unknown_scheme_returns_none(self, tmp_path: Path) -> None:
        result = resolve_attachment("https://example.com/img.png", tmp_path)
        assert result is None


# ── Copy attachments ──────────────────────────────────────


class TestCopyAttachments:
    def test_copies_and_updates_refs(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "convert"
        export_dir.mkdir()
        (export_dir / "file-ABC-photo.png").write_bytes(b"fake png")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        raw = _make_raw_conversation(
            parts=[
                {
                    "content_type": "image_asset_pointer",
                    "asset_pointer": "file-service://file-ABC",
                    "size_bytes": 8,
                },
            ]
        )
        converter = OpenAIConverter()
        convs = converter.convert_conversations(raw, owner_id="test")

        count = copy_attachments(convs, export_dir, output_dir)
        assert count == 1
        assert (output_dir / "attachments" / "file-ABC-photo.png").exists()
        # Ref updated to relative path
        att = convs[0].messages[0].attachments[0]
        assert att.ref == "attachments/file-ABC-photo.png"


# ── Full export integration ───────────────────────────────


class TestExportCLI:
    def test_conversations_index_built(self, tmp_path: Path) -> None:
        """Test the full export pipeline produces valid output."""
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        # Create fake export
        export_dir = tmp_path / "chatgpt"
        export_dir.mkdir()
        raw = _make_raw_conversation()
        (export_dir / "conversations.json").write_text(json.dumps(raw))

        output_dir = tmp_path / "pam-out"
        runner = CliRunner()
        result = runner.invoke(
            app, ["convert", str(export_dir), "-o", str(output_dir), "--owner-id", "test-user"]
        )
        assert result.exit_code == 0, result.output

        # Check memory-store.json
        store_path = output_dir / "memory-store.json"
        assert store_path.exists()
        store = json.loads(store_path.read_text())
        assert store["schema"] == "portable-ai-memory"
        assert len(store["conversations_index"]) == 1
        entry = store["conversations_index"][0]
        assert entry["id"] == "conv-1"
        assert entry["platform"] == "chatgpt"
        assert entry["storage"]["type"] == "file"
        assert entry["storage"]["ref"] == "conversations/conv-1.json"

    def test_integrity_block_present(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        export_dir = tmp_path / "chatgpt"
        export_dir.mkdir()
        raw = _make_raw_conversation()
        (export_dir / "conversations.json").write_text(json.dumps(raw))

        output_dir = tmp_path / "pam-out"
        runner = CliRunner()
        result = runner.invoke(app, ["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert store["integrity"] is not None
        assert store["integrity"]["checksum"].startswith("sha256:")
        assert store["integrity"]["total_memories"] == 0


# ── Convert command edge cases ────────────────────────────


class TestConvertCommand:
    def _setup_export(self, tmp_path: Path, **extra_files: str) -> tuple[Path, Path]:
        """Create a fake ChatGPT export dir and return (export_dir, output_dir)."""
        export_dir = tmp_path / "chatgpt"
        export_dir.mkdir()
        raw = _make_raw_conversation()
        (export_dir / "conversations.json").write_text(json.dumps(raw))
        for name, content in extra_files.items():
            (export_dir / name).write_text(content)
        return export_dir, tmp_path / "pam-out"

    def _run(self, args: list[str]) -> object:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        return CliRunner().invoke(app, args)

    def test_convert_with_file_path(self, tmp_path: Path) -> None:
        """Pass conversations.json file directly (not a dir) as source."""
        export_dir, output_dir = self._setup_export(tmp_path)
        conv_file = export_dir / "conversations.json"
        result = self._run(["convert", str(conv_file), "-o", str(output_dir), "--owner-id", "u1"])
        assert result.exit_code == 0, result.output
        assert (output_dir / "memory-store.json").exists()

    def test_convert_missing_conversations_json(self, tmp_path: Path) -> None:
        """Pass a dir without conversations.json → exit 1, 'Not found'."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "out"
        result = self._run(["convert", str(empty_dir), "-o", str(output_dir)])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_convert_unrecognized_provider(self, tmp_path: Path) -> None:
        """Pass a JSON file that doesn't match any provider → exit 1, 'Could not detect'."""
        bad_file = tmp_path / "weird.json"
        bad_file.write_text(json.dumps({"not": "a provider format"}))
        output_dir = tmp_path / "out"
        result = self._run(["convert", str(bad_file), "-o", str(output_dir)])
        assert result.exit_code == 1
        assert "No converter matched" in result.output

    def test_convert_with_user_json(self, tmp_path: Path) -> None:
        """user.json provides owner_id when --owner-id not given."""
        export_dir, output_dir = self._setup_export(
            tmp_path, **{"user.json": json.dumps({"id": "user-abc"})}
        )
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output
        store = json.loads((output_dir / "memory-store.json").read_text())
        assert store["owner"]["id"] == "user-abc"

    def test_convert_with_sora_json(self, tmp_path: Path) -> None:
        """sora.json user_display_name is shown in console output."""
        export_dir, output_dir = self._setup_export(
            tmp_path, **{"sora.json": json.dumps({"user_display_name": "Daniel"})}
        )
        result = self._run(["convert", str(export_dir), "-o", str(output_dir), "--owner-id", "x"])
        assert result.exit_code == 0, result.output

    def test_convert_with_message_feedback(self, tmp_path: Path) -> None:
        """message_feedback.json injects feedback into message raw_metadata."""
        feedback = [
            {
                "message_id": "msg-1",
                "conversation_id": "conv-1",
                "rating": "thumbsUp",
                "tags": ["helpful"],
                "create_time": 1700000000.0,
            }
        ]
        export_dir, output_dir = self._setup_export(
            tmp_path, **{"message_feedback.json": json.dumps(feedback)}
        )
        result = self._run(["convert", str(export_dir), "-o", str(output_dir), "--owner-id", "u"])
        assert result.exit_code == 0, result.output

        conv_file = list((output_dir / "conversations").glob("*.json"))[0]
        conv = json.loads(conv_file.read_text())
        msg = conv["messages"][0]
        assert "feedback" in msg["raw_metadata"]
        assert msg["raw_metadata"]["feedback"]["rating"] == "thumbsUp"
        assert msg["raw_metadata"]["feedback"]["tags"] == ["helpful"]

    def test_convert_owner_id_overrides_user_json(self, tmp_path: Path) -> None:
        """Explicit --owner-id wins over user.json."""
        export_dir, output_dir = self._setup_export(
            tmp_path, **{"user.json": json.dumps({"id": "user-abc"})}
        )
        result = self._run(
            ["convert", str(export_dir), "-o", str(output_dir), "--owner-id", "explicit-owner"]
        )
        assert result.exit_code == 0, result.output
        store = json.loads((output_dir / "memory-store.json").read_text())
        assert store["owner"]["id"] == "explicit-owner"


# ── Real data integration (skipped if not available) ──────


@skip_no_real_data
class TestRealExport:
    def test_full_export_chatgpt(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        output_dir = tmp_path / "pam-chatgpt"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "convert",
                str(real_data_path("chatgpt")),
                "-o",
                str(output_dir),
                "--owner-id",
                "daniel",
            ],
        )
        assert result.exit_code == 0, result.output

        # Verify output structure
        assert (output_dir / "memory-store.json").exists()
        assert (output_dir / "conversations").is_dir()

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert len(store["conversations_index"]) > 0
        assert store["integrity"] is not None

        # Check at least one conversation file is valid
        conv_files = list((output_dir / "conversations").glob("*.json"))
        assert len(conv_files) > 0


# ── Additional copy_attachments edge-case tests ──────────


class TestCopyAttachmentsEdgeCases:
    def _make_conversation_with_attachment(self, ref: str) -> Conversation:
        from portable_ai_memory.models.conversation import (
            Attachment,
            ConversationTemporal,
            Message,
            MessageContent,
            ProviderInfo,
        )
        from portable_ai_memory.models.enums import AttachmentType, ContentType, MessageRole

        return Conversation(
            schema_version="1.0",
            id="conv-1",
            provider=ProviderInfo(name="chatgpt"),
            temporal=ConversationTemporal(created_at="2026-01-01T00:00:00Z"),
            messages=[
                Message(
                    id="msg-1",
                    role=MessageRole.USER,
                    created_at="2026-01-01T00:00:00Z",
                    content=MessageContent(type=ContentType.TEXT, text="hi"),
                    attachments=[
                        Attachment(type=AttachmentType.IMAGE, ref=ref),
                    ],
                ),
            ],
        )

    def test_copy_sediment_user_subdir(self, tmp_path: Path) -> None:
        """Attachment from sediment:// URI in a user-*/ subdir is copied preserving subdir."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        user_dir = export_dir / "user-NReGOPwX"
        user_dir.mkdir()
        (user_dir / "file_cafebabe-img.png").write_bytes(b"img data")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        conv = self._make_conversation_with_attachment("sediment://file_cafebabe")
        count = copy_attachments([conv], export_dir, output_dir)
        assert count == 1
        assert (output_dir / "attachments" / "user-NReGOPwX" / "file_cafebabe-img.png").exists()
        assert (
            conv.messages[0].attachments[0].ref == "attachments/user-NReGOPwX/file_cafebabe-img.png"
        )

    def test_copy_deduplication(self, tmp_path: Path) -> None:
        """Same file referenced twice is only copied once."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        (export_dir / "file-ABC-photo.png").write_bytes(b"data")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        conv1 = self._make_conversation_with_attachment("file-service://file-ABC")
        conv2 = self._make_conversation_with_attachment("file-service://file-ABC")
        count = copy_attachments([conv1, conv2], export_dir, output_dir)
        assert count == 1
        # Both refs updated
        assert conv1.messages[0].attachments[0].ref == "attachments/file-ABC-photo.png"
        assert conv2.messages[0].attachments[0].ref == "attachments/file-ABC-photo.png"

    def test_copy_unresolvable_skipped(self, tmp_path: Path) -> None:
        """Attachment with URI that doesn't match any file is skipped."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        conv = self._make_conversation_with_attachment("file-service://file-NOTEXIST")
        count = copy_attachments([conv], export_dir, output_dir)
        assert count == 0

    def test_copy_non_uri_ref_skipped(self, tmp_path: Path) -> None:
        """Attachment with a non-URI ref (already resolved) is skipped."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        conv = self._make_conversation_with_attachment("attachments/already-resolved.png")
        count = copy_attachments([conv], export_dir, output_dir)
        assert count == 0


# ── Claude provider CLI integration ──────────────────────


def _make_claude_export(tmp_path: Path) -> Path:
    """Create a fake Claude export directory."""
    export_dir = tmp_path / "claude"
    export_dir.mkdir()

    conversations = [
        {
            "uuid": "conv-001",
            "name": "Test Chat",
            "created_at": "2026-01-01T10:00:00Z",
            "updated_at": "2026-01-01T11:00:00Z",
            "account": {"uuid": "acc-123"},
            "chat_messages": [
                {
                    "uuid": "msg-001",
                    "sender": "human",
                    "text": "Hello",
                    "content": [],
                    "created_at": "2026-01-01T10:00:00Z",
                    "files": [{"file_name": "doc.pdf"}],
                    "attachments": [],
                },
                {
                    "uuid": "msg-002",
                    "sender": "assistant",
                    "text": "Hi there!",
                    "content": [{"type": "text", "text": "Hi there!"}],
                    "created_at": "2026-01-01T10:01:00Z",
                },
            ],
        }
    ]
    (export_dir / "conversations.json").write_text(json.dumps(conversations))

    users = [
        {"uuid": "user-abc-123", "full_name": "Test User", "email_address": "test@example.com"}
    ]
    (export_dir / "users.json").write_text(json.dumps(users))

    memories = [
        {
            "conversations_memory": "User likes Python.",
            "project_memories": {"proj-1": "Project uses FastAPI."},
        }
    ]
    (export_dir / "memories.json").write_text(json.dumps(memories))

    projects = [
        {
            "uuid": "proj-001",
            "name": "TestProj",
            "docs": [
                {"uuid": "doc-001", "filename": "notes.md", "content": "# Notes\nSome content."}
            ],
        }
    ]
    (export_dir / "projects.json").write_text(json.dumps(projects))

    return export_dir


class TestClaudeConvertCLI:
    def _run(self, args: list[str]) -> object:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        return CliRunner().invoke(app, args)

    def test_claude_full_export(self, tmp_path: Path) -> None:
        export_dir = _make_claude_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        # Check memory-store has memories
        store = json.loads((output_dir / "memory-store.json").read_text())
        assert (
            len(store["memories"]) == 4
        )  # user_profile + conv_memory + project_memory + project_doc
        assert len(store["conversations_index"]) == 1
        assert store["conversations_index"][0]["platform"] == "claude"
        assert store["integrity"]["total_memories"] == 4

    def test_claude_owner_from_users_json(self, tmp_path: Path) -> None:
        export_dir = _make_claude_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert store["owner"]["id"] == "user-abc-123"

    def test_claude_owner_id_override(self, tmp_path: Path) -> None:
        export_dir = _make_claude_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(
            ["convert", str(export_dir), "-o", str(output_dir), "--owner-id", "my-id"]
        )
        assert result.exit_code == 0, result.output
        store = json.loads((output_dir / "memory-store.json").read_text())
        assert store["owner"]["id"] == "my-id"

    def test_claude_attachments_in_messages(self, tmp_path: Path) -> None:
        export_dir = _make_claude_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        conv_file = list((output_dir / "conversations").glob("*.json"))[0]
        conv = json.loads(conv_file.read_text())
        # First message has a file attachment
        assert len(conv["messages"][0]["attachments"]) == 1
        assert conv["messages"][0]["attachments"][0]["type"] == "file"


# ── Real Claude data integration (skipped if not available) ──


@skip_no_real_data
class TestRealClaudeExport:
    def test_full_export_claude(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        output_dir = tmp_path / "pam-claude"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["convert", str(real_data_path("claude")), "-o", str(output_dir)],
        )
        assert result.exit_code == 0, result.output

        assert (output_dir / "memory-store.json").exists()
        assert (output_dir / "conversations").is_dir()

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert len(store["conversations_index"]) > 0
        assert len(store["memories"]) > 0
        assert store["integrity"] is not None


# ── Grok / xAI CLI integration tests ──


def _make_grok_export(tmp_path: Path) -> Path:
    """Create a fake Grok export directory."""
    export_dir = tmp_path / "grok"
    export_dir.mkdir()

    backend = {
        "conversations": [
            {
                "conversation": {
                    "id": "conv-001",
                    "title": "Test Grok Chat",
                    "create_time": "2026-01-01T10:00:00Z",
                    "modify_time": "2026-01-01T11:00:00Z",
                    "user_id": "user-grok-1",
                },
                "responses": [
                    {
                        "response": {
                            "_id": "msg-001",
                            "sender": "human",
                            "message": "Hello Grok",
                            "create_time": {"$date": {"$numberLong": "1735689600000"}},
                        },
                        "share_link": None,
                    },
                    {
                        "response": {
                            "_id": "msg-002",
                            "sender": "ASSISTANT",
                            "message": "Hi!",
                            "model": "grok-3",
                            "create_time": {"$date": {"$numberLong": "1735689610000"}},
                            "parent_response_id": "msg-001",
                        },
                        "share_link": None,
                    },
                ],
            }
        ],
        "projects": [
            {
                "workspace_id": "proj-001",
                "name": "My Project",
                "custom_personality": "Be concise",
                "preferred_model": "grok-3",
                "conversation_starters": [],
            }
        ],
        "media_posts": [
            {
                "id": "post-001",
                "original_prompt": "A cat",
                "media_type": "image",
                "create_time": "2026-01-01T10:00:00Z",
                "link": "https://grok.com/imagine/post/post-001",
            }
        ],
        "tasks": [],
    }
    (export_dir / "prod-grok-backend.json").write_text(json.dumps(backend))

    auth = {
        "user": {
            "userId": "grok-user-123",
            "givenName": "Grok",
            "familyName": "User",
            "email": "grok@test.com",
        }
    }
    (export_dir / "prod-mc-auth-mgmt-api.json").write_text(json.dumps(auth))

    return export_dir


class TestGrokConvertCLI:
    def _run(self, args: list[str]) -> object:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        return CliRunner().invoke(app, args)

    def test_grok_full_export(self, tmp_path: Path) -> None:
        export_dir = _make_grok_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        store = json.loads((output_dir / "memory-store.json").read_text())
        # user_profile + project + media_post = 3
        assert len(store["memories"]) == 3
        assert len(store["conversations_index"]) == 1
        assert store["conversations_index"][0]["platform"] == "grok"

    def test_grok_owner_from_auth(self, tmp_path: Path) -> None:
        export_dir = _make_grok_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert store["owner"]["id"] == "grok-user-123"

    def test_grok_conversation_output(self, tmp_path: Path) -> None:
        export_dir = _make_grok_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        conv_file = list((output_dir / "conversations").glob("*.json"))[0]
        conv = json.loads(conv_file.read_text())
        assert conv["title"] == "Test Grok Chat"
        assert len(conv["messages"]) == 2
        assert conv["provider"]["name"] == "grok"


@skip_no_real_data
class TestRealGrokExport:
    def test_full_export_grok(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        output_dir = tmp_path / "pam-grok"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["convert", str(real_data_path("grok")), "-o", str(output_dir)],
        )
        assert result.exit_code == 0, result.output

        assert (output_dir / "memory-store.json").exists()
        assert (output_dir / "conversations").is_dir()

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert len(store["conversations_index"]) > 0
        assert len(store["memories"]) > 0


# ── Copilot / Microsoft CLI integration tests ──


def _make_copilot_export(tmp_path: Path) -> Path:
    """Create a fake Copilot export directory with CSV files."""
    export_dir = tmp_path / "copilot"
    export_dir.mkdir()

    (export_dir / "copilot-activity-history.csv").write_text(
        "Conversation,Time,Author,Message\n"
        "Test Chat,2026-01-01T10:00:00,Human,Hello\n"
        "Test Chat,2026-01-01T10:00:00,AI,Hi there!\n",
        encoding="utf-8",
    )

    (export_dir / "copilot-chat-activity.csv").write_text(
        "CreatedAt,MessageContent,Author,ChatName\n"
        "1/1/2026 10:00:00 +00:00,Query,user,Research\n"
        "1/1/2026 10:00:01 +00:00,Answer,bot,Research\n",
        encoding="utf-8",
    )

    (export_dir / "copilot-in-Microsoft-365-apps-activity.csv").write_text(
        "CreatedAt,MessageContent,Author,ChatName\n",
        encoding="utf-8",
    )

    return export_dir


class TestCopilotConvertCLI:
    def _run(self, args: list[str]) -> object:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        return CliRunner().invoke(app, args)

    def test_copilot_full_export(self, tmp_path: Path) -> None:
        export_dir = _make_copilot_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert len(store["conversations_index"]) == 2
        assert store["conversations_index"][0]["platform"] == "copilot"

    def test_copilot_message_order(self, tmp_path: Path) -> None:
        export_dir = _make_copilot_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        for f in (output_dir / "conversations").glob("*.json"):
            conv = json.loads(f.read_text())
            if len(conv["messages"]) == 2:
                assert conv["messages"][0]["role"] == "user"
                assert conv["messages"][1]["role"] == "assistant"

    def test_copilot_skips_empty_csv(self, tmp_path: Path) -> None:
        export_dir = _make_copilot_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output
        assert "Skipping" in result.output

    def test_copilot_import_metadata(self, tmp_path: Path) -> None:
        export_dir = _make_copilot_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        conv_file = list((output_dir / "conversations").glob("*.json"))[0]
        conv = json.loads(conv_file.read_text())
        assert conv["import_metadata"]["importer"].startswith("pam-cli/")
        assert conv["import_metadata"]["source_file"].endswith(".csv")


@skip_no_real_data
class TestRealCopilotExport:
    def test_full_export_copilot(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        output_dir = tmp_path / "pam-copilot"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["convert", str(real_data_path("copilot")), "-o", str(output_dir)],
        )
        assert result.exit_code == 0, result.output

        assert (output_dir / "memory-store.json").exists()
        assert (output_dir / "conversations").is_dir()

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert len(store["conversations_index"]) > 0


# ── Gemini / Google CLI integration tests ──


def _make_gemini_export(tmp_path: Path) -> Path:
    """Create a fake Gemini Takeout directory."""
    export_dir = tmp_path / "gemini"
    export_dir.mkdir()

    activity = [
        {
            "header": "Gemini",
            "title": "Used Gemini Apps",
            "titleUrl": "https://gemini.google.com/app/c/conv-001",
            "time": "2026-01-01T10:00:00Z",
            "products": ["Gemini Apps"],
            "details": [
                {"name": "Request", "value": "Hello Gemini"},
                {"name": "Response", "value": "Hi!"},
            ],
        },
        {
            "header": "Gemini",
            "title": "Used Gemini Apps",
            "titleUrl": "https://gemini.google.com/app/c/conv-002",
            "time": "2026-01-01T11:00:00Z",
            "products": ["Gemini Apps"],
            "userInteractions": [
                {
                    "userInteraction": {
                        "request": "What is AI?",
                        "response": "AI is...",
                        "endpoint": 2,
                        "latencySeconds": 0.5,
                    }
                }
            ],
        },
    ]
    (export_dir / "MyActivity.json").write_text(json.dumps(activity))

    return export_dir


class TestGeminiConvertCLI:
    def _run(self, args: list[str]) -> object:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        return CliRunner().invoke(app, args)

    def test_gemini_full_export(self, tmp_path: Path) -> None:
        export_dir = _make_gemini_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert len(store["conversations_index"]) == 2
        assert store["conversations_index"][0]["platform"] == "gemini"

    def test_gemini_conversation_content(self, tmp_path: Path) -> None:
        export_dir = _make_gemini_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        convs = []
        for f in (output_dir / "conversations").glob("*.json"):
            convs.append(json.loads(f.read_text()))
        assert len(convs) == 2

        for conv in convs:
            assert len(conv["messages"]) == 2
            assert conv["messages"][0]["role"] == "user"
            assert conv["messages"][1]["role"] == "assistant"

    def test_gemini_import_metadata(self, tmp_path: Path) -> None:
        export_dir = _make_gemini_export(tmp_path)
        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        conv_file = list((output_dir / "conversations").glob("*.json"))[0]
        conv = json.loads(conv_file.read_text())
        assert conv["import_metadata"]["source_file"] == "MyActivity.json"

    def test_gemini_takeout_dir_structure(self, tmp_path: Path) -> None:
        """Should find MyActivity.json in Takeout directory structure."""
        takeout_dir = tmp_path / "takeout"
        nested = takeout_dir / "My Activity" / "Gemini Apps"
        nested.mkdir(parents=True)

        activity = [
            {
                "header": "Gemini",
                "titleUrl": "https://gemini.google.com/app/c/c1",
                "time": "2026-01-01T10:00:00Z",
                "details": [{"name": "Request", "value": "Hi"}],
            }
        ]
        (nested / "MyActivity.json").write_text(json.dumps(activity))

        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(takeout_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output
        assert "gemini" in result.output.lower()


# ── Gemini HTML CLI integration tests ──


_GEMINI_HTML_CONTENT = """<html><body>
<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
  <div class="mdl-grid">
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">
      Prompted What is AI?<br>Feb 17, 2026, 3:08:20 PM EST<br>
      <p>AI is artificial intelligence.</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"></div>
  </div>
</div>
<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
  <div class="mdl-grid">
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">
      Prompted Tell me more<br>Feb 17, 2026, 3:10:00 PM EST<br>
      <p>Here is more info.</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"></div>
  </div>
</div>
</body></html>"""


class TestGeminiHtmlConvertCLI:
    def _run(self, args: list[str]) -> object:
        from typer.testing import CliRunner

        from portable_ai_memory.cli import app

        return CliRunner().invoke(app, args)

    def test_gemini_html_full_export(self, tmp_path: Path) -> None:
        """CLI convert should handle Gemini HTML activity exports."""
        export_dir = tmp_path / "gemini-html"
        export_dir.mkdir()
        (export_dir / "MyActivity.html").write_text(_GEMINI_HTML_CONTENT)

        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert len(store["conversations_index"]) >= 1
        assert store["conversations_index"][0]["platform"] == "gemini"

    def test_gemini_html_conversation_content(self, tmp_path: Path) -> None:
        """Converted HTML should produce user+assistant message pairs."""
        export_dir = tmp_path / "gemini-html"
        export_dir.mkdir()
        (export_dir / "MyActivity.html").write_text(_GEMINI_HTML_CONTENT)

        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        conv_files = list((output_dir / "conversations").glob("*.json"))
        assert len(conv_files) >= 1

        conv = json.loads(conv_files[0].read_text())
        roles = [m["role"] for m in conv["messages"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_gemini_html_import_metadata(self, tmp_path: Path) -> None:
        """HTML export should have proper import_metadata with source_file."""
        export_dir = tmp_path / "gemini-html"
        export_dir.mkdir()
        (export_dir / "MyActivity.html").write_text(_GEMINI_HTML_CONTENT)

        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(export_dir), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        conv_file = list((output_dir / "conversations").glob("*.json"))[0]
        conv = json.loads(conv_file.read_text())
        assert conv["import_metadata"]["source_file"] == "MyActivity.html"
        assert "sha256:" in conv["import_metadata"]["source_checksum"]

    def test_gemini_html_direct_file(self, tmp_path: Path) -> None:
        """Passing an HTML file directly (not a dir) should also work."""
        html_file = tmp_path / "Minhaatividade.html"
        html_file.write_text(_GEMINI_HTML_CONTENT)

        output_dir = tmp_path / "pam-out"
        result = self._run(["convert", str(html_file), "-o", str(output_dir)])
        assert result.exit_code == 0, result.output

        store = json.loads((output_dir / "memory-store.json").read_text())
        assert len(store["conversations_index"]) >= 1
