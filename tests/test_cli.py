"""Tests for PAM CLI validate and inspect commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from portable_ai_memory.cli import app
from portable_ai_memory.core.integrity import compute_content_hash, compute_integrity_checksum
from portable_ai_memory.core.io import save, to_dict
from portable_ai_memory.models.conversation import (
    Conversation,
    ConversationTemporal,
    Message,
    MessageContent,
    Participant,
    ProviderInfo,
)
from portable_ai_memory.models.enums import ContentType, MemoryStatus, MemoryType, MessageRole
from portable_ai_memory.models.memory_store import (
    IntegrityBlock,
    MemoryObject,
    MemoryStore,
    Owner,
    ProvenanceBlock,
    TemporalBlock,
)

runner = CliRunner()


def _make_store(memories: list[MemoryObject] | None = None) -> MemoryStore:
    return MemoryStore(schema_version="1.0", owner=Owner(id="test"), memories=memories or [])


def _make_memory(id: str = "m1", content: str = "hello world") -> MemoryObject:
    return MemoryObject(
        id=id,
        type=MemoryType.PREFERENCE,
        content=content,
        content_hash=compute_content_hash(content),
        temporal=TemporalBlock(created_at=datetime(2024, 1, 1, tzinfo=UTC)),
        provenance=ProvenanceBlock(platform="chatgpt"),
    )


def _make_conversation() -> Conversation:
    return Conversation(
        schema_version="1.0",
        id="conv-1",
        provider=ProviderInfo(name="chatgpt"),
        title="Test Conv",
        temporal=ConversationTemporal(created_at=datetime(2024, 1, 1, tzinfo=UTC)),
        messages=[
            Message(
                id="m1",
                role=MessageRole.USER,
                content=MessageContent(type=ContentType.TEXT, text="hi"),
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            )
        ],
        model="gpt-4",
        participants=[Participant(role=MessageRole.USER, name="Alice")],
    )


# ── validate command ──────────────────────────────────────


class TestValidate:
    def test_valid_memory_store(self, tmp_path: Path) -> None:
        store = _make_store()
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 0
        assert "Schema validation passed" in result.output
        assert "Valid" in result.output

    def test_valid_conversation(self, tmp_path: Path) -> None:
        conv = _make_conversation()
        f = tmp_path / "conv.json"
        save(conv, f)

        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 0
        assert "Schema validation passed" in result.output

    def test_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("{not json!!", encoding="utf-8")

        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output

    def test_invalid_schema(self, tmp_path: Path) -> None:
        f = tmp_path / "wrong.json"
        f.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 1
        assert "Schema validation failed" in result.output

    def test_deep_validation_content_hash_error(self, tmp_path: Path) -> None:
        mem = _make_memory()
        mem.content_hash = "sha256:" + "a" * 64  # wrong hash
        store = _make_store(memories=[mem])
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 1
        assert "error" in result.output.lower()
        assert "Memory 'm1'" in result.output

    def test_deep_validation_warnings(self, tmp_path: Path) -> None:
        mem = _make_memory()
        mem.status = MemoryStatus.SUPERSEDED
        # superseded but no superseded_by → warning
        store = _make_store(memories=[mem])
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["validate", str(f)])
        # Should still be valid (warnings only)
        assert result.exit_code == 0
        assert "warning" in result.output.lower() or "⚠" in result.output

    def test_no_deep_flag(self, tmp_path: Path) -> None:
        mem = _make_memory()
        mem.content_hash = "sha256:" + "a" * 64  # wrong hash
        store = _make_store(memories=[mem])
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["validate", str(f), "--no-deep"])
        # Should pass since deep validation is skipped
        assert result.exit_code == 0
        assert "Schema validation passed" in result.output


# ── inspect command ───────────────────────────────────────


class TestInspect:
    def test_inspect_memory_store(self, tmp_path: Path) -> None:
        store = _make_store()
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["inspect", str(f)])
        assert result.exit_code == 0
        assert "test" in result.output  # owner id
        assert "Memories" in result.output
        assert "Relations" in result.output
        assert "Conversations" in result.output

    def test_inspect_memory_store_with_memories(self, tmp_path: Path) -> None:
        store = _make_store(memories=[_make_memory("m1"), _make_memory("m2", "other content")])
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["inspect", str(f)])
        assert result.exit_code == 0
        assert "preference" in result.output.lower()

    def test_inspect_memory_store_with_integrity(self, tmp_path: Path) -> None:
        mems = [_make_memory()]
        mem_dicts = [to_dict(m) for m in mems]
        checksum = compute_integrity_checksum(mem_dicts)
        store = _make_store(memories=mems)
        store.integrity = IntegrityBlock(checksum=checksum, total_memories=1)
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["inspect", str(f)])
        assert result.exit_code == 0
        assert "Integrity" in result.output
        assert checksum[:30] in result.output

    def test_inspect_conversation(self, tmp_path: Path) -> None:
        conv = _make_conversation()
        f = tmp_path / "conv.json"
        save(conv, f)

        result = runner.invoke(app, ["inspect", str(f)])
        assert result.exit_code == 0
        assert "conv-1" in result.output
        assert "chatgpt" in result.output
        assert "Test Conv" in result.output
        assert "Messages" in result.output
        assert "gpt-4" in result.output
        assert "Alice" in result.output


class TestValidateDisplayBranches:
    def test_deep_warnings_only_exit_zero(self, tmp_path: Path) -> None:
        """Deep validation with only warnings should exit 0 and show warning symbols."""
        mem = _make_memory()
        mem.status = MemoryStatus.SUPERSEDED
        # superseded without superseded_by triggers a warning, not an error
        store = _make_store(memories=[mem])
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 0
        assert "⚠" in result.output
        assert "Valid" in result.output

    def test_deep_errors_and_warnings_together(self, tmp_path: Path) -> None:
        """Deep validation with both errors and warnings shows both."""
        mem = _make_memory()
        mem.content_hash = "sha256:" + "a" * 64  # wrong hash -> error
        mem.status = MemoryStatus.SUPERSEDED  # no superseded_by -> warning
        store = _make_store(memories=[mem])
        f = tmp_path / "store.json"
        save(store, f)

        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 1
        assert "error" in result.output.lower()
        # The warning symbol or text should also appear
        assert "⚠" in result.output or "warning" in result.output.lower()

    def test_validate_bundle_mix_valid_invalid(self, tmp_path: Path) -> None:
        """Validate a directory bundle with mix of valid and invalid files."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        # Valid memory store
        store = _make_store()
        save(store, bundle / "memory-store.json")

        # Conversations dir with one valid, one invalid
        conv_dir = bundle / "conversations"
        conv_dir.mkdir()
        conv = _make_conversation()
        save(conv, conv_dir / "conv-good.json")
        (conv_dir / "conv-bad.json").write_text('{"foo": "bar"}', encoding="utf-8")

        result = runner.invoke(app, ["validate", str(bundle)])
        assert result.exit_code == 1
        # Should report totals
        assert "valid" in result.output.lower()
        assert "invalid" in result.output.lower()


class TestConvertProviderFlag:
    def test_unknown_provider_error(self, tmp_path: Path) -> None:
        source = tmp_path / "data"
        source.mkdir()
        (source / "conversations.json").write_text("[]")
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["convert", str(source), "-o", str(out), "--provider", "nonexistent"]
        )
        assert result.exit_code == 1
        assert "Unknown provider" in result.output


class TestCLIErrorPaths:
    def test_validate_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("{not valid json}")
        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output

    def test_validate_invalid_schema(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text('{"schema": "portable-ai-memory", "schema_version": "1.0"}')
        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 1
        assert "Schema validation failed" in result.output

    def test_validate_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        result = runner.invoke(app, ["validate", str(d)])
        assert result.exit_code == 1
        assert "No PAM files found" in result.output

    def test_convert_no_source_file(self, tmp_path: Path) -> None:
        source = tmp_path / "empty"
        source.mkdir()
        out = tmp_path / "out"
        result = runner.invoke(app, ["convert", str(source), "-o", str(out)])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_convert_undetectable_provider(self, tmp_path: Path) -> None:
        source = tmp_path / "data"
        source.mkdir()
        (source / "conversations.json").write_text('{"unknown_format": true}')
        out = tmp_path / "out"
        result = runner.invoke(app, ["convert", str(source), "-o", str(out)])
        assert result.exit_code == 1
        assert "No converter matched" in result.output
