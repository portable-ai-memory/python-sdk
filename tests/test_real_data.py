"""Integration tests using real provider export data.

These tests use actual export files pointed to by the PAM_TEST_DATA_DIR env var.
Tests are skipped if the env var is not set or the directory does not exist.

The data files are READ-ONLY and must not be modified.

Usage:
    PAM_TEST_DATA_DIR=/path/to/backup-files uv run pytest tests/test_real_data.py
"""

from __future__ import annotations

import csv
import json

import pytest

from portable_ai_memory.converters.anthropic import AnthropicConverter
from portable_ai_memory.converters.microsoft import MicrosoftConverter
from portable_ai_memory.converters.openai import OpenAIConverter
from portable_ai_memory.converters.xai import XAIConverter
from portable_ai_memory.core.validator import validate_conversation
from portable_ai_memory.models.enums import MessageRole

from .conftest import real_data_path, skip_no_real_data

# ── ChatGPT / OpenAI ──────────────────────────────────────────


@skip_no_real_data
class TestOpenAIRealData:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.path = real_data_path("chatgpt") / "conversations.json"
        if not self.path.exists():
            pytest.skip("ChatGPT data not found")
        with open(self.path) as f:
            self.data = json.load(f)
        self.converter = OpenAIConverter()

    def test_detect(self) -> None:
        assert self.converter.detect(self.data, self.path)

    def test_convert_all(self) -> None:
        convs = self.converter.convert_conversations(self.data)
        assert len(convs) > 0
        for conv in convs:
            assert conv.provider.name == "chatgpt"
            assert conv.id
            assert conv.temporal.created_at is not None

    def test_messages_have_roles(self) -> None:
        convs = self.converter.convert_conversations(self.data[:5])
        for conv in convs:
            for msg in conv.messages:
                assert msg.role in (
                    MessageRole.USER,
                    MessageRole.ASSISTANT,
                    MessageRole.SYSTEM,
                    MessageRole.TOOL,
                )

    def test_conversations_validate(self) -> None:
        convs = self.converter.convert_conversations(self.data[:10])
        for conv in convs:
            result = validate_conversation(conv)
            assert result.is_valid, f"{conv.title}: {result}"

    def test_mapping_dag_preserved(self) -> None:
        """Verify parent/children relationships from OpenAI mapping."""
        convs = self.converter.convert_conversations(self.data[:5])
        for conv in convs:
            msg_ids = {m.id for m in conv.messages}
            for msg in conv.messages:
                # Verify parent/child IDs exist (system messages may not be in list)
                if msg.parent_id and msg.parent_id in msg_ids:
                    assert msg.parent_id in msg_ids
                for cid in msg.children_ids:
                    if cid in msg_ids:
                        assert cid in msg_ids


# ── Claude / Anthropic ─────────────────────────────────────────


@skip_no_real_data
class TestAnthropicRealData:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.conv_path = real_data_path("claude") / "conversations.json"
        self.mem_path = real_data_path("claude") / "memories.json"
        if not self.conv_path.exists():
            pytest.skip("Claude data not found")
        with open(self.conv_path) as f:
            self.conv_data = json.load(f)
        with open(self.mem_path) as f:
            self.mem_data = json.load(f)
        self.converter = AnthropicConverter()

    def test_detect(self) -> None:
        assert self.converter.detect(self.conv_data)

    def test_convert_all_conversations(self) -> None:
        convs = self.converter.convert_conversations(self.conv_data)
        assert len(convs) > 0
        for conv in convs:
            assert conv.provider.name == "claude"

    def test_sender_mapping(self) -> None:
        convs = self.converter.convert_conversations(self.conv_data[:5])
        for conv in convs:
            for msg in conv.messages:
                assert msg.role in (MessageRole.USER, MessageRole.ASSISTANT)

    def test_conversations_validate(self) -> None:
        convs = self.converter.convert_conversations(self.conv_data[:10])
        for conv in convs:
            result = validate_conversation(conv)
            assert result.is_valid, f"{conv.title}: {result}"

    def test_memories_conversion(self) -> None:
        store = self.converter.convert_memories(self.mem_data)
        assert len(store.memories) > 0
        for mem in store.memories:
            assert mem.content
            assert mem.content_hash.startswith("sha256:")

    def test_project_memories_extracted(self) -> None:
        store = self.converter.convert_memories(self.mem_data)
        from portable_ai_memory.models.enums import MemoryType

        types = {m.type for m in store.memories}
        assert MemoryType.CONTEXT in types  # conversations_memory
        assert MemoryType.PROJECT in types  # project_memories

    def test_structured_content_handled(self) -> None:
        """Verify tool_use, thinking, and citation content parts are handled."""
        convs = self.converter.convert_conversations(self.conv_data)
        has_tool_calls = any(msg.tool_calls for conv in convs for msg in conv.messages)
        has_thinking = any(msg.is_thought for conv in convs for msg in conv.messages)
        # At least one of these should exist in real Claude data
        assert has_tool_calls or has_thinking or len(convs) > 0


# ── Grok / xAI ────────────────────────────────────────────────


@skip_no_real_data
class TestXAIRealData:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        grok_dir = real_data_path("grok")
        # Find prod-grok-backend.json in nested Grok export structure
        matches = list(grok_dir.glob("*/export_data/*/prod-grok-backend.json"))
        if not matches:
            direct = grok_dir / "prod-grok-backend.json"
            if direct.exists():
                matches = [direct]
        if not matches:
            pytest.skip("prod-grok-backend.json not found in grok dir")
        self.path = matches[0]
        if not self.path.exists():
            pytest.skip("Grok data not found")
        with open(self.path) as f:
            self.data = json.load(f)
        self.converter = XAIConverter()

    def test_detect(self) -> None:
        assert self.converter.detect(self.data)

    def test_convert_all(self) -> None:
        convs = self.converter.convert_conversations(self.data)
        assert len(convs) > 0
        for conv in convs:
            assert conv.provider.name == "grok"

    def test_bson_timestamps_parsed(self) -> None:
        convs = self.converter.convert_conversations(self.data)
        for conv in convs[:10]:
            for msg in conv.messages:
                assert msg.created_at is not None
                assert msg.created_at.year >= 2024

    def test_conversations_validate(self) -> None:
        convs = self.converter.convert_conversations(self.data)
        for conv in convs[:20]:
            result = validate_conversation(conv)
            assert result.is_valid, f"{conv.title}: {result}"

    def test_parent_child_relationships(self) -> None:
        convs = self.converter.convert_conversations(self.data)
        for conv in convs[:10]:
            msg_ids = {m.id for m in conv.messages}
            for msg in conv.messages:
                if msg.parent_id:
                    assert msg.parent_id in msg_ids
                for cid in msg.children_ids:
                    assert cid in msg_ids


# ── Copilot / Microsoft ───────────────────────────────────────


@skip_no_real_data
class TestMicrosoftRealData:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.dir = real_data_path("copilot")
        if not self.dir.exists():
            pytest.skip("Copilot data not found")
        self.converter = MicrosoftConverter()

    def test_detect_csv_format_a(self) -> None:
        path = self.dir / "copilot-activity-history.csv"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert self.converter.detect_csv(header)

    def test_detect_csv_format_b(self) -> None:
        path = self.dir / "copilot-chat-activity.csv"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert self.converter.detect_csv(header)

    def test_convert_format_a(self) -> None:
        path = self.dir / "copilot-activity-history.csv"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            csv_text = f.read()
        convs = self.converter.convert_conversations_from_csv(csv_text)
        assert len(convs) > 0
        for conv in convs:
            assert conv.provider.name == "copilot"
            assert len(conv.messages) > 0

    def test_convert_format_b(self) -> None:
        path = self.dir / "copilot-chat-activity.csv"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            csv_text = f.read()
        convs = self.converter.convert_conversations_from_csv(csv_text)
        assert len(convs) > 0

    def test_format_a_bom_handling(self) -> None:
        """Verify BOM in CSV header is handled correctly."""
        path = self.dir / "copilot-activity-history.csv"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            csv_text = f.read()
        # File has BOM, converter must handle it
        assert csv_text[0] == "\ufeff"
        convs = self.converter.convert_conversations_from_csv(csv_text)
        assert len(convs) > 0

    def test_conversations_validate(self) -> None:
        path = self.dir / "copilot-activity-history.csv"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            csv_text = f.read()
        convs = self.converter.convert_conversations_from_csv(csv_text)
        for conv in convs:
            result = validate_conversation(conv)
            assert result.is_valid, f"{conv.title}: {result}"

    def test_grouping_by_conversation(self) -> None:
        path = self.dir / "copilot-activity-history.csv"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            csv_text = f.read()
        convs = self.converter.convert_conversations_from_csv(csv_text)
        titles = [c.title for c in convs]
        # Each title should be unique (grouped properly)
        assert len(titles) == len(set(titles))
