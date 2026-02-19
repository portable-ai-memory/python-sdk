"""Tests for Anthropic/Claude converter."""

from __future__ import annotations

from datetime import UTC, datetime

from portable_ai_memory.converters.anthropic import AnthropicConverter
from portable_ai_memory.models.conversation import ImportMetadata
from portable_ai_memory.models.enums import AttachmentType, MessageRole


class TestAnthropicDetection:
    def test_detects_claude_conversations(self) -> None:
        converter = AnthropicConverter()
        data = [{"uuid": "conv-1", "chat_messages": []}]
        assert converter.detect(data) is True

    def test_rejects_chatgpt(self) -> None:
        converter = AnthropicConverter()
        data = [{"id": "conv-1", "mapping": {}}]
        assert converter.detect(data) is False

    def test_rejects_empty(self) -> None:
        converter = AnthropicConverter()
        assert converter.detect([]) is False


class TestAnthropicConversion:
    def _sample_data(self) -> list:  # type: ignore[type-arg]
        return [
            {
                "uuid": "conv-001",
                "name": "Test Claude Chat",
                "created_at": "2026-01-01T10:00:00Z",
                "updated_at": "2026-01-01T11:00:00Z",
                "summary": "A test conversation",
                "account": {"uuid": "acc-123"},
                "chat_messages": [
                    {
                        "uuid": "msg-001",
                        "sender": "human",
                        "text": "Hello Claude",
                        "content": [],
                        "created_at": "2026-01-01T10:00:00Z",
                    },
                    {
                        "uuid": "msg-002",
                        "sender": "assistant",
                        "text": "Hello! How can I help?",
                        "content": [
                            {
                                "type": "text",
                                "text": "Hello! How can I help?",
                            }
                        ],
                        "created_at": "2026-01-01T10:01:00Z",
                    },
                ],
            }
        ]

    def test_convert_conversations(self) -> None:
        converter = AnthropicConverter()
        convs = converter.convert_conversations(self._sample_data())
        assert len(convs) == 1
        conv = convs[0]
        assert conv.title == "Test Claude Chat"
        assert conv.provider.name == "claude"
        assert len(conv.messages) == 2

    def test_role_mapping(self) -> None:
        converter = AnthropicConverter()
        convs = converter.convert_conversations(self._sample_data())
        msgs = convs[0].messages
        assert msgs[0].role == MessageRole.USER
        assert msgs[1].role == MessageRole.ASSISTANT

    def test_summary_in_raw_metadata(self) -> None:
        converter = AnthropicConverter()
        convs = converter.convert_conversations(self._sample_data())
        assert convs[0].raw_metadata.get("summary") == "A test conversation"

    def test_tool_use_extraction(self) -> None:
        converter = AnthropicConverter()
        data = [
            {
                "uuid": "conv-tools",
                "name": "Tool test",
                "created_at": "2026-01-01T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-001",
                        "sender": "assistant",
                        "text": "",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "web_search",
                                "input": {"query": "test"},
                                "id": "tool-1",
                            }
                        ],
                        "created_at": "2026-01-01T10:00:00Z",
                    }
                ],
            }
        ]
        convs = converter.convert_conversations(data)
        msg = convs[0].messages[0]
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "web_search"

    def test_thinking_extraction(self) -> None:
        converter = AnthropicConverter()
        data = [
            {
                "uuid": "conv-think",
                "name": "Thinking test",
                "created_at": "2026-01-01T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-001",
                        "sender": "assistant",
                        "text": "",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "Let me think about this...",
                            }
                        ],
                        "created_at": "2026-01-01T10:00:00Z",
                    }
                ],
            }
        ]
        convs = converter.convert_conversations(data)
        msg = convs[0].messages[0]
        assert msg.is_thought is True

    def test_convert_memories(self) -> None:
        converter = AnthropicConverter()
        mem_data = [
            {
                "conversations_memory": "User prefers dark mode.",
                "project_memories": {"proj-001": "Project X uses FastAPI and PostgreSQL."},
                "account_uuid": "acc-123",
            }
        ]
        store = converter.convert_memories(mem_data)
        assert len(store.memories) == 2
        types = [m.type.value for m in store.memories]
        assert "context" in types
        assert "project" in types

    def test_convert_memories_empty(self) -> None:
        converter = AnthropicConverter()
        store = converter.convert_memories([{}])
        assert len(store.memories) == 0

    def test_import_metadata_passed_to_conversations(self) -> None:
        converter = AnthropicConverter()
        meta = ImportMetadata(
            importer="pam-cli/0.1.0",
            imported_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_file="conversations.json",
            source_checksum="sha256:" + "a" * 64,
        )
        convs = converter.convert_conversations(
            self._sample_data(), owner_id="test", import_metadata=meta
        )
        assert convs[0].import_metadata is not None
        assert convs[0].import_metadata.importer == "pam-cli/0.1.0"

    def test_files_attachment_extraction(self) -> None:
        converter = AnthropicConverter()
        data = [
            {
                "uuid": "conv-files",
                "name": "File test",
                "created_at": "2026-01-01T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-001",
                        "sender": "human",
                        "text": "Check this file",
                        "content": [],
                        "created_at": "2026-01-01T10:00:00Z",
                        "files": [{"file_name": "report.pdf"}],
                        "attachments": [],
                    }
                ],
            }
        ]
        convs = converter.convert_conversations(data)
        msg = convs[0].messages[0]
        assert len(msg.attachments) == 1
        assert msg.attachments[0].type == AttachmentType.FILE
        assert msg.attachments[0].name == "report.pdf"

    def test_files_image_detection(self) -> None:
        converter = AnthropicConverter()
        data = [
            {
                "uuid": "conv-img",
                "name": "Image test",
                "created_at": "2026-01-01T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-001",
                        "sender": "human",
                        "text": "",
                        "content": [],
                        "created_at": "2026-01-01T10:00:00Z",
                        "files": [{"file_name": "photo.png"}],
                        "attachments": [],
                    }
                ],
            }
        ]
        convs = converter.convert_conversations(data)
        assert convs[0].messages[0].attachments[0].type == AttachmentType.IMAGE

    def test_attachments_extraction_with_content(self) -> None:
        converter = AnthropicConverter()
        data = [
            {
                "uuid": "conv-att",
                "name": "Attachment test",
                "created_at": "2026-01-01T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-001",
                        "sender": "human",
                        "text": "Check this",
                        "content": [],
                        "created_at": "2026-01-01T10:00:00Z",
                        "files": [],
                        "attachments": [
                            {
                                "file_name": "",
                                "file_size": 10414,
                                "file_type": "txt",
                                "extracted_content": "some terminal output here",
                            }
                        ],
                    }
                ],
            }
        ]
        convs = converter.convert_conversations(data)
        msg = convs[0].messages[0]
        assert len(msg.attachments) == 1
        assert msg.attachments[0].type == AttachmentType.DOCUMENT
        assert msg.attachments[0].size_bytes == 10414
        assert "attachment_extracts" in msg.raw_metadata

    def test_convert_project_docs(self) -> None:
        converter = AnthropicConverter()
        proj_data = [
            {
                "uuid": "proj-001",
                "name": "My Project",
                "docs": [
                    {
                        "uuid": "doc-001",
                        "filename": "README.md",
                        "content": "# My Project\nSome content here.",
                    },
                    {
                        "uuid": "doc-002",
                        "filename": "empty.md",
                        "content": "",
                    },
                ],
            },
            {
                "uuid": "proj-002",
                "name": "Empty Project",
                "docs": [],
            },
        ]
        memories = converter.convert_project_docs(proj_data)
        assert len(memories) == 1
        assert memories[0].type.value == "project"
        assert memories[0].summary is not None
        assert "My Project" in memories[0].summary
