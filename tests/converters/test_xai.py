"""Tests for xAI/Grok converter."""

from __future__ import annotations

from datetime import UTC, datetime

from portable_ai_memory.converters.xai import XAIConverter
from portable_ai_memory.models.conversation import ImportMetadata
from portable_ai_memory.models.enums import AttachmentType, MessageRole


class TestXAIDetection:
    def test_detects_grok_export(self) -> None:
        converter = XAIConverter()
        data = {
            "conversations": [
                {
                    "conversation": {"id": "conv-1"},
                    "responses": [],
                }
            ]
        }
        assert converter.detect(data) is True

    def test_rejects_non_grok(self) -> None:
        converter = XAIConverter()
        assert converter.detect({"conversations": [{"mapping": {}}]}) is False
        assert converter.detect([]) is False


class TestXAIConversion:
    def _sample_data(self) -> dict:  # type: ignore[type-arg]
        return {
            "conversations": [
                {
                    "conversation": {
                        "id": "conv-001",
                        "title": "Test Grok Chat",
                        "create_time": "2026-01-01T10:00:00Z",
                        "modify_time": "2026-01-01T11:00:00Z",
                        "user_id": "user-123",
                    },
                    "responses": [
                        {
                            "response": {
                                "_id": "msg-001",
                                "sender": "human",
                                "message": "Hello Grok",
                                "create_time": {"$date": {"$numberLong": "1735689600000"}},
                                "parent_response_id": None,
                            },
                            "share_link": None,
                        },
                        {
                            "response": {
                                "_id": "msg-002",
                                "sender": "ASSISTANT",
                                "message": "Hello! I'm Grok.",
                                "model": "grok-3",
                                "create_time": {"$date": {"$numberLong": "1735689610000"}},
                                "parent_response_id": "msg-001",
                                "cited_web_search_results": [
                                    {
                                        "title": "Grok",
                                        "url": "https://grok.x.ai",
                                        "preview": "AI assistant",
                                    }
                                ],
                            },
                            "share_link": None,
                        },
                    ],
                }
            ]
        }

    def test_convert_conversations(self) -> None:
        converter = XAIConverter()
        convs = converter.convert_conversations(self._sample_data())
        assert len(convs) == 1
        conv = convs[0]
        assert conv.title == "Test Grok Chat"
        assert conv.provider.name == "grok"
        assert len(conv.messages) == 2

    def test_role_normalization(self) -> None:
        converter = XAIConverter()
        convs = converter.convert_conversations(self._sample_data())
        msgs = convs[0].messages
        assert msgs[0].role == MessageRole.USER
        assert msgs[1].role == MessageRole.ASSISTANT

    def test_bson_timestamp_parsing(self) -> None:
        converter = XAIConverter()
        convs = converter.convert_conversations(self._sample_data())
        msg = convs[0].messages[0]
        assert msg.created_at.year >= 2024

    def test_citations(self) -> None:
        converter = XAIConverter()
        convs = converter.convert_conversations(self._sample_data())
        msg = convs[0].messages[1]
        assert len(msg.citations) == 1
        assert msg.citations[0].url == "https://grok.x.ai"

    def test_parent_child_relationships(self) -> None:
        converter = XAIConverter()
        convs = converter.convert_conversations(self._sample_data())
        msgs = convs[0].messages
        assert msgs[1].parent_id == "msg-001"
        assert "msg-002" in msgs[0].children_ids

    def test_sender_model_name_as_assistant(self) -> None:
        assert XAIConverter._normalize_role("grok-3") == MessageRole.ASSISTANT
        assert XAIConverter._normalize_role("ASSISTANT") == MessageRole.ASSISTANT
        assert XAIConverter._normalize_role("human") == MessageRole.USER

    def test_import_metadata_passed(self) -> None:
        converter = XAIConverter()
        meta = ImportMetadata(
            importer="pam-cli/0.1.0",
            imported_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_file="prod-grok-backend.json",
            source_checksum="sha256:" + "a" * 64,
        )
        convs = converter.convert_conversations(
            self._sample_data(), owner_id="test", import_metadata=meta
        )
        assert convs[0].import_metadata is not None
        assert convs[0].import_metadata.importer == "pam-cli/0.1.0"

    def test_all_response_extras_preserved(self) -> None:
        converter = XAIConverter()
        data = {
            "conversations": [
                {
                    "conversation": {
                        "id": "conv-extras",
                        "title": "Extras test",
                        "create_time": "2026-01-01T10:00:00Z",
                    },
                    "responses": [
                        {
                            "response": {
                                "_id": "msg-001",
                                "sender": "assistant",
                                "message": "Thinking...",
                                "model": "grok-3",
                                "create_time": {"$date": {"$numberLong": "1735689600000"}},
                                "thinking_trace": "Let me think...",
                                "thinking_start_time": {"$date": {"$numberLong": "1735689590000"}},
                                "thinking_end_time": {"$date": {"$numberLong": "1735689600000"}},
                                "metadata": {"ui_layout": {"effort": "HIGH"}},
                                "webpage_urls": ["https://example.com"],
                                "card_attachments_json": ['{"id":"abc"}'],
                                "xai_user_id": "user-123",
                                "partial": True,
                                "media_types": ["audio"],
                            },
                            "share_link": {"share_link_id": "sl-1", "is_public": True},
                        }
                    ],
                }
            ]
        }
        convs = converter.convert_conversations(data)
        msg = convs[0].messages[0]
        rm = msg.raw_metadata
        assert rm["thinking_trace"] == "Let me think..."
        assert "thinking_start_time" in rm
        assert "thinking_end_time" in rm
        assert rm["grok_metadata"] == {"ui_layout": {"effort": "HIGH"}}
        assert rm["webpage_urls"] == ["https://example.com"]
        assert rm["card_attachments_json"] == ['{"id":"abc"}']
        assert rm["xai_user_id"] == "user-123"
        assert rm["partial"] is True
        assert rm["media_types"] == ["audio"]
        assert rm["share_link"]["share_link_id"] == "sl-1"

    def test_file_attachments(self) -> None:
        converter = XAIConverter()
        data = {
            "conversations": [
                {
                    "conversation": {
                        "id": "conv-att",
                        "title": "Attachment test",
                        "create_time": "2026-01-01T10:00:00Z",
                    },
                    "responses": [
                        {
                            "response": {
                                "_id": "msg-001",
                                "sender": "human",
                                "message": "Check this",
                                "create_time": {"$date": {"$numberLong": "1735689600000"}},
                                "file_attachments": ["uuid-abc", "uuid-def"],
                            },
                            "share_link": None,
                        }
                    ],
                }
            ]
        }
        convs = converter.convert_conversations(data)
        msg = convs[0].messages[0]
        assert len(msg.attachments) == 2
        assert msg.attachments[0].type == AttachmentType.FILE
        assert msg.attachments[0].provider_id == "uuid-abc"

    def test_conversation_extras_preserved(self) -> None:
        converter = XAIConverter()
        data = {
            "conversations": [
                {
                    "conversation": {
                        "id": "conv-meta",
                        "title": "Meta test",
                        "create_time": "2026-01-01T10:00:00Z",
                        "summary": "A test conversation",
                        "starred": True,
                        "temporary": False,
                        "system_prompt_name": "Default",
                    },
                    "responses": [
                        {
                            "response": {
                                "_id": "msg-001",
                                "sender": "human",
                                "message": "Hi",
                                "create_time": {"$date": {"$numberLong": "1735689600000"}},
                            },
                            "share_link": None,
                        }
                    ],
                }
            ]
        }
        convs = converter.convert_conversations(data)
        rm = convs[0].raw_metadata
        assert rm["summary"] == "A test conversation"
        assert rm["starred"] is True
        assert rm["temporary"] is False
        assert rm["system_prompt_name"] == "Default"

    def test_convert_projects(self) -> None:
        converter = XAIConverter()
        data = {
            "conversations": [],
            "projects": [
                {
                    "workspace_id": "proj-001",
                    "name": "Test Project",
                    "custom_personality": "Be helpful",
                    "preferred_model": "grok-3",
                    "conversation_starters": ["Hello"],
                }
            ],
            "media_posts": [],
            "tasks": [],
        }
        mems = converter.convert_projects(data)
        assert len(mems) == 1
        assert mems[0].type.value == "project"
        assert "Test Project" in mems[0].content
        assert "Be helpful" in mems[0].content

    def test_non_dict_response_wrapper_skipped(self) -> None:
        """Non-dict entries in responses list should be silently skipped."""
        converter = XAIConverter()
        data = {
            "conversations": [
                {
                    "conversation": {
                        "id": "conv-skip",
                        "title": "Skip test",
                        "create_time": "2026-01-01T10:00:00Z",
                    },
                    "responses": [
                        "not a dict",
                        42,
                        None,
                    ],
                }
            ]
        }
        convs = converter.convert_conversations(data)
        # No valid messages -> conversation is skipped entirely
        assert len(convs) == 0

    def test_response_field_not_dict_skipped(self) -> None:
        """Wrapper where 'response' is not a dict should be skipped."""
        converter = XAIConverter()
        data = {
            "conversations": [
                {
                    "conversation": {
                        "id": "conv-bad-resp",
                        "title": "Bad response",
                        "create_time": "2026-01-01T10:00:00Z",
                    },
                    "responses": [
                        {"response": "string, not dict", "share_link": None},
                        {"response": 42, "share_link": None},
                    ],
                }
            ]
        }
        convs = converter.convert_conversations(data)
        assert len(convs) == 0

    def test_mix_valid_and_invalid_responses(self) -> None:
        """Only valid response wrappers should produce messages."""
        converter = XAIConverter()
        data = {
            "conversations": [
                {
                    "conversation": {
                        "id": "conv-mix",
                        "title": "Mixed",
                        "create_time": "2026-01-01T10:00:00Z",
                    },
                    "responses": [
                        "invalid-string",
                        {
                            "response": {
                                "_id": "msg-valid",
                                "sender": "human",
                                "message": "Hello",
                                "create_time": {"$date": {"$numberLong": "1735689600000"}},
                            },
                            "share_link": None,
                        },
                        {"response": None, "share_link": None},
                    ],
                }
            ]
        }
        convs = converter.convert_conversations(data)
        assert len(convs) == 1
        assert len(convs[0].messages) == 1
        assert convs[0].messages[0].id == "msg-valid"

    def test_convert_media_posts(self) -> None:
        converter = XAIConverter()
        data = {
            "conversations": [],
            "projects": [],
            "media_posts": [
                {
                    "id": "post-001",
                    "original_prompt": "A cat in space",
                    "media_type": "image",
                    "create_time": "2026-01-01T10:00:00Z",
                    "link": "https://grok.com/imagine/post/post-001",
                }
            ],
            "tasks": [],
        }
        mems = converter.convert_media_posts(data)
        assert len(mems) == 1
        assert "cat in space" in mems[0].content
        assert mems[0].summary == "Grok image generation"
