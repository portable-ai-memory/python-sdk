"""Tests for OpenAI/ChatGPT converter."""

from __future__ import annotations

from portable_ai_memory.converters.openai import OpenAIConverter
from portable_ai_memory.models.enums import MessageRole


class TestOpenAIDetection:
    def test_detects_conversations_json(self) -> None:
        converter = OpenAIConverter()
        data = [{"id": "conv-1", "mapping": {"node-1": {}}}]
        assert converter.detect(data) is True

    def test_rejects_empty_list(self) -> None:
        converter = OpenAIConverter()
        assert converter.detect([]) is False

    def test_rejects_non_chatgpt(self) -> None:
        converter = OpenAIConverter()
        assert converter.detect([{"chat_messages": []}]) is False


class TestOpenAIConversion:
    def _sample_data(self) -> list:  # type: ignore[type-arg]
        return [
            {
                "id": "conv-001",
                "title": "Test conversation",
                "create_time": 1700000000.0,
                "update_time": 1700001000.0,
                "default_model_slug": "gpt-4",
                "mapping": {
                    "node-1": {
                        "id": "node-1",
                        "parent": None,
                        "children": ["node-2"],
                        "message": {
                            "id": "msg-1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Hello"]},
                            "create_time": 1700000000.0,
                            "metadata": {},
                        },
                    },
                    "node-2": {
                        "id": "node-2",
                        "parent": "node-1",
                        "children": [],
                        "message": {
                            "id": "msg-2",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Hi there!"]},
                            "create_time": 1700000010.0,
                            "metadata": {"model_slug": "gpt-4"},
                        },
                    },
                },
            }
        ]

    def test_convert_conversations(self) -> None:
        converter = OpenAIConverter()
        convs = converter.convert_conversations(self._sample_data())
        assert len(convs) == 1
        conv = convs[0]
        assert conv.title == "Test conversation"
        assert conv.provider.name == "chatgpt"
        assert len(conv.messages) == 2

    def test_message_roles(self) -> None:
        converter = OpenAIConverter()
        convs = converter.convert_conversations(self._sample_data())
        msgs = convs[0].messages
        roles = [m.role for m in msgs]
        assert MessageRole.USER in roles
        assert MessageRole.ASSISTANT in roles

    def test_skips_null_messages(self) -> None:
        converter = OpenAIConverter()
        data = [
            {
                "id": "conv-001",
                "create_time": 1700000000.0,
                "mapping": {
                    "node-1": {"message": None, "children": ["node-2"]},
                    "node-2": {
                        "parent": "node-1",
                        "children": [],
                        "message": {
                            "id": "msg-1",
                            "author": {"role": "user"},
                            "content": {"parts": ["Hello"]},
                            "create_time": 1700000000.0,
                            "metadata": {},
                        },
                    },
                },
            }
        ]
        convs = converter.convert_conversations(data)
        assert len(convs[0].messages) == 1

    def test_convert_memories_empty(self) -> None:
        converter = OpenAIConverter()
        store = converter.convert_memories(self._sample_data())
        assert len(store.memories) == 0

    def test_role_normalization(self) -> None:
        assert OpenAIConverter._normalize_role("user") == MessageRole.USER
        assert OpenAIConverter._normalize_role("assistant") == MessageRole.ASSISTANT
        assert OpenAIConverter._normalize_role("system") == MessageRole.SYSTEM
        assert OpenAIConverter._normalize_role("tool") == MessageRole.TOOL
        assert OpenAIConverter._normalize_role("human") == MessageRole.USER


def _wrap_conv(mapping: dict, **kwargs) -> list:  # type: ignore[type-arg]
    """Helper: wrap a mapping dict into a minimal conversations.json list."""
    base = {
        "id": "conv-t",
        "create_time": 1700000000.0,
        "mapping": mapping,
    }
    base.update(kwargs)
    return [base]


class TestConvertConversationsNonList:
    """Line 62 — convert_conversations returns [] for non-list input."""

    def test_returns_empty_for_dict(self) -> None:
        converter = OpenAIConverter()
        assert converter.convert_conversations({"not": "a list"}) == []


class TestWeightZeroKept:
    """Messages with weight=0.0 are kept (alternate DAG branches)."""

    def test_keeps_weight_zero(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["alternate response"]},
                    "create_time": 1.0,
                    "weight": 0.0,
                    "metadata": {},
                },
            },
        }
        convs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))
        assert len(convs[0].messages) == 1
        assert convs[0].messages[0].content.text == "alternate response"


class TestIsVisuallyHidden:
    """Line 175 — visually hidden messages are skipped."""

    def test_skips_hidden(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["hidden"]},
                    "create_time": 1.0,
                    "metadata": {"is_visually_hidden_from_conversation": True},
                },
            },
        }
        convs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))
        assert len(convs[0].messages) == 0


class TestThoughtsContent:
    """Lines 191-196 — thoughts content type."""

    def test_thoughts_extracted(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "thoughts",
                        "thoughts": [
                            {"content": "step 1"},
                            {"content": "step 2"},
                            {"content": ""},
                        ],
                    },
                    "create_time": 1.0,
                    "metadata": {},
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert len(msgs) == 1
        assert msgs[0].is_thought is True
        assert "step 1" in msgs[0].content.text
        assert "step 2" in msgs[0].content.text


class TestReasoningRecap:
    """Lines 197-199 — reasoning_recap content type."""

    def test_reasoning_recap(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "reasoning_recap",
                        "content": "recap text",
                    },
                    "create_time": 1.0,
                    "metadata": {},
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert msgs[0].is_thought is True
        assert msgs[0].content.text == "recap text"


class TestSonicWebpage:
    """Lines 215-224 — sonic_webpage part."""

    def test_sonic_webpage(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "text",
                        "parts": [
                            {
                                "content_type": "sonic_webpage",
                                "text": "page content",
                                "title": "Page Title",
                                "url": "https://example.com",
                                "snippet": "snip",
                            }
                        ],
                    },
                    "create_time": 1.0,
                    "metadata": {},
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert "page content" in msgs[0].content.text
        assert len(msgs[0].citations) == 1
        assert msgs[0].citations[0].url == "https://example.com"


class TestTetherBrowsingDisplay:
    """Lines 225-227 — tether_browsing_display part."""

    def test_tether_browsing_display(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "text",
                        "parts": [
                            {
                                "content_type": "tether_browsing_display",
                                "result": "browsing result",
                            }
                        ],
                    },
                    "create_time": 1.0,
                    "metadata": {},
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert msgs[0].content.text == "browsing result"


class TestTetherQuote:
    """Lines 228-237 — tether_quote part."""

    def test_tether_quote(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "text",
                        "parts": [
                            {
                                "content_type": "tether_quote",
                                "text": "quoted text",
                                "title": "Source",
                                "url": "https://src.com",
                            }
                        ],
                    },
                    "create_time": 1.0,
                    "metadata": {},
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert "quoted text" in msgs[0].content.text
        assert msgs[0].citations[0].snippet == "quoted text"


class TestToolCalls:
    """Lines 290-307 — tool call extraction."""

    def test_assistant_tool_call(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["call code"]},
                    "create_time": 1.0,
                    "recipient": "python",
                    "metadata": {},
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0].name == "python"

    def test_tool_response(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "tool", "name": "python"},
                    "content": {"content_type": "text", "parts": ["result"]},
                    "create_time": 1.0,
                    "metadata": {},
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert msgs[0].role == MessageRole.TOOL
        assert msgs[0].tool_calls[0].name == "python"
        assert msgs[0].tool_calls[0].output == "result"


class TestMetadataCitations:
    """Lines 257-276 — citations from metadata and search_result_groups."""

    def test_metadata_citations(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["text"]},
                    "create_time": 1.0,
                    "metadata": {
                        "citations": [
                            {"metadata": {"title": "T", "url": "https://u.com", "text": "s"}}
                        ],
                    },
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert any(c.url == "https://u.com" for c in msgs[0].citations)

    def test_search_result_groups(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["text"]},
                    "create_time": 1.0,
                    "metadata": {
                        "search_result_groups": [
                            {"entries": [{"title": "R", "url": "https://r.com", "snippet": "sn"}]}
                        ],
                    },
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert any(c.url == "https://r.com" for c in msgs[0].citations)


class TestEmptyMessageSkipped:
    """Empty messages (no text, attachments, or citations) are skipped."""

    def test_empty_system_skipped(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "system"},
                    "content": {"content_type": "text", "parts": []},
                    "create_time": 1.0,
                    "metadata": {},
                },
            },
        }
        msgs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))[0].messages
        assert len(msgs) == 0


class TestIsArchived:
    def test_is_archived_set(self) -> None:
        conv_data = _wrap_conv({}, is_archived=True)
        convs = OpenAIConverter().convert_conversations(conv_data)
        assert convs[0].is_archived is True


class TestSystemInstruction:
    def test_user_editable_context(self) -> None:
        mapping = {
            "n1": {
                "message": {
                    "id": "m1",
                    "author": {"role": "system"},
                    "content": {
                        "content_type": "user_editable_context",
                        "user_instructions": "Be concise",
                        "user_profile": "Dev",
                    },
                    "create_time": 1.0,
                    "metadata": {"is_visually_hidden_from_conversation": True},
                },
            },
        }
        convs = OpenAIConverter().convert_conversations(_wrap_conv(mapping))
        assert "Be concise" in convs[0].system_instruction
        assert "Dev" in convs[0].system_instruction
