"""Tests for Microsoft/Copilot CSV converter."""

from __future__ import annotations

from datetime import UTC, datetime

from portable_ai_memory.converters.microsoft import MicrosoftConverter
from portable_ai_memory.models.conversation import ImportMetadata
from portable_ai_memory.models.enums import MessageRole


class TestMicrosoftDetection:
    def test_detects_csv_header_format_a(self) -> None:
        converter = MicrosoftConverter()
        assert converter.detect_csv(["Conversation", "Time", "Author", "Message"]) is True

    def test_detects_csv_header_format_b(self) -> None:
        converter = MicrosoftConverter()
        assert converter.detect_csv(["CreatedAt", "MessageContent", "Author", "ChatName"]) is True

    def test_detects_csv_header_format_c(self) -> None:
        converter = MicrosoftConverter()
        assert converter.detect_csv(["Timestamp", "ClientApp", "Prompt"]) is True

    def test_rejects_unknown_headers(self) -> None:
        converter = MicrosoftConverter()
        assert converter.detect_csv(["foo", "bar"]) is False


class TestMicrosoftConversion:
    def test_convert_format_a(self) -> None:
        converter = MicrosoftConverter()
        rows = [
            {
                "Conversation": "Test Chat",
                "Time": "2026-01-01T10:00:00",
                "Author": "user",
                "Message": "Hello",
            },
            {
                "Conversation": "Test Chat",
                "Time": "2026-01-01T10:01:00",
                "Author": "AI",
                "Message": "Hi there!",
            },
        ]
        convs = converter.convert_conversations(rows)
        assert len(convs) == 1
        assert len(convs[0].messages) == 2
        assert convs[0].messages[0].role == MessageRole.USER
        assert convs[0].messages[1].role == MessageRole.ASSISTANT

    def test_convert_format_b(self) -> None:
        converter = MicrosoftConverter()
        rows = [
            {
                "ChatName": "Research",
                "CreatedAt": "2026-01-01T10:00:00",
                "Author": "user",
                "MessageContent": "Query",
            },
            {
                "ChatName": "Research",
                "CreatedAt": "2026-01-01T10:01:00",
                "Author": "Copilot",
                "MessageContent": "Answer",
            },
        ]
        convs = converter.convert_conversations(rows)
        assert len(convs) == 1
        assert convs[0].title == "Research"

    def test_convert_from_csv_text(self) -> None:
        converter = MicrosoftConverter()
        csv_text = (
            "Conversation,Time,Author,Message\n"
            "Test,2026-01-01T10:00:00,user,Hello\n"
            "Test,2026-01-01T10:01:00,AI,Hi\n"
        )
        convs = converter.convert_conversations_from_csv(csv_text)
        assert len(convs) == 1
        assert len(convs[0].messages) == 2

    def test_groups_by_conversation_name(self) -> None:
        converter = MicrosoftConverter()
        rows = [
            {
                "Conversation": "Chat A",
                "Time": "2026-01-01T10:00:00",
                "Author": "user",
                "Message": "Hello A",
            },
            {
                "Conversation": "Chat B",
                "Time": "2026-01-01T10:00:00",
                "Author": "user",
                "Message": "Hello B",
            },
        ]
        convs = converter.convert_conversations(rows)
        assert len(convs) == 2

    def test_role_normalization_human(self) -> None:
        """Human author should map to USER role."""
        assert MicrosoftConverter._normalize_role("Human") == MessageRole.USER
        assert MicrosoftConverter._normalize_role("human") == MessageRole.USER
        assert MicrosoftConverter._normalize_role("user") == MessageRole.USER

    def test_role_normalization_assistant(self) -> None:
        """AI, bot, Copilot should all map to ASSISTANT role."""
        assert MicrosoftConverter._normalize_role("AI") == MessageRole.ASSISTANT
        assert MicrosoftConverter._normalize_role("bot") == MessageRole.ASSISTANT
        assert MicrosoftConverter._normalize_role("Copilot") == MessageRole.ASSISTANT

    def test_message_reorder_user_before_assistant(self) -> None:
        """When AI and Human share the same timestamp, Human should come first."""
        converter = MicrosoftConverter()
        rows = [
            {
                "Conversation": "Chat",
                "Time": "2026-01-01T10:00:00",
                "Author": "AI",
                "Message": "Response",
            },
            {
                "Conversation": "Chat",
                "Time": "2026-01-01T10:00:00",
                "Author": "Human",
                "Message": "Prompt",
            },
        ]
        convs = converter.convert_conversations(rows)
        assert convs[0].messages[0].role == MessageRole.USER
        assert convs[0].messages[1].role == MessageRole.ASSISTANT

    def test_import_metadata_passed(self) -> None:
        converter = MicrosoftConverter()
        meta = ImportMetadata(
            importer="pam-cli/0.1.0",
            imported_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_file="copilot-activity-history.csv",
            source_checksum="sha256:" + "a" * 64,
        )
        rows = [
            {
                "Conversation": "Test",
                "Time": "2026-01-01T10:00:00",
                "Author": "user",
                "Message": "Hello",
            },
        ]
        convs = converter.convert_conversations(rows, import_metadata=meta)
        assert convs[0].import_metadata is not None
        assert convs[0].import_metadata.source_file == "copilot-activity-history.csv"

    def test_source_file_in_raw_metadata(self) -> None:
        converter = MicrosoftConverter()
        csv_text = "Conversation,Time,Author,Message\nTest,2026-01-01T10:00:00,user,Hello\n"
        convs = converter.convert_conversations_from_csv(
            csv_text, source_file="copilot-activity-history.csv"
        )
        assert convs[0].raw_metadata["source_file"] == "copilot-activity-history.csv"

    def test_windows_app_format_c(self) -> None:
        """Format C (Timestamp, ClientApp, Prompt) should create single-message convos."""
        converter = MicrosoftConverter()
        rows = [
            {
                "Timestamp": "2026-01-01T10:00:00",
                "ClientApp": "Notepad",
                "Prompt": "Summarize this text",
            },
        ]
        convs = converter.convert_conversations(rows)
        assert len(convs) == 1
        assert convs[0].messages[0].role == MessageRole.USER
        assert convs[0].messages[0].content.text == "Summarize this text"
        assert convs[0].messages[0].raw_metadata["client_app"] == "Notepad"

    def test_us_date_format_parsing(self) -> None:
        """Copilot chat-activity uses M/D/YYYY H:MM:SS +00:00 format."""
        ts = MicrosoftConverter._parse_timestamp("12/13/2025 12:07:15 +00:00")
        assert ts.year == 2025
        assert ts.month == 12
        assert ts.day == 13

    def test_bom_handling_in_csv(self) -> None:
        converter = MicrosoftConverter()
        csv_text = "\ufeffConversation,Time,Author,Message\nTest,2026-01-01T10:00:00,user,Hello\n"
        convs = converter.convert_conversations_from_csv(csv_text)
        assert len(convs) == 1


class TestParseTimestamp:
    """Edge-case tests for MicrosoftConverter._parse_timestamp."""

    def test_us_date_no_seconds(self) -> None:
        """US date format without seconds: M/D/YYYY H:MM."""
        ts = MicrosoftConverter._parse_timestamp("01/15/2025 14:30")
        assert ts.year == 2025
        assert ts.month == 1
        assert ts.day == 15
        assert ts.hour == 14
        assert ts.minute == 30
        assert ts.tzinfo is not None  # should be UTC

    def test_invalid_timestamp_falls_back(self) -> None:
        """Completely invalid string falls back to datetime.now(tz=UTC)."""
        before = datetime.now(tz=UTC)
        ts = MicrosoftConverter._parse_timestamp("not-a-date-at-all")
        after = datetime.now(tz=UTC)
        assert before <= ts <= after
        assert ts.tzinfo is not None

    def test_empty_string_falls_back(self) -> None:
        """Empty string falls back to datetime.now(tz=UTC)."""
        before = datetime.now(tz=UTC)
        ts = MicrosoftConverter._parse_timestamp("")
        after = datetime.now(tz=UTC)
        assert before <= ts <= after
        assert ts.tzinfo is not None

    def test_iso_with_timezone_offset_preserved(self) -> None:
        """ISO format with explicit timezone offset preserves the original timezone."""
        ts = MicrosoftConverter._parse_timestamp("2024-01-15T10:30:00+05:00")
        assert ts.year == 2024
        assert ts.hour == 10
        assert ts.utcoffset() is not None
        # +05:00 offset in total seconds
        assert ts.utcoffset().total_seconds() == 5 * 3600  # type: ignore[union-attr]

    def test_iso_naive_gets_utc(self) -> None:
        """ISO format without timezone info gets UTC assigned."""
        ts = MicrosoftConverter._parse_timestamp("2024-01-15T10:30:00")
        assert ts.year == 2024
        assert ts.hour == 10
        assert ts.tzinfo is UTC
