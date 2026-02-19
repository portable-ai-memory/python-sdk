"""Tests for Google/Gemini Takeout converter."""

from __future__ import annotations

from datetime import UTC, datetime

from portable_ai_memory.converters.google import (
    GoogleConverter,
    _group_blocks_into_conversations,
    _html_to_text,
    _parse_html_blocks,
    _parse_timestamp,
)
from portable_ai_memory.models.conversation import ImportMetadata
from portable_ai_memory.models.enums import AttachmentType, MessageRole


class TestGoogleDetection:
    def test_detects_gemini_activity(self) -> None:
        converter = GoogleConverter()
        data = [{"header": "Gemini", "title": "Used Gemini Apps", "details": []}]
        assert converter.detect(data) is True

    def test_detects_gemini_apps_header(self) -> None:
        converter = GoogleConverter()
        data = [{"header": "Gemini Apps", "userInteractions": []}]
        assert converter.detect(data) is True

    def test_rejects_non_gemini(self) -> None:
        converter = GoogleConverter()
        assert converter.detect([{"mapping": {}}]) is False

    def test_rejects_non_gemini_header(self) -> None:
        """A list with 'header' key but not Gemini/Gemini Apps should be rejected."""
        converter = GoogleConverter()
        data = [{"header": "Google Maps", "title": "Used Google Maps", "details": []}]
        assert converter.detect(data) is False

    def test_rejects_details_without_header(self) -> None:
        """A list with 'details' but no 'header' should be rejected."""
        converter = GoogleConverter()
        data = [{"details": [{"name": "Request", "value": "test"}]}]
        assert converter.detect(data) is False

    def test_rejects_empty_list(self) -> None:
        """An empty list should be rejected."""
        converter = GoogleConverter()
        assert converter.detect([]) is False


class TestGoogleConversion:
    def _sample_details(self) -> list:  # type: ignore[type-arg]
        return [
            {
                "header": "Gemini",
                "title": "Used Gemini Apps",
                "titleUrl": "https://gemini.google.com/app/c/conv-001",
                "time": "2026-01-01T10:00:00Z",
                "products": ["Gemini Apps"],
                "details": [
                    {"name": "Request", "value": "What is Python?"},
                    {"name": "Response", "value": "Python is a programming language."},
                ],
            },
            {
                "header": "Gemini",
                "title": "Used Gemini Apps",
                "titleUrl": "https://gemini.google.com/app/c/conv-001",
                "time": "2026-01-01T10:05:00Z",
                "products": ["Gemini Apps"],
                "details": [
                    {"name": "Request", "value": "Tell me more"},
                    {"name": "Response", "value": "It was created by Guido van Rossum."},
                ],
            },
        ]

    def test_groups_by_conversation(self) -> None:
        converter = GoogleConverter()
        convs = converter.convert_conversations(self._sample_details())
        assert len(convs) == 1
        assert len(convs[0].messages) == 4

    def test_role_mapping(self) -> None:
        converter = GoogleConverter()
        convs = converter.convert_conversations(self._sample_details())
        msgs = convs[0].messages
        assert msgs[0].role == MessageRole.USER
        assert msgs[1].role == MessageRole.ASSISTANT

    def test_title_from_first_user_message(self) -> None:
        converter = GoogleConverter()
        convs = converter.convert_conversations(self._sample_details())
        assert convs[0].title == "What is Python?"

    def test_user_interactions_variant(self) -> None:
        converter = GoogleConverter()
        data = [
            {
                "header": "Gemini",
                "titleUrl": "https://gemini.google.com/app/c/conv-002",
                "time": "2026-01-01T10:00:00Z",
                "userInteractions": [
                    {
                        "userInteraction": {
                            "request": "Hello",
                            "response": "Hi there!",
                            "endpoint": 2,
                            "shown": False,
                            "latencySeconds": 1.5,
                        }
                    }
                ],
            }
        ]
        convs = converter.convert_conversations(data)
        assert len(convs) == 1
        assert len(convs[0].messages) == 2

    def test_interaction_extras_preserved(self) -> None:
        """endpoint, shown, latencySeconds from userInteraction should be preserved."""
        converter = GoogleConverter()
        data = [
            {
                "header": "Gemini",
                "titleUrl": "https://gemini.google.com/app/c/conv-002",
                "time": "2026-01-01T10:00:00Z",
                "userInteractions": [
                    {
                        "userInteraction": {
                            "request": "Hello",
                            "response": "Hi!",
                            "endpoint": 2,
                            "shown": False,
                            "latencySeconds": 1.5,
                        }
                    }
                ],
            }
        ]
        convs = converter.convert_conversations(data)
        msg = convs[0].messages[0]
        assert msg.raw_metadata["interaction"]["endpoint"] == 2
        assert msg.raw_metadata["interaction"]["shown"] is False
        assert msg.raw_metadata["interaction"]["latencySeconds"] == 1.5

    def test_event_extras_preserved(self) -> None:
        """products and other event-level fields should be preserved."""
        converter = GoogleConverter()
        data = [
            {
                "header": "Gemini",
                "titleUrl": "https://gemini.google.com/app/c/conv-003",
                "time": "2026-01-01T10:00:00Z",
                "products": ["Gemini Apps"],
                "activityControls": ["Web & App Activity"],
                "details": [
                    {"name": "Request", "value": "Test"},
                ],
            }
        ]
        convs = converter.convert_conversations(data)
        msg = convs[0].messages[0]
        assert msg.raw_metadata["event"]["products"] == ["Gemini Apps"]
        assert msg.raw_metadata["event"]["activityControls"] == ["Web & App Activity"]

    def test_import_metadata_passed(self) -> None:
        converter = GoogleConverter()
        meta = ImportMetadata(
            importer="pam-cli/0.1.0",
            imported_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_file="MyActivity.json",
            source_checksum="sha256:" + "a" * 64,
        )
        convs = converter.convert_conversations(self._sample_details(), import_metadata=meta)
        assert convs[0].import_metadata is not None
        assert convs[0].import_metadata.source_file == "MyActivity.json"

    def test_conv_id_from_title_url(self) -> None:
        converter = GoogleConverter()
        convs = converter.convert_conversations(self._sample_details())
        assert convs[0].id == "conv-001"
        assert convs[0].provider.conversation_id == "conv-001"


class TestTimestampParsing:
    def test_parses_portuguese(self) -> None:
        ts = _parse_timestamp("17 de fev. de 2026, 15:08:20 BRT")
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 2
        assert ts.day == 17
        assert ts.hour == 15

    def test_parses_english(self) -> None:
        ts = _parse_timestamp("Feb 17, 2026, 3:08:20 PM EST")
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 2
        assert ts.day == 17
        assert ts.hour == 15  # 3 PM = 15

    def test_parses_english_am(self) -> None:
        ts = _parse_timestamp("Jan 5, 2026, 11:30:00 AM UTC")
        assert ts is not None
        assert ts.hour == 11

    def test_parses_german(self) -> None:
        ts = _parse_timestamp("17. Feb. 2026, 15:08:20 CET")
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 2

    def test_parses_spanish(self) -> None:
        ts = _parse_timestamp("17 de feb. de 2026, 15:08:20 CET")
        assert ts is not None
        assert ts.month == 2

    def test_returns_none_for_invalid(self) -> None:
        assert _parse_timestamp("not a timestamp") is None


class TestHtmlToText:
    def test_strips_tags(self) -> None:
        assert _html_to_text("<p>Hello <strong>world</strong></p>") == "Hello world"

    def test_preserves_code_backticks(self) -> None:
        result = _html_to_text("<code>pip install</code>")
        assert "`pip install`" in result

    def test_list_items(self) -> None:
        result = _html_to_text("<ul><li>one</li><li>two</li></ul>")
        assert "- one" in result
        assert "- two" in result


_SAMPLE_HTML = """<html><body>
<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
  <div class="mdl-grid">
    <div class="header-cell mdl-cell mdl-cell--12-col mdl-typography--title">
      <p class="mdl-typography--title">Gemini Apps</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">
      Prompted What is PAM?<br>17 de fev. de 2026, 14:51:53 BRT<br>
      <p>PAM is Portable AI Memory.</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"></div>
    <div class="content-cell mdl-cell mdl-cell--12-col mdl-typography--caption">
      <b>Produtos:</b> Gemini Apps
    </div>
  </div>
</div>
<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
  <div class="mdl-grid">
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">
      Prompted Tell me more<br>17 de fev. de 2026, 14:53:25 BRT<br>
      <p>It is a universal format.</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"></div>
  </div>
</div>
<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
  <div class="mdl-grid">
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">
      Prompted Linux distro?<br>19 de jan. de 2026, 13:17:54 BRT<br>
      <p>Ubuntu</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"></div>
  </div>
</div>
</body></html>"""


class TestHtmlParsing:
    def test_parse_blocks(self) -> None:
        blocks = _parse_html_blocks(_SAMPLE_HTML)
        assert len(blocks) == 3

    def test_prompt_extraction(self) -> None:
        blocks = _parse_html_blocks(_SAMPLE_HTML)
        assert blocks[0]["prompt"] == "What is PAM?"
        assert blocks[1]["prompt"] == "Tell me more"

    def test_timestamp_extraction(self) -> None:
        blocks = _parse_html_blocks(_SAMPLE_HTML)
        ts = blocks[0]["timestamp"]
        assert isinstance(ts, datetime)
        assert ts.year == 2026
        assert ts.month == 2

    def test_response_extraction(self) -> None:
        blocks = _parse_html_blocks(_SAMPLE_HTML)
        assert "PAM is Portable AI Memory" in str(blocks[0]["response"])

    def test_groups_by_session_gap(self) -> None:
        blocks = _parse_html_blocks(_SAMPLE_HTML)
        groups = _group_blocks_into_conversations(blocks)
        # Two from Feb 17 (same session), one from Jan 19 (different day)
        assert len(groups) == 2
        assert len(groups[0]) == 1  # Jan 19
        assert len(groups[1]) == 2  # Feb 17


_HTML_WITH_FILES = """<html><body>
<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
  <div class="mdl-grid">
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">
      Prompted o que é este projeto/<br>Attached 2 files.<br>-  <a href="spec-abc123.md">spec.md</a><br>-  <a href="README-abc123.md">README.md</a><br>17 de fev. de 2026, 14:51:53 BRT<br>
      <p>This is a project.</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"></div>
  </div>
</div>
</body></html>"""


class TestHtmlAttachments:
    def test_file_refs_extracted(self) -> None:
        blocks = _parse_html_blocks(_HTML_WITH_FILES)
        assert len(blocks) == 1
        files = blocks[0]["files"]
        assert len(files) == 2
        assert files[0]["name"] == "spec.md"
        assert files[0]["href"] == "spec-abc123.md"

    def test_attachments_in_conversation(self) -> None:
        converter = GoogleConverter()
        convs = converter.convert_conversations_from_html(_HTML_WITH_FILES)
        assert len(convs) == 1
        user_msg = convs[0].messages[0]
        assert len(user_msg.attachments) == 2
        assert user_msg.attachments[0].type == AttachmentType.FILE
        assert user_msg.attachments[0].name == "spec.md"
        assert user_msg.attachments[0].ref == "spec-abc123.md"


class TestHtmlDetection:
    def test_detects_gemini_html(self) -> None:
        converter = GoogleConverter()
        assert converter.detect_html(_SAMPLE_HTML) is True

    def test_rejects_non_gemini_html(self) -> None:
        converter = GoogleConverter()
        assert converter.detect_html("<html><body>Hello</body></html>") is False


_ENGLISH_HTML = """<html><body>
<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
  <div class="mdl-grid">
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">
      Prompted What is AI?<br>Feb 17, 2026, 3:08:20 PM EST<br>
      <p>AI is artificial intelligence.</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"></div>
  </div>
</div>
</body></html>"""

_GERMAN_HTML = """<html><body>
<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
  <div class="mdl-grid">
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">
      Angefordert Was ist KI?<br>17. Feb. 2026, 15:08:20 CET<br>
      <p>KI ist künstliche Intelligenz.</p>
    </div>
    <div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"></div>
  </div>
</div>
</body></html>"""


class TestHtmlMultiLocale:
    def test_english_html(self) -> None:
        blocks = _parse_html_blocks(_ENGLISH_HTML)
        assert len(blocks) == 1
        assert blocks[0]["prompt"] == "What is AI?"
        ts = blocks[0]["timestamp"]
        assert isinstance(ts, datetime)
        assert ts.hour == 15  # 3 PM

    def test_german_html(self) -> None:
        blocks = _parse_html_blocks(_GERMAN_HTML)
        assert len(blocks) == 1
        assert blocks[0]["prompt"] == "Was ist KI?"

    def test_english_full_conversion(self) -> None:
        converter = GoogleConverter()
        convs = converter.convert_conversations_from_html(_ENGLISH_HTML)
        assert len(convs) == 1
        assert convs[0].messages[0].content.text == "What is AI?"
        assert "artificial intelligence" in convs[0].messages[1].content.text


class TestHtmlConversion:
    def test_full_conversion(self) -> None:
        converter = GoogleConverter()
        convs = converter.convert_conversations_from_html(_SAMPLE_HTML)
        assert len(convs) == 2
        total_msgs = sum(len(c.messages) for c in convs)
        assert total_msgs == 6  # 3 prompts + 3 responses

    def test_import_metadata(self) -> None:
        converter = GoogleConverter()
        meta = ImportMetadata(
            importer="pam-cli/0.1.0",
            imported_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_file="Minhaatividade.html",
            source_checksum="sha256:" + "b" * 64,
        )
        convs = converter.convert_conversations_from_html(_SAMPLE_HTML, import_metadata=meta)
        assert convs[0].import_metadata is not None
        assert convs[0].import_metadata.source_file == "Minhaatividade.html"

    def test_title_from_first_prompt(self) -> None:
        converter = GoogleConverter()
        convs = converter.convert_conversations_from_html(_SAMPLE_HTML)
        # Sorted chronologically: Jan 19 first
        titles = [c.title for c in convs]
        assert "Linux distro?" in titles
        assert "What is PAM?" in titles
