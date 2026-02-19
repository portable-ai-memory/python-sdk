"""Google (Gemini) Takeout export converter.

Handles MyActivity.json and Minhaatividade.html (localized) from Google Takeout → Gemini Apps.

See importer-mappings.md §3 for field-by-field mapping details.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from uuid import uuid4

from portable_ai_memory.converters.base import BaseConverter, register_converter
from portable_ai_memory.models.conversation import (
    Attachment,
    Conversation,
    ConversationTemporal,
    ImportMetadata,
    Message,
    MessageContent,
    Participant,
    ProviderInfo,
)
from portable_ai_memory.models.enums import AttachmentType, ContentType, MessageRole

_CONV_ID_RE = re.compile(r"/c/([a-zA-Z0-9_-]+)")

# Event-level keys already mapped to PAM fields — not duplicated in raw_metadata
_MAPPED_EVENT_KEYS = {
    "header",
    "title",
    "titleUrl",
    "time",
    "details",
    "userInteractions",
}

# Gap threshold: interactions >60 min apart on the same day = different conversations
_SESSION_GAP = timedelta(minutes=60)

# Month abbreviations across locales (first 3 letters, lowercase)
# Covers: EN, PT, ES, FR, DE, IT, NL, PL, TR, RO, and more
_MONTH_NAMES: dict[str, int] = {
    # English
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    # Portuguese
    "fev": 2,
    "abr": 4,
    "mai": 5,
    "ago": 8,
    "set": 9,
    "out": 10,
    "dez": 12,
    # Spanish
    "ene": 1,
    "dic": 12,
    # French
    "fév": 2,
    "avr": 4,
    "jui": 6,
    "aoû": 8,
    # German
    "jän": 1,
    "mär": 3,
    "okt": 10,
    # Italian
    "gen": 1,
    "mag": 5,
    "giu": 6,
    "lug": 7,
    "ott": 10,
    # Dutch
    "mrt": 3,
    "mei": 5,
}

# Common timezone abbreviations → UTC offset in hours
_TZ_OFFSETS: dict[str, int] = {
    "UTC": 0,
    "GMT": 0,
    "BRT": -3,
    "BRST": -2,
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "CET": 1,
    "CEST": 2,
    "WET": 0,
    "WEST": 1,
    "JST": 9,
    "KST": 9,
    "CST_CN": 8,
    "IST": 5,
}

# Pattern A: "17 de fev. de 2026, 15:08:20 BRT" (PT/ES/FR — day de month de year)
_TS_PATTERN_A = re.compile(
    r"(\d{1,2})\s+de\s+(\w+)\.?\s+de\s+(\d{4}),\s+"
    r"(\d{1,2}):(\d{2}):(\d{2})\s+(\w+)"
)

# Pattern B: "Feb 17, 2026, 3:08:20 PM EST" (EN — month day, year, time AM/PM tz)
_TS_PATTERN_B = re.compile(
    r"(\w+)\.?\s+(\d{1,2}),\s+(\d{4}),\s+"
    r"(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)?\s+(\w+)"
)

# Pattern C: "17. Feb. 2026, 15:08:20 CET" (DE — day. month. year)
_TS_PATTERN_C = re.compile(
    r"(\d{1,2})\.?\s+(\w+)\.?\s+(\d{4}),\s+"
    r"(\d{1,2}):(\d{2}):(\d{2})\s+(\w+)"
)


def _resolve_tz(tz_abbr: str) -> timezone:
    """Resolve timezone abbreviation to offset."""
    offset = _TZ_OFFSETS.get(tz_abbr.upper())
    if offset is not None:
        return timezone(timedelta(hours=offset))
    return timezone(timedelta(hours=0))  # fallback to UTC


def _parse_month(month_str: str) -> int | None:
    """Parse month name/abbreviation in any supported language."""
    key = month_str.lower()[:3]
    return _MONTH_NAMES.get(key)


def _parse_timestamp(text: str) -> datetime | None:
    """Parse timestamp from Gemini HTML in any supported locale."""
    # Pattern A: PT/ES/FR — "17 de fev. de 2026, 15:08:20 BRT"
    match = _TS_PATTERN_A.search(text)
    if match:
        day, month_str, year = int(match.group(1)), match.group(2), int(match.group(3))
        hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
        month = _parse_month(month_str)
        if month:
            tz = _resolve_tz(match.group(7))
            return datetime(year, month, day, hour, minute, second, tzinfo=tz)

    # Pattern B: EN — "Feb 17, 2026, 3:08:20 PM EST"
    match = _TS_PATTERN_B.search(text)
    if match:
        month_str, day, year = match.group(1), int(match.group(2)), int(match.group(3))
        hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
        ampm = match.group(7)
        if ampm:
            if ampm.upper() == "PM" and hour != 12:
                hour += 12
            elif ampm.upper() == "AM" and hour == 12:
                hour = 0
        month = _parse_month(month_str)
        if month:
            tz = _resolve_tz(match.group(8))
            return datetime(year, month, day, hour, minute, second, tzinfo=tz)

    # Pattern C: DE — "17. Feb. 2026, 15:08:20 CET"
    match = _TS_PATTERN_C.search(text)
    if match:
        day, month_str, year = int(match.group(1)), match.group(2), int(match.group(3))
        hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
        month = _parse_month(month_str)
        if month:
            tz = _resolve_tz(match.group(7))
            return datetime(year, month, day, hour, minute, second, tzinfo=tz)

    return None


class _HtmlToText(HTMLParser):
    """Convert HTML to plain text preserving minimal structure."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("p", "br", "hr", "tr"):
            self._parts.append("\n")
        elif tag in ("li",):
            self._parts.append("\n- ")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")
        elif tag == "code":
            self._parts.append("`")
        elif tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")
        elif tag == "code":
            self._parts.append("`")
        elif tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Normalize multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    """Convert HTML fragment to plain text."""
    parser = _HtmlToText()
    parser.feed(html)
    return parser.get_text()


# Known activity keywords across locales (used as hint, not required)
_ACTIVITY_KEYWORDS = re.compile(
    r"^(?:Prompted|Solicitou|Solicitó|Demandé|Angefordert|Richiesto|Prompted)\s+",
    re.IGNORECASE,
)

# Broad timestamp anchor: find the line containing "YYYY, H:MM:SS"
# We search for this pattern and then back-track to find the full timestamp start
_TS_ANCHOR_RE = re.compile(r"\d{4},\s+\d{1,2}:\d{2}:\d{2}")


def _parse_html_blocks(html: str) -> list[dict[str, Any]]:
    """Parse Gemini HTML activity export into structured blocks.

    Language-agnostic: uses HTML structure (outer-cell → content-cell → <br> separators)
    rather than locale-specific keywords.
    """
    blocks: list[dict[str, Any]] = []

    # Split by outer-cell divs
    parts = re.split(r'<div\s+class="outer-cell[^"]*"', html)
    for part in parts[1:]:  # skip before first outer-cell
        # Find body-1 content-cell (first match; second is empty right column)
        content_match = re.search(
            r'class="content-cell[^"]*mdl-typography--body-1">(.*?)</div>',
            part,
            re.DOTALL,
        )
        if not content_match:
            continue

        content = content_match.group(1).strip()

        # Structure: {keyword} {prompt}<br>{optional attachments}<br>{timestamp}<br>{response}
        # Find timestamp anchor (YYYY, H:MM:SS) and back-track to <br> before it
        anchor = _TS_ANCHOR_RE.search(content)
        if not anchor:
            continue

        # Back-track from anchor to find start of timestamp line (after last <br>)
        pre_anchor = content[: anchor.start()]
        last_br = pre_anchor.rfind("<br>")
        ts_line_start = last_br + len("<br>") if last_br >= 0 else 0

        # Everything before the timestamp line is prompt + optional attachments
        before_ts = content[:ts_line_start].rstrip()
        # Remove trailing <br> before timestamp
        before_ts = re.sub(r"<br>\s*$", "", before_ts)

        # Extract the raw prompt: first segment before first <br>
        first_br = before_ts.find("<br>")
        raw_prompt = before_ts[:first_br] if first_br >= 0 else before_ts

        # Strip activity keyword prefix if present
        prompt_text = _ACTIVITY_KEYWORDS.sub("", raw_prompt).strip()
        prompt_text = _html_to_text(prompt_text)

        if not prompt_text:
            continue

        # Parse the timestamp from the timestamp line
        timestamp = _parse_timestamp(content[ts_line_start:])

        # Extract attached files (local hrefs, not http)
        file_refs: list[dict[str, str]] = []
        for file_match in re.finditer(r'<a href="([^"]+)">([^<]+)</a>', content):
            href = file_match.group(1)
            name = file_match.group(2)
            if not href.startswith("http"):
                file_refs.append({"href": href, "name": name})

        # Response: everything after timestamp+timezone+<br>
        # Find the full timestamp+tz match and skip past it
        response_html = ""
        for pattern in (_TS_PATTERN_A, _TS_PATTERN_B, _TS_PATTERN_C):
            full_ts = pattern.search(content)
            if full_ts:
                after_ts = content[full_ts.end() :]
                br_match = re.match(r"\s*<br>", after_ts)
                response_html = after_ts[br_match.end() :] if br_match else after_ts
                break

        response_text = _html_to_text(response_html) if response_html else ""

        blocks.append(
            {
                "prompt": prompt_text,
                "response": response_text,
                "timestamp": timestamp,
                "files": file_refs,
                "raw_html_response": response_html.strip() if response_html else "",
            }
        )

    return blocks


def _group_blocks_into_conversations(
    blocks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group blocks into conversations by temporal proximity.

    Blocks are in reverse chronological order from the HTML. We reverse to chronological,
    then group by session gap (>60 min between interactions = new conversation).
    """
    if not blocks:
        return []

    # Sort chronologically
    sorted_blocks = sorted(
        blocks,
        key=lambda b: (
            b["timestamp"]
            if isinstance(b["timestamp"], datetime)
            else datetime.min.replace(tzinfo=UTC)
        ),
    )

    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        prev_ts = current_group[-1].get("timestamp")
        curr_ts = block.get("timestamp")

        if prev_ts and curr_ts and isinstance(prev_ts, datetime) and isinstance(curr_ts, datetime):
            gap = curr_ts - prev_ts
            if gap > _SESSION_GAP:
                groups.append(current_group)
                current_group = [block]
                continue

        current_group.append(block)

    if current_group:
        groups.append(current_group)

    return groups


@register_converter
class GoogleConverter(BaseConverter):
    """Convert Google Gemini Takeout exports to PAM format."""

    @property
    def provider_name(self) -> str:
        return "gemini"

    @staticmethod
    def find_activity_file(source: Path) -> Path | None:
        """Find Gemini activity file (JSON or HTML) in Takeout directory structure."""
        _GEMINI_NAMES = (
            "MyActivity.json",
            "Minhaatividade.html",
            "Minha atividade.html",
            "MyActivity.html",
        )
        for name in _GEMINI_NAMES:
            direct = source / name
            if direct.exists():
                return direct

        for pattern in (
            "My Activity/Gemini Apps/MyActivity.json",
            "*/My Activity/Gemini Apps/MyActivity.json",
            "Takeout/My Activity/Gemini Apps/MyActivity.json",
            "My Activity/Gemini Apps/*.html",
            "*/My Activity/Gemini Apps/*.html",
        ):
            matches = list(source.glob(pattern))
            if matches:
                return matches[0]

        for html_file in source.glob("*.html"):
            try:
                snippet = html_file.read_text(encoding="utf-8", errors="ignore")[:2000]
                if "Gemini Apps" in snippet and ("Prompted" in snippet or "Solicitou" in snippet):
                    return html_file
            except OSError:
                continue

        return None

    def detect(self, data: dict[str, Any] | list[Any], path: Path | None = None) -> bool:
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict) and first.get("header") in ("Gemini", "Gemini Apps"):
                return True
        return False

    def detect_html(self, html: str, path: Path | None = None) -> bool:
        """Detect Gemini HTML activity export."""
        if "Gemini Apps" in html and ("Prompted" in html or "Solicitou" in html):
            return True
        return bool(path and path.stem in ("MyActivity", "Minhaatividade", "Minha atividade"))

    def convert_conversations(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
        import_metadata: ImportMetadata | None = None,
    ) -> list[Conversation]:
        if not isinstance(data, list):
            return []

        # Group activity events by conversation ID
        groups: dict[str, list[dict[str, Any]]] = {}
        for event in data:
            if not isinstance(event, dict):
                continue
            conv_id = self._extract_conv_id(event)
            groups.setdefault(conv_id, []).append(event)

        conversations: list[Conversation] = []
        for conv_id, events in groups.items():
            conv = self._build_conversation(
                conv_id,
                events,
                owner_id=owner_id,
                import_metadata=import_metadata,
            )
            if conv is not None:
                conversations.append(conv)
        return conversations

    def convert_conversations_from_html(
        self,
        html: str,
        *,
        owner_id: str = "unknown",
        import_metadata: ImportMetadata | None = None,
        source_dir: Path | None = None,
    ) -> list[Conversation]:
        """Convert Gemini HTML activity export to PAM conversations."""
        blocks = _parse_html_blocks(html)
        if not blocks:
            return []

        conv_groups = _group_blocks_into_conversations(blocks)
        conversations: list[Conversation] = []

        for group in conv_groups:
            conv = self._build_conversation_from_blocks(
                group,
                owner_id=owner_id,
                import_metadata=import_metadata,
                source_dir=source_dir,
            )
            if conv is not None:
                conversations.append(conv)

        return conversations

    def _extract_conv_id(self, event: dict[str, Any]) -> str:
        title_url = event.get("titleUrl", "")
        match = _CONV_ID_RE.search(title_url)
        if match:
            return match.group(1)
        return str(uuid4().hex[:12])

    def _build_conversation_from_blocks(
        self,
        blocks: list[dict[str, Any]],
        *,
        owner_id: str,
        import_metadata: ImportMetadata | None = None,
        source_dir: Path | None = None,
    ) -> Conversation | None:
        messages: list[Message] = []
        timestamps: list[datetime] = []
        conv_id = uuid4().hex[:12]

        for block in blocks:
            ts = block.get("timestamp")
            if isinstance(ts, datetime):
                timestamps.append(ts)
            else:
                ts = datetime.now(tz=UTC)

            prompt = str(block.get("prompt", ""))
            response = str(block.get("response", ""))
            files = block.get("files", [])

            # Attachments from file references
            attachments: list[Attachment] = []
            if isinstance(files, list):
                for f in files:
                    if isinstance(f, dict):
                        name = str(f.get("name", ""))
                        href = str(f.get("href", ""))
                        attachments.append(
                            Attachment(
                                type=AttachmentType.FILE,
                                name=name,
                                ref=href,
                            )
                        )

            raw_metadata: dict[str, Any] = {}
            raw_html = block.get("raw_html_response", "")
            if raw_html:
                raw_metadata["response_html"] = raw_html

            if prompt:
                messages.append(
                    Message(
                        id=str(uuid4()),
                        role=MessageRole.USER,
                        content=MessageContent(type=ContentType.TEXT, text=prompt),
                        created_at=ts,
                        attachments=attachments if attachments else [],
                        raw_metadata={},
                    )
                )
            if response:
                messages.append(
                    Message(
                        id=str(uuid4()),
                        role=MessageRole.ASSISTANT,
                        content=MessageContent(type=ContentType.TEXT, text=response),
                        created_at=ts,
                        raw_metadata=raw_metadata if raw_metadata else {},
                    )
                )

        if not messages:
            return None

        created_at = min(timestamps) if timestamps else datetime.now(tz=UTC)
        updated_at = max(timestamps) if len(timestamps) > 1 else None

        title = None
        for msg in messages:
            if msg.role == MessageRole.USER and msg.content and msg.content.text:
                title = msg.content.text[:100]
                break

        return Conversation(
            schema_version="1.0",
            id=conv_id,
            provider=ProviderInfo(
                name="gemini",
                conversation_id=conv_id,
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
        )

    def _build_conversation(
        self,
        conv_id: str,
        events: list[dict[str, Any]],
        *,
        owner_id: str,
        import_metadata: ImportMetadata | None = None,
    ) -> Conversation | None:
        # Sort events by time
        events.sort(key=lambda e: e.get("time", ""))

        messages: list[Message] = []
        timestamps: list[datetime] = []

        for event in events:
            event_time_str = event.get("time")
            event_time = (
                datetime.fromisoformat(event_time_str) if event_time_str else datetime.now(tz=UTC)
            )
            timestamps.append(event_time)

            # Collect event-level extras (products, activityControls, etc.)
            event_extras: dict[str, Any] = {}
            for key, val in event.items():
                if key in _MAPPED_EVENT_KEYS:
                    continue
                if val is None:
                    continue
                event_extras[key] = val

            # Variant A: details array
            details = event.get("details", [])
            if details:
                for detail in details:
                    if not isinstance(detail, dict):
                        continue
                    name = detail.get("name", "")
                    value = detail.get("value", "")
                    if not value:
                        continue
                    role = MessageRole.USER if name == "Request" else MessageRole.ASSISTANT

                    # Preserve non-mapped detail fields
                    detail_extras: dict[str, Any] = {}
                    for k, v in detail.items():
                        if k in ("name", "value") or v is None:
                            continue
                        detail_extras[k] = v

                    raw_metadata: dict[str, Any] = {}
                    if event_extras:
                        raw_metadata["event"] = event_extras
                    if detail_extras:
                        raw_metadata["detail"] = detail_extras

                    messages.append(
                        Message(
                            id=str(uuid4()),
                            role=role,
                            content=MessageContent(type=ContentType.TEXT, text=value),
                            created_at=event_time,
                            raw_metadata=raw_metadata if raw_metadata else {},
                        )
                    )

            # Variant B: userInteractions array
            interactions = event.get("userInteractions", [])
            if interactions and not details:
                for interaction in interactions:
                    if not isinstance(interaction, dict):
                        continue
                    ui = interaction.get("userInteraction", {})
                    if not isinstance(ui, dict):
                        continue
                    request = ui.get("request", "")
                    response = ui.get("response", "")

                    # Preserve interaction extras (endpoint, shown, latencySeconds)
                    ui_extras: dict[str, Any] = {}
                    for k, v in ui.items():
                        if k in ("request", "response") or v is None:
                            continue
                        ui_extras[k] = v

                    if request:
                        raw_meta_req: dict[str, Any] = {}
                        if event_extras:
                            raw_meta_req["event"] = event_extras
                        if ui_extras:
                            raw_meta_req["interaction"] = ui_extras

                        messages.append(
                            Message(
                                id=str(uuid4()),
                                role=MessageRole.USER,
                                content=MessageContent(
                                    type=ContentType.TEXT,
                                    text=str(request),
                                ),
                                created_at=event_time,
                                raw_metadata=raw_meta_req if raw_meta_req else {},
                            )
                        )
                    if response:
                        raw_meta_resp: dict[str, Any] = {}
                        if event_extras:
                            raw_meta_resp["event"] = event_extras
                        if ui_extras:
                            raw_meta_resp["interaction"] = ui_extras

                        messages.append(
                            Message(
                                id=str(uuid4()),
                                role=MessageRole.ASSISTANT,
                                content=MessageContent(
                                    type=ContentType.TEXT,
                                    text=str(response),
                                ),
                                created_at=event_time,
                                raw_metadata=raw_meta_resp if raw_meta_resp else {},
                            )
                        )

        if not messages:
            return None

        created_at = min(timestamps) if timestamps else datetime.now(tz=UTC)
        updated_at = max(timestamps) if len(timestamps) > 1 else None

        # Title from first user message
        title = None
        for msg in messages:
            if msg.role == MessageRole.USER and msg.content and msg.content.text:
                title = msg.content.text[:100]
                break

        return Conversation(
            schema_version="1.0",
            id=conv_id,
            provider=ProviderInfo(
                name="gemini",
                conversation_id=conv_id,
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
        )
