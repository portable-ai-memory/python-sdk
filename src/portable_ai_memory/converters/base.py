"""Base converter and auto-detection for provider exports.

Each provider converter subclasses BaseConverter and implements the
conversion from raw provider export to PAM format.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

    from portable_ai_memory.models.conversation import Conversation, ImportMetadata
    from portable_ai_memory.models.memory_store import MemoryObject, MemoryStore


class BaseConverter(ABC):
    """Abstract base for provider export converters.

    To create a custom converter::

        from portable_ai_memory.converters import BaseConverter, register_converter

        @register_converter
        class MyConverter(BaseConverter):
            @property
            def provider_name(self) -> str:
                return "my_provider"

            def detect(self, data, path=None) -> bool:
                return isinstance(data, dict) and "my_key" in data

            def convert_conversations(self, data, *, owner_id="unknown",
                                      import_metadata=None):
                # ... build list[Conversation] from data ...
                return conversations

    Optional overrides:

    - ``convert_memories()`` — if the provider exports memories directly.
    - ``extract_owner_info()`` — read supplementary files (profiles, etc.).
    - ``post_process()`` — inject feedback, copy attachments after conversion.

    Note: ``post_process()`` is called by the CLI automatically, but must be
    called manually when using converters programmatically.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short provider identifier (e.g., 'chatgpt', 'claude')."""

    @abstractmethod
    def detect(self, data: dict[str, Any] | list[Any], path: Path | None = None) -> bool:
        """Return True if this converter can handle the given data.

        Args:
            data: Parsed JSON from the export file.
            path: Optional file path for filename-based heuristics.
        """

    @abstractmethod
    def convert_conversations(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
        import_metadata: ImportMetadata | None = None,
    ) -> list[Conversation]:
        """Convert raw provider conversations to PAM Conversation format.

        Args:
            data: Raw parsed JSON from the provider export.
            owner_id: Owner identifier for provenance.
            import_metadata: Optional import provenance metadata.

        Returns:
            List of PAM Conversation objects.
        """

    def convert_memories(
        self,
        data: dict[str, Any] | list[Any],
        *,
        owner_id: str = "unknown",
    ) -> MemoryStore:
        """Convert raw provider data to a PAM MemoryStore.

        Override in subclasses that export memories directly (e.g. Claude).
        Default returns an empty MemoryStore.

        Args:
            data: Raw parsed JSON from the provider export.
            owner_id: Owner identifier.

        Returns:
            A PAM MemoryStore.
        """
        from portable_ai_memory.models.memory_store import MemoryStore as MS
        from portable_ai_memory.models.memory_store import Owner

        return MS(schema_version="1.0", owner=Owner(id=owner_id), memories=[])

    def extract_owner_info(
        self,
        source_dir: Path,
        owner_id: str = "unknown",
    ) -> tuple[str, list[MemoryObject]]:
        """Extract owner ID and extra memories from supplementary files.

        Override in subclasses to read provider-specific files (user profiles,
        memories, projects) alongside the main conversations export.

        Args:
            source_dir: Directory containing the provider export.
            owner_id: Current owner identifier (may be updated).

        Returns:
            Tuple of (possibly updated owner_id, list of extra MemoryObjects).
        """
        return owner_id, []

    def post_process(
        self,
        conversations: list[Conversation],
        source_dir: Path,
        output_dir: Path,
    ) -> int:
        """Post-process conversations after conversion.

        Override in subclasses to inject feedback, copy attachments, or
        perform other provider-specific post-processing.

        Args:
            conversations: Converted conversations (may be mutated in-place).
            source_dir: Directory containing the provider export.
            output_dir: Output bundle directory.

        Returns:
            Number of attachments copied/processed.
        """
        return 0


def validate_converter_output(conversations: list[Conversation]) -> None:
    """Validate converter output, raising PAMValidationError on invalid conversations.

    Call this after ``convert_conversations()`` to catch converter bugs early.

    Raises:
        PAMValidationError: If any conversation fails deep validation.
    """
    from portable_ai_memory.core.validator import validate_conversation
    from portable_ai_memory.exceptions import PAMValidationError

    for conv in conversations:
        result = validate_conversation(conv)
        if not result.is_valid:
            errors = "; ".join(str(e) for e in result.errors)
            raise PAMValidationError(
                f"Converter produced invalid conversation '{conv.id}': {errors}"
            )


# ── Registry & auto-detection ─────────────────────────────

_CONVERTERS: list[type[BaseConverter]] = []


def register_converter(cls: type[BaseConverter]) -> type[BaseConverter]:
    """Class decorator to register a converter for auto-detection.

    Registered converters are tried by ``detect_provider()`` in registration order.

    Example::

        @register_converter
        class MyConverter(BaseConverter):
            ...
    """
    _CONVERTERS.append(cls)
    return cls


def detect_provider(path: str | Path) -> BaseConverter:
    """Auto-detect the provider of an export file.

    Tries each registered converter's detect() method.

    Args:
        path: Path to the export file.

    Returns:
        An instantiated converter.

    Raises:
        ProviderNotDetectedError: If no converter matches the file.
    """
    from portable_ai_memory.exceptions import ProviderNotDetectedError

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProviderNotDetectedError(f"Cannot parse {path.name}: {e}") from e

    for converter_cls in _CONVERTERS:
        converter = converter_cls()
        if converter.detect(raw, path):
            return converter

    raise ProviderNotDetectedError(f"No converter matched the format of {path.name}")


def list_converters() -> list[str]:
    """Return provider names of all registered converters.

    Returns:
        List of provider name strings (e.g., ``['chatgpt', 'claude', 'gemini', ...]``).
    """
    return [cls().provider_name for cls in _CONVERTERS]


def parse_iso_or_now(ts: str | None) -> datetime:
    """Parse an ISO 8601 timestamp string, or return current UTC time if None/empty.

    Args:
        ts: ISO 8601 timestamp string, or None.

    Returns:
        Timezone-aware datetime (UTC if no timezone in source).
    """
    from datetime import UTC, datetime

    if not ts:
        return datetime.now(tz=UTC)
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
