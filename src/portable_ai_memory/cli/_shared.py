"""Shared CLI state and helpers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from portable_ai_memory import __version__
from portable_ai_memory.models.conversation import ImportMetadata

try:
    from rich.console import Console
except ImportError as e:
    raise SystemExit(
        "CLI dependencies not installed. Run: pip install 'portable-ai-memory[cli]'"
    ) from e

console = Console()


def compute_source_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum string for source file bytes."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def create_import_metadata(source_file: str, source_checksum: str) -> ImportMetadata:
    """Build standard ImportMetadata for CLI-converted files."""
    return ImportMetadata(
        importer=f"pam-cli/{__version__}",
        imported_at=datetime.now(tz=UTC),
        source_file=source_file,
        source_checksum=source_checksum,
    )
