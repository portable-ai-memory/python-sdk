"""OpenAI attachment resolver and copier.

Resolves asset URIs from ChatGPT exports to disk files and copies them
to the PAM output directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portable_ai_memory.models.conversation import Conversation


def resolve_attachment(ref: str, export_dir: Path) -> Path | None:
    """Resolve an OpenAI asset URI to a disk file path.

    Handles two URI schemes:
    - file-service://file-{ID} → export_dir/file-{ID}-*
    - sediment://file_{hash}   → export_dir/file_{hash}-* or export_dir/user-*/file_{hash}-*
    """
    if ref.startswith("file-service://"):
        # file-service://file-XXXXX → file-XXXXX-*.ext
        file_id = ref.removeprefix("file-service://")
        matches = list(export_dir.glob(f"{file_id}-*"))
        if matches:
            return matches[0]
    elif ref.startswith("sediment://"):
        # sediment://file_HASH → file_HASH-* in root or user-*/ subdirs
        file_id = ref.removeprefix("sediment://")
        # Try root level first
        matches = list(export_dir.glob(f"{file_id}-*"))
        if matches:
            return matches[0]
        # Try user-*/ subdirectories (DALL-E images)
        matches = list(export_dir.glob(f"user-*/{file_id}-*"))
        if matches:
            return matches[0]
    return None


def copy_attachments(
    conversations: list[Conversation],
    export_dir: Path,
    output_dir: Path,
) -> int:
    """Copy attachment files and update refs to relative paths.

    Returns:
        Number of files copied.
    """
    attachments_dir = output_dir / "attachments"
    copied = 0
    seen: set[str] = set()

    for conv in conversations:
        for msg in conv.messages:
            for att in msg.attachments:
                if not att.ref:
                    continue
                # Skip non-URI refs (already resolved)
                if not att.ref.startswith(("file-service://", "sediment://")):
                    continue
                source = resolve_attachment(att.ref, export_dir)
                if source is None:
                    continue

                # Determine destination, preserving parent dir for user-*/ paths
                rel = source.relative_to(export_dir)
                if rel.parts[0].startswith("user-"):
                    dest = attachments_dir / rel.parts[0] / source.name
                else:
                    dest = attachments_dir / source.name

                dest_ref = str(dest.relative_to(output_dir))
                if dest_ref not in seen:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                    seen.add(dest_ref)
                    copied += 1

                att.ref = dest_ref

    return copied
