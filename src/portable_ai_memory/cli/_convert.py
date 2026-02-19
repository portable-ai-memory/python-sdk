"""Convert command for PAM CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from portable_ai_memory.cli._shared import compute_source_checksum, console, create_import_metadata
from portable_ai_memory.exceptions import PAMError

if TYPE_CHECKING:
    from portable_ai_memory.models.conversation import Conversation


def convert(
    source: Path = typer.Argument(..., help="Provider export file or directory"),
    output: Path = typer.Option(..., "-o", "--output", help="Output directory for PAM bundle"),
    owner_id: str = typer.Option("unknown", "--owner-id", help="Owner ID for the PAM export"),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Force provider (chatgpt, claude, gemini, grok, copilot).",
    ),
) -> None:
    """Convert a provider export to a complete PAM bundle."""
    import json

    from portable_ai_memory.converters import detect_provider, list_converters
    from portable_ai_memory.converters.base import _CONVERTERS
    from portable_ai_memory.converters.google import GoogleConverter
    from portable_ai_memory.converters.xai import XAIConverter

    # --provider flag: resolve forced converter
    forced_converter = None
    if provider:
        for cls in _CONVERTERS:
            if cls().provider_name == provider:
                forced_converter = cls()
                break
        if forced_converter is None:
            names = ", ".join(list_converters())
            console.print(f"[red]✗ Unknown provider '{provider}'. Available: {names}[/red]")
            raise typer.Exit(1)

    # Handle forced copilot/gemini shortcuts
    if forced_converter and forced_converter.provider_name == "copilot" and source.is_dir():
        _convert_copilot_csv_dir(source, output, owner_id)
        return
    if forced_converter and forced_converter.provider_name == "gemini":
        html_files = list(source.glob("*.html")) if source.is_dir() else []
        if source.suffix.lower() == ".html" or html_files:
            html_file = source if source.is_file() else html_files[0]
            _convert_gemini_html(html_file, html_file.parent, output, owner_id)
            return

    # Find conversations file
    is_copilot_csv = False
    if source.is_dir():
        conv_file = source / "conversations.json"
        if not conv_file.exists():
            grok_file = XAIConverter.find_backend_file(source)
            if grok_file:
                conv_file = grok_file
                source = grok_file.parent
            else:
                gemini_file = GoogleConverter.find_activity_file(source)
                if gemini_file:
                    conv_file = gemini_file
                    source = gemini_file.parent
                else:
                    csv_files = list(source.glob("*.csv"))
                    if csv_files:
                        is_copilot_csv = True
                        conv_file = csv_files[0]
    else:
        conv_file = source
        source = source.parent

    if not conv_file.exists():
        console.print(f"[red]✗ Not found:[/red] {conv_file}")
        raise typer.Exit(1)

    # Handle Copilot CSV directory
    if is_copilot_csv:
        _convert_copilot_csv_dir(source, output, owner_id)
        return

    # Handle Gemini HTML export
    if conv_file.suffix.lower() == ".html":
        _convert_gemini_html(conv_file, source, output, owner_id)
        return

    # Detect provider (or use forced)
    raw_bytes = conv_file.read_bytes()
    raw = json.loads(raw_bytes)
    if forced_converter:
        converter = forced_converter
    else:
        try:
            converter = detect_provider(conv_file)
        except PAMError as e:
            console.print(f"[red]✗ {e}[/red]")
            raise typer.Exit(1) from e

    console.print(f"[bold]Provider:[/bold] {converter.provider_name}")

    import_meta = create_import_metadata(
        conv_file.name,
        compute_source_checksum(raw_bytes),
    )

    # Extract owner info and extra memories (provider-specific)
    old_owner_id = owner_id
    owner_id, memories = converter.extract_owner_info(source, owner_id)
    if owner_id != old_owner_id:
        console.print(f"[dim]  Owner ID:[/dim] {owner_id}")
    if memories:
        console.print(f"[dim]  Extra memories:[/dim] {len(memories)}")

    # Convert conversations
    conversations = converter.convert_conversations(
        raw, owner_id=owner_id, import_metadata=import_meta
    )

    # Post-process (feedback injection, attachment copying, etc.)
    attachment_count = converter.post_process(conversations, source, output)

    # Build and save the bundle (saves conversations + memory-store.json)
    _build_and_save_bundle(conversations, memories, output, owner_id, converter.provider_name)

    # Summary
    console.print(f"\n[green]✓[/green] Exported {len(conversations)} conversations")
    if memories:
        console.print(f"[green]✓[/green] Exported {len(memories)} memories")
    if attachment_count > 0:
        console.print(f"[green]✓[/green] Copied {attachment_count} attachment files")
    console.print(f"  Output: {output}")


def _build_and_save_bundle(
    conversations: list[Conversation],
    memories: list[Any],
    output: Path,
    owner_id: str,
    platform: str,
) -> None:
    """Save conversations and build a MemoryStore bundle with index and integrity."""
    from portable_ai_memory.core.integrity import compute_integrity_checksum
    from portable_ai_memory.core.io import save
    from portable_ai_memory.models.enums import StorageType
    from portable_ai_memory.models.memory_store import (
        ConversationIndexEntry,
        IntegrityBlock,
        MemoryStore,
        Owner,
        StorageReference,
    )
    from portable_ai_memory.models.memory_store import (
        ConversationTemporal as IndexTemporal,
    )

    # Save conversations
    for conv in conversations:
        save(conv, output / "conversations" / f"{conv.id}.json")

    # Build conversations index
    index_entries: list[ConversationIndexEntry] = []
    for conv in conversations:
        index_entries.append(
            ConversationIndexEntry(
                id=conv.id,
                platform=platform,
                title=conv.title,
                message_count=len(conv.messages),
                temporal=IndexTemporal(
                    created_at=conv.temporal.created_at,
                    updated_at=conv.temporal.updated_at,
                ),
                storage=StorageReference(
                    type=StorageType.FILE,
                    ref=f"conversations/{conv.id}.json",
                ),
            )
        )

    # Build MemoryStore with integrity
    # IMPORTANT: by_alias=True is required so field names match the saved JSON
    # (e.g. RelationObject.from_ → "from"). The validator uses by_alias=True,
    # so the checksum must be computed the same way to avoid mismatch.
    checksum = compute_integrity_checksum(
        [m.model_dump(mode="json", by_alias=True, exclude_none=True) for m in memories]
    )
    store = MemoryStore(
        schema_version="1.0",
        owner=Owner(id=owner_id),
        memories=memories,
        conversations_index=index_entries,
        integrity=IntegrityBlock(
            checksum=checksum,
            total_memories=len(memories),
        ),
    )
    save(store, output / "memory-store.json")


def _convert_gemini_html(
    html_file: Path,
    source: Path,
    output: Path,
    owner_id: str,
) -> None:
    """Handle Gemini HTML activity export."""
    from portable_ai_memory.converters.google import GoogleConverter

    converter = GoogleConverter()
    console.print(f"[bold]Provider:[/bold] {converter.provider_name}")

    raw_bytes = html_file.read_bytes()
    html_text = raw_bytes.decode("utf-8", errors="replace")

    import_meta = create_import_metadata(
        html_file.name,
        compute_source_checksum(raw_bytes),
    )

    conversations = converter.convert_conversations_from_html(
        html_text,
        owner_id=owner_id,
        import_metadata=import_meta,
        source_dir=source,
    )

    _build_and_save_bundle(conversations, [], output, owner_id, "gemini")

    console.print(f"[green]✓ Converted {len(conversations)} conversations[/green]")
    total_msgs = sum(len(c.messages) for c in conversations)
    console.print(f"[dim]  {total_msgs} messages total[/dim]")
    console.print(f"[green]✓ Output:[/green] {output}")


def _convert_copilot_csv_dir(source: Path, output: Path, owner_id: str) -> None:
    """Handle Copilot CSV directory: read all CSV files and produce PAM bundle."""
    from portable_ai_memory.converters.microsoft import MicrosoftConverter

    converter = MicrosoftConverter()
    console.print(f"[bold]Provider:[/bold] {converter.provider_name}")

    all_conversations: list[Conversation] = []

    for csv_file in sorted(source.glob("*.csv")):
        raw_bytes = csv_file.read_bytes()
        csv_text = raw_bytes.decode("utf-8-sig")

        lines = csv_text.strip().splitlines()
        if len(lines) <= 1:
            console.print(f"[dim]  Skipping {csv_file.name} (empty)[/dim]")
            continue

        import_meta = create_import_metadata(
            csv_file.name,
            compute_source_checksum(raw_bytes),
        )

        convs = converter.convert_conversations_from_csv(
            csv_text,
            owner_id=owner_id,
            import_metadata=import_meta,
            source_file=csv_file.name,
        )
        if convs:
            console.print(f"[dim]  {csv_file.name}:[/dim] {len(convs)} conversations")
        all_conversations.extend(convs)

    _build_and_save_bundle(all_conversations, [], output, owner_id, "copilot")

    console.print(f"\n[green]✓[/green] Exported {len(all_conversations)} conversations")
    console.print(f"  Output: {output}")
