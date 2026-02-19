"""Inspect command for PAM CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from portable_ai_memory.cli._shared import console
from portable_ai_memory.core.io import load
from portable_ai_memory.models.conversation import Conversation
from portable_ai_memory.models.memory_store import MemoryStore


def inspect(
    file: Path = typer.Argument(..., help="Path to a PAM JSON file", exists=True),
) -> None:
    """Show a summary of a PAM file."""
    doc = load(file)

    if isinstance(doc, MemoryStore):
        _inspect_memory_store(doc)
    elif isinstance(doc, Conversation):
        _inspect_conversation(doc)
    else:
        console.print(f"Document type: {type(doc).__name__}")


def _inspect_memory_store(store: MemoryStore) -> None:
    console.print("\n[bold]PAM Memory Store[/bold]")
    console.print(f"  Version:  {store.schema_version}")
    console.print(f"  Owner:    {store.owner.id}")
    console.print(f"  Memories: {len(store.memories)}")
    console.print(f"  Relations: {len(store.relations)}")
    console.print(f"  Conversations: {len(store.conversations_index)}")

    if store.memories:
        table = Table(title="\nMemories by Type")
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Active", justify="right", style="green")

        from collections import Counter

        type_counts = Counter(m.type for m in store.memories)
        active_counts = Counter(m.type for m in store.memories if m.status == "active")

        for mem_type, count in type_counts.most_common():
            table.add_row(mem_type, str(count), str(active_counts.get(mem_type, 0)))

        console.print(table)

    if store.integrity:
        console.print(f"\n  Integrity: checksum={store.integrity.checksum[:30]}...")


def _inspect_conversation(conv: Conversation) -> None:
    console.print("\n[bold]PAM Conversation[/bold]")
    console.print(f"  ID:       {conv.id}")
    console.print(f"  Provider: {conv.provider.name}")
    console.print(f"  Title:    {conv.title or '(none)'}")
    console.print(f"  Messages: {len(conv.messages)}")
    console.print(f"  Model:    {conv.model or '(none)'}")

    if conv.participants:
        roles = ", ".join(f"{p.role}:{p.name or '?'}" for p in conv.participants)
        console.print(f"  Participants: {roles}")
