"""Validate command for PAM CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from portable_ai_memory.cli._shared import console
from portable_ai_memory.core.io import load_dict
from portable_ai_memory.core.validator import validate_conversation, validate_memory_store
from portable_ai_memory.exceptions import PAMError
from portable_ai_memory.models.conversation import Conversation
from portable_ai_memory.models.memory_store import MemoryStore


def validate(
    path: Path = typer.Argument(
        ...,
        help="Path to a PAM JSON file or bundle directory",
        exists=True,
    ),
    deep: bool = typer.Option(
        True,
        help="Run deep validation (cross-refs, integrity, temporal)",
    ),
) -> None:
    """Validate a PAM file or bundle directory against schemas and run integrity checks."""
    if path.is_dir():
        _validate_directory(path, deep=deep)
    else:
        if not _validate_file(path, deep=deep):
            raise typer.Exit(1)


def _validate_directory(directory: Path, *, deep: bool) -> None:
    """Validate all PAM files in a bundle directory."""
    errors = 0
    valid = 0
    files: list[Path] = []

    store_file = directory / "memory-store.json"
    if store_file.exists():
        files.append(store_file)

    conv_dir = directory / "conversations"
    if conv_dir.is_dir():
        files.extend(sorted(conv_dir.glob("*.json")))

    if not files:
        console.print(f"[red]✗ No PAM files found in {directory}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Validating bundle:[/bold] {directory} ({len(files)} files)\n")

    for f in files:
        ok = _validate_file(f, deep=deep, quiet=True)
        if ok:
            valid += 1
        else:
            errors += 1

    console.print(f"\n[bold]Results:[/bold] {valid} valid, {errors} invalid, {len(files)} total")
    if errors:
        raise typer.Exit(1)


def _validate_file(file: Path, *, deep: bool, quiet: bool = False) -> bool:
    """Validate a single PAM file. Returns True if valid."""
    import json

    from pydantic import ValidationError

    label = file.name
    if not quiet:
        console.print(f"\n[bold]Validating:[/bold] {file}\n")

    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        msg = (
            f"[red]✗ Invalid JSON:[/red] {e}"
            if not quiet
            else f"[red]✗ {label}:[/red] Invalid JSON"
        )
        console.print(msg)
        return False

    try:
        doc = load_dict(raw)
        if not quiet:
            console.print("[green]✓[/green] Schema validation passed")
    except (PAMError, ValidationError) as e:
        msg = (
            f"[red]✗ Schema validation failed:[/red]\n{e}"
            if not quiet
            else f"[red]✗ {label}:[/red] Schema failed"
        )
        console.print(msg)
        return False

    if deep:
        if isinstance(doc, MemoryStore):
            result = validate_memory_store(doc)
        elif isinstance(doc, Conversation):
            result = validate_conversation(doc)
        else:
            if not quiet:
                console.print("[green]✓[/green] No deep checks for this document type")
            return True

        if not quiet:
            for issue in result.warnings:
                console.print(f"[yellow]⚠ {issue}[/yellow]")
            for issue in result.errors:
                console.print(f"[red]✗ {issue}[/red]")

        if result.is_valid:
            if not quiet:
                checks = len(result.issues)
                warns = len(result.warnings)
                console.print(f"\n[green]✓ Valid[/green] ({checks} checks, {warns} warnings)")
            elif result.warnings:
                console.print(f"[yellow]⚠ {label}:[/yellow] {len(result.warnings)} warning(s)")
            return True
        else:
            if quiet:
                console.print(f"[red]✗ {label}:[/red] {len(result.errors)} error(s)")
                for issue in result.errors:
                    console.print(f"  [red]{issue}[/red]")
            else:
                errs = len(result.errors)
                warns = len(result.warnings)
                console.print(f"\n[red]✗ {errs} error(s), {warns} warning(s)[/red]")
            return False
    else:
        if not quiet:
            console.print("[green]✓[/green] Schema OK")
        return True
