"""PAM CLI — validate, convert, and inspect PAM files.

Usage:
    pam validate <file>       Validate a PAM file (schema + deep checks)
    pam convert <source> -o <out>  Convert a provider export to a complete PAM bundle
    pam inspect <file>        Show summary of a PAM file
"""

from __future__ import annotations

try:
    import typer
except ImportError as e:
    raise SystemExit(
        "CLI dependencies not installed. Run: pip install 'portable-ai-memory[cli]'"
    ) from e

from portable_ai_memory.cli._convert import convert
from portable_ai_memory.cli._inspect import inspect
from portable_ai_memory.cli._validate import validate

app = typer.Typer(
    name="pam",
    help="Portable AI Memory — CLI tools for the PAM interchange format.",
    no_args_is_help=True,
)
app.command()(validate)
app.command()(convert)
app.command()(inspect)


def cli() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    cli()
