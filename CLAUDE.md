# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python SDK for **Portable AI Memory (PAM)** — a universal interchange format for AI user memories. The SDK provides Pydantic models mapping 1:1 to PAM JSON Schemas, validation, I/O, integrity hashing, and converters from provider-specific exports (e.g., OpenAI) to PAM format.

## Commands

```bash
# Install with all deps (dev + cli)
uv sync --all-extras

# Run all tests
uv run pytest

# Run a single test file or specific test
uv run pytest tests/test_models.py
uv run pytest tests/test_models.py::test_name -v

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Run CLI
uv run pam --help
```

## Architecture

**Source layout:** `src/portable_ai_memory/` (hatchling build, wheel packages from `src/portable_ai_memory`)

- **`models/`** — Pydantic v2 models mirroring PAM JSON Schemas. Three root document types: `MemoryStore` (memories + relations), `Conversation` (messages), `EmbeddingsFile` (vectors). Enums in `enums.py`, sub-blocks (temporal, provenance, access, integrity, etc.) in `memory_store.py`.
- **`core/`** — `io.py` (load/save PAM files with auto-detection of document type), `validator.py` (schema validation returning `ValidationResult`), `integrity.py` (content hashing and checksums).
- **`converters/`** — Provider export converters. `base.py` defines `BaseConverter` ABC and a registry with `@register_converter` decorator + `detect_provider()` auto-detection. Implementations: `openai.py` (ChatGPT), `anthropic.py` (Claude), `google.py` (Gemini Takeout), `xai.py` (Grok), `microsoft.py` (Copilot CSV).
- **`schemas/`** — PAM JSON Schemas (symlinked from `vendor/portable-ai-memory` submodule; `force-include` in wheel build).
- **`cli/`** — Typer-based CLI entry point (`pam` command).

## Key Conventions

- Python 3.11+, strict mypy with Pydantic plugin
- Ruff for linting/formatting (line length 100)
- All models use `from __future__ import annotations`
- Type-checking imports guarded with `TYPE_CHECKING`
