# Changelog

## [1.0.1](https://github.com/portable-ai-memory/python-sdk/compare/portable-ai-memory-v1.0.0...portable-ai-memory-v1.0.1) (2026-02-19)


### CI

* trigger initial CI run ([d821ced](https://github.com/portable-ai-memory/python-sdk/commit/d821cede1eb478f7b959a5d031c0d37af55e8d8a))

## [1.0.0] - 2026-02-19

First stable release.

### Features

- Pydantic v2 models for all PAM document types (MemoryStore, Conversation, EmbeddingsFile)
- Full PAM 1.0 spec compliance with deep validation (cross-refs, integrity, temporal)
- Provider converters: ChatGPT, Claude, Gemini (JSON + HTML), Grok, Copilot (CSV)
- Auto-detection of provider export format via `detect_provider()`
- CLI tool (`pam`) with validate, convert, and inspect commands
- `--provider` flag for forced provider detection in CLI
- `MemoryObject.create()` factory for convenient memory construction
- `MemoryStore` convenience methods (get_memory_by_id, get_memories_by_type, get_memories_with_tag)
- Unified `validate()` dispatcher for any PAM document type
- `validate_converter_output()` helper for catching converter bugs
- Extensible converter registry via `@register_converter` decorator
- PAMError exception hierarchy (PAMSchemaError, PAMValidationError, ProviderNotDetectedError)
- 281 tests, strict mypy, full ruff compliance
