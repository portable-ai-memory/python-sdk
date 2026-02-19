# Portable AI Memory (PAM) — Python SDK

A Python SDK for the **Portable AI Memory (PAM)** interchange format — a universal way to store, validate, and convert AI user memories across providers.

## What is PAM?

AI assistants learn about you over time — your preferences, facts about your life, project context. But these memories are locked inside each provider. If you switch from ChatGPT to Claude, or use both, your context doesn't follow you.

**PAM** solves this with an open interchange format. It defines three document types:

- **MemoryStore** — your memories (preferences, facts, context) with integrity checksums and semantic relations
- **Conversation** — full chat history with messages, tool calls, citations, and attachments
- **EmbeddingsFile** — vector embeddings linked to memories for semantic search

This SDK lets you:

1. **Convert** exports from ChatGPT, Claude, Gemini, Grok, and Copilot into PAM format
2. **Validate** PAM documents with deep integrity checks (cross-references, temporal ordering, content hashes)
3. **Build** PAM documents programmatically with type-safe Pydantic models

## Installation

```bash
pip install portable-ai-memory        # core SDK (models, I/O, validation, converters)
pip install 'portable-ai-memory[cli]' # + CLI tool: typer, rich (pam command)
pip install 'portable-ai-memory[dev]' # + dev tools: pytest, ruff, mypy
pip install 'portable-ai-memory[all]' # cli + dev combined
```

## Quick Start

### Load and validate a PAM file

```python
from portable_ai_memory import load, validate_memory_store

store = load("memory-store.json")
result = validate_memory_store(store)

if result.is_valid:
    print(f"Valid — {len(store.memories)} memories")
else:
    for issue in result.errors:
        print(issue)
```

### Convert a provider export

```python
import json
from pathlib import Path

from portable_ai_memory.converters import detect_provider
from portable_ai_memory import ProviderNotDetectedError

try:
    converter = detect_provider("conversations.json")
    data = json.loads(Path("conversations.json").read_text())
    conversations = converter.convert_conversations(
        data, owner_id="user-123",
    )
except ProviderNotDetectedError as e:
    print(f"Unknown format: {e}")
```

### Build a memory store from scratch

```python
from portable_ai_memory import MemoryStore, MemoryObject, Owner, save

# MemoryObject.create() auto-fills content_hash, temporal, provenance
store = MemoryStore(
    schema_version="1.0",
    owner=Owner(id="user-123"),
    memories=[
        MemoryObject.create(
            id="mem-001",
            type="preference",
            content="User prefers dark mode.",
            platform="my-app",
        )
    ],
)
save(store, "memory-store.json")

# Convenience lookups
mem = store.get_memory_by_id("mem-001")
prefs = store.get_memories_by_type("preference")
```

## CLI

```bash
# Validate a PAM file or bundle directory
pam validate memory-store.json
pam validate ./my-pam-bundle/

# Convert a provider export to a PAM bundle
pam convert ~/chatgpt-export/ -o ./pam-bundle/ --owner-id user-123

# Inspect a PAM file
pam inspect memory-store.json
```

## Supported Providers

| Provider | Format |
|---|---|
| OpenAI (ChatGPT) | `conversations.json` |
| Anthropic (Claude) | `conversations.json` + `memories.json` |
| Google (Gemini) | Takeout JSON or HTML |
| xAI (Grok) | `prod-grok-backend.json` |
| Microsoft (Copilot) | CSV exports |

To list registered converters programmatically:

```python
from portable_ai_memory.converters import list_converters

print(list_converters())  # ['chatgpt', 'claude', 'gemini', 'grok', 'copilot']
```

## Development

```bash
git clone --recurse-submodules git@github.com:portable-ai-memory/python-sdk.git
cd python-sdk
uv sync --all-extras
uv run pytest
```

> **Note:** The PAM JSON Schemas live in the main
> [portable-ai-memory](https://github.com/portable-ai-memory/portable-ai-memory)
> repo and are included here as a git submodule under `vendor/portable-ai-memory`.
> If you cloned without `--recurse-submodules`, run:
> `git submodule update --init --recursive`

## Links

- [PAM Specification](https://portable-ai-memory.org)
- [GitHub Repository](https://github.com/portable-ai-memory/python-sdk)

## License

Apache License 2.0
