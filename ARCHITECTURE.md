# Architecture

## Package Structure

```
src/portable_ai_memory/
  __init__.py          # Public API re-exports (models, I/O, validation, exceptions)
  exceptions.py        # PAMError hierarchy
  models/              # Pydantic v2 models (1:1 with PAM JSON Schemas)
    memory_store.py    # MemoryStore, MemoryObject, RelationObject, etc.
    conversation.py    # Conversation, Message, MessageContent, etc.
    embeddings.py      # EmbeddingsFile, EmbeddingEntry
    enums.py           # MemoryType, MessageRole, ContentType, etc.
  core/                # I/O, validation, integrity
    io.py              # load(), save(), to_dict() — auto-detects document type
    validator.py       # Deep validation (cross-refs, temporal, integrity)
    integrity.py       # SHA-256 content hashing, RFC 8785 checksums
  converters/          # Provider export converters
    base.py            # BaseConverter ABC, @register_converter, detect_provider()
    openai.py          # ChatGPT conversations.json
    anthropic.py       # Claude conversations.json + memories.json
    google.py          # Gemini Takeout JSON and HTML
    xai.py             # Grok prod-grok-backend.json
    microsoft.py       # Copilot CSV exports
  cli/                 # CLI tool (pam command)
    __init__.py        # App setup, command registration
    _validate.py       # pam validate
    _convert.py        # pam convert + bundle builder
    _inspect.py        # pam inspect
    _shared.py         # Shared console instance
  schemas/             # Bundled PAM JSON Schema files
```

## Converter System

### How It Works

Each provider converter is a subclass of `BaseConverter` registered via the `@register_converter` decorator. When `detect_provider()` is called, it tries each converter's `detect()` method until one matches.

```
@register_converter
class MyConverter(BaseConverter):
    detect()                    # Does this data match my format?
    convert_conversations()     # Raw data → list[Conversation]
    convert_memories()          # Raw data → MemoryStore (optional)
    extract_owner_info()        # Read supplementary files (optional)
    post_process()              # Inject feedback, copy attachments (optional)
```

### Writing a Custom Converter

```python
from portable_ai_memory.converters import BaseConverter, register_converter
from portable_ai_memory.models.conversation import (
    Conversation, ConversationTemporal, ImportMetadata,
    Message, MessageContent, Participant, ProviderInfo,
)
from portable_ai_memory.models.enums import ContentType, MessageRole

@register_converter
class MyProviderConverter(BaseConverter):
    @property
    def provider_name(self) -> str:
        return "my_provider"

    def detect(self, data, path=None) -> bool:
        # Return True if data matches your provider's export format.
        # Check structural keys, NOT filenames.
        return isinstance(data, dict) and "my_unique_key" in data

    def convert_conversations(self, data, *, owner_id="unknown",
                              import_metadata=None):
        # Convert raw parsed JSON into PAM Conversation objects.
        conversations = []
        for raw_conv in data["conversations"]:
            messages = [
                Message(
                    id=msg["id"],
                    role=MessageRole.USER if msg["author"] == "user"
                         else MessageRole.ASSISTANT,
                    content=MessageContent(
                        type=ContentType.TEXT, text=msg["text"],
                    ),
                    created_at=msg["timestamp"],
                )
                for msg in raw_conv["messages"]
            ]
            conversations.append(Conversation(
                schema_version="1.0",
                id=raw_conv["id"],
                provider=ProviderInfo(name="my_provider"),
                temporal=ConversationTemporal(
                    created_at=messages[0].created_at,
                ),
                participants=[
                    Participant(role=MessageRole.USER),
                    Participant(role=MessageRole.ASSISTANT),
                ],
                messages=messages,
                import_metadata=import_metadata,
            ))
        return conversations
```

After importing your module, `detect_provider()` will automatically try your converter.

### Validating Converter Output

Use `validate_converter_output()` to catch bugs in your converter:

```python
from portable_ai_memory.converters import validate_converter_output

conversations = my_converter.convert_conversations(data)
validate_converter_output(conversations)  # Raises PAMValidationError if invalid
```

## Validation Architecture

Validation happens at two levels:

1. **Schema validation** (Pydantic) — enforced on model construction. Catches type errors, missing required fields, invalid enum values, regex pattern mismatches.

2. **Deep validation** (`validator.py`) — run explicitly via `validate()`. Catches semantic issues:

| Check | What it verifies |
|-------|-----------------|
| `content-hash` | SHA-256 hash matches content |
| `integrity-total` | Memory count matches integrity block |
| `integrity-checksum` | RFC 8785 checksum matches |
| `id-unique` | No duplicate IDs across memories, relations, conversations |
| `xref-relation` | Relation from/to point to existing memories |
| `xref-conversation` | conversation_ref points to conversations_index |
| `xref-superseded` | superseded_by points to existing memory |
| `xref-derived` | derived_memories point to existing memories |
| `xref-bidirectional` | conversation_ref and derived_memories are consistent |
| `temporal` | created_at <= updated_at, valid_from <= valid_until |
| `custom-type` | type='custom' requires custom_type |
| `status` | superseded status matches superseded_by |
| `dag` | Message parent_id references exist |
| `dag-bidirectional` | Parent/children consistency |

## Integrity Model

Content integrity uses two mechanisms:

- **Content hash**: Each memory has a `content_hash` field (`sha256:<hex>`). Content is normalized (lowercase, whitespace collapse, NFC unicode) before hashing.
- **Integrity checksum**: The `MemoryStore.integrity` block contains a checksum over all memories, computed using RFC 8785 (JCS) canonical JSON serialization for deterministic ordering.
