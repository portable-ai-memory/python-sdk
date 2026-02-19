"""Tests for pam.schemas module."""

from __future__ import annotations

import pytest

from portable_ai_memory.schemas import load_schema


class TestLoadSchema:
    def test_load_memory_store(self) -> None:
        schema = load_schema("memory-store")
        assert isinstance(schema, dict)
        assert "$schema" in schema or "type" in schema

    def test_load_conversation(self) -> None:
        schema = load_schema("conversation")
        assert isinstance(schema, dict)

    def test_load_embeddings(self) -> None:
        schema = load_schema("embeddings")
        assert isinstance(schema, dict)

    def test_unknown_schema_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schema: 'bogus'"):
            load_schema("bogus")
