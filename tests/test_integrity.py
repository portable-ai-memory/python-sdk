"""Tests for integrity utilities — RFC 8785, content hash, integrity checksum."""

from __future__ import annotations

from portable_ai_memory.core.integrity import (
    canonicalize_jcs,
    compute_content_hash,
    compute_integrity_checksum,
    normalize_content,
    verify_content_hash,
    verify_integrity,
)


class TestNormalization:
    def test_basic_normalization(self) -> None:
        assert normalize_content("  Hello  World  ") == "hello world"

    def test_unicode_nfc(self) -> None:
        decomposed = "caf\u0065\u0301"
        composed = "caf\u00e9"
        assert normalize_content(decomposed) == normalize_content(composed)

    def test_collapse_whitespace(self) -> None:
        assert normalize_content("a\t\nb") == "a b"


class TestContentHash:
    def test_deterministic(self) -> None:
        h1 = compute_content_hash("Hello World")
        h2 = compute_content_hash("Hello World")
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_verify(self) -> None:
        content = "Some content"
        h = compute_content_hash(content)
        assert verify_content_hash(content, h)
        assert not verify_content_hash("Different", h)


class TestRFC8785Canonicalization:
    def test_sorted_keys(self) -> None:
        result = canonicalize_jcs({"b": 1, "a": 2})
        assert result == b'{"a":2,"b":1}'

    def test_float_integer_distinction(self) -> None:
        """RFC 8785 represents 1.0 as 1 (no trailing .0)."""
        result = canonicalize_jcs({"v": 1.0})
        assert result == b'{"v":1}'

    def test_nested_objects(self) -> None:
        result = canonicalize_jcs({"a": {"c": 1, "b": 2}})
        assert result == b'{"a":{"b":2,"c":1}}'

    def test_array_order_preserved(self) -> None:
        result = canonicalize_jcs([3, 1, 2])
        assert result == b"[3,1,2]"


class TestIntegrityChecksum:
    def test_deterministic(self) -> None:
        memories = [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
        c1 = compute_integrity_checksum(memories)
        c2 = compute_integrity_checksum(memories)
        assert c1 == c2
        assert c1.startswith("sha256:")

    def test_order_independent(self) -> None:
        m1 = [{"id": "b"}, {"id": "a"}]
        m2 = [{"id": "a"}, {"id": "b"}]
        assert compute_integrity_checksum(m1) == compute_integrity_checksum(m2)

    def test_verify(self) -> None:
        memories = [{"id": "a", "content": "x"}]
        checksum = compute_integrity_checksum(memories)
        assert verify_integrity(memories, checksum)
        assert not verify_integrity([{"id": "b"}], checksum)
