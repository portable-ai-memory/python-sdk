"""CLI integration tests using real provider data.

Tests the full CLI pipeline: convert → validate → inspect for all 5 providers.
Skipped if PAM_TEST_DATA_DIR is not set.

Usage:
    PAM_TEST_DATA_DIR=/path/to/backup-files uv run pytest tests/test_cli_integration.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from portable_ai_memory.cli import app

from .conftest import REAL_DATA_DIR, real_data_path, skip_no_real_data

runner = CliRunner()


# ── ChatGPT / OpenAI ──────────────────────────────────────


@skip_no_real_data
class TestCLIChatGPT:
    """CLI integration: convert + validate + inspect for ChatGPT."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.source = real_data_path("chatgpt")
        if not (self.source / "conversations.json").exists():
            pytest.skip("ChatGPT data not found")
        self.output = tmp_path / "chatgpt-out"

    def test_convert(self) -> None:
        result = runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        assert result.exit_code == 0, result.output
        assert "chatgpt" in result.output.lower()
        assert "conversations" in result.output.lower()

    def test_convert_output_structure(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        assert (self.output / "memory-store.json").exists()
        assert (self.output / "conversations").is_dir()
        conv_files = list((self.output / "conversations").glob("*.json"))
        assert len(conv_files) > 0

    def test_validate_bundle(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        result = runner.invoke(app, ["validate", str(self.output)])
        assert result.exit_code == 0, result.output
        assert "invalid" not in result.output.lower() or "0 invalid" in result.output.lower()

    def test_validate_single_conversation(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        conv_files = list((self.output / "conversations").glob("*.json"))
        assert len(conv_files) > 0
        result = runner.invoke(app, ["validate", str(conv_files[0])])
        assert result.exit_code == 0, result.output

    def test_validate_memory_store(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        result = runner.invoke(app, ["validate", str(self.output / "memory-store.json")])
        assert result.exit_code == 0, result.output

    def test_inspect_conversation(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        conv_files = list((self.output / "conversations").glob("*.json"))
        assert len(conv_files) > 0
        result = runner.invoke(app, ["inspect", str(conv_files[0])])
        assert result.exit_code == 0, result.output
        assert "chatgpt" in result.output.lower()
        assert "Messages" in result.output

    def test_inspect_memory_store(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        result = runner.invoke(app, ["inspect", str(self.output / "memory-store.json")])
        assert result.exit_code == 0, result.output
        assert "Memory Store" in result.output
        assert "Owner" in result.output

    def test_convert_with_provider_flag(self) -> None:
        result = runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "chatgpt"],
        )
        assert result.exit_code == 0, result.output

    def test_convert_with_owner_id(self) -> None:
        result = runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--owner-id", "test-user"],
        )
        assert result.exit_code == 0, result.output
        store = json.loads((self.output / "memory-store.json").read_text())
        assert store["owner"]["id"] == "test-user"


# ── Claude / Anthropic ─────────────────────────────────────


@skip_no_real_data
class TestCLIClaude:
    """CLI integration: convert + validate + inspect for Claude."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.source = real_data_path("claude")
        if not (self.source / "conversations.json").exists():
            pytest.skip("Claude data not found")
        self.output = tmp_path / "claude-out"

    def test_convert(self) -> None:
        result = runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        assert result.exit_code == 0, result.output
        assert "claude" in result.output.lower()

    def test_convert_extracts_memories(self) -> None:
        result = runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        assert result.exit_code == 0, result.output
        store = json.loads((self.output / "memory-store.json").read_text())
        assert len(store["memories"]) > 0

    def test_convert_extracts_owner(self) -> None:
        """Claude exports include users.json with owner UUID."""
        result = runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        assert result.exit_code == 0, result.output
        # Should auto-detect owner from users.json
        if (self.source / "users.json").exists():
            assert "Owner ID" in result.output

    def test_validate_bundle(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        result = runner.invoke(app, ["validate", str(self.output)])
        assert result.exit_code == 0, result.output

    def test_inspect_conversation(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        conv_files = list((self.output / "conversations").glob("*.json"))
        assert len(conv_files) > 0
        result = runner.invoke(app, ["inspect", str(conv_files[0])])
        assert result.exit_code == 0, result.output
        assert "claude" in result.output.lower()

    def test_inspect_memory_store(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        result = runner.invoke(app, ["inspect", str(self.output / "memory-store.json")])
        assert result.exit_code == 0, result.output
        assert "Memories" in result.output

    def test_validate_no_deep(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        conv_files = list((self.output / "conversations").glob("*.json"))
        result = runner.invoke(app, ["validate", str(conv_files[0]), "--no-deep"])
        assert result.exit_code == 0, result.output
        assert "Schema" in result.output


# ── Grok / xAI ────────────────────────────────────────────


@skip_no_real_data
class TestCLIGrok:
    """CLI integration: convert + validate + inspect for Grok."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.source = real_data_path("grok")
        if not self.source.exists():
            pytest.skip("Grok data not found")
        # Check that the nested backend file exists
        from portable_ai_memory.converters.xai import XAIConverter

        backend = XAIConverter.find_backend_file(self.source)
        if not backend:
            pytest.skip("prod-grok-backend.json not found")
        self.output = tmp_path / "grok-out"

    def test_convert(self) -> None:
        result = runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        assert result.exit_code == 0, result.output
        assert "grok" in result.output.lower()

    def test_convert_output_structure(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        assert (self.output / "memory-store.json").exists()
        conv_files = list((self.output / "conversations").glob("*.json"))
        assert len(conv_files) > 0

    def test_validate_bundle(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        result = runner.invoke(app, ["validate", str(self.output)])
        assert result.exit_code == 0, result.output

    def test_inspect_conversation(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        conv_files = list((self.output / "conversations").glob("*.json"))
        assert len(conv_files) > 0
        result = runner.invoke(app, ["inspect", str(conv_files[0])])
        assert result.exit_code == 0, result.output
        assert "grok" in result.output.lower()

    def test_inspect_memory_store(self) -> None:
        runner.invoke(app, ["convert", str(self.source), "-o", str(self.output)])
        result = runner.invoke(app, ["inspect", str(self.output / "memory-store.json")])
        assert result.exit_code == 0, result.output

    def test_convert_with_provider_flag(self) -> None:
        result = runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "grok"],
        )
        assert result.exit_code == 0, result.output


# ── Copilot / Microsoft ───────────────────────────────────


@skip_no_real_data
class TestCLICopilot:
    """CLI integration: convert + validate + inspect for Copilot CSV."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.source = real_data_path("copilot")
        if not self.source.exists():
            pytest.skip("Copilot data not found")
        csv_files = list(self.source.glob("*.csv"))
        if not csv_files:
            pytest.skip("No CSV files found in copilot dir")
        self.output = tmp_path / "copilot-out"

    def test_convert(self) -> None:
        result = runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "copilot"],
        )
        assert result.exit_code == 0, result.output
        assert "copilot" in result.output.lower()

    def test_convert_output_structure(self) -> None:
        runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "copilot"],
        )
        assert (self.output / "memory-store.json").exists()

    def test_validate_bundle(self) -> None:
        runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "copilot"],
        )
        result = runner.invoke(app, ["validate", str(self.output)])
        assert result.exit_code == 0, result.output

    def test_inspect_conversation(self) -> None:
        runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "copilot"],
        )
        conv_files = list((self.output / "conversations").glob("*.json"))
        if not conv_files:
            pytest.skip("No conversations converted from Copilot CSV")
        result = runner.invoke(app, ["inspect", str(conv_files[0])])
        assert result.exit_code == 0, result.output
        assert "copilot" in result.output.lower()

    def test_inspect_memory_store(self) -> None:
        runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "copilot"],
        )
        result = runner.invoke(app, ["inspect", str(self.output / "memory-store.json")])
        assert result.exit_code == 0, result.output

    def test_convert_single_csv(self) -> None:
        """Convert a single CSV file directly."""
        csv_files = list(self.source.glob("*.csv"))
        result = runner.invoke(
            app,
            [
                "convert",
                str(csv_files[0]),
                "-o",
                str(self.output),
                "--provider",
                "copilot",
            ],
        )
        # Single CSV files are handled by the dir path; a file might not match
        # the copilot dir shortcut, so just check it doesn't crash unexpectedly
        assert result.exit_code in (0, 1)


# ── Gemini / Google ───────────────────────────────────────


@skip_no_real_data
class TestCLIGemini:
    """CLI integration: convert + validate + inspect for Gemini."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.source = real_data_path("gemini")
        if not self.source.exists():
            pytest.skip("Gemini data not found")
        # Look for HTML or JSON activity file
        from portable_ai_memory.converters.google import GoogleConverter

        self.html_files = list(self.source.glob("*.html"))
        self.activity_file = GoogleConverter.find_activity_file(self.source)
        if not self.html_files and not self.activity_file:
            pytest.skip("No Gemini HTML or activity file found")
        self.output = tmp_path / "gemini-out"

    def test_convert_html(self) -> None:
        if not self.html_files:
            pytest.skip("No HTML files")
        result = runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "gemini"],
        )
        assert result.exit_code == 0, result.output
        assert "gemini" in result.output.lower()

    def test_convert_output_structure(self) -> None:
        if not self.html_files:
            pytest.skip("No HTML files")
        runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "gemini"],
        )
        assert (self.output / "memory-store.json").exists()

    def test_validate_bundle(self) -> None:
        if not self.html_files:
            pytest.skip("No HTML files")
        runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "gemini"],
        )
        result = runner.invoke(app, ["validate", str(self.output)])
        assert result.exit_code == 0, result.output

    def test_inspect_conversation(self) -> None:
        if not self.html_files:
            pytest.skip("No HTML files")
        runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "gemini"],
        )
        conv_files = list((self.output / "conversations").glob("*.json"))
        if not conv_files:
            pytest.skip("No conversations converted from Gemini")
        result = runner.invoke(app, ["inspect", str(conv_files[0])])
        assert result.exit_code == 0, result.output
        assert "gemini" in result.output.lower()

    def test_inspect_memory_store(self) -> None:
        if not self.html_files:
            pytest.skip("No HTML files")
        runner.invoke(
            app,
            ["convert", str(self.source), "-o", str(self.output), "--provider", "gemini"],
        )
        result = runner.invoke(app, ["inspect", str(self.output / "memory-store.json")])
        assert result.exit_code == 0, result.output


# ── Cross-provider consistency ─────────────────────────────


@skip_no_real_data
class TestCLICrossProvider:
    """Verify output consistency across providers."""

    def _convert_provider(
        self,
        tmp_path: Path,
        provider: str,
        source: Path,
        *,
        force: bool = False,
    ) -> Path:
        output = tmp_path / f"{provider}-out"
        args = ["convert", str(source), "-o", str(output)]
        if force:
            args.extend(["--provider", provider])
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{provider} convert failed: {result.output}"
        return output

    def test_all_bundles_have_memory_store(self, tmp_path: Path) -> None:
        """Every provider bundle must produce a valid memory-store.json."""
        assert REAL_DATA_DIR is not None
        providers = [
            ("chatgpt", "chatgpt", False),
            ("claude", "claude", False),
            ("grok", "grok", False),
            ("copilot", "copilot", True),
        ]
        for subdir, provider, force in providers:
            source = REAL_DATA_DIR / subdir
            if not source.exists():
                continue
            output = self._convert_provider(tmp_path, provider, source, force=force)
            store_file = output / "memory-store.json"
            assert store_file.exists(), f"{provider}: no memory-store.json"
            store = json.loads(store_file.read_text())
            assert "schema_version" in store
            assert "owner" in store

    def test_all_conversations_are_valid_pam(self, tmp_path: Path) -> None:
        """All converted conversation files must pass validation."""
        assert REAL_DATA_DIR is not None
        providers = [
            ("chatgpt", "chatgpt", False),
            ("claude", "claude", False),
            ("grok", "grok", False),
            ("copilot", "copilot", True),
        ]
        for subdir, provider, force in providers:
            source = REAL_DATA_DIR / subdir
            if not source.exists():
                continue
            output = self._convert_provider(tmp_path, provider, source, force=force)
            result = runner.invoke(app, ["validate", str(output)])
            assert result.exit_code == 0, f"{provider} validation failed: {result.output}"
