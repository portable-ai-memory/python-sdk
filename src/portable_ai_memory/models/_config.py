"""Shared Pydantic model configuration."""

from __future__ import annotations

from pydantic import ConfigDict

STRICT = ConfigDict(extra="forbid", populate_by_name=True)
