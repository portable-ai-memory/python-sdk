"""PAM SDK exception hierarchy.

All exceptions raised by the SDK inherit from ``PAMError``,
allowing consumers to catch any PAM-related error with a single handler::

    try:
        store = pam.load("file.json")
    except PAMError as e:
        ...
"""

from __future__ import annotations


class PAMError(Exception):
    """Base exception for all PAM SDK errors."""


class PAMSchemaError(PAMError):
    """Raised when a document has an unknown or missing ``schema`` field."""


class PAMValidationError(PAMError):
    """Raised when a document fails Pydantic model validation.

    Wraps :class:`pydantic.ValidationError` so callers do not need
    to depend on Pydantic directly.

    Attributes:
        pydantic_error: The original ``pydantic.ValidationError``, if available.
    """

    def __init__(self, message: str, *, pydantic_error: Exception | None = None) -> None:
        super().__init__(message)
        self.pydantic_error = pydantic_error


class ProviderNotDetectedError(PAMError):
    """Raised when ``detect_provider`` cannot identify the export format."""
