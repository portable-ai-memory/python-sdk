# Contributing to PAM Python SDK

Thank you for your interest in contributing to the Portable AI Memory Python SDK!

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
git clone --recurse-submodules git@github.com:portable-ai-memory/python-sdk.git
cd python-sdk
uv sync --all-extras
uv run pre-commit install
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) to run checks before every commit:

| Hook | What it does |
|------|-------------|
| `ruff check` | Lint with auto-fix |
| `ruff format` | Code formatting |
| `mypy` | Static type checking |
| `pytest` | Unit tests (excludes real data tests) |

Hooks are installed with `uv run pre-commit install` (see Setup above). To run all hooks manually:

```bash
uv run pre-commit run --all-files
```

### Running Tests

```bash
uv run pytest              # unit tests (289 tests)
uv run pytest -v           # verbose output
uv run mypy src/           # type checking
uv run ruff check src/ tests/   # linting
uv run ruff format src/ tests/  # formatting
```

To run tests with real provider data (all 352 tests):

```bash
PAM_TEST_DATA_DIR=/path/to/backup-files uv run pytest
```

See `tests/conftest.py` for the expected directory structure.

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](https://github.com/portable-ai-memory/python-sdk/issues/new?template=bug_report.yml) issue template
- Include Python version, SDK version, and steps to reproduce
- If the bug involves a specific provider export, mention the provider but **do not share personal data**

### Suggesting Features

- Use the [Feature Request](https://github.com/portable-ai-memory/python-sdk/issues/new?template=feature_request.yml) issue template
- Explain the use case, not just the solution

### Submitting Pull Requests

1. **Fork** the repository and create a branch from `main`
2. **Follow existing patterns** — look at similar code in the codebase before writing
3. **Write tests** for new functionality
4. **Run the full check suite** before submitting:
   ```bash
   uv run pre-commit run --all-files
   ```
5. **Keep PRs focused** — one feature or fix per PR
6. **Write clear commit messages** describing the "why", not just the "what"

### Adding a New Provider Converter

If you want to add support for a new AI provider:

1. Create `src/portable_ai_memory/converters/<provider>.py`
2. Subclass `BaseConverter` and implement:
   - `provider_name` property
   - `detect()` — return True if this converter handles the data
   - `convert_conversations()` — convert raw data to PAM Conversations
3. Decorate with `@register_converter`
4. Optionally override `extract_owner_info()` and `post_process()`
5. Add tests in `tests/test_real_data.py` (if you have sample data)
6. Update `README.md` supported providers table

See `BaseConverter` docstring in `src/portable_ai_memory/converters/base.py` for a complete example.

## Code Style

- **Python 3.11+** with `from __future__ import annotations`
- **Strict mypy** with Pydantic plugin — all code must pass `mypy src/`
- **Ruff** for linting and formatting (line length 100)
- **Type-checking imports** guarded with `if TYPE_CHECKING:`
- **No docstring required** for private helpers, but all public API must have docstrings

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed overview of the codebase structure.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
