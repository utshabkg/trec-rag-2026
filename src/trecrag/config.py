"""Config loading and the `.env.local` secret-loading convention used across
the TREC RAG 2026 skills (Pyserini token, LLM API keys). Never logs values."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a top-level YAML mapping")
    return config


def load_dotenv_local(path: str | Path = ".env.local") -> None:
    """Populate `os.environ` from a local `KEY=VALUE` file, without overwriting
    variables already set and without ever printing file contents."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def require_env(name: str, *, hint: str = "") -> str:
    value = os.environ.get(name)
    if not value:
        suffix = f" ({hint})" if hint else ""
        raise RuntimeError(f"required environment variable {name!r} is not set{suffix}")
    return value


def get_env(name: str | None, default: str | None = None) -> str | None:
    if not name:
        return default
    return os.environ.get(name, default)
