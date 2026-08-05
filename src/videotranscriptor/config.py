"""Minimal ``.env`` support.

A single API key does not justify a dependency on python-dotenv, and keeping
the parser here means the loading rules are visible: existing environment
variables always win, so `DEEPGRAM_API_KEY=... videotranscriptor ...` overrides
the file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

ENV_FILENAME = ".env"


def parse_env(text: str) -> Dict[str, str]:
    """Parse ``KEY=value`` lines, tolerating ``export``, quotes and comments."""
    values: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # An unquoted trailing comment is not part of the value.
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """Look for a ``.env`` in ``start`` and each parent directory."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_env(path: Optional[Path] = None, override: bool = False) -> Dict[str, str]:
    """Load a ``.env`` into ``os.environ`` and return what it contained."""
    env_path = path or find_env_file()
    if env_path is None or not env_path.is_file():
        return {}
    values = parse_env(env_path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values
