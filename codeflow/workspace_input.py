"""Parse workspace paths pasted or dragged into the CodeFlow prompt."""

import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def _unquote_path(raw_path):
    value = str(raw_path or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    if value.lower().startswith("file://"):
        parsed = urlparse(value)
        value = unquote(parsed.path)
        if parsed.netloc:
            value = f"\\\\{parsed.netloc}{value}"
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    return os.path.expandvars(os.path.expanduser(value))


def resolve_workspace_path(raw_path):
    """Return a usable workspace directory from a dropped/pasted path.

    A dropped file is mapped to its parent directory so users can drag either
    a repository folder or one file from that repository. Natural-language
    messages return ``None`` because only existing filesystem paths qualify.
    """

    value = _unquote_path(raw_path)
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.is_file():
        return candidate.parent.resolve()
    return None
