"""Narrow context passed from runtime into tool functions."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolContext:
    root: Path
    path_resolver: Callable[[str], Path]
    shell_env_provider: Callable[[], dict]
    depth: int
    max_depth: int
    spawn_delegate: Callable[[dict], str]
    shell_backend: str = "srt"
    srt_cli_path: Path | None = None
    srt_settings_path: Path | None = None
    read_paths: tuple[Path, ...] | None = None
    write_paths: tuple[Path, ...] | None = None
    denied_capabilities: tuple[str, ...] = ()

    def path(self, raw_path):
        return self.path_resolver(str(raw_path))

    def require_path_scope(self, path, capability):
        allowed = self.read_paths if capability == "filesystem.read" else self.write_paths
        if allowed is None:
            return
        resolved = Path(path).resolve()
        for scope in allowed:
            scope = Path(scope).resolve()
            if resolved == scope or scope in resolved.parents:
                return
        raise ValueError(f"path is outside configured {capability} scopes")

    def shell_env(self):
        return self.shell_env_provider()

    def require_capability(self, capability):
        if capability in self.denied_capabilities:
            raise PermissionError(f"capability '{capability}' is disabled in this run")
