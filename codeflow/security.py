"""Security and redaction helpers for runtime artifacts."""

import os
from pathlib import Path

SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED_VALUE = "<redacted>"


def _normalized_secret_names(secret_env_names):
    return {str(name).upper() for name in (secret_env_names or ())}


def looks_sensitive_env_name(name):
    upper = str(name).upper()
    return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)


def is_secret_env_name(name, secret_env_names=None):
    upper = str(name).upper()
    return upper in _normalized_secret_names(secret_env_names) or looks_sensitive_env_name(upper)


def configured_secret_env_items(env=None, secret_env_names=None):
    env = os.environ if env is None else env
    configured_names = _normalized_secret_names(secret_env_names)
    items = [
        (name, value)
        for name, value in env.items()
        if str(name).upper() in configured_names and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def detected_secret_env_items(env=None, secret_env_names=None):
    env = os.environ if env is None else env
    items = [
        (name, value)
        for name, value in env.items()
        if is_secret_env_name(name, secret_env_names=secret_env_names) and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def secret_env_summary(env=None, secret_env_names=None):
    names = [name for name, _ in configured_secret_env_items(env=env, secret_env_names=secret_env_names)]
    return {
        "secret_env_count": len(names),
        "secret_env_names": names,
    }


def detected_secret_env_summary(env=None, secret_env_names=None):
    names = [name for name, _ in detected_secret_env_items(env=env, secret_env_names=secret_env_names)]
    return {
        "secret_env_count": len(names),
        "secret_env_names": names,
    }


def redact_text(text, env=None, secret_env_names=None):
    text = str(text)
    for _, value in sorted(
        detected_secret_env_items(env=env, secret_env_names=secret_env_names),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        text = text.replace(value, REDACTED_VALUE)
    return text


def redact_artifact(value, key=None, env=None, secret_env_names=None):
    if key and is_secret_env_name(key, secret_env_names=secret_env_names):
        return REDACTED_VALUE
    if isinstance(value, dict):
        return {
            str(item_key): redact_artifact(item_value, key=item_key, env=env, secret_env_names=secret_env_names)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_artifact(item, key=key, env=env, secret_env_names=secret_env_names) for item in value]
    if isinstance(value, tuple):
        return [redact_artifact(item, key=key, env=env, secret_env_names=secret_env_names) for item in value]
    if isinstance(value, str):
        return redact_text(value, env=env, secret_env_names=secret_env_names)
    return value


def shell_env(env=None, allowlist=(), root="."):
    env = os.environ if env is None else env
    if os.name == "nt":
        # Windows 环境变量名不区分大小写，但 Python dict 区分。先统一匹配，
        # 否则默认名单里的 COMSPEC 会漏掉真实环境中的 ComSpec。
        source = {str(name).upper(): value for name, value in env.items()}
        filtered = {
            str(name).upper(): source[str(name).upper()]
            for name in allowlist
            if str(name).upper() in source
        }
        allowed = {str(name).upper() for name in allowlist}
        system_root = source.get("SYSTEMROOT") or source.get("WINDIR") or r"C:\Windows"
        if "SYSTEMROOT" in allowed:
            filtered["SystemRoot"] = system_root
            filtered.pop("SYSTEMROOT", None)
        if "WINDIR" in allowed:
            filtered["WINDIR"] = system_root
        if "COMSPEC" in allowed:
            filtered["ComSpec"] = source.get("COMSPEC") or str(
                Path(system_root) / "System32" / "cmd.exe"
            )
            filtered.pop("COMSPEC", None)
        if "PATHEXT" in allowed and "PATHEXT" not in filtered:
            filtered["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
        if "PATH" in allowed and "PATH" not in filtered:
            filtered["PATH"] = ";".join(
                [str(Path(system_root) / "System32"), system_root]
            )
    else:
        filtered = {
            name: env[name]
            for name in allowlist
            if name in env
        }
    filtered["PWD"] = str(root)
    if "PATH" not in filtered and env.get("PATH"):
        filtered["PATH"] = env["PATH"]
    return filtered
