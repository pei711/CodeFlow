"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import os
import shutil
import signal
import subprocess
import textwrap
import threading
import time
from functools import partial
from pathlib import Path

from .workspace import IGNORED_PATH_NAMES

BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "capability": "filesystem.read",
        "risky": False,
        "description": "List files in the workspace.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "capability": "filesystem.read",
        "risky": False,
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "capability": "filesystem.read",
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback.",
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "capability": "process.execute",
        "risky": True,
        "description": "Run a shell command in the repo root.",
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "capability": "filesystem.write",
        "risky": True,
        "description": "Write a text file.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "capability": "filesystem.write",
        "risky": True,
        "description": "Replace one exact text block in a file.",
    },
}

DELEGATE_TOOL_SPEC = {
    "schema": {"task": "str", "max_steps": "int=3"},
    "capability": "agent.delegate",
    "risky": False,
    "description": "Ask a bounded read-only child agent to investigate.",
}

TOOL_CAPABILITIES = {
    "filesystem.read",
    "filesystem.write",
    "process.execute",
    "agent.delegate",
}
MAX_PROCESS_CAPTURE_BYTES = 128 * 1024


def legal_tool_names():
    return set(BASE_TOOL_SPECS) | {"delegate"}

TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
}


def build_tool_registry(context):
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], context)}
        for name, spec in BASE_TOOL_SPECS.items()
    }
    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，
    # 就连 delegate 这个工具都不再暴露给模型。
    if context.depth < context.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPEC, "run": partial(tool_delegate, context)}
    return tools


def tool_example(name):
    return TOOL_EXAMPLES.get(name, "")


def validate_tool(context, name, args):
    args = args or {}
    capability = DELEGATE_TOOL_SPEC["capability"] if name == "delegate" else BASE_TOOL_SPECS.get(name, {}).get("capability")
    if capability:
        context.require_capability(capability)

    for key, maximum in (("path", 4096), ("command", 20000), ("pattern", 2000), ("content", 1_000_000), ("old_text", 100_000), ("new_text", 100_000), ("task", 4000)):
        if key in args and len(str(args[key])) > maximum:
            raise ValueError(f"{key} exceeds the {maximum} character limit")

    if name == "list_files":
        path = context.path(args.get("path", "."))
        context.require_path_scope(path, "filesystem.read")
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return

    if name == "read_file":
        path = context.path(args["path"])
        context.require_path_scope(path, "filesystem.read")
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start or end - start + 1 > 2000:
            raise ValueError("line range must contain between 1 and 2000 lines")
        return

    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        context.path(args.get("path", "."))
        context.require_path_scope(context.path(args.get("path", ".")), "filesystem.read")
        return

    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        return

    if name == "write_file":
        path = context.path(args["path"])
        context.require_path_scope(path, "filesystem.write")
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        return

    if name == "patch_file":
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
        path = context.path(args["path"])
        context.require_path_scope(path, "filesystem.write")
        if not path.is_file():
            raise ValueError("path is not a file")
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError("patch_file does not accept files larger than 5 MiB")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        return

    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        max_steps = int(args.get("max_steps", 3))
        if max_steps < 1 or max_steps > 3:
            raise ValueError("delegate max_steps must be in [1, 3]")
        if context.depth >= context.max_depth:
            raise ValueError("delegate depth exceeded")
        return


def tool_list_files(context, args):
    _guard_tool(context, "list_files", args)
    path = context.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(context.root)}")
    return "\n".join(lines) or "(empty)"


def tool_read_file(context, args):
    _guard_tool(context, "read_file", args)
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            if number > end:
                break
            if number >= start:
                lines.append(f"{number:>4}: {line.rstrip()}")
    body = "\n".join(lines)
    return f"# {path.relative_to(context.root)}\n{body}"


def tool_search(context, args):
    _guard_tool(context, "search", args)
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = context.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = _run_bounded_process(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=context.root,
            timeout=10,
        )
        return result["stdout"].strip() or result["stderr"].strip() or "(no matches)"

    matches = []
    deadline = time.monotonic() + 10
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(context.root).parts)
    ]
    for file_path in files:
        with file_path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("search", 10)
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(context.root)}:{number}:{line.rstrip()}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_run_shell(context, args):
    _guard_tool(context, "run_shell", args)
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    env = context.shell_env()
    if context.shell_backend == "srt":
        srt_cli_path = Path(context.srt_cli_path or "")
        settings_path = Path(context.srt_settings_path or "")
        node_path = shutil.which("node", path=env.get("PATH"))
        if not node_path:
            raise RuntimeError("SRT sandbox unavailable: Node.js was not found on PATH")
        if not srt_cli_path.is_file():
            raise RuntimeError(
                "SRT sandbox unavailable: CLI not found; run "
                "`npm install --save-exact @anthropic-ai/sandbox-runtime`"
            )
        if not settings_path.is_file():
            raise RuntimeError(f"SRT sandbox unavailable: settings file not found: {settings_path}")
        # 外层不用 host shell 解析模型给出的 command。SRT 会先建立系统级边界，
        # 再把 command 交给沙箱内部的 shell，避免命令在隔离生效前执行。
        invocation = [
            node_path,
            str(srt_cli_path),
            "--settings",
            str(settings_path),
            "-c",
            command,
        ]
        result = _run_bounded_process(
            invocation,
            cwd=context.root,
            shell=False,
            timeout=timeout,
            env=env,
        )
    elif context.shell_backend == "direct":
        # 仅供显式测试/调试。生产 CLI 默认不会走这个分支。
        executable = env.get("ComSpec") if os.name == "nt" else None
        result = _run_bounded_process(
            command,
            cwd=context.root,
            shell=True,
            executable=executable,
            timeout=timeout,
            env=env,
        )
    else:
        raise RuntimeError(f"unknown shell backend: {context.shell_backend}")
    return textwrap.dedent(
        f"""\
        sandbox: {context.shell_backend}
        exit_code: {result['returncode']}
        stdout:
        {result['stdout'].strip() or "(empty)"}{result['stdout_suffix']}
        stderr:
        {result['stderr'].strip() or "(empty)"}{result['stderr_suffix']}
        """
    ).strip()


def _run_bounded_process(command, *, cwd, timeout, env=None, shell=False, executable=None):
    """Drain child output without retaining unbounded stdout/stderr in memory."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=shell,
        executable=executable,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}

    def drain(name, stream):
        while True:
            chunk = stream.read1(8192)
            if not chunk:
                return
            remaining = MAX_PROCESS_CAPTURE_BYTES - len(captured[name])
            if remaining > 0:
                captured[name].extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[name] = True

    readers = [
        threading.Thread(target=drain, args=(name, getattr(process, name)), daemon=True)
        for name in ("stdout", "stderr")
    ]
    for reader in readers:
        reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        for reader in readers:
            reader.join()
        raise
    for reader in readers:
        reader.join()

    return {
        "returncode": returncode,
        "stdout": captured["stdout"].decode("utf-8", errors="replace"),
        "stderr": captured["stderr"].decode("utf-8", errors="replace"),
        "stdout_suffix": "\n...[stdout capture limit reached]" if truncated["stdout"] else "",
        "stderr_suffix": "\n...[stderr capture limit reached]" if truncated["stderr"] else "",
    }


def tool_write_file(context, args):
    _guard_tool(context, "write_file", args)
    path = context.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(context.root)} ({len(content)} chars)"


def tool_patch_file(context, args):
    _guard_tool(context, "patch_file", args)
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    return f"patched {path.relative_to(context.root)}"


def tool_delegate(context, args):
    _guard_tool(context, "delegate", args)
    if context.depth >= context.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")
    return context.spawn_delegate(args)


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}


def _guard_tool(context, name, args):
    validate_tool(context, name, args)
