"""命令行入口。

这个模块负责把“用户怎么启动 CodeFlow”翻译成 runtime 能理解的对象：
解析参数、挑模型后端、构建工作区快照、恢复或新建 session，
最后进入 one-shot 或交互式循环。
"""

import argparse
import os
import shutil
import sys
import textwrap

from . import tools as toolkit
from .config import load_project_env, provider_env
from .model_switch import ModelProfile
from .providers.clients import (
    AnthropicCompatibleModelClient,
    OllamaModelClient,
    OpenAICompatibleModelClient,
)
from .runtime import CodeFlow, SessionStore
from .terminal_ui import thinking_indicator
from .workspace import WorkspaceContext, middle, workspace_state_path
from .workspace_input import resolve_workspace_path

DEFAULT_SECRET_ENV_NAMES = (
    "CODEFLOW_OPENAI_API_KEY",
    "PICO_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "CODEFLOW_ANTHROPIC_API_KEY",
    "PICO_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CODEFLOW_DEEPSEEK_API_KEY",
    "PICO_DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEY",
    "CODEFLOW_RIGHT_CODES_API_KEY",
    "PICO_RIGHT_CODES_API_KEY",
    "RIGHT_CODES_API_KEY",
    "GITHUB_PAT",
    "GH_PAT",
)

WELCOME_ART = (
    "        /\\___/\\\\",
    "       (  o o  )",
    "       /   ^   \\\\",
    "      /|       |\\\\",
)
WELCOME_NAME = "CodeFlow"
WELCOME_SUBTITLE = "local coding agent"
WELCOME_STATUS = "calm shell, ready for work"
HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
             You can also say “切换到 flash”, “切换到 deepseek-v4.1-flash” or “换成 pro”.
    /memory  Show the agent's distilled working memory.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /workspace <path>
             Switch workspace. You can also drag a folder into this prompt.
    /exit    Exit the agent.
    """
).strip()


DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OPENAI_MODEL = "deepseek-v4-flash"
DEFAULT_OPENAI_BASE_URL = "https://api.deepseek.com"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_BASE_URL = "https://www.right.codes/claude/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_PROVIDER = "deepseek"
PROVIDER_CHOICES = ("ollama", "openai", "anthropic", "deepseek")
SECRET_ENV_NAMES_VAR = "CODEFLOW_SECRET_ENV_NAMES"
LEGACY_SECRET_ENV_NAMES_VAR = "PICO_SECRET_ENV_NAMES"


def _effective_provider(args):
    # Provider 选择优先级：
    # 1. 用户显式传入 --provider
    # 2. 项目 .env / shell 里的 PICO_PROVIDER
    # 3. 代码里的默认 provider
    provider = getattr(args, "provider", None) or provider_env(
        "PICO_PROVIDER", default=DEFAULT_PROVIDER
    )
    if provider not in PROVIDER_CHOICES:
        choices = ", ".join(PROVIDER_CHOICES)
        raise ValueError(f"unknown provider: {provider}. expected one of: {choices}")
    return provider


def _effective_model(args, provider):
    # 模型选择优先级：
    # 1. 用户显式传入 --model
    # 2. provider 对应的环境变量
    # 3. 代码里的默认值
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        return explicit_model
    if provider == "openai":
        model = provider_env("PICO_OPENAI_MODEL", ("OPENAI_MODEL",))
        if model:
            return model
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        model = provider_env("PICO_ANTHROPIC_MODEL", ("ANTHROPIC_MODEL",))
        if model:
            return model
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "deepseek":
        model = provider_env("PICO_DEEPSEEK_MODEL", ("DEEPSEEK_MODEL",))
        if model:
            return model
        return DEFAULT_DEEPSEEK_MODEL
    return DEFAULT_OLLAMA_MODEL


def _configured_secret_names(args):
    configured_secret_names = set(DEFAULT_SECRET_ENV_NAMES)
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR) or os.environ.get(LEGACY_SECRET_ENV_NAMES_VAR, "")
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")
            if item.strip()
        )
    return sorted(configured_secret_names)


def _build_model_client(args):
    provider = _effective_provider(args)
    # CLI 只负责把 provider 选择翻译成具体 client。
    # 真正的提示词格式、缓存支持、HTTP 协议差异，都封装在 models.py 里。
    if provider == "openai":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("PICO_OPENAI_API_BASE", ("OPENAI_API_BASE",), DEFAULT_OPENAI_BASE_URL)
        api_key = provider_env(
            "PICO_OPENAI_API_KEY",
            (
                "OPENAI_API_KEY",
                "PICO_RIGHT_CODES_API_KEY",
                "RIGHT_CODES_API_KEY",
                "PICO_ANTHROPIC_API_KEY",
                "ANTHROPIC_API_KEY",
                "PICO_DEEPSEEK_API_KEY",
                "DEEPSEEK_API_KEY",
            ),
        )
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    if provider == "anthropic":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("PICO_ANTHROPIC_API_BASE", ("ANTHROPIC_API_BASE",), DEFAULT_ANTHROPIC_BASE_URL)
        api_key = provider_env(
            "PICO_ANTHROPIC_API_KEY",
            ("ANTHROPIC_API_KEY", "PICO_RIGHT_CODES_API_KEY", "RIGHT_CODES_API_KEY", "PICO_OPENAI_API_KEY", "OPENAI_API_KEY"),
        )
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    if provider == "deepseek":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("PICO_DEEPSEEK_API_BASE", ("DEEPSEEK_API_BASE",), DEFAULT_DEEPSEEK_BASE_URL)
        api_key = provider_env("PICO_DEEPSEEK_API_KEY", ("DEEPSEEK_API_KEY",))
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )

    model = _effective_model(args, provider)
    host = getattr(args, "host", DEFAULT_OLLAMA_HOST)
    return OllamaModelClient(
        model=model,
        host=host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
    )


def build_welcome(agent, model, host):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center(WELCOME_NAME),
            center(WELCOME_SUBTITLE),
            center(WELCOME_STATUS),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args):
    """根据 CLI 参数装配出一个可运行的 CodeFlow 实例。

    为什么存在：
    命令行参数只是字符串和开关，runtime 需要的是已经装配好的对象图：
    model client、workspace snapshot、session store、secret 配置等。
    这个函数负责把“启动参数”翻译成“agent 运行现场”。

    输入 / 输出：
    - 输入：`argparse` 解析后的 `args`
    - 输出：一个新的 `CodeFlow`，或一个从旧 session 恢复出来的 `CodeFlow`

    在 agent 链路里的位置：
    它是整个程序启动链路里最靠近 runtime 的装配点。`main()` 先调它，
    得到 agent 后，后面无论是 one-shot 还是 REPL 模式，都会落到 `ask()`。
    """
    # 这里是 CLI 到 runtime 的装配点：
    # 先采集工作区快照和加载项目级环境，再整理 secret 名单、模型后端和 session。
    workspace = WorkspaceContext.build(args.cwd)
    load_project_env(workspace.repo_root)
    configured_secret_names = _configured_secret_names(args)
    store = SessionStore(workspace_state_path(workspace.repo_root, "sessions"))
    model = _build_model_client(args)

    def model_factory(profile: ModelProfile):
        switch_args = argparse.Namespace(**vars(args))
        switch_args.provider = profile.provider
        switch_args.model = profile.model
        switch_args.base_url = profile.base_url
        return _build_model_client(switch_args)

    shell_backend = getattr(args, "shell_backend", "srt")
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return CodeFlow.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            denied_capabilities=getattr(args, "deny_capability", ()),
            read_paths=getattr(args, "read_path", None),
            write_paths=getattr(args, "write_path", None),
            approval_grant_seconds=getattr(args, "approval_grant_seconds", 60),
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            secret_env_names=configured_secret_names,
            shell_backend=shell_backend,
            model_factory=model_factory,
        )
    return CodeFlow(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        denied_capabilities=getattr(args, "deny_capability", ()),
        read_paths=getattr(args, "read_path", None),
        write_paths=getattr(args, "write_path", None),
        approval_grant_seconds=getattr(args, "approval_grant_seconds", 60),
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=configured_secret_names,
        shell_backend=shell_backend,
        model_factory=model_factory,
    )


def _agent_host(agent):
    return getattr(
        agent.model_client,
        "host",
        getattr(agent.model_client, "base_url", DEFAULT_OLLAMA_HOST),
    )


def switch_workspace(agent, args, raw_path):
    """Build a fresh agent rooted at a user-selected directory."""
    target = resolve_workspace_path(raw_path)
    if target is None:
        print(
            "无法识别工作区目录：请拖入一个真实存在的文件夹，"
            "或使用 /workspace <目录>。"
        )
        return agent

    current_root = os.path.normcase(os.path.abspath(agent.workspace.repo_root))
    target_root = os.path.normcase(os.path.abspath(target))
    if current_root == target_root:
        print(f"当前工作区已经是：{target}")
        return agent

    previous_cwd = args.cwd
    previous_resume = args.resume
    args.cwd = str(target)
    # 不把旧仓库的 session 恢复到新仓库；新仓库从独立会话开始。
    args.resume = None
    try:
        new_agent = build_agent(args)
    except Exception as exc:
        args.cwd = previous_cwd
        args.resume = previous_resume
        print(f"切换工作区失败：{exc}")
        return agent

    print(f"工作区已切换到：{new_agent.workspace.repo_root}")
    model = getattr(new_agent.model_client, "model", getattr(args, "model", DEFAULT_OLLAMA_MODEL))
    print(build_welcome(new_agent, model=model, host=_agent_host(new_agent)))
    return new_agent


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="CodeFlow local coding agent for DeepSeek, OpenAI-compatible, Anthropic-compatible, or Ollama models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=None,
        help="Model backend to use. Defaults to CODEFLOW_PROVIDER (legacy PICO_PROVIDER) or deepseek.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to qwen3.5:4b for Ollama or the matching CODEFLOW_*_MODEL (legacy PICO_* names are supported).",
    )
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama server URL.")
    parser.add_argument("--base-url", default=None, help="Provider API base URL for deepseek, openai, or anthropic.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="OpenAI-compatible request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--deny-capability",
        choices=tuple(sorted(toolkit.TOOL_CAPABILITIES)),
        action="append",
        default=[],
        help="Disable a tool capability for this run; may be repeated.",
    )
    parser.add_argument(
        "--read-path",
        action="append",
        default=None,
        help="Limit file listing, reading, and search to this workspace path; may be repeated.",
    )
    parser.add_argument(
        "--write-path",
        action="append",
        default=None,
        help="Limit file writes and patches to this workspace path; may be repeated.",
    )
    parser.add_argument(
        "--approval-grant-seconds",
        type=int,
        default=60,
        help="Lifetime of an explicitly granted exact-command or exact-path approval.",
    )
    parser.add_argument(
        "--shell-backend",
        choices=("srt", "direct"),
        default="srt",
        help="run_shell backend. SRT is fail-closed; direct is only for explicit testing/debugging.",
    )
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)

    model = getattr(agent.model_client, "model", getattr(args, "model", DEFAULT_OLLAMA_MODEL))
    host = _agent_host(agent)
    print(build_welcome(agent, model=model, host=host))

    if args.prompt:
        # one-shot 模式：只跑一次 ask，不进入 REPL 循环。
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                with thinking_indicator():
                    response = agent.ask(prompt)
                print(response)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        # 交互模式：每次读取一条用户输入，交给同一个 agent，
        # 因此 session history 和 working memory 会跨轮延续。
        try:
            user_input = input("\nCodeFlow> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        if user_input == "/workspace":
            print(f"当前工作区：{agent.workspace.repo_root}")
            continue
        if user_input.startswith("/workspace "):
            agent = switch_workspace(agent, args, user_input[len("/workspace ") :])
            continue

        # Windows Explorer 拖入终端后通常会得到一个带引号的完整路径。
        # 只对“整条输入就是现存路径”的情况自动切换，避免误伤普通对话。
        dropped_workspace = resolve_workspace_path(user_input)
        if dropped_workspace is not None:
            agent = switch_workspace(agent, args, user_input)
            continue

        print()
        try:
            with thinking_indicator():
                response = agent.ask(user_input)
            print(response)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
