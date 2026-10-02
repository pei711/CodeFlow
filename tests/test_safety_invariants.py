import io
import os
import shlex
import subprocess
import sys
from unittest.mock import patch

import pytest

from codeflow import CodeFlow, FakeModelClient, SessionStore, WorkspaceContext
from codeflow import cli as codeflow_cli
from codeflow.task_state import TaskState


class FakeProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = io.BytesIO(stdout.encode("utf-8"))
        self.stderr = io.BytesIO(stderr.encode("utf-8"))

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".codeflow" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    shell_backend = kwargs.pop("shell_backend", "direct")
    return CodeFlow(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        shell_backend=shell_backend,
        **kwargs,
    )


def test_workspace_escape_is_rejected(tmp_path):
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("read_file", {"path": "../outside.txt"})

    assert "path escapes workspace" in result


def test_symlink_path_traversal_is_rejected(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        (tmp_path / "linked.txt").symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("read_file", {"path": "linked.txt"})

    assert "path escapes workspace" in result


def test_risky_tool_deny_behavior(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="never")

    result = agent.run_tool("run_shell", {"command": "echo hi", "timeout": 20})

    assert result == "error: approval denied for run_shell"


def test_cli_build_agent_wires_secret_env_names_from_parser(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(os.environ, {"GITHUB_PAT": "ghp-1", "GH_PAT": "ghp-2"}, clear=True), patch(
        "codeflow.cli.OllamaModelClient",
        DummyModelClient,
    ):
        args = codeflow_cli.build_arg_parser().parse_args(
            [
                "--cwd",
                str(tmp_path),
                "--approval",
                "auto",
                "--secret-env-name",
                "GITHUB_PAT",
                "--secret-env-name",
                "GH_PAT",
            ]
        )
        agent = codeflow_cli.build_agent(args)
        assert set(agent.secret_env_summary()["secret_env_names"]) == {"GITHUB_PAT", "GH_PAT"}


def test_cli_build_agent_uses_default_configured_secret_names(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(os.environ, {"GH_PAT": "ghp-default-1"}, clear=True), patch(
        "codeflow.cli.OllamaModelClient",
        DummyModelClient,
    ):
        args = codeflow_cli.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--approval", "auto"])
        agent = codeflow_cli.build_agent(args)
        assert agent.secret_env_summary()["secret_env_names"] == ["GH_PAT"]


def test_cli_build_agent_loads_project_env_secrets_before_redaction_setup(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / ".env").write_text("CODEFLOW_DEEPSEEK_API_KEY=sk-project-secret\n", encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True), patch("codeflow.cli.AnthropicCompatibleModelClient", DummyModelClient):
        args = codeflow_cli.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "deepseek"])
        agent = codeflow_cli.build_agent(args)
        assert agent.secret_env_summary()["secret_env_names"] == ["CODEFLOW_DEEPSEEK_API_KEY"]


def test_cli_build_agent_reads_secret_names_from_environment_config(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(
        os.environ,
        {
            "CODEFLOW_CUSTOM_SECRET": "custom-secret-value",
            "CODEFLOW_SECRET_ENV_NAMES": "CODEFLOW_CUSTOM_SECRET",
        },
        clear=True,
    ), patch("codeflow.cli.OllamaModelClient", DummyModelClient):
        args = codeflow_cli.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--approval", "auto"])
        agent = codeflow_cli.build_agent(args)
        assert agent.secret_env_summary()["secret_env_names"] == ["CODEFLOW_CUSTOM_SECRET"]


def test_run_shell_uses_allowlisted_environment_only(tmp_path):
    secret = "shh-allowlist-secret"
    agent = build_agent(tmp_path, [], approval_policy="auto")
    script = 'import os; print(os.getenv("CODEFLOW_ALLOWLIST_SECRET", "missing"))'
    command = (
        subprocess.list2cmdline([sys.executable, "-c", script])
        if os.name == "nt"
        else f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    )

    with patch.dict(os.environ, {"CODEFLOW_ALLOWLIST_SECRET": secret}, clear=False):
        result = agent.run_tool("run_shell", {"command": command, "timeout": 20})

    assert secret not in result
    assert "missing" in result


def test_bound_tool_methods_delegate_into_tools_module(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="auto")

    with patch("codeflow.tools.subprocess.Popen", return_value=FakeProcess(stdout="toolkit-shell\n")) as fake_run:
        shell_result = agent.tool_run_shell({"command": "echo bypass", "timeout": 20})

    assert "toolkit-shell" in shell_result
    fake_run.assert_called_once()
    assert agent.tool_run_shell.__func__.__module__ == "codeflow.runtime"

    with patch("codeflow.tools.tool_delegate", return_value="toolkit-delegate") as fake_delegate:
        delegate_result = agent.tool_delegate({"task": "inspect README.md", "max_steps": 2})

    assert delegate_result == "toolkit-delegate"
    fake_delegate.assert_called_once()


def test_run_shell_uses_srt_without_host_shell_and_records_metadata(tmp_path):
    cli_path = tmp_path / "cli.js"
    settings_path = tmp_path / "srt-settings.json"
    cli_path.write_text("// test stub\n", encoding="utf-8")
    settings_path.write_text("{}\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [],
        shell_backend="srt",
        srt_cli_path=cli_path,
        srt_settings_path=settings_path,
    )

    with patch("codeflow.tools.shutil.which", return_value="C:/Program Files/nodejs/node.exe"), patch(
        "codeflow.tools.subprocess.Popen"
    ) as fake_run:
        fake_run.return_value = FakeProcess(stdout="safe\n")
        result = agent.run_tool("run_shell", {"command": "echo safe", "timeout": 20})

    invocation = fake_run.call_args.args[0]
    assert invocation[-2:] == ["-c", "echo safe"]
    assert fake_run.call_args.kwargs["shell"] is False
    assert "sandbox: srt" in result
    assert agent._last_tool_result_metadata["sandbox_backend"] == "srt"
    assert agent._last_tool_result_metadata["sandbox_enforced"] is True


def test_run_shell_fails_closed_when_srt_is_missing(tmp_path):
    agent = build_agent(
        tmp_path,
        [],
        shell_backend="srt",
        srt_cli_path=tmp_path / "missing-cli.js",
        srt_settings_path=tmp_path / "missing-settings.json",
    )

    with patch("codeflow.tools.subprocess.Popen") as fake_run:
        result = agent.run_tool("run_shell", {"command": "echo must-not-run", "timeout": 20})

    assert "SRT sandbox unavailable" in result
    fake_run.assert_not_called()
    assert agent._last_tool_result_metadata["tool_error_code"] == "sandbox_unavailable"
    assert agent._last_tool_result_metadata["sandbox_enforced"] is False


def test_run_shell_marks_srt_initialization_failure_as_unavailable(tmp_path):
    cli_path = tmp_path / "cli.js"
    settings_path = tmp_path / "srt-settings.json"
    cli_path.write_text("// test stub\n", encoding="utf-8")
    settings_path.write_text("{}\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [],
        shell_backend="srt",
        srt_cli_path=cli_path,
        srt_settings_path=settings_path,
    )

    with patch("codeflow.tools.shutil.which", return_value="C:/Program Files/nodejs/node.exe"), patch(
        "codeflow.tools.subprocess.Popen"
    ) as fake_run:
        fake_run.return_value = FakeProcess(returncode=1, stderr="WFP egress fence could not be verified")
        result = agent.run_tool("run_shell", {"command": "echo blocked", "timeout": 20})

    assert "exit_code: 1" in result
    assert agent._last_tool_result_metadata["tool_error_code"] == "sandbox_unavailable"
    assert agent._last_tool_result_metadata["security_event_type"] == "sandbox_unavailable"
    assert agent._last_tool_result_metadata["sandbox_enforced"] is False


def test_delegate_depth_limit_is_enforced(tmp_path):
    agent = build_agent(tmp_path, [], depth=1, max_depth=1)

    try:
        agent.validate_tool("delegate", {"task": "inspect README.md", "max_steps": 2})
    except ValueError as exc:
        assert "delegate depth exceeded" in str(exc)
    else:
        raise AssertionError("delegate depth validation did not fail")


def test_delegate_child_is_read_only(tmp_path):
    target = tmp_path / "child-was-not-allowed.txt"
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"write a file","max_steps":2}}</tool>',
            '<tool>{"name":"write_file","args":{"path":"child-was-not-allowed.txt","content":"nope"}}</tool>',
            "<final>child done</final>",
            "<final>parent done</final>",
        ],
    )

    result = agent.ask("Delegate the work")

    assert result == "parent done"
    assert not target.exists()
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_configured_secret_env_names_are_redacted_in_trace_and_report(tmp_path):
    github_pat = "test-github-pat-placeholder"
    gh_pat = "test-gh-pat-placeholder"
    with patch.dict(os.environ, {"GITHUB_PAT": github_pat, "GH_PAT": gh_pat}, clear=True):
        agent = build_agent(
            tmp_path,
            [],
            secret_env_names=("GITHUB_PAT", "GH_PAT"),
        )
        state = TaskState.create(run_id="run_001", task_id="task_001", user_request="Mask configured secrets")
        agent.run_store.start_run(state)

        assert set(agent.secret_env_summary()["secret_env_names"]) == {"GITHUB_PAT", "GH_PAT"}

        payload = {
            "GITHUB_PAT": github_pat,
            "GH_PAT": gh_pat,
            "nested": {"GITHUB_PAT": github_pat, "GH_PAT": gh_pat},
            "list": [github_pat, gh_pat],
        }
        agent.emit_trace(state, "tool_executed", payload)
        agent.run_store.write_report(
            state,
            agent.redact_artifact({"task_state": state.to_dict(), "payload": payload}),
        )

    run_dir = agent.run_store.run_dir(state.run_id)
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    report_text = (run_dir / "report.json").read_text(encoding="utf-8")

    assert github_pat not in trace_text
    assert gh_pat not in trace_text
    assert github_pat not in report_text
    assert gh_pat not in report_text
    assert trace_text.count("<redacted>") >= 4
    assert report_text.count("<redacted>") >= 4
