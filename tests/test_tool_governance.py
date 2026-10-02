import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from codeflow import CodeFlow, FakeModelClient, SessionStore, WorkspaceContext
from codeflow.tools import MAX_PROCESS_CAPTURE_BYTES, _run_bounded_process


def build_agent(tmp_path, **kwargs):
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("root readme\n", encoding="utf-8")
    return CodeFlow(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".pico" / "sessions"),
        approval_policy=kwargs.pop("approval_policy", "auto"),
        shell_backend="direct",
        **kwargs,
    )


def test_capabilities_remove_tools_and_guard_direct_runner(tmp_path):
    agent = build_agent(tmp_path, denied_capabilities=["process.execute"])

    assert "run_shell" not in agent.tools
    denied = agent.execute_tool("run_shell", {"command": "echo blocked"})
    assert denied.metadata["tool_error_code"] == "capability_denied"
    with pytest.raises(PermissionError, match="process.execute"):
        agent.tool_run_shell({"command": "echo blocked", "timeout": 1})


def test_file_capabilities_are_scoped_to_configured_workspace_paths(tmp_path):
    agent = build_agent(tmp_path, read_paths=["src"], write_paths=["src"])

    assert "app.py" in agent.run_tool("read_file", {"path": "src/app.py", "start": 1, "end": 2})
    assert "outside configured filesystem.read scopes" in agent.run_tool(
        "read_file", {"path": "README.md", "start": 1, "end": 2}
    )
    assert "outside configured filesystem.write scopes" in agent.run_tool(
        "write_file", {"path": "README.md", "content": "changed"}
    )
    denied = agent.execute_tool("write_file", {"path": "README.md", "content": "changed"})
    assert denied.metadata["tool_error_code"] == "path_scope_denied"
    assert denied.metadata["security_event_type"] == "path_scope_violation"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "root readme\n"


def test_approval_preview_and_short_lived_exact_scope_grant(tmp_path, capsys):
    agent = build_agent(tmp_path, approval_policy="ask", approval_grant_seconds=60)
    tool = agent.tools["write_file"]
    first_args = {"path": "src/app.py", "content": "first"}
    second_args = {"path": "src/app.py", "content": "second"}
    other_scope = {"path": "README.md", "content": "root"}

    with patch("builtins.input", side_effect=["g", "n"]) as ask:
        assert agent.approve("write_file", first_args, tool) is True
        assert agent.approve("write_file", second_args, tool) is True
        assert agent.approve("write_file", other_scope, tool) is False

    assert ask.call_count == 2
    preview = capsys.readouterr().out
    assert '"capability": "filesystem.write"' in preview
    assert "src\\\\app.py" in preview or "src/app.py" in preview
    assert "exact scope for 60s" in ask.call_args_list[0].args[0]


def test_tool_audit_arguments_are_redacted_and_bounded(tmp_path):
    agent = build_agent(tmp_path)
    secret = "sk-test-secret-123456"
    audit = agent.audit_tool_args({"command": f"echo {secret}", "content": "x" * 5000})

    assert secret not in json.dumps(audit)
    assert audit["content"]["truncated"] is True
    assert audit["content"]["characters"] == 5000
    assert len(audit["content"]["sha256"]) == 64


def test_subprocess_capture_is_bounded_and_timeout_kills_child(tmp_path):
    output = _run_bounded_process(
        [sys.executable, "-c", "print('x' * 200000)"], cwd=tmp_path, timeout=10
    )
    assert len(output["stdout"].encode("utf-8")) <= MAX_PROCESS_CAPTURE_BYTES
    assert "capture limit reached" in output["stdout_suffix"]

    with pytest.raises(subprocess.TimeoutExpired):
        _run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(5)"], cwd=tmp_path, timeout=0.1
        )
