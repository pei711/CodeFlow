"""Structured tool execution for the agent runtime."""

import re
import subprocess
from dataclasses import dataclass

from . import tools as toolkit
from .workspace import MAX_TOOL_OUTPUT, clip


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    metadata: dict


def _metadata(
    tool_status,
    tool_error_code="",
    security_event_type="",
    risk_level="low",
    read_only=True,
    affected_paths=None,
    workspace_changed=False,
    workspace_fingerprint="",
    diff_summary=None,
    capability="",
    approval_decision="",
    approval_scope_type="",
    approval_grant_seconds=0,
):
    result = {
        "tool_status": tool_status,
        "tool_error_code": tool_error_code,
        "security_event_type": security_event_type,
        "risk_level": risk_level,
        "read_only": read_only,
        "affected_paths": list(affected_paths or []),
        "workspace_changed": bool(workspace_changed),
        "diff_summary": list(diff_summary or []),
    }
    if workspace_fingerprint:
        result["workspace_fingerprint"] = workspace_fingerprint
    if capability:
        result["capability"] = capability
    if approval_decision:
        result["approval_decision"] = approval_decision
    if approval_scope_type:
        result["approval_scope_type"] = approval_scope_type
    if approval_grant_seconds:
        result["approval_grant_seconds"] = approval_grant_seconds
    return result


class ToolExecutor:
    def __init__(self, agent):
        self.agent = agent

    def execute(self, name, args):
        agent = self.agent
        capability = (
            toolkit.DELEGATE_TOOL_SPEC.get("capability")
            if name == "delegate"
            else toolkit.BASE_TOOL_SPECS.get(name, {}).get("capability")
        )
        if capability in agent.denied_capabilities:
            return ToolExecutionResult(
                content=f"error: capability '{capability}' is disabled in this run",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="capability_denied",
                    security_event_type="capability_denied",
                    risk_level="high",
                    read_only=False,
                    capability=capability,
                ),
            )
        if agent.allowed_tools is not None and name not in agent.allowed_tools:
            return ToolExecutionResult(
                content=f"error: tool '{name}' is not allowed in this run",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="tool_not_allowed",
                    risk_level="high",
                    read_only=False,
                ),
            )

        tool = agent.tools.get(name)
        if tool is None:
            return ToolExecutionResult(
                content=f"error: unknown tool '{name}'",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="unknown_tool",
                    risk_level="high",
                    read_only=False,
                ),
            )

        capability = tool.get("capability", "general")
        if capability in agent.denied_capabilities:
            return ToolExecutionResult(
                content=f"error: capability '{capability}' is disabled in this run",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="capability_denied",
                    security_event_type="capability_denied",
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                    capability=capability,
                ),
            )

        try:
            agent.validate_tool(name, args)
        except Exception as exc:
            example = agent.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            error_text = str(exc)
            security_event_type = (
                "path_escape"
                if "path escapes workspace" in error_text
                else "path_scope_violation"
                if "outside configured filesystem." in error_text
                else ""
            )
            tool_error_code = "path_scope_denied" if security_event_type == "path_scope_violation" else "invalid_arguments"
            return ToolExecutionResult(
                content=message,
                metadata=_metadata(
                    "rejected",
                    tool_error_code=tool_error_code,
                    security_event_type=security_event_type,
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                    capability=capability,
                ),
            )

        if agent.repeated_tool_call(name, args):
            return ToolExecutionResult(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="repeated_identical_call",
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                    capability=capability,
                ),
            )

        approval_metadata = {"approval_decision": "not_required"}
        if tool["risky"]:
            approved = agent.approve(name, args, tool)
            approval_metadata = dict(agent._last_approval_metadata)
        else:
            approved = True
        if not approved:
            return ToolExecutionResult(
                content=f"error: approval denied for {name}",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="approval_denied",
                    security_event_type="read_only_block" if agent.read_only else "approval_denied",
                    risk_level="high",
                    read_only=False,
                    capability=capability,
                    **approval_metadata,
                ),
            )

        before_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        try:
            raw_content = str(tool["run"](args))
            output_truncated = len(raw_content) > MAX_TOOL_OUTPUT or "capture limit reached" in raw_content
            content = clip(raw_content)
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = "ok"
            tool_error_code = ""
            if name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", content)
                exit_code = int(match.group(1)) if match else 0
                lowered = content.lower()
                sandbox_unavailable = agent.shell_backend == "srt" and any(
                    marker in lowered
                    for marker in (
                        "sandbox dependencies not available",
                        "sandbox user is not provisioned",
                        "wfp egress fence could not be verified",
                        "could not load settings",
                    )
                )
                sandbox_violation = agent.shell_backend == "srt" and any(
                    marker in lowered
                    for marker in (
                        "<sandbox_violations>",
                        "connection blocked by network allowlist",
                        "operation not permitted",
                    )
                )
                if sandbox_unavailable:
                    tool_status = "error"
                    tool_error_code = "sandbox_unavailable"
                elif exit_code != 0 and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            agent.update_memory_after_tool(name, args, content)
            metadata = _metadata(
                tool_status,
                tool_error_code=tool_error_code,
                security_event_type=(
                    "sandbox_unavailable"
                    if name == "run_shell" and sandbox_unavailable
                    else "sandbox_violation"
                    if name == "run_shell" and sandbox_violation
                    else ""
                ),
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
                capability=capability,
            )
            metadata["capability"] = capability
            metadata.update(approval_metadata)
            metadata["output_chars"] = len(raw_content)
            metadata["output_truncated"] = output_truncated
            if name == "run_shell":
                metadata["sandbox_backend"] = agent.shell_backend
                metadata["sandbox_enforced"] = agent.shell_backend == "srt" and not sandbox_unavailable
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=content, metadata=metadata)
        except Exception as exc:
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            exception_text = str(exc)
            sandbox_unavailable = name == "run_shell" and "SRT sandbox unavailable" in exception_text
            security_event_type = (
                "sandbox_unavailable"
                if sandbox_unavailable
                else "path_escape"
                if "path escapes workspace" in exception_text
                else ""
            )
            metadata = _metadata(
                "partial_success" if workspace_changed else "error",
                tool_error_code=(
                    "tool_partial_success"
                    if workspace_changed
                    else "sandbox_unavailable"
                    if sandbox_unavailable
                    else "tool_failed"
                ),
                security_event_type=security_event_type,
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
                capability=capability,
            )
            metadata["capability"] = capability
            metadata.update(approval_metadata)
            if isinstance(exc, subprocess.TimeoutExpired):
                metadata["tool_error_code"] = "tool_timeout"
                metadata["tool_status"] = "timeout"
            if name == "run_shell":
                metadata["sandbox_backend"] = agent.shell_backend
                metadata["sandbox_enforced"] = False
            agent.record_process_note_for_tool(name, metadata)
            error_message = "timed out" if metadata["tool_error_code"] == "tool_timeout" else str(exc)
            return ToolExecutionResult(content=f"error: tool {name} failed: {error_message}", metadata=metadata)
