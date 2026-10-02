"""Structured context handoff generation.

Compaction is intentionally isolated from ``ContextManager``.  The latter is
deterministic prompt rendering; this module owns the optional extra model call
and its deterministic fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .workspace import clip

SUMMARY_HEADINGS = (
    "Goal",
    "Constraints",
    "Files Read / Files Modified",
    "Key Decisions",
    "Rejected Paths",
    "Blockers",
    "Next Steps",
    "Critical Context",
)
PATH_PATTERN = re.compile(r"(?<![\w.-])(?:[A-Za-z]:[\\/])?[-\w./\\]+\.[A-Za-z0-9_-]+")
ERROR_PATTERN = re.compile(r"(?:error|traceback|failed|failure|exception)[:\s].*", re.IGNORECASE)


@dataclass
class CompactionResult:
    summary: str
    mode: str
    included_event_id: str
    source_event_count: int
    call_metadata: dict
    failure: str = ""


def _event_text(item: dict) -> str:
    if item.get("role") == "tool":
        return f"[tool:{item.get('name', 'tool')}] {item.get('args', {})}\n{item.get('content', '')}"
    return f"[{item.get('role', 'event')}] {item.get('content', '')}"


def _latest_user(events, fallback=""):
    for item in reversed(events):
        if item.get("role") == "user" and str(item.get("content", "")).strip():
            return clip(item["content"], 500)
    return clip(fallback, 500)


def _lines_matching(events, pattern, limit):
    values = []
    for item in events:
        for line in str(item.get("content", "")).splitlines():
            line = line.strip()
            if line and pattern.search(line) and line not in values:
                values.append(clip(line, 300))
                if len(values) >= limit:
                    return values
    return values


def deterministic_summary(events, current_request="", prior_summary="") -> str:
    """Build a bounded, zero-cost summary from observable session facts."""

    files = []
    errors = []
    for item in events:
        args = item.get("args", {}) or {}
        path = str(args.get("path", "")).strip()
        if path and path not in files:
            files.append(path)
        for match in PATH_PATTERN.findall(_event_text(item)):
            if match not in files:
                files.append(match)
        errors.extend(_lines_matching([item], ERROR_PATTERN, 3))

    constraints = _lines_matching(
        events,
        re.compile(r"(?:不要|必须|只|禁止|don't|must|only|never)", re.IGNORECASE),
        5,
    )
    decisions = _lines_matching(
        events,
        re.compile(r"(?:选择|决定|采用|最终|decided|selected|approach|switched to)", re.IGNORECASE),
        3,
    )
    rejected = _lines_matching(
        events,
        re.compile(r"(?:不行|失败|回退|拒绝|doesn't work|failed|reverted)", re.IGNORECASE),
        3,
    )
    last_event_id = str(events[-1].get("event_id", "")) if events else ""
    prior = clip(prior_summary, 700).strip()
    lines = [
        "## Goal",
        _latest_user(events, current_request) or "-",
        "## Constraints",
        *([f"- {item}" for item in constraints] or ["- none recorded"]),
        "## Files Read / Files Modified",
        *([f"- {item}" for item in files[:12]] or ["- none recorded"]),
        "## Key Decisions",
        *([f"- {item}" for item in decisions] or ["- none recorded"]),
        "## Rejected Paths",
        *([f"- {item}" for item in rejected] or ["- none recorded"]),
        "## Blockers",
        *([f"- {item}" for item in errors[:3]] or ["- none recorded"]),
        "## Next Steps",
        "- Continue from the latest checkpoint and verify the next tool result.",
        "## Critical Context",
        f"- compacted {len(events)} history items; last_event_id={last_event_id or '-'}",
    ]
    if prior:
        lines.extend(["- Prior summary retained:", f"  {prior}"])
    return clip("\n".join(lines), 2000)


def _render_for_llm(events, prior_summary=""):
    chunks = []
    for item in events:
        limit = 3000 if item.get("role") == "tool" else 2000
        chunks.append(clip(_event_text(item), limit))
    body = clip("\n\n".join(chunks), 20_000)
    prior = clip(prior_summary, 3000)
    return (
        "You are a context handoff writer for a coding agent. Summarize only the "
        "observable task facts below. Preserve exact paths, test names, variable "
        "names, and error text. Return markdown with every heading exactly once: "
        + ", ".join(f"## {heading}" for heading in SUMMARY_HEADINGS)
        + "\n\nPrior summary:\n"
        + (prior or "- none")
        + "\n\nEvents:\n"
        + body
    )


def valid_summary(text: str) -> bool:
    value = str(text or "")
    return all(f"## {heading}" in value for heading in ("Goal", "Next Steps")) and bool(
        re.search(r"## Goal\s+(.+)", value, re.DOTALL)
    )


def compact(agent, events, current_request, prior_summary="") -> CompactionResult:
    """Try LLM handoff first, then guarantee deterministic progress."""

    included = str(events[-1].get("event_id", "")) if events else ""
    call_metadata = {}
    try:
        prompt = _render_for_llm(events, prior_summary=prior_summary)
        result = agent.model_client.complete(prompt, min(1024, max(256, agent.max_new_tokens)))
        call_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
        if valid_summary(result):
            return CompactionResult(
                summary=clip(result, 5000),
                mode="llm",
                included_event_id=included,
                source_event_count=len(events),
                call_metadata=call_metadata,
            )
        raise ValueError("summary missing required Goal or Next Steps")
    except Exception as exc:  # noqa: BLE001 - compaction must always fall back
        # A provider response can be usable at the transport level but fail
        # summary validation.  Keep its real usage metadata so evaluation and
        # observability include the failed LLM compaction request as cost.
        if not call_metadata:
            call_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
        return CompactionResult(
            summary=deterministic_summary(events, current_request=current_request, prior_summary=prior_summary),
            mode="deterministic_fallback",
            included_event_id=included,
            source_event_count=len(events),
            call_metadata=call_metadata,
            failure=str(exc),
        )
