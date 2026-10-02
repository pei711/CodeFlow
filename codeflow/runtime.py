"""CodeFlow Agent 运行时核心逻辑。

CodeFlow 是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import checkpoint as checkpointlib
from . import security as securitylib
from . import tools as toolkit
from .checkpoint import CHECKPOINT_NONE_STATUS
from .context_manager import ContextManager
from .context_orchestrator import ContextOrchestrator
from .features import memory as memorylib
from .model_switch import ModelProfile, detect_model_switch
from .prompt_prefix import build_prompt_prefix, tool_signature
from .run_store import RunStore
from .security import REDACTED_VALUE
from .session_store import SessionStore
from .tool_context import ToolContext
from .tool_executor import ToolExecutor
from .verifier import RuntimeVerifier
from .workspace import (
    IGNORED_PATH_NAMES,
    MAX_HISTORY,
    WorkspaceContext,
    clip,
    now,
    workspace_state_path,
)

DEFAULT_SHELL_ENV_ALLOWLIST = (
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "LOGNAME",
    "PATH",
    "PATHEXT",
    "PWD",
    "SHELL",
    "APPDATA",
    "SYSTEMROOT",
    "TERM",
    "TMPDIR",
    "TMP",
    "TEMP",
    "USER",
    "USERPROFILE",
    "WINDIR",
)
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "relevant_memory": True,
    "context_reduction": True,
    "prompt_cache": True,
    "context_orchestrator": True,
}
DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")

__all__ = ["CodeFlow", "SessionStore"]


class CodeFlow:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        shell_backend="srt",
        srt_cli_path=None,
        srt_settings_path=None,
        secret_env_names=None,
        feature_flags=None,
        allowed_tools=None,
        denied_capabilities=None,
        read_paths=None,
        write_paths=None,
        approval_grant_seconds=60,
        model_factory=None,
        verification_contract=None,
        max_verification_attempts=2,
    ):
        self.model_client = model_client
        self.model_factory = model_factory
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.max_verification_attempts = max(1, int(max_verification_attempts))
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        if shell_backend not in {"srt", "direct"}:
            raise ValueError("shell_backend must be 'srt' or 'direct'")
        package_root = Path(__file__).resolve().parents[1]
        self.shell_backend = shell_backend
        self.srt_cli_path = Path(
            srt_cli_path
            or package_root / "node_modules" / "@anthropic-ai" / "sandbox-runtime" / "dist" / "cli.js"
        ).resolve()
        self.srt_settings_path = Path(srt_settings_path or package_root / "srt-settings.json").resolve()
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.allowed_tools = self._normalize_allowed_tools(allowed_tools)
        self.denied_capabilities = self._normalize_denied_capabilities(denied_capabilities)
        self.read_paths = self._normalize_scoped_paths(read_paths)
        self.write_paths = self._normalize_scoped_paths(write_paths)
        self.approval_grant_seconds = max(0, int(approval_grant_seconds))
        self._approval_grants = {}
        self._last_approval_metadata = {"approval_decision": "not_required"}
        self.run_store = run_store or RunStore(workspace_state_path(workspace.repo_root, "runs"))
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        self.runtime_verifier = (
            RuntimeVerifier(self.root, verification_contract)
            if verification_contract is not None
            else None
        )
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tools = self._apply_tool_allowlist(self.build_tools())
        self.tool_executor = ToolExecutor(self)
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        self.context_manager = ContextManager(self)
        self.context_orchestrator = ContextOrchestrator(self)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_rejections = []
        self.last_durable_superseded = []
        self._last_tool_result_metadata = {}
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _ensure_session_shape(self):
        self.session.setdefault("history", [])
        # Older sessions predate event ids. Add stable local ids lazily so a
        # compaction boundary can still be resumed after a restart.
        for item in self.session.get("history", []):
            if isinstance(item, dict):
                item.setdefault("event_id", "evt_" + uuid.uuid4().hex[:12])
                item.setdefault("turn_id", "session")
        self.session.setdefault("memory", memorylib.default_memory_state())
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        resume_state = self.session.setdefault("resume_state", {})
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}

    def current_runtime_identity(self):
        return checkpointlib.current_runtime_identity(self)

    def checkpoint_state(self):
        return checkpointlib.checkpoint_state(self)

    def current_checkpoint(self):
        return checkpointlib.current_checkpoint(self)

    def invalidate_stale_memory(self):
        invalidated = self.memory.invalidate_stale_file_summaries()
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def evaluate_resume_state(self):
        return checkpointlib.evaluate_resume_state(self)

    def render_checkpoint_text(self):
        return checkpointlib.render_checkpoint_text(self)

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        return toolkit.build_tool_registry(self.tool_context())

    @staticmethod
    def _normalize_allowed_tools(allowed_tools):
        if allowed_tools is None:
            return None
        normalized = tuple(str(name).strip() for name in allowed_tools)
        if not normalized or any(not name for name in normalized):
            raise ValueError("allowed_tools must be a non-empty sequence of tool names")
        return normalized

    @staticmethod
    def _normalize_denied_capabilities(denied_capabilities):
        if isinstance(denied_capabilities, str):
            denied_capabilities = (denied_capabilities,)
        normalized = tuple(sorted({str(name).strip() for name in (denied_capabilities or ()) if str(name).strip()}))
        unknown = sorted(set(normalized) - toolkit.TOOL_CAPABILITIES)
        if unknown:
            raise ValueError(f"unknown denied capability: {', '.join(unknown)}")
        return normalized

    def _normalize_scoped_paths(self, paths):
        if paths is None:
            return None
        if isinstance(paths, (str, os.PathLike)):
            paths = (paths,)
        normalized = tuple(self.path(path) for path in paths)
        if not normalized:
            raise ValueError("path scopes must contain at least one path")
        return normalized

    def _apply_tool_allowlist(self, tools):
        legal_names = toolkit.legal_tool_names()
        unknown = [name for name in (self.allowed_tools or ()) if name not in legal_names]
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        allowed = set(self.allowed_tools) if self.allowed_tools is not None else None
        return {
            name: tool
            for name, tool in tools.items()
            if (allowed is None or name in allowed) and tool.get("capability") not in self.denied_capabilities
        }

    def tool_signature(self):
        return tool_signature(self.tools)

    def build_prefix(self):
        return build_prompt_prefix(workspace=self.workspace, tools=self.tools)

    def _apply_prefix_state(self, prefix_state):
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force=False):
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", None)

        # 工作区事实相对稳定，所以这里按整体刷新；
        # 只有这些事实真的变化了，才重建完整 prefix。
        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace

        prefix_state = self.build_prefix() if workspace_changed or force or previous_hash is None else self.prefix_state
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def memory_text(self):
        return self.memory.render_memory_text()

    def skills_text(self):
        """Return the active skill projection without coupling the core to a skill registry."""

        skills = getattr(self, "active_skills", None) or []
        if not skills:
            return "Skills:\n- none"
        lines = ["Skills:"]
        lines.extend(f"- {str(item).strip()}" for item in skills if str(item).strip())
        return "\n".join(lines)

    def verification_prompt_text(self):
        if self.runtime_verifier is None:
            return ""
        return self.runtime_verifier.prompt_text()

    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item):
        item = dict(item)
        item.setdefault("event_id", "evt_" + uuid.uuid4().hex[:12])
        item.setdefault("turn_id", str(getattr(self, "_active_turn_id", "session")))
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    @staticmethod
    def looks_sensitive_env_name(name):
        return securitylib.looks_sensitive_env_name(name)

    def is_secret_env_name(self, name):
        return securitylib.is_secret_env_name(name, secret_env_names=self.secret_env_names)

    def configured_secret_env_items(self):
        return securitylib.configured_secret_env_items(secret_env_names=self.secret_env_names)

    def detected_secret_env_items(self):
        return securitylib.detected_secret_env_items(secret_env_names=self.secret_env_names)

    def secret_env_summary(self):
        return securitylib.secret_env_summary(secret_env_names=self.secret_env_names)

    def detected_secret_env_summary(self):
        return securitylib.detected_secret_env_summary(secret_env_names=self.secret_env_names)

    def redact_text(self, text):
        return securitylib.redact_text(text, secret_env_names=self.secret_env_names)

    def redact_artifact(self, value, key=None):
        return securitylib.redact_artifact(value, key=key, secret_env_names=self.secret_env_names)

    def shell_env(self):
        return securitylib.shell_env(allowlist=self.shell_env_allowlist, root=self.root)

    def prompt_metadata(self, user_message, prompt):
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    def _build_prompt_and_metadata(self, user_message):
        refresh = self.refresh_prefix()
        self.resume_state = self.evaluate_resume_state()
        prompt, metadata = self.context_orchestrator.prepare(user_message)
        prompt_cache_key = self.prefix_state.hash
        if self.runtime_verifier is not None:
            # The contract is part of the stable prompt prefix from the model's
            # perspective, so a different contract must not reuse an old cache.
            contract_text = self.runtime_verifier.prompt_text()
            prompt_cache_key = hashlib.sha256(
                f"{prompt_cache_key}\n{contract_text}".encode()
            ).hexdigest()
        # 这里把“这轮 prompt 是怎么拼出来的”连同缓存相关状态一起记下来，
        # 后面 trace/report 才能解释清楚：为什么这一轮 prefix 变了、缓存有没有命中。
        metadata.update(
            {
                "prefix_chars": len(self.prefix),
                "workspace_chars": len(self.workspace.text()),
                "memory_chars": len(self.memory_text()),
                "history_chars": len(self.history_text()),
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": prompt_cache_key,
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,
                "tool_signature": self.prefix_state.tool_signature,
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return prompt, metadata

    def emit_trace(self, task_state, event, payload=None):
        payload = self.redact_artifact(payload or {})
        payload["event"] = event
        payload["created_at"] = now()
        # trace 是运行中的逐事件时间线，适合回答“这一轮 agent 到底做了什么”。
        self.run_store.append_trace(task_state, payload)
        return payload

    def audit_tool_args(self, args):
        safe_args = securitylib.redact_artifact(args, secret_env_names=self.secret_env_names)
        safe_args = self._scrub_secret_shaped_values(safe_args)
        for key in ("content", "old_text", "new_text", "command", "task"):
            value = safe_args.get(key) if isinstance(safe_args, dict) else None
            if isinstance(value, str) and len(value) > 2000:
                safe_args[key] = {
                    "preview": value[:1000] + " ... " + value[-500:],
                    "characters": len(value),
                    "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    "truncated": True,
                }
        return safe_args

    @staticmethod
    def _scrub_secret_shaped_values(value):
        if isinstance(value, dict):
            return {key: CodeFlow._scrub_secret_shaped_values(item) for key, item in value.items()}
        if isinstance(value, list):
            return [CodeFlow._scrub_secret_shaped_values(item) for item in value]
        if isinstance(value, str):
            value = re.sub(
                r"(?i)\b(api[_ -]?key|token|secret|password)\b\s*[:=]\s*[^\s,;]+",
                lambda match: match.group(1) + "=<redacted>",
                value,
            )
            return re.sub(r"(?i)\bsk-[A-Za-z0-9_-]{6,}\b", "<redacted>", value)
        return value

    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        return checkpointlib.create_checkpoint(self, task_state, user_message, trigger)

    def infer_next_step(self, task_state):
        return checkpointlib.infer_next_step(task_state)

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """
        if not self.feature_enabled("memory"):
            return
        path = args.get("path")
        if not path:
            return

        canonical_path = self.memory.canonical_path(path)
        # 不是所有工具结果都进入工作记忆。
        # 读文件会生成摘要；写文件/patch 会让旧摘要失效，因为它们可能过期了。
        if name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        elif name in {"write_file", "patch_file"}:
            self.memory.invalidate_file_summary(canonical_path)

    def remember_conversation_turn(self, user_message, final_answer, turn_id=""):
        if not self.feature_enabled("memory"):
            return
        summary = memorylib.summarize_conversation_turn(user_message, final_answer)
        if not summary:
            return
        self.memory.append_turn_summary(summary, turn_id=turn_id)
        self.session["memory"] = self.memory.to_dict()
        self.session_path = self.session_store.save(self.session)

    def note_tool(self, name, args, result):
        self.update_memory_after_tool(name, args, result)

    def record_process_note_for_tool(self, name, metadata):
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        path_text = ", ".join(affected_paths) or "workspace"
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def reject_durable_reason(self, note_text):
        text = str(note_text or "").strip()
        lowered = text.lower()
        if not text:
            return "empty"
        if REDACTED_VALUE in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
            return "secret_shaped"
        checkpoint_like_prefixes = (
            "current goal",
            "current blocker",
            "next step",
            "current phase",
            "key files",
            "freshness",
            "当前目标",
            "当前卡点",
            "下一步",
            "当前阶段",
            "关键文件",
            "已完成",
            "已排除",
        )
        if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
            return "transient_task_state"
        if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
            return "noisy_output"
        return ""

    def extract_durable_promotions(self, user_message, final_answer):
        user_text = str(user_message or "")
        if not (DURABLE_MEMORY_INTENT_PATTERN.search(user_text) or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)):
            return [], []
        promotions = []
        rejections = []
        for line in str(final_answer or "").splitlines():
            text = line.strip()
            if not text or REDACTED_VALUE in text:
                continue
            for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
                match = pattern.match(text)
                if not match:
                    continue
                note_text = match.group(1).strip()
                if note_text:
                    reason = self.reject_durable_reason(note_text)
                    if reason:
                        rejections.append(f"{topic}:{reason}")
                        break
                    promotions.append((topic, note_text))
                break
        return promotions, rejections

    def promote_durable_memory(self, user_message, final_answer):
        promotions, rejections = self.extract_durable_promotions(user_message, final_answer)
        promoted, superseded = self.memory.promote_durable(promotions)
        self.session["memory"] = self.memory.to_dict()
        self.last_durable_promotions = promoted
        self.last_durable_rejections = rejections
        self.last_durable_superseded = superseded
        return promoted, rejections, superseded

    def ask(self, user_message):
        from .agent_loop import AgentLoop

        profile = detect_model_switch(user_message)
        if profile is not None:
            return self._handle_model_switch(user_message, profile)
        return AgentLoop(self).run(user_message)

    @staticmethod
    def _model_label(model_client):
        model = str(getattr(model_client, "model", "unknown-model"))
        provider = model_client.__class__.__name__.replace("ModelClient", "")
        return f"{provider}/{model}"

    def _handle_model_switch(self, user_message, profile: ModelProfile):
        self.record({"role": "user", "content": user_message, "created_at": now()})
        previous = self._model_label(self.model_client)
        if self.model_factory is None:
            response = (
                f"当前运行实例没有配置模型切换器，仍使用 {previous}。"
                "请重新启动 CodeFlow 后再切换。"
            )
            self.record({"role": "assistant", "content": response, "created_at": now()})
            self.remember_conversation_turn(user_message, response, "model-switch")
            return response

        try:
            next_client = self.model_factory(profile)
        except Exception as exc:
            response = f"切换到 {profile.name} 失败，仍使用 {previous}：{exc}"
            self.record({"role": "assistant", "content": response, "created_at": now()})
            self.remember_conversation_turn(user_message, response, "model-switch")
            return response

        self.model_client = next_client
        response = (
            f"已切换到 {profile.description}："
            f"provider={profile.provider}，model={profile.model}。"
        )
        self.record({"role": "assistant", "content": response, "created_at": now()})
        self.remember_conversation_turn(user_message, response, "model-switch")
        return response

    def execute_tool(self, name, args):
        result = self.tool_executor.execute(name, args)
        self._last_tool_result_metadata = dict(result.metadata)
        return result

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """
        return self.execute_tool(name, args).content

    def repeated_tool_call(self, name, args):
        # agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        # 这里提前挡掉最简单的这种循环。
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "durable_promotions": list(self.last_durable_promotions),
            "durable_rejections": list(self.last_durable_rejections),
            "durable_superseded": list(self.last_durable_superseded),
            "verification": {
                "configured": self.runtime_verifier is not None,
                "contract": self.runtime_verifier.contract if self.runtime_verifier else None,
            },
            "redacted_env": self.detected_secret_env_summary(),
        }

    def tool_example(self, name):
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self.tool_context(), name, args)

    def tool_context(self):
        return ToolContext(
            root=self.root,
            path_resolver=self.path,
            shell_env_provider=self.shell_env,
            depth=self.depth,
            max_depth=self.max_depth,
            spawn_delegate=self.spawn_delegate,
            shell_backend=self.shell_backend,
            srt_cli_path=self.srt_cli_path,
            srt_settings_path=self.srt_settings_path,
            read_paths=self.read_paths,
            write_paths=self.write_paths,
            denied_capabilities=self.denied_capabilities,
        )

    def spawn_delegate(self, args):
        task = str(args.get("task", "")).strip()
        child = CodeFlow(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            run_store=self.run_store,
            approval_policy="never",
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
            secret_env_names=self.secret_env_names,
            denied_capabilities=self.denied_capabilities,
            read_paths=self.read_paths,
            write_paths=self.write_paths,
            shell_env_allowlist=self.shell_env_allowlist,
            shell_backend=self.shell_backend,
            srt_cli_path=self.srt_cli_path,
            srt_settings_path=self.srt_settings_path,
            model_factory=self.model_factory,
        )
        # 委派的目标是“调查”，不是“放权执行”。
        # 子 agent 以只读方式运行、步数更少，最后只把结论文本返回给父 agent。
        child.session["memory"]["task"] = task
        child.session["memory"]["notes"] = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)

    def tool_list_files(self, args):
        return toolkit.tool_list_files(self.tool_context(), args)

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self.tool_context(), args)

    def tool_search(self, args):
        return toolkit.tool_search(self.tool_context(), args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self.tool_context(), args)

    def tool_write_file(self, args):
        return toolkit.tool_write_file(self.tool_context(), args)

    def tool_patch_file(self, args):
        return toolkit.tool_patch_file(self.tool_context(), args)

    def tool_delegate(self, args):
        return toolkit.tool_delegate(self.tool_context(), args)

    def approve(self, name, args, tool=None):
        if self.read_only:
            self._last_approval_metadata = {"approval_decision": "denied"}
            return False
        if self.approval_policy == "auto":
            self._last_approval_metadata = {"approval_decision": "auto"}
            return True
        if self.approval_policy == "never":
            self._last_approval_metadata = {"approval_decision": "denied"}
            return False
        tool = tool or self.tools.get(name, {})
        capability = tool.get("capability", "unknown")
        resource = self._approval_resource(name, args, capability)
        grant_key = (capability, resource)
        expires_at = self._approval_grants.get(grant_key, 0)
        if expires_at > time.monotonic():
            self._last_approval_metadata = {
                "approval_decision": "temporary_grant",
                "approval_scope_type": "exact_path" if capability == "filesystem.write" else "exact_command",
                "approval_grant_seconds": self.approval_grant_seconds,
            }
            return True
        self._approval_grants.pop(grant_key, None)
        preview = {
            "tool": name,
            "capability": capability,
            "workspace": str(self.root),
            "arguments": self._approval_args(args),
        }
        if "path" in args:
            preview["resolved_path"] = str(self.path(args["path"]))
        print("Tool approval preview:")
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        grant_hint = f"/g=grant this exact scope for {self.approval_grant_seconds}s" if self.approval_grant_seconds else ""
        try:
            answer = input(f"Approve once? [y/N{grant_hint}] ")
        except EOFError:
            self._last_approval_metadata = {"approval_decision": "denied"}
            return False
        answer = answer.strip().lower()
        if answer in {"y", "yes"}:
            self._last_approval_metadata = {"approval_decision": "once"}
            return True
        if answer in {"g", "grant"} and self.approval_grant_seconds:
            self._approval_grants[grant_key] = time.monotonic() + self.approval_grant_seconds
            self._last_approval_metadata = {
                "approval_decision": "temporary_grant",
                "approval_scope_type": "exact_path" if capability == "filesystem.write" else "exact_command",
                "approval_grant_seconds": self.approval_grant_seconds,
            }
            return True
        self._last_approval_metadata = {"approval_decision": "denied"}
        return False

    def _approval_args(self, args):
        safe_args = securitylib.redact_artifact(args, secret_env_names=self.secret_env_names)
        safe_args = self._scrub_secret_shaped_values(safe_args)
        for key in ("content", "old_text", "new_text"):
            value = safe_args.get(key) if isinstance(safe_args, dict) else None
            if isinstance(value, str) and len(value) > 4000:
                safe_args[key] = {
                    "preview": value[:2000] + " ... " + value[-500:],
                    "characters": len(value),
                    "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                }
        return safe_args

    def _approval_resource(self, name, args, capability):
        if capability == "filesystem.write" and "path" in args:
            return str(self.path(args["path"]))
        if capability == "process.execute":
            return str(args.get("command", "")).strip()
        return name

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        """
        raw = str(raw)
        # 这里支持两种工具格式：
        # 1. <tool>...</tool> 里包 JSON，适合简短调用
        # 2. XML 风格属性/子标签，适合写文件这类多行内容
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = CodeFlow.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", CodeFlow.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", CodeFlow.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", CodeFlow.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", CodeFlow.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = CodeFlow.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", CodeFlow.retry_notice()
        # DeepSeek Flash 可能返回 DSML 工具调用，而不是 CodeFlow 自己的
        # <tool> 格式。这里把 DSML 只转换成统一的内部 payload，后续仍然
        # 经过 execute_tool -> ToolExecutor 的参数校验、工作区限制和审批。
        if CodeFlow.looks_like_dsml(raw):
            payload = CodeFlow.parse_dsml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", CodeFlow.retry_notice("model returned malformed DSML tool output")
        if "<final>" in raw:
            final = CodeFlow.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", CodeFlow.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", CodeFlow.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call, a DeepSeek DSML tool call, "
            "or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def looks_like_dsml(raw):
        """判断文本里是否出现 DeepSeek DSML 工具调用标记。"""
        return re.search(
            r"<[|｜]\s*DSML\s*[|｜]\s*(?:tool_calls|function_calls|calls|invoke)\b",
            raw,
            re.IGNORECASE,
        ) is not None

    @staticmethod
    def parse_dsml_tool(raw):
        """解析 DeepSeek DSML 的第一个工具调用。

        DeepSeek 的不同版本/网关会使用 ASCII `|` 或全角 `｜`，并且有的
        输出会把 `calls`、`invoke`、`parameter` 与 DSML 标记之间写成带空格
        的形式。因此这里故意保持格式宽容，但只提取明确的 invoke 节点。
        `string="false"` 的参数按 JSON 解码，其他参数按原始字符串处理。
        """
        marker = r"[|｜]\s*DSML\s*[|｜]"
        invoke = re.search(
            rf"<{marker}\s*invoke\s+name\s*=\s*(?P<quote>['\"])(?P<name>[^'\"]+)"
            rf"(?P<quote2>['\"])\s*>(?P<body>.*?)</{marker}\s*invoke\s*>",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if not invoke:
            return None

        name = invoke.group("name").strip()
        if not name:
            return None
        # 某些兼容层可能把工具名写成 functions.read_file；CodeFlow 的
        # 工具注册表使用最后一段名称即可，真正的合法性仍由 executor 校验。
        name = name.rsplit(".", 1)[-1]

        parameter_pattern = re.compile(
            rf"<{marker}\s*parameter\s+name\s*=\s*(?P<name_quote>['\"])(?P<param>[^'\"]+)"
            rf"(?P<name_quote2>['\"])\s*"
            rf"(?:string\s*=\s*(?P<string_quote>['\"])(?P<string>[^'\"]+)(?P<string_quote2>['\"])\s*)?"
            rf">(?P<value>.*?)</{marker}\s*parameter\s*>",
            re.DOTALL | re.IGNORECASE,
        )
        args = {}
        for parameter in parameter_pattern.finditer(invoke.group("body")):
            key = parameter.group("param").strip()
            value = parameter.group("value").strip("\r\n")
            string_flag = (parameter.group("string") or "true").strip().lower()
            if string_flag == "false":
                try:
                    value = json.loads(value.strip())
                except (TypeError, ValueError):
                    # 不让单个格式问题丢掉整个工具调用；executor 仍会做
                    # 具体 schema 校验，并把错误反馈给模型。
                    value = value.strip()
            args[key] = value

        return {"name": name, "args": args}

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.DOTALL)
        if not match:
            return None
        attrs = CodeFlow.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = CodeFlow.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.session_store.save(self.session)

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved
