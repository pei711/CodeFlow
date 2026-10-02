"""Model-call preflight for dynamic budgets, pressure tiers, and compaction."""

from __future__ import annotations

from .compaction import compact
from .token_estimator import context_budget_chars, estimate_text_tokens

SECTION_WEIGHTS = {
    "prefix": 0.20,
    "memory": 0.13,
    "skills": 0.07,
    "relevant_memory": 0.10,
    "history": 0.50,
}


class ContextOrchestrator:
    def __init__(self, agent):
        self.agent = agent

    def _identity(self):
        client = self.agent.model_client
        return "|".join(
            [
                str(getattr(client, "model", "unknown-model")),
                str(getattr(client, "base_url", "")),
                str(getattr(self.agent.prefix_state, "hash", "")),
            ]
        )

    def _calibration_ratio(self):
        runtime = dict(self.agent.session.get("context_runtime", {}) or {})
        calibration = dict(runtime.get("token_calibration", {}) or {})
        item = calibration.get(self._identity(), {})
        try:
            return float(item.get("ratio", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def _update_calibration(self, prompt, metadata):
        usage = dict(getattr(self.agent, "last_completion_metadata", {}) or {})
        actual = usage.get("input_tokens")
        if not actual:
            return
        estimated = estimate_text_tokens(prompt)
        if estimated <= 0:
            return
        ratio = min(3.0, max(0.25, float(actual) / estimated))
        runtime = self.agent.session.setdefault("context_runtime", {})
        calibration = runtime.setdefault("token_calibration", {})
        previous = calibration.get(self._identity(), {})
        old_ratio = float(previous.get("ratio", ratio) or ratio)
        calibration[self._identity()] = {
            "ratio": round(old_ratio * 0.7 + ratio * 0.3, 4),
            "last_actual_tokens": int(actual),
            "last_estimated_tokens": int(estimated),
        }
        metadata["usage_source"] = "actual_calibration"
        metadata["actual_input_tokens"] = int(actual)

    @staticmethod
    def _tier(ratio):
        if ratio >= 0.95:
            return "tier3_summary"
        if ratio >= 0.80:
            return "tier2_prune"
        if ratio >= 0.60:
            return "tier1_snip"
        return "tier0_observe"

    @staticmethod
    def _raw_tokens(metadata):
        sections = metadata.get("sections", {}) or {}
        return sum(int(value.get("raw_tokens", 0) or 0) for value in sections.values())

    def _skills_text(self):
        method = getattr(self.agent, "skills_text", None)
        if callable(method):
            return str(method() or "Skills:\n- none")
        return "Skills:\n- none"

    def _build(self, user_message, budget, section_budgets, skills_text):
        return self.agent.context_manager.build(
            user_message,
            total_budget=budget,
            section_budgets=section_budgets,
            skills_text=skills_text,
        )

    def prepare(self, user_message):
        client = self.agent.model_client
        previous_completion_metadata = dict(getattr(self.agent, "last_completion_metadata", {}) or {})
        context_window = getattr(client, "context_window", None)
        dynamic = bool(context_window) and self.agent.feature_enabled("context_orchestrator")
        if not dynamic:
            prompt, metadata = self.agent.context_manager.build(user_message)
            metadata.update({"context_orchestrator": False, "pressure_tier": "legacy"})
            return prompt, metadata

        budget_info = context_budget_chars(context_window)
        budget = int(budget_info["budget_chars"])
        section_budgets = {
            section: max(20, int(budget * weight))
            for section, weight in SECTION_WEIGHTS.items()
        }
        skills_text = self._skills_text()
        prompt, metadata = self._build(user_message, budget, section_budgets, skills_text)
        estimate_tokens = max(1, round(estimate_text_tokens(prompt) * self._calibration_ratio()))
        raw_tokens = max(1, self._raw_tokens(metadata))
        pressure = max(estimate_tokens, raw_tokens) / max(1, int(budget_info["budget_tokens"]))
        tier = self._tier(pressure)

        if tier in {"tier1_snip", "tier2_prune"}:
            adjusted = dict(section_budgets)
            adjusted["relevant_memory"] = max(20, int(adjusted["relevant_memory"] * 0.7))
            if tier == "tier2_prune":
                adjusted["skills"] = max(20, int(adjusted["skills"] * 0.5))
            prompt, metadata = self._build(user_message, budget, adjusted, skills_text)
            section_budgets = adjusted

        compacted = False
        compact_meta = {}
        history = list(self.agent.session.get("history", []))
        prior = dict(self.agent.session.get("context_summary", {}) or {})
        boundary = str(prior.get("last_included_event_id", ""))
        delta = history
        if boundary:
            boundary_index = next(
                (index for index, item in enumerate(history) if str(item.get("event_id", "")) == boundary),
                -1,
            )
            if boundary_index >= 0:
                delta = history[boundary_index + 1 :]
        if tier == "tier3_summary" and len(history) > 4 and len(delta) >= 4:
            runtime = self.agent.session.setdefault("context_runtime", {})
            if not runtime.get("circuit_open", False):
                result = compact(
                    self.agent,
                    delta[:-2] if len(delta) > 2 else delta,
                    user_message,
                    prior_summary=str(prior.get("content", "")),
                )
                self.agent.session["context_summary"] = {
                    "content": result.summary,
                    "last_included_event_id": result.included_event_id,
                    "summary_mode": result.mode,
                    "created_at": __import__("datetime").datetime.now().isoformat(),
                    "source_event_count": result.source_event_count,
                }
                if result.failure:
                    failures = int(runtime.get("compaction_failures", 0)) + 1
                    runtime["compaction_failures"] = failures
                    if failures >= 3:
                        runtime["circuit_open"] = True
                runtime["last_compact_call"] = result.call_metadata
                # The compaction request is not the primary model request.
                # Keep the previous provider usage available for calibration;
                # the compaction usage is stored separately above.
                self.agent.last_completion_metadata = previous_completion_metadata
                compacted = True
                compact_meta = {
                    "summary_mode": result.mode,
                    "source_event_count": result.source_event_count,
                    "included_event_id": result.included_event_id,
                    "failure": result.failure,
                    "call_metadata": dict(result.call_metadata or {}),
                }
                prompt, metadata = self._build(user_message, budget, section_budgets, skills_text)

        metadata.update(
            {
                "context_orchestrator": True,
                "context_budget": budget_info,
                "estimated_input_tokens": estimate_tokens,
                "raw_input_tokens": raw_tokens,
                "pressure_ratio": round(pressure, 4),
                "pressure_tier": tier,
                "context_compacted": compacted,
                "compaction": compact_meta,
                "usage_source": "estimated_proxy",
            }
        )
        self._update_calibration(prompt, metadata)
        self.agent.session_path = self.agent.session_store.save(self.agent.session)
        return prompt, metadata
