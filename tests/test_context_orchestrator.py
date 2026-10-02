from codeflow import CodeFlow, FakeModelClient, SessionStore, WorkspaceContext


def build_agent(tmp_path, outputs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    client = FakeModelClient(outputs)
    client.context_window = 20_000
    return CodeFlow(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".pico" / "sessions"),
        approval_policy="auto",
    )


def test_dynamic_budget_reports_pressure_and_keeps_skills_separate(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.active_skills = ["Use the verifier before claiming completion."]
    for index in range(8):
        agent.record({"role": "user", "content": f"history-{index} " + ("x" * 2000)})

    prompt, metadata = agent._build_prompt_and_metadata("latest request")

    assert metadata["context_orchestrator"] is True
    assert metadata["context_budget"]["budget_chars"] == 14_464
    assert metadata["section_order"] == [
        "prefix",
        "memory",
        "skills",
        "relevant_memory",
        "history",
        "current_request",
    ]
    assert "Skills:\n- Use the verifier" in prompt
    assert metadata["pressure_tier"] in {"tier0_observe", "tier1_snip", "tier2_prune", "tier3_summary"}


def test_compaction_preserves_raw_history_and_renders_summary_projection(tmp_path):
    summary = """## Goal
Keep the task state.
## Constraints
- preserve paths
## Files Read / Files Modified
- sample.py
## Key Decisions
- keep the latest tail
## Rejected Paths
- none
## Blockers
- none
## Next Steps
- run tests
## Critical Context
- compacted old events"""
    agent = build_agent(tmp_path, [summary, "<final>Done.</final>"])
    for index in range(8):
        agent.record({"role": "user", "content": f"history-{index} " + ("x" * 2000)})

    assert agent.ask("continue") == "Done."
    assert agent.session["context_summary"]["summary_mode"] == "llm"
    assert len(agent.session["history"]) >= 9
    assert "Context summary:" in agent.model_client.prompts[-1]
    assert "continue" in agent.model_client.prompts[-1]


def test_compaction_falls_back_deterministically_when_summary_is_invalid(tmp_path):
    agent = build_agent(tmp_path, ["invalid summary", "<final>Done.</final>"])
    for index in range(8):
        agent.record({"role": "user", "content": f"history-{index} " + ("x" * 2000)})

    assert agent.ask("continue") == "Done."
    summary = agent.session["context_summary"]
    assert summary["summary_mode"] == "deterministic_fallback"
    assert "## Goal" in summary["content"]
