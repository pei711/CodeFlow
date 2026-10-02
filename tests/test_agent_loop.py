from codeflow import CodeFlow, FakeModelClient, SessionStore, WorkspaceContext
from codeflow.agent_loop import AgentLoop


def build_agent(tmp_path, outputs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".pico" / "sessions")
    return CodeFlow(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )


def test_agent_loop_runs_same_control_flow_as_codeflow_ask(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Done.</final>",
        ],
    )

    answer = AgentLoop(agent).run("Inspect hello.txt")

    assert answer == "Done."
    assert agent.current_task_state.status == "completed"
    assert agent.run_store.report_path(agent.current_task_state.run_id).exists()


def test_codeflow_ask_delegates_to_agent_loop(tmp_path):
    agent = build_agent(tmp_path, ["<final>Facade works.</final>"])

    assert agent.ask("Use facade") == "Facade works."


def test_runtime_verifier_rejects_final_then_allows_repair(tmp_path):
    (tmp_path / "sample.txt").write_text("beta\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".pico" / "sessions")
    agent = CodeFlow(
        model_client=FakeModelClient(
            [
                "<final>Done.</final>",
                '<tool>{"name":"patch_file","args":{"path":"sample.txt","old_text":"beta","new_text":"beta-locked"}}</tool>',
                "<final>Done after verification.</final>",
            ]
        ),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        verification_contract={
            "name": "sample replacement",
            "checks": [
                {"type": "file_contains", "path": "sample.txt", "text": "beta-locked"},
                {"type": "file_not_contains", "path": "sample.txt", "text": "\nbeta\n"},
            ],
        },
    )

    answer = AgentLoop(agent).run("Replace beta with beta-locked")

    assert answer == "Done after verification."
    assert agent.current_task_state.verification_status == "passed"
    assert agent.current_task_state.verification_attempts == 2
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "beta-locked\n"
    events = [
        line
        for line in agent.run_store.trace_path(agent.current_task_state).read_text(encoding="utf-8").splitlines()
        if '"event": "verification_' in line
    ]
    assert len(events) == 2


def test_runtime_verifier_fails_closed_after_retry_budget(tmp_path):
    (tmp_path / "sample.txt").write_text("beta\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".pico" / "sessions")
    agent = CodeFlow(
        model_client=FakeModelClient(["<final>Done.</final>", "<final>Still done.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        verification_contract={
            "checks": [{"type": "file_contains", "path": "sample.txt", "text": "beta-locked"}]
        },
    )

    answer = AgentLoop(agent).run("Replace beta with beta-locked")

    assert "Verification failed" in answer
    assert agent.current_task_state.status == "failed"
    assert agent.current_task_state.stop_reason == "verification_failed"
