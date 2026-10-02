"""Run the real DeepSeek Flash context A/B evaluation.

This benchmark intentionally uses the provider client, not FakeModelClient.  It
creates an isolated workspace for every cell, seeds the same history for the
baseline/governed pair, and records provider-reported input usage from trace
events.  The output directory contains JSON (machine-readable) and Markdown
(reviewable) summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codeflow.providers.clients import AnthropicCompatibleModelClient
from codeflow.runtime import CodeFlow
from codeflow.session_store import SessionStore
from codeflow.task_state import VERIFICATION_PASSED
from codeflow.workspace import WorkspaceContext

DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "real-deepseek-flash-context"

PRESSURE_LEVELS = [
    {"id": "L1", "name": "低压观察", "context_window": 50_000, "history_chars": 3_000},
    {"id": "L2", "name": "中压收缩", "context_window": 50_000, "history_chars": 15_000},
    {"id": "L3", "name": "高压裁剪", "context_window": 60_000, "history_chars": 27_000},
    {"id": "L4", "name": "极限压缩", "context_window": 70_000, "history_chars": 40_000},
]

SCENARIOS = [
    ("long_session", "长会话", "long"),
    ("cross_turn_dependency", "跨轮次依赖", "cross"),
    ("compression_degradation", "压缩降级", "degrade"),
]


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def make_history(scenario: str, target_chars: int, marker: str) -> list[dict]:
    """Create deterministic, semantically plausible prior turns."""

    now = "2026-09-22T00:00:00+08:00"
    history: list[dict] = []
    if scenario == "cross":
        history.extend(
            [
                {
                    "role": "user",
                    "content": (
                        "跨轮次约定：后续维护 target.txt 时，必须写入目标标记 "
                        f"{marker}，不要改动其他文件。"
                    ),
                    "created_at": now,
                },
                {
                    "role": "assistant",
                    "content": f"已记录约定，目标标记为 {marker}。",
                    "created_at": now,
                },
            ]
        )
    elif scenario == "degrade":
        history.append(
            {
                "role": "user",
                "content": f"压缩前的关键任务：把 target.txt 更新为包含 {marker}。",
                "created_at": now,
            }
        )
    else:
        history.append(
            {
                "role": "user",
                "content": "请先分析仓库里的维护记录，再完成 target.txt 的小修复。",
                "created_at": now,
            }
        )

    filler = (
        "历史工具输出：完成一次只读检查。路径 target.txt；状态 clean；"
        "没有发现阻塞。请保留可复现的上下文边界和验证结果。"
    )
    index = 0
    while sum(len(str(item.get("content", ""))) for item in history) < target_chars:
        index += 1
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": f"第 {index} 轮分析：{filler}",
                    "created_at": now,
                },
                {
                    "role": "tool",
                    "name": "run_shell",
                    "args": {"command": f"python check_{index}.py"},
                    "content": (f"check_{index}: " + filler + "\n") * 3,
                    "created_at": now,
                },
            ]
        )
    return history


def build_request(scenario: str, marker: str) -> str:
    if scenario == "cross":
        return (
            "请继续完成前几轮已经约定的 target.txt 维护任务。必须使用工具修改文件，"
            "从历史上下文找回目标标记，不要向我索要标记；完成后回复 <final>done</final>。"
        )
    if scenario == "degrade":
        return (
            "长会话可能已经经历上下文压缩或降级。请根据仍然可用的上下文继续完成"
            " target.txt 维护任务，使用工具写入之前约定的标记，最后回复 <final>done</final>。"
        )
    return (
        f"请完成当前维护任务：使用工具读取 target.txt，并将其更新为包含 {marker}。"
        "不要只解释，实际修改文件；完成后回复 <final>done</final>。"
    )


def make_client(env: dict[str, str], context_window: int) -> AnthropicCompatibleModelClient:
    client = AnthropicCompatibleModelClient(
        model="deepseek-flash",
        base_url=env.get("PICO_DEEPSEEK_API_BASE", "https://api.deepseek.com/anthropic"),
        api_key=env.get("PICO_DEEPSEEK_API_KEY", ""),
        temperature=0.0,
        timeout=300,
    )
    # The override controls benchmark pressure only; the actual request still
    # goes to DeepSeek Flash and reports server-side usage.
    client.context_window = context_window
    return client


def read_trace(run_dir: Path) -> list[dict]:
    trace = run_dir / "trace.jsonl"
    if not trace.exists():
        return []
    events = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def run_one(env: dict[str, str], output_root: Path, scenario: str, level: dict, governed: bool) -> dict:
    label = "governed" if governed else "baseline"
    # Keep the target and seeded history byte-for-byte equivalent across the
    # baseline/governed pair; only the feature flags differ.
    marker = f"PICO_EVAL_{scenario.upper()}_{level['id']}"
    run_root = Path(tempfile.mkdtemp(prefix=f"{scenario}-{level['id']}-{label}-", dir=output_root))
    (run_root / "target.txt").write_text("initial content\n", encoding="utf-8")
    history = make_history(scenario, int(level["history_chars"]), marker)
    memory = {
        "working": {"task_summary": "", "recent_files": []},
        "episodic_notes": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [f"关键跨轮次事实：target.txt 需要包含 {marker}"],
        "next_note_index": 0,
    }
    session = {
        "id": f"eval-{scenario}-{level['id']}-{label}-{int(time.time() * 1000)}",
        "created_at": "2026-09-22T00:00:00+08:00",
        "workspace_root": str(run_root),
        "history": history,
        "memory": memory,
    }
    initial_history_chars = sum(len(str(item.get("content", ""))) for item in history)
    initial_history_events = len(history)
    workspace = WorkspaceContext.build(run_root, repo_root_override=run_root)
    client = make_client(env, int(level["context_window"]))
    flags = {
        "memory": True,
        "relevant_memory": True,
        "context_reduction": governed,
        "context_orchestrator": governed,
        "prompt_cache": False,
    }
    contract = {
        "name": f"{scenario}-{level['id']}-{label}",
        "checks": [{"type": "file_contains", "path": "target.txt", "text": marker}],
    }
    agent = CodeFlow(
        model_client=client,
        workspace=workspace,
        session_store=SessionStore(run_root / ".pico" / "sessions"),
        session=session,
        run_store=None,
        approval_policy="auto",
        max_steps=10,
        max_new_tokens=768,
        shell_backend="direct",
        feature_flags=flags,
        verification_contract=contract,
        max_verification_attempts=2,
    )
    error = ""
    try:
        answer = agent.ask(build_request(scenario, marker))
    except Exception as exc:  # noqa: BLE001 - preserve failed cells in aggregate report
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
    state = agent.current_task_state.to_dict() if agent.current_task_state is not None else {}
    run_dir = Path(agent.current_run_dir) if agent.current_run_dir else None
    events = read_trace(run_dir) if run_dir else []
    main_usage = [
        event.get("completion_metadata", {})
        for event in events
        if event.get("event") == "model_parsed" and event.get("completion_metadata")
    ]
    main_input = sum(int(item.get("input_tokens", 0) or 0) for item in main_usage)
    main_output = sum(int(item.get("output_tokens", 0) or 0) for item in main_usage)
    compaction_calls = []
    for event in events:
        if event.get("event") != "context_compacted":
            continue
        metadata = dict(event.get("call_metadata", {}) or {})
        if metadata:
            compaction_calls.append(metadata)
    compaction_input = sum(int(item.get("input_tokens", 0) or 0) for item in compaction_calls)
    compaction_output = sum(int(item.get("output_tokens", 0) or 0) for item in compaction_calls)
    prompt_events = [event for event in events if event.get("event") == "prompt_built"]
    prompt_meta = [dict(event.get("prompt_metadata", {}) or {}) for event in prompt_events]
    last_meta = prompt_meta[-1] if prompt_meta else {}
    pressure_ratios = [float(item.get("pressure_ratio")) for item in prompt_meta if item.get("pressure_ratio") is not None]
    max_pressure_ratio = max(pressure_ratios, default=None)
    if max_pressure_ratio is None:
        max_pressure_tier = last_meta.get("pressure_tier")
    elif max_pressure_ratio >= 0.95:
        max_pressure_tier = "tier3_summary"
    elif max_pressure_ratio >= 0.80:
        max_pressure_tier = "tier2_prune"
    elif max_pressure_ratio >= 0.60:
        max_pressure_tier = "tier1_snip"
    else:
        max_pressure_tier = "tier0_observe"
    verification = state.get("verification_status") == VERIFICATION_PASSED
    return {
        "scenario": scenario,
        "level": level["id"],
        "level_name": level["name"],
        "mode": label,
        "model": client.model,
        "context_window_override": client.context_window,
        "seed_history_chars": initial_history_chars,
        "seed_history_events": initial_history_events,
        "main_input_tokens": main_input,
        "main_output_tokens": main_output,
        "compaction_input_tokens": compaction_input,
        "compaction_output_tokens": compaction_output,
        "total_input_tokens": main_input + compaction_input,
        "total_output_tokens": main_output + compaction_output,
        "actual_usage_complete": bool(main_usage) and all("input_tokens" in item for item in main_usage),
        "pressure_tier": max_pressure_tier,
        "pressure_ratio": max_pressure_ratio,
        "context_compacted": any(bool(item.get("context_compacted")) for item in prompt_meta),
        "compaction_modes": [
            event.get("summary_mode", "unknown")
            for event in events
            if event.get("event") == "context_compacted"
        ],
        "status": state.get("status"),
        "stop_reason": state.get("stop_reason"),
        "attempts": state.get("attempts"),
        "tool_steps": state.get("tool_steps"),
        "verification_passed": verification,
        "verification_status": state.get("verification_status"),
        "answer": answer,
        "error": error,
        "run_dir": str(run_dir) if run_dir else "",
    }


def pair_summary(baseline: dict, governed: dict) -> dict:
    b = int(baseline.get("total_input_tokens", 0) or 0)
    g = int(governed.get("total_input_tokens", 0) or 0)
    reduction = ((b - g) / b * 100.0) if b > 0 and governed.get("actual_usage_complete") else None
    return {
        "scenario": baseline["scenario"],
        "level": baseline["level"],
        "level_name": baseline["level_name"],
        "baseline": baseline,
        "governed": governed,
        "input_token_reduction": (b - g) if reduction is not None else None,
        "input_token_reduction_pct": reduction,
        "verifier_both_passed": bool(baseline.get("verification_passed") and governed.get("verification_passed")),
    }


def render_markdown(pairs: list[dict], started_at: str, finished_at: str) -> str:
    lines = [
        "# DeepSeek Flash 上下文治理真实评测",
        "",
        f"- 模型：`deepseek-flash`；运行时间：{started_at} ~ {finished_at}",
        "- 口径：治理前/后使用同一场景、同一压力级别、同一真实 provider；治理后总输入 Token 包含压缩调用 Token。",
        "- 压力级别由评测用 context-window override 控制，服务器调用仍为真实 DeepSeek Flash，usage 来自 provider 返回值。",
        "",
        "| 场景 | 压力 | 治理前输入 | 治理后主调用 | 治理后压缩调用 | 治理后总输入 | 降低 Token | 降低比例 | 前/后 Verifier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for pair in pairs:
        b = pair["baseline"]
        g = pair["governed"]
        pct = pair["input_token_reduction_pct"]
        lines.append(
            "| {scenario} | {level} {name} | {b} | {gm} | {gc} | {gt} | {red} | {pct} | {bp}/{gp} |".format(
                scenario=b["scenario"], level=b["level"], name=b["level_name"],
                b=b["total_input_tokens"], gm=g["main_input_tokens"], gc=g["compaction_input_tokens"],
                gt=g["total_input_tokens"], red=pair["input_token_reduction"] if pct is not None else "N/A",
                pct=f"{pct:.2f}%" if pct is not None else "N/A",
                bp="PASS" if b["verification_passed"] else "FAIL",
                gp="PASS" if g["verification_passed"] else "FAIL",
            )
        )
    valid = [p for p in pairs if p["input_token_reduction_pct"] is not None]
    base_total = sum(p["baseline"]["total_input_tokens"] for p in valid)
    gov_total = sum(p["governed"]["total_input_tokens"] for p in valid)
    weighted = ((base_total - gov_total) / base_total * 100.0) if base_total else None
    mean = (sum(p["input_token_reduction_pct"] for p in valid) / len(valid)) if valid else None
    lines.extend(
        [
            "",
            f"有效配对：{len(valid)}/{len(pairs)}；加权平均降低：{weighted:.2f}%" if weighted is not None else "有效配对不足，无法计算加权平均。",
            f"等权平均每组降低：{mean:.2f}%" if mean is not None else "等权平均：N/A。",
            f"Verifier 双通过组数：{sum(1 for p in pairs if p['verifier_both_passed'])}/{len(pairs)}。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    env = read_env(ROOT / ".env")
    if not env.get("PICO_DEEPSEEK_API_KEY"):
        raise SystemExit("PICO_DEEPSEEK_API_KEY is missing from .env")
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    pairs = []
    for scenario, _label, scenario_key in SCENARIOS:
        for level in PRESSURE_LEVELS:
            base = run_one(env, output_root, scenario_key, level, governed=False)
            governed = run_one(env, output_root, scenario_key, level, governed=True)
            pairs.append(pair_summary(base, governed))
            print(json.dumps({"scenario": scenario, "level": level["id"], "baseline": base, "governed": governed}, ensure_ascii=False), flush=True)
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    valid = [p for p in pairs if p["input_token_reduction_pct"] is not None]
    result = {
        "schema_version": "real-context-eval-v1",
        "model": "deepseek-flash",
        "provider_base_url": env.get("PICO_DEEPSEEK_API_BASE", "https://api.deepseek.com/anthropic"),
        "started_at": started_at,
        "finished_at": finished_at,
        "pairs": pairs,
        "summary": {
            "pair_count": len(pairs),
            "valid_pair_count": len(valid),
            "verifier_both_passed_count": sum(1 for p in pairs if p["verifier_both_passed"]),
            "baseline_verifier_passed_count": sum(1 for p in pairs if p["baseline"]["verification_passed"]),
            "governed_verifier_passed_count": sum(1 for p in pairs if p["governed"]["verification_passed"]),
            "weighted_reduction_pct": (
                (sum(p["baseline"]["total_input_tokens"] for p in valid) - sum(p["governed"]["total_input_tokens"] for p in valid))
                / max(1, sum(p["baseline"]["total_input_tokens"] for p in valid))
                * 100
                if valid else None
            ),
            "unweighted_mean_reduction_pct": (
                sum(p["input_token_reduction_pct"] for p in valid) / len(valid) if valid else None
            ),
        },
    }
    (output_root / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "results.md").write_text(render_markdown(pairs, started_at, finished_at), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
