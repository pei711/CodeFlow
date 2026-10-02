import os
from unittest.mock import patch

from codeflow import CodeFlow
from codeflow.config import provider_env
from codeflow.features import memory
from codeflow.workspace import workspace_state_path


def _state(*texts):
    return {
        "turn_summaries": [
            {"text": text, "turn_id": f"turn-{index}", "created_at": f"2026-09-{index + 1:02d}T00:00:00Z"}
            for index, text in enumerate(texts)
        ]
    }


def test_bm25_ranks_chinese_terms_and_exact_code_identifiers():
    documents = [
        "讨论历史压缩和上下文窗口预算",
        "修改 context_budget_chars 的计算方法",
        "分析文件读写工具的安全权限",
    ]

    hits = memory._bm25_rank(documents, "context_budget_chars 的计算")

    assert hits[0][0] == 1
    assert hits[0][1] > 0


def test_rrf_rewards_a_summary_found_by_both_retrievers():
    summaries = memory.normalize_memory_state(_state("向量首位", "两路都命中", "关键词首位"))["turn_summaries"]

    results = memory._rrf_fuse(
        [(0, 0.9), (1, 0.8)],
        [(2, 8.0), (1, 7.0)],
        summaries,
        limit=3,
    )

    assert results[0]["text"] == "两路都命中"
    assert results[0]["retrieval"]["vector"]["rank"] == 2
    assert results[0]["retrieval"]["bm25"]["rank"] == 2


def test_rrf_collapses_duplicate_summary_text():
    summaries = memory.normalize_memory_state(
        _state("same historical turn", " same   historical turn ", "different turn")
    )["turn_summaries"]

    results = memory._rrf_fuse([(0, 0.9), (1, 0.8), (2, 0.7)], [(1, 8.0)], summaries, limit=3)

    normalized = [" ".join(item["text"].split()).casefold() for item in results]
    assert normalized.count("same historical turn") == 1
    assert len(set(normalized)) == len(results)


def test_retrieval_falls_back_to_tfidf_and_keeps_top_three(monkeypatch):
    monkeypatch.setattr(memory, "_semantic_rank", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")))
    state = _state(
        "之前讨论上下文预算与历史压缩",
        "之前讨论模型请求失败后的恢复",
        "之前讨论文件工具的安全权限",
        "之前讨论上下文窗口的 token 估算",
    )

    results = memory.retrieve_turn_summaries(state, "上下文预算", limit=3)

    assert len(results) == 3
    assert results[0]["retrieval"]["vector_mode"] == "tfidf_fallback"
    assert results[0]["retrieval"]["bm25"] is not None


def test_legacy_summaries_get_stable_ids_when_normalized():
    state = _state("旧 session 里的轮次摘要")

    first = memory.normalize_memory_state(state)["turn_summaries"]
    second = memory.normalize_memory_state({"turn_summaries": first})["turn_summaries"]

    assert first[0]["summary_id"]
    assert second[0]["summary_id"] == first[0]["summary_id"]


def test_new_public_name_and_environment_keys_keep_legacy_compatibility():
    assert CodeFlow is CodeFlow
    with patch.dict(os.environ, {"CODEFLOW_PROVIDER": "openai", "PICO_PROVIDER": "deepseek"}, clear=True):
        assert provider_env("PICO_PROVIDER") == "openai"
    with patch.dict(os.environ, {"PICO_PROVIDER": "deepseek"}, clear=True):
        assert provider_env("PICO_PROVIDER") == "deepseek"


def test_workspace_state_path_prefers_existing_data_and_new_name_for_new_workspaces(tmp_path):
    assert workspace_state_path(tmp_path, "sessions") == tmp_path / ".codeflow" / "sessions"
    legacy = tmp_path / ".pico" / "sessions"
    legacy.mkdir(parents=True)
    assert workspace_state_path(tmp_path, "sessions") == legacy
    current = tmp_path / ".codeflow" / "sessions"
    current.mkdir(parents=True)
    assert workspace_state_path(tmp_path, "sessions") == current
