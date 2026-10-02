from pathlib import Path

import codeflow
from codeflow import (
    CodeFlow,
    SessionStore,
    WorkspaceContext,
    build_agent,
    build_arg_parser,
    build_welcome,
    main,
)


def test_public_api_exports_codeflow_names_without_old_brand_aliases():
    assert CodeFlow is not None
    assert SessionStore is not None
    assert WorkspaceContext is not None
    assert callable(build_agent)
    assert callable(build_arg_parser)
    assert callable(build_welcome)
    assert callable(main)
    assert not hasattr(codeflow, "MiniAgent")
    assert "MiniAgent" not in codeflow.__all__
    assert "CodeFlow" in codeflow.__all__


def test_build_agent_returns_codeflow(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    args = build_arg_parser().parse_args(["--cwd", str(tmp_path), "--approval", "auto"])

    agent = build_agent(args)

    assert isinstance(agent, CodeFlow)


def test_package_split_uses_codeflow_paths_only():
    from codeflow.evaluation.evaluator import BenchmarkEvaluator
    from codeflow.evaluation.metrics import run_context_ablation_v2
    from codeflow.features.memory import LayeredMemory
    from codeflow.providers.clients import FakeModelClient as ProviderFakeModelClient

    assert BenchmarkEvaluator is not None
    assert LayeredMemory is not None
    assert ProviderFakeModelClient is not None
    assert callable(run_context_ablation_v2)
    for legacy_module in ("evaluator.py", "metrics.py", "models.py", "memory.py"):
        assert not (Path("codeflow") / legacy_module).exists()
    assert (Path("codeflow") / "__init__.py").exists()
    assert (Path("codeflow") / "__main__.py").exists()


def test_packaging_discovers_codeflow_subpackages():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.packages.find]" in pyproject_text
    assert 'include = ["codeflow*"]' in pyproject_text
