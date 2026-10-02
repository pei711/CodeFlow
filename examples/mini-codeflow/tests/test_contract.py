import subprocess
import sys
from pathlib import Path

import mini_codeflow


def test_mini_codeflow_module_and_public_exports():
    assert mini_codeflow.CodeFlow is not None
    assert mini_codeflow.FakeModelClient is not None
    assert not hasattr(mini_codeflow, "MiniAgent")
    result = subprocess.run([sys.executable, "-m", "mini_codeflow", "--help"], capture_output=True, text=True, check=True)
    assert "Teaching-sized CodeFlow agent harness" in result.stdout


def test_readme_main_mapping_points_to_existing_files():
    repo_root = Path(__file__).resolve().parents[3]
    main_files = [
        "codeflow/cli.py",
        "codeflow/runtime.py",
        "codeflow/agent_loop.py",
        "codeflow/context_manager.py",
        "codeflow/providers/clients.py",
        "codeflow/tool_executor.py",
        "codeflow/tools.py",
        "codeflow/task_state.py",
        "codeflow/run_store.py",
        "codeflow/workspace.py",
    ]
    for path in main_files:
        assert (repo_root / path).exists()
