from codeflow.workspace_input import resolve_workspace_path


def test_resolve_workspace_path_accepts_quoted_directory(tmp_path):
    assert resolve_workspace_path(f'"{tmp_path}"') == tmp_path.resolve()


def test_resolve_workspace_path_maps_dropped_file_to_parent(tmp_path):
    selected_file = tmp_path / "README.md"
    selected_file.write_text("demo\n", encoding="utf-8")

    assert resolve_workspace_path(str(selected_file)) == tmp_path.resolve()


def test_resolve_workspace_path_rejects_natural_language(tmp_path):
    assert resolve_workspace_path("请分析这个仓库") is None
    assert resolve_workspace_path(str(tmp_path / "missing")) is None
