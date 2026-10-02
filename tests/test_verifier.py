import pytest

from codeflow.verifier import RuntimeVerifier


def test_runtime_verifier_checks_workspace_contract(tmp_path):
    (tmp_path / "sample.txt").write_text("beta-locked\n", encoding="utf-8")
    verifier = RuntimeVerifier(
        tmp_path,
        {
            "name": "locked replacement",
            "checks": [
                {"type": "file_exists", "path": "sample.txt"},
                {"type": "file_contains", "path": "sample.txt", "text": "beta-locked"},
                {"type": "file_not_contains", "path": "sample.txt", "text": "\nbeta\n"},
            ],
        },
    )

    result = verifier.verify()

    assert result.passed is True
    assert all(check["passed"] for check in result.checks)


def test_runtime_verifier_returns_repairable_failures(tmp_path):
    (tmp_path / "sample.txt").write_text("beta\n", encoding="utf-8")
    verifier = RuntimeVerifier(
        tmp_path,
        {"checks": [{"type": "file_contains", "path": "sample.txt", "text": "beta-locked"}]},
    )

    result = verifier.verify()

    assert result.passed is False
    assert "must contain" in result.feedback()


def test_runtime_verifier_rejects_workspace_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes workspace"):
        RuntimeVerifier(
            tmp_path,
            {"checks": [{"type": "file_exists", "path": "../outside.txt"}]},
        ).verify()
