"""Runtime acceptance checks for tasks that opt into a verification contract.

The benchmark evaluator has its own post-run verifier commands. This module is
intentionally smaller and safer: runtime checks are typed, workspace-scoped,
and never execute an arbitrary command supplied by a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VERIFIER_SCHEMA_VERSION = "runtime-v1"
SUPPORTED_CHECKS = {"file_exists", "file_contains", "file_not_contains"}


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    checks: tuple[dict, ...]
    failures: tuple[str, ...]

    def to_dict(self):
        return {
            "passed": self.passed,
            "checks": [dict(item) for item in self.checks],
            "failures": list(self.failures),
        }

    def feedback(self):
        if self.passed:
            return "Verification passed: all acceptance checks are satisfied."
        lines = ["Verification failed. Fix these acceptance checks before returning a final answer:"]
        lines.extend(f"- {failure}" for failure in self.failures)
        return "\n".join(lines)


class RuntimeVerifier:
    """Evaluate a trusted, typed acceptance contract inside the workspace."""

    def __init__(self, workspace_root, contract):
        self.root = Path(workspace_root).resolve()
        self.contract = self._normalize_contract(contract)

    @staticmethod
    def _normalize_contract(contract):
        if not isinstance(contract, dict):
            raise TypeError("verification_contract must be a mapping")
        checks = contract.get("checks", [])
        if not isinstance(checks, (list, tuple)) or not checks:
            raise ValueError("verification_contract.checks must be a non-empty list")
        normalized = []
        for index, check in enumerate(checks):
            if not isinstance(check, dict):
                raise TypeError(f"verification check {index} must be a mapping")
            check_type = str(check.get("type", "")).strip()
            if check_type not in SUPPORTED_CHECKS:
                raise ValueError(
                    f"unsupported verification check {check_type!r}; "
                    f"supported checks: {', '.join(sorted(SUPPORTED_CHECKS))}"
                )
            path = str(check.get("path", "")).strip()
            if not path:
                raise ValueError(f"verification check {index} is missing path")
            item = {"type": check_type, "path": path}
            if check_type != "file_exists":
                if "text" not in check:
                    raise ValueError(f"verification check {index} is missing text")
                item["text"] = str(check["text"])
            normalized.append(item)
        return {
            "schema_version": str(contract.get("schema_version", VERIFIER_SCHEMA_VERSION)),
            "name": str(contract.get("name", "runtime acceptance contract")).strip(),
            "checks": normalized,
        }

    def _path(self, raw_path):
        path = Path(raw_path)
        resolved = (path if path.is_absolute() else self.root / path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"verification path escapes workspace: {raw_path}") from exc
        return resolved

    def prompt_text(self):
        lines = ["Acceptance contract (system-enforced; verify before claiming completion):"]
        for check in self.contract["checks"]:
            check_type = check["type"]
            if check_type == "file_exists":
                lines.append(f"- file exists: {check['path']}")
            elif check_type == "file_contains":
                lines.append(f"- {check['path']} contains: {check['text']}")
            else:
                lines.append(f"- {check['path']} does not contain: {check['text']}")
        return "\n".join(lines)

    def verify(self):
        checks = []
        failures = []
        for check in self.contract["checks"]:
            path = self._path(check["path"])
            passed = False
            detail = ""
            try:
                if check["type"] == "file_exists":
                    passed = path.exists()
                    detail = "exists" if passed else "missing"
                else:
                    text = path.read_text(encoding="utf-8")
                    expected = check["text"]
                    if check["type"] == "file_contains":
                        passed = expected in text
                        detail = "found" if passed else "not found"
                    else:
                        passed = expected not in text
                        detail = "absent" if passed else "still present"
            except (OSError, UnicodeError) as exc:
                detail = f"read error: {exc}"

            checks.append({**check, "passed": passed, "detail": detail})
            if not passed:
                if check["type"] == "file_exists":
                    failures.append(f"{check['path']} must exist ({detail})")
                elif check["type"] == "file_contains":
                    failures.append(f"{check['path']} must contain {check['text']!r} ({detail})")
                else:
                    failures.append(f"{check['path']} must not contain {check['text']!r} ({detail})")

        return VerificationResult(passed=not failures, checks=tuple(checks), failures=tuple(failures))
