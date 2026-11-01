"""Automated regression and compliance tests to prevent real personal/corporate data leakage in code and tests."""

import base64
import os
from pathlib import Path

# Base64-encoded forbidden signatures to prevent accidental literal matching
# during static code scans.
FORBIDDEN_SIGNATURES_B64 = [
    # Signature 1 (real signatory)
    "RElOSVMgREUgQUxNRUlEQQ==",
    # Signature 2 (real document number)
    "MjUwMzIwMTgwMzA1MTQ5MDAwMDAwMTcxMzM0NjI=",
    # Signature 3 (real page ID)
    "MjE3NzczNjY4OA==",
    # CRASHCRED
    "Q1JBU0hDUkVE",
    # Alberto Bezzarro
    "QWxiZXJ0byBCZXp6YXJybw==",
    # Fernando Cavalcanti
    "RmVybmFuZG8gQ2F2YWxjYW50aQ==",
    # 174.312.508-94
    "MTc0LjMxMi41MDgtOTQ=",
    # 47.344.286/0001-96
    "NDcuMzQ0LjI4Ni8wMDAxLTk2",
]


def test_no_real_sensitive_data_in_code_or_tests():
    """Verify that no real names, company names, process IDs or CPF/CNPJ from real investigation

    documents are present in src/, tests/ (excluding this test file), or configs/.
    """
    forbidden_terms = [
        base64.b64decode(b64).decode("utf-8") for b64 in FORBIDDEN_SIGNATURES_B64
    ]

    project_root = Path(__file__).resolve().parent.parent.parent
    directories_to_scan = [
        project_root / "src",
        project_root / "tests",
        project_root / "configs",
    ]

    violations = []

    for directory in directories_to_scan:
        if not directory.exists():
            continue
        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = Path(root) / file_name

                # Skip this test file and non-text artifacts
                if file_path.resolve() == Path(__file__).resolve():
                    continue
                if file_path.suffix not in {".py", ".yaml", ".yml", ".json", ".md", ".sh"}:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except (OSError, UnicodeDecodeError):
                    content = ""

                for term in forbidden_terms:
                    if term.lower() in content.lower():
                        violations.append(f"{file_path.relative_to(project_root)}: contains '{term}'")

    assert (
        not violations
    ), "Privacy compliance violation detected! Real entity data found in code/tests:\n" + "\n".join(
        violations
    )
