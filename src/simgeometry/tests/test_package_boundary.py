"""Keep geometry importable after copying it into a separate package."""

from __future__ import annotations

import ast
from pathlib import Path


def test_geometry_has_no_picnic_imports() -> None:
    source_root = Path(__file__).resolve().parents[1]
    for path in source_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert not (node.module or "").startswith("picnic"), path
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("picnic") for alias in node.names
                ), path
