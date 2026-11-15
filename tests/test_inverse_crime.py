"""Guard the inverse-crime separation: the reconstruction side must not reuse
the simulator sampling code (mlx_sr.sensor / mlx_sr.scene)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "mlx_sr"
FORBIDDEN = {"sensor", "scene"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = (node.module or "").split(".")
            mods.update(base)
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                mods.update(a.name.split("."))
    return mods


@pytest.mark.parametrize("module", ["operators.py", "reconstruct.py"])
def test_reconstruction_does_not_import_simulator(module):
    assert not (_imported_modules(PKG / module) & FORBIDDEN)
