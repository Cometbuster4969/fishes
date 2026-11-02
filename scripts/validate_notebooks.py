#!/usr/bin/env python3
"""Static validation of the generated notebooks (no execution, no GPU, no data).

Checks
  1. every notebook is valid nbformat-4 JSON with unique cell ids,
  2. every code cell compiles as Python,
  3. no cell still contains the old `__import__(...)` hacks or leftover stubs,
  4. every `src.*` symbol imported in a notebook actually exists (AST import check),
  5. banned patterns: from-scratch initialisation, hard-coded /kaggle paths outside the
     config, and any `pretrained=False`.

Run: .venv/bin/python scripts/validate_notebooks.py
"""
from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
BANNED = [
    (r"pretrained\s*=\s*False", "pretrained=False (from-scratch) is forbidden"),
    (r"weights\s*=\s*None", "weights=None (random init) is forbidden"),
    (r"yolo\w*\.yaml", "building an Ultralytics net from a .yaml means random init"),
    (r"__import__\(", "old __import__ hack"),
    (r"NotImplementedError", "leftover stub"),
    (r"/kaggle/input/[a-z]", "dataset paths must come from configs/kaggle.yaml"),
    (r"np\.random\.seed\(\d+\)", "use cfg.seed / set_seed instead of a literal seed"),
]


def check_module_symbols(tree: ast.AST, nb_name: str, problems: list[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            try:
                mod = importlib.import_module(node.module)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{nb_name}: cannot import {node.module} ({exc})")
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if not hasattr(mod, alias.name):
                    problems.append(f"{nb_name}: {node.module}.{alias.name} does not exist")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    problems: list[str] = []
    notebooks = sorted(NB_DIR.glob("0*.ipynb"))
    if len(notebooks) != 6:
        problems.append(f"expected 6 notebooks, found {len(notebooks)}")

    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text())
        if nb.get("nbformat") != 4:
            problems.append(f"{nb_path.name}: nbformat is not 4")
        ids = [c.get("id") for c in nb["cells"]]
        if len(ids) != len(set(ids)):
            problems.append(f"{nb_path.name}: duplicate cell ids")
        n_code = 0
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell["source"])
            if cell["cell_type"] == "code":
                n_code += 1
                try:
                    tree = ast.parse(src)
                except SyntaxError as exc:
                    problems.append(f"{nb_path.name} cell {i}: SyntaxError {exc.msg} "
                                    f"(line {exc.lineno})")
                    continue
                check_module_symbols(tree, f"{nb_path.name} cell {i}", problems)
            for pattern, why in BANNED:
                if re.search(pattern, src):
                    # /kaggle/working is fine; only /kaggle/input dataset paths are banned
                    if pattern == r"/kaggle/input/[a-z]" and "input_root" in src:
                        continue
                    problems.append(f"{nb_path.name} cell {i}: {why}: "
                                    f"{re.search(pattern, src).group(0)!r}")
        if n_code < 5:
            problems.append(f"{nb_path.name}: only {n_code} code cells")

    print(f"checked {len(notebooks)} notebooks")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
