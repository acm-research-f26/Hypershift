"""Regenerate the notebook from its percent-format source and optionally execute.

Run from the repository root:
    uv run --group notebooks python notebooks/build_hyperedge_tutorial.py --execute

Execution uses a real IPython kernel with the calling interpreter. Kernel and
runtime files stay in ignored local-data rather than modifying global kernels.
The source notebook stays output-free; executed copies also stay in local-data.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def _percent_cells(source):
    cells, kind, lines = [], None, []

    def _append():
        if kind is None:
            return
        content = "".join(lines).strip("\n") + "\n"
        cell = {"cell_type": kind, "metadata": {},
                "id": hashlib.sha256(f"{len(cells)}:{content}".encode()).hexdigest()[:12],
                "source": content.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in source.splitlines(keepends=True):
        if line.startswith("# %%"):
            _append()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind == "markdown":
            if not line.strip():
                lines.append("\n")
            elif line.startswith("# "):
                lines.append(line[2:])
            elif line.startswith("#"):
                lines.append(line[1:])
            else:
                raise ValueError("Markdown cells must be comments in percent source.")
        elif kind == "code":
            lines.append(line)
        elif line.strip():
            raise ValueError("Content precedes the first cell marker.")
    _append()
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "hyperedge_walkthrough.py"
    destination = source.with_suffix(".ipynb")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python (Hypershift .venv)",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": sys.version.split()[0]}},
        "cells": _percent_cells(source.read_text(encoding="utf-8")),
    }
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(notebook['cells'])} cells to {destination}", flush=True)
    if not args.execute:
        return

    executed_destination = root / "local-data" / "tutorial-artifacts" / "hyperedge_walkthrough.executed.ipynb"
    executed_destination.parent.mkdir(parents=True, exist_ok=True)

    import nbformat
    from nbclient import NotebookClient

    runtime = root / "local-data" / "tutorial-runtime"
    kernel = runtime / "kernels" / "hypershift-tutorial"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (Hypershift tutorial validation)", "language": "python",
    }), encoding="utf-8")
    environment = {"JUPYTER_PATH": str(runtime) + (os.pathsep + os.environ["JUPYTER_PATH"]
                                                  if os.environ.get("JUPYTER_PATH") else ""),
                   "JUPYTER_RUNTIME_DIR": str(runtime / "runtime"),
                   "IPYTHONDIR": str(runtime / "ipython"),
                   "MPLCONFIGDIR": str(runtime / "matplotlib")}
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    for key in ("JUPYTER_RUNTIME_DIR", "IPYTHONDIR", "MPLCONFIGDIR"):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(destination, as_version=4)
    nbformat.validate(nb)
    client = NotebookClient(nb, timeout=300, kernel_name="hypershift-tutorial",
                            resources={"metadata": {"path": str(root)}})

    def _progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Executing code cell {cell_index + 1}/{len(nb.cells)}", flush=True)

    client.on_cell_start = _progress
    try:
        client.execute()
    finally:
        nbformat.write(nb, executed_destination)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    nbformat.validate(nb)
    code = [cell for cell in nb.cells if cell.cell_type == "code"]
    if not all(cell.execution_count is not None for cell in code):
        raise RuntimeError("Not all code cells executed.")
    errors = [output for cell in code for output in cell.outputs if output.output_type == "error"]
    if errors:
        raise RuntimeError("Executed notebook contains error outputs.")
    print(f"Verified {len(code)} executed code cells without errors.", flush=True)
    print(f"Saved executed notebook to {executed_destination}", flush=True)


if __name__ == "__main__":
    main()
