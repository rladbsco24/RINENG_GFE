"""Execute the standalone study notebook in the selected Python kernel."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-name", default="python3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / "RINENG_GFE_Reproduction.ipynb"
    output = root / "RINENG_GFE_Reproduction_executed.ipynb"
    notebook = nbformat.read(source, as_version=4)

    def checkpoint(cell, cell_index, **kwargs):
        print(f"Completed cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
        nbformat.write(notebook, output)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=args.kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(root)}},
        iopub_timeout=900,
        raise_on_iopub_timeout=True,
        on_cell_executed=checkpoint,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)


if __name__ == "__main__":
    main()
