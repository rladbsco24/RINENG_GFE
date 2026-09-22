"""Produce Appendix B1/B2 from their numerical caches.

The former B3/B4 sensitivity figures are now C4/C5 and are generated separately
by sensitivity_b.render into the combined Appendix C sensitivity section.
"""
from pathlib import Path
import reproduce_solutions
import render_b12

ROOT = Path(__file__).resolve().parent

def render(root=ROOT):
    root = Path(root).resolve()
    manifest = reproduce_solutions.compute_nominal_missing(root)
    if len(manifest) != 8:
        raise RuntimeError(f"Appendix B requires eight solutions, got {len(manifest)}")
    return render_b12.render(root)

if __name__ == "__main__":
    render()
