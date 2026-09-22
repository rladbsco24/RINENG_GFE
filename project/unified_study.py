"""Isolated Appendix G generation used by the standalone notebook."""
from pathlib import Path
import json
import os
import subprocess
import sys

G_STEMS = (
    "Appendix_0_ArrayLayouts", "Appendix_1A_Pressure_Arrays",
    "Appendix_1B_Pressure_Frequencies", "Appendix_2A_Decision_Arrays",
    "Appendix_2B_Decision_Frequencies", "Appendix_3A_ACS_Arrays",
    "Appendix_3B_ACS_Frequencies", "Appendix_4A_EquilibriumPressure_Arrays",
    "Appendix_4B_EquilibriumPressure_Frequencies",
)


def generate_generality(ctx, workers=4):
    root = Path(ctx.memo["notebook_root"])
    # Appendix population and sampling stay at the established smoke scale.
    requested_mode = "smoke"
    modes = ("smoke",)
    output = root / "unified_outputs" / "generality"
    def subprocess_path(path):
        text = str(path.resolve())
        prefix = chr(92) * 2 + "?" + chr(92)
        return prefix + text if os.name == "nt" and not text.startswith(prefix) else text
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONHASHSEED="0")
    projects = {}
    for mode in modes:
        command = [
            sys.executable,
            subprocess_path(root / "generality/revision_f1_f2/unified_generation.py"),
            "--mode", mode,
            "--cache-policy", ctx.memo.get("cache_policy", "resume"),
            "--output-root", subprocess_path(output),
            "--workers", str(int(workers)),
        ]
        subprocess.run(command, check=True, cwd=root, env=env)
        status = json.loads((output / mode / "generation_status.json").read_text(encoding="utf-8"))
        project = Path(status["active_project"]).resolve()
        allowed = {"smoke": {"generated_smoke", "resumed_smoke"},
                   "full": {"generated_full", "resumed_full"}}
        if status.get("state") != "completed" or status.get("evidence_mode") not in allowed[mode]:
            raise RuntimeError(f"Appendix G did not complete in {mode} mode")
        for stem in G_STEMS:
            for suffix in (".png", ".pdf"):
                path = project / "figures" / (stem + suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
        projects[mode] = project
    ctx.memo["generality_projects"] = {key: str(value) for key, value in projects.items()}
    if "smoke" in projects:
        ctx.memo["generality_paper_smoke_project"] = str(projects["smoke"])
    project = projects[requested_mode]
    ctx.memo["generality_project"] = str(project)
    return project
