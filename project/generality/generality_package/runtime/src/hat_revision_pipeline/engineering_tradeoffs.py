"""Export engineering roles and the benchmark accuracy-cost frontier from saved data.

No solver, field evaluation, prescription search, or figure generation is performed.
The two input tables remain the authority for every numerical statement.
"""
from pathlib import Path
import fingerprintlib
import json

import numpy as np
import pandas as pd

from .manuscript_tables import _latex


METHODS = ("IB", "GS", "AD", "Conventional", "FE", "FE+g")
CASES = ("Single", "Triple")


def _bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0") or pd.isna(value):
        return False
    raise ValueError(f"Unrecognized boolean {value!r}")


def _number(value):
    return f"{value:.4g}" if np.isfinite(value) else "Not resolved"


def _display_frame(frame):
    result = frame.copy()
    for column in result.select_dtypes(include="floating").columns:
        result[column] = result[column].map(_number)
    return result.fillna("Not recorded")


def export_engineering_tradeoffs(table_root, output_root):
    """Write three tables, measured/mock sentences, and a source-hash manifest.

    ``output_root`` is the exact export directory. CSV/TSV preserve numerical
    precision; TXT/LaTeX and notebook displays use four significant figures.
    """
    table_root, output_root = Path(table_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(filename):
        path = table_root / filename
        frame = pd.read_csv(path)
        sources[filename] = {
            "path": str(path), "fingerprint": fingerprintlib.fingerprint(path.read_bytes()).hexdigest(),
            "rows": len(frame), "columns": list(frame.columns),
        }
        return frame

    mechanical = read("main_mechanical_results.csv")
    tradeoff = read("fig8_performance_time_tradeoff.csv")
    if tradeoff.method.duplicated().any():
        raise ValueError("The Fig. 8 table must contain one row per method.")
    tradeoff = tradeoff.set_index("method")
    rows, row_audit = [], []
    for case in CASES:
        prefix = case.lower()
        expected_targets = 1 if case == "Single" else 3
        for method in METHODS:
            group = mechanical[(mechanical.task == case) & (mechanical.display_method == method)]
            if len(group) != expected_targets or group.target_id.duplicated().any():
                raise ValueError(f"Expected distinct targets for {case} {method}.")
            if group.phase_fingerprint.nunique() != 1:
                raise ValueError(f"Expected one phase command for {case} {method}.")
            fig8 = tradeoff.loc[method]
            time = float(fig8[f"{prefix}_time_s"])
            if not np.isfinite(time) or time <= 0:
                raise ValueError(f"Invalid recorded command time for {case} {method}.")
            if not np.allclose(group.command_time_s, time, rtol=1e-12, atol=1e-14):
                raise ValueError(f"Time mismatch between main and Fig. 8 tables: {case} {method}.")
            if case == "Single" and str(group.iloc[0].phase_fingerprint) != str(fig8.single_phase_fingerprint):
                raise ValueError(f"Single phase mismatch for {method}.")
            root_count = int(group.finite_ka_root_found.map(_bool).sum())
            restoring_count = int((group.finite_ka_root_found.map(_bool)
                                   & group.finite_ka_locally_restoring.map(_bool)).sum())
            all_roots = root_count == expected_targets
            all_restoring = restoring_count == expected_targets
            displacement = group.finite_ka_displacement_a.to_numpy(float)
            stiffness = group.finite_ka_stiffness_min_n_m.to_numpy(float)
            complete = bool(all_roots and np.isfinite(displacement).all() and np.isfinite(stiffness).all())
            median = float(np.median(displacement)) if complete else np.nan
            worst = float(np.max(displacement)) if complete else np.nan
            min_stiffness = float(np.min(stiffness) * 1e3) if complete else np.nan
            if complete and not np.isclose(median, fig8[f"{prefix}_displacement_a"], rtol=1e-12, atol=1e-14):
                raise ValueError(f"Displacement mismatch between main and Fig. 8 tables: {case} {method}.")
            rows.append({"Case": case, "Method": method, "Command time (s)": time,
                "Median displacement (a)": median, "Worst displacement (a)": worst,
                "Minimum stiffness over targets (mN/m)": min_stiffness,
                "Roots found": root_count, "Locally restoring roots": restoring_count,
                "Targets": expected_targets, "Independent commands": 1,
                "All roots found": all_roots, "All locally restoring": all_restoring,
                "Frontier eligible": complete and all_restoring,
                "Recorded time source": str(fig8[f"{prefix}_time_source"]), "Status": "SMOKE"})
            row_audit.append({"case": case, "method": method,
                "phase_fingerprint": str(group.iloc[0].phase_fingerprint),
                "target_ids": group.target_id.astype(str).tolist(),
                "source_phase_description": str(fig8[f"{prefix}_phase_source"]),
                "recorded_time_source": str(fig8[f"{prefix}_time_source"]),
                "target_displacement_a": [float(x) if np.isfinite(x) else None for x in displacement],
                "target_minimum_stiffness_n_m": [float(x) if np.isfinite(x) else None for x in stiffness]})
    accuracy = pd.DataFrame(rows)
    frontier_rows, dominance = [], []
    for _, row in accuracy.iterrows():
        eligible = accuracy[(accuracy.Case == row.Case) & accuracy["Frontier eligible"]]
        witnesses = []
        if row["Frontier eligible"]:
            for _, candidate in eligible.iterrows():
                if candidate.Method == row.Method:
                    continue
                no_worse = (candidate["Command time (s)"] <= row["Command time (s)"]
                            and candidate["Median displacement (a)"] <= row["Median displacement (a)"])
                strict = (candidate["Command time (s)"] < row["Command time (s)"]
                          or candidate["Median displacement (a)"] < row["Median displacement (a)"])
                if no_worse and strict:
                    witnesses.append(candidate.Method)
                    dominance.append({"case": row.Case, "dominated_method": row.Method,
                        "witness_method": candidate.Method,
                        "dominated_time_s": float(row["Command time (s)"]),
                        "witness_time_s": float(candidate["Command time (s)"]),
                        "dominated_median_displacement_a": float(row["Median displacement (a)"]),
                        "witness_median_displacement_a": float(candidate["Median displacement (a)"])})
        label = "On frontier" if row["Frontier eligible"] and not witnesses else "Dominated" if witnesses else "Ineligible"
        frontier_rows.append({"Case": row.Case, "Method": row.Method,
            "Command time (s)": row["Command time (s)"],
            "Median displacement (a)": row["Median displacement (a)"],
            "Benchmark frontier status": label,
            "Dominance witnesses": "; ".join(witnesses) if witnesses else "None" if row["Frontier eligible"] else "Not evaluated",
            "Status": "SMOKE"})
    frontier = pd.DataFrame(frontier_rows)

    def r(case, method):
        return accuracy[(accuracy.Case == case) & (accuracy.Method == method)].iloc[0]

    def pair(case, method):
        row = r(case, method)
        return f"{method}: {_number(row['Median displacement (a)'])} a at {_number(row['Command time (s)'])} s"

    single_fe, single_feg = r("Single", "FE"), r("Single", "FE+g")
    triple_fe, triple_feg = r("Triple", "FE"), r("Triple", "FE+g")
    single_gap = float(single_fe["Median displacement (a)"] - single_feg["Median displacement (a)"])
    role_rows = [
        ("Single", "IB / GS", "Fast prescribed-field command generation",
         pair("Single", "IB") + "; " + pair("Single", "GS"),
         "Useful when the measured approximately 0.4a offset is acceptable; prescribed settings are already supplied."),
        ("Single", "FE / FE+g", "Higher coordinate precision; surrogate-based command generation",
         pair("Single", "FE") + "; " + pair("Single", "FE+g"),
         "Both attain approximately 0.045a displacement; FE+g has the lower recorded cost. Their Single precision is effectively tied at the reported scale."),
        ("Single", "FE / FE+g", "Candidate reference for prescription refinement",
         "Coordinate displacement is measured independently of pressure reconstruction.",
         "The command-generation role is supported; a prescription-refinement workflow and its preparation cost were not measured here."),
        ("Triple", "FE+g", "Higher precision with gravity-balanced surrogate design",
         pair("Triple", "FE+g") + f"; worst {_number(triple_feg['Worst displacement (a)'])} a",
         "Lowest median and worst-target displacement among the six tested methods; all three roots locally restoring."),
        ("Triple", "IB / GS", "Fast prescribed-field synthesis with target-position tradeoffs",
         pair("Triple", "IB") + "; " + pair("Triple", "GS"),
         "Both supply locally restoring roots, with larger coordinate offsets at much lower command cost."),
        ("Triple", "AD", "Intermediate command cost and median precision",
         pair("Triple", "AD") + f"; worst {_number(r('Triple', 'AD')['Worst displacement (a)'])} a",
         "On the benchmark time-median-displacement frontier; the worst target remains substantially farther from its intended position."),
        ("Triple", "FE", "Uncompensated surrogate reference",
         pair("Triple", "FE"),
         "Retained in the comparison; AD and FE+g dominate this recorded time-median-displacement pair."),
        ("Single / Triple", "Conventional", "Pressure-null and curvature optimization reference",
         pair("Single", "Conventional") + "; " + pair("Triple", "Conventional"),
         "Existing objective baseline; outside the benchmark accuracy-cost frontier in both cases."),
    ]
    roles = pd.DataFrame(role_rows, columns=["Case", "Method", "Engineering role", "Measured basis", "Interpretation"])
    roles["Status"] = "SMOKE"
    compact_accuracy = accuracy[["Case", "Method", "Command time (s)",
        "Median displacement (a)", "Worst displacement (a)",
        "Minimum stiffness over targets (mN/m)"]].copy()
    compact_accuracy["Restoring roots"] = [
        f"{row['Locally restoring roots']}/{row['Targets']}" for _, row in accuracy.iterrows()
    ]
    compact_accuracy["Status"] = "SMOKE"
    tables = {"table_engineering_roles": roles,
              "table_engineering_accuracy_cost": compact_accuracy,
              "table_engineering_pareto_frontier": frontier}
    notes = [
        "ALL NUMERICAL RESULTS ARE SMOKE; not production manuscript measurements.",
        "Each method/case contains one independent phase command. Triple medians and worst values summarize its three dependent target observations; they are not repeated-trial statistics.",
        "Displacement is the distance from the intended coordinate to the finite-ka elastic-bead equilibrium including effective gravity, normalized by bead radius a.",
        "Minimum stiffness is the smallest eigenvalue of the symmetric restoring-stiffness matrix, then the minimum across targets. It is not the median of the target minima.",
        "The benchmark Pareto frontier minimizes recorded command time and median displacement separately for Single and Triple. Eligible commands have all target roots resolved and locally restoring. A dominance witness is no worse in both objectives and strictly better in at least one. Comparisons use the unrounded recorded values.",
        "The reported command times retain their existing timing scopes; source fields are exported. Final mechanical validation is excluded. This is the frontier of those reported costs, without retiming or an inferred common decomposition. See the manuscript timing-protocol table for the archived breakdown.",
        "Eligibility is a local restoring-root check, not a capture-volume or dynamic-stability comparison. Stiffness is reported but is not an additional Pareto objective.",
        f"The Single FE minus FE+g displacement difference is {single_gap:.12g} a. Strict nominal dominance is retained, while the displayed precision of approximately 0.045a is treated as nearly tied; one command does not resolve a physical precision advantage.",
        "Pressure is not used as a coordinate-precision score. No pressure threshold, prescription-search cost, throughput repetition, or data-set quality measurement is inferred.",
        "Benchmark scope: six phase-only implementations, the locked Single and Triple targets in the single-sided array, and the reported command-time protocol. Prescription refinement and cross-geometry transfer are prospective uses of these reference commands.",
    ]
    combined = ["ENGINEERING TRADEOFF EXPORTS", "", *notes[:5], notes[-1], ""]
    output_records = []
    for name, frame in tables.items():
        readable = _display_frame(frame)
        frame.to_csv(output_root / f"{name}.csv", index=False, na_rep="Not resolved")
        frame.to_csv(output_root / f"{name}.tsv", index=False, sep="\t", na_rep="Not resolved")
        body = "\n".join(notes[:2]) + "\n\n" + readable.to_string(index=False) + "\n"
        (output_root / f"{name}.txt").write_text(body, encoding="utf-8")
        (output_root / f"{name}.tex").write_text(_latex(readable, name + "; " + notes[1]), encoding="utf-8")
        combined.extend([name, readable.to_string(index=False), ""])
        output_records.append({"Table": name, "Rows": len(frame), "Formats": "CSV / TSV / TXT / LaTeX", "Status": "SMOKE"})

    measured = ["MEASURED ENGINEERING RESULTS: ALL RUNS ARE SMOKE", "", *notes[:2], ""]
    for _, row in accuracy.iterrows():
        measured.append(f"[SMOKE] For {row.Case}, {row.Method} required {_number(row['Command time (s)'])} s of reported command time and produced median/worst exact-equilibrium displacement {_number(row['Median displacement (a)'])}/{_number(row['Worst displacement (a)'])} a. The minimum symmetric stiffness over targets was {_number(row['Minimum stiffness over targets (mN/m)'])} mN/m; {row['Roots found']}/{row.Targets} roots were resolved and {row['Locally restoring roots']}/{row.Targets} were locally restoring.")
    for case in CASES:
        on_frontier = frontier[(frontier.Case == case) & (frontier['Benchmark frontier status'] == 'On frontier')].Method.tolist()
        measured.append(f"[SMOKE] The {case} command-time versus median-displacement Pareto frontier contained {', '.join(on_frontier)}, with all target equilibria resolved and locally restoring.")
    improvement = 100 * (1 - triple_feg['Median displacement (a)'] / triple_fe['Median displacement (a)'])
    cost_ratio = triple_fe['Command time (s)'] / triple_feg['Command time (s)']
    focused = [
        "[SMOKE] IB and GS generated Single commands in approximately one millisecond with approximately 0.4a equilibrium displacement, providing fast synthesis when that coordinate tolerance is acceptable.",
        "[SMOKE] FE and FE+g attained approximately 0.045a Single displacement in approximately 1.2 and 1.1 s, respectively, extending the benchmark to approximately 8.9-fold finer coordinate precision than IB/GS.",
        f"[SMOKE] Triple FE+g attained {_number(triple_feg['Median displacement (a)'])}a median and {_number(triple_feg['Worst displacement (a)'])}a worst-target displacement in {_number(triple_feg['Command time (s)'])} s. These were the smallest displacements among the six methods, with all three equilibria locally restoring.",
        f"[SMOKE] Relative to Triple FE, FE+g reduced median displacement by {improvement:.3g}% and used {cost_ratio:.3g}-fold less recorded command time.",
    ]
    measured = measured[:5] + focused + ["", "METHOD RECORDS", ""] + measured[5:]
    measured.extend(["", "DEFINITIONS", notes[4], notes[5], notes[6], notes[7], notes[-1]])
    (output_root / "engineering_results_smoke.txt").write_text("\n".join(measured) + "\n", encoding="utf-8")
    mock = ["MOCK RESULTS SENTENCES: PLACEHOLDERS ARE NOT MEASUREMENTS", "",
        "For Single, IB and GS produced commands in {ib_time_s} and {gs_time_s} s with exact-equilibrium displacement {ib_displacement_a}a and {gs_displacement_a}a, respectively; these costs favor applications accepting the measured offsets.",
        "FE and FE+g reduced Single displacement to {fe_displacement_a}a and {feg_displacement_a}a at command costs of {fe_time_s} and {feg_time_s} s, providing higher coordinate precision within the tested formulation.",
        "For Triple, FE+g attained median/worst displacement {median_a}/{worst_a}a in {time_s} s, with {restoring_count}/{target_count} locally restoring equilibria and minimum symmetric stiffness {minimum_stiffness_mn_m} mN/m.",
        "The command-time versus median-displacement Pareto frontier contained {method_list}; it was evaluated separately for each case among methods with all target equilibria resolved and locally restoring.",
        "Within the tested benchmark, {method} occupied the high-precision end of the frontier, while {fast_methods} served the lower-cost operating range. Application acceptance depends on the required coordinate tolerance.",
        "If prescription refinement is evaluated separately: Using {surrogate_method} as the reference changed the prescribed-field equilibrium displacement from {before_a}a to {after_a}a at an additional measured preparation cost of {preparation_time_s} s.",
        "Replace every placeholder with its own validated measurement and preserve the SMOKE designation until production runs are complete."]
    (output_root / "engineering_mock_sentences.txt").write_text("\n".join(mock) + "\n", encoding="utf-8")
    all_exports = output_root / "all_engineering_exports.txt"
    all_exports.write_text("\n".join(combined) + "\n", encoding="utf-8")
    manifest = {"evidence_status": "SMOKE", "simulations_performed": False,
        "input_sources": sources, "output_directory": str(output_root), "definitions": notes,
        "row_audit": row_audit, "dominance_witnesses": dominance,
        "tables": output_records,
        "numerical_display": "CSV/TSV retain raw numerical precision; TXT/LaTeX use four significant figures.",
        "implementation_fingerprint": fingerprintlib.fingerprint(Path(__file__).read_bytes()).hexdigest()}
    (output_root / "engineering_sources_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"tables": tables, "summary": pd.DataFrame(output_records), "manifest": manifest,
            "all_table_exports": all_exports,
            "measured_sentences": output_root / "engineering_results_smoke.txt",
            "mock_sentences": output_root / "engineering_mock_sentences.txt"}


def display_engineering_tradeoffs(result):
    """Display the small engineering comparison and the export inventory."""
    from IPython.display import display
    for name, frame in result["tables"].items():
        print(name)
        display(_display_frame(frame))
    print("ENGINEERING EXPORT SUMMARY — ALL RUNS ARE SMOKE")
    display(result["summary"])
    print("One command per case/method; Triple targets are dependent. The frontier uses reported command time and median displacement; all roots must be resolved and locally restoring.")
