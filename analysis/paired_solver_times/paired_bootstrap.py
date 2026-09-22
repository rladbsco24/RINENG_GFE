"""Summarize all saved paired starts without executing scientific models or solvers."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
SOURCE = PACKAGE / "results/production_outputs/tables/fig2_paired_initialization_commands.csv"
RECORDS = HERE / "full_start_statistics.json"
METHODS = ("Conventional", "GFE")
with SOURCE.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
records = {}
integer_fields = ("seed", "pair_index", "iterations", "evaluations", "iteration_cap")
number_fields = (
    "wall_time_s", "command_time_s", "objective", "terminal_gradient_l2",
    "terminal_gradient_rms", "terminal_gradient_inf", "alpha_per_m",
)
boolean_fields = ("stationary_gtol", "native_success")
text_fields = (
    "method", "task", "population", "initial_phase_content_id", "phase_content_id",
    "optimizer", "evaluator_backend", "native_message", "run_mode",
)
for method in METHODS:
    selected = sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: int(row["pair_index"]),
    )
    records[method] = [
        {
            **{key: int(row[key]) for key in integer_fields},
            **{key: float(row[key]) for key in number_fields},
            **{key: row[key] == "True" for key in boolean_fields},
            **{key: row[key] for key in text_fields},
        }
        for row in selected
    ]
c, g = records["Conventional"], records["GFE"]
sample_size = len(c)
assert sample_size == len(g) == 100, "The full population must contain 100 pairs."
assert len(rows) == 2 * sample_size, "Unexpected or unpaired method records."
seeds = [row["seed"] for row in c]
assert seeds == [row["seed"] for row in g]
assert len(set(seeds)) == sample_size
for method in METHODS:
    assert [row["pair_index"] for row in records[method]] == list(range(sample_size))
    assert all(row["run_mode"] == "full" for row in records[method])
assert all(
    left["initial_phase_content_id"] == right["initial_phase_content_id"]
    for left, right in zip(c, g)
), "Each comparison must use the same initial phase vector."

statistics = {}
for method in METHODS:
    statistics[method] = {}
    for field in ("wall_time_s", "command_time_s", "iterations", "objective", "terminal_gradient_l2"):
        values = np.array([row[field] for row in records[method]], dtype=float)
        assert np.isfinite(values).all()
        quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 1], method="linear")
        statistics[method][field] = dict(zip(
            ("minimum", "q25", "median", "q75", "maximum"),
            map(float, quantiles),
        ))
        statistics[method][field]["iqr"] = float(quantiles[3] - quantiles[1])
        statistics[method][field]["mean"] = float(np.mean(values))
        statistics[method][field]["sample_sd"] = float(np.std(values, ddof=1))
    statistics[method]["stationary_count"] = sum(
        row["stationary_gtol"] for row in records[method]
    )
    statistics[method]["native_success_count"] = sum(
        row["native_success"] for row in records[method]
    )
    statistics[method]["precision_loss_count"] = sum(
        "precision loss" in row["native_message"] for row in records[method]
    )
times_c = np.array([row["wall_time_s"] for row in c])
times_g = np.array([row["wall_time_s"] for row in g])
assert (times_c > 0).all() and (times_g > 0).all()
indices = np.random.default_rng(20260906).integers(
    0, sample_size, size=(10000, sample_size)
)
summaries = {}
for name, values, units in (
    ("Conventional_to_GFE_ratio", times_c / times_g, "dimensionless"),
    ("Conventional_minus_GFE_difference", times_c - times_g, "s"),
):
    summaries[name] = {
        "estimate": float(np.median(values)),
        "percentile_95_interval": np.quantile(
            np.median(values[indices], axis=1), [0.025, 0.975], method="linear"
        ).tolist(),
        "units": units,
    }
source_metadata = {
    "source_run": "RINENG_GFE_COMPACT_FULL_FIGURE7_BFGS_20260920",
    "source_cohort": SOURCE.relative_to(PACKAGE).as_posix(),
    "sample_size": sample_size,
    "sampling_unit": "paired initial phase vector",
    "population": "all 100 completed central-target Conventional/GFE pairs",
    "timing_field": "wall_time_s",
    "timing_scope": "solver-only; objective/transfer construction and compact precomputation excluded",
}
RECORDS.write_text(json.dumps({
    **source_metadata,
    **records,
    "summary_timing_scopes": {
        "wall_time_s": "native solver-history interval; setup and compact precomputation excluded",
        "command_time_s": "complete command synthesis including initialization, transfer/objective construction, compact precomputation, solver invocation and phase extraction",
    },
    "summary_statistics": statistics,
}, indent=2) + "\n", encoding="utf-8")
result = {
    **source_metadata,
    "source_records": RECORDS.name,
    "source_command_pattern": "results/production_outputs/commands/Single/{Conventional,GFE}_seed{seed}_controlled*.json",
    "seed_ids": seeds,
    "method": {
        "estimands": "median paired time ratio and median paired time difference",
        "sampling_unit": "paired initial phase vector",
        "sample_size": sample_size,
        "resampling": f"{sample_size} paired indices with replacement",
        "replicates": 10000,
        "rng": "numpy.random.default_rng",
        "rng_seed": 20260906,
        "interval": "2.5th and 97.5th percentiles",
        "quantile_interpolation": "linear",
    },
    "paired_solver_time_records": [
        {"seed": seed, "Conventional_s": float(tc), "GFE_s": float(tg)}
        for seed, tc, tg in zip(seeds, times_c, times_g)
    ],
    "summaries": summaries,
}
(HERE / "paired_bootstrap.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summaries, indent=2))
