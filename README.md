# Holographic acoustic tweezers: reproduction data and code

This package contains the numerical implementation, fixed scientific inputs, and exported results for the accompanying manuscript. `RINENG_GFE_Paper_Figures.pdf` contains the supplied collection of 31 main and appendix figures. `FIGURES.md` maps its pages to the figure assets and manuscript sections.

## Run the study

Use Python 3.12.4. The recorded execution used Windows 11, NumPy 2.3.5, SciPy 1.17.0, CPU computation, and one BLAS thread. Exact package versions are listed in `requirements.txt`.

1. Extract the complete package into a writable directory.
2. Open `RINENG_GFE_Reproduction.ipynb` in a Python 3 notebook environment.
3. Keep `MODE = "full"`, `CACHE_POLICY = "resume"`, `WORK_DIR = None`, and `CACHE_SOURCE_ROOT = None` in the first cell. The notebook installs missing or mismatched dependencies when `INSTALL_PACKAGES = True`.
4. Restart the kernel and run all cells in order.

The notebook creates `RINENG_GFE_Standalone_Run` beneath the current notebook working directory. Its source modules, configuration, and fixed numerical inputs are embedded, so no previous run directory or external data path is required. `project/` provides readable copies of those same modules and starting inputs.

For command-line execution in an installed environment:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name rineng-gfe --display-name "RINENG GFE"
python run_reproduction.py --kernel-name rineng-gfe
```

The command-line runner uses the same notebook cells and stores their outputs in `RINENG_GFE_Reproduction_executed.ipynb`.

## Numerical protocol

| Study component | Protocol |
| --- | --- |
| Standard single-target FE/GFE | Compact evaluator and L-BFGS-B |
| Multitrap FE/GFE | Compact evaluator and BFGS |
| Figure 7 alpha sweep | BFGS with the full-stencil single-target objective at alpha = 1e3, 1e6, and 1e7 m^-1, common initial phases, and a 10,000-iteration cap |
| Conventional | Native objective evaluation and BFGS |
| IB, GS, and AD | Their respective recorded synthesis procedures |
| Figure 2 | Representative objective-progress and pressure-field panels; a 5 by 3 target scan with one initialization per target |
| Table 2 convergence summary | All 100 paired central-target Conventional/GFE initializations |
| Figure 3 | Endpoint analysis of the same 100 paired central-target initializations; a 49-target branch chart |
| Figure 8 | Five stochastic cases per method and task; mean and sample standard deviation across cases |
| Appendix B force-model comparison | Nine paired targets, one FE and one Conventional command per target |
| Appendix E Single generality | Four matched starts for each of 14 distinct configurations; four FE plus four GFE references per configuration |
| Appendix E Triple generality | One command per method/configuration; 14 configurations, five methods, three target equilibria per command |
| Appendix F objective-weight sweeps | One initial phase vector per task, reused across parameter settings |
| Appendix F Triple ablations | Three starts for each of five variants; uniformity and smoothing sweeps use one start |
| Appendix G optimizer comparison | Three starts for each of five objective/solver combinations |
| Appendix H directional-field examples | One shared initial phase vector per formulation |
| Appendix I phase quantization | 15 continuous FE commands; separately, one representative Single and one Triple GFE command |

The notebook's full profile expands the central-target comparison and engineering benchmark while preserving the stated sampling for the other studies. Its first-result helper uses 16 paired initializations for representative diagnostics; this is separate from the 100-pair Table 2 analysis. Figure 2 shows representative histories and fields plus 15 spatial timing pairs. The archived `first_result_seed_count` is not the Table 2 denominator. `STUDY_POPULATIONS.md` identifies the saved evidence and aggregation unit for each study.

For Figure 8, each Triple command contributes the median pressure and displacement across its three targets and the minimum target stiffness. These per-command values are aggregated across the five cases. IB and GS use one deterministic command per task and five synthesis timing repeats; repeated timings do not create independent mechanical samples.

The optimizer uses Gor'kov-based objectives. Independent equilibrium and stiffness evaluation uses the finite-ka elastic-sphere force model. The viscosity comparison is a separate sensitivity analysis.

The first Results convergence figure and optimizer comparison report solver time. The engineering benchmark reports end-to-end command synthesis time, including setup and compact preparation. Subsequent physical validation and diagnostic reevaluation are outside the synthesis timer. Hardware-dependent runtimes should be measured on the reproduction machine.

## Fixed inputs and regeneration

The included fixed inputs define branch-chart initializations, matched comparison commands, selected target cases, and array settings. They are part of the study specification. The notebook regenerates the study commands, physical evaluations, tables, and plots through the supplied numerical routines.

On an empty working directory, `CACHE_POLICY = "resume"` computes missing records. On subsequent runs, it reuses records whose numerical keys match. Set `WORK_DIR` to an existing working directory to continue it. An optional `CACHE_SOURCE_ROOT` can stage compatible raw caches from another complete run directory. The exported `results/` directory is for analysis and comparison; it is not a complete execution cache.

For a separate calculation with fresh current-study caches, restart the kernel and set `CACHE_POLICY = "fresh"`. This creates a separate working directory while retaining the fixed scientific inputs. Regenerating the original construction of those fixed reference banks is outside this entry point.

To inspect the deposited outcomes without computation, open the figure PDF, the tables under `results/`, or the completed notebook in `run_record/`. Running the reproduction notebook computes any missing numerical records needed to generate its figures.

`run_record/RINENG_GFE_Compact_Full_executed.ipynb` is the supplied completed notebook, including its saved outputs and original execution settings. Use `RINENG_GFE_Reproduction.ipynb` for portable reruns.

`analysis/paired_solver_times/` contains the 100 paired central-target records summarized in Table 2 and the reported bootstrap intervals. From the package directory, `python analysis/paired_solver_times/paired_bootstrap.py` recomputes the convergence statistics and intervals directly from `results/production_outputs/tables/fig2_paired_initialization_commands.csv`, using all 100 pairs, 10,000 paired resamples, and seed 20260906. This analysis requires NumPy and uses the saved records. It writes `full_start_statistics.json` and `paired_bootstrap.json`. Figure 3 uses the same population for its endpoint comparison; the five-case engineering benchmark remains a separate analysis. The same output now also summarizes `command_time_s` from all 100 pairs: mean and sample standard deviation are 11.321772 ± 3.285283 ms for GFE and 28.928314 ± 5.075682 s for Conventional. These complete-synthesis records support the Abstract timing comparison; Table 2 continues to use the separately recorded solver-only interval. The five engineering seeds overlap the 100-pair population and have separate timing measurements, so the populations are not pooled as 105 starts.

## Outputs

The notebook writes:

- `unified_outputs/full/RINENG_GFE_full_current_paper_figures.pdf`: the complete ordered figure collection.
- `smoke_outputs/`: the components using the unchanged sampling profile.
- `production_outputs/`: the expanded central-target and engineering populations.
- `appendix_outputs/`, `appendix_B_pressure/`, and `generality/revision_f1_f2/`: appendix calculations and figures.
- `table6_exports/full/`: per-command and population optimizer diagnostics. `table6` is the code export label; manuscript table numbering may differ.

The supplied `results/` preserves the available scientific exports from the recorded run, including command arrays, mechanical evaluations, sensitivity tables, and optimizer diagnostics. `results/notebook_tables/representative_force_stiffness_display.csv` preserves the 24-row force/stiffness table displayed in the executed notebook, at its displayed numerical precision. Optimizer stopping status and physical equilibrium status are recorded separately. `NaN`/N/A denotes an inapplicable or unrecorded quantity as specified by the corresponding table.
