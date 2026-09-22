# Study populations and saved-result sources

This map uses the **printed manuscript numbering**. It follows the executed notebook, the functions actually called, and the saved records selected by each figure or table. A full notebook execution does not assign the same number of random starts to every study.

## Main manuscript

| Display or analysis | Sampling unit and actual population | Saved evidence or producer |
| --- | --- | --- |
| Figure 1 target grid | 49 spatial targets | `results/production_outputs/tables/fig2_fixed_feg_conventional_evolution.csv` |
| Figure 2 convergence and fields | Representative command histories/fields; 15 spatial timing pairs, one matched initialization per target | Executed notebook's first-figure call; `main_data_generation.prepare_baseline_context`; `main_feg_benchmarks` and baseline figure producers |
| Table 2 convergence statistics | 100 matched Conventional/GFE initial phase vectors, 200 command records | `results/production_outputs/tables/fig2_paired_initialization_commands.csv`; `analysis/paired_solver_times/paired_bootstrap.py` |
| Figure 3 central endpoint comparison | The same 100 matched pairs; GFE endpoint classes contain 45 and 55 commands | `production_fig2_central_correspondence.csv` in the same tables directory |
| Figure 3 spatial branch comparison | 49 targets and 98 GFE branch anchors | `fig2_fixed_feg_conventional_evolution.csv`; fixed branch protocol |
| Figures 4–6 local sections and fields | Selected representative commands; three simultaneous target equilibria belong to one Triple command | Executed notebook's separate diagnostic calls; `results/notebook_tables/representative_force_stiffness_display.csv` preserves the displayed mechanical results |
| Tables 3–4 optimizer diagnostics | Table 3: 12 first engineering representatives, one per method/task. Table 4: the eight optimized-method representatives; IB/GS have no scalar-objective loss | `results/table6/full/methods_selected_gradients.csv` |
| Figure 7 force weighting | Three force weights, one shared initial phase vector | `results/figure7_bfgs/tables/main_alpha_force_results.csv` |
| Figure 8 and Table 6 engineering | Five commands per stochastic method/task; one deterministic command and five timings per IB/GS method/task | `results/production_outputs/tables/production_command_metrics.csv`, `production_timing_observations.csv`, and target-mechanics tables |

The notebook's first-result helper retains a 16-pair diagnostic bank. It does not define the denominator of the current Table 2: that table is calculated from all 100 saved central-target pairs also used in Figure 3. Figure 2 remains a representative display with its separate fifteen-target timing scan.

For the engineering comparison, Conventional, FE and GFE use matching initial phase vectors. AD has five cases with its separately specified JAX initialization. There are 44 command endpoints, 60 synthesis timings and 88 target equilibria: 22 Single and 66 Triple. Triple displacement and pressure are reduced by the median across the three targets of each command; stiffness uses their minimum. Means and sample standard deviations are then calculated across five commands. The five IB/GS timings quantify runtime variation for one deterministic design.

The 100-start records contain both solver-only and complete command-synthesis time. Table 2 uses solver-only time; the Abstract and central timing claim use the complete synthesis interval from the same 100 pairs. The five-case engineering statistics also use complete command-synthesis time, measured separately. Their five seeds are included among the central 100, so these are not 105 independent starts. Timing boundaries and populations must remain distinct.

## Appendices

| Printed appendix / analysis | Sampling unit and actual population | Aggregation or selection |
| --- | --- | --- |
| B, force-model comparison | Nine paired targets, one FE and one Conventional command at each | Bootstrap resamples nine target pairs. The 756 spatial-force samples and 252 root-search candidates are not extra commands. |
| B, viscosity | Six representative Single commands; five FE and five GFE initialization outcomes also summarized | Representative mechanics and five-case ranges have separate roles. |
| D, opposed-array prescription transfer | Two methods × five separations × 29 radii = 290 candidates | Ten selected method/separation results; profile samples are not independent commands. |
| D, single-sided radius comparison | Two methods × ten radii = 20 candidates, with one FE reference | The plotted preset and the sampled RMSE minimum are distinct. |
| E, Single generality | Four matched starts per method in 14 distinct configurations | Four FE plus four GFE references per configuration; ACS averages four Conventional trajectories. Nine array panels and six frequency panels share the 40 kHz Square case. |
| E, Triple generality | 14 configurations × five methods = 70 commands, 210 target equilibria | One command per method/configuration; displayed summaries describe its three simultaneous targets. |
| F, continuous sensitivity | One initial phase vector per task across parameter settings | The 177 parameter-slot rows include reused nominal results; they do not represent 177 independent starts. Each displayed weight curve has nine settings. |
| F, curvature/stencil table | Seven axial settings, seven transverse settings and six stencil settings per task | Counts refer to settings, with one initialization per task. Unresolved roots remain in success denominators. |
| F, Triple ablations | Five variants × three matched starts = 15 selected records | Selected from a 30-row file containing additional configurations. Medians/quartiles are over three starts. |
| F, uniformity and smoothing | One Triple initialization | Uses the continuous-sensitivity records, separately from the three-start ablation records. Its nominal 2777 iterations must not be replaced by the ablation campaign's 2482. |
| F, directional/RH controls | Seven directional ratios × two formulations × three objectives = 42 commands; ten RH pressure factors | One shared initial phase vector. Reused nominal commands explain the distinction between command count and new-solve count. |
| G, optimizer comparison | Five objective/solver groups × three matched starts = 15 trials | Medians and gradient distributions use three starts per group. |
| H, Twin/Bottle fields and mechanics | One shared initialization for formulation examples; three selected commands for mechanical validation | The eight method/formulation/sign records are not eight random starts. |
| I, Rayleigh quantization | 15 continuous FE commands × seven resolutions × three rules = 315 evaluations | 105 target/resolution pairs. Each conversion timing is a median of three repeats; plotted timing is the median across 15 targets. |
| I, elastic FE quantization | 105 quantized commands plus 15 continuous references | Added displacement is measured against the corresponding continuous equilibrium. |
| I, GFE hardware quantization | One Single and one Triple reference; nine resolutions × two rules × four targets = 72 quantized target evaluations | Add four continuous target references for 76 total rows. The 4608 offset candidates are selection candidates, not independent initializations. |

Appendix raw tables are under `results/appendix_outputs/`, `results/appendix_B_pressure/`, and `results/appendix_G/`. The executed notebook and readable `project/` sources establish the active reader, filter and aggregation for each table.

## Verification scope

The saved 100-pair convergence records and engineering records support direct numerical recalculation. The compact run archive preserves generality execution receipts, figure manifests and plotted summaries but omits some generated command/trajectory and target-outcome arrays. Generality population and plotted-value checks therefore use those retained records; they are not independent recalculations of absent force residuals. Some baseline representative exports are likewise absent, so a shared seed alone must not be used to assert equality with an engineering command. The deposited notebook display and figure positions provide separate checks of the reported representative mechanics.

No scientific simulation or optimization was rerun during manuscript reconciliation. The original executed notebook and original measurement records are preserved.
