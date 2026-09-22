"""Source-grounded parameter and reproducibility appendix; no numerical solves.

``render(package_root)`` refreshes readable tables and complete CSV/TSV/JSON
inventories. Source defaults, expressions, runtime settings and observed cache
metadata are deliberately distinct. A documented option is not evidence that
an experiment used it. Retained historical commands keep their own settings.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, replace

from rineng_content_id import content_identity
import importlib.metadata
import inspect
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from types import SimpleNamespace
import xml.sax.saxutils as xml

import numpy as np
import pandas as pd

SCHEMA = "parameter-reproducibility-appendix-v3-clear-appendices-main-triple-30000"


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _content_id(path):
    return content_identity(Path(path).read_bytes()).hexdigest()


def _fmt(value):
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(value, default=_json, ensure_ascii=True)
    if isinstance(value, float): return f"{value:.12g}"
    return str(value)


def _write(out, name, rows):
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(out / (name + '.csv'), index=False)
    frame.to_csv(out / (name + '.tsv'), index=False, sep='\t')
    (out / (name + '.json')).write_text(frame.to_json(orient='records', indent=2), encoding='utf-8')
    return frame


def _source(root, obj, term=None):
    path = Path(inspect.getsourcefile(obj)).resolve()
    lines, number = inspect.getsourcelines(obj)
    if term:
        found = next((i for i, line in enumerate(lines) if term in line), None)
        if found is not None: number += found
    try: path = path.relative_to(root)
    except ValueError: pass
    return str(path), getattr(obj, '__qualname__', str(obj)), number


def _unit(name):
    explicit = {'lmax':'order', 'gtol':'objective units per radian', 'alpha_per_m':'m^-1',
        'alpha_force':'m^-1', 'beta_curvature_per_pa':'N m^-1 Pa^-1',
        'beta_pressure':'N m^-1 Pa^-1', 'gamma_uniformity':'1',
        'pressure_epsilon_rel':'1', 'smooth_stage_factors':'1',
        'smooth_stage_maxiters':'accepted iterations', 'fit_shell_wavelengths':'wavelengths',
        'fit_rcond':'1', 'random_seed':'integer identity', 'array_side':'elements per side',
        'mode':'text', 'budget_mode':'text'}
    if name in explicit:return explicit[name]
    for suffix,unit in [('_kg_m3','kg m^-3'),('_pa_s','Pa s'),('_m_s2','m s^-2'),
                        ('_m_s','m s^-1'),('_hz','Hz'),('_rad','rad'),('_n_m','N m^-1'),
                        ('_pa','Pa'),('_n','N'),('_m','m'),('_a','particle radii'),('_s','s')]:
        if name.endswith(suffix): return unit
    if any(x in name for x in ('maxiter','iterations','max_nfev','n_mu','n_phi','points','count','half_stencil','gauge_index')): return 'count'
    return 'see expression / source'


def _readable_parameters(root):
    from . import config, gorkov_core as gc, exact_validator as ev, pipeline as p
    from . import fixed_fe_endpoints as fixed, multitrap as mt, branch_figures as branch
    from . import main_feg, main_feg_branch, triple_long_iteration
    rows=[]
    def add(group, name, value, unit, scope, obj, term=None, note=''):
        file,function,line = _source(root,obj,term)
        rows.append(dict(group=group, parameter=name, value=_fmt(value), units=unit,
            applicability=scope, source_file=file, source_function=function,
            source_line=line, note=note))
    def dataclass(group, value, scope, obj):
        for name,v in asdict(value).items():add(group,name,v,_unit(name),scope,obj,name)
    dataclass('Physical model', ev.Medium(),'Current medium; viscosity only in separate viscous sensitivity',ev.Medium)
    dataclass('Physical model', ev.ElasticSphere(),'Declared elastic bead; simulation inputs',ev.ElasticSphere)
    sphere=ev.ElasticSphere(); medium=ev.Medium(); cfg=config.RunConfig()
    add('Physical model','wavelength lambda',medium.sound_speed_m_s/cfg.frequency_hz,'m','40 kHz baseline',config.RunConfig,'frequency_hz','Computed as c/f; not a tunable regularizer.')
    add('Physical model','ka',2*math.pi*cfg.frequency_hz*sphere.radius_m/medium.sound_speed_m_s,'1','40 kHz baseline',ev.ElasticSphere,'radius_m','Radius a=0.65 mm, not diameter.')
    add('Physical model','GFE upward force target',mt.effective_weight_force_target_n(mt.SphereInFluid()),'N','Single and Triple GFE',mt.effective_weight_force_target_n,note='F_target=(rho_p-rho_0) V g e_z; fixed from material, not fitted.')
    add('Physical model','Gor\'kov sign','Kp |p|^2 - Kg |grad p|^2','J','Main, A, C1-C2, D and F; B and C3 also test the alternative-sign formulation',gc.gorkov_potential_from_pressure,note='Alternative-sign formulation means Kp |p|^2 + Kg |grad p|^2. This is a changed relative sign, not a change of phasor convention. Existing cache key strings retain their original provenance.')
    add('Physical model','pressure convention','peak phasor, time exp(-i omega t)','Pa','Current pressure synthesis / elastic validation',ev.ArbitraryArrayPressureField,note='Source reference sqrt(2)*20*0.30 Pa m; equivalent table scaling is separately inventoried.')
    add('Physical model','source strength',gc.REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,'Pa m','On-axis peak reference',gc.transfer_matrix,note='Murata angular amplitude interpolation is evaluated at each sample.')
    add('Physical model','table source scale',gc.SOURCE_SCALE_PA_M_PER_MURATA_UNIT,'Pa m per Murata unit','gorkov_core transfer matrix',gc.transfer_matrix,'source_scale_pa_m')
    coefficients=gc.gorkov_coefficients(cfg.frequency_hz)
    add('Physical model','Gor\'kov pressure coefficient Kp',coefficients.pressure_j_pa2,'J Pa^-2','Peak-phasor surrogate; declared Rayleigh sphere',gc.gorkov_coefficients,'pressure =')
    add('Physical model','Gor\'kov gradient coefficient Kg',coefficients.gradient_j_m2_pa2,'J m^2 Pa^-2','Positive for declared denser sphere, subtracted in U',gc.gorkov_coefficients,'gradient =')
    add('Physical model','Gor\'kov contrast f1 / f2',[coefficients.monopole_contrast_f1,coefficients.dipole_contrast_f2],'1','Surrogate uses declared compressible-sphere material',gc.gorkov_coefficients,'f1 =')
    for mode in ('quick','full'):
        dataclass('Run configuration',config.RunConfig.for_mode(mode),f'{mode} mode; actual supplied run is SMOKE',config.RunConfig)
    add('Array and objective','element count',256,'elements','16 x 16 baseline',config.square_positions)
    add('Array and objective','positions','Cartesian square at z=0; pitch 0.010; normals +z','m','Single-sided baseline',config.square_positions)
    for label,method in [('GFE / FE Single',fixed.fixed_single_fe_method_spec()),('Conventional Single',gc.method_spec('Conventional'))]:
        dataclass('Array and objective',method,label,fixed.fixed_single_fe_method_spec if label.startswith('GFE') else gc.method_spec)
    conventional_triple,fe_triple=triple_long_iteration._configs(
        SimpleNamespace(config=cfg),triple_long_iteration.MAIN_TRIPLE_ITERATION_CAP)
    triple=replace(fe_triple,compensate_effective_gravity=True)
    dataclass('Main Triple objective',triple,'Current main GFE; shared 30000-iteration main budget',triple_long_iteration._configs)
    add('Main Triple objective','FE control gravity compensation',False,'boolean','Main FE control; other objective values and stage caps match GFE',main_feg.feg_triple_run,note='GFE sets compensate_effective_gravity=True; FE sets it False. Seeds, initial commands, alpha and curvature weights remain paired.')
    dataclass('Main Triple Conventional',conventional_triple,'Current main Conventional; cumulative 30000 accepted-iteration budget',triple_long_iteration._configs)
    add('Solvers and stopping','Main Triple iteration cap',triple_long_iteration.MAIN_TRIPLE_ITERATION_CAP,'accepted iterations','Current main FE / GFE / Conventional',triple_long_iteration._configs,note='FE/GFE stage budgets are 250 and 29750; Conventional retains checkpoints at 10000, 20000 and 30000. Current C1/C2 Triple uses30000; the C3 RH Single control uses10000. Earlier C1/C3/C6 file names in archived data retain their recorded protocols.')
    add('Solvers and stopping','Main Triple Conventional checkpoints',triple_long_iteration.CHECKPOINTS,'accepted iterations','Cumulative Conventional run; checkpoint observations do not reset the budget',triple_long_iteration._run_cumulative_conventional,note='Precision-loss restarts begin from the identical terminal phase under the deposited restart policy; individual native stops remain recorded.')
    add('Solvers and stopping','Main Triple Conventional restart policy',triple_long_iteration.RESTART_POLICY,'rule','Current main cumulative budget',triple_long_iteration._run_cumulative_conventional)
    for n in ('half_stencil','smooth_pressure_relative','gauge_index'):
        v=inspect.signature(gc.SingleTargetObjective).parameters[n].default
        add('Array and objective',n,v,_unit(n),'SingleTargetObjective defaults',gc.SingleTargetObjective,n)
    add('Array and objective','GFE loss','-W:Hess(U) + alpha ||F_Gorkov-F_target|| + beta P','N m^-1','Single GFE',main_feg.feg_single_objective,note='FE sets F_target=0. GFE is the displayed objective name; FE+g occurs only in retained internal cache keys.')
    add('Triple objective','per-target loss ell_j','-W:Hess(U_j) + alpha rho_epsilon(F_j-F_target) + beta P_j','N m^-1','Triple GFE; matched ablations remove one term',mt.evaluate_multitrap_objective)
    add('Triple objective','pooled loss','mean(ell_j) + gamma_u [sqrt(var(ell_j)+epsilon_std^2)-epsilon_std]','N m^-1','Triple GFE / FE',mt.evaluate_multitrap_objective,note='Population variance of per-target loss; not pressure variance.')
    add('Triple objective','epsilon_g','stage_factor * initial median force scale','N','Force smoothing enabled',mt.estimate_smooth_scales,note='Fixed within each stage. Realized values per command are exported, not replaced with a universal number.')
    add('Triple objective','epsilon_std','multiplier * stage_factor * initial per-target loss std scale','N m^-1','Uniformity enabled; nominal multiplier 1',mt.estimate_smooth_scales,note='C6 varies the multiplier independently of force smoothing. Ablating a loss term may change the derived initial scale; actual stage values are exported.')
    add('Triple objective','epsilon_p','pressure_epsilon_rel * local stencil RMS(|p|)','Pa','Pressure retention enabled',mt.evaluate_multitrap_objective,note='Analytic phase derivative of the RMS-dependent smoothing is included.')
    from . import sota
    add('Solvers and stopping','FE/GFE/RH Single solver','L-BFGS-B; compact analytic evaluator; unbounded reduced phases','algorithm','All single-target main and appendix FE/GFE/RH solves',sota.solve_corrected_gorkov_fe)
    add('Solvers and stopping','FE/GFE multi-target solver','BFGS; compact analytic evaluator','algorithm','All multi-target FE/GFE solves',mt.run_multitrap)
    add('Solvers and stopping','Conventional solver','BFGS; native analytic evaluator','algorithm','Conventional production solves',sota.solve_corrected_gorkov_fe)
    add('Solvers and stopping','native gradient threshold',1e-8,'N m^-1 rad^-1','Raw reduced-phase infinity norm; L-BFGS-B for single FE/GFE/RH; BFGS for multi FE/GFE and Conventional',sota.solve_corrected_gorkov_fe,'gtol','Reporting L2 threshold 1e-3 is separate; neither cap nor precision-loss exit is relabeled convergence.')
    add('Solvers and stopping','reporting threshold',1e-3,'N m^-1 rad^-1','Reduced-phase gradient L2 summary',mt.MultitrapObjectiveConfig,'report_gradient_tol')
    from scipy.optimize._optimize import _minimize_bfgs
    for n in ('norm','eps','xrtol','c1','c2','finite_diff_rel_step','hess_inv0'):
        v=inspect.signature(_minimize_bfgs).parameters[n].default
        add('Solvers and stopping','SciPy BFGS '+n,v,'SciPy native option','Conventional BFGS defaults when caller omits keyword',_minimize_bfgs,n,note='These are installed-library defaults, not recovered historical environment settings. eps unused with supplied analytic jacobian.')
    for mode in ('quick','full'):
        proto=p._finite_ka_protocol(SimpleNamespace(config=config.RunConfig.for_mode(mode)))
        for n,v in proto['numerics'].items():add('Elastic validation',n,v,_unit(n),f'Main {mode} protocol',p._partial_wave_numerics,n)
        for n in ('root_starts_a','search_half_width_a','max_nfev_per_start','numerical_root_tolerance','least_squares','boundary_fraction_epsilon','jacobian_step_a'):
            add('Elastic validation',n,proto[n],_unit(n),f'Main {mode} protocol',p._finite_ka_protocol,n)
    add('Elastic validation','spherical quadrature','Gauss-Legendre in cos(theta); uniform periodic phi','rule','Incident fitting and momentum-flux integration',ev.spherical_quadrature)
    add('Elastic validation','solid boundary','Radial velocity / normal stress continuous; tangential traction zero','rule','Elastic sphere in inviscid host',ev.elastic_sphere_scattering_coefficients,note='Longitudinal and shear internal modes retained; viscosity is a separate sensitivity evaluator, not a full viscous finite-ka solve.')
    add('Elastic validation','stiffness','K_sym = -(J_F + J_F^T)/2; report min eigenvalue','N m^-1','At resolved total-force equilibrium',ev.validate_static_trap)
    for name,expression,unit in [('weighted curvature','W:Hess(U)','N m^-1'),('Laplacian','trace(Hess(U))','N m^-1'),('force','-grad(U)','N'),('displacement','||x_equilibrium - x_target|| / a','1'),('command similarity','|u^H v| for normalized target-demodulated complex states','1'),('projector distance','sqrt(max(0, 1 - similarity^2))','1')]:
        add('Notation and metrics',name,expression,unit,'Current exported physical / command metrics',branch.projective_similarity if name in ('command similarity','projector distance') else ev.validate_static_trap if name=='displacement' else gc.SingleTargetObjective)
    add('Embedding and branch','dimension',3,'coordinates','Main Figure 2 fixed projective MDS',branch.fit_fixed_3d_frame,'[:3]')
    add('Embedding and branch','landmarks','Independently reoptimized GFE medoids, both endpoint components at every deposited chart target','phase commands','Frozen throughout Conventional projection',main_feg_branch.fixed_feg_branch_pack,note='Source alpha=9 medoids initialize GFE alpha=10 solves; source endpoints are not relabeled as GFE.')
    add('Embedding and branch','historical component assignment','Average-linkage hierarchical clustering, target-local k=2','rule','Source FE endpoint organization; no k-means in this frame',branch._local_two_component_labels,note='Fixed k does not itself prove physical uniqueness or robustness.')
    add('Embedding and branch','MDS fit','B=-0.5 H D^2 H; top 3 positive eigenpairs','rule','GFE landmarks only',branch.fit_fixed_3d_frame)
    add('Embedding and branch','orientation','Orthogonal Procrustes to centered [xi, eta, B1=-1/B2=+1]','rule','Display orientation only',branch.fit_fixed_3d_frame,'orthogonal_procrustes')
    add('Embedding and branch','out-of-sample placement','Centered squared distances to frozen GFE landmarks','rule','Conventional iterations',branch.FrozenBranchFrame3D.project)
    add('Embedding and branch','landmark self-reprojection atol',2e-10,'coordinate units','rtol=0',branch.fit_fixed_3d_frame,'atol=')
    add('Embedding and branch','embedding random seed','none','not applicable','Deterministic eigendecomposition / Procrustes',branch.fit_fixed_3d_frame,note='Optimizer initial seeds and source branch selection seeds are separately recorded.')
    source_config=root/'runtime/data/corrected_branch_evolution/anchor_config.json'
    if source_config.exists():
        old=json.loads(source_config.read_text())['config']
        for n,v in old.items():
            rows.append(dict(group='Retained source branch',parameter=n,value=_fmt(v),units=_unit(n),applicability='Historical branch archive only; not current GFE solver',source_file=str(source_config.relative_to(root)),source_function='JSON config',source_line=1,note='Archived gtol=0 and 2000 anchor cap are preserved explicitly; new GFE anchors use 10000 and 1e-8.'))
    from . import diff_pat, appendix_a
    dataclass('Holography baseline',diff_pat.DIFF_PAT_CONFIG,'AD / Diff-PAT; supplied transfer/signature',diff_pat.DiffPATConfig)
    add('Holography baseline','AD controls',diff_pat.DIFF_PAT_CONTROL_POINTS,'points','Single / Triple',diff_pat.diff_pat_phase_only)
    add('Holography baseline','AD RNG','JAX PRNGKey((base_seed + 181) % 2**32)','integer key','No inherited global NumPy seed',diff_pat.diff_pat_phase_only,'effective_seed')
    add('Holography baseline','AD normalization','mean[(|A exp(i phi)| / row_capacity - target / target_RMS)^2]','1','Phase-only pressure-amplitude loss',diff_pat._optimizer_kernel,note='row_capacity=sum(abs(A),axis=1); Gor\'kov and force terms absent.')
    add('Holography baseline','AD timer','JIT kernel invocation through command and loss block_until_ready','s','Returned jit_sync_wall_s',diff_pat.diff_pat_phase_only,'started =',note='PRNG/device-array preparation occurs before this inner timer. Wrapper end-to-end time must be consulted separately.')
    dataclass('Appendix protocols',appendix_a.NUMERICS,'A validator numerics; 18 paired active commands, 27 historical FE commands retained as context',appendix_a._inputs)
    add('Appendix protocols','A root protocol',appendix_a.ROOT_PROTOCOL,'see field units','Seven Cartesian starts: target plus +/-1a axes',appendix_a._inputs)
    from . import appendix_c_feg
    appendix_triple=fixed.fixed_triple_fe_config(gtol=1e-8)
    add('Appendix protocols','C frozen Triple iteration cap',sum(appendix_triple.smooth_stage_maxiters),'accepted iterations','C1 and archived C2 reference protocol',fixed.fixed_triple_fe_config,note='The fixed 10000-iteration contract is retained for existing frozen references. Revised C2/C6 use 30000 while preserving the nominal objective and paired seeds.')
    add('Appendix protocols','C frozen Triple stage caps',appendix_triple.smooth_stage_maxiters,'accepted iterations','Appendix C FE / GFE reference and matched perturbations',fixed.fixed_triple_fe_config,note='Nominal alpha3000, curvature weights, objective definitions and actual seeds remain unchanged; removal experiments retain their exported stage configurations.')
    add('Appendix protocols','C alpha factors',appendix_c_feg.ALPHA_FACTORS,'relative to nominal','Single nominal alpha=10; Triple alpha=3000',appendix_c_feg.specifications)
    add('Appendix protocols','C Single seed',appendix_c_feg.SINGLE_SEED,'integer seed','Frozen GFE reference and same-start alpha sweep',appendix_c_feg.specifications)
    add('Appendix protocols','C Triple seeds',appendix_c_feg.TRIPLE_SEEDS,'integer seeds','Paired GFE ablations and FE control',appendix_c_feg.specifications)
    add('Appendix protocols','C ablations',appendix_c_feg.ABLATIONS,'configuration','Each seed matched; targets within one command are dependent',appendix_c_feg.specifications)
    add('Appendix protocols','C frozen reference','Nominal GFE command at same task/seed; independently validated elastic total-force root','phase command','No parameter-wise reference refitting',appendix_c_feg.specifications)
    add('Appendix protocols','Sensitivity coverage','C1 alpha; C2 component removals; C6 independent gamma_u and epsilon_std brackets','executed scope','GFE sensitivity; C4/C5 separately cover morphology-specific weights',appendix_c_feg.specifications,note='Nonzero GFE beta, epsilon_p, epsilon_g and main curvature-ratio brackets remain separate coverage items. C6 isolates STD smoothing from force smoothing. See SENSITIVITY_VARIABLES.md and its CSV for actual ranges.')
    from . import appendix_c_std_gfe
    add('Statistics','C2/C6 paired bootstrap','10000 resamples; seed20260906; 2.5/97.5 percentile limits of median paired changes','95% interval','Three command/seed pairs; dependent targets summarized within command',appendix_c_std_gfe._write_text,note='Exploratory n=3 interval; no target pseudo-replication. B3 is a representative mechanical comparison, not an endpoint-frequency estimate.')
    add('Appendix protocols','C2/C6 Triple stage caps',appendix_c_std_gfe.CAPS,'accepted iterations','Current 30000-cap ablations and isolated STD sensitivity',appendix_c_std_gfe.specifications)
    add('Appendix protocols','C6 uniformity weight',appendix_c_std_gfe.GAMMA_VALUES,'dimensionless','gamma_u; nominal 1',appendix_c_std_gfe.specifications,note='Vary only weight; retain nominal force and STD smoothing schedules.')
    add('Appendix protocols','C6 STD smoothing multiplier',appendix_c_std_gfe.STD_FACTORS,'relative to nominal','epsilon_STD only; force smoothing unchanged',appendix_c_std_gfe.specifications,note='Zero is unsmoothed STD with gamma_u=1, not removal of the uniformity penalty. All dimensional stage epsilons are exported per seed.')
    add('Appendix protocols','C complex-pressure volume','9 x 9 x 9 grid within +/-0.75 wavelength of each prescribed target','729 points per target','Identical samples for candidate and frozen GFE reference',appendix_c_feg._correspondence,'offsets=')
    bpath=root/'appendix_B_pressure/data/sensitivity_cache'
    bconfigs={}
    for path in sorted(bpath.glob('*.json')):
        record=json.loads(path.read_text())
        key=(record.get('requested'),record.get('method'))
        if float(record.get('eta',0))==.01 and record.get('beta_factor') in (None,1.):bconfigs[key]=(path,record)
    for (shape,method),(path,record) in bconfigs.items():
        value={key:record.get(key) for key in ('seed','alpha_per_m','beta_curvature_per_pa','curvature_weights','pressure_mode','smooth_pressure_relative','maxiter','gtol')}
        rows.append(dict(group='Appendix protocols',parameter=f'B / C4-C5 nominal {shape} {method}',value=_fmt(value),units='alpha m^-1; beta N m^-1 Pa^-1; W dimensionless',applicability='B source fields; C4 shows standard Twin and alternative Bottle, C5 alternative Bottle only; seed260905',source_file=str(path.relative_to(root)),source_function='deposited numerical metadata',source_line=1,note='Eta sweep .001/.01/.1 at fixed trace; RH beta factors0/.1/1/10. Archived combinations remain in the raw cache; plotted combinations are explicitly filtered.'))
    warm=root/'runtime/src/hat_revision_pipeline/appendix_c_warm_gfe.py'
    if warm.exists():
        rows.append(dict(group='Appendix protocols',parameter='C3 controlled warm start',value='Nominal alpha10 GFE at seed260828; sequence alpha1000,1000000,10000000',units='m^-1; integer seed',applicability='Same-seed cold/warm comparison; explicit continuation experiment',source_file=str(warm.relative_to(root)),source_function='See source numeric inventory / actual cache metadata',source_line=1,note='These warm starts are not silently substituted into the cold-start Single/Triple benchmarks.'))
    from . import appendix_support_gfe as deployment, appendix_de
    add('Phase resolution and deployment','Current GFE phase levels Q',deployment.LEVELS,'levels per 2 pi cycle','D exported sweep; D3 illustrates Q32',deployment._d3,note='Delta phi = 2 pi / Q. Nearest and elastic-mechanics common-offset rounding share each continuous GFE reference. D3 shows field contours and per-target displacement vectors; the original sweep plot is retired.')
    add('Publication layout','Figure 2 marker scaling','Original areas 27 / 20 pt^2 at 10.8 inch width; final 11.625 / 8.611 pt^2 at 180 mm','point squared','Conventional / GFE branch-web markers',main_feg.figure_2_feg,note='The Figure 2 branch markers bypass the global readability area floor. Fixed GFE landmarks and embedding coordinates unchanged. Connection opacity 0.2; markers remain opaque. Other marker readability floors remain unchanged.')
    add('Publication layout','Displacement display unit','m; d_m = (d/a) times radius_m','meter','Main 8 and active A/C/D/F displacement plots',ev.ElasticSphere,note='Points, intervals, limits and exported SI aliases use dimensional values. Normalized audit columns remain. Pressure-field axes in mm remain dimensional; frozen branch coordinates remain dimensionless.')
    add('Publication layout','Final plot annotations','No temporary figure-review titles or SMOKE footers','display rule','All 22 active figures',main_feg.figure_2_feg,note='Case labels, panel letters, units, legends and quantitative annotations remain. Numerical evidence status is retained in captions, notebook and exports.')
    add('Phase resolution and deployment','Historical FE phase levels Q',appendix_de.LEVELS,'levels per 2 pi cycle','D1 Rayleigh comparison and D2 elastic validation bridge',deployment.run,note='Frozen historical FE grid. Its selection model and command population differ from D3; intermediate points are added to the current GFE experiment.')
    for reference in deployment.PHASE_RESOLUTION_LITERATURE:
        add('Phase resolution and deployment',f"Implemented literature resolution Q={reference['levels']}",reference['exact_phase_step'],'rad','40 kHz hardware; resolution anchor only',deployment._write_phase_resolution_provenance,note=f"{reference['authors']}; {reference['title']} ({reference['year']}); DOI {reference['doi']}. {reference['interpretation']} The present simulation does not claim to reproduce that hardware.")
    a_protocol = root / 'appendix_outputs/A_GFE/A_protocol.json'
    if a_protocol.exists():
        protocol = json.loads(a_protocol.read_text())
        selected = {
            'sample_selection': ('Paired command selection', '9 target pairs; 18 commands'),
            'statistical_unit': ('Sampling unit', 'paired target'),
            'command_scope': ('Frozen command scope', 'see coefficients'),
            'neighbor_radii_a': ('3D force-sampling radii', 'particle radii'),
            'direction_threshold': ('Force-direction threshold', 'fraction of effective weight'),
            'pressure_minimum': ('Local pressure-minimum protocol', 'particle radii; relative intensity'),
            'field_display': ('XZ force-field display', 'particle radii; grid counts'),
            'bootstrap': ('Paired bootstrap', 'seed; resample count; percentile'),
        }
        for name, (label, units) in selected.items():
            if name not in protocol:
                continue
            value = protocol[name]
            if name == 'pressure_minimum':
                value = dict(value)
                value['shell_directions'] = '26 normalized nonzero vectors in {-1,0,1}^3; exact vectors in A_protocol.json'
            rows.append(dict(group='Appendix A force comparison',parameter=label,value=_fmt(value),units=units,applicability='A1 force fields and A2 target-grouped statistics',source_file=str(a_protocol.relative_to(root)),source_function='Executed comparison protocol',source_line=1,note='Full resolved values, source hashes, and paired command identities are retained in A_protocol.json and the A portable data exports. Physical constants and validator settings are listed in the corresponding groups above.'))
    from . import cache
    add('Reproducibility and timing','NumPy RNG','np.random.default_rng(seed); explicit per-experiment seeds','generator','Current NumPy optimizers; actual integer lists exported',main_feg.feg_single_bank,'default_rng')
    add('Reproducibility and timing','global seed42 / PyTorch flags','Historical manuscript only; not applied to current NumPy/JAX pipeline','not applicable','Replaces predecessor Appendix C controls',main_feg.feg_single_bank,note='Do not copy torch/cudnn settings or V100/RTX4060 hardware claims into current runs.')
    add('Reproducibility and timing','cache identity','Canonical payload + implementation namespace + dependency snapshot; phase/source hashes when supplied','content_id','Cached numerical commands remain tied to original metadata',cache.CacheStore,note='No seed/phase/force-target edits solely for presentation. Cached wall times are original run times, not replay costs.')
    add('Reproducibility and timing','Current main Triple timing scope','Matched main run at the shared 30000-iteration budget','s','Current Triple timings only; smoke_outputs/tables/main_triple30k/main_triple30k_commands.csv and main_triple30k_protocol.json',triple_long_iteration._cached_node,note='Single timings and historical Appendix C timings retain their own recorded runs. An iteration cap is not an assertion that every method reaches it. Timing values must not be pooled across these scopes; setup, solve and final-evaluation components are exported per command.')
    timing_audit=root/'data_exports/conventional_timing_audit.csv'
    if timing_audit.exists():
        rows.append(dict(group='Reproducibility and timing',parameter='Conventional timing before the 30000 update',value='Original Triple 276.208885 s; integrated/deposited Triple 88.056407555 s; Single 61.734891728 s',units='s',applicability='Historical Figure 8 records at the same 10000-iteration budget and cap-limited outcome',source_file=str(timing_audit.relative_to(root)),source_function='Deposited notebook/CSV timing audit',source_line=1,note='The Triple timing change preceded promotion of GFE and was not a relaxed maxiter setting. The recorded evidence does not identify a hardware or environment cause. Current matched 30000-iteration timings are reported separately.'))
    add('Reproducibility and timing','bitwise repeatability','Not guaranteed across BLAS, CPU, JAX, compiler or library versions','scope','Explicit seeds make initialization reproducible',cache.dependency_snapshot,note='Software versions and threadpool runtime are recorded. Existing command caches provide exact deposited phases.')
    transfer_path = root / 'manuscript_appendix_transfer/transfer_status.json'
    if transfer_path.exists():
        transfer = json.loads(transfer_path.read_text())
        for experiment in transfer['experiments']:
            rows.append(dict(group='Prerevision appendix transfer',parameter=experiment['id'],value=experiment['status'],units='execution / transfer status',applicability='Original manuscript C-D-E; current appendix lettering is independent',source_file=str(transfer_path.relative_to(root)),source_function='Audited manuscript and retained current C3 data',source_line=1,note='Editable archival and GFE-adapted LaTeX, source-page extracts, experiment definitions and exact missing-input records are in manuscript_appendix_transfer. Historical optimizer values are not current GFE measurements.'))
    from . import reviewer_robustness as rr
    for name,value in rr.SOLVER.items():
        add('Current reviewer optimizer controls',name,value,_unit(name),'Current-objective C7/C9/C10/C11; active overrides in each command row',rr._optimize,
            note='Native unscaled objective; AdamW nonzero decay is a labeled phase prior. C7 timings are serial and other sweep timings are not a speed comparison.')
    for name,values,tasks,methods,units in rr.GRIDS:
        add('Expanded sensitivity design',name,values,units,'/'.join(tasks)+'; '+ '/'.join(methods),rr.sensitivity_design,
            note='Complete grid and measured completion are separate exports under C_GFE/reviewer_robustness. Full mode uses ten paired seeds.')
    add('Conventional notation','lambda_p',1.,'N m^-1 Pa^-1','Single/Triple Conventional pressure weight; distinct from lambda_ac=c/f',rr._build,
        note='Explicitly swept at nine multipliers. The source field is named beta for API compatibility; this does not make it wavelength.')
    return rows


def _source_inventory(root):
    paths=sorted((root/'runtime/src/hat_revision_pipeline').glob('*.py'))
    paths += sorted(root.glob('*.py'))
    paths += sorted((root/'appendix_B_pressure').glob('*.py'))
    rows=[]; sources=[]
    for path in paths:
        source=path.read_text(encoding='utf-8'); tree=ast.parse(source)
        rel=str(path.relative_to(root)); sources.append(dict(source_file=rel,content_id=_content_id(path),bytes=path.stat().st_size))
        parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        def context(node):
            scope=[]
            while node in parents:
                node=parents[node]
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):scope.append(node.name)
            return '.'.join(reversed(scope)) or '<module>'
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,(int,float,complex)) and not isinstance(node.value,bool):
                rows.append(dict(source_file=rel,source_function=context(node),source_line=node.lineno,
                    parameter=f'literal@{node.lineno}:{node.col_offset}',expression=ast.get_source_segment(source,node) or repr(node.value),
                    numeric_literals_json=json.dumps([str(node.value)]),declaration_type='numeric literal',
                    execution_evidence='Source literal only; includes plotting and validation constants',
                    units='Context-dependent; see containing expression',source_content_id=sources[-1]['content_id']))
        for node in ast.walk(tree):
            kind=None;name=None;value=None
            if isinstance(node,ast.Assign):kind='assignment';name=ast.unparse(node.targets[0]);value=node.value
            elif isinstance(node,ast.AnnAssign) and node.value is not None:kind='annotated assignment';name=ast.unparse(node.target);value=node.value
            elif isinstance(node,ast.keyword) and node.arg:kind='call keyword';name=node.arg;value=node.value
            elif isinstance(node,ast.arg):continue
            if kind is None:continue
            constants=[n.value for n in ast.walk(value) if isinstance(n,ast.Constant) and isinstance(n.value,(int,float,complex)) and not isinstance(n.value,bool)]
            if not constants:continue
            expr=ast.get_source_segment(source,value) or ast.unparse(value)
            if len(expr)>1200:continue
            scope=context(node)
            numeric=[str(x) for x in constants]
            rows.append(dict(source_file=rel,source_function=scope,source_line=node.lineno,
                parameter=name,expression=expr,numeric_literals_json=json.dumps(numeric),
                declaration_type=kind,execution_evidence='Source declaration only; runtime values require run metadata',
                units=_unit(name),source_content_id=sources[-1]['content_id']))
        for node in ast.walk(tree):
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            args=node.args.posonlyargs+node.args.args
            defaults=list(zip(args[-len(node.args.defaults):],node.args.defaults)) if node.args.defaults else []
            defaults+=list(zip(node.args.kwonlyargs,node.args.kw_defaults))
            for arg,value in defaults:
                if value is None:continue
                constants=[n.value for n in ast.walk(value) if isinstance(n,ast.Constant) and isinstance(n.value,(int,float,complex)) and not isinstance(n.value,bool)]
                if not constants:continue
                rows.append(dict(source_file=rel,source_function=node.name,source_line=arg.lineno,
                    parameter=arg.arg,expression=ast.unparse(value),numeric_literals_json=json.dumps([str(x) for x in constants]),
                    declaration_type='function default',execution_evidence='Default only; may be overridden by caller',
                    units=_unit(arg.arg),source_content_id=sources[-1]['content_id']))
    # These machine-readable settings and protocol documents govern the same
    # production campaign but are not Python AST inputs.  Hash them beside the
    # executable sources so the active P/reviewer inventories identify the
    # exact timing and sample-plan revision used for release.
    for path in [root/'production_config.json', root/'FULL_RUN_SAMPLE_PLAN.csv',
                 root/'FULL_RUN_SAMPLE_PLAN.md', root/'CONVENTIONAL_TIMING_AUDIT.md',
                 root/'MAIN_FIG8_TIMING_PROTOCOL.md']:
        if path.is_file():
            sources.append(dict(source_file=str(path.relative_to(root)),content_id=_content_id(path),bytes=path.stat().st_size))
    return rows,sources


def _is_seed_identity(name):
    # Seed counts, starting-seed rules, and dimensions are not realized RNG keys.
    leaf=str(name).split('.')[-1].lower()
    return (leaf=='seed' or leaf.endswith('_seed')) and not leaf.startswith(('n_','num_','count_','min_','max_','minimum_','maximum_'))


def _data_inventory(root,out):
    folders=[root/'smoke_outputs/tables',root/'smoke_outputs/manuscript_tables',root/'appendix_B_pressure']
    folders += [root/'runtime/data/corrected_branch_evolution']
    folders += [x for x in (root/'appendix_outputs').glob('*') if x.is_dir() and x.name not in {'B','C', 'P_parameters'}]
    folders += [root/'data_exports']
    # The replacement GFE C directory is inventoried by its actual output name.
    rows=[];seeds=[];actual=[];seen=set()
    for folder in folders:
        if not folder.exists():continue
        for path in sorted(folder.rglob('*.csv')):
            if 'data/cache' in str(path) or path in seen:continue
            seen.add(path)
            try:df=pd.read_csv(path)
            except (pd.errors.EmptyDataError,pd.errors.ParserError,UnicodeDecodeError):continue
            rel=str(path.relative_to(root))
            rows.append(dict(source_file=rel,rows=len(df),columns_json=json.dumps(list(df.columns)),
                             content_id=_content_id(path),bytes=path.stat().st_size,evidence_status='SMOKE / consult source status'))
            for col in df.columns:
                if _is_seed_identity(col) and pd.api.types.is_numeric_dtype(df[col]):
                    values=pd.to_numeric(df[col],errors='coerce').dropna()
                    values=values[np.isfinite(values) & (values==np.floor(values))]
                    for val,count in values.value_counts().sort_index().items():
                        seeds.append(dict(source_file=rel,seed_column=col,seed=int(val),record_count=int(count),kind='observed exported integer seed'))
            if any(t in path.stem.lower() for t in ('parameter','protocol','definition','notation')):
                for i,rec in enumerate(df.to_dict('records')):
                    actual.append(dict(source_file=rel,record_index=i,record_json=json.dumps(rec,default=_json,allow_nan=True)))
    sensitivity=root/'SENSITIVITY_VARIABLES.csv'
    if sensitivity.exists():
        frame=pd.read_csv(sensitivity)
        rows.append(dict(source_file=str(sensitivity.relative_to(root)),rows=len(frame),
            columns_json=json.dumps(list(frame.columns)),content_id=_content_id(sensitivity),
            bytes=sensitivity.stat().st_size,evidence_status='Sensitivity/control coverage inventory; consult per-variable status'))
    return rows,seeds,actual


def _cache_inventory(root):
    """Read JSON headers only; do not unpickle historical numerical archives."""
    rows=[];seeds=[]
    candidates=list((root/'appendix_B_pressure/data').rglob('*.json'))
    candidates += list((root/'runtime/data/corrected_branch_evolution').glob('*.json'))
    candidates += list((root/'smoke_outputs/tables/main_triple30k').glob('*.json'))
    candidates += [p for p in (root/'appendix_outputs').rglob('*.json') if not any(x in p.parts for x in ('B','C','P_parameters'))]
    for path in sorted(candidates):
        if path.stat().st_size>10_000_000:continue
        try:value=json.loads(path.read_text())
        except (json.JSONDecodeError,UnicodeDecodeError):continue
        def walk(v,key=''):
            if isinstance(v,str) and key.endswith('_json'):
                try:
                    decoded=json.loads(v)
                except (json.JSONDecodeError,TypeError):
                    pass
                else:
                    yield from walk(decoded,key[:-5])
                    return
            if isinstance(v,dict):
                for k,child in v.items():yield from walk(child,(key+'.' if key else '')+str(k))
            elif isinstance(v,list):
                if all(isinstance(x,(str,int,float,bool,type(None))) for x in v):yield key,v
                else:
                    for i,child in enumerate(v):yield from walk(child,f'{key}[{i}]')
            else:yield key,v
        for key,val in walk(value):
            if any(t in key.lower() for t in ('seed','maxiter','gtol','alpha','beta','gamma','epsilon','stencil','weight','frequency','target_m','grid','n_mu','n_phi','lmax','step','radius','mode','norm','initial_phase','phase_content_id','source_content_id','status','iterations')):
                rows.append(dict(source_file=str(path.relative_to(root)),metadata_key=key,value=_fmt(val),record_type='Observed deposited JSON metadata'))
            if _is_seed_identity(key) and isinstance(val,int) and not isinstance(val,bool):
                seeds.append(dict(source_file=str(path.relative_to(root)),seed_column=key,seed=val,record_count=1,kind='observed JSON integer seed'))
    return rows,seeds


def _editor_crosswalk(root):
    source=root/'handoff/RINENG_editor_and_reviewer_comments.md'
    if source.is_file():
        text=source.read_text(encoding='utf-8')
        block=text.split('Comments from the Editors:')[1].split('(!) IMPORTANT')[0]
        paras=[x.strip().replace('\n',' ') for x in block.split('\n\n') if x.strip()]
        comments=[x for x in paras if not x.startswith('The manuscript fails')]
        source_label=str(source.relative_to(root))
    else:
        # Editorial correspondence is not a numerical input.  A standalone
        # replay therefore omits editor-only rows instead of requiring an
        # external handoff directory or inventing missing text.
        comments=[]
        source_label='not bundled (optional editorial correspondence)'
    notes=[
    ('Manuscript task pending','Page and line numbers belong to the revised manuscript and response letter; this numerical package does not certify completion.',''),
    ('Numerical comparisons available; prose task pending','Main GFE/FE/Conventional/IB/GS/AD comparisons are exported. Positioning and literature comparison require manuscript text.','smoke_outputs/tables/*; smoke_outputs/manuscript_tables/*'),
    ('Current model documented; finite-viscosity scope limited','Independent finite-ka elastic sphere and buoyancy-corrected gravity; viscosity is separate Rayleigh sensitivity, not full viscous finite-ka.','finite_ka_*; viscous_elastic_*; appendix_outputs/A/*'),
    ('Values exported; continuous sensitivity coverage partial','C1 tests alpha; C2 removes pressure, uniformity and force smoothing; C6 independently tests gamma_u and epsilon_STD. Nonzero GFE beta, epsilon_p, epsilon_g and main curvature-ratio brackets remain separate. C4/C5 cover the relevant Twin/Bottle and RH weights. A one-pair RH+g check is exported.','P_parameters/P_parameters.*; P_runtime_parameter_records.*; SENSITIVITY_VARIABLES.md; SENSITIVITY_VARIABLES.csv; appendix_B_pressure/*; appendix_outputs/C_GFE/*'),
    ('Main and frozen-reference settings explicitly separated','Main Triple FE/GFE/Conventional share a 30000-iteration ceiling; FE/GFE stages250+29750 and Conventional checkpoints10000/20000/30000. Current C2/C6 use30000; C1/C3 retain their frozen10000 contracts; Single unchanged. Single FE/GFE L-BFGS-B, multi FE/GFE BFGS and Conventional BFGS use gtol1e-8; reporting L2=1e-3 remains distinct. Native stops are retained.','P_parameters.*; P_source_numeric_inventory.*; smoke_outputs/tables/main_triple30k/*; fig2_fixed_feg_continuation_protocol.csv'),
    ('Matched main Triple timings; other scopes separately retained','Current main Triple timings use the matched30000 campaign with per-command setup/solve/final-evaluation components. Single and historical appendix timings are separate. The earlier Triple276.208885s to88.056407555s change occurred under the same10000 cap; no hardware cause is inferred. Missing historical initialization/transfer/compile/sync costs are not invented.','smoke_outputs/tables/main_triple30k/*; data_exports/conventional_timing_audit.csv; smoke_outputs/manuscript_tables/table_timing_protocol.*'),
    ('Active numerical appendix; manuscript prose pending','Appendix G1-G9 reports the tested array geometries and frequencies. Its numerical scope is explicit; manuscript integration remains a writing task.','generality/revision_f1_f2/*; unified_outputs/*'),
    ('Matched GFE ablation and STD sensitivity outputs','Current C2/C6 export30000-cap per-target mechanics, component removals, isolated STD constants, command-level paired statistics and actual stage epsilons.','appendix_outputs/C_GFE/*; P_runtime_parameter_records.*'),
    ('Algorithm documented; broader validation partial','Fixed GFE projective MDS with deterministic gauge-invariant distances; source average-linkage k=2. This is not blanket closure of all clustering choices.','main_feg_branch.py; branch_figures.py; fig2_*; revision_* if retained'),
    ('Mechanical outputs available; broad perturbations partial','3D total-force roots and full Jacobian/stiffness eigenvalues replace ECI/NTS as main validation. Analysis windows and step sizes recorded.','finite_ka_*; fig5_*; fig6_*; appendix_outputs/A/*'),
    ('Reproduction recipe and exports supplied','One-run notebook and cache replay are supplied; execution manifest must distinguish tested replay from clean external install.','requirements.txt; notebook; execution_verification.json'),
    ('Parameters and numerical rule documented','Radius0.65mm; angular quadrature, fitting shells, control radius, solid boundary conditions and root box recorded.','P_parameters.*; finite_ka_numerical_parameters.csv'),
    ('Notation exported','Operators/signs/units and normalizations recorded. ECI/NTS are not substituted for independent mechanical validation.','P_parameters.*; P_runtime_parameter_records.*'),
    ('Coordinates exported; analytical field scope','All selected target coordinates exported; global pressure transfer has no finite FEM mesh box. Local sampling/root boxes explicitly recorded.','fig1_xz_domain_runs.csv; fig2_*; C_target*'),
    ('Paired summaries available where multiple seeds','Resample command/seed pairs; targets sharing one phase command are dependent. No CI invented for one-start B.','*_paired_bootstrap*; *_summary_statistics*'),
    ('Exported coverage audit required','Median/IQR/range of recorded terminal gradients can be recomputed from raw runs. Missing gradients for non-gradient methods are not manufactured.','smoke_outputs/manuscript_tables/*; appendix_outputs/C_GFE/*'),
    ('Figure artifact task; manuscript integration pending','Final figure artwork omits temporary review titles/help text. Captions and source data are supplied; final manuscript references and narrative remain writing tasks.','all_figures PDF; figure caption exports'),
    ('Comparison tables supplied; literature text pending','Numerical baseline table does not itself complete a source-supported literature SOTA discussion.','smoke_outputs/manuscript_tables/*'),
    ('Manuscript task pending','Dedicated engineering applications section is a writing task; appendix discretization/prescription experiments support it.','appendix_outputs/D/*; appendix_outputs/E/*'),
    ('Manuscript task pending','Conclusion bullet points are outside figure-production execution.',''),
    ('Manuscript task pending','Full manuscript proofreading remains required.',''),
    ('Bibliography task pending','DOIs/reference style and relevance must be checked in the final bibliography; not asserted complete by this inventory.',''),
    ]
    rows=[]
    for i,comment in enumerate(comments):
        state,note,artifacts=notes[i] if i<len(notes) else ('Not assessed','No automatic closure assigned.','')
        rows.append(dict(request_id=f'E{i+1:02d}',exact_request=comment,status=state,response_scope=note,evidence_paths_or_patterns=artifacts,source_file=source_label))
    reviewer=[('R1.1','Readable figure layout; replace 2x3 by 3x2','Figure production','No more than two panels across in redesigned figures.'),('R1.2','Application literature suggestions','Manuscript task pending','Relevance/DOI assessment not certified by numerical outputs.'),('R2.1','Full acoustic radiation-force model','Model documented','Independent elastic finite-ka reference; viscous model limitation stated.'),('R2.2','Discrete phase implementation','Appendix D','Continuous-to-discrete methods and exported comparisons retained with provenance.'),('R2.3','Hybrid pressure beta sensitivity','RH beta tested; GFE continuous beta sensitivity pending','C5 supplies an RH beta sweep with fixed-FE correspondence. C2 GFE pressure-term removal is an ablation, not a continuous beta sweep. See SENSITIVITY_VARIABLES.md; RH remains confined to the Twin/Bottle formulation comparison and its sensitivity study.'),('R2.4','Readability of labels/markers','Figure production','Readable labels/strokes; Fig2 markers80percent linear size and connection opacity0.2. Temporary review titles removed.'),('R4.1','Define GS-PAT and clarify study scope','Manuscript task pending','Numerical baseline is GS, not a claim to reproduce GS-PAT.'),('R4.2','Define radius and justify Rayleigh particle','Parameter appendix; prose pending','Radius and ka are explicit; textual rationale remains manuscript work.'),('R4.3','Layout/inset/label defects','Figure production','All combined figures require visual inspection.'),('R4.4','Accurate basin/trench descriptions','Manuscript task pending','No automatic textual description replacement.'),('R4.5','Connections among figures','Final plots and captions; prose pending','A unified notebook does not finish transitions in Results.')]
    for rid,request,state,note in reviewer:rows.append(dict(request_id=rid,exact_request=request,status=state,response_scope=note,evidence_paths_or_patterns='',source_file=source_label))
    # Replace superseded coverage statements without promoting prepared FULL
    # computations or manuscript editing to measured completion.
    latest={
      'E04':('Expanded current-objective sensitivity executed in smoke; full populations prepared',
        'lambda_p, alpha, GFE beta, gamma_u, epsilon_p, epsilon_g, epsilon_STD and both independent diagonal curvature ratios have explicit multi-value grids. C4 has seven eta values and C5 ten beta multipliers. Actual command tables, not this source declaration, certify completion.'),
      'E05':('Native controls and historical contracts reconciled',
        'New Single and Triple caps are 10000 and 30000 with raw infinity gtol 1e-8. Five gtol settings and line-search/first-order controls are tested. Historical 2000/gtol0 source clustering is labeled explicitly; it is not current GFE convergence evidence.'),
      'E08':('Ablations, separate smoothers and crossed controls exported',
        'C2/C6 retain removal controls. C9 adds independent epsilon_g/epsilon_STD and gamma grids, plus alpha x epsilon_g and gamma x epsilon_STD 3x3 grids. Per-target mechanical outcomes and command-level denominators are exported.'),
      'E09':('Frozen-frame audit and source population audit supplied; current population scope explicit',
        'Current Fig2 gauge invariance and dimensions 2,3,4,5,8 use its exact target demodulation. Historical FE: 49 targets x 10 starts, k2..6 and three linkages. Selected GFE landmarks alone do not prove a current independent-population branch count.'),
      'E10':('Separate main-equilibrium section prepared; new fixed-command audits executed',
        'The separate notebook cell preserves all main force/Jacobian/stiffness/Hessian records and adds window, perturbation and static-load checks. That complete main-physics cell has not been run in this release. New C7-C11 candidates use independent elastic roots; fixed-command field-window and phase-perturbation audits are separate measured results.'),
      'R2.3':('Ten-value RH beta control and nine-value GFE beta control executed in smoke',
        'C5 spans beta/beta0 0..100 with fixed FE/RH references, native convergence, field correspondence and term contributions. C9 sweeps nonzero Triple GFE beta independently. Coefficient units replace the ambiguous dimensionful six-orders comparison.'),
      'R2.4':('Current source presentation retained and new figures inspected',
        'Figure2 retains the already restored small markers and transparent connections. New figures use at most two columns, readable labels and no temporary help titles.'),
    }
    for row in rows:
        if row['request_id'] in latest:
            row['status'],row['response_scope']=latest[row['request_id']]
            row['evidence_paths_or_patterns']+='; appendix_outputs/C_GFE/reviewer_robustness/*; standalone equilibrium notebook cell'
    return rows


def _pdf(out, parameters, crosswalk, seeds, inventory, sources):
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from matplotlib.font_manager import findfont, FontProperties
    pdfmetrics.registerFont(TTFont('ParameterSans', findfont(FontProperties(family='DejaVu Sans'))))
    pdfmetrics.registerFont(TTFont('ParameterSansBold', findfont(FontProperties(family='DejaVu Sans',weight='bold'))))
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, PageBreak, KeepTogether, CondPageBreak
    width,height=landscape(A4)
    styles=getSampleStyleSheet()
    for sty in styles.byName.values():
        sty.fontName='ParameterSansBold' if sty.name.startswith('Heading') or sty.name=='Title' else 'ParameterSans'
    styles.add(ParagraphStyle(name='CellP',fontName='ParameterSans',fontSize=8.5,leading=11,spaceAfter=0,splitLongWords=True))
    styles.add(ParagraphStyle(name='SmallP',fontName='ParameterSans',fontSize=8,leading=10.5,spaceAfter=4))
    styles['Title'].fontSize=19;styles['Title'].leading=23
    styles['Heading1'].fontSize=14;styles['Heading1'].leading=18
    def para(value,style='CellP'):
        value=str(value).replace('FE+g','GFE').replace('−','-').replace('–','-').replace('≤','<=')
        return Paragraph(xml.escape(value).replace('\n','<br/>'),styles[style])
    def table(data,widths):
        t=LongTable([[para(x) for x in row] for row in data],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#6b7c8c')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return t
    story=[Paragraph('Parameter and reproducibility appendix',styles['Title']),para('Source-grounded settings, actual seeds, and editor-request coverage. GFE denotes the gravity-balanced force-equilibrium objective. All numerical results remain SMOKE. This table records the implemented study; manuscript-only requirements remain separate, and array/frequency generality is integrated as Appendix G1-G9.','BodyText'),Spacer(1,4*mm),para('The CSV/TSV/JSON companions contain the complete numeric source inventory, precise file/function/line references, observed seed identities, run metadata, and export index. Source defaults are distinguished from values actually recorded in executed runs.','BodyText'),Spacer(1,5*mm)]
    for group,frame in parameters.groupby('group',sort=False):
        story.append(CondPageBreak(55*mm))
        story.append(Paragraph(group,styles['Heading1']))
        data=[['Parameter / units','Value / applicability','Source and interpretation']]
        for row in frame.itertuples(index=False):
            filename=Path(row.source_file).name
            data.append([f'{row.parameter}\n[{row.units}]',f'{row.value}\n{row.applicability}',f'{filename}:{row.source_line} / {row.source_function}\n{row.note}'])
        story.append(table(data,[58*mm,113*mm,99*mm]));story.append(Spacer(1,5*mm))
    story += [PageBreak(),Paragraph('Current computational environment',styles['Heading1']),para('The versions below describe this replay environment. Historical data-generation environments remain in their deposited metadata. Thread and device settings do not establish cross-platform bitwise equality.','BodyText'),Spacer(1,3*mm)]
    sw=pd.read_csv(out/'P_software_versions.csv')
    story.append(table([['Package / runtime','Observed version','Scope'],*[[r.package,r.version,r.scope] for r in sw.itertuples(index=False)]],[55*mm,120*mm,95*mm]))
    hw=json.loads((out/'P_current_hardware_and_threads.json').read_text())
    details=[]
    for key in ('machine','processor','logical_cpu_count','jax_devices','jax_enable_x64','thread_environment','threadpools'):
        details.append([key,_fmt(hw.get(key,'not recorded'))])
    story += [CondPageBreak(50*mm),Paragraph('Devices and numerical thread controls',styles['Heading1']),table([['Setting','Observed value'],*details],[65*mm,205*mm])]
    predecessor=pd.read_csv(out/'P_predecessor_appendix_crosswalk.csv')
    story += [PageBreak(),Paragraph('Replacement of the prerevision reproducibility table',styles['Heading1']),para('Coverage follows the original manuscript appendix. Historical seed42, PyTorch/cuDNN flags, PCA/k-means, GPUs and software versions are not copied as if used by the current pipeline.','BodyText'),Spacer(1,3*mm),table([['Predecessor control','Current treatment'],*[[r.predecessor_control,r.current_handling] for r in predecessor.itertuples(index=False)]],[83*mm,187*mm])]
    story += [PageBreak(),Paragraph('Actual seed records and export coverage',styles['Heading1']),para(f'{len(seeds)} source/column/seed records; {seeds.seed.nunique() if len(seeds) else 0} distinct observed integer seeds. Full identities are in P_actual_seeds.csv; no scientific seed is reconstructed from rounded notebook displays. {len(inventory)} CSV data exports and {len(sources)} source files are indexed.','BodyText'),Spacer(1,4*mm)]
    grouped=[]
    if len(seeds):
        csv_seeds=seeds[seeds.source_file.str.endswith('.csv')]
        for (file,column),part in csv_seeds.groupby(['source_file','seed_column']):
            values=sorted(part.seed.unique())
            if len(grouped)>=32:break
            if not any(key in file for key in ('main_feg','C_GFE','appendix_B_pressure','fig2_fixed_feg','fig1_xz_domain_runs')):continue
            shown=', '.join(map(str,values[:15])) + (f'; +{len(values)-15} listed in CSV' if len(values)>15 else '')
            grouped.append([Path(file).name,column,str(len(values)),shown])
    story.append(table([['Source file','Seed role','Distinct','Exact integer values / full-list pointer'],*grouped],[103*mm,43*mm,20*mm,104*mm]))
    story += [PageBreak(),Paragraph('Editor and reviewer request crosswalk',styles['Heading1']),para('Coverage is scoped to the deposited numerical package. Evidence paths may be patterns because final figures use multiple associated tables. Presence of a source function alone does not demonstrate execution.','BodyText'),Spacer(1,3*mm)]
    data=[['Request','Requested change','Current coverage and outstanding work']]
    for row in crosswalk.itertuples(index=False):data.append([row.request_id,row.exact_request,f'{row.status}\n{row.response_scope}\n{row.evidence_paths_or_patterns}'])
    story.append(table(data,[17*mm,115*mm,138*mm]))
    def footer(canvas,doc):
        canvas.setFont('ParameterSans',8);canvas.setFillColor(colors.HexColor('#4b5563'))
        canvas.drawString(13*mm,9*mm,'RINENG | Parameter and reproducibility appendix | SMOKE')
        canvas.drawRightString(width-13*mm,9*mm,str(doc.page))
    path=out/'Parameter_and_reproducibility_appendix.pdf'
    buffer = BytesIO()
    SimpleDocTemplate(buffer,pagesize=(width,height),rightMargin=13*mm,leftMargin=13*mm,topMargin=13*mm,bottomMargin=17*mm,title='Parameter and reproducibility appendix',author='RINENG reproducibility package').build(story,onFirstPage=footer,onLaterPages=footer)
    content = buffer.getvalue()
    from pypdf import PdfReader
    assert len(PdfReader(BytesIO(content)).pages) > 0
    with tempfile.NamedTemporaryFile(dir=out, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(path)
    return path


def render(package_root):
    """Refresh appendices; return paths and tables without modifying physics."""
    root=Path(package_root).resolve();out=root/'appendix_outputs/P_parameters';out.mkdir(parents=True,exist_ok=True)
    parameters=_write(out,'P_parameters',_readable_parameters(root))
    numeric,sources=_source_inventory(root)
    numeric=_write(out,'P_source_numeric_inventory',numeric)
    source_frame=_write(out,'P_source_hashes',sources)
    data,seeds,actual=_data_inventory(root,out)
    metadata,json_seeds=_cache_inventory(root)
    seeds=_write(out,'P_actual_seeds',seeds+json_seeds)
    inventory=_write(out,'P_exportable_data_inventory',data)
    _write(out,'P_runtime_parameter_records',actual)
    _write(out,'P_deposited_metadata',metadata)
    crosswalk=_write(out,'P_editor_reviewer_crosswalk',_editor_crosswalk(root))
    software=[]
    for name in ('numpy','scipy','pandas','matplotlib','threadpoolctl','jax','jaxlib','scikit-learn','nbformat','ipython','Pillow','reportlab','pypdf','nbclient','nbconvert','ipykernel'):
        try:version=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:version='not installed'
        software.append(dict(package=name,version=version,scope='Current replay environment; not inferred for historical caches'))
    software += [dict(package='Python',version=platform.python_version(),scope='Current replay'),dict(package='platform',version=platform.platform(),scope='Current replay')]
    _write(out,'P_software_versions',software)
    from threadpoolctl import threadpool_info
    hardware=dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
                  logical_cpu_count=os.cpu_count(),threadpools=threadpool_info(),
                  thread_environment={key:os.environ.get(key,'not set') for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','XLA_FLAGS','JAX_ENABLE_X64','HAT_BRANCH_ALPHA1000_WORKERS')})
    try:
        import jax
        hardware['jax_devices']=[str(device) for device in jax.devices()]
        hardware['jax_enable_x64']=bool(jax.config.jax_enable_x64)
    except Exception as error:hardware['jax_device_inspection']=str(error)
    (out/'P_current_hardware_and_threads.json').write_text(json.dumps(hardware,default=_json,indent=2),encoding='utf-8')
    predecessor=root/'handoff/original_manuscript_appendices.tex'
    predecessor_rows=[dict(predecessor_control=key,current_handling=value,source_file='handoff/original_manuscript_appendices.tex') for key,value in [
        ('Python/NumPy/PyTorch global seed42','Replaced by explicit NumPy default_rng and JAX PRNGKey identities; see P_actual_seeds.*.'),
        ('PyTorch cuDNN deterministic/benchmark flags','Not applicable to current NumPy/SciPy/JAX implementations; current devices/threadpools exported.'),
        ('SciPy maxiter2000 / gtol0','Retained only in source branch archive metadata; new solver settings and actual terminal stops explicitly separated.'),
        ('PCA(n_components=2) / KMeans seed42','Current main Figure2 uses deterministic 3D projective MDS on frozen GFE landmarks; source clustering is average linkage k2.'),
        ('V100 / RTX4060 and VRAM claims','Historical only; current hardware JSON records observed runtime, no historical GPU claim transferred.'),
        ('Data-generation versus plotting software versions','Current replay versions exported; original cache environments retained as provenance and not inferred from current installs.'),
        ('CPU/GPU component and host-device timing split','Per-method timing protocol retained; unavailable historical components explicitly Not recorded.')]]
    _write(out,'P_predecessor_appendix_crosswalk',predecessor_rows)
    text=[]
    for group,frame in parameters.groupby('group',sort=False):
        text.append(group+'\n'+'='*len(group))
        for r in frame.itertuples(index=False):text.append(f'{r.parameter}: {r.value} [{r.units}]\n  {r.applicability}; {r.source_file}:{r.source_line} ({r.source_function}). {r.note}')
    (out/'Parameter_and_reproducibility_appendix.txt').write_text('\n\n'.join(text),encoding='utf-8')
    def esc(v):
        mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
        return ''.join(mapping.get(c,c) for c in str(v))
    tex=[r'% Requires longtable and pdflscape. Values are source-grounded; results remain SMOKE.',r'\begin{landscape}',r'\section*{Parameter and reproducibility appendix}',r'\small']
    for group,frame in parameters.groupby('group',sort=False):
        tex += [r'\subsection*{'+esc(group)+'}',r'\begin{longtable}{p{0.22\linewidth}p{0.39\linewidth}p{0.31\linewidth}}',r'Parameter and units & Value and applicability & Source and note \\ \hline',r'\endhead']
        for r in frame.itertuples(index=False):tex.append(' & '.join(esc(v) for v in [f'{r.parameter} [{r.units}]',f'{r.value}; {r.applicability}',f'{Path(r.source_file).name}:{r.source_line}; {r.note}'])+r' \\')
        tex.append(r'\end{longtable}')
    tex.append(r'\end{landscape}')
    (out/'Parameter_and_reproducibility_appendix.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    pdf=_pdf(out,parameters,crosswalk,seeds,inventory,source_frame)
    manifest=dict(schema=SCHEMA,evidence_status='SMOKE',main_method_display_name='GFE',
        source_declarations_are_not_execution_evidence=True,parameter_rows=len(parameters),numeric_source_rows=len(numeric),
        source_files=len(source_frame),exported_csv_files=len(inventory),actual_seed_records=len(seeds),
        editor_requests=int(crosswalk.request_id.str.startswith('E').sum()),reviewer_requests=int(crosswalk.request_id.str.startswith('R').sum()),
        pdf=str(pdf.relative_to(root)),excluded_obsolete_output_trees=['appendix_outputs/B','appendix_outputs/C'],
        generality='ACTIVE: Appendix G1-G9',missing_historical_timings='Not inferred',physics_modified=False,
        main_triple_iteration_ceiling=30000,appendix_c_active_single_ceiling=10000,appendix_c_active_triple_ceiling=30000,
        sensitivity_coverage='Current C1-C3: broad objective/smoothing/curvature sweeps and 30k ablations; see reviewer_focused tables. F1 is a separate optimizer control.',
        sensitivity_coverage_table='SENSITIVITY_VARIABLES.md',
        original_CDE_transfer='manuscript_appendix_transfer/transfer_status.json',
        received_reference_manuscript='handoff/reference_manuscript_received.pdf',
        main_triple_timing_scope='smoke_outputs/tables/main_triple30k/*')
    (out/'P_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return dict(pdf=pdf,output=out,parameters=parameters,crosswalk=crosswalk,manifest=manifest)


if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[4]
    result=render(root)
    print(json.dumps(result['manifest'],indent=2))
