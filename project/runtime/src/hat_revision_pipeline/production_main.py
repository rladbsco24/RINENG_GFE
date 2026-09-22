"""Explicit production sampling for the eight established GFE main figures.

No computation is performed on import. All synthesis uses the deposited solver
implementations and native stopping rules. Timed records belong to the current
machine campaign; archived phase-only references remain a separate cache source.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace, is_dataclass, fields
from pathlib import Path

from rineng_content_id import content_identity
import json
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import pipeline as p, final_figures as ff, main_feg as mf
from . import main_feg_branch as branch_source, fig1_domain_volume as domain
from .cache import CacheStore, CacheMissError, digest_array, digest_payload, dependency_snapshot
from .config import RunConfig
from .gorkov_core import Condition, MethodSpec, SingleTargetObjective, method_spec
from .compact_single import CompactSingleObjective
from .sota import SynthesisResult, solve_corrected_gorkov_fe, vortex_ring_metrics
from .multitrap import (
    SphereInFluid,
    effective_weight_force_target_n,
    run_multitrap,
    evaluate_multitrap_objective,
)
from .triple_long_iteration import _configs, _run_cumulative_conventional, CHECKPOINTS, FIXED_BASE_TARGETS_M
from .fig4_threepoint_selection import make_problem
from .branch_figures import projective_similarity, target_demodulated_states, COMPONENTS
from .fixed_fe_endpoints import fixed_single_fe_method_spec

SINGLE30 = (172817509, 425727192, 469986874, 519693606, 681979972,
            693201818, 713854823, 731661603, 764705224, 991245467,
            *range(260828, 260848))
TRIPLE30 = tuple(range(260869, 260899))
ALPHAS = (1e3, 1e6, 1e7)
METHODS = ('Conventional', 'FE', 'GFE', 'IB', 'GS', 'AD')
ENGINEERING_LBFGS_OPTIONS = dict(gtol=1e-8, ftol=0.0, maxcor=10, maxls=40, maxfun=1000000)
SCHEMA = 'gfe-production-main-v2'
TIMING_PROTOCOL_ID = 'main-fig8-comparable-end-to-end-v2'
TIMING_SCOPE = (
    'fresh command construction from declared inputs through terminal host phase: '
    'problem/objective initialization, acoustic transfer construction, applicable '
    'host-to-device transfer, compilation/warmup, device synchronization, native '
    'solve, and result extraction included; independent mechanical validation, '
    'cache retrieval, table export, and rendering excluded'
)
_ORIGINAL_FIXED_BRANCH = branch_source.fixed_feg_branch_pack
_ORIGINAL_SINGLE_BENCHMARK = ff._exact_benchmark_with_concentration
_ORIGINAL_TRIPLE_BENCHMARK = ff._triple_exact_benchmark_with_concentration


def _study_scale(ctx):
    """Sampling controls only; objective, discretization and solver caps are unchanged."""
    scale = dict(ctx.memo.get('study_scale', {}))
    for name, default in (
        ('first_result_seed_count', 16), ('second_result_seed_count', 100),
        ('last_result_seed_count', 5), ('deterministic_timing_repeats', 5),
        ('first_result_domain_x_points', 5), ('first_result_domain_z_points', 3),
        ('first_result_domain_restarts', 1),
    ):
        value = int(scale.get(name, default))
        if value < 1:
            raise ValueError(f'{name} must be a positive integer')
        scale[name] = value
    if min(scale['first_result_domain_x_points'], scale['first_result_domain_z_points']) < 2:
        raise ValueError('Each first-result domain axis needs at least 2 points for the XZ panel')
    scale.setdefault('profile', 'legacy-production')
    return scale


def _selected_seeds(base, count, *, include=None):
    """Keep the deposited prefix and extend it with reproducible independent seeds."""
    seeds = list(base[:count])
    rng = np.random.default_rng(20260918)
    while len(seeds) < count:
        seed = int(rng.integers(1, 2**31 - 1))
        if seed not in seeds:
            seeds.append(seed)
    if include is not None and include not in seeds:
        seeds[-1] = int(include)
    return tuple(seeds)


def _first_result_seeds(ctx):
    display_seed = (ctx.memo.get('production_single_display_seed', 260828)
                    if ctx.memo.get('production_main') else None)
    return _selected_seeds(SINGLE30, _study_scale(ctx)['first_result_seed_count'],
                           include=display_seed)


def _benchmark_seeds(ctx, task):
    return _selected_seeds(SINGLE30 if task == 'Single' else TRIPLE30,
                           _study_scale(ctx)['last_result_seed_count'])


def _scale_key(ctx, stem):
    return stem + ':' + digest_payload(_study_scale(ctx))[:16]


def configure_study_scale(ctx, scale):
    """Apply an explicit notebook profile without performing any numerical work."""
    previous = _study_scale(ctx)
    ctx.memo['study_scale'] = dict(scale)
    configured = _study_scale(ctx)
    ctx.memo['study_scale'] = configured
    if previous != configured:
        for key in list(ctx.memo):
            if any(str(key).startswith(stem) for stem in
                   ('production_commands', 'production_benchmarks', 'production_xz_bank', 'production_fixed_branch')):
                ctx.memo.pop(key)
        for stem in list(ctx.figures):
            if str(stem).startswith(('Figure_1_', 'Figure_2_', 'Figure_8_')):
                ctx.figures.pop(stem)
    ctx.memo['production_single_seeds'] = _benchmark_seeds(ctx, 'Single')
    ctx.memo['production_triple_seeds'] = _benchmark_seeds(ctx, 'Triple')
    path = ctx.output_root / 'production_main_settings.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        count = configured['last_result_seed_count']
        repeats = configured['deterministic_timing_repeats']
        x_points = configured['first_result_domain_x_points']
        z_points = configured['first_result_domain_z_points']
        manifest.update(study_scale=configured,
            first_result_single_seeds=list(_first_result_seeds(ctx)),
            first_result_paired_initializations=configured['first_result_seed_count'],
            second_result_paired_initializations=configured['second_result_seed_count'],
            single_seeds=list(_benchmark_seeds(ctx, 'Single')),
            triple_seeds=list(_benchmark_seeds(ctx, 'Triple')),
            nominal_command_count=8 * count + 4,
            nominal_target_validation_count=16 * count + 8,
            deterministic_timing_repeats=repeats,
            timing_observations_per_stochastic_method_task=count,
            timing_observations_per_deterministic_method_task=repeats,
            figure1_xz_shape=[x_points, z_points],
            figure1_xz_target_count=x_points * z_points,
            figure1_xz_restarts=configured['first_result_domain_restarts'],
            exact_protocol=p._finite_ka_protocol(ctx))
        manifest.pop('timing_observations_per_method_task', None)
        path.write_text(json.dumps(manifest, indent=2, default=str))
    return ctx


@dataclass(frozen=True)
class ComparableTiming:
    """One fresh command observation under the common Figure 8 boundary.

    ``None`` means a separately measured substage is not available or is not
    applicable; it is never serialized as a numeric zero.  Accelerated stages
    that a native solver measures only as one synchronized interval are marked
    as bundled in ``solve_s``.  Their work is therefore included exactly once
    in ``total_s`` even though no fabricated decomposition is reported.
    """

    protocol_id: str
    scope: str
    backend: str
    problem_initialization_s: float
    acoustic_transfer_construction_s: float
    device_transfer_s: float | None
    compilation_warmup_s: float | None
    pre_solve_synchronization_s: float | None
    solve_s: float
    post_solve_synchronization_s: float | None
    result_extraction_s: float
    orchestration_overhead_s: float
    total_s: float
    device_transfer_status: str
    compilation_warmup_status: str
    pre_solve_synchronization_status: str
    post_solve_synchronization_status: str
    decomposition_complete: bool
    comparable_total: bool
    exclusion_reason: str | None


@dataclass(frozen=True)
class ProductionBenchmarkRun:
    """Minimal benchmark record retained by the production command bank."""

    method_id: str
    display_name: str
    initialization_kind: str
    target_m: np.ndarray
    phase_rad: np.ndarray
    timing: ComparableTiming
    solver_result: SynthesisResult
    metadata: dict[str, object]


def _elapsed_call(function):
    started = time.perf_counter()
    value = function()
    return value, float(time.perf_counter() - started)


def _execute_comparable_timing(
    initializer,
    transfer_builder,
    solver,
    extractor,
    *,
    backend: str,
    accelerated: bool = False,
):
    """Execute one command with the same inclusive boundary for every method."""

    total_started = time.perf_counter()
    initialized, initialization_s = _elapsed_call(initializer)
    prepared, transfer_s = _elapsed_call(lambda: transfer_builder(initialized))
    solved, solve_s = _elapsed_call(lambda: solver(prepared))
    extracted, extraction_s = _elapsed_call(lambda: extractor(solved, prepared))
    total_s = float(time.perf_counter() - total_started)
    measured = initialization_s + transfer_s + solve_s + extraction_s
    overhead_s = max(0.0, total_s - measured)

    metadata = dict(getattr(solved, 'metadata', {}) or {})
    if accelerated:
        device_transfer_confirmed = bool(metadata.get('device_transfer_included', False))
        compilation_confirmed = bool(metadata.get('jit_included', False))
        synchronization_confirmed = bool(metadata.get('device_synchronization_included', False))
        comparable = device_transfer_confirmed and compilation_confirmed and synchronization_confirmed
        missing = []
        if not device_transfer_confirmed:
            missing.append('host-to-device transfer inclusion not confirmed')
        if not compilation_confirmed:
            missing.append('compilation/warmup inclusion not confirmed')
        if not synchronization_confirmed:
            missing.append('device synchronization inclusion not confirmed')
        included = 'included_unseparated_in_native_solve'
        unconfirmed = 'unconfirmed; observation excluded from comparison'
        timing = ComparableTiming(
            protocol_id=TIMING_PROTOCOL_ID,
            scope=TIMING_SCOPE,
            backend=str(backend),
            problem_initialization_s=initialization_s,
            acoustic_transfer_construction_s=transfer_s,
            device_transfer_s=None,
            compilation_warmup_s=None,
            pre_solve_synchronization_s=None,
            solve_s=solve_s,
            post_solve_synchronization_s=None,
            result_extraction_s=extraction_s,
            orchestration_overhead_s=overhead_s,
            total_s=total_s,
            device_transfer_status=included if device_transfer_confirmed else unconfirmed,
            compilation_warmup_status=included if compilation_confirmed else unconfirmed,
            pre_solve_synchronization_status=included if synchronization_confirmed else unconfirmed,
            post_solve_synchronization_status=included if synchronization_confirmed else unconfirmed,
            decomposition_complete=False,
            comparable_total=bool(comparable),
            exclusion_reason='; '.join(missing) if missing else None,
        )
    else:
        status = 'not_applicable_synchronous_cpu'
        timing = ComparableTiming(
            protocol_id=TIMING_PROTOCOL_ID,
            scope=TIMING_SCOPE,
            backend=str(backend),
            problem_initialization_s=initialization_s,
            acoustic_transfer_construction_s=transfer_s,
            device_transfer_s=None,
            compilation_warmup_s=None,
            pre_solve_synchronization_s=None,
            solve_s=solve_s,
            post_solve_synchronization_s=None,
            result_extraction_s=extraction_s,
            orchestration_overhead_s=overhead_s,
            total_s=total_s,
            device_transfer_status=status,
            compilation_warmup_status=status,
            pre_solve_synchronization_status=status,
            post_solve_synchronization_status=status,
            decomposition_complete=True,
            comparable_total=True,
            exclusion_reason=None,
        )
    return extracted, timing


def _effective_timing_record(timing: ComparableTiming, ctx=None) -> dict[str, object]:
    """Apply machine-contract eligibility to one native timing record."""
    record = asdict(timing)
    comparable = bool(record['comparable_total'])
    exclusion_reason = record['exclusion_reason']
    if ctx is not None:
        machine = ctx.memo.get('production_machine', {})
        if not bool(machine.get('blas_thread_contract_satisfied', False)):
            comparable = False
            detail = 'recorded BLAS thread pools do not satisfy the declared timing contract'
            exclusion_reason = '; '.join(v for v in (exclusion_reason,detail) if v)
        requested = str(machine.get('requested_device','auto'))
        backend = str(machine.get('jax_backend','unknown'))
        if requested in ('cpu','gpu') and backend != requested:
            comparable = False
            detail = f'requested JAX device {requested!r} resolved to {backend!r}'
            exclusion_reason = '; '.join(v for v in (exclusion_reason,detail) if v)
    record['comparable_total']=comparable
    record['exclusion_reason']=exclusion_reason
    return record


def _timing_columns(timing: ComparableTiming, ctx=None) -> dict[str, object]:
    """Flatten the authoritative timing record without replacing N/A by zero."""

    record = _effective_timing_record(timing,ctx)
    comparable = bool(record.pop('comparable_total'))
    exclusion_reason = record.pop('exclusion_reason')
    return {
        'timing_protocol_id': record.pop('protocol_id'),
        'timing_scope': record.pop('scope'),
        'timing_backend': record.pop('backend'),
        'timing_comparable': comparable,
        'timing_exclusion_reason': exclusion_reason,
        **{f'timing_{key}': ('N/A' if value is None and key.endswith('_s') else value)
           for key, value in record.items()},
        'command_time_s': float(timing.total_s),
    }


def _timing_json(timing: ComparableTiming, ctx=None) -> str:
    """Serialize the same effective eligibility exposed by flattened columns."""

    return json.dumps(_effective_timing_record(timing,ctx),default=str)


class ProductionCache(CacheStore):
    """Reuse exact immutable non-timing inputs without importing smoke times."""
    IMMUTABLE_KINDS = frozenset({'alpha10_feg_branch_medoid_continuation',
                                'decision_endpoint_run'})

    def __init__(self, root, study_root, *, read_only=False):
        source_root=Path(__file__).parent
        self.source_hashes={name:content_identity((source_root/name).read_bytes()).hexdigest()
                            for name in ('production_main.py','pipeline.py','fixed_fe_endpoints.py',
                                         'gorkov_core.py','multitrap.py','sota.py',
                                         'compact_single.py','compact_multitrap.py',
                                         'diff_pat.py','exact_validator.py','triple_long_iteration.py',
                                         'fig4_threepoint_selection.py')}
        from .fe_solver_policy import compatible_source_ids, COMPATIBLE_PRODUCTION_NAMESPACES
        # Solver-only edits are represented in each affected command key.
        # Preserve the previous namespace for unrelated physics and baselines.
        cache_sources = compatible_source_ids(self.source_hashes)
        super().__init__(root, namespace=SCHEMA+'-'+digest_payload(cache_sources)[:16], read_only=read_only)
        self.compatible_stores = [CacheStore(root, namespace=namespace, read_only=True)
                                  for namespace in COMPATIBLE_PRODUCTION_NAMESPACES
                                  if namespace != self.namespace]
        self.immutable = CacheStore(Path(study_root) / 'smoke_outputs/cache',
                                    namespace=p.PIPELINE_IMPLEMENTATION, read_only=True)

    def get_or_compute(self, kind, payload, producer, *, recompute=False):
        def compute():
            if kind in self.IMMUTABLE_KINDS and not recompute:
                try:
                    value, _, source = self.immutable.get_or_compute(kind, payload, producer)
                    self.access_records.append(dict(kind=kind, status='immutable phase reference', path=str(source)))
                    return value
                except CacheMissError:
                    pass
            return producer()
        result = None
        if not recompute and not self.path_for(kind, payload, create_parent=False).is_file():
            for store in self.compatible_stores:
                try:
                    result = store.get_or_compute(kind, payload, producer)
                    self.access_records.append(dict(kind=kind, status='cached compatible namespace',
                                                    path=str(result[2])))
                    break
                except CacheMissError:
                    continue
        if result is None:
            result = super().get_or_compute(kind, payload, compute, recompute=recompute)
        value = result[0]
        mapping = value if isinstance(value, dict) else {f.name:getattr(value,f.name) for f in fields(value)} if is_dataclass(value) else {}
        arrays = {str(key): item for key, item in mapping.items()
                  if isinstance(item, np.ndarray) and item.dtype.kind != 'O'}
        if isinstance(value, (tuple, list)):
            arrays.update({f'array_{index}':item for index,item in enumerate(value)
                           if isinstance(item,np.ndarray) and item.dtype.kind != 'O'})
        if arrays and not self.read_only:
            output = self.root.parent / 'numerical_arrays' / kind
            output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output / (result[2].stem+'.npz'), **arrays,
                                payload_json=json.dumps(payload,default=str))
        return result


def _machine_identity(settings):
    """Fingerprint the actual backend without running a synthesis command."""
    host = dict(dependency_snapshot(), machine=platform.machine(), processor=platform.processor(), logical_cpu_count=os.cpu_count(),
                requested_blas_threads=settings['blas_threads'],
                timing_clock='time.perf_counter',
                timing_clock_resolution_s=time.get_clock_info('perf_counter').resolution,
                timing_clock_monotonic=time.get_clock_info('perf_counter').monotonic)
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
    except (ImportError, OSError):
        pools=[]
    host['threadpools']=[{key:item.get(key) for key in
        ('user_api','internal_api','prefix','version','num_threads','threading_layer','architecture')}
        for item in pools]
    host['blas_thread_contract_satisfied']=all(
        item.get('num_threads') in (None,int(settings['blas_threads'])) for item in pools)
    from importlib.metadata import version, PackageNotFoundError, distributions
    for package in ('jax','jaxlib'):
        try: host[package] = version(package)
        except PackageNotFoundError: host[package] = 'not installed'
    host['requested_device'] = settings['device']
    host['jax_platform_request'] = os.environ.get('JAX_PLATFORMS', '')
    # The requested "auto" mode is insufficient to identify a timing machine.
    # Query the initialized backend before the campaign/cache key is constructed.
    import jax
    host['jax_backend'] = str(jax.default_backend())
    host['jax_enable_x64'] = bool(jax.config.jax_enable_x64)
    host['jax_devices'] = [dict(platform=str(device.platform),
        device_kind=str(device.device_kind), id=int(device.id),
        process_index=int(device.process_index),
        platform_version=str(getattr(device.client, 'platform_version', 'unreported')))
        for device in jax.devices()]
    host['accelerator_package_versions'] = {
        str(dist.metadata['Name']): str(dist.version) for dist in distributions()
        if str(dist.metadata.get('Name', '')).lower().startswith(('jax-cuda', 'nvidia-'))}
    return host


def prepare(root, settings=None):
    """Create a full numerical context; optimization starts only in producers."""
    root = Path(root).resolve()
    from production_settings import load_settings
    settings = load_settings(root, settings)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = str(settings['blas_threads'])
    os.environ['HAT_FIG1_DOMAIN_WORKERS'] = '1'
    os.environ['HAT_HIDE_MODE_NOTE'] = '1'
    os.environ['HAT_PROTOTYPE_LABEL'] = '0'
    output = root / 'production_outputs'
    ctx = p.prepare_pipeline(run_mode='full', recompute=False, output_root=output)
    ctx.config = replace(ctx.config, multitrap_maxiter=30000)
    ctx.cache = ProductionCache(output / 'cache', root)
    host = _machine_identity(settings)
    if not host['blas_thread_contract_satisfied']:
        raise RuntimeError(
            'Production timing requires the declared BLAS thread count before NumPy/SciPy '
            'initialization. Restart the kernel and enter through run_notebook.prepare or '
            'run_production.prepare before importing scientific libraries.')
    if settings['device'] in ('cpu','gpu') and host['jax_backend'] != settings['device']:
        raise RuntimeError(
            f"Requested JAX device {settings['device']!r} resolved to "
            f"{host['jax_backend']!r}; no timing campaign was started.")
    campaign = settings['campaign_id'] + '-' + digest_payload(host)[:20]
    ctx.memo.update(production_root=root, production_settings=settings,
                    production_timing_campaign=str(campaign), production_machine=host,
                    benchmark_timing_campaign=settings['benchmark_campaign_id'] + '-' + digest_payload(host)[:20],
                    production_single_seeds=tuple(SINGLE30), production_triple_seeds=tuple(TRIPLE30),
                    production_main=True, production_single_display_seed=260828)
    manifest = dict(schema=SCHEMA, status='configured; execution starts in figure cells',
        single_seeds=list(SINGLE30), triple_seeds=list(TRIPLE30), single_cap=10000,
        triple_cap=30000, triple_checkpoints=list(CHECKPOINTS),
        field_samples=181, local_plane_samples=101, figure1_xz_shape=[7,7],
        figure1_xz_restarts=3, nominal_command_count=244,
        nominal_target_validation_count=488, deterministic_timing_repeats=30,
        machine=host, timing_campaign=campaign, source_hashes=ctx.cache.source_hashes,
        benchmark_timing_campaign=ctx.memo['benchmark_timing_campaign'],
        timing_protocol_id=TIMING_PROTOCOL_ID, timing_scope=TIMING_SCOPE,
        timing_observations_per_method_task=30,
        timing_total_ranking_policy='only timing_comparable=true observations from the common protocol',
        timing_missing_component_policy='JSON null / CSV N/A; never numeric zero',
        exact_protocol=p._finite_ka_protocol(ctx),
        generality='ACTIVE: Appendix G1-G9', optional_experiments=False)
    (output / 'production_main_settings.json').write_text(json.dumps(manifest, indent=2, default=str))
    return ctx


def _timed_payload(ctx, payload):
    final_benchmark = payload.get('role') == 'engineering' or payload.get('method') in ('IB', 'GS', 'AD')
    campaign = ctx.memo.get('benchmark_timing_campaign', ctx.memo['production_timing_campaign']) if final_benchmark else ctx.memo['production_timing_campaign']
    return dict(payload, timing_campaign=campaign,
                machine=ctx.memo['production_machine'], schema=SCHEMA,
                timing_protocol_id=TIMING_PROTOCOL_ID, timing_scope=TIMING_SCOPE)


def _save_command(ctx, task, method, seed, value, *, suffix='nominal'):
    directory = ctx.output_root / 'commands' / task
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'{method}_seed{seed}_{suffix}'
    row = value['row']
    initial = ({'initial_phase_rad': value['initial_phase']} if 'initial_phase' in value else {})
    np.savez_compressed(directory / (stem+'.npz'), phase_rad=value['phase'], **initial,
        targets_m=value.get('targets', np.asarray([p.MAIN_TARGET_M])),
        metadata_json=json.dumps(row, default=str))
    (directory / (stem+'.json')).write_text(json.dumps(row, indent=2, default=str))


def _single_objective_definition(ctx, method, alpha):
    """Return transfer-free inputs for a timed Single objective build."""

    if method == 'Conventional':
        spec = method_spec('Conventional')
        force_target = np.zeros(3, dtype=float)
    elif method in ('FE', 'GFE'):
        spec = fixed_single_fe_method_spec()
        force_target = (
            effective_weight_force_target_n(SphereInFluid(frequency_hz=ctx.config.frequency_hz))
            if method == 'GFE' else np.zeros(3, dtype=float)
        )
    else:
        raise ValueError(method)
    if alpha is not None:
        spec = MethodSpec(name='Force-Equilibrium', alpha_per_m=float(alpha),
                          beta_curvature_per_pa=0., pressure_mode='abs', curvature_weight=np.eye(3))
    return spec, np.asarray(force_target, dtype=float)


def _optimizer_policy(method, role, *, task):
    """Use the same task-specific compact FE/GFE policy in every section."""
    if task not in ('Single', 'Triple'):
        raise ValueError(task)
    if method in ('FE', 'GFE') and task == 'Single':
        return 'L-BFGS-B', {key: value for key, value in ENGINEERING_LBFGS_OPTIONS.items() if key != 'gtol'}
    return 'BFGS', {}


def ensure_single_run(ctx, method, seed, *, target=None, alpha=None, role='controlled'):
    """Return one Single command and a comparable end-to-end timing record."""

    if method not in ('Conventional','FE','GFE'):
        raise ValueError(method)
    target = np.asarray(p.MAIN_TARGET_M if target is None else target, float)
    spec, force_target = _single_objective_definition(ctx, method, alpha)
    optimizer, optimizer_options = _optimizer_policy(method, role, task='Single')
    compact = method in ('FE', 'GFE')
    evaluator_backend = CompactSingleObjective.backend_id if compact else 'native'
    expected_initial = np.random.default_rng(int(seed)).uniform(-np.pi,np.pi,len(ctx.positions_m))
    payload = _timed_payload(ctx, dict(method=method, objective=asdict(spec),
        role=role, optimizer=optimizer, optimizer_options=optimizer_options,
        target_m=target.tolist(), force_target_n=force_target.tolist(),
        seed=int(seed), initial_phase_content_id=digest_array(expected_initial),
        positions=digest_array(ctx.positions_m), normals=digest_array(ctx.normals),
        frequency_hz=ctx.config.frequency_hz, maxiter=10000, gtol=ctx.config.gtol))
    if compact:
        payload['evaluator_backend'] = evaluator_backend
    key = ('production_single', digest_payload(payload))
    if key in ctx.memo:
        return ctx.memo[key]

    def produce():
        def initialize():
            label = ('canonical-Conventional' if method == 'Conventional' else
                     f'fixed-single-{method.lower()}-alpha{spec.alpha_per_m:g}')
            condition = Condition(label=label, positions=ctx.positions_m,
                target_m=tuple(target), frequency_hz=ctx.config.frequency_hz,
                curvature_weight=spec.curvature_weight, normals=ctx.normals)
            initial = np.random.default_rng(int(seed)).uniform(
                -np.pi, np.pi, len(ctx.positions_m))
            return condition, initial

        def build_transfer(initialized):
            condition, initial = initialized
            objective = SingleTargetObjective(
                condition, spec, force_target_n=force_target)
            if compact:
                objective = CompactSingleObjective(objective)
            return objective, initial

        extracted, timing = _execute_comparable_timing(
            initialize,
            build_transfer,
            lambda prepared: solve_corrected_gorkov_fe(
                prepared[0], prepared[1], maxiter=10000, gtol=ctx.config.gtol,
                optimizer=optimizer, optimizer_options=optimizer_options),
            lambda run, prepared: dict(
                run=run,
                objective=prepared[0],
                initial_phase=np.asarray(prepared[1], dtype=float),
                phase=np.asarray(run.phase_rad, dtype=float),
            ),
            backend='numpy/scipy-cpu',
        )
        return dict(extracted, timing=timing)

    timed, status, path = ctx.cache.get_or_compute(
        'production_single_run', payload, produce)
    run = timed['run']
    objective = timed['objective']
    initial = np.asarray(timed['initial_phase'], dtype=float)
    timing = timed['timing']
    if digest_array(initial) != digest_array(expected_initial):
        raise RuntimeError('Timed Single initialization does not match its cache identity')
    row = dict(method=method, task='Single', seed=int(seed), alpha_per_m=float(objective.method.alpha_per_m),
        iteration_cap=10000, maxiter=10000, cap=10000, iterations=int(run.iterations),
        evaluations=int(run.evaluations), objective=float(run.objective),
        gradient_norm=float(run.gradient_norm), terminal_gradient_norm=float(run.gradient_norm),
        terminal_gradient_l2=float(run.gradient_norm), terminal_gradient_rms=float(run.gradient_norm/np.sqrt(len(initial)-1)),
        terminal_gradient_inf=float(run.gradient_inf_norm), stationary_gtol=bool(run.stationary_gtol),
        terminal_gradient_units='N m^-1 rad^-1',
        terminal_gradient_status='native optimized-objective gradient',
        optimizer_success=bool(run.success), optimizer_status=int(run.status), optimizer_message=str(run.message),
        native_success=bool(run.success), native_status=int(run.status), native_message=str(run.message),
        wall_time_s=float(run.history_wall_s[-1]),
        native_reported_interval_s=float(run.history_wall_s[-1]),
        native_reported_interval_scope='FESolveResult history interval; objective/transfer construction excluded; not used for cross-method ranking',
        phase_content_id=digest_array(run.phase_rad), initial_phase_content_id=digest_array(initial),
        actual_force_target_z_n=float(objective.force_target_n[2]), force_target_z_n=float(objective.force_target_n[2]),
        cache_status=status, cache_path=str(path), cache_source=str(path),
        timing_campaign=payload['timing_campaign'],
        timing_observation_origin='fresh execution retained in cache' if status == 'new' else 'same-machine protocol record replayed from cache',
        timing_json=_timing_json(timing,ctx),
        role=role, optimizer=optimizer, optimizer_options_json=json.dumps(optimizer_options,sort_keys=True),
        evaluator_backend=evaluator_backend,
        correct_sign_gorkov=True, run_mode=ctx.config.mode, evidence_status='PRODUCTION' if ctx.config.full else 'SMOKE', manuscript_numerics=bool(ctx.config.full),
        near_stationary=bool(run.gradient_norm <= ff.REPORT_GRADIENT_NORM),
        **_timing_columns(timing,ctx),
        **{f'target_{axis}_m':float(target[i]) for i,axis in enumerate('xyz')})
    value=dict(run=run,phase=np.asarray(timed['phase']),row=row,objective=objective,
               timing=timing,initial_phase=initial,targets=np.asarray([target]))
    ctx.memo[key]=value
    _save_command(ctx,'Single',method,seed,value,suffix=f'{role}_{digest_array(target)[:10]}_a{objective.method.alpha_per_m:g}')
    return value


def triple_problem(ctx):
    if 'production_triple_problem' not in ctx.memo:
        ctx.memo['production_triple_problem'] = make_problem(ctx,FIXED_BASE_TARGETS_M,0.,0.,case_prefix='fixed-long-triple-observation')
    return ctx.memo['production_triple_problem']


def ensure_triple_run(ctx, method, seed, *, role='controlled'):
    """Return one Triple command and a comparable end-to-end timing record."""
    if method not in ('Conventional','FE','GFE'):
        raise ValueError(method)
    optimizer, optimizer_options = _optimizer_policy(method, role, task='Triple')
    compact = method in ('FE', 'GFE')
    evaluator_backend = 'compact-multitrap-discrete-v1' if compact else 'native'
    conventional,fe=_configs(ctx,30000)
    expected_config=conventional if method=='Conventional' else replace(fe,compensate_effective_gravity=method=='GFE')
    expected_initial=np.random.default_rng(int(seed)).uniform(-np.pi,np.pi,len(ctx.positions_m))
    payload=_timed_payload(ctx,dict(method=method,seed=int(seed),
        role=role,optimizer=optimizer,optimizer_options=optimizer_options,
        base_targets_m=np.asarray(FIXED_BASE_TARGETS_M).tolist(),
        positions=digest_array(ctx.positions_m),normals=digest_array(ctx.normals),
        frequency_hz=ctx.config.frequency_hz,stencil_m=ctx.config.stencil_m,
        config=expected_config.to_payload(),initial_phase_content_id=digest_array(expected_initial),maxiter=30000,
        checkpoints=list(CHECKPOINTS) if method=='Conventional' else []))
    if compact:
        payload['evaluator_backend'] = evaluator_backend
    key=('production_triple',digest_payload(payload))
    if key in ctx.memo:return ctx.memo[key]

    def produce():
        def initialize():
            conventional_config, fe_config = _configs(ctx, 30000)
            config = (conventional_config if method == 'Conventional' else
                      replace(fe_config, compensate_effective_gravity=method == 'GFE'))
            initial = np.random.default_rng(int(seed)).uniform(-np.pi, np.pi, len(ctx.positions_m))
            return config, initial

        def build_transfer(initialized):
            config, initial = initialized
            problem = make_problem(ctx, FIXED_BASE_TARGETS_M, 0., 0.,
                                   case_prefix='fixed-long-triple-observation')
            return problem, config, initial

        def solve(prepared):
            problem, config, initial = prepared
            if method == 'Conventional':
                return _run_cumulative_conventional(
                    problem, config, seed=int(seed), initial_phases=initial,
                    cumulative_cap=30000, checkpoints=CHECKPOINTS,
                    record_stalled_terminal=True)
            return run_multitrap(
                problem, config, seed=int(seed), initial_phases=initial, record_history=True,
                optimizer=optimizer, optimizer_options=optimizer_options,
                evaluator_backend='compact' if compact else 'native',
                metadata=dict(display_method=method,evidence_status='PRODUCTION'))

        def extract(run, prepared):
            problem, config, initial = prepared
            if method == 'Conventional':
                phase = np.asarray(run.terminal_phase_rad, dtype=float)
                last = run.segment_records[-1]
                values = dict(iterations=int(run.accepted_iterations), evaluations=int(run.nfev),
                    gradient=float(run.terminal_gradient_norm), objective=float(run.terminal_objective),
                    native_solve_s=float(run.solve_sec),
                    success=bool(last.get('success',False)) and not run.termination_reason.startswith('zero-accepted'),
                    status=int(last.get('status',1)), message=str(run.termination_reason))
            else:
                phase = np.asarray(run.phases, dtype=float)
                last = run.stages[-1]
                values = dict(iterations=int(run.iterations), evaluations=int(run.nfev),
                    native_solve_s=float(run.end_to_end_sec), success=bool(last.success),
                    status=int(last.status), message=str(last.message))
            return dict(run=run,problem=problem,config=config,initial_phase=np.asarray(initial,dtype=float),
                        phase=phase,**values)

        extracted,timing=_execute_comparable_timing(
            initialize,build_transfer,solve,extract,backend='numpy/scipy-cpu')
        if method != 'Conventional':
            last=extracted['run'].stages[-1]
            evaluation=evaluate_multitrap_objective(
                extracted['problem'],extracted['phase'],extracted['config'],
                force_epsilon=last.force_epsilon,uniformity_epsilon=last.uniformity_epsilon)
            extracted.update(gradient=float(np.linalg.norm(evaluation.gradient_reduced)),
                             objective=float(evaluation.value))
        return dict(extracted,timing=timing)

    timed,status,path=ctx.cache.get_or_compute('production_triple_run',payload,produce)
    run=timed['run'];problem=timed['problem'];config=timed['config']
    initial=np.asarray(timed['initial_phase'],dtype=float);phase=np.asarray(timed['phase'],dtype=float)
    timing=timed['timing'];iterations=timed['iterations'];evals=timed['evaluations']
    grad=timed['gradient'];loss=timed['objective'];elapsed=timed['native_solve_s']
    success=timed['success'];code=timed['status'];message=timed['message']
    if digest_array(initial) != digest_array(expected_initial):
        raise RuntimeError('Timed Triple initialization does not match its cache identity')
    if config.to_payload() != expected_config.to_payload():
        raise RuntimeError('Timed Triple configuration does not match its cache identity')
    row=dict(method=method,task='Triple',seed=int(seed),iterations=int(iterations),evaluations=int(evals),
        gradient_norm=float(grad),terminal_gradient_norm=float(grad),terminal_gradient_l2=float(grad),
        terminal_gradient_inf=float(run.metadata.get('terminal_gradient_inf_norm',np.nan)) if method!='Conventional' else np.nan,
        stationary_gtol=bool(run.metadata.get('stationary_gtol',False)) if method!='Conventional' else None,
        terminal_gradient_units='N m^-1 rad^-1',
        terminal_gradient_status='native optimized-objective gradient',
        objective=float(loss),wall_time_s=float(elapsed),native_reported_interval_s=float(elapsed),
        native_reported_interval_scope=('CumulativeBFGSResult.solve_sec; internal solve/checkpoint interval; not used for cross-method ranking'
            if method=='Conventional' else
            'MultitrapRunResult.end_to_end_sec after a prepared problem; not used for cross-method ranking'),
        iteration_cap=30000,maxiter=30000,cap=30000,optimizer_success=bool(success),optimizer_status=int(code),
        optimizer_message=str(message),native_success=bool(success),native_status=int(code),native_message=str(message),
        phase_content_id=digest_array(phase),initial_phase_content_id=digest_array(initial),cache_status=status,
        cache_path=str(path),cache_source=str(path),timing_campaign=payload['timing_campaign'],
        timing_observation_origin='fresh execution retained in cache' if status == 'new' else 'same-machine protocol record replayed from cache',
        timing_json=_timing_json(timing,ctx),config_json=json.dumps(config.to_payload()),
        native_stage_records_json=json.dumps(list(run.segment_records) if method=='Conventional' else [asdict(stage) for stage in run.stages],default=str),
        role=role,optimizer=optimizer,optimizer_options_json=json.dumps(optimizer_options,sort_keys=True),
        evaluator_backend=evaluator_backend,
        evidence_status='PRODUCTION' if ctx.config.full else 'SMOKE',manuscript_numerics=bool(ctx.config.full),**_timing_columns(timing,ctx))
    value=dict(run=run,phase=phase,row=row,config=config,problem=problem,timing=timing,
               initial_phase=initial,targets=np.array([s.target_m for s in problem.stencils]))
    ctx.memo[key]=value;_save_command(ctx,'Triple',method,seed,value,suffix=role)
    return value


def ensure_alpha_bank(ctx, seed):
    """The three cold commands shared exactly with production Appendix C3."""
    values=[ensure_single_run(ctx,'GFE',int(seed),alpha=alpha,role='cold_alpha') for alpha in ALPHAS]
    table=pd.DataFrame([dict(v['row'],prescribed_radiation_force_z_n=v['row']['force_target_z_n']) for v in values])
    return dict(runs=[v['run'] for v in values],phases=np.stack([v['phase'] for v in values]),table=table,seed=int(seed))


def _holography_run(ctx, task, method, seed=None, repeat=0):
    target=p.MAIN_TARGET_M if task=='Single' else np.mean(FIXED_BASE_TARGETS_M,axis=0)
    spec=p.SINGLE_SIDED_GS_VORTEX_SPEC if task=='Single' and method=='GS' else p.IB_VORTEX_SPEC
    if method=='AD':fn=ff.diff_pat_phase_only;kwargs=dict(task=task,random_seed=int(seed))
    elif method=='IB':fn=ff.iterative_back_projection;kwargs=dict(maxiter=200,tolerance_rad=.01)
    elif method=='GS':fn=ff.phase_only_signature_projection_audit;kwargs=dict(iterations=100)
    else:raise ValueError(method)
    if task=='Triple' and method!='AD':kwargs['control_group_size']=8
    payload=_timed_payload(ctx,dict(task=task,method=method,seed=seed,repeat=int(repeat),
        settings=kwargs,control_spec=asdict(spec),positions=digest_array(ctx.positions_m),
        normals=digest_array(ctx.normals),frequency_hz=ctx.config.frequency_hz,
        targets_m=np.asarray([target] if task=='Single' else FIXED_BASE_TARGETS_M).tolist(),
        diff_pat=asdict(ff.DIFF_PAT_CONFIG) if method=='AD' else None))

    def produce():
        def initialize():
            local_target = (np.asarray(p.MAIN_TARGET_M,dtype=float) if task == 'Single'
                            else np.mean(FIXED_BASE_TARGETS_M,axis=0))
            local_spec = (p.SINGLE_SIDED_GS_VORTEX_SPEC
                          if task == 'Single' and method == 'GS' else p.IB_VORTEX_SPEC)
            local_kwargs = dict(kwargs)
            return local_target, local_spec, local_kwargs

        def build_transfer(initialized):
            local_target, local_spec, local_kwargs = initialized
            factory = (p._vortex_transfer_factory(ctx,local_target,spec=local_spec)
                       if task == 'Single' else
                       p._multivortex_transfer_factory(ctx,FIXED_BASE_TARGETS_M,spec=local_spec))
            transfer, signature = factory()
            return local_target,local_spec,local_kwargs,np.asarray(transfer),np.asarray(signature)

        def solve(prepared):
            _,_,local_kwargs,transfer,signature = prepared
            return fn(transfer,signature,**local_kwargs)

        def extract(solver,prepared):
            local_target,_,_,transfer,signature = prepared
            phase=np.asarray(solver.phase_rad,dtype=float)
            return dict(solver=solver,phase=phase,target=np.asarray(local_target,dtype=float),
                        transfer=transfer,signature=signature)

        backend=('jax-'+str(ctx.memo['production_machine'].get('jax_backend','unknown'))
                 if method == 'AD' else 'numpy-cpu')
        extracted,timing=_execute_comparable_timing(
            initialize,build_transfer,solve,extract,backend=backend,accelerated=method=='AD')
        extracted['metrics']=vortex_ring_metrics(
            extracted.pop('transfer'),extracted.pop('signature'),extracted['phase'])
        return dict(extracted,timing=timing)

    timed,status,path=ctx.cache.get_or_compute('production_holography_run',payload,produce)
    solver=timed['solver'];timing=timed['timing'];phase=np.asarray(timed['phase'],dtype=float)
    if method=='AD':
        from .diff_pat import diff_pat_terminal_gradient_l2
        diagnostic_factory=(p._vortex_transfer_factory(ctx,target,spec=spec) if task=='Single'
                            else p._multivortex_transfer_factory(ctx,FIXED_BASE_TARGETS_M,spec=spec))
        diagnostic_transfer,diagnostic_signature=diagnostic_factory()
        terminal_gradient_l2=diff_pat_terminal_gradient_l2(
            diagnostic_transfer,diagnostic_signature,phase,task=task)
        gradient_units='dimensionless rad^-1'
        gradient_status='evaluated after and excluded from the common command-timing boundary'
    else:
        terminal_gradient_l2=np.nan
        gradient_units='N/A'
        gradient_status='not applicable: prescribed IB/GS command has no scalar optimized phase objective'
    run=ProductionBenchmarkRun(method_id=str(solver.method_id),display_name=str(solver.display_name),
        initialization_kind='method-declared command initialization',target_m=np.asarray(timed['target']),
        phase_rad=phase,timing=timing,solver_result=solver,
        metadata={**dict(solver.metadata),**dict(timed['metrics']),
                  'acoustic_transfer_construction_included':True,
                  'common_timing_protocol_id':TIMING_PROTOCOL_ID})
    row=dict(task=task,method=method,seed=seed,repeat=int(repeat),iterations=int(solver.iterations),
        phase_content_id=digest_array(run.phase_rad),native_reported_interval_s=float(timing.solve_s),
        native_reported_interval_scope=('native AD invocation with device transfer/JIT/synchronization bundled; not used for cross-method ranking'
            if method=='AD' else 'native prescribed-field method invocation; not used for cross-method ranking'),
        cache_path=str(path),cache_status=status,timing_campaign=payload['timing_campaign'],
        timing_observation_origin='fresh execution retained in cache' if status == 'new' else 'same-machine protocol record replayed from cache',
        timing_json=_timing_json(timing,ctx),
        solver_metadata_json=json.dumps(dict(solver.metadata),default=str),
        native_success=solver.converged,terminal_projective_change_rad=solver.terminal_projective_change_rad,
        terminal_gradient_l2=terminal_gradient_l2,terminal_gradient_units=gradient_units,
        terminal_gradient_status=gradient_status,
        evidence_status='PRODUCTION',**_timing_columns(timing,ctx))
    return dict(run=run,phase=np.asarray(run.phase_rad),row=row,timing=timing,
                targets=np.asarray([p.MAIN_TARGET_M] if task=='Single' else FIXED_BASE_TARGETS_M))


def ensure_command_bank(ctx):
    """Synthesize only the selected last-result population, with serial timings."""
    key=_scale_key(ctx,'production_commands')
    if key in ctx.memo:return ctx.memo[key]
    scale=_study_scale(ctx)
    count=scale['last_result_seed_count']
    repeats=scale['deterministic_timing_repeats']
    bank={};timing_rows=[]
    for task in ('Single','Triple'):
        seeds=_benchmark_seeds(ctx,task)
        stochastic_methods=('Conventional','FE','GFE','AD')
        for seed_index,seed in enumerate(seeds):
            offset=seed_index%len(stochastic_methods)
            scheduled=stochastic_methods[offset:]+stochastic_methods[:offset]
            seed_values={}
            for method in scheduled:
                value=(_holography_run(ctx,task,method,seed) if method=='AD' else
                       ensure_single_run(ctx,method,seed,role='engineering') if task=='Single' else ensure_triple_run(ctx,method,seed,role='engineering'))
                seed_values[method]=value
                timing_rows.append(dict(value['row'],repeat=0,timing_observation_index=int(seed_index),
                    timing_schedule='four-position cyclic order balanced across command seeds'))
                _save_command(ctx,task,method,seed,value)
            for method in stochastic_methods:
                bank[(task,method,int(seed))]=seed_values[method]
        firsts={'IB':None,'GS':None};deterministic_rows={'IB':[],'GS':[]}
        for repeat in range(repeats):
            scheduled=('IB','GS') if repeat%2==0 else ('GS','IB')
            for method in scheduled:
                value=_holography_run(ctx,task,method,repeat=repeat)
                if firsts[method] is None:firsts[method]=value
                elif not np.allclose(np.exp(1j*value['phase']),np.exp(1j*firsts[method]['phase']),atol=1e-10,rtol=0.):
                    raise RuntimeError(f'{task} {method} repeat changed deterministic phase')
                observation=dict(value['row'],timing_observation_index=int(repeat),
                    timing_schedule='IB/GS order alternated across complete repeats')
                timing_rows.append(observation);deterministic_rows[method].append(observation)
        for method in ('IB','GS'):
            first=firsts[method];method_rows=deterministic_rows[method]
            if first is None or len(method_rows)!=repeats:
                raise RuntimeError(f'{task} {method} timing schedule is incomplete')
            median_total=float(np.median([r['command_time_s'] for r in method_rows]))
            first['row']=dict(first['row'],command_time_s=median_total,timing_total_s=median_total,
                timing_observation_count=repeats,
                timing_record_role=f'one deterministic endpoint; command time is median of {repeats} protocol observations (fresh or same-machine replay)',
                timing_json=json.dumps(dict(protocol_id=TIMING_PROTOCOL_ID,
                    aggregate=f'median total over {repeats} complete observations',
                    observation_table='production_timing_observations.csv'),default=str))
            bank[(task,method,None)]=first;_save_command(ctx,task,method,'deterministic',first)
    expected_commands=8*count+4
    if len(bank)!=expected_commands:
        raise AssertionError(f'Expected {expected_commands} command slots, got {len(bank)}')
    timing_table=pd.DataFrame(timing_rows)
    expected={(task,method):(repeats if method in ('IB','GS') else count)
              for task in ('Single','Triple') for method in METHODS}
    observed=timing_table.groupby(['task','method'],sort=False).size().to_dict()
    if observed != expected:
        raise RuntimeError(f'Timing campaign observation counts differ: expected {expected}; got {observed}')
    if set(timing_table.timing_protocol_id) != {TIMING_PROTOCOL_ID}:
        raise RuntimeError('Production timing observations mix protocol identities')
    if set(timing_table.timing_scope) != {TIMING_SCOPE}:
        raise RuntimeError('Production timing observations mix timing boundaries')
    excluded=timing_table[~timing_table.timing_comparable.astype(bool)].copy()
    ctx.tables['production_command_runs']=pd.DataFrame([v['row'] for v in bank.values()])
    ctx.tables['production_timing_observations']=timing_table
    ctx.tables['production_timing_exclusions']=excluded
    ctx.tables['production_timing_protocol']=pd.DataFrame([dict(
        timing_protocol_id=TIMING_PROTOCOL_ID,scope=TIMING_SCOPE,
        study_profile=scale['profile'],
        required_observations_per_stochastic_method_task=count,
        required_observations_per_deterministic_method_task=repeats,
        deterministic_endpoint_policy=f'IB/GS: one endpoint, {repeats} complete timing repeats',
        stochastic_endpoint_policy=f'Conventional/FE/GFE/AD: {count} commands and {count} complete timing observations',
        execution_order_policy='cyclic four-position order for stochastic methods; alternating IB/GS order',
        absent_numeric_stage_policy='N/A with explicit status; never zero',
        rank_policy='only timing_comparable=true observations from this protocol',
        requested_device=ctx.memo['production_machine'].get('requested_device'),
        actual_jax_backend=ctx.memo['production_machine'].get('jax_backend'),
        requested_blas_threads=ctx.memo['production_machine'].get('requested_blas_threads'),
        blas_thread_contract_satisfied=ctx.memo['production_machine'].get('blas_thread_contract_satisfied'),
        machine_record_json=json.dumps(ctx.memo['production_machine'],default=str),
        independent_validation_included=False,cache_retrieval_included=False,
        rendering_included=False,export_included=False)])
    p.export_all_tables(ctx)
    if len(excluded):
        detail=excluded[['task','method','timing_exclusion_reason']].drop_duplicates().to_dict('records')
        raise RuntimeError(f'Non-comparable production timing observations were excluded: {detail}')
    ctx.memo[key]=bank
    return bank


def pressure_descriptor(ctx, phase):
    """The established 80-radius × 256-azimuth morphology diagnostic."""
    from .appendix_b3_mechanics import transverse_descriptor
    payload=dict(phase=digest_array(phase),positions=digest_array(ctx.positions_m),
        normals=digest_array(ctx.normals),frequency_hz=ctx.config.frequency_hz,
        target_m=p.MAIN_TARGET_M.tolist(),radial_points=80,azimuth_points=256,
        descriptor_source=content_identity(Path(__file__).with_name('appendix_b3_mechanics.py').read_bytes()).hexdigest())
    value,_,_=ctx.cache.get_or_compute('production_pressure_descriptor',payload,
        lambda:transverse_descriptor(p._array_field(ctx,phase),p.MAIN_TARGET_M,343./ctx.config.frequency_hz))
    return dict(value[0])


def _single_plot_bank(ctx):
    seeds=_first_result_seeds(ctx)
    runs={};rows=[]
    for index,seed in enumerate(seeds):
        for method in ('Conventional','GFE'):
            value=ensure_single_run(ctx,method,seed)
            label='FE' if method=='GFE' else method
            runs[(label,index)]=value['run'];rows.append(dict(value['row'],method=label,pair_index=index))
    table=pd.DataFrame(rows)
    ctx.tables['main_feg_single_runs']=table.assign(method=table.method.replace({'FE':'GFE'}))
    ctx.tables['fig1_single_vortex_iterations_for_table_only']=table[['method','pair_index','seed','iterations','evaluations','iteration_cap']]
    return dict(runs=runs,table=table,seeds=list(seeds),caps={'Conventional':10000,'FE':10000})


def _domain_plot_bank(ctx):
    key=_scale_key(ctx,'production_xz_bank')
    if key in ctx.memo:return ctx.memo[key]
    scale=_study_scale(ctx)
    x_points=scale['first_result_domain_x_points']
    z_points=scale['first_result_domain_z_points']
    restarts=scale['first_result_domain_restarts']
    methods=('Conventional','GFE') if scale['profile']=='supersmoke' else ('Conventional','FE','GFE')
    rows=[]
    for z in np.linspace(*domain.Z_BOUNDS_M,z_points):
        for x in np.linspace(*domain.X_BOUNDS_M,x_points):
            target=np.array([x,0.,z]);target_id=domain._coordinate_id(target)
            for restart in range(restarts):
                seed=domain._matched_seed(ctx.config.random_seed+401,target_id,restart)
                for method in methods:
                    value=ensure_single_run(ctx,method,seed,target=target,role='xz_domain')
                    rows.append(dict(value['row'],target_id=target_id,restart=restart))
    raw=pd.DataFrame(rows)
    ctx.tables['main_fe_control_xz_domain']=raw[raw.method.eq('FE')].copy()
    raw=raw[~raw.method.eq('FE')].copy()
    summary=raw.groupby(['method','alpha_per_m','target_id','target_x_m','target_y_m','target_z_m'],sort=True).agg(
        run_count=('restart','size'),wall_time_median_s=('wall_time_s','median'),
        wall_time_q25_s=('wall_time_s',lambda a:a.quantile(.25)),wall_time_q75_s=('wall_time_s',lambda a:a.quantile(.75)),
        terminal_gradient_rms_median=('terminal_gradient_rms','median')).reset_index()
    expected_rows=2*x_points*z_points*restarts
    if len(raw)!=expected_rows:
        raise AssertionError(f'XZ map must have {expected_rows} displayed-method runs')
    ctx.tables['fig1_xz_domain_runs']=raw;ctx.tables['fig1_xz_domain_summary']=summary
    definition=pd.DataFrame([dict(x_points=x_points,z_points=z_points,restarts=restarts,target_count=x_points*z_points,methods='Conventional;GFE',
        x_min_m=domain.X_BOUNDS_M[0],x_max_m=domain.X_BOUNDS_M[1],z_min_m=domain.Z_BOUNDS_M[0],z_max_m=domain.Z_BOUNDS_M[1])])
    ctx.tables['fig1_xz_domain_definition']=definition
    value=dict(raw=raw,summary=summary.assign(method=summary.method.replace({'GFE':'FE'})),definition=definition)
    ctx.memo[key]=value;return value


def _branch_pack(ctx):
    """Preserve the fixed chart; expand only the root paired-initialization audit."""
    count = _study_scale(ctx)['second_result_seed_count']
    key = f'production_fixed_branch:{count}'
    if key in ctx.memo:
        return ctx.memo[key]
    base = _ORIGINAL_FIXED_BRANCH(ctx)
    if len(base['anchors']) != 98 or base['anchors'].target_id.nunique() != 49:
        raise AssertionError('The branch chart must preserve 98 anchors at 49 targets')
    bank = base['bank']
    ix, iy = bank.root_target
    root_table = base['anchors'].query('ix == @ix and iy == @iy')
    anchors = np.stack([bank.target_medoid(ix, iy, component) for component in COMPONENTS])
    seeds = _selected_seeds(SINGLE30, count)
    command_rows, commands = [], {}
    for pair_index, seed in enumerate(seeds):
        for method in ('Conventional', 'GFE'):
            value = ensure_single_run(ctx, method, seed)
            commands[(method, pair_index)] = value
            command_rows.append(dict(value['row'], pair_index=pair_index,
                                     population='Figure 3 root paired initializations'))
        left = commands[('Conventional', pair_index)]['initial_phase']
        right = commands[('GFE', pair_index)]['initial_phase']
        if not np.array_equal(left, right):
            raise RuntimeError('Root correspondence requires identical paired initial phases')
    target_rows = pd.DataFrame([
        dict(target_x_m=p.MAIN_TARGET_M[0], target_y_m=p.MAIN_TARGET_M[1],
             target_z_m=p.MAIN_TARGET_M[2]) for _ in seeds
    ])
    similarities = {}
    for method in ('Conventional', 'GFE'):
        phases = np.stack([commands[(method, index)]['phase'] for index in range(count)])
        states = target_demodulated_states(phases, target_rows,
            positions_m=ctx.positions_m, frequency_hz=ctx.config.frequency_hz)
        similarities[method] = projective_similarity(states, anchors)
    records = []
    for index, seed in enumerate(seeds):
        component_index = int(np.argmax(similarities['Conventional'][index]))
        similarity = float(similarities['Conventional'][index, component_index])
        gfe_index = int(np.argmax(similarities['GFE'][index]))
        records.append(dict(
            trajectory_index=index, pair_index=index, seed=seed,
            target_id=str(root_table.iloc[0].target_id), component=COMPONENTS[component_index],
            objective=commands[('Conventional', index)]['row']['objective'],
            similarity_to_matched_fe=similarity,
            distance_to_matched_fe=float(np.sqrt(max(0., 1-similarity*similarity))),
            reference_method='GFE', similarity_to_matched_feg=similarity,
            distance_to_matched_feg=float(np.sqrt(max(0., 1-similarity*similarity))),
            paired_gfe_component=COMPONENTS[gfe_index],
            paired_gfe_similarity_to_endpoint_class=float(similarities['GFE'][index, gfe_index]),
            initial_phase_content_id=commands[('Conventional', index)]['row']['initial_phase_content_id'],
        ))
    correlation = pd.DataFrame(records)
    if len(correlation) != count or correlation.seed.nunique() != count:
        raise AssertionError('The root audit must contain the requested unique initialization count')
    protocol = base['protocol'].copy()
    protocol['root_paired_initializations'] = count
    protocol['root_methods'] = 'Conventional;GFE'
    protocol['chart_target_count'] = 49
    value = dict(base, correlation=correlation, protocol=protocol)
    ctx.memo[key] = value
    ctx.tables['production_fig2_central_correspondence'] = correlation.copy()
    ctx.tables['fig2_paired_initialization_commands'] = pd.DataFrame(command_rows)
    ctx.tables['fig2_fixed_feg_continuation_protocol'] = protocol.copy()
    return value


def _representative_triple_bank(ctx, compensated=False, *, role='controlled'):
    c=ensure_triple_run(ctx,'Conventional',260869,role=role);fe=ensure_triple_run(ctx,'GFE' if compensated else 'FE',260869,role=role)
    node=dict(problem=c['problem'],conventional_config=c['config'],fe_config=fe['config'],
        Conventional=c['run'],FE=fe['run'],FE_phase=fe['phase'],
        Conventional_checkpoint_phases=c['run'].checkpoint_phases_rad,targets_m=c['targets'])
    return dict(objects={(1,1):node},initial_phase=c['initial_phase'],seed=260869,
        conventional_cap=30000,fe_cap=30000,checkpoints=CHECKPOINTS,nodes=((1,1),))


def _representative_gfe_triple(ctx):
    value=ensure_triple_run(ctx,'GFE',260869)
    return dict(value,bank=_representative_triple_bank(ctx),cache_status=value['row']['cache_status'])


def _spec(task, method, seed, value):
    label='FE+g' if method=='GFE' else method
    return dict(method=label,display_method=label,task=task,seed=seed,pair_id=f'{task}-seed-{seed}',
        phase=value['phase'],target_m=value['targets'][0],source_index=-1,
        phase_source='current production command with fixed declared objective',
        terminal_iteration=int(value['row']['iterations']),command_time_s=float(value['row']['command_time_s']),
        terminal_gradient_l2=float(value['row'].get('terminal_gradient_l2',np.nan)),
        terminal_gradient_units=value['row'].get('terminal_gradient_units','N m^-1 rad^-1'),
        terminal_gradient_status=value['row'].get('terminal_gradient_status','native optimized-objective gradient'),
        optimizer=value['row'].get('optimizer','native'),role=value['row'].get('role','engineering'),
        timing_scope=value['row']['timing_scope'],timing_protocol_id=value['row']['timing_protocol_id'],
        timing_comparable=bool(value['row']['timing_comparable']),
        timing_exclusion_reason=value['row']['timing_exclusion_reason'],command_record=value,
        method_family='prescribed phase-only' if method in ('IB','GS') else 'automatic-differentiation holography' if method=='AD' else 'optimization',
        preset_role='matched' if method in ('IB','GS') else 'matched-amplitude' if method=='AD' else 'not_applicable',
        preset_radius_lambda=.45 if method in ('IB','GS','AD') else np.nan)


def _validate_spec(ctx, spec, target_index, target):
    plot_only=_study_scale(ctx)['profile']=='supersmoke'
    phase=spec['phase'];label=spec['method'];key=f"production-{spec['task']}-{label}-{spec['seed']}-T{target_index+1}"
    result=p._finite_ka_validation(ctx,phase,target,key=key)
    row=p._static_validation_row(label,result)
    if not plot_only:
        viscous=p._viscous_elastic_validation(ctx,phase,target,key=key+'-viscous')
        row.update(p._viscous_static_validation_row(label,viscous))
    field=p._array_field(ctx,phase);resolved=bool(result.equilibrium.numerical_root_found)
    equilibrium=np.asarray(result.equilibrium.equilibrium_m)
    row.update(pressure_abs_at_intended_target_pa=float(abs(field.pressure(np.asarray(target)[None,:])[0])),
        pressure_abs_at_exact_equilibrium_pa=float(abs(field.pressure(equilibrium[None,:])[0])) if resolved else np.nan,
        task=spec['task'],method=label,display_method=label,seed=spec['seed'],pair_id=spec['pair_id'],
        target_index=target_index,target_id='Single' if spec['task']=='Single' else f'T{target_index+1}',
        phase_content_id=digest_array(phase),phase_source=spec['phase_source'],terminal_iteration=spec['terminal_iteration'],
        terminal_gradient_l2=spec['terminal_gradient_l2'],terminal_gradient_units=spec['terminal_gradient_units'],
        terminal_gradient_status=spec['terminal_gradient_status'],
        command_time_s=spec['command_time_s'],timing_scope=spec['timing_scope'],
        timing_protocol_id=spec['timing_protocol_id'],timing_comparable=spec['timing_comparable'],
        timing_exclusion_reason=spec['timing_exclusion_reason'],
        optimizer=spec['optimizer'],role=spec['role'],
        method_family=spec['method_family'],preset_role=spec['preset_role'],preset_radius_lambda=spec['preset_radius_lambda'],
        **{f'target_{axis}_m':float(target[i]) for i,axis in enumerate('xyz')},
        evidence_status='PRODUCTION',manuscript_numerics=True)
    if plot_only:
        # Population table consumers require this column; Figure 8 does not use it.
        row.update(field_concentration_fraction=np.nan,
                   field_concentration_status='not_computed_supersmoke_unplotted_diagnostic')
    else:
        row.update(ff._field_concentration(ctx,phase,target))
    directory=ctx.output_root/'mechanics';directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/(key+'.npz'),phase_rad=phase,target_m=target,equilibrium_m=equilibrium,
        displacement_m=result.equilibrium.displacement_m,force_jacobian_n_m=result.force_jacobian_n_m,
        stiffness_eigenvalues_n_m=result.symmetric_stiffness_eigenvalues_n_m,
        metadata_json=json.dumps(row,default=str),protocol_json=json.dumps(p._finite_ka_protocol(ctx),default=str))
    return dict(row=row,result=result,spec=dict(spec))


def ensure_benchmarks(ctx):
    """Validate the selected command population and expose every target record."""
    key=_scale_key(ctx,'production_benchmarks')
    if key in ctx.memo:return ctx.memo[key]
    bank=ensure_command_bank(ctx);output={}
    for task in ('Single','Triple'):
        specs=[];evaluations=[]
        for (bank_task,method,seed),value in bank.items():
            if bank_task!=task:continue
            spec=_spec(task,method,seed,value);specs.append(spec)
            for index,target in enumerate(value['targets']):
                evaluations.append(_validate_spec(ctx,spec,index,target))
        table=pd.DataFrame([item['row'] for item in evaluations])
        value=dict(raw=table,table=table,all_table=table,concentration_table=table,
            specs=specs,method_specs=tuple(specs),results=[item['result'] for item in evaluations],
            evaluations=evaluations,targets_m=np.asarray([p.MAIN_TARGET_M] if task=='Single' else FIXED_BASE_TARGETS_M))
        if task=='Triple':value.update(node=_representative_triple_bank(ctx,role='engineering')['objects'][(1,1)],long_bank=_representative_triple_bank(ctx,role='engineering'))
        output[task]=value;ctx.tables[f'production_{task.lower()}_target_mechanics']=table
    expected_records=16*_study_scale(ctx)['last_result_seed_count']+8
    if sum(len(v['table']) for v in output.values())!=expected_records:
        raise AssertionError(f'Expected {expected_records} target-level records')
    ctx.memo[key]=output
    ctx.tables['fig5_exact_force_field_concentration']=pd.concat([v['table'] for v in output.values()],ignore_index=True)
    _population_summaries(ctx,ctx.tables['fig5_exact_force_field_concentration'])
    p.export_all_tables(ctx);return output


def _command_metrics(table):
    records=[]
    for (task,method,phase,seed),group in table.groupby(['task','method','phase_content_id','seed'],sort=False,dropna=False):
        all_roots=bool(group.finite_ka_root_found.all())
        resolved=group[group.finite_ka_root_found.astype(bool)]
        target_columns=['target_index','target_x_m','target_y_m','target_z_m']
        target_condition=(group[target_columns].sort_values('target_index')
                          .to_dict(orient='records'))
        records.append(dict(task=task,method=method,phase_content_id=phase,seed=group.iloc[0].seed,
            command_time_s=float(group.command_time_s.iloc[0]),target_count=len(group),
            target_condition_content_id=digest_payload(target_condition),
            root_found=all_roots,all_restoring=bool(all_roots and group.finite_ka_locally_restoring.all()),
            field_concentration_fraction=float(group.field_concentration_fraction.median()),
            finite_ka_displacement_a=float(resolved.finite_ka_displacement_a.median()) if all_roots else np.nan,
            finite_ka_stiffness_min_n_m=float(resolved.finite_ka_stiffness_min_n_m.min()) if all_roots else np.nan,
            pressure_abs_at_exact_equilibrium_pa=float(resolved.pressure_abs_at_exact_equilibrium_pa.median()) if all_roots else np.nan))
    return pd.DataFrame(records)


def _bootstrap_controls(ctx):
    settings=ctx.memo.get('production_settings',{})
    resamples=int(settings.get('bootstrap_resamples',10000))
    seed=int(settings.get('bootstrap_seed',20260906))
    if resamples!=10000:
        raise ValueError('The production statistical protocol requires exactly 10000 bootstrap resamples')
    return resamples,seed


def _stream_seed(master_seed, identity):
    from rineng_protocol import bootstrap_stream
    return bootstrap_stream(master_seed, identity)

def _distribution_statistics(values, *, resamples, master_seed, identity,
                             inferential=True, inapplicable_reason=''):
    """Median/IQR/range and a reproducible command-level bootstrap interval."""
    array=pd.to_numeric(pd.Series(values),errors='coerce').to_numpy(float)
    array=array[np.isfinite(array)]
    result=dict(n_valid=len(array),median=np.nan,q25=np.nan,q75=np.nan,
        minimum=np.nan,maximum=np.nan,iqr=np.nan,range=np.nan,
        ci_low=np.nan,ci_high=np.nan,bootstrap_resamples_performed=0,
        bootstrap_stream_seed=np.nan,interval_status='no_finite_values')
    if len(array):
        result.update(median=float(np.median(array)),minimum=float(np.min(array)),
                      maximum=float(np.max(array)),range=float(np.ptp(array)))
    if not inferential:
        # Preserve the deterministic point estimate but do not serialize a
        # singleton min=max or zero range as if it described population spread.
        result.update(minimum=np.nan,maximum=np.nan,range=np.nan)
        result['interval_status']=inapplicable_reason or 'not_applicable'
        return result
    if not len(array):
        return result
    if len(array)<2:
        result['interval_status']='insufficient_independent_replicates'
        return result
    q25,q75=np.quantile(array,[.25,.75])
    stream_seed=_stream_seed(master_seed,identity)
    rng=np.random.default_rng(stream_seed)
    indices=rng.integers(0,len(array),size=(resamples,len(array)))
    bootstrap_medians=np.median(array[indices],axis=1)
    ci_low,ci_high=np.quantile(bootstrap_medians,[.025,.975])
    result.update(q25=float(q25),q75=float(q75),iqr=float(q75-q25),
        ci_low=float(ci_low),ci_high=float(ci_high),
        bootstrap_resamples_performed=resamples,
        bootstrap_stream_seed=stream_seed,interval_status='estimated')
    return result


def _assign_distribution(row, metric, statistics):
    row[metric]=statistics['median']
    for name in ('q25','q75','minimum','maximum','iqr','range','ci_low','ci_high','n_valid',
                 'bootstrap_resamples_performed','bootstrap_stream_seed','interval_status'):
        row[f'{metric}_{name}']=statistics[name]


def _terminal_gradient_summaries(ctx):
    """Report objective-specific phase gradients without inventing IB/GS values."""
    commands=ctx.tables.get('production_command_runs')
    if commands is None:
        raise RuntimeError('Production command records must precede gradient statistics')
    commands=commands.copy()
    commands['display_method']=commands.method.replace({'GFE':'FE+g'})
    resamples,seed=_bootstrap_controls(ctx)
    rows=[]
    for task in ('Single','Triple'):
        for method in METHODS:
            display='FE+g' if method=='GFE' else method
            group=commands[commands.task.eq(task)&commands.display_method.eq(display)]
            applicable=method not in ('IB','GS')
            values=(group['terminal_gradient_l2'] if 'terminal_gradient_l2' in group
                    else pd.Series(dtype=float))
            reason=('not_applicable: deterministic prescribed-field method has no optimized '
                    'scalar phase objective')
            statistics=_distribution_statistics(values,resamples=resamples,master_seed=seed,
                identity=('terminal_gradient_l2',task,display),inferential=applicable,
                inapplicable_reason=reason)
            if applicable and len(group) and statistics['n_valid']==0:
                statistics['interval_status']='missing_required_terminal_gradient'
            row=dict(task=task,method=display,public_method='GFE' if display=='FE+g' else display,
                command_count=len(group),
                independent_command_count=int(group.seed.notna().sum()),
                gradient_applicable=applicable,
                gradient_units=('dimensionless rad^-1' if method=='AD' else
                                'N m^-1 rad^-1' if applicable else 'N/A'),
                gradient_definition=('L2 norm of the terminal gradient of the declared '
                    'amplitude-only Diff-PAT/AD loss' if method=='AD' else
                    'L2 norm of the terminal reduced-phase objective gradient' if applicable else
                    'N/A: IB/GS return prescribed phase commands without a scalar optimized objective'),
                cross_objective_comparison='not valid; objective scales and units differ')
            _assign_distribution(row,'terminal_gradient_l2',statistics)
            row['terminal_gradient_l2_median']=statistics['median']
            rows.append(row)
    result=pd.DataFrame(rows)
    ctx.tables['production_terminal_phase_gradient_summary']=result
    return result


def _population_summaries(ctx, table):
    commands=_command_metrics(table);summary=[];intervals=[]
    resamples,seed=_bootstrap_controls(ctx)
    metrics=('field_concentration_fraction','finite_ka_displacement_a','finite_ka_stiffness_min_n_m','pressure_abs_at_exact_equilibrium_pa','command_time_s')
    for (task,method),group in commands.groupby(['task','method'],sort=False):
        timing_values=_timing_values(ctx,task,method)
        deterministic=method in ('IB','GS')
        row=dict(task=task,method=method,public_method='GFE' if method=='FE+g' else method,
                 endpoint_count=len(group),resolved_count=int(group.root_found.sum()),
                 timing_observation_count=len(timing_values),
                 independent_endpoint_count=0 if deterministic else int(group.seed.notna().sum()),
                 root_found=bool(group.root_found.any()),
                 aggregation=('one deterministic endpoint; timing repetitions are measurement repeats, '
                              'not endpoint replicates' if deterministic else
                              'within-command target metric, then median/IQR across command seeds'),
                 bootstrap_resamples=resamples,bootstrap_seed=seed,
                 triple_dependence='targets are nested within each command and never resampled as independent observations')
        for metric in metrics:
            is_timing=metric=='command_time_s'
            values=timing_values if is_timing else group[metric]
            inferential=is_timing or not deterministic
            statistics=_distribution_statistics(values,resamples=resamples,master_seed=seed,
                identity=('method_population',task,method,metric),inferential=inferential,
                inapplicable_reason='not_applicable: one deterministic endpoint, not an independent endpoint population')
            _assign_distribution(row,metric,statistics)
            sampling_unit=(('serial timing repetition of one deterministic command' if deterministic else
                            'independent command-seed timing observation') if is_timing else
                'deterministic endpoint (descriptive value only)' if deterministic else 'independent command seed')
            row[f'{metric}_sampling_unit']=sampling_unit
            intervals.append(dict(task=task,method=method,
                public_method='GFE' if method=='FE+g' else method,metric=metric,
                estimate=statistics['median'],q25=statistics['q25'],q75=statistics['q75'],
                iqr=statistics['iqr'],minimum=statistics['minimum'],maximum=statistics['maximum'],
                range=statistics['range'],ci95_low=statistics['ci_low'],ci95_high=statistics['ci_high'],
                n_valid=statistics['n_valid'],interval_status=statistics['interval_status'],
                bootstrap_resamples_requested=resamples,
                bootstrap_resamples_performed=statistics['bootstrap_resamples_performed'],
                bootstrap_seed=seed,bootstrap_stream_seed=statistics['bootstrap_stream_seed'],
                sampling_unit=sampling_unit,
                triple_dependence='target outcomes reduced within command before any resampling'))
        summary.append(row)
    ctx.tables['production_command_metrics']=commands
    ctx.tables['production_method_summary']=pd.DataFrame(summary)
    ctx.tables['production_method_bootstrap_intervals']=pd.DataFrame(intervals)
    paired=[]
    for task in ('Single','Triple'):
        selected=commands[commands.task.eq(task)]
        reference=selected[selected.method.eq('FE+g')]
        for method in ('Conventional','FE','IB','GS','AD'):
            other=selected[selected.method.eq(method)]
            deterministic=method in ('IB','GS')
            if deterministic:
                joined=pd.DataFrame()
            else:
                joined=reference.merge(other,on='seed',suffixes=('_reference','_comparison'),
                    validate='one_to_one')
                if len(joined) and not joined.target_condition_content_id_reference.equals(
                        joined.target_condition_content_id_comparison):
                    raise RuntimeError(f'Paired {task} {method} commands do not share target conditions')
            for metric in metrics:
                if deterministic:
                    delta=np.array([],float)
                else:
                    left=pd.to_numeric(joined[f'{metric}_reference'],errors='coerce').to_numpy(float)
                    right=pd.to_numeric(joined[f'{metric}_comparison'],errors='coerce').to_numpy(float)
                    valid=np.isfinite(left)&np.isfinite(right)
                    delta=left[valid]-right[valid]
                reason='not_applicable: deterministic method has no same-seed endpoint population'
                statistics=_distribution_statistics(delta,resamples=resamples,master_seed=seed,
                    identity=('paired_FE+g_difference',task,method,metric),
                    inferential=not deterministic,inapplicable_reason=reason)
                paired.append(dict(task=task,reference_method='GFE',internal_reference_method='FE+g',
                    comparison_method=method,comparison=f'GFE minus {method}',metric=metric,
                    planned_pairs=0 if deterministic else len(joined),valid_pairs=len(delta),
                    median_difference=statistics['median'],q25_difference=statistics['q25'],
                    q75_difference=statistics['q75'],minimum_difference=statistics['minimum'],
                    maximum_difference=statistics['maximum'],ci_low=statistics['ci_low'],
                    ci_high=statistics['ci_high'],interval_status=statistics['interval_status'],
                    bootstrap_resamples=resamples,
                    bootstrap_resamples_performed=statistics['bootstrap_resamples_performed'],
                    bootstrap_seed=seed,bootstrap_stream_seed=statistics['bootstrap_stream_seed'],
                    pairing_keys='task + seed + identical target-condition hash',
                    pairing_detail=('same cold phase for Conventional/FE/FE+g' if method in ('Conventional','FE') else
                                    'same base-seed block; AD retains its declared native seed offset' if method=='AD' else
                                    'N/A: no stochastic endpoint population'),
                    sampling_unit='paired command seed; Triple target outcomes reduced within command'))
    ctx.tables['production_paired_GFE_differences']=pd.DataFrame(paired)
    _terminal_gradient_summaries(ctx)
    return ctx.tables['production_method_summary']


def _force_aggregates(table, *, task):
    commands=_command_metrics(table.assign(task=task));rows=[]
    for method in ff.FIG5_METHOD_ORDER:
        group=commands[commands.method.eq(method)];resolved=group[group.root_found]
        deterministic=method in ('IB','GS')
        row=dict(task=task,method=method,display_method=method,endpoint_count=len(group),resolved_count=len(resolved),
                 independent_endpoint_count=0 if deterministic else int(group.seed.notna().sum()),
                 root_found=bool(len(resolved)),
                 aggregation=('one deterministic endpoint; no population interval' if deterministic else
                              'per-command targets then median/IQR across seeds'),
                 bootstrap_resamples=10000,bootstrap_seed=20260906)
        for metric in ('field_concentration_fraction','finite_ka_displacement_a','finite_ka_stiffness_min_n_m'):
            statistics=_distribution_statistics(group[metric],resamples=10000,
                master_seed=20260906,identity=('figure5',task,method,metric),
                inferential=not deterministic,
                inapplicable_reason='not_applicable: one deterministic endpoint, not an independent endpoint population')
            _assign_distribution(row,metric,statistics)
        rows.append(row)
    return pd.DataFrame(rows)


def _timing_values(ctx, task, method):
    """Return only complete, like-for-like observations from the common protocol."""
    observations=ctx.tables.get('production_timing_observations' if ctx.config.full else 'engineering_timing_observations')
    if observations is None:
        raise RuntimeError('Current command-timing observations must precede population summaries')
    required={'timing_protocol_id','timing_scope','timing_comparable','timing_exclusion_reason','command_time_s','timing_campaign'}
    if not required.issubset(observations.columns):
        raise RuntimeError('Legacy or historical timing rows cannot enter the Figure 8 production comparison')
    public_method='GFE' if method=='FE+g' else method
    selected=observations[observations.task.eq(task)&observations.method.eq(public_method)]
    if selected.empty:
        raise RuntimeError(f'Missing production timing observations for {task} {method}')
    if public_method in ('FE','GFE'):
        if not {'optimizer','evaluator_backend'}.issubset(selected.columns):
            raise RuntimeError(f'{task} {method} lacks the required solver identity')
        expected_backend=CompactSingleObjective.backend_id if task=='Single' else 'compact-multitrap-discrete-v1'
        expected_optimizer = 'L-BFGS-B' if task == 'Single' else 'BFGS'
        if not selected.optimizer.eq(expected_optimizer).all() or not selected.evaluator_backend.eq(expected_backend).all():
            raise RuntimeError(f'{task} {method} timing is not from compact {expected_optimizer}')
    current_campaign=str(ctx.memo.get('benchmark_timing_campaign',ctx.memo['production_timing_campaign']))
    if set(selected.timing_campaign.astype(str)) != {current_campaign}:
        raise RuntimeError(f'{task} {method} contains timing observations from another campaign')
    selected=selected[selected.timing_protocol_id.eq(TIMING_PROTOCOL_ID)&selected.timing_scope.eq(TIMING_SCOPE)]
    eligible=selected[selected.timing_comparable.astype(bool)]
    values=pd.to_numeric(eligible.command_time_s,errors='coerce').dropna()
    scale=_study_scale(ctx)
    expected_count=(scale['deterministic_timing_repeats'] if public_method in ('IB','GS') else scale['last_result_seed_count']) if ctx.config.full else 1
    if len(values)!=expected_count:
        exclusions=selected[~selected.timing_comparable.astype(bool)][
            ['timing_exclusion_reason']].drop_duplicates().to_dict('records')
        raise RuntimeError(
            f'{task} {method} has {len(values)}/{expected_count} comparable timing observations; '
            f'excluded records: {exclusions}')
    if not np.isfinite(values.to_numpy(float)).all() or (values<0).any():
        raise RuntimeError(f'{task} {method} contains an invalid command time')
    return values.reset_index(drop=True)


def _tradeoff_table(ctx, benchmark, triple_benchmark, command_time):
    resamples,seed=_bootstrap_controls(ctx)
    scale=_study_scale(ctx)
    records=[]
    for method in ('Conventional','FE','FE+g','IB','GS','AD'):
        deterministic=method in ('IB','GS')
        expected_count=(scale['deterministic_timing_repeats'] if deterministic else scale['last_result_seed_count']) if ctx.config.full else 1
        record=dict(method=method,
            aggregation=('one deterministic endpoint plus repeated timing measurements' if deterministic else
                         'mean and sample SD across independent command seeds; Triple metrics first reduced to a median within each command'),
            spread_definition='sample standard deviation, ddof=1; unavailable when n<2',
            displayed_center='arithmetic mean across command cases',
            bootstrap_interval_center='median; retained in exported columns only',
            endpoint_interval_status=('not applicable to deterministic endpoint' if deterministic else
                                      'estimated when at least two finite command outcomes exist'),
            bootstrap_resamples=resamples,bootstrap_seed=seed,
            timing_protocol_id=TIMING_PROTOCOL_ID,timing_scope=TIMING_SCOPE,
            timing_rank_policy=f'{expected_count} timing_comparable observations from the common protocol only')
        for task,source in [('Single',benchmark['concentration_table']),('Triple',triple_benchmark['table'])]:
            frame=_command_metrics(source);group=frame[frame.method.eq(method)];prefix=task.lower()
            valid=group[group.root_found]
            timing_values=_timing_values(ctx,task,method)
            values={'time_s':timing_values,'displacement_a':valid.finite_ka_displacement_a,
                    'stiffness_min_n_m':valid.finite_ka_stiffness_min_n_m,
                    'pressure_abs_at_exact_equilibrium_pa':valid.pressure_abs_at_exact_equilibrium_pa}
            for name,v in values.items():
                is_timing=name=='time_s'
                statistics=_distribution_statistics(v,resamples=resamples,master_seed=seed,
                    identity=('figure8',task,method,name),
                    inferential=is_timing or not deterministic,
                    inapplicable_reason='not_applicable: one deterministic endpoint, not an independent endpoint population')
                stem=f'{prefix}_{name}'
                _assign_distribution(record,stem,statistics)
                finite=pd.to_numeric(pd.Series(v),errors='coerce').to_numpy(float)
                finite=finite[np.isfinite(finite)]
                record[f'{stem}_median']=statistics['median']
                record[f'{stem}_mean']=float(np.mean(finite)) if len(finite) else np.nan
                record[f'{stem}_std']=float(np.std(finite,ddof=1)) if len(finite)>1 and (is_timing or not deterministic) else np.nan
                record[stem]=record[f'{stem}_mean']
                record[f'{stem}_sampling_unit']=(('serial timing repetition of one deterministic command' if deterministic else
                    'independent command-seed timing observation') if is_timing else
                    'deterministic endpoint (descriptive value only)' if deterministic else 'independent command seed')
            record.update({f'{prefix}_resolved_count':len(valid),f'{prefix}_endpoint_count':len(group),
                           f'{prefix}_independent_endpoint_count':0 if deterministic else int(group.seed.notna().sum()),
                           f'{prefix}_timing_observation_count':len(timing_values),
                           f'{prefix}_population_count':len(group),f'{prefix}_root_found':bool(len(valid)),
                           f'{prefix}_restoring_count':int(group.all_restoring.sum()),
                           f'{prefix}_unresolved_count':int(len(group)-len(valid))})
            record[f'{prefix}_displacement_min_a']=record[f'{prefix}_displacement_a']-record[f'{prefix}_displacement_a_std']
            record[f'{prefix}_displacement_max_a']=record[f'{prefix}_displacement_a']+record[f'{prefix}_displacement_a_std']
            record[f'{prefix}_pressure_min_pa']=record[f'{prefix}_pressure_abs_at_exact_equilibrium_pa']-record[f'{prefix}_pressure_abs_at_exact_equilibrium_pa_std']
            record[f'{prefix}_pressure_max_pa']=record[f'{prefix}_pressure_abs_at_exact_equilibrium_pa']+record[f'{prefix}_pressure_abs_at_exact_equilibrium_pa_std']
        records.append(record)
    ctx.tables['fig8_single_endpoint_population']=_command_metrics(benchmark['concentration_table'])
    ctx.tables['fig8_triple_endpoint_population']=_command_metrics(triple_benchmark['table'])
    return pd.DataFrame(records)


def _representative_triple_benchmark(ctx):
    full=ensure_benchmarks(ctx)['Triple'];items=[]
    for item in full['evaluations']:
        seed=item['spec']['seed'];method=item['spec']['method']
        if (method in ('IB','GS') or (seed==260869 and method!='AD')):
            items.append(item)
    # The fixed published AD illustration has seed260828, outside TRIPLE30.
    # Retain it separately; it is not a31st sample in the statistical bank.
    ad=_holography_run(ctx,'Triple','AD',260828)
    spec=_spec('Triple','AD',260828,ad)
    for index,target in enumerate(FIXED_BASE_TARGETS_M):items.append(_validate_spec(ctx,spec,index,target))
    table=pd.DataFrame([item['row'] for item in items])
    ctx.tables['production_fixed_triple_illustration_mechanics']=table
    return dict(full,table=table,all_table=table,evaluations=items,results=[i['result'] for i in items])


def _compensation(ctx, benchmark):
    value=ensure_triple_run(ctx,'GFE',260869,role='engineering')
    items=[item for item in benchmark['evaluations'] if item['spec']['method']=='FE+g' and item['spec']['seed']==260869]
    return dict(value,evaluations=items,cache_status=value['row']['cache_status'])


def _alpha_display_bank(ctx):
    rows=[]
    for seed in SINGLE30:
        value=ensure_alpha_bank(ctx,seed)
        for run,row in zip(value['runs'],value['table'].to_dict('records')):
            rows.append(dict(row,**{'morphology_'+k:v for k,v in pressure_descriptor(ctx,run.phase_rad).items()}))
    table=pd.DataFrame(rows)
    ctx.tables['fig7_feg_alpha_examples']=table
    ctx.tables['fig7_alpha_transition']=table
    descriptor=[c for c in table if c.startswith('morphology_') and pd.api.types.is_numeric_dtype(table[c])]
    summary=table.groupby('alpha_per_m')[descriptor].agg(['median','min','max']).reset_index()
    summary.columns=['_'.join(str(part) for part in key if str(part)) if isinstance(key,tuple) else key for key in summary.columns]
    ctx.tables['production_fig7_morphology_summary']=summary
    return ensure_alpha_bank(ctx,260828)


@contextmanager
def _hooks(ctx):
    """Scope adapters to serial figure composition; restore all legacy producers."""
    import main_feg_benchmarks as benchmarks
    from . import fig8_methods_only as tradeoff
    updates=[(mf,'_single_plot_adapter',_single_plot_bank),
             (mf,'_domain_plot_adapter',_domain_plot_bank),
             (mf,'fixed_feg_branch_pack',_branch_pack),
             (mf,'feg_triple_run',_representative_gfe_triple),
             (mf,'_triple_plot_adapter',lambda c:_representative_triple_bank(c,True)),
             (ff,'_exact_benchmark_with_concentration',lambda c:ensure_benchmarks(c)['Single']),
             (ff,'_triple_exact_benchmark_with_concentration',lambda c:ensure_benchmarks(c)['Triple']),
             (ff,'_aggregate_force_concentration',_force_aggregates),
             (ff,'_figure6_vertical_compensated_fe',_compensation),
             (benchmarks,'alpha_bank',_alpha_display_bank),
             (tradeoff,'methods_only_tradeoff_table',_tradeoff_table)]
    old=[(obj,name,getattr(obj,name)) for obj,name,_ in updates]
    try:
        for obj,name,value in updates:setattr(obj,name,value)
        yield benchmarks
    finally:
        for obj,name,value in reversed(old):setattr(obj,name,value)


def _expand_interval_limits(ax, *, xvalues=(), yvalues=()):
    """Retain existing framing while making every finite quartile visible."""
    for values,getter,setter in ((xvalues,ax.get_xlim,ax.set_xlim),(yvalues,ax.get_ylim,ax.set_ylim)):
        values=np.asarray(values,dtype=float)
        values=values[np.isfinite(values)]
        if not len(values):continue
        original=getter(); low,high=sorted(original)
        span=max(high-low,float(np.ptp(values)),np.finfo(float).eps)
        low=min(low,float(values.min())-.035*span)
        high=max(high,float(values.max())+.035*span)
        setter((low,high) if original[0]<=original[1] else (high,low))


def _add_population_intervals(ctx, figure, number):
    if number==5:
        tables=ctx.tables['fig5_exact_force_field_concentration_aggregates']
        for ax,task in zip(figure.axes,('Single','Triple')):
            xvalues=[];yvalues=[]
            for row in tables[tables.task.eq(task)].itertuples(index=False):
                x=float(row.field_concentration_fraction);y=1e3*float(row.finite_ka_stiffness_min_n_m)
                if not np.isfinite(y):continue
                bounds=(row.field_concentration_fraction_q25,row.field_concentration_fraction_q75,
                        row.finite_ka_stiffness_min_n_m_q25,row.finite_ka_stiffness_min_n_m_q75)
                if not np.all(np.isfinite(bounds)):continue
                xerr=np.array([[max(0.,x-row.field_concentration_fraction_q25)],[max(0.,row.field_concentration_fraction_q75-x)]])
                yerr=1e3*np.array([[max(0.,row.finite_ka_stiffness_min_n_m-row.finite_ka_stiffness_min_n_m_q25)],
                                 [max(0.,row.finite_ka_stiffness_min_n_m_q75-row.finite_ka_stiffness_min_n_m)]])
                ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='none',ecolor=ff.COLORS[row.method],capsize=3,elinewidth=1.5,zorder=4)
                xvalues.extend((row.field_concentration_fraction_q25,row.field_concentration_fraction_q75))
                yvalues.extend((1e3*row.finite_ka_stiffness_min_n_m_q25,1e3*row.finite_ka_stiffness_min_n_m_q75))
            _expand_interval_limits(ax,xvalues=xvalues,yvalues=yvalues)
    elif number==8:
        # Figure 8 renders sample SD for both task classes directly. Do not add
        # the former Single-only IQR overlay to the same measurements.
        return


def main_figure(ctx, number):
    """Render an established main figure with its complete production population."""
    number=int(number)
    if number not in range(1,9):raise ValueError(number)
    if number == 5:
        raise ValueError(
            "The obsolete concentration-versus-stiffness graphic is disabled; "
            "its numerical benchmark records remain available."
        )
    with _hooks(ctx) as benchmarks:
        if number<=4:figure=mf.MAIN_FEG_FIRST_FOUR[number-1](ctx)
        elif number==6:
            original=ff._triple_exact_benchmark_with_concentration
            ff._triple_exact_benchmark_with_concentration=_representative_triple_benchmark
            try:figure=benchmarks.figure_6(ctx)
            finally:ff._triple_exact_benchmark_with_concentration=original
        else:figure={7:benchmarks.figure_7,8:benchmarks.figure_8}[number](ctx)
        _add_population_intervals(ctx,figure,number)
        stem=list(ctx.figures)[-1]
        # Reserialize after adding IQRs so notebook and standalone files agree.
        from .publication_layout import polish_main_figure
        polish_main_figure(figure)
        for extension in ('png','pdf','svg'):
            figure.savefig(ctx.output_root/'figures'/f'{stem}.{extension}',dpi=240,bbox_inches='tight',facecolor='white')
        if number == 6:
            benchmarks.finalize_figure_6_visual(ctx)
    for name,table in list(ctx.tables.items()):
        if isinstance(table,pd.DataFrame):ctx.tables[name]=benchmarks.with_si_displacements(table)
    p.export_all_tables(ctx)
    (ctx.output_root/'main_cache_access.json').write_text(json.dumps(ctx.cache.access_records,indent=2))
    path=ctx.output_root/'figures'/f'{stem}.png';plt.close(figure);return path


def run(root, ctx=None):
    """Run the current figures only; root orchestration controls appendices/export."""
    ctx=prepare(root) if ctx is None else ctx
    figures=[str(main_figure(ctx,number)) for number in (1,2,3,4,6,7,8)]
    return dict(figures=figures,output_dir=str(ctx.output_root),context=ctx)
