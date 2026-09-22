"""Export diagnostics from the commands already used in the final benchmark."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def _objective_diagnostics(task, method, value):
    fields = dict(objective_initial=None, objective_final=None,
                  gradient_norm_initial=None, gradient_norm_final=None,
                  gradient_ratio=None)
    if method in ('IB', 'GS'):
        return dict(fields, gradient_applicable=False,
                    diagnostic_status='N/A: prescribed method has no scalar optimized objective')
    if method == 'AD':
        metadata = value['run'].solver_result.metadata
        fields.update(objective_initial=metadata.get('initial_loss'),
                      objective_final=metadata.get('final_loss'),
                      gradient_norm_final=value['row'].get('terminal_gradient_l2'))
        return dict(fields, gradient_applicable=True,
                    diagnostic_status='Initial phase-gradient unavailable; stored losses and terminal gradient reported')
    initial = np.asarray(value['initial_phase'], dtype=float)
    terminal = np.asarray(value['phase'], dtype=float)
    if task == 'Single':
        from hat_revision_pipeline.gorkov_core import reduce_gauge
        objective = value['objective']
        j0, g0 = objective.fun_grad(reduce_gauge(initial, objective.gauge_index))
        jf, gf = objective.fun_grad(reduce_gauge(terminal, objective.gauge_index))
    else:
        from hat_revision_pipeline.multitrap import evaluate_multitrap_objective
        run = value['run']
        last = run if method == 'Conventional' else run.stages[-1]
        options = dict(force_epsilon=last.force_epsilon,
                       uniformity_epsilon=last.uniformity_epsilon)
        before = evaluate_multitrap_objective(value['problem'], initial, value['config'], **options)
        after = evaluate_multitrap_objective(value['problem'], terminal, value['config'], **options)
        j0, g0, jf, gf = before.value, before.gradient_reduced, after.value, after.gradient_reduced
    norm0, normf = float(np.linalg.norm(g0)), float(np.linalg.norm(gf))
    return dict(objective_initial=float(j0), objective_final=float(jf),
                gradient_norm_initial=norm0, gradient_norm_final=normf,
                gradient_ratio=normf / norm0 if norm0 > 0 else None,
                gradient_applicable=True,
                diagnostic_status='Initial and terminal commands evaluated with identical final-stage objective settings')


def export_current_table6(ctx):
    """Evaluate saved commands; never invoke a solver or physical validator."""
    from hat_revision_pipeline import production_main
    key = production_main._scale_key(ctx, 'production_commands') if ctx.config.full else 'engineering_commands'
    if key not in ctx.memo:
        raise RuntimeError('Generate the final benchmark before exporting Table 6 diagnostics')
    bank = ctx.memo[key]
    rows = []
    for (task, method, seed), value in bank.items():
        record = value['row']
        expected_optimizer = 'L-BFGS-B' if task == 'Single' else 'BFGS'
        if method in ('FE', 'GFE') and record.get('optimizer') != expected_optimizer:
            raise RuntimeError('FE/GFE Table 6 command does not use the required solver')
        rows.append(dict(task=task, method=method, seed=seed,
            phase_content_id=record.get('phase_content_id'),
            optimizer=record.get('optimizer', 'Adam' if method == 'AD' else method),
            evaluator_backend=record.get('evaluator_backend', 'method-native'),
            actual_iterations=record.get('iterations'),
            native_success=record.get('native_success'),
            stationary_gtol=record.get('stationary_gtol'),
            terminal_gradient_inf=record.get('terminal_gradient_inf'),
            gradient_units=record.get('terminal_gradient_units', 'N/A'),
            recorded_endpoint_outcome=record.get('native_message', record.get('terminal_gradient_status')),
            **_objective_diagnostics(task, method, value)))
    frame = pd.DataFrame(rows)
    frame['terminal_gradient_norm'] = frame.gradient_norm_final
    # The first predeclared seed is the representative; no selection by outcome.
    selected = frame.groupby(['task', 'method'], sort=False, as_index=False).head(1).copy()
    if len(selected) != 12 or selected[['task', 'method']].duplicated().any():
        raise RuntimeError('Table 6 requires one representative for each of six methods in two tasks')
    folder = Path(ctx.memo['notebook_root']) / 'table6_exports' / str(ctx.memo['notebook_mode'])
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / 'methods_all_command_gradients.csv', index=False)
    selected.to_csv(folder / 'methods_selected_gradients.csv', index=False)
    measures = ['objective_initial', 'objective_final', 'gradient_norm_initial',
                'gradient_norm_final', 'gradient_ratio', 'actual_iterations']
    summaries = []
    for (task, method), group in frame.groupby(['task', 'method'], sort=False):
        summary = dict(task=task, method=method, command_count=len(group))
        for metric in measures:
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            summary[metric + '_n'] = len(values)
            if len(values):
                q1, q3 = values.quantile([.25, .75])
                summary.update({metric+'_median': float(values.median()),
                    metric+'_q25': float(q1), metric+'_q75': float(q3),
                    metric+'_iqr': float(q3-q1), metric+'_min': float(values.min()),
                    metric+'_max': float(values.max())})
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(folder / 'methods_gradient_population.csv', index=False)
    metadata = dict(title='Current benchmark native objective diagnostics',
        mode=ctx.memo['notebook_mode'], representative_selection='first predeclared seed per method and task',
        command_count=len(frame), rows=json.loads(selected.to_json(orient='records')),
        note='Objective scales differ across methods. IB/GS gradients are N/A. Initial and terminal FE/GFE diagnostics use identical final-stage settings. Diagnostics are outside synthesis timing.')
    (folder / 'methods_selected_gradients.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    def tex(value):
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return 'N/A'
        if isinstance(value, (float, np.floating)): return f'{value:.6g}'
        return str(value).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
    lines = [r'\begin{table}[htbp]\centering\scriptsize',
        r'\caption{Native objective diagnostics for the first predeclared final-benchmark command per method and task. Initial and terminal values use identical final-stage settings. IB/GS gradients are not applicable; the AD initial gradient is unavailable.}',
        r'\label{tab:selected_gradients}', r'\begin{tabular}{llrrrrrr}\toprule',
        r'Task & Method & $J_0$ & $J_f$ & $\|g_0\|_2$ & $\|g_f\|_2$ & Ratio & Iter.\\\midrule']
    for row in selected.to_dict('records'):
        lines.append(' & '.join(tex(row[k]) for k in ['task','method',*measures])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (folder / 'methods_selected_gradients.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return selected, folder
