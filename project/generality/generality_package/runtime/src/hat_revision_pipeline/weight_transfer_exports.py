"""Compact manuscript records from the bounded weight-transfer experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def export_weight_results(raw,summary,output_root):
    out=Path(output_root)/'weight_test';out.mkdir(parents=True,exist_ok=True)
    raw.to_csv(out/'weight_test_raw_with_curvature.csv',index=False)
    summary.to_csv(out/'weight_test_summary_with_curvature.csv',index=False)
    display=summary[['task','method','alpha_per_m','wzz','median_displacement_a','worst_displacement_a',
        'median_dz_um','min_stiffness_mn_m','all_roots','all_restoring','command_time_s','iterations']].copy()
    display.columns=['Case','Method','Alpha (1/m)','Wzz','Median displacement (a)','Worst displacement (a)',
        'Median dz (um)','Minimum stiffness (mN/m)','All roots found','All locally restoring','Command time (s)','Iterations']
    for c in display.select_dtypes(include='number').columns:
        if c not in ['Alpha (1/m)','Wzz','Iterations']:display[c]=display[c].map(lambda x:f'{x:.4g}' if np.isfinite(x) else 'Not resolved')
    for suffix,sep in [('csv',','),('tsv','\t')]:display.to_csv(out/f'table_weight_transfer.{suffix}',index=False,sep=sep)
    (out/'table_weight_transfer.txt').write_text('ALL RUNS ARE SMOKE.\n\n'+display.to_string(index=False)+'\n')
    from .manuscript_tables import _latex
    (out/'table_weight_transfer.tex').write_text(_latex(display, 'Paired weight-transfer SMOKE test; all parameters are recorded in the raw table.'))
    refs={task:summary[(summary.task==task)&(summary.method=='FE+g')&(summary.alpha_per_m==(10 if task=='Single' else 3000))&(summary.wzz==(1 if task=='Single' else 10))].iloc[0] for task in ['Single','Triple']}
    candidates=[]
    for wz in (10.,1.):
        for alpha in (10.,100.,1000.):
            single=summary[(summary.task=='Single')&(summary.method=='FE+g')&(summary.alpha_per_m==alpha)]
            triple=summary[(summary.task=='Triple')&(summary.method=='FE+g')&(summary.alpha_per_m==alpha)&(summary.wzz==wz)]
            if single.empty or triple.empty:continue
            a,b=single.iloc[0],triple.iloc[0]
            # Preserve current precision within 1% numerical tolerance; require every root restoring.
            passed=all(bool(r.all_roots and r.all_restoring) and r.median_displacement_a<=1.01*refs[t].median_displacement_a and
                r.worst_displacement_a<=1.01*refs[t].worst_displacement_a for t,r in [('Single',a),('Triple',b)])
            candidates.append(dict(alpha_per_m=alpha,triple_wzz=wz,preserves_current_precision=passed,
                single_displacement_ratio=a.median_displacement_a/refs['Single'].median_displacement_a,
                triple_displacement_ratio=b.median_displacement_a/refs['Triple'].median_displacement_a))
    eligible=[x for x in candidates if x['preserves_current_precision']]
    decision=dict(evidence_status='SMOKE',candidate_alpha_per_m=[10,100,1000],
        rule='Current FE+g median and worst-target displacement preserved within 1%; all exact roots locally restoring. No hypothesis of equivalence is inferred from one seed.',
        candidates=candidates,shared_alpha_adopted=bool(eligible),selected_alpha_per_m=min((x['alpha_per_m'] for x in eligible),default=None),
        baseline_single_alpha=10,baseline_triple_alpha=3000,main_figures_changed=False)
    (out/'alpha_decision.json').write_text(json.dumps(decision,indent=2))
    sentences=['ALL RUNS ARE SMOKE. ONE LOCKED INITIALIZATION PER CASE. NOT MANUSCRIPT RESULTS.',
        'Single uses one target; Triple summaries use T1-T3 of one shared command, not three independent trials.','']
    for row in summary.itertuples(index=False):
        sentences.append(f'[SMOKE; {row.run_id}] At alpha={row.alpha_per_m:g} 1/m and Wzz={row.wzz:g}, {row.task} {row.method} '
          f'had median/worst exact equilibrium displacement {row.median_displacement_a:.6g}/{row.worst_displacement_a:.6g} a, '
          f'median vertical offset {row.median_dz_um:.6g} um, minimum symmetric stiffness {row.min_stiffness_mn_m:.6g} mN/m, '
          f'and measured command time {row.command_time_s:.6g} s over {row.iterations} iterations. '
          f'All roots found: {row.all_roots}; all locally restoring: {row.all_restoring}.')
    sentences+=['','Shared alpha adopted: '+str(decision['shared_alpha_adopted']),
        'Gor\'kov remains the design surrogate; all listed equilibria use the independent finite-ka elastic bead plus effective gravity.',
        'FE+g changes the equilibrium force residual; the added linear gravitational potential has zero Hessian.']
    (out/'weight_transfer_results_smoke.txt').write_text('\n'.join(sentences)+'\n')
    (out/'weight_transfer_mock_sentences.txt').write_text('MOCK TEMPLATES; PLACEHOLDERS ARE NOT MEASUREMENTS.\n\n'
        'At the shared force weight alpha={alpha} 1/m, FE+g produced Single and Triple median exact-equilibrium displacements of '
        '{single_displacement_a} a and {triple_displacement_a} a, with command times of {single_time_s} s and {triple_time_s} s.\n'
        'Relative to the task-specific defaults, the shared setting changed Single displacement by {single_change_percent}% '
        'and Triple displacement by {triple_change_percent}%; the worst Triple displacement was {worst_triple_a} a.\n')
    return display,decision
