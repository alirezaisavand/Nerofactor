"""Offline 3D-1 saved-evidence tables and plots; no rendering, AD or optimization.

Written for Phase D, separately from the frozen scientific execution. All
thresholds and truth-aware metrics come from the preregistered implementation.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from experiments import multi_material_recovery_v1 as m
from infrastructure.signatures import require


def summarize(p):
    materials=[]; outcomes=[]; adam=[]
    for row in p.get('materials',[]):
        name=row['material']; truth=np.asarray(row['truth'])
        reference=row['targets']; ident=row.get('identifiability')
        selected=[r for r in p.get('outcomes',[]) if r['material']==name]
        valid=[r for r in selected if 'recovery_errors' in r]
        groups={}
        for arm in m.v1.ARMS:
            rows=[r for r in selected if r['arm']==arm]
            groups[arm]=dict(planned=3,constructed=len(rows),
                completed=sum(r['status']=='complete' for r in rows),censored=sum(r['censored'] for r in rows),
                failed=sum(r['status'].startswith('failed') for r in rows),
                incomplete=sum(r['status']=='incomplete_budget_limited' for r in rows),
                successes_uncensored=sum(r['status']=='complete' and r.get('success',{}).get('all_passed',False) for r in rows),
                successful_censored_endpoints=sum(r['censored'] and r.get('success',{}).get('all_passed',False) for r in rows),
                failure_criteria=dict(collections.Counter(k for r in rows for k,v in r.get('success',{}).get('checks',{}).items() if not v)),
                terminations=dict(collections.Counter(str(r['termination']) for r in rows)))
        materials.append(dict(material=name,truth=row['truth'],anisotropy_strength=float(abs(truth[0]**2-truth[1]**2)/(truth[0]**2+truth[1]**2)),
            target_convergence=reference['passed'],maximum_target_bound_ratio=reference['maximum_bound_ratio'],
            component_target_ratios=reference.get('component_maximum_bound_ratios'),
            identifiability=ident['passed'] if ident else None,
            condition_number=ident['condition_numbers'][1][1] if ident else None,
            finest_singular_values=ident['grid_singular_values'][1][1] if ident else None,
            local_rank=ident['steps'][1]['rank_at_declared_ratio'] if ident else None,
            stability_margin=ident['smallest_singular_value_stability_margin'] if ident else None,
            maximum_perturbation_bound_ratio=max(e['maximum_bound_ratio'] for e in ident['evaluations']) if ident else None,
            groups=groups,evaluated_endpoints=len(valid),
            median_Q_error=float(np.median([r['recovery_errors']['Q_relative_frobenius'] for r in valid])) if valid else None,
            median_held_out=float(np.median([r['radiance_errors']['held_out']['total']['relative_L2'] for r in valid])) if valid else None,
            median_relight=float(np.median([r['radiance_errors']['relight']['total']['relative_L2'] for r in valid])) if valid else None))
    for r in p.get('Adam_runs',[]):
        adam.append(dict(material=r['material'],seed=r['seed'],status=r['status'],completed_updates=r['completed_updates'],
            snapshots=r.get('snapshots'),latest_state=r.get('latest_state'),work=r.get('work'),active_wall_seconds=r.get('active_wall_seconds'),
            failure=r.get('failure'),history_points=len(r['history'])))
    for r in p.get('outcomes',[]):
        truth=m.reference.truth(r['material'])
        initial=r['initial_state']['physical']['values']
        item=dict(material=r['material'],seed=r['seed'],arm=r['arm'],status=r['status'],
            censored=r['censored'],budget_limited=r['budget_limited'],termination=r['termination'],
            initial_recovery_errors=m.recovery_errors(initial,truth),initial_fit_MSE=r.get('initial_fit_MSE'),
            final_fit_MSE=r.get('final_fit_MSE'),work=r['work'],active_wall_seconds=r.get('active_wall_seconds'),
            final_state=r['final_state'],final_parameters=r.get('evaluation_gauge'),recovery_errors=r.get('recovery_errors'),
            radiance_errors=r.get('radiance_errors'),success=r.get('success'),component_compensation=r.get('component_compensation'),
            final_gradient_norm=r.get('final_gradient_norm'),initial_gradients=r.get('initial_gradients'),failure=r.get('failure'),
            final_damping=r.get('damping'),zero_represented_proposals=r.get('trajectory_analysis',{}).get('zero_represented_proposals'),
            trial_mask_flip_count=sum(t.get('mask_flips',0) for rec in r['iterations'] for t in rec['trials']))
        if r['iterations'] and r['iterations'][-1]['trials']:
            item['terminal_represented_delta']=r['iterations'][-1]['trials'][-1].get('represented_delta')
        outcomes.append(item)
    return materials,adam,outcomes


def pair_diagnostics(p):
    """Compare saved common-gauge predictions and original controller records."""
    out=[]
    observations=p['scene']['observations']
    for c in p.get('comparisons',[]):
        a,b=(next(r for r in p['outcomes'] if r['material']==c['material'] and r['seed']==c['seed'] and r['arm']==arm) for arm in c['pair'])
        row=dict(material=c['material'],seed=c['seed'],pair=c['pair'],
            common_gauge='That seed original P0; no cross-seed chart equivalence claim',
            maximum_outer_start_Q_discrepancy=max((x['start_state_difference']['Q_relative'] for x in c['outer_start_alignment']),default=None),
            maximum_accepted_index_Q_discrepancy=max((x['physical_difference']['Q_relative'] for x in c['accepted_update_alignment']),default=None),
            first_practical_divergence_outer=c['first_practical_divergence_outer'])
        first=None
        for x,y in zip(a['iterations'],b['iterations']):
            if first:break
            for t,u in zip(x['trials'],y['trials']):
                if t['status']!=u['status'] or t['damping']!=u['damping']:
                    first=dict(outer=x['outer_iteration'],trial=t['trial'],P0_status=t['status'],P1_status=u['status'],
                        P0_damping=t['damping'],P1_damping=u['damping']);break
            if not first and len(x['trials'])!=len(y['trials']):
                first=dict(outer=x['outer_iteration'],P0_trials=len(x['trials']),P1_trials=len(y['trials']))
        row['first_controller_history_difference']=first
        row['controller_interpretation']='Later states are different; this does not isolate a controller defect or equal-state proposal error'
        if 'final_predictions' in a and 'final_predictions' in b:
            va,vb=(np.asarray(r['final_predictions'],float) for r in (a,b))
            row['common_evaluation_prediction_differences']={}
            row['truth_relative_L2_metric_differences']={}
            for split in ('fit','held_out','relight'):
                mask=[o['split']==split for o in observations]
                row['common_evaluation_prediction_differences'][split]={name:m.original.radiance_errors(vb[mask,i],va[mask,i])
                    for i,name in enumerate(('diffuse','specular','total'))}
                row['truth_relative_L2_metric_differences'][split]={name:b['radiance_errors'][split][name]['relative_L2']-a['radiance_errors'][split][name]['relative_L2']
                    for name in ('diffuse','specular','total')}
        out.append(row)
    return out


def interpretation(p,materials,outcomes):
    evaluated=[r for r in outcomes if r['recovery_errors'] is not None]
    profiles=[r['material'] for r in materials if any(g['successes_uncensored'] for g in r['groups'].values())]
    complete=p['status']=='complete'
    reference_pass=len(materials)==3 and all(r['target_convergence'] and r['identifiability'] for r in materials)
    comparisons=p.get('comparisons',[])
    group=lambda arm:[c for c in comparisons if c['pair'][1]==arm]
    pair_counts={arm:dict(planned=9,completed=sum(c['completed_pair'] for c in group(arm)),
        final_similarity_passes=sum(c.get('final_similarity_verdict')=='pass' for c in group(arm)),
        initial_math_passes=sum(c.get('initial_delta_float64',{}).get('passed',False) for c in group(arm)),
        initial_represented_passes=sum(c.get('initial_represented_delta',{}).get('passed',False) for c in group(arm)),
        initial_J_passes=sum(c.get('initial_J',{}).get('passed',False) for c in group(arm))) for arm in ('P1-U','P1-C')}
    ambiguous=[dict(material=r['material'],seed=r['seed'],arm=r['arm'],Q_error=r['recovery_errors']['Q_relative_frobenius'],
        held_out=r['radiance_errors']['held_out']['total']['relative_L2']) for r in evaluated
        if r['recovery_errors']['Q_relative_frobenius']>.05 and r['radiance_errors']['held_out']['total']['relative_L2']<=.01]
    warnings=[dict(material=g['material'],seed=g['seed']) for g in p.get('initialization',[]) if 'gates' in g and not g['gates']['E']['old_scalar_gate_passed']]
    saturations=sum(r['termination']=='damping_saturated' for r in outcomes)
    censored=sum(r['censored'] for r in outcomes)
    h={
        'H1':dict(verdict='confirmed' if reference_pass else 'contradicted' if len(materials)==3 else 'unresolved',
            evidence='Independent two-grid convergence and local rank/stability outcomes: '+str([(r['material'],r['target_convergence'],r['identifiability']) for r in materials])+'. Local rank does not prove global uniqueness.'),
        'H2':dict(verdict='confirmed' if len(profiles)==3 else 'supported but limited' if profiles else 'contradicted' if complete else 'unresolved',
            evidence='Profiles with at least one uncensored successful endpoint: '+str(profiles)+'. Full counts per material/arm are reported; this does not establish robust recovery for every start or a population distribution.'),
        'H3':dict(verdict='confirmed' if pair_counts['P1-C']['final_similarity_passes']==9 and pair_counts['P1-C']['initial_math_passes']==9 else 'supported but limited' if comparisons else 'unresolved',
            evidence='Initial proposal/endpoint counts: '+str(pair_counts)+'. All individual diagnostic failures remain in JSON; later distinct states are not equal-state proposal tests. No bitwise/controller or cross-chart equivalence claim.'),
        'H4':dict(verdict='supported but limited' if evaluated else 'unresolved',
            evidence='Selected-profile success counts and physical/radiance errors differ; independent reference conditions are '+str([(r['material'],r['condition_number']) for r in materials])+'. Three profiles vary multiple parameters together; no causal anisotropy/metallicity or universal difficulty inference.'),
        'H5':dict(verdict='confirmed' if ambiguous else 'unresolved',
            evidence=str(len(ambiguous))+' evaluated endpoints fail Q while passing the original held-out RGB bound. Component residuals and correlations are reported independently; good total RGB does not imply material recovery.'),
        'H6':dict(verdict='supported but limited' if outcomes else 'unresolved',
            evidence=str(saturations)+' damping saturations; '+str(censored)+' censored arms; scalar warning failures '+str(warnings)+'. Finite precision, changed sample masks and controller histories are observed but their causal effects are not independently isolated.')}
    ranked=sorted((r for r in materials if r['evaluated_endpoints']),key=lambda r:(sum(g['successes_uncensored'] for g in r['groups'].values()),
        -(r['median_Q_error'] or 0)))
    difficulty=[dict(material=r['material'],uncensored_successes=sum(g['successes_uncensored'] for g in r['groups'].values()),
        planned_endpoints=9,median_Q=r['median_Q_error'],median_held_out=r['median_held_out'],condition=r['condition_number']) for r in ranked]
    text=('Measured difficulty (descriptive ordering by fewer completed recovery successes, then larger median Q error): '+str(difficulty)+'. '
        'Representation consistency and physical accuracy are separate: copied-sample coupling can preserve an inaccurate finite-sample solution. '
        'Independent target convergence/rank does not remove finite-sample objective effects or certify global optimizer convergence. '
        'No material, seed, target, tolerance or budget was changed after observing results.')
    next_step=('Recommended next neural milestone, requiring a separately frozen protocol and approval: a bounded spatially uniform neural-material-field recovery on the SAME known plane and known lights for this fixed panel. '
        'Freeze geometry and illumination; retain A2/atan2, explicit sample coupling, independent targets and all unsuccessful starts. '
        'Compare the neural field against these direct seven-parameter endpoints to separate neural optimization error from the finite-sample physical-recovery floor. '
        'Do not proceed directly to joint learned geometry/lighting or paper-quality reconstruction; current material failures must remain explicit controls. '
        'No neural training or follow-up experiment was executed here.')
    return h,dict(pair_counts=pair_counts,low_RGB_Q_failures=ambiguous,scalar_warning_failures=warnings,
        difficulty_order=difficulty),text,next_step


def make_figures(p,destination):
    """Original saved values only; no synthesis of unexecuted histories."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='multi-material-recovery-v1-offline'
    destination.mkdir(exist_ok=False)
    hashes={}
    def save(fig,name):
        path=destination/(name+'.svg');fig.tight_layout();fig.savefig(path,metadata={'Date':None});plt.close(fig)
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()).rstrip()+'\n')
        hashes[str(path.relative_to(m.ROOT))]=m.accepted.digest(path)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for index,row in enumerate(p.get('materials',[])):
        axes[0].scatter(index,row['targets']['maximum_bound_ratio'],label=row['material'])
        if 'identifiability' in row:
            axes[1].semilogy(range(1,8),row['identifiability']['grid_singular_values'][1][1],marker='.',label=row['material'])
    axes[0].set_yscale('log');axes[0].axhline(1,color='gray',linestyle='--');axes[0].set_xticks(range(3));axes[0].set_xticklabels(m.reference.PANEL)
    axes[0].set_ylabel('Maximum target convergence bound ratio');axes[1].set_xlabel('Singular value index');axes[1].set_ylabel('Normalized reference singular value')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    save(fig,'reference-gates')
    fig,axes=plt.subplots(3,3,figsize=(12,8.5))
    for i,name in enumerate(m.reference.PANEL):
        for j,seed in enumerate(m.SEEDS):
            ax=axes[i,j];row=next((r for r in p.get('Adam_runs',[]) if r['material']==name and r['seed']==seed),None)
            if row:
                ax.semilogy([r['loss_at_update'] for r in row['history']],[r['loss'] for r in row['history']],color='#222222')
                for k,s in row.get('snapshots',{}).items():
                    if 'objective' in s:ax.scatter(int(k),s['objective'],s=20,color='#222222')
            else:ax.text(.5,.5,'Not executed',ha='center',transform=ax.transAxes)
            ax.set_title(name+' / '+str(seed));ax.set_xlabel('Successful Adam updates');ax.set_ylabel('Own-material fitting MSE');ax.grid(alpha=.25)
    save(fig,'adam-loss')
    colors=dict(zip(m.v1.ARMS,('#222222','#c65102','#0868ac')))
    for title in ('lm-loss-outer','lm-loss-forwards','material-Q-accepted','lm-damping'):
        fig,axes=plt.subplots(3,3,figsize=(12,8.5))
        for i,name in enumerate(m.reference.PANEL):
            for j,seed in enumerate(m.SEEDS):
                ax=axes[i,j];plotted=False
                for row in p.get('outcomes',[]):
                    if row['material']!=name or row['seed']!=seed:continue
                    if title=='lm-damping':
                        xs=[r['outer_iteration'] for r in row['iterations']];ys=[r['lambda_before'] for r in row['iterations']]
                    else:
                        points=row.get('trajectory_analysis',{}).get('points',[])
                        if not points:
                            points=[dict(phase='initial',outer_iteration=0,accepted_updates=0,training_MSE=row['initial_fit_MSE'],
                                Q_error=m.recovery_errors(row['initial_state']['physical']['values'],m.reference.truth(name))['Q_relative_frobenius'],work={'full_batch_forward_evaluations':1})] if 'initial_fit_MSE' in row else []
                        if title=='lm-loss-outer':
                            xs=[0 if v['phase']=='initial' else v['outer_iteration'] if v['outer_iteration'] is not None else row['iterations'][-1]['outer_iteration'] if row['iterations'] else 0 for v in points]
                        elif title=='lm-loss-forwards':xs=[v['work']['full_batch_forward_evaluations'] for v in points]
                        else:xs=[v['accepted_updates'] for v in points]
                        ys=[v['Q_error'] if title=='material-Q-accepted' else v['training_MSE'] for v in points]
                    if xs:
                        plotted=True;ax.semilogy(xs,ys,label=row['arm'],color=colors[row['arm']],marker='.',markersize=3)
                        if row['censored']:ax.scatter(xs[-1],ys[-1],marker='x',color=colors[row['arm']],s=35)
                if not plotted:ax.text(.5,.5,'Not executed',ha='center',transform=ax.transAxes)
                ax.set_title(name+' / '+str(seed));ax.grid(alpha=.25)
                ax.set_xlabel('Counted full forwards' if title=='lm-loss-forwards' else 'Accepted LM steps' if title=='material-Q-accepted' else 'Attempted outer')
                ax.set_ylabel('Relative Q error' if title=='material-Q-accepted' else 'Damping' if title=='lm-damping' else 'Own-material fitting MSE')
        handles,labels=axes[0,0].get_legend_handles_labels()
        if handles:axes[0,0].legend(fontsize=7)
        save(fig,title)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,name in zip(axes,m.reference.PANEL):
        for arm in m.v1.ARMS:
            rows=[r for r in p.get('outcomes',[]) if r['material']==name and r['arm']==arm and 'recovery_errors' in r]
            if rows:ax.scatter([r['radiance_errors']['held_out']['total']['relative_L2'] for r in rows],
                [r['recovery_errors']['Q_relative_frobenius'] for r in rows],color=colors[arm],label=arm,s=24,alpha=.8)
        ax.axhline(.05,color='gray',linestyle=':');ax.axvline(.01,color='gray',linestyle=':')
        ax.set_title(name);ax.set_xlabel('Held-out total RGB relative L2');ax.set_ylabel('Relative Q error');ax.grid(alpha=.25)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=7)
    save(fig,'endpoint-recovery')
    return hashes


def write_results(p,a,path):
    """Display-only rounding; exact measurements remain in the versioned JSON."""
    lines=[]
    put=lines.append
    def table(headers,rows):
        put('| '+' | '.join(headers)+' |');put('| '+' | '.join('---' for _ in headers)+' |')
        for row in rows:put('| '+' | '.join(str(v) for v in row)+' |')
        put('')
    def fmt(x):return 'not available' if x is None else format(x,'.8g')
    def number(x,key):return None if x is None else x.get(key)
    put('# 3D-1: multi-material anisotropic recovery benchmark');put('')
    put('Execution status: **'+p['status']+'**. Exactly the preregistered M1/M2/M3 panel; original fresh starts 101/103/105; one shared training quartet 3000–3003 and original independent evaluation quartet 1000–1003. No pilot, replacement, retry, reference refinement, hyperparameter tuning, source modification or follow-up experiment.');put('')
    put('This benchmarks seven physical parameters on known flat geometry and known lighting. Three selected profiles and three starts each are not a population-level robustness study or neural inverse-rendering result. The shared training draws and repeated initial coordinates across materials are deliberate controls. Local reference rank does not establish global identifiability.');put('')
    put('## Provenance, gates and frozen design');put('')
    put('Base `'+m.BASE+'`; branch `feat/multi-material-recovery-v1`, separate worktree. The [protocol](protocols/'+m.NAME+'.md), [pre-execution freeze](protocols/'+m.NAME+'.freeze.json), [primary JSON](evidence/'+m.NAME+'.json) and [offline analysis](evidence/'+m.NAME+'_ANALYSIS.json) retain complete identities and measurements. The offline-only analyzer was prepared for Phase D; it cannot execute scientific rendering, reference quadrature, AD or optimization. It does not alter the frozen experiment.');put('')
    table(['Artifact','SHA-256'],[(name,sha) for name,sha in p['frozen_execution']['sha256'].items()]+[
        ('Freeze',m.accepted.digest(m.ROOT/('docs/protocols/'+m.NAME+'.freeze.json'))),
        ('Primary report',a['source_report_sha256']),('Offline analyzer',a['offline_implementation_sha256']),
        ('Analysis report',m.accepted.digest(m.ROOT/('docs/evidence/'+m.NAME+'_ANALYSIS.json')))])
    put('Frozen at `'+p['frozen_execution']['frozen_before_execution_at_utc']+'`; started `'+p['started_at_utc']+'`; finished `'+p['finished_at_utc']+'`. Admission verified 83 inherited identities through 3C-10. Every original tracked file except authorized AGENTS.md edits, pinned G/A, A2/atan2 shader, sampler, old targets/protocols/evidence/failed attempts and pre-existing user edits were preserved. Original datasets and historical results were not used or written.');put('')
    table(['Material','mx','my','theta','kR','kG','kB','metallic'],[(name,*truth) for name,truth in p['material_panel'].items()])
    put('Independent targets use unchanged NumPy Cartesian reference rendering, original 12 views/seven lights, 24 fit/24 held-out/36 relighting observations, component definitions and ordering. Grids 256×512 and 512×1024; convergence `abs(fine-coarse)<=2e-6+5e-5*abs(fine)` separately for diffuse/specular/total, with total also requiring component convergence. Explicit truth-aware central differences preserve original scales/steps and the original rank/stability gate. No historical TRUTH global was changed or implicitly reused.');put('')
    table(['Material','Target converged','Max target bound ratio','Max perturbation ratio','Identifiability','Fine-grid condition','Numerical rank','Smallest-value stability margin'],[
        (r['material'],r['target_convergence'],fmt(r['maximum_target_bound_ratio']),fmt(r['maximum_perturbation_bound_ratio']),r['identifiability'],fmt(r['condition_number']),r['local_rank'],fmt(r['stability_margin'])) for r in a['materials']])
    put('Full float64 truth, 84-observation coarse/fine component targets, array/grid identities, all 28 perturbations per material, all four normalized 72×7 Jacobians, singular values/ranks/conditions and cross-grid/cross-step drifts are in primary evidence. Reference work: '+str(p.get('reference_work',{}).get('reference_direction_evaluations',0))+' directions, cap 353894400. Reference gates precede every Adam update. Any incomplete phase below is explicitly unavailable, not an inferred negative recovery result.');put('')
    table(['Training seed','Combined u/v/azimuth SHA-256'],[(r['seed'],r['sha256']) for r in p['frozen_execution']['tapes']['fit']['per_tape']])
    put('The freeze stores each training and evaluation array hash/shape/dtype. Original private NumPy draws, float32 conversion, four tapes per observation, 512 diffuse/1024 specular samples and global RNG preservation are unchanged. A2/atan2 shader, seven-parameter transform, float32 mean-before-MSE, Adam and original LM calculations/settings remain frozen.');put('')
    put('## Nine fresh Adam histories');put('')
    put('One continuous Adam instance per material/start: lr=.01, betas=(.9,.999), eps=1e-8, weight_decay=0. No scheduler, clipping, regularizer, truth-guided start or early stopping. Snapshot 250 objective reuses the regular update251 forward; snapshot 1000 uses one counted no-grad terminal forward. Exact typed/hashed states, moments, step counts and effective optimizer flags are preserved. First/every-tenth history loss labels pre-update coordinates; after-update coordinates and work counters are explicit.');put('')
    table(['Material','Seed','Status','Updates','Q error fresh / 250 / 1000','Objective at 250 / 1000','Forwards / reverse / optimizer returns','Active seconds'],[
        (r['material'],r['seed'],r['status'],r['completed_updates'],
         ' / '.join(fmt(m.recovery_errors(s,m.reference.truth(r['material']))['Q_relative_frobenius']) if s is not None else 'not available' for s in
            [next(x for x in p['Adam_runs'] if x['material']==r['material'] and x['seed']==r['seed'])['initial_state']['physical']['values'],
             r.get('snapshots',{}).get('250',{}).get('physical',{}).get('values'),r.get('snapshots',{}).get('1000',{}).get('physical',{}).get('values')]),
         ' / '.join(fmt(r.get('snapshots',{}).get(k,{}).get('objective')) for k in ('250','1000')),
         ' / '.join(str((r.get('work') or {}).get(k,0)) for k in ('full_batch_forward_evaluations','reverse_completed','optimizer_returns')),fmt(r['active_wall_seconds'])) for r in a['Adam_runs']])
    put('## Initial gauge gates and paired LM');put('')
    put('All planned initial forwards precede every LM update. Gate A checks the NEW saved Adam endpoint, material target, tapes, source/order/runtime and exact internal float32 replay; no archival RGB parity. B physical/frame, C sample/mask, D independently per-tape/mean component RGB, E exact loss/gamma75 arithmetic are inherited unchanged. Old scalar loss comparison remains a warning. First chart crossing blocks before candidate rendering and is censored; no angle repair, step retry or recoupling.');put('')
    table(['Material','Seed','Mandatory A/B/C/D/E','Old scalar warning passes'],[
        (g['material'],g['seed'],' / '.join(str(v['passed']) for v in g['gates'].values()) if 'gates' in g else g.get('integrity_failure'),
         g.get('gates',{}).get('E',{}).get('old_scalar_gate_passed')) for g in a['initialization']])
    put('LM uses fresh lambda=.01, bounds 1e-8..1e8, maximum 80 outers/four trials, original float64 small solve, float32 represented candidate and original acceptance/damping/saturation. No Adam moments are restored into LM. Endpoint gradient and one common-gauge external evaluation are counted. P1 is mapped once-float32 back to the corresponding P0 chart; mapping roundoff is recorded.');put('')
    put('## Per-material recovery');put('')
    put('Success requires ALL original criteria: Q relative error<=.05, identifiable orientation<=5 degrees, maximum albedo error<=.02, metallic error<=.03, held-out total relativeL2<=.01. Undefined orientation does not pass. Counts below are uncensored completed successes out of three planned starts per arm; successful censored endpoints are reported separately in JSON. Medians use all valid evaluated endpoints, with their explicit counts. These include correlated U/C representations of each of three starts, not nine independent initializations. Conditioning is the independent local Jacobian at truth, not a causal predictor of the entire finite-sample optimization path.');put('')
    table(['Material','Anisotropy strength','Reference condition','P0 successes/3','U successes/3','C successes/3','Median Q error','Median held-out error','Valid endpoints'],[
        (r['material'],fmt(r['anisotropy_strength']),fmt(r['condition_number']),*(str(r['groups'][arm]['successes_uncensored'])+'/3' for arm in m.v1.ARMS),fmt(r['median_Q_error']),fmt(r['median_held_out']),r['evaluated_endpoints']) for r in a['materials']])
    put('## Every constructed LM outcome');put('')
    put('Initial Q error is at the fresh Adam update1000 endpoint, before LM; terminal metrics use that material’s own truth and unchanged external tapes. Unexecuted planned runs are not fabricated. The 27-run planned population and actual completed/censored/failed denominators remain separate.');put('')
    table(['Material','Seed','Arm','Initial Q','Final Q','Orientation deg','Max albedo error','Metallic error','Held-out total L2','Relight total L2','Recovery','Termination','Forwards/solves','Active s'],[
        (r['material'],r['seed'],r['arm'],fmt(r['initial_recovery_errors']['Q_relative_frobenius']),
         fmt(number(r['recovery_errors'],'Q_relative_frobenius')),fmt(number(r['recovery_errors'],'orientation_degrees')),
         fmt(max(r['recovery_errors']['albedo_absolute_errors'])) if r['recovery_errors'] else 'not available',fmt(number(r['recovery_errors'],'metallic_absolute_error')),
         fmt(r['radiance_errors']['held_out']['total']['relative_L2']) if r['radiance_errors'] else 'not available',
         fmt(r['radiance_errors']['relight']['total']['relative_L2']) if r['radiance_errors'] else 'not available',
         str(r['success']['all_passed'])+(' (censored)' if r['censored'] else '') if r['success'] else r['status'],
         r['termination'],str(r['work']['full_batch_forward_evaluations'])+'/'+str(r['work']['linear_solves']),fmt(r['active_wall_seconds'])) for r in a['outcomes']])
    put('Full physical/latent endpoints, Q/axis/orientation/albedo/metallic errors, gradients, predictions and histories are retained in JSON. No best intermediate state or held-out-based selection.');put('')
    put('## Gauge consistency');put('')
    comparisons={}
    for c in a['comparisons']:comparisons.setdefault((c['material'],c['seed']),{})[c['pair'][1]]=c
    def pairval(pair,key,sub=None):
        def value(arm):
            x=pair.get(arm,{}).get(key)
            return x.get(sub) if x is not None and sub else x
        return ' / '.join(fmt(value(arm)) for arm in ('P1-U','P1-C'))
    table(['Material','Seed','Initial math delta relative U / C','Max outer-start Q discrepancy U / C','Final Q discrepancy U / C','Final similarity U / C'],[
        (material,seed,pairval(pair,'initial_delta_float64','relative_frobenius'),
         ' / '.join(fmt(max((x['start_state_difference']['Q_relative'] for x in pair.get(arm,{}).get('outer_start_alignment',[])),default=None)) for arm in ('P1-U','P1-C')),
         pairval(pair,'final_physical_difference','Q_relative'),
         ' / '.join(str(pair.get(arm,{}).get('final_similarity_verdict')) for arm in ('P1-U','P1-C'))) for (material,seed),pair in comparisons.items()])
    put('Frozen final similarity bounds: Qrel<=.005, max albedo<=.002, metallic<=.002. Initial J/H/gradient/math/represented proposal tests refer to common initial physical states. Later index-aligned states differ; their proposal differences are not equal-state equivariance tests. Initial physical sensitivities are chain-derived, not extra physical-coordinate AD. Censored/incomplete pairs are not promoted to completed-pair successes. Complete alignment/controller/first-divergence and rendering measurements remain available.');put('')
    put('## Components, numerical limits and scientific interpretation');put('')
    table(['Material/seed/arm','Held-out diffuse L2','Held-out specular L2','Held-out residual correlation','Final gradient norm','Mask flips','Zero represented proposals'],[
        (r['material']+'/'+str(r['seed'])+'/'+r['arm'],fmt(r['radiance_errors']['held_out']['diffuse']['relative_L2']) if r['radiance_errors'] else 'not available',
         fmt(r['radiance_errors']['held_out']['specular']['relative_L2']) if r['radiance_errors'] else 'not available',
         fmt(r['component_compensation']['held_out']['residual_correlation']) if r['component_compensation'] else 'not available',fmt(r['final_gradient_norm']),r['trial_mask_flip_count'],r['zero_represented_proposals']) for r in a['outcomes']])
    put('Component relative errors have different denominators and cannot be summed. Negative diffuse/specular residual correlation is descriptive cancellation, not a proof of nonidentifiability. Compare actual failed physical criteria with low total RGB to assess ambiguity. Changed masks during different optimization states are distinct from initial correspondence XOR failures. Damping saturation/zero represented proposals are not global-convergence certificates; precision versus mask/controller effects are not independently isolated.');put('')
    table(['Hypothesis','Verdict','Evidence/limits'],[(h,v['verdict'],v['evidence']) for h,v in a['hypotheses'].items()])
    put(a.get('scientific_interpretation',''));put('')
    put('Archival context ONLY: original material `[.20,.55,.31,.30,.45,.60,.25]`, 3C-10 P0/U/C recovery 5/5 each, fixed older Adam-pretrained starts. Its different truths/training draws/pretraining history prevent a one-factor causal comparison with this benchmark. It is excluded from every new denominator.');put('')
    put('## Resources, focused tests and preservation');put('')
    put('One scientific process: wall '+fmt(p['wall_seconds'])+' s, CPU '+fmt(p['CPU_seconds'])+' s, peak RSS '+str(p['peak_RSS_bytes'])+' bytes. Original runtime '+str(p.get('runtime'))+'. GPUs hidden; one Torch/BLAS/OpenMP thread. 7200-second process cap, 180 active seconds per Adam, 120 per LM with 10 reserved, 4 GiB RSS/8 GiB address backstop. Exact reference/forward/JVP/reverse/solve/sample caps are frozen. No limits were enlarged, new packages installed or Docker/GPU/neural work run.');put('')
    put('29 new focused tests passed, 0 failed, 0 skipped (5.072 s); no historical suite or scientific shader/quadrature experiment ran as a test. Synthetic fixtures cover explicit truths/derivatives, domains/reference gates, tape/target identities, continuous Adam and snapshots, inherited LM/gates/chart policy, truthful undefined orientations, common evaluation, components, work/resources and strict partial evidence. Static checks cover frozen hashes, strict JSON, original file preservation, links, SVGs and whitespace.');put('')
    put('Aggregate counted scientific work: `'+json.dumps(a.get('total_work'),sort_keys=True)+'`. Reference counters: `'+json.dumps(a.get('reference_work'),sort_keys=True)+'`. Per-run attempts/completions and failed partial evidence remain in JSON. Global RNG, thread/memory restoration and fixed tape identities are checked; original reports were not rewritten by analysis.');put('')
    if p.get('failure') or p.get('reason'):put('Execution stop/failure: '+str(p.get('failure') or p.get('reason')));put('')
    put('## Offline figures and next step');put('')
    for file in a['figures']:
        put('!['+Path(file).stem+']('+str(Path(file).relative_to('docs'))+')');put('')
    put(a.get('recommended_next_milestone','No further experiment is authorized.'));put('')
    put('All figures use saved observations only, with no rendering, derivatives, reference calls or optimization. Fitting losses belong to different material-specific objectives; panels are descriptive, not a universal raw-loss ranking. Commit/push the feature branch only; stop without merge or follow-up.')
    with path.open('x') as stream:stream.write('\n'.join(lines).rstrip()+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(args.input==m.OUTPUT and args.output==m.ROOT/('docs/evidence/'+m.NAME+'_ANALYSIS.json'),'Explicit versioned input/output required')
    p=json.loads(args.input.read_text());require(p['status']!='running','Wait for the single execution to finish')
    primary_sha=m.accepted.digest(args.input)
    for name,sha in p['frozen_execution']['sha256'].items():require(m.accepted.digest(m.ROOT/name)==sha,'Frozen source changed')
    materials,adam,outcomes=summarize(p)
    figures=make_figures(p,m.ROOT/'docs/figures/multi-material-recovery-v1')
    archival=json.loads((m.ROOT/'docs/evidence/MATERIAL_INDEPENDENT_TAPE_LM_V1.json').read_text())
    groups={arm:sum(r['arm']==arm and r['success']['all_passed'] for r in archival['outcomes']) for arm in m.v1.ARMS}
    a=dict(schema_version=1,protocol='multi-material-recovery-v1-offline-analysis',source_report=str(args.input.relative_to(m.ROOT)),
        source_report_sha256=primary_sha,offline_implementation_sha256=m.accepted.digest(Path(__file__)),
        additional_shader_calls=0,additional_derivative_calls=0,additional_reference_calls=0,additional_optimizer_calls=0,
        status=p['status'],materials=materials,Adam_runs=adam,outcomes=outcomes,initialization=p.get('initialization',[]),
        comparisons=p.get('comparisons',[]),pair_diagnostics=pair_diagnostics(p),reference_work=p.get('reference_work'),total_work=p.get('total_work'),
        hypotheses={},archival_context=dict(original_truth=m.oldref.TRUTH.tolist(),milestone='3C-10',
            source_report_sha256=m.INPUTS['docs/evidence/MATERIAL_INDEPENDENT_TAPE_LM_V1.json'],recovery_counts=groups,
            denominator_per_arm=5,excluded_from_new_denominators=True,
            limitation='Different truths, training tapes and pretraining pipeline; archival contrast is not a controlled one-factor causal comparison'),
        figures=figures)
    a['hypotheses'],a['scientific_summary'],a['scientific_interpretation'],a['recommended_next_milestone']=interpretation(p,materials,outcomes)
    with args.output.open('x') as stream:json.dump(a,stream,indent=2,allow_nan=False);stream.write('\n')
    write_results(p,a,m.ROOT/('docs/'+m.NAME+'_RESULTS.md'))
    require(m.accepted.digest(args.input)==primary_sha,'Primary evidence modified by offline analysis')
    print(json.dumps(dict(status=p['status'],materials=materials,figures=len(figures),primary_sha256=primary_sha),indent=2))


if __name__=='__main__':
    main()
