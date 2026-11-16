"""Saved-response analysis for the fixed E.14 two-light panel; no rendering."""
import numpy as np
from benchmarks.material_information_e14 import information_row,secants
from benchmarks.bell_e11 import convergence
from experiments.bell_multiview_e11g import CASES,SCREENS,spectrum
from experiments.analyze_bell_highlight_e11d import response_fields
from infrastructure.signatures import require

FACTORS=('orientation','major','minor','isotropic','logmean','logratio')


def analyze(fine,coarse,valid):
    require(fine.shape==coarse.shape==(2,2,4,84,12,3,3) and valid.shape==(4,84) and
            valid.dtype==np.bool_ and valid[0].all(),'Fixed E14 paired layout required')
    convergence_rows=[];png_domain=True
    for arm in range(2):
        for light in range(2):
            require(np.isnan(fine[arm,light,~valid]).all() and np.isnan(coarse[arm,light,~valid]).all(),
                    'Unmeasured correspondences must remain NaN')
            for view in range(4):
                f=fine[arm,light,view,valid[view]];c=coarse[arm,light,view,valid[view]]
                require(len(f)>0 and np.isfinite(f).all() and np.isfinite(c).all() and (f>=0).all() and (c>=0).all(),
                        'Missing/nonfinite valid observations')
                require(np.array_equal(f[:,:,0]+f[:,:,1],f[:,:,2]),'Component sums differ')
                png_domain &= bool((f<=1).all() and (c<=1).all())
                for k,case in enumerate(CASES):
                    convergence_rows.append(dict(material=arm,light=light,view_index=view,case=case,
                        **convergence(f[:,k,2],c[:,k,2])))
    converged=all(x['passed'] for x in convergence_rows)
    base=dict(convergence=convergence_rows,all_cases_converged=converged,png_domain_valid=png_domain,
              valid_view_counts=valid.sum(0).tolist(),new_geometry_queries=0,old_decisions_replaced=False,
              full_dataset_admitted=False)
    if not png_domain:
        base.update(decision='INCONCLUSIVE_CLIPPED_REFERENCE',reason='Frozen legacy response gates require unclipped [0,1] radiance; no clipping or exposure repair.')
        return base
    rows=[];coverage=[];conditioning=[]
    resolved=np.zeros((2,2,4,84,len(FACTORS)),bool)
    code_gate=np.zeros_like(resolved)
    amplitudes=np.full(resolved.shape,np.nan)
    for arm in range(2):
        for light in range(2):
            for view in range(4):
                ix=np.flatnonzero(valid[view]);f=fine[arm,light,view,ix];c=coarse[arm,light,view,ix]
                fields=response_fields(f[:,:8],c[:,:8])
                for name,pair in [('logmean',(8,9)),('logratio',(10,11))]:
                    order=[0,pair[0],pair[1],3,4,5,6,7]
                    fields[name]=response_fields(f[:,order],c[:,order])['orientation']
                for k,name in enumerate(FACTORS):
                    resolved[arm,light,view,ix,k]=fields[name]['resolved']
                    code_gate[arm,light,view,ix,k]=fields[name]['code_gate']
                    amplitudes[arm,light,view,ix,k]=fields[name]['amplitude']
        for point in range(84):
            views=np.flatnonzero(valid[:,point]);f=fine[arm,:,:,point][:,views];c=coarse[arm,:,:,point][:,views]
            info={}
            for name,a,b in [('L2',f[0],c[0]),('rotated_L2',f[1],c[1]),('two_light',f.reshape(-1,12,3,3),c.reshape(-1,12,3,3))]:
                info[name]=information_row(a,b)
            rows.append(dict(material=arm,probe_index=point,valid_view_count=len(views),view_indices=views.tolist(),information=info))
            row=dict(material=arm,probe_index=point,valid_view_count=len(views))
            for basis in ('legacy_scaled','theta_logmean_logratio'):
                if basis=='legacy_scaled':
                    rgb=f[...,2,:];low=c[...,2,:]
                    J=np.stack([(rgb[...,a,:]-rgb[...,b,:])/2 for a,b in [(1,2),(3,4),(5,6)]],-1)
                    K=np.stack([(low[...,a,:]-low[...,b,:])/2 for a,b in [(1,2),(3,4),(5,6)]],-1)
                else:J,K=secants(f),secants(c)
                single=J[0].reshape(-1,3)/np.sqrt(len(views));joint=J.reshape(-1,3)/np.sqrt(2*len(views))
                error=float(np.linalg.norm((J-K).reshape(-1,3)/np.sqrt(2*len(views)),ord=2))
                row[basis]=dict(L2_multiview_RMS=spectrum(single),two_light_multiview_RMS=spectrum(joint),
                               grid_difference_operator_norm_RMS=error,
                               grid_proxy_smin_lower=max(0.,spectrum(joint)['singular_values'][-1]-error))
            conditioning.append(row)
        groups=[('all',np.ones(84,bool)),('at_least_two',valid.sum(0)>=2)]
        groups += [('exactly_'+str(n),valid.sum(0)==n) for n in range(1,5)]
        for name,matched in groups:
            for k,factor in enumerate(FACTORS):
                a=resolved[arm,0,:,:,k].any(0);b=resolved[arm,1,:,:,k].any(0);both=a|b
                coverage.append(dict(material=arm,group=name,matched_points=int(matched.sum()),factor=factor,
                    L2_resolved=int((a&matched).sum()),rotated_L2_resolved=int((b&matched).sum()),two_light_resolved=int((both&matched).sum()),
                    L2_two_codes=int((code_gate[arm,0,:,:,k].any(0)&matched).sum()),
                    two_light_two_codes=int((code_gate[arm,:,:,:,k].any((0,1))&matched).sum()),
                    L2_median_max_response=float(np.nanmedian(np.nanmax(amplitudes[arm,0,:,:,k],axis=0)[matched])) if matched.any() else None,
                    two_light_median_max_response=float(np.nanmedian(np.nanmax(amplitudes[arm,:,:,:,k],axis=(0,1))[matched])) if matched.any() else None))
    screens=[]
    for arm in range(2):
        matched=[x for x in conditioning if x['material']==arm and x['valid_view_count']>=2];n=len(matched)
        widths={k:next(x['two_light_resolved'] for x in coverage if x['material']==arm and x['group']=='at_least_two' and x['factor']==k) for k in ('major','minor')}
        for basis in ('legacy_scaled','theta_logmean_logratio'):
            usable=[x[basis] for x in matched if x[basis]['L2_multiview_RMS']['singular_values'][-1]>0 and x[basis]['two_light_multiview_RMS']['condition_number'] is not None]
            gain=float(np.median([x['two_light_multiview_RMS']['singular_values'][-1]/x['L2_multiview_RMS']['singular_values'][-1] for x in usable])) if usable else None
            ratio=float(np.median([x['two_light_multiview_RMS']['condition_number']/x['L2_multiview_RMS']['condition_number'] for x in usable])) if usable else None
            passed=bool(converged and len(usable)==n and n>0 and gain>=SCREENS['median_smin_gain'] and ratio<=SCREENS['median_condition_ratio'] and all(v/n>=SCREENS['width_coverage'] for v in widths.values()))
            screens.append(dict(material=arm,basis=basis,matched_points=n,median_paired_RMS_smin_gain=gain,
                                median_paired_condition_ratio=ratio,major_resolved=widths['major'],minor_resolved=widths['minor'],
                                original_screen_passed=passed))
    base.update(per_probe_information=rows,per_probe_conditioning=conditioning,response_coverage=coverage,retained_screen=screens,
                decision=('REFERENCE_ACCURACY_UNRESOLVED' if not converged else 'DESCRIPTIVE_SCREEN_PASSED_RESEARCH_REVIEW_REQUIRED' if all(x['original_screen_passed'] for x in screens) else 'FROZEN_SCREEN_NOT_MET'),
                interpretation='Conditional three-parameter finite-secant diagnostics; known color/metal. No optimizer or global/seven-parameter recovery claim. Noise is hypothetical, grid errors deterministic.')
    return base
