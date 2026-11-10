"""Bounded SDF/mesh/normal measurements; native imports are isolation-gated.

No training, data loaders, rendering, second derivatives, mesh modifications or
checkpoint writes. Descriptive local measurements never establish provenance.
"""
from copy import deepcopy
import datetime
import inspect
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from infrastructure import geometry_scientific_audit as evidence
from infrastructure.signatures import require,read_report,write_report
from infrastructure.stage_sources import ROOT,PINS,baseline,activate,imported_sources,digest
from infrastructure.g_checkpoint import load_checked,historical_schema,network_comparison,strict_network_load,identities
from infrastructure.paths import PathResolver
from infrastructure.runs import RunStore,validate_run_id
from infrastructure.g_lifecycle import runtime_guards,IMAGE

PROTOCOL='bell-geometry-sdf-mesh-audit-v1'
BASE='3fe197bbdc37957c187c8589fffdb123aaac692d'
DOC='docs/GEOMETRY_SDF_MESH_AUDIT_V1.md'
H=2/127
SETTINGS={'face_limit':1920,'vertex_limit':256,'controls_per_view':32,'batch_size':128,'point_budget':20000,
          'gradient_epsilon':1e-6,'sign_epsilon':1e-7,'direction_cosine':0.8660254037844386,
          'CPU_depth_absolute':2e-4,'CPU_tie_absolute':5e-5,'CPU_barycentric_absolute':1e-9,
          'near_edge_barycentric':1e-5,'tiny_triangle_area':1e-10,'CPU_parallel_relative':1e-12}
EXPECTED_BACK={'14':0,'58':2,'11':1}
COORDINATE_AUDIT_SHA='9067af609b5289440ce07d9ca47370f830952c9dfa64bd72d24aa12e80f5c327'


def mesh_arrays(ref):
    with evidence.verified(ref) as stream:
        header=[]
        for _ in range(32):
            line=stream.readline(256);require(line and len(line)<256,'Invalid PLY header')
            text=line.decode('ascii').strip()
            if not text.startswith('comment '):header.append(text)
            if text=='end_header':break
        require(len(header)==9 and header[:2]==['ply','format binary_little_endian 1.0'] and
                header[3:6]==['property float x','property float y','property float z'] and
                header[7:]==['property list uchar int vertex_indices','end_header'],'Unreviewed PLY layout')
        nv=int(header[2].split()[-1]);nf=int(header[6].split()[-1]);raw=stream.read()
        require(0<nv<=100000 and 0<nf<=100000 and len(raw)==12*nv+13*nf,'Invalid mesh sizes')
        v=np.frombuffer(raw[:12*nv],dtype='<f4').reshape(nv,3).astype(np.float64)
        records=np.frombuffer(raw[12*nv:],dtype=np.dtype([('count','u1'),('indices','<i4',(3,))]))
        f=records['indices'].astype(np.int64)
        require((records['count']==3).all() and f.min()>=0 and f.max()<nv and np.isfinite(v).all(),'Invalid mesh payload')
    return v,f


def face_geometry(vertices,faces):
    require(len(faces)<=100000 and faces.ndim==2 and faces.shape[1]==3,'Triangle CPU budget/shape exceeded')
    a,b,c=(vertices[faces[:,i]] for i in range(3));cross=np.cross(b-a,c-a);length=np.linalg.norm(cross,axis=1)
    require(np.isfinite(cross).all() and (length>0).all(),'Nonfinite/degenerate original faces; no repair')
    return (a+b+c)/3,cross/length[:,None],length/2


def spaced(count,limit):
    require(count>0,'Cannot sample empty population')
    return np.linspace(0,count-1,min(count,limit),dtype=np.int64)


def select_faces(centers,areas):
    if len(areas)<=SETTINGS['face_limit']:return np.arange(len(areas),dtype=np.int64)
    radius=np.linalg.norm(centers,axis=1);indices=np.arange(len(areas))
    by_area=np.lexsort((indices,areas));chosen=[]
    for group in np.array_split(by_area,4):
        ordered=group[np.lexsort((group,radius[group]))]
        chosen.extend(ordered[spaced(len(ordered),SETTINGS['face_limit']//4)].tolist())
    # Preserve both extrema without hiding the smallest triangle or sphere rim.
    for offset,index in enumerate([int(np.argmin(areas)),int(np.argmax(radius))]):
        if index not in chosen:chosen[-1-offset]=index
    require(len(set(chosen))==len(chosen),'Duplicate deterministic face selection')
    return np.asarray(chosen,np.int64)


def cpu_intersections(vertices,faces,origin,direction):
    """Two-sided float64 Moller-Trumbore; no BVH, culling or geometry repair."""
    centers,normals,areas=face_geometry(vertices,faces)
    direction=np.asarray(direction,np.float64)
    require(direction.shape==(3,) and np.isfinite(direction).all() and np.linalg.norm(direction)>0,'Invalid CPU ray direction')
    direction=direction/np.linalg.norm(direction)
    a,b,c=(vertices[faces[:,i]] for i in range(3));e1=b-a;e2=c-a
    p=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,p)
    scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)
    valid=np.abs(det)>SETTINGS['CPU_parallel_relative']*scale
    inv=np.zeros_like(det);inv[valid]=1/det[valid]
    tvec=np.asarray(origin,np.float64)-a;u=np.einsum('ij,ij->i',tvec,p)*inv
    q=np.cross(tvec,e1);v=q@direction*inv;distance=np.einsum('ij,ij->i',e2,q)*inv
    eps=SETTINGS['CPU_barycentric_absolute']
    ids=np.flatnonzero(valid & (u>=-eps) & (v>=-eps) & (u+v<=1+eps) & (distance>1e-8))
    ids=ids[np.lexsort((ids,distance[ids]))]
    return [{'face_index':int(i),'distance':float(distance[i]),'position':(origin+distance[i]*direction).tolist(),
             'normal':normals[i].tolist(),'area':float(areas[i]),'barycentric':[float(1-u[i]-v[i]),float(u[i]),float(v[i])],
             'near_edge':bool(min(u[i],v[i],1-u[i]-v[i])<=SETTINGS['near_edge_barycentric']),
             'tiny_triangle':bool(areas[i]<=SETTINGS['tiny_triangle_area'])} for i in ids]


def ray_comparison(vertices,faces,row):
    hits=cpu_intersections(vertices,faces,np.asarray(row['origin']),np.asarray(row['direction']))
    result={'positive_intersections':len(hits),'candidates':hits[:8],'candidate_limit':8,
            'classification':'unresolved','basis':'No CPU triangle intersection'}
    if not hits:return result
    first=hits[0];closest=min(hits,key=lambda h:abs(h['distance']-row['saved_depth']))
    ties=[h for h in hits if h['distance']-first['distance']<=SETTINGS['CPU_tie_absolute']]
    matched=abs(closest['distance']-row['saved_depth'])<=SETTINGS['CPU_depth_absolute']
    first_match=abs(first['distance']-row['saved_depth'])<=SETTINGS['CPU_depth_absolute']
    raw=np.asarray(row['raw_normal']);raw=raw/np.linalg.norm(raw)
    cosine=float(np.dot(raw,closest['normal']))
    distinct=[h['distance'] for h in hits if h['distance']>first['distance']+SETTINGS['CPU_tie_absolute']]
    ambiguous=len(ties)>1 or first['near_edge'] or first['tiny_triangle'] or closest['near_edge'] or closest['tiny_triangle']
    result.update(first=first,closest_saved_depth=closest,first_tie_count=len(ties),saved_depth_match=matched,
                  first_hit_match=first_match,raw_vs_matched_face_cosine=cosine,
                  position_error=float(np.max(np.abs(np.asarray(closest['position'])-row['position']))),
                  next_distinct_gap=(min(distinct)-first['distance']) if distinct else None,
                  geometric_exterior_eligible=bool(first_match and not ambiguous and
                      np.any((np.asarray(row['origin'])<vertices.min(0)) | (np.asarray(row['origin'])>vertices.max(0)))))
    if ambiguous:result.update(classification='ambiguous',basis='Near-edge/tiny-face/first-hit tie observed; numerical cause is not established')
    elif matched and not first_match:result.update(classification='supported_by_evidence',basis='Saved hit is later than an independently identified CPU first hit; cause unassigned')
    elif matched and cosine<SETTINGS['direction_cosine']:result.update(classification='supported_by_evidence',basis='Saved raw normal disagrees with uniquely matched original triangle cross normal; cause unassigned')
    elif matched:result.update(classification='supported_by_evidence',basis='Saved hit/normal supported by unique CPU triangle geometry; no global outward or causal claim')
    else:result.update(classification='unresolved',basis='No CPU candidate matches saved depth within the frozen bound')
    return result


def silhouette_disagreements(hit,mask):
    """Chebyshev pixel distance to foreground 4-neighbor silhouette boundary."""
    require(hit.shape==mask.shape and hit.dtype==mask.dtype==np.bool_,'Boolean silhouette dimensions required')
    padded=np.pad(mask,1,constant_values=False)
    eroded=mask & padded[:-2,1:-1] & padded[2:,1:-1] & padded[1:-1,:-2] & padded[1:-1,2:]
    boundary=mask & ~eroded;zone=boundary.copy();fp=hit & ~mask;fn=~hit & mask
    result={'false_positive':int(fp.sum()),'false_negative':int(fn.sum()),'boundary_pixels':int(boundary.sum()),'within_pixels':{}}
    for distance in range(1,5):
        p=np.pad(zone,1,constant_values=False)
        zone=np.logical_or.reduce([p[dy:dy+mask.shape[0],dx:dx+mask.shape[1]] for dy in range(3) for dx in range(3)])
        if distance in (1,2,4):result['within_pixels'][str(distance)]={'false_positive':int((fp & zone).sum()),'false_negative':int((fn & zone).sum())}
    result['deeper_than_4']={'false_positive':int((fp & ~zone).sum()),'false_negative':int((fn & ~zone).sum())}
    result['reference']='Saved mask foreground boundary, 4-neighbor erosion with outside-image background; Chebyshev inclusive distance'
    return result


def samples(vertices,faces,views,expected=EXPECTED_BACK):
    centers,normal,areas=face_geometry(vertices,faces);fi=select_faces(centers,areas);vi=spaced(len(vertices),SETTINGS['vertex_limit'])
    rim=int(np.argmax(np.linalg.norm(vertices,axis=1)))
    if rim not in vi:vi[-1]=rim
    points=[*centers[fi],*vertices[vi]];rows=[];silhouettes=[]
    for camera,arrays in views:
        hit=arrays['hit'][:,0];ids=np.flatnonzero(hit);d=arrays['directions'][hit].astype(np.float64);n=arrays['normals'][hit].astype(np.float64)
        require(np.isfinite(n).all() and (np.linalg.norm(n,axis=1)>0).all(),'Invalid saved hit normals')
        cosine=-np.einsum('ij,ij->i',n/np.linalg.norm(n,axis=1)[:,None],d/np.linalg.norm(d,axis=1)[:,None])
        bad=ids[cosine<0];good=ids[cosine>0]
        require(len(bad)==expected[camera],'Back-facing count differs from accepted evidence; do not drop anomalies')
        controls=good[spaced(len(good),SETTINGS['controls_per_view'])] if len(good) else np.asarray([],np.int64)
        for index in [*bad,*controls]:
            index=int(index);point=arrays['positions'][index].astype(np.float64)
            require(np.isfinite(point).all(),'Nonfinite selected hit')
            row={'camera_id':camera,'ray_index':index,'pixel_row':index//arrays['mask'].shape[1],'pixel_column':index%arrays['mask'].shape[1],
                 'anomalous':bool(index in bad),'sample_index':len(points),'position':point.tolist(),
                 'origin':arrays['origins'][index].tolist(),'direction':arrays['directions'][index].tolist(),
                 'saved_depth':float(arrays['depth'][index,0]),'raw_normal':arrays['raw_normals'][index].tolist(),
                 'final_normal':arrays['normals'][index].tolist(),'viewing_cosine':float(cosine[np.searchsorted(ids,index)])}
            row['CPU']=ray_comparison(vertices,faces,row);rows.append(row);points.append(point)
        silhouettes.append({'camera_id':camera,**silhouette_disagreements(hit.reshape(arrays['mask'].shape),arrays['mask'])})
    nominal=np.asarray(points,np.float64);p=nominal.astype(np.float32)
    require(len(fi)<=2048 and len(vi)<=256 and len(rows)<=99 and len(p)*5+4*len(fi)+6*len(rows)<=SETTINGS['point_budget'],
            'SDF plan exceeds approved deterministic query budget')
    return {'points':p,'face_indices':fi,'vertex_indices':vi,'face_normals':normal[fi],'face_areas':areas[fi],
            'face_count':len(fi),'vertex_count':len(vi),'rays':rows,'silhouettes':silhouettes,
            'mesh_population':{'faces':len(faces),'vertices':len(vertices),'face_area_quantiles':quantiles(areas),
                'centroid_radius_quantiles':quantiles(np.linalg.norm(centers,axis=1)),
                'tiny_faces':int((areas<=SETTINGS['tiny_triangle_area']).sum())},
            'float32_point_rounding_maximum':float(np.abs(nominal-p).max()),
            'sample_kind':np.asarray([0]*len(fi)+[1]*len(vi)+[2]*len(rows),np.int32)}


class TorchQueries:
    """Only original sdf(x), first input derivatives, fixed float32 batches."""
    def __init__(self,sdf,torch,device='cuda'):
        self.sdf=sdf;self.torch=torch;self.device=device;self.points=0;self.calls=0;self.gradient_points=0
    def __call__(self,points,gradients=False):
        points=np.asarray(points,np.float32)
        require(points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all() and
                self.points+len(points)<=SETTINGS['point_budget'],'SDF point budget/finite input check failed')
        if not len(points):
            self.last_inside=np.empty(0,bool)
            return np.empty(0,np.float32),np.empty((0,3),np.float32) if gradients else None
        self.points+=len(points);values=[];derivatives=[];masks=[];t=self.torch
        for start in range(0,len(points),SETTINGS['batch_size']):
            x=t.tensor(points[start:start+SETTINGS['batch_size']],dtype=t.float32,device=self.device,requires_grad=gradients)
            self.calls+=1
            masks.append((t.norm(x.detach(),dim=-1)<1.).cpu().numpy())  # Exact original extraction mask precision/operator.
            with t.enable_grad() if gradients else t.no_grad():
                y=self.sdf(x);require(y.shape==(len(x),1) and y.dtype==t.float32 and bool(t.isfinite(y).all().item()),'Invalid original SDF result')
                if gradients:
                    g=t.autograd.grad(y,x,grad_outputs=t.ones_like(y),create_graph=False,retain_graph=False,only_inputs=True)[0]
                    require(not g.requires_grad and bool(t.isfinite(g).all().item()),'Invalid/second-order input gradients')
                    derivatives.append(g.detach().cpu().numpy());self.gradient_points+=len(x)
            values.append(y.detach().cpu().numpy()[:,0])
        self.last_inside=np.concatenate(masks)
        return np.concatenate(values),np.concatenate(derivatives) if gradients else None


def quantiles(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return dict(zip(['minimum','p05','median','p95','maximum'],map(float,np.quantile(values,[0,.05,.5,.95,1])))) if len(values) else None


def evaluate_samples(plan,query):
    """First-order measurements only; return bounded numerical arrays and JSON."""
    p=plan['points'];nf=plan['face_count'];nv=plan['vertex_count'];rows=plan['rays'];nh=len(rows)
    y,g=query(p,True);norm=np.linalg.norm(g.astype(np.float64),axis=1);valid=norm>SETTINGS['gradient_epsilon']
    radii=np.linalg.norm(p.astype(np.float64),axis=1)
    inside=query.last_inside.copy()
    unit=np.zeros_like(g,dtype=np.float64);unit[valid]=g[valid]/norm[valid,None]
    proxy=np.full(len(p),np.nan);proxy[valid]=np.abs(y[valid])/norm[valid]
    probe_points=[];base_ids=[];directions=[];offsets=[]
    def probes(ids,vectors,code,factors):
        for factor in factors:
            probe_points.extend((p[ids].astype(np.float64)+H*factor*vectors).astype(np.float32))
            base_ids.extend(ids);directions.extend([code]*len(ids));offsets.extend([factor]*len(ids))
    probes(np.flatnonzero(valid),unit[valid],0,[.25,-.25,.5,-.5])
    probes(np.arange(nf),plan['face_normals'],1,[.25,-.25,.5,-.5])
    hit_ids=np.arange(nf+nv,len(p))
    if nh:
        final=np.asarray([r['final_normal'] for r in rows],np.float64);final/=np.linalg.norm(final,axis=1)[:,None]
        view=-np.asarray([r['direction'] for r in rows],np.float64);view/=np.linalg.norm(view,axis=1)[:,None]
        probes(hit_ids,final,2,[.25,-.25,.5,-.5]);probes(hit_ids,view,3,[.5,-.5])
    qp=np.asarray(probe_points,np.float32).reshape(-1,3);require(len(p)+len(qp)<=SETTINGS['point_budget'],'SDF probe budget exceeded')
    values,_=query(qp,False);bi=np.asarray(base_ids,np.int32);codes=np.asarray(directions,np.int32);factors=np.asarray(offsets,np.float32)
    probe_inside=query.last_inside.copy()
    lookup={(int(i),int(c),float(f)):k for k,(i,c,f) in enumerate(zip(bi,codes,factors))}
    crossings={};crossed=np.zeros(len(p),bool);eligible=valid & inside
    for code,label in enumerate(['gradient','original_face','A_final','toward_camera']):
        scale_report={}
        for scale in ([.5] if code==3 else [.25,.5]):
            ids=[i for i in range(len(p)) if (i,code,scale) in lookup]
            count=0;excluded=0;positive=0;negative=0
            for i in ids:
                a,b=lookup[(i,code,scale)],lookup[(i,code,-scale)]
                if not inside[i] or not (probe_inside[a] and probe_inside[b]):
                    excluded+=1
                    if code==0 and scale==.5:eligible[i]=False
                    continue
                plus,minus=values[a],values[b]
                positive+=int(plus>SETTINGS['sign_epsilon']);negative+=int(plus<-SETTINGS['sign_epsilon'])
                changes=bool((plus>SETTINGS['sign_epsilon'] and minus<-SETTINGS['sign_epsilon']) or
                             (plus<-SETTINGS['sign_epsilon'] and minus>SETTINGS['sign_epsilon']))
                count+=int(changes)
                if code==0 and scale==.5:crossed[i]=changes
            scale_report[str(scale)]={'points':len(ids),'sphere_mask_excluded':excluded,'sign_changes':count,
                                      'positive_side_positive_sign':positive,'positive_side_negative_sign':negative}
        crossings[label]=scale_report
    groups={}
    for kind,label in enumerate(['face_centroids','vertices','saved_intersections']):
        selected=plan['sample_kind']==kind;ok=selected & valid & inside
        groups[label]={'count':int(selected.sum()),'raw_abs_SDF':quantiles(np.abs(y[selected])),
            'first_order_distance_proxy':quantiles(proxy[ok]),'proxy_over_grid_h':quantiles(proxy[ok]/H),
            'gradient_lengths':quantiles(norm[selected]),'near_zero_gradients':int((selected & ~valid).sum()),
            'outside_sphere':int((selected & ~inside).sum()),'near_unit_sphere_within_h':int((selected & (np.abs(radii-1)<=H)).sum()),
            'radius':quantiles(radii[selected])}
    face_cos=np.full(nf,np.nan);ok=valid[:nf] & inside[:nf]
    face_cos[ok]=np.einsum('ij,ij->i',unit[:nf][ok],plan['face_normals'][ok])
    finite=face_cos[np.isfinite(face_cos)];same=bool(len(finite) and (finite>=SETTINGS['direction_cosine']).all())
    opposite=bool(len(finite) and (finite<=-SETTINGS['direction_cosine']).all())
    A_cos=[];outward_controls=[];exterior_signs=[]
    for j,row in enumerate(rows):
        i=nf+nv+j;excluded=[];row['raw_SDF']=float(y[i]);row['gradient']=g[i].tolist();row['gradient_length']=float(norm[i])
        row['radius']=float(radii[i]);row['first_order_distance_proxy']=float(proxy[i]) if valid[i] else None
        if not valid[i]:excluded.append('near-zero gradient')
        if not inside[i]:excluded.append('base outside unit sphere')
        if valid[i]:
            n=np.asarray(row['final_normal']);n/=np.linalg.norm(n);cos=float(np.dot(unit[i],n));A_cos.append(cos)
            row['gradient_vs_A_final_signed_cosine']=cos
            row['gradient_vs_A_final_angle_degrees']=float(np.degrees(np.arccos(np.clip(cos,-1,1))))
            cpu=row['CPU']
            if cpu.get('closest_saved_depth'):
                raw_face=np.asarray(cpu['closest_saved_depth']['normal']);v=float(np.dot(unit[i],raw_face))
                row['gradient_vs_matched_original_face_signed_cosine']=v
                row['gradient_vs_negated_matched_face_signed_cosine']=-v
            a,b=lookup[(i,3,.5)],lookup[(i,3,-.5)]
            plus,minus=float(values[a]),float(values[b]);gap=cpu.get('next_distinct_gap')
            row['ray_side_SDF']={'toward_camera':plus,'away_from_camera':minus,'offset_scene_units':H/2,
                                 'both_inside_sphere':bool(probe_inside[a] and probe_inside[b])}
            if not cpu.get('geometric_exterior_eligible'):excluded.append('CPU first-hit/edge/origin gate not established')
            if gap is not None and gap<=H/2+SETTINGS['CPU_depth_absolute']:excluded.append('offset may cross another triangle intersection')
            if not (probe_inside[a] and probe_inside[b]):excluded.append('ray-side probe affected by outside-sphere masking')
            if not (plus*minus<0 and abs(plus)>SETTINGS['sign_epsilon'] and abs(minus)>SETTINGS['sign_epsilon']):
                excluded.append('no robust ray-side sign change')
            if not excluded:
                sign=1 if plus>0 else -1
                row['measured_camera_exterior_SDF_sign']=sign;exterior_signs.append(sign)
                row['A_alignment_with_exterior_signed_gradient']=sign*cos
                if not row['anomalous']:outward_controls.append(sign*cos)
        row['exterior_interpretation_exclusions']=excluded
        row['causal_attribution']='unresolved; classifications concern observations, not an invented failure cause'
    zero_pass=bool(eligible.any() and (proxy[eligible]<=H/2).all() and crossed[eligible].all())
    anomalies=[r for r in rows if r['anomalous']]
    statuses={
        'sdf_mesh_zero_level_consistency':{'status':'pass' if zero_pass else ('fail' if eligible.any() else 'unresolved'),
            'scope':'Eligible sampled points only: proxy<=h/2 and a +/-h/2 gradient sign crossing; not surface accuracy',
            'eligible':int(eligible.sum()),'excluded':int((~eligible).sum()),'failures':int((eligible & ((proxy>H/2)|~crossed)).sum())},
        'sdf_gradient_quality':{'status':'pass' if valid.all() else 'unresolved','near_zero_excluded':int((~valid).sum()),
            'length_quantiles':quantiles(norm),'unit_magnitude_assumed':False},
        'face_winding_vs_sdf_orientation':{'status':'pass' if same or opposite else 'unresolved',
            'scope':'Signed relation at eligible interior centroid samples only, not correct global outward orientation',
            'observed_relation':'same signed gradient' if same else ('opposite signed gradient' if opposite else 'mixed/insufficient'),
            'signed_cosine':quantiles(face_cos),'negated_face_signed_cosine':quantiles(-face_cos),
            'angle_degrees':quantiles(np.degrees(np.arccos(np.clip(face_cos,-1,1)))),
            'negated_face_angle_degrees':quantiles(np.degrees(np.arccos(np.clip(-face_cos,-1,1)))),
            'positive_cosines':int((finite>0).sum()),'negative_cosines':int((finite<0).sum()),'excluded':int((~ok).sum()),
            'global_outward_orientation_verified':False},
        'A_shading_normal_consistency':{'status':('pass' if min(outward_controls)>=SETTINGS['direction_cosine'] else 'fail') if outward_controls else 'unresolved',
            'scope':'Only eligible non-anomalous controls with geometric first-hit and measured ray-side sign; no normal is changed',
            'gradient_signed_cosine':quantiles(A_cos),'eligible_controls':len(outward_controls),
            'gradient_angle_degrees':quantiles(np.degrees(np.arccos(np.clip(A_cos,-1,1)))),
            'gradient_positive_cosines':sum(v>0 for v in A_cos),'gradient_negative_cosines':sum(v<0 for v in A_cos),
            'directional_exclusions':nh-len(A_cos),'exterior_control_exclusions':sum(not r['anomalous'] for r in rows)-len(outward_controls),
            'exterior_signed_alignment':quantiles(outward_controls),'measured_exterior_positive_sign_count':exterior_signs.count(1),
            'measured_exterior_negative_sign_count':exterior_signs.count(-1)},
        'three_anomalous_hits':{'status':'pass' if anomalies and all(r['CPU']['classification']=='supported_by_evidence' for r in anomalies) else 'unresolved',
            'classification_counts':{name:sum(r['CPU']['classification']==name for r in anomalies) for name in ['supported_by_evidence','ambiguous','unresolved']},
            'scope':'CPU geometric observations plus explicit SDF comparisons; no causal/global orientation claim'},
        'silhouette_disagreement_characterization':{'status':'pass','views':plan['silhouettes'],'quality_admission':False},
        'historical_checkpoint_provenance':{'status':'unresolved','reason':'Producing revision/configuration/split remain unattributed'},
        'research_geometry_admission':{'status':'unresolved','reason':'Historical r128 diagnostic; no attributable fresh G/canonical512 handoff'}}
    for check in statuses.values():
        if check['status']=='unresolved' and 'reason' not in check:
            check['reason']='Insufficient eligible or unambiguous measurements for the frozen criterion; inspect exclusions and per-ray records'
    numeric={'points':p,'SDF_raw':y,'base_masked_extraction_field':np.where(inside,y,1.),'gradient':g,'gradient_length':norm,'distance_proxy':proxy,'inside_sphere':inside,
        'float64_radius':radii,
        'sample_kind':plan['sample_kind'],'probe_points':qp,'probe_SDF_raw':values,'probe_masked_extraction_field':np.where(probe_inside,values,1.),
        'probe_inside_sphere':probe_inside,'probe_base_index':bi,'probe_direction_code':codes,'probe_offset_over_h':factors,
        'face_indices':plan['face_indices'],'vertex_indices':plan['vertex_indices'],'face_areas':plan['face_areas'],
        'face_cross_normals':plan['face_normals'],'gradient_face_cosine':face_cos}
    return {'statuses':statuses,'groups':groups,'sign_probes':crossings,'face_area_quantiles':quantiles(plan['face_areas']),
            'tiny_selected_faces':int((plan['face_areas']<=SETTINGS['tiny_triangle_area']).sum()),
            'query_point_rounding_maximum':plan['float32_point_rounding_maximum'],
            'mask_precision':'Original torch.norm(float32 point) >= 1, not a float64 sphere substitution',
            'rays':rows,'point_evaluations':len(p)+len(qp),'gradient_points':len(p)},numeric


def implementation():
    names=['infrastructure/geometry_sdf_mesh_audit.py','run_geometry_sdf_mesh_audit.py',DOC,
           'infrastructure/geometry_scientific_audit.py','infrastructure/g_checkpoint.py','infrastructure/stage_sources.py']
    return {n:digest(ROOT/n) for n in names}


def prepare(inputs,sha,output,g_commit=PINS['geometry'],a_commit=PINS['material']):
    require(g_commit==PINS['geometry'] and a_commit==PINS['material'],'Exact frozen G/A commits required')
    path,_=evidence.reference({'path':str(inputs),'sha256':sha});selection,selection_id=read_report(path,sha)
    require(set(selection)=={'schema_version','protocol','artifacts','views'} and type(selection['schema_version']) is int and
            selection['schema_version']==1 and selection['protocol']==PROTOCOL,'Wrong SDF audit selection version/schema')
    refs=selection['artifacts']
    require(set(refs)=={'source_manifest','extraction','mesh','rays','checkpoint','coordinate_audit'},'Explicit complete artifact set required')
    require(type(selection['views']) is list and all(type(r) is dict and set(r)=={'role','source_camera_id','path','sha256'}
            for r in selection['views']),'Explicit complete view bindings required; unused fields refused')
    for ref in [*refs.values(),*[{k:r[k] for k in ['path','sha256']} for r in selection['views']]]:evidence.reference(ref)
    base_selection={'schema_version':1,'protocol':evidence.PROTOCOL,
                    'artifacts':{k:refs[k] for k in ['source_manifest','extraction','mesh','rays']},'views':selection['views']}
    require([(v['role'],v['source_camera_id']) for v in selection['views']]==list(evidence.VIEWS),'Explicit selected camera IDs/order required')
    extraction,rays,sources=evidence.admit(base_selection)  # Identity only; NEVER the 3A-1 numerical audit.
    require(refs['checkpoint']['sha256']==evidence.CHECKPOINT_SHA,'Explicit admitted historical best checkpoint required')
    require(refs['coordinate_audit']['sha256']==COORDINATE_AUDIT_SHA,'Explicit accepted 3A-1 report SHA-256 required')
    for ref in [refs['checkpoint'],*[{k:r[k] for k in ['path','sha256']} for r in selection['views']]]:
        with evidence.verified(ref):pass
    prior,_=read_report(*evidence.reference(refs['coordinate_audit']))
    require(prior.get('protocol')==evidence.PROTOCOL and prior.get('schema_version')==1 and prior.get('status')=='pass' and
            prior.get('blockers')==[] and prior.get('research_handoff_admitted') is False and
            prior.get('tolerances')==evidence.TOLERANCES and len(prior.get('views',[]))==3 and
            prior.get('categories',{}).get('saved_evidence_coordinate_consistency',{}).get('status')=='pass',
            'Accepted completed 3A-1 coordinate report required')
    # Paths in the old report belong to its original host/worktree. Bind bytes
    # and camera IDs, never follow those old paths inside the isolated container.
    old=prior['artifacts']
    require(len(old['views'])==3 and all(old['artifacts'][k]['sha256']==refs[k]['sha256'] for k in base_selection['artifacts']) and
            all((v['role'],v['source_camera_id'],v['sha256'])==(n['role'],n['source_camera_id'],n['sha256'])
                for v,n in zip(old['views'],selection['views'])),'3A-1 camera/mesh/NPZ identities differ')
    for row,new in zip(prior['views'],selection['views']):
        require((row['role'],row['source_camera_id'])==(new['role'],new['source_camera_id']) and
                all(c.get('status')=='pass' for c in row['checks'].values()) and
                row['checks']['final_normal_facing']['measured']['back_facing_count']==EXPECTED_BACK[row['source_camera_id']],
                '3A-1 anomaly/check identities incomplete or changed')
    require(set(prior['implementation'])=={'infrastructure/geometry_scientific_audit.py','audit_geometry_scientific.py',
        'infrastructure/signatures.py','infrastructure/stage_sources.py','infrastructure/paths.py','docs/GEOMETRY_SCIENTIFIC_AUDIT_V1.md'},
        'Incomplete 3A-1 implementation identity')
    require(all(digest(ROOT/n)==h for n,h in prior['implementation'].items()),'Accepted 3A-1 implementation bytes changed')
    for stage,item in sources.items():
        marker='/'+item['root']+'/'
        recorded=prior.get('source_files',{}).get(stage,{})
        require(recorded and all(marker in n for n in recorded) and
                {n.split(marker,1)[1]:h for n,h in recorded.items()}=={n:v['sha256'] for n,v in item['files'].items()},
                'Incomplete/different accepted 3A-1 scientific source closure')
    roots={k:Path(v['path']).parent for k,v in refs.items()};roots['selection']=path.parent
    roots.update({'view'+str(i):Path(v['path']).parent for i,v in enumerate(selection['views'])})
    # Existing storage adapter requires these labels. "datasets" is immutable
    # CODE here, not a dataset selection; no dataset API/root is ever loaded.
    resolver=PathResolver({'datasets':ROOT,'thesis':roots['checkpoint'],**roots},output)
    require(resolver.output_root!=ROOT and ROOT not in resolver.output_root.parents and resolver.output_root not in ROOT.parents and
            all(p not in resolver.output_root.parents and p!=resolver.output_root for p in roots.values()),'Output overlaps source/evidence roots')
    import yaml
    science=ROOT/sources['geometry']['root'];cfg=yaml.load((science/'configs/shape/syn/bell.yaml').read_text(),Loader=yaml.FullLoader)
    return {'selection':selection,'selection_id':selection_id,'extraction':extraction,'rays':rays,'prior':prior,
            'sources':sources,'science':science,'cfg':cfg,'resolver':resolver}


class NativeCalls:
    def __init__(self,renderer,sdf_class,torch):
        from network.field import extract_geometry,extract_fields
        from dataset.database import GlossySyntheticDatabase
        from utils.base_utils import read_pickle,save_pickle
        self.forbidden={inspect.unwrap(f).__code__ for f in [renderer.forward,renderer._init_dataset,renderer.train_step,
            renderer.test_step,renderer.nvs,sdf_class.gradient,torch.optim.Adam.__init__,torch.optim.Adam.step,
            torch.optim.Optimizer.load_state_dict,torch.Tensor.backward,torch.save,extract_geometry,extract_fields,
            GlossySyntheticDatabase.__init__,read_pickle,save_pickle]}
        self.sdf=sdf_class.sdf.__code__;self.grad=torch.autograd.grad.__code__;self.phase='construction'
        self.points=0;self.calls=0;self.grad_calls=0
    def observe(self,frame,event,value):
        if event!='call':return
        require(frame.f_code not in self.forbidden,'Forbidden training/rendering/data/serialization/second-order helper reached')
        if frame.f_code is self.sdf:
            x=frame.f_locals['x'];require(self.phase=='measurement' and x.dtype.__str__()=='torch.float32' and x.is_cuda and len(x)<=128,'Unapproved SDF call')
            self.points+=len(x);self.calls+=1;require(self.points<=20000,'Observed SDF query budget exceeded')
        elif frame.f_code is self.grad:
            require(self.phase=='measurement' and frame.f_locals['create_graph'] is False,'Second derivatives/unapproved autograd query refused')
            self.grad_calls+=1
    def __enter__(self):
        require(sys.implementation.name=='cpython' and sys.getprofile() is None and sys.gettrace() is None,'Unobserved CPython required')
        sys.setprofile(self.observe);return self
    def __exit__(self,*args):sys.setprofile(None)


def native(p,run,report,persist):
    import torch
    require(sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and
            torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe',
            'Reviewed Python/Torch/CUDA/one A100 runtime required')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu','Original CPU float32 constructor context required')
    from network.renderer import NeROShapeRenderer
    from network.field import SDFNetwork
    from infrastructure.geometry_training_smoke import runtime_identity
    from infrastructure.bell_inference import rng_identity
    report['runtime']=runtime_identity(torch);network=None;guard=NativeCalls(NeROShapeRenderer,SDFNetwork,torch)
    try:
        with guard:
            report['phase']='original_G_constructor_and_strict_load';persist()
            network=NeROShapeRenderer(deepcopy(p['cfg']),training=False)
            ref=p['selection']['artifacts']['checkpoint']
            state=load_checked(Path(ref['path']),ref['sha256'],torch,trusted=True)
            report['checkpoint_metadata']=historical_schema(state)
            comparison=network_comparison(network,state['network_state_dict'],torch)
            require(comparison==p['extraction']['network_comparison'],'Network catalog differs from accepted G extraction')
            strict_network_load(network,state['network_state_dict'],torch);del state
            network.eval().cuda();before=identities(network.state_dict())
            require(before==p['extraction']['network_state_before'] and before==p['extraction']['network_state_after'],
                    'Loaded weights/buffers differ from admitted extraction identities')
            report['network_state_before']=before;report['strict_network_loaded']=True
            report['original_configuration']=p['cfg'];report['effective_renderer']=network.cfg
            guard.phase='measurement';query=TorchQueries(network.sdf_network.sdf,torch)
            report['RNG_before_measurements']=rng_identity(torch,np)
            report['phase']='bounded_original_sdf_and_first_derivatives';persist()
            result,numeric=evaluate_samples(p['sample_plan'],query)
            require(query.points==guard.points==result['point_evaluations'] and query.calls==guard.calls and
                    guard.grad_calls==(query.gradient_points+127)//128,'Unaccounted original SDF/gradient calls')
            require(identities(network.state_dict())==before and all(v.grad is None for v in network.parameters()),
                    'SDF audit mutated model weights/buffers or accumulated parameter gradients')
            report['RNG_after_measurements']=rng_identity(torch,np)
            require(report['RNG_before_measurements']==report['RNG_after_measurements'],'SDF measurements consumed unexpected RNG state')
            report.update(result=result,network_state_after=identities(network.state_dict()),network_unchanged=True,
                          query_accounting={'points':query.points,'calls':query.calls,'first_derivative_points':query.gradient_points,
                                            'autograd_calls':guard.grad_calls,'second_derivatives':0})
            with run.output('measurements.npz').open('xb') as stream:np.savez_compressed(stream,**numeric)
            report['numeric_output']={'path':'measurements.npz','sha256':digest(run.output('measurements.npz'))}
    finally:
        report['observed_calls']={'points':guard.points,'calls':guard.calls,'autograd_calls':guard.grad_calls}
        del network;torch.set_default_tensor_type('torch.FloatTensor');torch.cuda.empty_cache()
        report['cleanup']={'profile_restored':sys.getprofile() is None,'trace_restored':sys.gettrace() is None,'default_tensor_type':'torch.FloatTensor'}


def execute(p,run_id,image_id,infrastructure_commit,trusted=False):
    validate_run_id(run_id);require(trusted is True,'Explicit trusted historical checkpoint pickle acknowledgement required')
    runtime_guards(p,image_id)
    for ref in [*p['selection']['artifacts'].values(),*p['selection']['views'],p['selection_id']]:
        require(os.statvfs(ref['path']).f_flag & os.ST_RDONLY,'Every selected input file must be mounted read-only')
    activate('geometry',PINS['geometry'])
    run=RunStore(p['resolver'],layout='mounted').create(run_id)
    code=implementation()
    report={'schema_version':1,'protocol':PROTOCOL,'status':'running','phase':'allocated','run_id':run_id,'reviewed_base_commit':BASE,
            'scientific_sources':p['sources'],'artifacts':p['selection'],'implementation':code,
            'infrastructure_commit_declared':infrastructure_commit,'image_id_declared':image_id,'settings':{**SETTINGS,'grid_h':H},
            'research_handoff_admitted':False,'historical_checkpoint_provenance':'unresolved','optimizer_updates':0,
            'optimizer_state_restored':False,'dataset_loaded':False,'rendering_performed':False,
            'statuses':{name:{'status':'unresolved','reason':'Action not completed'} for name in [
                'evidence_identity','sdf_mesh_zero_level_consistency','sdf_gradient_quality','face_winding_vs_sdf_orientation',
                'A_shading_normal_consistency','three_anomalous_hits','silhouette_disagreement_characterization',
                'historical_checkpoint_provenance','research_geometry_admission']},
            'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with run.output('sdf-mesh-audit.json').open('x') as stream:
        def persist():write_report(stream,report)
        persist()
        try:
            report['phase']='bounded_mesh_ray_and_silhouette_measurements';persist()
            vertices,faces=mesh_arrays(p['selection']['artifacts']['mesh'])
            arrays=[(row['source_camera_id'],evidence.load_arrays({k:row[k] for k in ['path','sha256']})) for row in p['selection']['views']]
            p['sample_plan']=samples(vertices,faces,arrays);del arrays,vertices,faces
            report['mesh_population']=p['sample_plan']['mesh_population']
            report['selection']={'face_indices':p['sample_plan']['face_indices'].tolist(),'vertex_indices':p['sample_plan']['vertex_indices'].tolist(),
                'ray_indices':[{k:r[k] for k in ['camera_id','ray_index','anomalous','sample_index']} for r in p['sample_plan']['rays']]}
            workspace=run.output('cache/sdf-audit-workspace');workspace.mkdir()
            for name in ['assets','configs']:(workspace/name).symlink_to(p['science']/name,target_is_directory=True)
            from infrastructure.bell_inference import runtime_environment
            with runtime_environment(run):
                os.chdir(workspace)
                try:native(p,run,report,persist)
                except BaseException:traceback.print_exc();raise
            require(report.get('network_unchanged') is True and report['cleanup']['profile_restored'] and report['cleanup']['trace_restored'],
                    'Incomplete immutable-model/cleanup acceptance')
            report['statuses'].update(report['result'].pop('statuses'))
            require(code==implementation(),'Audit implementation changed')
            for ref in [*p['selection']['artifacts'].values(),*[{k:r[k] for k in ['path','sha256']} for r in p['selection']['views']],
                        {'path':p['selection_id']['path'],'sha256':p['selection_id']['sha256']}]:
                with evidence.verified(ref):pass
            require(all(not list(run.output(folder).iterdir()) for folder in ['checkpoints','meshes','renders']),
                    'Unexpected checkpoint/mesh writer')
            report['statuses']['evidence_identity']={'status':'pass','scope':'Explicit hashes, accepted 3A-1 binding, strict full weights and post-run immutability'}
            report['imported_scientific_modules']=imported_sources(p['science'])
            report['output_hashes']={q.relative_to(run.root).as_posix():digest(q) for q in run.root.rglob('*') if q.is_file() and q.resolve()==q and
                q.name not in ['sdf-mesh-audit.json','.experiment.json'] and not any(v in ('torch_extensions','home','tmp') for v in q.relative_to(run.root).parts)}
            report.update(status='complete',phase='complete')
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc));persist();raise
        finally:
            report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();persist()
    return report
