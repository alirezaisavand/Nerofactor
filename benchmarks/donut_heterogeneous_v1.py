"""Geometry-defined heterogeneous Kurt oracle and admitted Donut correspondences."""
import json
from pathlib import Path

import numpy as np

from benchmarks.bell_e11 import TOL
from benchmarks.beckmann_reflectance_v2 import NODES_PER_POINT, ORDERS
from benchmarks.donut_recovery_v1 import (ENVIRONMENT, GRIDS, IMAGE, PANEL, REFERENCE_LIMITS,
                                           SIZE, SPLITS, camera_rays, inputs, unit)
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest
from infrastructure.verified_ray_backend import identity as backend_identity

PROTOCOL = 'donut-kurt-heterogeneous-reference-v1'
FREEZE = 'docs/protocols/DONUT_HETEROGENEOUS_REFERENCE_V1.freeze.json'
SOURCE_REPORT = '2052306bdf87a0beaa0a3f9b0bddeeb12d787f1d65357d520b5ce2fe0a33ee64'
SOURCE_ADMISSION = '02e4f7ca909de6b7c2c9fa2a2e291850a9bff190b426e319daf82e3ecdb82c9e'
COUNTS = [4536, 2636, 4144, 3970, 4849, 2495]
GEOMETRY_KEYS = ('pixel_indices','hit_mask','shader_valid','position','outgoing','raw_BVH_normals',
                 'depth','normals','triangle','barycentric','face_normal','surface_residual')
BRDF = 'Kurt-a0-metallic-additive-v1'
POINTS = sum(COUNTS)
DIRECTIONS = POINTS*sum(g*g for g in GRIDS)
LIMITS = dict(REFERENCE_LIMITS, primary_rays=0, CPU_primary_crosschecks=0,
              shadow_directions=DIRECTIONS, BRDF_point_directions=DIRECTIONS,
              slope_nodes_attempted=POINTS*NODES_PER_POINT, slope_nodes_returned=POINTS*NODES_PER_POINT)
FIELD = dict(center=[0.,0.,0.], torus_axis=[0.,0.,1.], boundary_band=.035,
             orientation_gap=.1, width_families=[[.42,.17],[.30,.112]],
             weak_widths=[.223,.214], width_modulation=[.035,.016],
             weak_lobe_azimuth=0., weak_lobe_meridian=.5)
TOLERANCES = {key:TOL[key] for key in ('convergence_rgb_abs','convergence_relative_l2')}


def read(path):
    return json.loads(path.read_text())


def check_artifact(path, item):
    require(path.is_file() and path.resolve()==path and path.stat().st_size==item['bytes']
            and digest(path)==item['sha256'], 'Frozen artifact differs: '+str(path))
    return path


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha, 'Heterogeneous freeze identity differs')
    f = read(path)
    require(f['protocol']==PROTOCOL and f['field']==FIELD and f['limits']==LIMITS
            and f['image']==IMAGE and f['panel']==PANEL and f['splits']==SPLITS
            and f['grids']==list(GRIDS) and f['slope_orders']==list(ORDERS)
            and f['environment']==ENVIRONMENT and f['tolerances']==TOLERANCES,
            'Heterogeneous scientific definition differs')
    for name, hash_ in f['sources'].items():
        p = ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==hash_, 'Frozen source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(approved=True, protocol=PROTOCOL, commit=commit, freeze_sha256=sha,
                image=IMAGE, selected_gpu='0', limits=LIMITS, maximum_execution_launches=1,
                host_output=f['host_output'], source_generation_sha256=SOURCE_REPORT,
                source_admission_sha256=SOURCE_ADMISSION, training_authorized=False)


def smoothstep(value):
    x = np.clip(value, 0, 1)
    return x*x*(3-2*x)


def material_field(points, normals, ring_radius):
    """Azimuth-sector steps, continuous scalar colors, and a smooth weak patch."""
    require(points.shape==normals.shape and points.ndim==2 and points.shape[1]==3
            and np.isfinite(points).all() and np.isfinite(normals).all()
            and np.max(abs(np.linalg.norm(normals,axis=1)-1))<2e-12
            and np.isfinite(ring_radius) and ring_radius>0, 'Invalid material field coordinates')
    x,y,z = (points - FIELD['center']).T
    radial = np.hypot(x,y)
    require((radial>0).all(), 'Undefined azimuth')
    u,v = np.arctan2(y,x), np.arctan2(z,radial-ring_radius)
    sector = np.cos(2*u)
    region = (sector<0).astype(np.uint8)
    # All four interfaces share position-dependent albedo/metallicity and a
    # continuous director. Only the two Beckmann widths step.
    angle = .40*np.sin(u+v)+.28*np.cos(2*u-v)
    azimuthal = np.stack((-np.sin(u),np.cos(u),np.zeros_like(u)),axis=1)
    tangent = unit(azimuthal-(azimuthal*normals).sum(1)[:,None]*normals)
    axis = unit(np.cos(angle)[:,None]*tangent + np.sin(angle)[:,None]*np.cross(normals,tangent))
    other = unit(np.cross(normals,axis))
    widths = np.asarray(FIELD['width_families'])[region] + np.asarray(FIELD['width_modulation'])*np.stack(
        (np.sin(u+v), np.cos(2*u-v)),axis=1)
    weak = (smoothstep((np.cos(u-FIELD['weak_lobe_azimuth'])-.90)/.075)
            *smoothstep((np.cos(v-FIELD['weak_lobe_meridian'])-.35)/.45))
    widths = (1-weak[:,None])*widths+weak[:,None]*FIELD['weak_widths']
    color = np.stack((.20+.05*np.sin(u-v), .58+.10*np.cos(v), .13+.04*np.cos(u+v)),axis=1)
    metallic = (.68+.08*np.sin(u-v))[:,None]
    major,minor = widths.T**2
    gap = (major-minor)/(major+minor)
    distance = .5*radial*abs(np.arcsin(np.clip(sector,-1,1)))
    Q = major[:,None,None]*axis[:,:,None]*axis[:,None,:]+minor[:,None,None]*other[:,:,None]*other[:,None,:]
    require(((widths>=.04)&(widths<=1)).all() and (major>minor).all()
            and ((color>=0)&(color<=1)).all() and ((metallic>=0)&(metallic<=1)).all(),
            'Material outside supported Kurt domain')
    return dict(axis=axis,Q_world=Q,color=color,metallic=metallic,widths_major_minor=widths,
                anisotropy_gap=gap,material_region=region,boundary_distance_native=distance,
                near_boundary=distance<=FIELD['boundary_band'],nearly_isotropic=gap<FIELD['orientation_gap'],
                weak_patch_weight=weak)


def geometry_fields(geometry):
    xyz = geometry['vertices_native']
    radial = np.linalg.norm(xyz[:,:2],axis=1)
    ring_radius = float((radial.min()+radial.max())/2)
    faces = xyz[geometry['triangles']]
    area = np.linalg.norm(np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]),axis=1)/2
    require((area>0).all(), 'Positive oracle triangle areas required')
    return ring_radius, area


def source(f, geometry_root, cameras, reference, admission):
    geometry,meta,identity = inputs(geometry_root,cameras)
    require(all(identity[k]['sha256']==f['geometry_inputs'][k]['sha256'] for k in identity),
            'Source geometry does not match freeze')
    require(digest(admission)==SOURCE_ADMISSION and digest(reference/'generation.json')==SOURCE_REPORT,
            'Exact admitted ordinary Donut evidence required')
    authorization_ = read(admission)
    g = read(reference/'generation.json')
    require(authorization_['researcher_admitted_dataset'] is True
            and authorization_['generation']['sha256']==SOURCE_REPORT
            and g['status']=='complete_pending_researcher_admission' and g['scientific_work_complete']
            and g['cleanup_verified'] and g['backend']==backend_identity() and g['environment']==ENVIRONMENT
            and g['panel']==PANEL and g['splits']==SPLITS and g['BRDF']==BRDF
            and g['indirect']=='zero' and g['point_light']=='absent', 'Admitted source reference differs')
    views = {}
    for camera,count in zip(PANEL,COUNTS):
        name = f'view-{camera}.npz'
        require(f['geometry_view_records'][name]==g['outputs'][name], 'Frozen geometry view identity differs')
        path = check_artifact(reference/name,g['outputs'][name])
        # Load geometry only. Old rendered RGB and material arrays are outside this data path.
        with np.load(path,allow_pickle=False) as rows:
            view = {name:rows[name] for name in GEOMETRY_KEYS}
        ids,valid,n,wo = (view[k] for k in ('pixel_indices','shader_valid','normals','outgoing'))
        origins,directions,_ = camera_rays(meta,camera)
        require(view['hit_mask'].shape==(SIZE,SIZE) and view['hit_mask'].dtype==bool
                and np.array_equal(ids,np.flatnonzero(view['hit_mask'])) and valid.dtype==bool
                and int(valid.sum())==count and np.array_equal(valid,(n*wo).sum(-1)>0)
                and all(np.isfinite(x).all() for x in view.values())
                and np.max(abs(wo+directions[ids]))<=2e-12
                and np.max(abs(view['position']-(origins[ids]-view['depth'][:,None]*wo)))<=2e-5
                and view['surface_residual'].max()<=2e-6
                and ((view['face_normal']*n).sum(-1)>0).all(),
                'Archived primary geometry convention differs')
        views[camera] = view
    return geometry, views


def labels(geometry, views):
    radius,area = geometry_fields(geometry)
    fields = {c:dict(v,**material_field(v['position'],v['normals'],radius)) for c,v in views.items()}
    return fields,radius,area


def seen_area_weights(triangles, areas):
    """Split each observed triangle's area among its observations in the chosen split."""
    require(triangles.ndim==areas.ndim==1 and triangles.dtype.kind in 'iu'
            and len(triangles)>0 and triangles.min()>=0 and triangles.max()<len(areas)
            and np.isfinite(areas).all() and (areas>0).all(), 'Invalid observed triangle inventory')
    n = np.bincount(triangles,minlength=len(areas))
    weights = areas[triangles]/n[triangles]
    return weights, float(areas[n>0].sum()/areas.sum())
