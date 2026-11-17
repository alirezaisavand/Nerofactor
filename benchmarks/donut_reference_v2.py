"""Versioned continuation metadata and admission of immutable camera0 evidence."""
from pathlib import Path
import json
import numpy as np
from benchmarks.donut_recovery_v1 import REFERENCE_LIMITS, SIZE, PANEL, numeric
from benchmarks.beckmann_reflectance_v2 import ORDERS,NODES_PER_POINT
from infrastructure.stage_sources import ROOT,digest
from infrastructure.signatures import require

PROTOCOL='donut-kurt-reference-v2'
PREVIOUS='docs/evidence/donut_reference_reporting_fix_v1/offline-analysis.json'
GRAZING_PIXELS=[12635,4037,12581,4825,4027]
ORDINARY_COUNT=8
ORDINARY_COSINE_MIN=.25
LIMITS=dict(REFERENCE_LIMITS,primary_rays=5*SIZE**2,
    shadow_directions=5*SIZE**2*(256**2+512**2),
    BRDF_point_directions=5*SIZE**2*(256**2+512**2),CPU_primary_crosschecks=40,
    slope_nodes_attempted=(4536+5*SIZE**2)*NODES_PER_POINT,
    slope_nodes_returned=(4536+5*SIZE**2)*NODES_PER_POINT)


def previous(root):
    require(root.is_absolute() and root.resolve()==root,'Explicit unaliased previous reference root required')
    evidence=json.loads((ROOT/PREVIOUS).read_text())
    for name,item in evidence['original_artifacts'].items():
        p=root/name
        require(root in p.resolve().parents and p.resolve()==p and p.is_file() and
            p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'Original failed evidence changed: '+name)
    data=numeric(root/'reference/view-0.npz')
    require(data['hit_mask'].shape==(SIZE,SIZE) and data['hit_mask'].dtype==bool and
        data['shader_valid'].dtype==bool and len(data['pixel_indices'])==4539 and
        int(data['shader_valid'].sum())==4536 and np.array_equal(np.flatnonzero(data['hit_mask'].reshape(-1)),data['pixel_indices']),
        'Camera0 geometry/observation identity differs')
    v=data['shader_valid'];fail=v&(abs(data['reflectance']-data['coarse_reflectance']).max(-1)>.003)
    require(set(data['pixel_indices'][fail])==set(GRAZING_PIXELS),'Saved grazing failure identities changed')
    require(np.isfinite(data['linear_components']).all() and
        np.array_equal(data['linear_components'][:,0]+data['linear_components'][:,1],data['linear_components'][:,2]),
        'Invalid preserved camera0 image components')
    return data,evidence


def validation_indices(data):
    pixels=data['pixel_indices'];v=data['shader_valid']
    grazing=np.array([np.flatnonzero(pixels==i)[0] for i in GRAZING_PIXELS])
    cosine=(data['outgoing']*data['normals']).sum(-1)
    domain=np.flatnonzero(v&(cosine>=ORDINARY_COSINE_MIN))
    require(len(domain)>=ORDINARY_COUNT,'Insufficient original ordinary-angle observations; no replacements')
    ordinary=domain[np.linspace(0,len(domain)-1,ORDINARY_COUNT).astype(int)]
    require(not np.intersect1d(grazing,ordinary).size,'Validation sets must be disjoint')
    return grazing,ordinary
