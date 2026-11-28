"""Freeze common saved observations; no Torch, checkpoint or ray queries."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks import bell_material_geometry_gate13 as m

OUTPUT=Path('/home/isavand/paper/evidence/bell-material-gate13-f1a471f430a443f4927cbdc5fa2dd430')


def inventory():
    g=m.g;f=g.freeze('f20bab2ebcddc48bbd8d22dc384c06d0ee3f6231d181b09560d32c3968fe5e9a')
    roots=g.paths();audit=roots['O1'].parent/'evaluation.json';old=g.read(audit)
    gate12=Path(f['host_output']);native=g.read(gate12/'render/receipt.json')
    require=m.require
    require(native['status']=='complete' and native['cleanup_verified'] and not native['refined_pixels_total'] and
        all(x['passed'] for v in native['views'].values() for a in v['convergence'].values() for x in a),
        'Accepted Gate 12 completion required')
    require(m.digest(m.ROOT/m.K1_FREEZE)==m.K1_SHA,'Accepted K1 recipe differs')
    base=m.read(m.ROOT/m.K1_FREEZE)['arms']['K1-Dense']
    p=dict(oracle=roots['oracle'],O1=roots['O1'],gate12=gate12/'render',initial=Path(base['initial']['path']),
           admission=g.prior.original.ADMISSION,verification=g.prior.original.ARCHIVE/'offline-verification-v1.json')
    files={k:{} for k in p};gen=m.read(p['oracle']/'geometry/generation.json')
    catalog={x['path']:{k:x[k] for k in ('sha256','bytes')} for x in gen['outputs']}
    names=['geometry/mesh.ply','geometry/calibrations-safe.npz']
    for c in m.panel():names.extend([f'geometry/rays/{c}.npz',f'geometry/labels/{c}.npz',f'references/{c}.npz'])
    for name in names:g.check(p['oracle']/name,catalog[name]);files['oracle'][name]=catalog[name]
    files['oracle']['geometry/generation.json']=g.item(p['oracle']/'geometry/generation.json')
    for name in ['candidate.ply']+[('fitting/' if str(c) in m.split()['train'] else 'rasters/')+f'{c}.npz' for c in m.panel()]:
        expected=old['artifacts']['300000/'+name];g.check(p['O1']/name,expected);files['O1'][name]=expected
    for camera in g.PANEL:
        for arm in ('oracle','O1'):
            for n in g.BASE:
                name=f'{camera}-{arm}-base-{n}.npz';g.check(p['gate12']/name,native['outputs'][name]);files['gate12'][name]=native['outputs'][name]
    files['gate12']['receipt.json']=g.item(p['gate12']/'receipt.json')
    for key in ('initial','admission','verification'):files[key]['']=g.item(p[key])
    g.check(p['initial'],{k:base['initial'][k] for k in ('sha256','bytes')})
    require(g.digest(p['admission'])==m.ADMISSION and g.digest(p['verification'])==g.prior.original.VERIFY_SHA,
            'Original dataset admission differs')
    vertices,_=g.mesh_arrays(p['oracle']/'geometry/mesh.ply');bounds=np.stack([vertices.min(0),vertices.max(0)]).astype(float)
    data={};coverage=[]
    for c in m.panel():
        vo,cov=m.view(c,'oracle',p,bounds);vr,_=m.view(c,'O1',p,bounds);data[c]=vo
        for key in ('pixel_indices','linear_components','outgoing'):np.testing.assert_array_equal(vo[key],vr[key])
        coverage.append(dict(camera=c,oracle_foreground=int(cov['oracle_hit'].sum()),O1_foreground=int(cov['O1_hit'].sum()),
            common_fitting_or_evaluation=len(vo['pixel_indices']),original_valid=int(cov['valid'].sum()),
            positive_rule_exclusions=int(cov['valid'].sum())-len(vo['pixel_indices']),
            missing=int((cov['oracle_hit']&~cov['O1_hit']).sum()),excess=int((~cov['oracle_hit']&cov['O1_hit']).sum()),
            undefined_O1=int((cov['O1_hit']&~cov['positive_pool']).sum()),
            exclusions_overlap={k:int(cov[k].sum()) for k in ('distance_failure','normal_failure','facing_failure')}))
    support=m.support_identity(data)
    root=m.ROOT/'docs/results/publication_gate13';root.mkdir()
    g.write(root/'input-inventory.json',dict(protocol=m.PROTOCOL,roots={k:str(v) for k,v in p.items()},files=files,
        backend=f['backend'],reference_freeze_sha256=gen['freeze_sha256'],O1_handoff=g.item(m.ROOT/g.HANDOFF),
        accepted_K1_freeze_sha256=m.K1_SHA,support=support,coverage=coverage,
        masks='Oracle-assisted same-ray geometric restriction only; no material labels used for selection.',
        observed_execution=dict(checkpoint_loads=0,SDF_queries=0,BVH_rays=0,shader_integrations=0,optimizer_updates=0)))
    print('Saved common fitting inventory:',support['fitting_observations'],'points;',support['effective_training_passes'],'effective passes')


if __name__=='__main__':
    c=argparse.ArgumentParser();c.add_argument('phase',choices=('inventory','freeze'));a=c.parse_args()
    if a.phase=='inventory':inventory()
    else:
        m.write(m.ROOT/m.FREEZE,m.definition(OUTPUT));print(m.digest(m.ROOT/m.FREEZE))
