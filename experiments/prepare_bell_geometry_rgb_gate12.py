"""Saved-file Gate 12 inventory/admission and fitting-camera angular audit only."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks import bell_geometry_rgb_gate12 as g
from benchmarks.bell_sam2_ordering_gate12 import camera_order
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline,PINS

O1_SOURCE='2d5a1ac0d2f04e701e0a75276d7ac67aee1ad9a1'
O1_FREEZE='4b09eb7871eb2f0d8e52f142a814d7148808a3908320c464d8d4d05a003cc81c'
RESULTS='e04795c68fe914a3d42d2fb3987d81798b2b6ccb'
CHECKPOINT='b9a5fb4c461de77ec9f387466e05373f959df65baebd5151f02278d1648c46e7'
MESH='0becad75d5f5404b84f5a154870eaf66e71f00ba5b843776f50dd6b1989d7ac3'
OUT=Path('/home/isavand/paper/evidence/bell-geometry-rgb-gate12-0ea32eb2081a435aa20ffef242773ce5')


def main():
    old=g.prior.freeze(O1_FREEZE);root=Path(old['host_output']);dest=g.ROOT/'docs/results/publication_gate12';dest.mkdir()
    for stage,commit in PINS.items():baseline(stage,commit)
    completed=g.read(root/'offline-verification.json');training=g.read(root/'train/training.json')
    audit=g.read(root/'evaluate/evaluation.json');endpoint=audit['steps'][-1]
    require(completed['quality_eligible'] and not completed['geometry_admitted'] and training['commit']==O1_SOURCE and
            training['adam_returns']==300000 and training['cleanup_verified'] and audit['cleanup_verified'] and
            endpoint['SDF_gate']['passed'] and all(v['passed'] for v in endpoint['views']),
            'O1 original evidence incomplete; do not fabricate admission')
    require(training['endpoint_checkpoint']['sha256']==CHECKPOINT and endpoint['mesh']['sha256']==MESH,'O1 identity differs')
    g.check(root/'train/checkpoints/model.pth',training['endpoint_checkpoint'])
    g.check(root/'evaluate/300000/candidate.ply',endpoint['mesh'])
    for phase in ('train','evaluate'):
        require((root/f'{phase}-exit-code.txt').read_text().strip()=='0' and
                (root/f'{phase}-container-absent.txt').read_text().strip()=='absent','Owned resource cleanup not verified')
    for path,key in [('train/training.json','training_receipt'),('evaluate/evaluation.json','evaluation_receipt'),
                     ('analysis/verification.json','verification')]:g.check(root/path,completed[key])
    replay=g.read(root/'analysis/verification.json')
    require(replay['substantive_mismatches']==0 and not replay['roundoff_events'],'O1 accepted verification differs')
    handoff=dict(protocol='authenticated-O1-controlled-geometry-handoff-v1',
        status='admitted_for_controlled_downstream_evaluation',variant=g.prior.VARIANT,
        researcher_approval='Gate 12: Admit the authenticated O1 candidate for controlled downstream evaluation as an explicitly identified perfect-synthetic-mask-assisted geometry derivative.',
        accepted_results_commit=RESULTS,producing_commit=O1_SOURCE,producing_freeze_sha256=O1_FREEZE,
        authoritative_G=PINS['geometry'],derivative='original foreground L1 plus coefficient-1 background opacity under original mask schedule',
        seed=6033,optimizer_returns=300000,split=old['splits'],image=old['image'],backend=old['backend'],
        checkpoint=training['endpoint_checkpoint'],checkpoint_path=str(root/'train/checkpoints/model.pth'),
        mesh=endpoint['mesh'],mesh_path=str(root/'evaluate/300000/candidate.ply'),
        training_receipt=g.item(root/'train/training.json'),evaluation_receipt=g.item(root/'evaluate/evaluation.json'),
        verification=g.item(root/'analysis/verification.json'),prior_offline_verification=g.item(root/'offline-verification.json'),
        exact_float32_cleanup=endpoint['cleanup'],topology=endpoint['topology_after'],
        extraction='Gate 10.2 conservative fully-inside-cell raw-SDF zero crossings; exact float32 weld/zero-face cleanup only',
        perfect_synthetic_mask_assisted=True,unchanged_original_G=False,realistic_SAM2_validation=False,
        O0_admission='remains blocked; diagnostic comparison only',
        material_training_authorized=False,new_rendering_authorized=False,
        limitations=['one seed, Bell development data','normal/depth tails and small recall tradeoff','open boundaries and disconnected components retained'])
    external=root.parent/(root.name+'-controlled-admission-v1.json')
    require(not external.exists(),'Separate admission record already exists; do not overwrite')
    g.write(external,handoff);g.write(dest/'O1-controlled-handoff.json',handoff)
    oracle=Path(old['acquisition']);O0=g.prior.evolution.ACCEPTED/'run';O1=root/'evaluate/300000'
    files={'oracle':{},'O0':{},'O1':{}}
    generation=g.read(oracle/'geometry/generation.json');catalog={x['path']:{k:x[k] for k in ('sha256','bytes')} for x in generation['outputs']}
    require(g.digest(oracle/'geometry/generation.json')==g.prior.original.GEN_SHA,'Admitted Bell generation differs')
    oracle_names=['geometry/mesh.ply','geometry/calibrations-safe.npz']
    for c in g.PANEL:oracle_names.extend([f'geometry/rays/{c}.npz',f'geometry/labels/{c}.npz',f'references/{c}.npz'])
    for name in oracle_names:files['oracle'][name]=catalog[name];g.check(oracle/name,catalog[name])
    files['oracle']['geometry/generation.json']=g.item(oracle/'geometry/generation.json')
    proofs={'O0':g.read(g.ROOT/'docs/results/publication_gate102/results-v2/artifact-manifest.json')['files'],
            'O1':g.read(g.ROOT/'docs/results/publication_gate11/results-v1/artifact-manifest.json')['native_files']}
    for arm,folder,mesh in [('O0',O0,'domain-aware-300000.ply'),('O1',O1,'candidate.ply')]:
        names=[mesh,'topology-after-cleanup.npz']+[f'rasters/{c}.npz' for c in g.PANEL]
        for name in names:
            expected=proofs[arm][('run/' if arm=='O0' else 'evaluate/300000/')+name]
            g.check(folder/name,expected);files[arm][name]=expected
    with np.load(oracle/'geometry/calibrations-safe.npz',allow_pickle=False) as z:
        ids=list(map(int,old['splits']['train']));poses=z['pose'][ids];K=z['K'][ids]
    ordering=camera_order(ids,poses)
    excluded=old['splits']['validation']+old['splits']['held_view_nvs']
    require(set(ordering['frame_to_camera'])==set(old['splits']['train']) and
            not set(ordering['frame_to_camera'])&set(excluded),'Held-view leakage in prospective SAM2 order')
    masks=g.ROOT/'docs/results/publication_gate11/realistic-mask-definition.json';original=g.read(masks)
    ordering.update(protocol='bell-fitting-image-SAM2-angular-order-v2',status='prospective_no_inference',
        original_definition=g.item(masks),calibration=g.item(oracle/'geometry/calibrations-safe.npz'),
        used_rows=list(map(str,ids)),excluded=excluded,endpoint_anchors=['0','127'],prompts=original['prompts'],
        original_numeric_definition_preserved=True,old_trimap_rule_unchanged=True,
        manual_annotations='Reuse original image-only P1/P2 on cameras 0 and 127 without new prompts.',
        calibration_subset_sha256=g.prior.original.calibration_sha(dict(source_camera_ids=np.array(ids,dtype=np.int64),pose=poses,K=K)),
        segmentation_inference_authorized=False,checkpoint_loads=0)
    g.write(dest/'sam2-ordering-v2.json',ordering)
    inv=dict(protocol=g.PROTOCOL,image=old['image'],backend=old['backend'],
        roots=dict(oracle=str(oracle),O0=str(O0),O1=str(O1)),files=files,
        O1_admission_path=str(external),O1_admission=g.item(external),
        O1_native_evaluation=g.item(root/'evaluate/evaluation.json'),
        O0_accepted_audit=g.item(O0/'receipt.json'),
        domain_band=old['domain']['cell_counts']['maximum_boundary_omission_depth'],counts={})
    # Inventory from accepted arrays only, no new rendering or geometry queries.
    vertices,_=g.mesh_arrays(oracle/'geometry/mesh.ply');bounds=np.stack([vertices.min(0),vertices.max(0)]).astype(float)
    rows=[];p={k:Path(v) for k,v in inv['roots'].items()}
    for c in g.PANEL:
        d=g.prepare(c,p,bounds);row=dict(camera=c,oracle=len(d['views']['oracle']['pixels']))
        for arm in ('O0','O1'):
            masks=d['cohorts'][arm];row[arm]=dict(defined=len(d['views'][arm]['pixels']),
                undefined=int(masks['undefined'].sum()),excess=int(masks['reconstruction_only'].sum()),
                missing=int(masks['oracle_only'].sum()),grazing=int(masks['grazing'].sum()),
                near_open_boundary=int(masks['near_open_boundary'].sum()),
                large_depth_shift=int(masks['large_depth_shift'].sum()),
                audit_vs_label_normal_max_abs=d['transfers'][arm]['audit_vs_label_normal_max_abs'])
        rows.append(row)
    inv['counts']=dict(per_camera=rows,oracle_total=sum(x['oracle'] for x in rows),
        defined={arm:sum(x[arm]['defined'] for x in rows) for arm in ('O0','O1')},
        undefined_total=sum(x[arm]['undefined'] for x in rows for arm in ('O0','O1')))
    g.write(dest/'input-inventory.json',inv)
    g.write(g.ROOT/g.FREEZE,g.definition(OUT))
    shutil.copyfile('/tmp/gate12-tests.xml',dest/'focused-cpu-tests.junit.xml')
    print(json.dumps(dict(admission=str(external),counts=inv['counts'],ordering=ordering['angular_order'],
                         freeze_sha256=g.digest(g.ROOT/g.FREEZE)),indent=2))


if __name__=='__main__':main()
