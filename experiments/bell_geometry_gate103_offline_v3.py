"""File-only verification of completed Gate 10.3 v2, without new native execution."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks import bell_geometry_gate103_v2 as p
from benchmarks.bell_geometry_gate103_replay_v3 import compare_surface,POLICY
from benchmarks.bell_geometry_e11 import mesh_arrays
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS,baseline

SOURCES=('experiments/bell_geometry_gate103_offline_v3.py',
         'benchmarks/bell_geometry_gate103_replay_v3.py',
         'tests/scientific/test_bell_geometry_gate103_replay_v3.py')

def verify(plan,args,output):
    from experiments.bell_geometry_gate10_v1 import verify_rasters,verify_sdf
    root=Path(plan['host_output']);out=root/'run';r=p.read(out/'receipt.json')
    roundoff_events=[]
    require((root/'exit-code.txt').read_text().strip()=='0' and (root/'container-absent.txt').read_text().strip()=='absent' and
        (root/'image-id.txt').read_text().strip()==plan['image'],'Host execution/image/cleanup receipts differ')
    require(r['commit']==args.commit and r['freeze_sha256']==args.freeze_sha256 and
        r['authorization_sha256']==args.authorization_sha256 and r['status']=='complete' and
        r['scientific_work_complete'] and r['cleanup_verified'] and not r['geometry_admitted'] and
        not r['handoff_created'] and r['observed']['loads']==4 and r['observed']['renders']==128 and
        r['observed']['grid_SDF']==p.LIMITS['dense_SDF_points'],'Incomplete frozen study')
    require([row['step'] for row in r['steps']]==list(p.STEPS) and not r['optimizer_state_restored'] and
            r['optimizer_returns']==0 and r['scientific_commit']==PINS['geometry'],'Study identity/history inventory differs')
    ceilings=dict(consistency_SDF='consistency_SDF_points',neural_SDF='neural_SDF_forward_points',
        neural_gradients='neural_gradient_points',appearance_calls='appearance_forward_calls',
        appearance_points='appearance_point_slots',occ_origins='occ_probe_origins',validation_origins='validation_probe_origins')
    require(all(r['observed'][key]<=p.LIMITS[limit] for key,limit in ceilings.items()) and r['observed']['jit_loads']<=1,
            'Observed primitive work exceeds freeze')
    require(sum(row['MC_attempts'] for row in r['steps'])==sum(row['MC_returns'] for row in r['steps'])==2 and
        sum(row['grid_tiles'] for row in r['steps'])==1024 and
        sum(row['primary_BVH_rays'] for row in r['steps'])==p.LIMITS['audit_BVH_rays'] and
        sum(row['CPU_ray_checks']+row['fitting_CPU_checks'] for row in r['steps'])==p.LIMITS['CPU_ray_checks'] and
        sum(row['fitting_BVH_rays'] for row in r['steps'])==p.LIMITS['fitting_BVH_rays'],
        'Total extraction/ray inventories differ')
    for name,identity in r['artifacts'].items():p.check(out/name,identity)
    outside=np.load(Path(p.policy.read(p.ROOT/p.policy.FREEZE)['diagnostic'])/'run/stages/outside-sphere.npy',allow_pickle=False,mmap_mode='r')
    valid,any_inside=p.policy.domain_cells(outside)
    for row in r['steps']:
        step=row['step'];folder=out/str(step)
        require(row['network_preserved'] and row['finite_checkpoint_tensors'] and
                row['checkpoint']==plan['checkpoints'][str(step)]['identity'],'Checkpoint identity/state differs')
        if step==15000:
            previous=p.completed_15k(plan,p.FAILED)
            require(row['MC_attempts']==row['MC_returns']==row['grid_tiles']==row['primary_BVH_rays']==row['CPU_ray_checks']==0
                and row['reused_15k'] and row['views']==previous['views'] and row['SDF_gate']==previous['SDF_gate'],
                '15k extraction/audit was repeated or changed')
            prefix='run/15000/'
            for name,identity in plan['completed_15k']['files'].items():
                if name.startswith(prefix) and not name[len(prefix):].startswith(('fitting/','neural/')):
                    p.check(folder/name[len(prefix):],identity)
        if step!=300000:
            voxel=np.load(folder/'raw-voxel-vertices.npy',allow_pickle=False);faces=np.load(folder/'raw-faces.npy',allow_pickle=False)
            world=np.load(folder/'raw-world-vertices.npy',allow_pickle=False)
            require(np.array_equal(world,voxel/511*2-1),'Coordinate conversion differs')
            keep,reason,possible,good=p.policy.face_domain(voxel,faces,valid,any_inside)
            with np.load(folder/'face-domain.npz',allow_pickle=False) as z:
                require(all(np.array_equal(z[k],v) for k,v in dict(retained=keep,reason=reason,possible_cells=possible,supported_cells=good).items()),'Domain replay differs')
            v,f,mapping,retained,removed,cleanup=p.policy.numerical_cleanup(world,faces,keep)
            cv,cf=mesh_arrays(folder/'candidate.ply')
            require(np.array_equal(cv,v) and np.array_equal(cf,f) and row['cleanup']==cleanup,'Cleanup replay differs')
            with np.load(folder/'cleanup-mappings.npz',allow_pickle=False) as z:
                require(np.array_equal(z['raw_vertex_to_candidate'],mapping) and np.array_equal(z['candidate_face_to_raw'],retained)
                        and np.array_equal(z['removed_exact_zero_raw_faces'],removed),'Cleanup map replay differs')
            topo,labels,edges=p.policy.topology(v,f)
            require(topo==row['topology_after'],'Topology replay differs')
            with np.load(folder/'topology-after-cleanup.npz',allow_pickle=False) as z:
                require(np.array_equal(z['face_labels'],labels) and np.array_equal(z['boundary_edges'],edges),'Topology arrays differ')
            verify_rasters(folder,row['views']);verify_sdf(folder/'sdf-witnesses.npz',row['SDF_gate'])
        else:
            require(row['MC_attempts']==row['MC_returns']==row['grid_tiles']==row['primary_BVH_rays']==0 and
                    row['views']==p.read(p.ACCEPTED/'run/receipt.json')['views'],'300k audit was changed or repeated')
        require(row['fitting_BVH_rays']==122*128**2 and row['fitting_CPU_checks']==64,'Fitting inventory incomplete')
        require([v['camera'] for v in row['fitting']]==plan['fitting_ids'] and
                row['neural_rays']==2*sum(len(v['indices']) for v in plan['neural_panel'].values()),
                'Fitting camera or neural-ray inventory differs')
        for fit in row['fitting']:
            camera=fit['camera']
            with np.load(folder/'fitting'/f'{camera}.npz',allow_pickle=False) as z:
                expected=p.silhouette(z['oracle_hit'],z['hit'])
                require(all(fit[k]==v for k,v in expected.items()),'Fitting silhouette replay differs')
                with np.load(Path(plan['acquisition'])/'geometry/rays'/f'{camera}.npz',allow_pickle=False) as a:
                    oracle=dict(hit=a['hit'],normal=a['normals'],position=a['positions'],depth=a['depth'])
                    reconstructed=dict(hit=z['hit'],normal=z['normal'],position=z['position'],depth=z['depth'])
                    observed,masks=p.policy.original.paired_surface(oracle,reconstructed,a['directions'])
                    extras={'camera','split','tp','fp','fn','tn','iou','precision','recall','extra_radial_interior',
                            'oracle_domain_supported','oracle_domain_unsupported','missing_inside_domain','missing_outside_domain'}
                    if camera in plan['neural_panel']:extras|={'CPU_witnesses','camera_ray_errors','neural'}
                    require(set(fit)==set(observed)|extras,'Fitting report fields missing or extra')
                    events=compare_surface({k:fit[k] for k in observed},observed)
                    roundoff_events.extend(dict(step=step,camera=camera,**event) for event in events)
                    require(all(z[k].dtype==v.dtype and np.array_equal(z[k],v) for k,v in masks.items()),
                            'Fitting eligibility replay differs')
            if camera in plan['neural_panel']:
                with np.load(folder/'neural'/f'{camera}-rays.npz',allow_pickle=False) as z:
                    expected=dict(origin=float(abs(z['origins']-z['archived_origins']).max()),
                                  direction=float(abs(z['directions']-z['archived_directions']).max()))
                    require(expected==fit['camera_ray_errors'] and max(expected.values())<=2e-6 and
                            np.array_equal(z['indices'],plan['neural_panel'][camera]['indices']),
                            'CPU camera witness replay differs')
                opacity=[]
                for draw in range(2):
                    with np.load(folder/'neural'/f'{camera}-{draw}.npz',allow_pickle=False) as z:
                        require(np.array_equal(z['indices'],plan['neural_panel'][camera]['indices']) and
                                int(z['seed'])==6033+1000*int(camera)+draw and int(z['loop_index'])==step-1 and
                                all(np.isfinite(z[k]).all() for k in ('fg_acc','acc','ray_rgb','fg_rgb','bg_rgb')),
                                'Neural panel/values differ')
                        opacity.append(z['fg_acc'])
                        oracle_hit=z['oracle_hit'];mesh_hit=z['mesh_hit']
                require(fit['neural']==p.opacity_summary(plan['neural_panel'][camera],oracle_hit,mesh_hit,np.stack(opacity)),
                        'Opacity summary replay differs')
    require(sum(row.get('camera_precision_probe_calls',0) for row in r['steps'])==p.LIMITS['camera_precision_probe_calls'],
            'Camera precision witness inventory differs')
    witness=r['steps'][0]['camera_precision_diagnostic']
    require(witness['restored'] and witness['used_for_rendering']=='cpu','Renderer precision policy changed')
    with np.load(out/'15000/neural/camera-precision.npz',allow_pickle=False) as z:
        for label in ('cpu','default','TF32_disabled'):
            expected=dict(origin=float(abs(z[label+'_origins']-z['archived_origins']).max()),
                          direction=float(abs(z[label+'_directions']-z['archived_directions']).max()))
            require(witness['ray_errors'][label]==expected,'Precision-witness replay differs')
    p.write(output/'verification.json',dict(status='complete_verified_diagnostic',
        replay_policy=POLICY,roundoff_events=roundoff_events,substantive_mismatches=0,
        captured_results_modified=False,verifier_sources={name:p.digest(p.ROOT/name) for name in SOURCES},receipt=p.item(out/'receipt.json'),
        geometry_admitted=False,handoff_created=False,publication_endpoint=300000,
        comparison_is_development_evidence=True,quality_decisions={str(row['step']):dict(
            original_six_views=[dict(camera=x['camera'],passed=x['passed']) for x in row['views']],SDF=row['SDF_gate']['passed']) for row in r['steps']}))
    print(output/'verification.json')

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True);a=cli.parse_args()
    sha='e027b4ab9320049086b4e3d7b83c47646f6b228cf9399b43a015a84b089bb862'
    plan=p.freeze(sha);root=Path(plan['host_output']);r=p.read(root/'run/receipt.json')
    require(r['commit']=='53662125ba4184bc9bb48d7a6845a4358aa04fcb','Unexpected native source')
    auth=root/'authorization.json'
    require(p.digest(auth)==r['authorization_sha256'] and p.read(auth)==p.authorization(plan,r['commit'],sha),
            'Approved source/freeze identity differs')
    p.host_inputs(plan)
    for stage,commit in PINS.items():baseline(stage,commit)
    output=a.output
    require(output.resolve()==output and output.parent==root.parent and
            output.name==root.name+'-offline-v3' and not output.exists(),'Fresh separate offline output required')
    output.mkdir()
    p.write(output/'reported-offline-failure.json',dict(source='researcher-supplied traceback',
        failed_verifier='experiments/bell_geometry_gate103_v2.py',line=488,
        message='Fitting physical/eligibility replay differs',native_status=r['status'],
        native_receipt=p.item(root/'run/receipt.json'),original_verifier_preserved=True))
    args=SimpleNamespace(commit=r['commit'],freeze_sha256=sha,authorization_sha256=r['authorization_sha256'])
    verify(plan,args,output)


if __name__=='__main__':main()
