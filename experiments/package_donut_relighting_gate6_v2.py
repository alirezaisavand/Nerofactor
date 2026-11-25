"""Export the verified Gate 6 tables and small figures with source identities."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks import donut_relighting_gate6_v2 as g
from benchmarks.donut_recovery_v1 import json_new
from infrastructure.signatures import require


PACKAGED = ('relighting-results.json','offline-replay.json','independent-relighting.png',
            'independent-relighting.pdf','native-relighting.png','native-relighting.pdf')
ROWS = ('K1-Oracle','Final-Chart-Oracle','Ambient-Oracle')


def identity(path):
    return dict(bytes=path.stat().st_size,sha256=g.digest(path))


def write_csv(path,rows):
    require(bool(rows),'Empty table')
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def metrics_table(report):
    rows=[]
    for camera,panel in [*report['views'].items(),('pooled',None)]:
        for arm in ROWS:
            for route in ('independent','native'):
                scores=panel[route][arm] if panel is not None else report['pooled'][arm][route]
                for scope,group in scores.items():
                    for component,metric in group.items():
                        values=metric or dict(pixels=0,rmse=None,relative_L2=None,PSNR_unit_peak_dB=None,exact=None)
                        rows.append(dict(camera=camera,arm=arm,route=route,scope=scope,component=component,**values))
    return rows


def diagnostic_tables(report,receipt):
    uncertainty,convergence,association=[],[],[]
    for camera,panel in report['views'].items():
        for role,metric in panel['convergence'].items():
            convergence.append(dict(camera=camera,field=role,
                                    refined_pixels=' '.join(map(str,receipt['views'][camera]['refined_pixels'])),**metric))
        for arm in ROWS:
            u=panel['uncertainty'][arm]
            for i,component in enumerate(report['component_order']):
                uncertainty.append(dict(camera=camera,arm=arm,component=component,
                    grid_reference_RMSE=u['reference_grid_difference'][component]['rmse'],
                    grid_prediction_RMSE=u['prediction_grid_difference'][component]['rmse'],
                    paired_grid_residual_change_RMSE=u['paired_grid_residual_change'][component]['rmse'],
                    native_minus_independent_RMSE=u['native_minus_independent'][component]['rmse'],
                    native_draw_SE_RMS=u['native_standard_error_RMS'][i],
                    native_split_half_RMS=u['native_split_half_difference_RMS'][i]))
            a=panel['physical_associations'][arm]
            for physical,values in a.items():
                if physical=='orientation_eligible_rows':
                    continue
                count=a['orientation_eligible_rows'] if physical=='orientation_degrees' else panel['support']['primary_common']
                for component,value in values.items():
                    association.append(dict(camera=camera,arm=arm,physical_error=physical,relighting_component=component,
                                            observations=count,spearman_rho=value))
    return {'convergence.csv':convergence,'rendering-uncertainty.csv':uncertainty,'physical-associations.csv':association}


def package(sha,analysis,out):
    f=g.freeze(sha)
    batch=Path(f['host_output'])
    auth_path=batch/'authorization.json'
    auth=g.read(auth_path)
    native=batch/'run/receipt.json'
    r=g.read(native)
    report=g.read(analysis/'relighting-results.json')
    replay=g.read(analysis/'offline-replay.json')
    require(auth==g.authorization(f,auth['commit'],sha) and g.digest(batch/'freeze.json')==sha
            and g.digest(auth_path)==g.digest(batch.with_name(batch.name+'-authorization.json'))
            and (batch/'exit-code.txt').read_text().strip()=='0' and r['status']=='complete' and r['cleanup_verified']
            and r['authorization_sha256']==g.digest(auth_path) and r['freeze_sha256']==sha
            and r['commit']==auth['commit'] and r['backend']==f['backend']
            and report['status']=='verified_complete' and report['freeze_sha256']==replay['freeze_sha256']==sha
            and report['authorization_sha256']==r['authorization_sha256']
            and report['receipt_sha256']==replay['receipt_sha256']==g.digest(native)
            and r['counters']==report['work'], 'Expected exact verified Gate 6 launch and offline replay')
    for name,item in r['outputs'].items():
        g.check(native.parent/name,item)
    failed=g.read(g.ROOT/g.FAILURE)
    for name,item in failed['files'].items():
        g.check(Path(failed['batch'])/name,item)
    require(out.resolve()==out and not out.exists() and out.parent.is_dir()
            and batch not in out.parents and analysis not in out.parents,'Fresh publication output required')
    out.mkdir()
    for name in PACKAGED:
        shutil.copyfile(analysis/name,out/name)
        require(identity(analysis/name)==identity(out/name),'Packaged copy differs')
    write_csv(out/'relighting-metrics.csv',metrics_table(report))
    for name,rows in diagnostic_tables(report,r).items():
        write_csv(out/name,rows)
    artifacts={name:dict(identity(batch/name),path=str(batch/name)) for name in
               ('freeze.json','authorization.json','run/receipt.json','console.txt','exit-code.txt','container.cid')}
    offline={p.name:dict(identity(p),path=str(p)) for p in sorted(analysis.iterdir()) if p.is_file()}
    json_new(out/'artifact-manifest.json',dict(schema_version=1,protocol=g.PROTOCOL,status='verified_complete',
        commit=auth['commit'],image=f['image'],backend=r['backend'],frozen_artifacts=f['artifacts'],
        v1_failure_receipt_sha256=f['v1_failure_receipt_sha256'],host_output=str(batch),execution_artifacts=artifacts,
        native_outputs=r['outputs'],checkpoint_hashes=auth['checkpoints'],gates=r['views'],field_admission=r['field_admission'],
        native_counters=r['counters'],cleanup_verified=r['cleanup_verified'],offline_analysis=offline,
        packaged_files={p.name:identity(p) for p in sorted(out.iterdir())},
        offline_new_renders=0,offline_checkpoint_loads=0,offline_optimizer_updates=0))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    package(a.freeze_sha256,a.analysis,a.output)
