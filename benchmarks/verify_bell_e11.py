"""Offline integrity/encoding verification only; never renders or loads a model."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from infrastructure.signatures import require,read_report
from infrastructure.stage_sources import digest
from infrastructure.stage_workspace import source_records
from audits.bell_calibration_e11 import decode_camera
from benchmarks.bell_e11 import MESH,convergence,linear_to_srgb,verify_labels
from benchmarks.bell_geometry_e11 import mesh_arrays


def verify(root,report_sha256):
    root=Path(root);g=root/'geometry';report,_=read_report(g/'generation.json',report_sha256)
    require(root.is_absolute() and root.resolve()==root,'Explicit unaliased generated root required')
    seal=json.loads((root/'completion.json').read_text())
    require(seal.get('generation_sha256')==report_sha256 and report.get('status')=='complete' and
            seal.get('cleanup_verified') is True and report.get('scientific_work_complete') is True,'Incomplete generation/cleanup seal')
    require(digest(g/'mesh.ply')==MESH==report['mesh_sha256'],'Oracle identity mismatch')
    for r in report['outputs']:
        p=root/r['path'];require(p.resolve()==p and root in p.parents and p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],'Generated artifact changed: '+r['path'])
    scene=root/'datasets/GlossySynthetic/bell';require(source_records(scene)==report['dataset_files'],'Native dataset identity changed')
    vertices,_=mesh_arrays(g/'mesh.ply');bounds=np.stack([vertices.min(0),vertices.max(0)]).astype(float)
    views=[]
    with np.load(g/'calibrations-safe.npz',allow_pickle=False) as cameras:
        for i in range(128):
            P,K=decode_camera((scene/(str(i)+'-camera.pkl')).read_bytes())
            require(np.array_equal(P,cameras['pose'][i]) and np.array_equal(K,cameras['K'][i]),'Native camera/archive mismatch')
            with np.load(root/'references'/(str(i)+'.npz'),allow_pickle=False) as ref:
                fine=ref['linear_components'];coarse=ref['coarse_components'];hit=ref['hit']
                require(fine.shape==(128,128,3,3) and fine.dtype==np.float64 and hit.shape==(128,128),'Reference shape/dtype mismatch')
                gate=convergence(fine[hit,2],coarse[hit,2]);require(gate['passed'],'Stored convergence does not reproduce')
                with Image.open(scene/(str(i)+'.png')) as png:
                    require(np.array_equal(np.asarray(png),linear_to_srgb(fine[:,:,2])),'Native PNG/reference encoding mismatch')
                with Image.open(scene/'masks_train'/(str(i)+'.png')) as png:
                    require(np.array_equal(np.asarray(png)>127,hit),'Native mask/reference mismatch')
            with np.load(g/'rays'/(str(i)+'.npz'),allow_pickle=False) as saved:
                expected_depth=np.rint(np.where(saved['hit'],saved['depth'],15).reshape(128,128)/15*65535).astype(np.uint16)
                with Image.open(scene/(str(i)+'-depth.png')) as png:
                    require(np.array_equal(np.asarray(png),expected_depth),'Native ray-distance depth encoding mismatch')
            if str(i) in report['splits']['validation']+report['splits']['held_view_nvs']:
                with np.load(root/'references/relight'/(str(i)+'.npz'),allow_pickle=False) as ref:
                    fine=ref['linear_components'];coarse=ref['coarse_components']
                    require(convergence(fine[hit,2],coarse[hit,2])['passed'],'Relight convergence does not reproduce')
                    with Image.open(root/'references/relight'/(str(i)+'.png')) as png:
                        require(np.array_equal(np.asarray(png),linear_to_srgb(fine[:,:,2])),'Relight PNG/reference mismatch')
            with np.load(g/'labels'/(str(i)+'.npz'),allow_pickle=False) as saved:
                data={k:saved[k] for k in saved.files};labels=verify_labels(data['position'],data['normals'],bounds,data)
                require(np.array_equal(data['pixel_indices'],np.flatnonzero(hit)),'Labels do not match foreground pixels')
            views.append({'camera_id':str(i),'convergence':gate,'labels':labels})
    return {'schema_version':1,'protocol':'bell-oracle-offline-verification-e11-v1','status':'pass',
            'generation_sha256':report_sha256,'views':views,'scientific_queries':0,
            'researcher_admission':'pending separate review; never inferred from hashes'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=verify(a.root,a.report_sha256)
    with a.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
