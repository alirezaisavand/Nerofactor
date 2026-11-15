"""Explicit admitted tracing dependency selection; metadata operations import no Torch."""
import importlib.util
from pathlib import Path
import sys
from infrastructure.signatures import require,read_report
from infrastructure.stage_sources import ROOT,digest
from infrastructure.ray_backend_e11c import BASE,ORIGINAL_BINARY

CORRECTED_IMAGE='sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331'
CORRECTED_BINARY='800606e1544c42f3e750978d0cce747c5b69c15996391a8709fdc3669274f3e5'
ADMISSION_SHA='1131ff9e1fb28907202f6aae14420debc159a47c2ef537ac482952c16e9eac15'
ADMISSION=ROOT/'docs/evidence/RAY_BACKEND_E11C_ACCEPTANCE_V2.json'
PATHS={'original':Path('/usr/local/lib/python3.8/dist-packages/_raytracing.cpython-38-x86_64-linux-gnu.so'),
       'predicate_fp64':Path('/opt/e11c/corrected/_raytracing.cpython-38-x86_64-linux-gnu.so')}


def identity(variant='predicate_fp64'):
    require(variant in PATHS,'Explicit known tracing backend required')
    if variant=='predicate_fp64':
        proof,_=read_report(ADMISSION,ADMISSION_SHA)
        require(proof['status']=='PROCEED' and proof['corrected_reference_matches']==44 and not proof['remaining_mismatches'] and
                proof['corrected_binary']['sha256']==CORRECTED_BINARY and proof['image']['image_id']==CORRECTED_IMAGE,
                'Corrected native tracing admission mismatch')
    return dict(variant=variant,image_id=CORRECTED_IMAGE if variant=='predicate_fp64' else BASE,
        binary_sha256=CORRECTED_BINARY if variant=='predicate_fp64' else ORIGINAL_BINARY,binary_path=str(PATHS[variant]),
        acceptance_sha256=ADMISSION_SHA if variant=='predicate_fp64' else None)


def image_for(variant):return identity(variant)['image_id']


def load(variant,image_id):
    selected=identity(variant);require(image_id==selected['image_id'],'Tracing backend/image mismatch')
    require('_raytracing' not in sys.modules,'Tracing extension already imported; fresh process required')
    path=PATHS[variant]
    require(path.is_file() and path.resolve()==path and digest(path)==selected['binary_sha256'],'Tracing binary absent or changed; no fallback')
    spec=importlib.util.spec_from_file_location('_raytracing',path);module=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module);sys.modules['_raytracing']=module
    except BaseException:
        sys.modules.pop('_raytracing',None);raise
    return selected
