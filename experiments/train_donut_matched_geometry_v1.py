"""Select one frozen trainer recipe and supply the matched Donut acquisition."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_matched_geometry_recovery_v1 as m
from benchmarks.bell_known_l2_v1 import route_plan, validate_map
from experiments import bell_final_tensor_trainer_v1 as native
from experiments.bell_final_tensor_admission_v1 import tensor_admission
from experiments.donut_reconstructed_geometry_v1 import interrupted
from experiments.train_bell_dense_known_l2_v1 import RecordedPositions
from infrastructure.signatures import require


def trainer_source(ambient):
    raw = (m.ROOT/native.SOURCE).read_bytes()
    require(sha256(raw).hexdigest() == native.SOURCE_SHA, 'Accepted native trainer differs')
    code = native.transformed_source() if ambient else raw.decode()
    changes = (
        ("    if faceted:\n        report['counters'].update(boundary_predictor_calls=0, boundary_predictor_points=0)",
         "    report.update(geometry=bound['geometry'], matched_support=freeze['support'],\n"
         "                  reference_Gate3_receipt_sha256=auth['gate3_receipt_sha256'])"),
        ("        reference=first['linear_components'][ix,2];coarse=first['coarse_components'][ix,2]",
         "        coarse,reference=acquisition.compatibility_rgb(panel[0],ix)"),
    )
    for old,new in changes:
        require(code.count(old)==1, 'Expected unique geometry adapter site')
        code=code.replace(old,new,1)
    # Remove prior Bell/Donut evaluation-only off-image diagnostics. Gate 4
    # scores the frozen boundary-neighbor pairs from already exported pixels.
    for start,end in (
        ('        # The historical diagnostic edges remain disjoint from D6 training edges.', '        def evaluate(step):'),
        ('                q=[]\n                diagnostic_widths=[]', '                y,x=np.mgrid[:128,:256]'),
    ):
        require(code.count(start)==code.count(end)==1 and code.index(start)<code.index(end), 'Expected ordered diagnostic sites')
        code=code[:code.index(start)]+code[code.index(end):]
    return code


class Acquisition:
    FACETED = True  # Dispatch to explicit geometry admission. No shading-normal change.
    PANEL, SPLITS = m.PANEL, m.SPLITS
    ADMISSION, TARGET_KEYS = m.ADMISSION, m.TARGET_KEYS
    point_errors = staticmethod(m.old.point_errors)
    validate_map = staticmethod(validate_map)
    route_plan = staticmethod(route_plan)
    tensor_admission = staticmethod(tensor_admission)
    freeze = staticmethod(m.freeze)

    def __init__(self, f, p, args):
        self.f, self.p, self.args = f,p,args
        self.arm = args.phase
        self.recipe = f['arms'][self.arm]
        self.geometry_kind = self.recipe['geometry']
        self.record = None
        self.g, self.views, self.references, _, self.geometries = m.admit(f,p)

    def runtime(self, args):
        a = m.rgb.geo.h.read(args.authorization)
        require(a == m.authorization(self.f,args.commit,args.freeze_sha256)
                and m.digest(args.authorization)==args.authorization_sha256, 'Exact matched recovery approval required')
        require(args.image_id==m.IMAGE and args.benchmark_sha256==m.GENERATION
                and args.initial_sha256==self.f['base_initial']['sha256'], 'Acquisition identity differs')
        recipe = dict(self.recipe, initial_sha256=args.initial_sha256, benchmark_generation_sha256=m.GENERATION,
                      protocol=m.PROTOCOL)
        active = dict(self.f,arm_training=recipe)
        if recipe['representation'] is not None:
            active['tensor_training'] = dict(recipe, arms=list(m.bell.d9.ARMS), initialization_policy=m.bell.INITIALIZATION,
                                             surface_q_enabled=False)
        return active,a

    def geometry(self, freeze, path):
        require(path==self.p['oracle'], 'Explicit oracle geometry member required')
        f = self.f['gate3']['geometry_audit']
        key = 'oracle' if self.geometry_kind=='oracle' else 'mesh'
        m.rgb.geo.h.check_artifact(self.p[key],f[key])
        return self.geometries[self.geometry_kind],None,{self.geometry_kind:f[key]}

    def dataset(self, freeze, root, admission, verification):
        require((root,admission,verification)==tuple(self.p[k] for k in ('reference','admission','verification')),
                'Admitted Donut observations required')
        if self.record is not None:
            self.record.verify(self.f['support']['photometric_index_sha256'])
            m.admit(self.f,self.p)
        return self.g,self.views[self.geometry_kind]

    def compatibility_rgb(self, camera, ix):
        return self.references[self.geometry_kind,camera][:,ix]

    def fit_pool(self, data):
        require(self.record is None, 'Fitting pool already constructed')
        fit = m.old.fit_pool(data)
        require(set(fit)==set(m.FIT_KEYS) and len(fit['position'])==self.f['support']['fitting_observations'],
                'Unexpected optimizer input inventory')
        self.record = RecordedPositions(fit['position'], self.args.output)
        return dict(fit,position=self.record)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--phase',choices=m.ARMS,required=True)
    for flag in ('freeze-sha256','commit','authorization-sha256'):
        cli.add_argument('--'+flag,required=True)
    cli.add_argument('--execute', action='store_true', required=True)
    args = cli.parse_args()
    args.freeze=m.ROOT/m.FREEZE
    f=m.freeze(args.freeze,args.freeze_sha256)
    p=m.paths(f,native=True)
    args.output=Path('/outputs')/args.phase
    args.authorization=Path('/authorization.json')
    require(os.geteuid()!=0 and args.output.resolve()==args.output and not args.output.exists()
            and not os.statvfs('/outputs').f_flag&os.ST_RDONLY
            and all(os.statvfs(path).f_flag&os.ST_RDONLY for path in (Path('/'),m.ROOT,args.authorization,*p.values()))
            and {ln.split(':')[0].strip() for ln in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Fresh isolated non-root matched geometry runtime required')
    args.image_id=m.IMAGE
    args.initial, args.oracle, args.benchmark = (p[k] for k in ('initial','oracle','reference'))
    args.admission,args.verification=p['admission'],p['verification']
    args.initial_sha256=f['base_initial']['sha256']
    args.benchmark_sha256=m.GENERATION
    ambient=f['arms'][args.phase]['representation'] is not None
    args.arm_initial=p['ambient_initial' if ambient else 'initial']
    code = trainer_source(ambient)
    require(sha256(code.encode()).hexdigest()==f['arms'][args.phase]['trainer_sha256'], 'Frozen trainer overlay differs')
    acquisition = Acquisition(f,p,args)
    namespace = dict(__name__='donut_gate4_native',__file__=str(m.ROOT/native.SOURCE))
    exec(compile(code,'donut-matched-geometry-v1','exec'),namespace)
    namespace['train'](args,acquisition=acquisition)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    main()
