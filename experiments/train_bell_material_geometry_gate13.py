"""Minimal exact-site Bell acquisition overlay of the accepted Gate 4 K1 loop."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks import bell_material_geometry_gate13 as m
from experiments.train_donut_matched_geometry_v1 import trainer_source as inherited_source
from experiments.train_bell_dense_known_l2_v1 import RecordedPositions
from infrastructure.bell_material_pair_gate13 import SamplingAudit
from infrastructure.signatures import require


def trainer_source():
    code=inherited_source(False)
    changes=[
        ("bound.get('illumination')=='known_L2'","bound.get('illumination')=='known_Bell_v3'"),
        ('from infrastructure.known_l2_audit import KnownL2Audit','from infrastructure.bell_material_pair_gate13 import KnownBellV3Audit as KnownL2Audit'),
        ("'known analytic L2'","'known original Bell v3 environment'"),
        ("report.update(geometry=bound['geometry'], matched_support=freeze['support'],\n                  reference_Gate3_receipt_sha256=auth['gate3_receipt_sha256'])",
         "report.update(geometry=bound['geometry'], matched_support=freeze['support'],\n                  O1_handoff_sha256=auth['O1_handoff_sha256'],oracle_assisted_correspondence=True,heterogeneous_dataset=False)"),
        ("first=data[panel[0]];valid=", "first=data[58];valid="),
        ('acquisition.compatibility_rgb(panel[0],ix)','acquisition.compatibility_rgb(58,ix)'),
        ('from benchmarks.donut_recovery_v1 import ENVIRONMENT','ENVIRONMENT=acquisition.illumination'),
        ("truth=np.zeros((SIZE*SIZE,3));truth[d['pixel_indices']]=d['linear_components'][:,2]",
         "truth=d['full_linear_components'][:,2].copy()"),
        ("split='fitting' if camera in splits['fitting'] else 'held_out'",
         "split='fitting' if camera in splits['fitting'] else 'validation' if camera==58 else 'held_out'"),
        ("report['native_reference_gate_lighting']='known L2, same compatibility control as D1/D2'",
         "report['native_reference_gate_lighting']='original Bell v3, accepted Gate 12 geometry-specific reference'"),
        ('                        # Deterministic field maps from the same returned forward, no extra material query.',
         '                        acquisition.evaluation_draws(step,camera,ids,np.stack(draws))\n'
         '                        # Deterministic field maps from the same returned forward, no extra material query.'),
    ]
    for old,new in changes:
        require(code.count(old)==1,'Expected unique Gate 13 acquisition overlay site: '+old);code=code.replace(old,new,1)
    start='                        with calibration_scope(False):\n'
    end='                        loss=training_objective('
    require(code.count(start)==code.count(end)==1,'Expected one original sampling scope')
    a,b=code.index(start),code.index(end)
    block=code[a:b];code=code[:a]+"                        with acquisition.sampling_step(step,torch):\n"+''.join('    '+line if line.strip() else line for line in block.splitlines(True))+code[b:]
    marker="                require(report['optimizer_returns']==10000,'Incomplete recovery history')"
    require(code.count(marker)==1,'Original two-update-independent endpoint boundary missing')
    code=code.replace(marker,marker+'\n                acquisition.sampling_complete()',1)
    return code


class Acquisition:
    FACETED=True
    PANEL=m.panel()
    SPLITS=dict(fitting=list(map(int,m.split()['train'])),validation=[58],held_out=list(map(int,m.split()['held_view_nvs'])))
    ADMISSION,TARGET_KEYS=m.ADMISSION,m.TARGET_KEYS
    point_errors=staticmethod(m.point_errors)
    freeze=staticmethod(m.freeze)
    illumination=m.g.ENV

    def __init__(self,f,p,args):
        self.f,self.p,self.args=f,p,args;self.kind=f['arms'][args.phase]['geometry'];self.record=None;self.sampling=None
        self.draw_seconds=0.;self.draw_buffer=None

    def runtime(self,args):
        require(m.read(args.authorization)==m.authorization(self.f,args.commit,args.freeze_sha256) and
            m.digest(args.authorization)==args.authorization_sha256 and args.image_id==m.IMAGE and
            args.benchmark_sha256==m.GENERATION and args.initial_sha256==self.f['base_initial']['sha256'],
            'Exact separate Gate 13 approval required')
        for path in (Path('/'),m.ROOT,args.authorization,*self.p.values()):
            require(os.statvfs(path).f_flag&os.ST_RDONLY,'Read-only source, inputs and root required')
        require(os.getuid()!=0 and args.output==Path('/outputs')/args.phase and not args.output.exists() and
            {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Fresh networkless non-root runtime required')
        if args.phase=='K1-O1':
            require(os.statvfs('/paired').f_flag&os.ST_RDONLY and
                    os.statvfs('/outputs/K1-Oracle').f_flag&os.ST_RDONLY,'Completed oracle evidence must be read-only')
        recipe=dict(self.f['arms'][args.phase],initial_sha256=args.initial_sha256,benchmark_generation_sha256=m.GENERATION)
        return dict(self.f,arm_training=recipe),m.read(args.authorization)

    def geometry(self,f,path):
        require(path==self.p['oracle'],'Explicit immutable scene input required')
        root=self.p['oracle'] if self.kind=='oracle' else self.p['O1'];name='geometry/mesh.ply' if self.kind=='oracle' else 'candidate.ply'
        vertices,faces=m.g.mesh_arrays(root/name)
        return dict(vertices_native=vertices,triangles=faces),None,{self.kind:m.item(root/name)}

    def dataset(self,f,root,admission,verification):
        require(root==self.p['oracle'] and admission==self.p['admission'] and verification==self.p['verification'],
                'Frozen data/admission bindings differ')
        if self.record:self.record.verify(self.f['support']['photometric_index_sha256'])
        return m.dataset(self.f,self.p,self.kind)

    def fit_pool(self,data):
        require(self.record is None,'One fitting pool per history')
        pool=m.fit_pool(data);require(set(pool)==set(m.FIT_KEYS),'Material truth reached optimizer inventory')
        self.record=RecordedPositions(pool['position'],self.args.output)
        self.sampling=SamplingAudit(self.args.output,Path('/paired/sampling-tapes.jsonl') if self.args.phase=='K1-O1' else None)
        return dict(pool,position=self.record)

    def sampling_step(self,step,torch):return self.sampling.step(step,torch)
    def sampling_complete(self):
        self.sampling.close()
        m.write(self.args.output/'observer-cost.json',dict(raw_draw_output_seconds=self.draw_seconds,
            sampling_observer_seconds=self.sampling.observer_seconds,
            caveat='Observed host bookkeeping/copy time, including probe synchronization; not isolated GPU overhead.'))

    def evaluation_draws(self,step,camera,indices,draws):
        start=time.monotonic();n=next(x['common'] for x in self.f['support']['cameras'] if x['camera']==camera)
        if indices[0]==0:
            require(self.draw_buffer is None,'Prior camera draw capture incomplete')
            self.draw_buffer=np.empty((8,n,3,3),np.float32)
        self.draw_buffer[:,indices]=draws
        if indices[-1]==n-1:
            root=self.args.output/'native-draws';root.mkdir(exist_ok=True)
            with (root/f'step-{step:05d}-camera-{camera}.npz').open('xb') as stream:
                # Primitive numeric schema, compatible with the accepted strict
                # evidence loader: archive component order is diffuse/spec/RGB.
                np.savez_compressed(stream,draws=self.draw_buffer,component_reference_indices=np.array([2,0,1],np.int8))
            self.draw_buffer=None
        self.draw_seconds+=time.monotonic()-start

    def compatibility_rgb(self,camera,indices):
        require(camera==58,'Frozen compatibility camera differs')
        base=self.data_pixels(camera)
        arm='oracle' if self.kind=='oracle' else 'O1'
        values=[]
        for size in m.g.BASE:
            with np.load(self.p['gate12']/f'{camera}-{arm}-base-{size}.npz',allow_pickle=False) as z:
                rows=np.searchsorted(z['pixels'],base[indices]);np.testing.assert_array_equal(z['pixels'][rows],base[indices])
                values.append(z['components'][rows,0,2])
        return values

    def data_pixels(self,camera):
        v,_=m.g.mesh_arrays(self.p['oracle']/'geometry/mesh.ply');bounds=np.stack([v.min(0),v.max(0)]).astype(float)
        return m.view(camera,self.kind,self.p,bounds)[0]['pixel_indices']

    @staticmethod
    def validate_map(value,expected):
        from benchmarks.bell_known_l2_v1 import validate_map
        return validate_map(value,expected)

    @staticmethod
    def route_plan(work,paired):
        from benchmarks.bell_known_l2_v1 import route_plan
        return route_plan(work,paired)


def main():
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('--phase',choices=m.ARMS,required=True)
    for flag in ('commit','freeze-sha256','authorization-sha256'):c.add_argument('--'+flag,required=True)
    c.add_argument('--execute',action='store_true',required=True);a=c.parse_args();a.freeze=m.ROOT/m.FREEZE
    f=m.freeze(a.freeze,a.freeze_sha256);p=m.paths(True)
    a.output=Path('/outputs')/a.phase;a.authorization=Path('/authorization.json');a.image_id=m.IMAGE
    a.initial=a.arm_initial=p['initial'];a.oracle=a.benchmark=p['oracle'];a.admission=p['admission'];a.verification=p['verification']
    a.initial_sha256=f['base_initial']['sha256'];a.benchmark_sha256=m.GENERATION
    code=trainer_source();require(sha256(code.encode()).hexdigest()==f['arms'][a.phase]['trainer_sha256'],'Trainer overlay differs')
    acq=Acquisition(f,p,a);namespace=dict(__name__='bell_gate13_native',__file__=str(m.ROOT/'experiments/donut_material_recovery_v1.py'))
    exec(compile(code,'bell-material-geometry-gate13-v1','exec'),namespace)
    namespace['SHADER']=m.SHADER;namespace['POLICY']=m.POLICY
    namespace['train'](a,acquisition=acq)


if __name__=='__main__':
    from experiments.donut_reconstructed_geometry_v1 import interrupted
    signal.signal(signal.SIGTERM,interrupted);main()
