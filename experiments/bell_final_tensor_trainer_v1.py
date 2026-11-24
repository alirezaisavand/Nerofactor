"""Exact, opt-in overlay of the frozen native Bell trainer for the final pair."""
from pathlib import Path
from hashlib import sha256

from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT

SOURCE = 'experiments/donut_material_recovery_v1.py'
SOURCE_SHA = 'ac511516e3b05b3c27b3a55b7f8d5e5a5e97f01eba2deb83f67ae968a341baa7'
REGULARIZATION = 'legacy_scalar_energy_specular_white_no_surface_Q_v1'
# Every edit is an exact, unique site in SOURCE_SHA. Keep the legacy file
# byte-identical so the previously frozen runs remain verifiable.
EDITS = (
    ("    spatial=freeze.get('surface_q_training') or chart or tensor_plan",
     "    require(tensor_plan and tensor_plan['surface_q_enabled'] is False and not chart\n"
     "            and freeze.get('surface_q_training') is None,'Final pair requires tensor shader without surface-Q')\n"
     "    spatial=None"),
    ("if args.phase in D9_ARMS:",
     "if tensor_plan['initialization_policy']=='validated_D9_weak_start_v1':"),
    ("effective_regularization='legacy_scalar_energy_specular_white_plus_surface_Q_v1',",
     "effective_regularization='legacy_scalar_energy_specular_white_no_surface_Q_v1',\n"
     "                      optimization_terms=['RGB_loss','material_prior','diffuse_white_prior'],"),
    ("    if tensor_plan:\n        report.update(representation",
     "    if tensor_plan:\n        report['counters'].update(tensor_admission_predictor_calls=0,tensor_admission_reverse_sweeps=0)\n"
     "        report.update(representation"),
    ("                        np.savez_compressed(args.output/'paired-initial-gate.npz',**capture)\n"
     "                        report['paired_initial_gate']=gate",
     "                        np.savez_compressed(args.output/'paired-initial-gate.npz',**capture)\n"
     "                        report['paired_initial_gate']=gate\n"
     "                        require(gate['passed'],'Final pair initial Q/RGB/scalar parity failed')\n"
     "                        report['physical_tensor_admission']=acquisition.tensor_admission(\n"
     "                            shader,sibling,p,n,wo,args.output,budget,torch)"),
)


def transformed_source():
    raw = (ROOT/SOURCE).read_bytes()
    require(sha256(raw).hexdigest()==SOURCE_SHA, 'Frozen native trainer differs')
    source = raw.decode()
    for before,after in EDITS:
        require(source.count(before)==1,'Expected single frozen trainer edit site')
        source = source.replace(before,after,1)
    return source


def transformed_sha256():
    return sha256(transformed_source().encode()).hexdigest()


def run(args, acquisition, expected):
    code = transformed_source()
    require(sha256(code.encode()).hexdigest()==expected,'Versioned trainer overlay differs')
    namespace = dict(__name__=__name__,__file__=str(Path(ROOT/SOURCE)))
    exec(compile(code,'bell-final-tensor-trainer-v1','exec'),namespace)
    return namespace['train'](args,acquisition=acquisition)
