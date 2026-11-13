"""Future explicitly authorized v2 execution. D.3 never invokes recovery."""
import argparse
import copy
import json
import os
from pathlib import Path

from experiments import neural_material_recovery_logged_v2 as d2
from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.neural_material_journal_runtime_v2 import RuntimeOutput
from infrastructure.neural_material_resources_v2 import capacity
from infrastructure.signatures import require
from infrastructure.runs import validate_run_id


def validate_execution_review(review,root=None,run_id=None):
    from experiments import freeze_neural_material_recovery_logged_v2 as f
    require(type(review) is dict and review.get('phase')==f.PHASE and review.get('scientific_execution_authorized') is True,
        'Separate explicit researcher v2 scientific authorization required; D3 does not authorize execution')
    require(set(review)=={'phase','scientific_execution_authorized','freeze_path','freeze_sha256','source_inventory_sha256',
            'approved_output_parent','approved_output_root','approved_run_id','external_watchdog'},'Exact v2 authorization schema required')
    require(review['external_watchdog']==f.WATCHDOG,'Unchanged external watchdog required')
    require(review['freeze_sha256']!=d2.FREEZE_SHA,'Original v1 freeze cannot authorize reporting v2')
    freeze=f.validate_freeze(Path(review['freeze_path']),review['freeze_sha256'])
    require(review['source_inventory_sha256']==log.digest(freeze['sha256']),'Authorization source inventory changed')
    parent=Path(freeze['output_policy']['approved_parent']);root=Path(root) if root is not None else None
    validate_run_id(run_id)
    require(root is not None and root.is_absolute() and root.resolve()==root and root==parent/run_id and
        run_id.startswith(freeze['output_policy']['required_run_prefix']) and not root.exists(),
        'Fresh unique v2 root/run ID required; no reuse, overwrite or implicit resume')
    require(review['approved_output_parent']==str(parent) and review['approved_output_root']==str(root) and review['approved_run_id']==run_id,'Authorization must select exact parent/root/run ID')
    capacity(parent,freeze['resource_budget']['minimum_free_bytes'])
    h=d2._scientific_v1()
    require(all(os.environ.get(k)==v for k,v in h.previous.original.REQUIRED_ENV.items()),'Hidden GPU/one-thread environment required before allocation')
    import platform
    require(dict(python=platform.python_version(),numpy=h.np.__version__,torch=h.torch.__version__,device='cpu',torch_threads=h.torch.get_num_threads())==freeze['inputs']['runtime'] and
        h.torch.get_default_dtype()==h.torch.float32 and str(h.torch.get_default_device())=='cpu','Exact accepted CPU runtime/dtype required')
    require(h.previous.accepted.arithmetic_runtime()==freeze['inputs']['arithmetic_runtime'],'Accepted arithmetic runtime/rounding changed')
    return freeze


def _execution_graph(freeze,review):
    """Private graph: only output/authorization bindings differ from admitted D2."""
    namespace=d2._execution_graph()
    class BoundOutput(RuntimeOutput):pass
    BoundOutput.freeze=freeze;BoundOutput.review=copy.deepcopy(review)
    namespace['OwnedOutput']=BoundOutput
    def checked(actual,root,run_id):
        admitted=validate_execution_review(actual,root,run_id)
        require(admitted==freeze and actual==review,'Private graph freeze/review differs from admitted selection')
        return admitted
    namespace['validate_execution_review']=checked
    return namespace


def source_equivalence():
    audit=d2.source_equivalence_audit();namespace=_execution_graph({},{});h=d2._scientific_v1()
    changed={n for n,value in vars(h).items() if namespace[n] is not value}
    require(changed=={'OwnedOutput','validate_execution_review','train_history','execute_reviewed'} and
        namespace['train_history'].__code__ is h.train_history.__code__,'Scientific dataflow/loop changed')
    return dict(D2=audit,scientific_bytecode_identical=True,unchanged_scientific_globals=True,
        intentional_D3_changes=['durable commit anchors','bounded seal instead of in-process finalization','new strict freeze/researcher authorization gate','storage admission and derived-artifact caps'],
        new_scientific_calls=0,default_execution_authorized=False)


def execute_reviewed(review,root,run_id):
    # Full admission before mkdir/model/shader/optimizer. The private original
    # execution function performs the same gate again before scientific work.
    freeze=validate_execution_review(review,root,run_id)
    return _execution_graph(freeze,review)['execute_reviewed'](review,root,run_id)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--describe',required=True,action='store_true');p.parse_args()
    print(json.dumps(source_equivalence(),sort_keys=True,indent=2))


if __name__=='__main__':main()
