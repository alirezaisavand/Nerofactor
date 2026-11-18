"""Cold-process entry for the separately authorized reference or recovery arm."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmarks.donut_recovery_v1 import inputs,verify_freeze,PANEL,SPLITS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('reference','D1','D2','D3','D4','D5','D6'),required=True)
    for name in ('oracle','cameras','freeze','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--execute',action='store_true')
    for name in ('authorization','benchmark','initial','previous'):p.add_argument('--'+name,type=Path)
    for name in ('authorization-sha256','benchmark-sha256','initial-sha256','commit','image-id'):
        p.add_argument('--'+name)
    args=p.parse_args()
    freeze=verify_freeze(args.freeze,args.freeze_sha256);inputs(args.oracle,args.cameras)
    if not args.execute:
        from infrastructure.signatures import require
        require(not any(n.split('.')[0] in ('torch','network','raytracing','bpy') for n in sys.modules), 'Metadata validation imported scientific runtime')
        print(json.dumps(dict(status='metadata_only',panel=PANEL,splits=SPLITS,scientific_execution=False)))
        return
    from infrastructure.signatures import require
    require(all((args.authorization,args.authorization_sha256,args.commit,args.image_id)), 'Explicit execution approval required')
    if args.phase=='reference':
        if freeze.get('reference_protocol')=='donut-kurt-reference-v2':
            require(args.previous is not None,'Explicit immutable failed reference batch required')
            from benchmarks.generate_donut_reference_v2 import generate
        else:
            from benchmarks.generate_donut_recovery_v1 import generate
        generate(args)
    else:
        require(args.benchmark is not None and args.benchmark_sha256 is not None,'Admitted immutable benchmark required')
        from experiments.donut_material_recovery_v1 import train
        train(args)


if __name__=='__main__':main()
