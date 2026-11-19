"""File-only comparison of the completed Bell recovery arms."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks import bell_heterogeneous_training_v1 as bell
from experiments.analyze_donut_heterogeneous_training_v1 import analyze


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--output',type=Path,required=True)
    args = cli.parse_args()
    print(f'Verified and scored {analyze(args.freeze_sha256,args.output,acquisition=bell)} Bell arm/checkpoint/split combinations')
