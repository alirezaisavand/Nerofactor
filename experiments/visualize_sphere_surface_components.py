"""One file-only GT/RGB/component comparison; no material predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from infrastructure.stage_sources import digest
from infrastructure.signatures import require


def render(report_path,output):
    r=json.loads(report_path.read_text());require(not output.exists(),'Fresh figure path required')
    paths=[next(Path(p) for p in r['external_artifacts'] if '/sphere-core-v2-' in p and p.endswith('materials.npz'))]
    paths.extend(Path(r['batch'])/arm/'run/renders/step-008000/validation/camera-36/materials.npz'
                 for arm in ('surface_chart_control_v1','surface_world_q_v1'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,6,figsize=(15,10))
    for row,(path,label) in enumerate(zip(paths,('Initial4000','Chart control8000','World-Q8000'))):
        require(digest(path)==r['external_artifacts'][str(path)]['sha256'],'Saved archive changed')
        with np.load(path,allow_pickle=False) as a:
            gt,pr=a['rgb_gt'],a['rgb_pr']
            values=(gt,pr,a['kd'],a['diffuse_color'],a['specular_color'],abs(gt-pr)*4)
        for col,(v,name) in enumerate(zip(values,('GT','RGB','kd (linear display)','diffuse sRGB','specular sRGB','abs sRGB error ×4'))):
            axes[row,col].imshow(np.clip(v[::3,::3],0,1),vmin=0,vmax=1);axes[row,col].set_title(label+'\n'+name,fontsize=9);axes[row,col].axis('off')
    fig.tight_layout();fig.savefig(output,dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.report,args.output)
