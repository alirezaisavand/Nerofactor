"""Display-only world-Q panels from hash-verified saved sphere arrays."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from infrastructure.sphere_visuals import orientation_maps
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


def render(audit_path,output):
    require(not output.exists(),'Fresh figure file required')
    report=json.loads(audit_path.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,5,figsize=(13,10))
    for row,step in enumerate((2000,10000,20000,40000)):
        path=next(Path(p) for p in report['artifacts'] if ('step-%06d/'%step) in p)
        require(digest(path)==report['artifacts'][str(path)]['sha256'],'Saved map hash mismatch')
        with np.load(path,allow_pickle=False) as data:
            # Subsample DISPLAY only. All reported statistics use full arrays.
            mask=data['hit_mask'][::4,::4];mx=data['mx'][::4,::4];my=data['my'][::4,::4]
            r=data['rotation'][::4,::4];t=2*data['tangents'][::4,::4]-1;b=2*data['bitangents'][::4,::4]-1
            _,raw=orientation_maps(r,np.ones_like(mx),np.zeros_like(my),mask,np)
            _,major=orientation_maps(r,mx,my,mask,np)
            a=np.where(my>mx,b,t);projector=a[..., :,None]*a[...,None,:]
            image=np.stack([projector[...,0,1],projector[...,0,2],projector[...,1,2]],-1)+.5
            image*=mask[...,None]
            q=mx[...,None]**2*t[..., :,None]*t[...,None,:]+my[...,None]**2*b[..., :,None]*b[...,None,:]
            diagonal=np.stack([q[...,0,0],q[...,1,1],q[...,2,2]],-1)*mask[...,None]
            gap=np.abs(mx**2-my**2)/np.maximum(mx**2+my**2,1e-12)
        for col,(field,label) in enumerate(zip((raw,major,image,diagonal,gap[...,0]),
            ('raw chart angle','major chart angle','world unoriented projector','Qxx / Qyy / Qzz','anisotropy gap'))):
            axes[row,col].imshow(np.clip(field,0,1),vmin=0,vmax=1,cmap='gray' if field.ndim==2 else None)
            axes[row,col].set_title(str(step)+' '+label,fontsize=8);axes[row,col].axis('off')
    fig.tight_layout();fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.audit,a.output)
