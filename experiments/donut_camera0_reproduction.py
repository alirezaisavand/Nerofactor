"""One authorized native Blender frame, followed by file-only comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='donut-camera0-blender-reproduction-v1'
LIMITS=dict(wall_seconds=3600,kill_grace_seconds=30,CPU_threads=8,
            RSS_bytes=16*1024**3,address_space_bytes=32*1024**3,
            output_bytes=512*1024**2,render_calls=1,samples=4096,resolution=[1080,1080])


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def verify_freeze(path,sha):
    require(digest(path)==sha,'Freeze hash mismatch')
    f=json.loads(path.read_text())
    require(f['protocol']==PROTOCOL and f['limits']==LIMITS,'Unexpected frozen experiment')
    for name,r in f['inputs'].items():
        p=Path(name);require(p.is_absolute() and p.resolve()==p and p.is_file() and
            p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],'Input changed: '+name)
    for name,h in f['sources'].items():
        p=ROOT/name;require(ROOT in p.resolve().parents and p.resolve()==p and digest(p)==h,'Source changed: '+name)
    return f


def authorize(path,f,sha,commit):
    a=json.loads(path.read_text())
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==commit and
        a.get('freeze_sha256')==sha and a.get('output_directory')==f['output_directory'] and
        a.get('limits')==LIMITS,'Explicit exact-source single-render authorization required')
    return a


def comparison(ref,pred,fixed_mask,render_mask):
    import numpy as np
    from scipy.ndimage import binary_erosion
    from skimage.metrics import structural_similarity
    require(ref.dtype==pred.dtype==np.uint16 and ref.shape==pred.shape and ref.ndim==3 and ref.shape[-1]==3,'Original16-bit RGB required')
    require(fixed_mask.shape==render_mask.shape==ref.shape[:2] and fixed_mask.dtype==render_mask.dtype==bool,'Aligned boolean masks required')
    x=ref.astype(np.float64)/65535;y=pred.astype(np.float64)/65535
    error=y-x;absolute=abs(error)
    _,ssim_map=structural_similarity(x,y,win_size=11,channel_axis=2,data_range=1.,full=True)
    ssim_map=ssim_map.mean(-1)
    common=fixed_mask&render_mask;union=fixed_mask|render_mask
    regions=dict(full_image=np.ones(fixed_mask.shape,bool),fixed_original_foreground=fixed_mask,
        common_foreground=common,common_SSIM_window_interior=binary_erosion(common,structure=np.ones((11,11),bool)))
    result={}
    for name,mask in regions.items():
        require(mask.any(),'Empty comparison region: '+name)
        e=error[mask];a=absolute[mask];mse=float(np.mean(e*e))
        result[name]=dict(pixels=int(mask.sum()),MSE=mse,PSNR_dB=float(-10*np.log10(mse)) if mse else None,
            exact_match=bool(mse==0),MAE=float(a.mean()),max_absolute_error=float(a.max()),
            absolute_error_quantiles=np.quantile(a,[.5,.95,.99]).tolist(),
            mean_signed_RGB=e.mean(0).tolist(),SSIM=float(ssim_map[mask].mean()),
            fraction_RGB_components_within_2_LSB=float((a<=2/65535+1e-15).mean()))
    return dict(regions=result,foreground_IoU=float(common.sum()/union.sum()),
        original_foreground_pixels=int(fixed_mask.sum()),render_foreground_pixels=int(render_mask.sum()),
        original_only_pixels=int((fixed_mask&~render_mask).sum()),render_only_pixels=int((render_mask&~fixed_mask).sum())),absolute,ssim_map


def srgb_preview(linear):
    import numpy as np
    x=np.maximum(linear,0)
    return np.clip(np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055),0,1)


def analyze(output):
    import numpy as np
    from PIL import Image
    report_path=output/'render-receipt.json';receipt=json.loads(report_path.read_text())
    require(receipt['status']=='complete' and receipt['successful_render_returns']==1,'One completed render required')
    for name,r in receipt['outputs'].items():
        p=output/name;require(p.is_file() and digest(p)==r['sha256'] and p.stat().st_size==r['bytes'],'Output changed: '+name)
    with np.load(output/'comparison-inputs.npz',allow_pickle=False) as data:
        ref=data['reference_RGB16'];pr=data['reproduced_RGB16'];rt=data['roundtrip_RGB16'];linear=data['linear_RGB'];mask=data['reference_mask'];render_mask=data['render_mask']
    require(np.isfinite(linear).all(),'Nonfinite EXR radiance')
    values,error,ssim=comparison(ref,pr,mask,render_mask)
    roundtrip=np.abs(pr.astype(np.int32)-rt.astype(np.int32))
    result=dict(schema_version=1,protocol=PROTOCOL,status='file_only_comparison_complete',render_receipt_sha256=digest(report_path),
        comparison=values,EXR_to_AgX_roundtrip=dict(max_LSB=int(roundtrip.max()),different_components=int((roundtrip!=0).sum()),within_2_LSB=bool((roundtrip<=2).all())),
        linear_RGB=dict(min_per_channel=linear.min((0,1)).tolist(),max_per_channel=linear.max((0,1)).tolist(),
            negative_components=int((linear<0).sum()),components_above_one=int((linear>1).sum())),
        interpretation='No camera, exposure, color gain, crop or registration fitting. Scores alone cannot establish historical renderer identity.',
        new_render_calls=0)
    np.savez_compressed(output/'RGB-errors.npz',absolute_RGB_error=error,SSIM_map=ssim,reference_mask=mask,render_mask=render_mask)
    def save(name,im):Image.fromarray(np.clip(np.rint(im*255),0,255).astype(np.uint8)).save(output/name)
    save('reference-preview.png',ref/65535);save('reproduction-preview.png',pr/65535);save('linear-sRGB-preview.png',srgb_preview(linear))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,4.6))
    for axis,image,title in zip(axes[:3],(ref/65535,pr/65535,srgb_preview(linear)),
        ('Original dataset / AgX','Reproduced / AgX','Linear EXR → sRGB preview\nclip0…1; no AgX')):
        axis.imshow(image);axis.set_title(title);axis.axis('off')
    im=axes[3].imshow(error.max(-1),vmin=0,vmax=.1,cmap='magma');axes[3].set_title('Max-channel absolute error\nfixed range0…0.1');axes[3].axis('off');fig.colorbar(im,ax=axes[3],fraction=.046)
    fig.tight_layout();fig.savefig(output/'camera0-comparison.png',dpi=160);plt.close(fig)
    with (output/'comparison.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


def launch(args):
    f=verify_freeze(args.freeze,args.freeze_sha256)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=str(ROOT),text=True),'Clean reviewed source required')
    authorize(args.authorization,f,args.freeze_sha256,commit)
    output=Path(f['output_directory']);require(output.is_absolute() and output.resolve()==output and not output.exists(),'Fresh exact output required')
    output.mkdir();(output/'runtime').mkdir()
    (output/'freeze.json').write_bytes(args.freeze.read_bytes());(output/'authorization.json').write_bytes(args.authorization.read_bytes())
    command=[f['Blender_executable'],'--background','--factory-startup','--disable-autoexec','--threads','8','--python-exit-code','2',
        '--python',str(ROOT/'experiments/render_donut_camera0.py'),'--','--execute','--output',str(output),
        '--freeze',str(args.freeze),'--freeze-sha256',args.freeze_sha256,'--authorization',str(args.authorization),'--commit',commit]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',
        PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(output/'runtime/cache'),BLENDER_USER_CONFIG=str(output/'runtime/config'))
    def bounds():
        resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['wall_seconds']*8,)*2)
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    started=time.monotonic();cause=None;peak_RSS=0;child=None
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):raise KeyboardInterrupt('Supervisor interrupted')
    for s in previous:signal.signal(s,interrupted)
    try:
        with (output/'console.log').open('x') as console:
            child=subprocess.Popen(command,cwd=str(output),env=env,stdout=console,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=bounds)
            while child.poll() is None:
                elapsed=time.monotonic()-started
                try:
                    status=Path('/proc')/str(child.pid)/'status';rss=int(next(l for l in status.read_text().splitlines() if l.startswith('VmRSS:')).split()[1])*1024
                    peak_RSS=max(peak_RSS,rss)
                except (FileNotFoundError,StopIteration):rss=0
                size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.is_symlink())
                if elapsed>LIMITS['wall_seconds'] or rss>LIMITS['RSS_bytes'] or size>LIMITS['output_bytes']:
                    cause='wall/RSS/output limit';break
                time.sleep(2)
    except BaseException as exc:
        cause=type(exc).__name__+': '+str(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=LIMITS['kill_grace_seconds'])
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for s,handler in previous.items():signal.signal(s,handler)
        code=child.returncode if child else None
        receipt=dict(command=command,exit_code=code,stop_reason=cause,wall_seconds=time.monotonic()-started,
            observed_peak_RSS_bytes=peak_RSS,supervisor_cleanup_complete=child is not None and child.poll() is not None)
        (output/'execution-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    require(code==0 and cause is None,'Render failed; preserve evidence, no retry')
    verify_freeze(args.freeze,args.freeze_sha256)
    result=analyze(output);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--authorization',type=Path,required=True)
    launch(parser.parse_args())
