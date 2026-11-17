"""Lossless sphere maps and file-only historical comparison. No model imports."""
import csv
import json
from pathlib import Path

from infrastructure.signatures import require
from infrastructure.stage_sources import digest


def uint8(value, np):
    # Exact pinned utils.base_utils.color_map_backward (no gamma here).
    return np.clip(value * 255, 0, 255).astype(np.uint8)


def orientation_maps(rotation, mx, my, mask, np):
    require(rotation.shape == (*mask.shape, 2), 'Orientation/image shape mismatch')
    theta = .5 * np.arctan2(rotation[..., 1], rotation[..., 0])
    major_is_my = my[..., 0] > mx[..., 0]
    theta = np.mod(theta + major_is_my * (np.pi / 2), np.pi)
    hue = theta / np.pi
    # Vectorized HSV(hue, saturation, value); grey only at exactly equal widths.
    sector = hue * 6
    f = sector - np.floor(sector)
    p, q, t = np.zeros_like(f), 1 - f, f
    one = np.ones_like(f)
    choices = np.stack([np.stack(c, -1) for c in ((one,t,p),(q,one,p),(p,one,t),
                                                       (p,q,one),(t,p,one),(one,p,q))])
    rgb = np.take_along_axis(choices, (np.floor(sector).astype(int) % 6)[None,...,None], axis=0)[0]
    rgb = np.where((mx == my), .5, rgb)
    return theta, np.where(mask[..., None], rgb, 0)


def write_view(directory, arrays, identity, np):
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    mask = arrays['hit_mask']
    require(mask.ndim == 2 and mask.any(), 'Empty/invalid visualization mask')
    require(all(np.isfinite(v).all() for v in arrays.values()), 'Nonfinite material visualization data')
    angle, orientation_rgb = orientation_maps(arrays['rotation'], arrays['mx'], arrays['my'], mask, np)
    arrays = dict(arrays, major_width=np.maximum(arrays['mx'], arrays['my']),
                  minor_width=np.minimum(arrays['mx'], arrays['my']), major_angle_radians=angle)
    t,b=2*arrays['tangents']-1,2*arrays['bitangents']-1
    Q=arrays['mx'][...,None]**2*t[..., :,None]*t[...,None,:]+arrays['my'][...,None]**2*b[..., :,None]*b[...,None,:]
    axis=np.where(arrays['my']>arrays['mx'],b,t)
    projector=axis[..., :,None]*axis[...,None,:]
    gap=np.abs(arrays['mx']**2-arrays['my']**2)/np.maximum(arrays['mx']**2+arrays['my']**2,1e-12)
    raw_angle=np.mod(.5*np.arctan2(arrays['rotation'][...,1],arrays['rotation'][...,0]),np.pi)
    arrays.update(Q_world=Q,major_axis_projector_world=projector,anisotropy_gap=gap,raw_angle_radians=raw_angle)
    # Keep raw head coordinates, linear components and original encoded frames.
    np.savez_compressed(directory / 'materials.npz', **arrays)
    names = dict(ground_truth='rgb_gt', reconstructed='rgb_pr', albedo='kd', metallic='metallic',
                 mx='mx', my='my', major_width='major_width', minor_width='minor_width',
                 diffuse='diffuse_color', specular='specular_color', normal='normals')
    for filename, key in names.items():
        image = uint8(arrays[key], np)
        if image.shape[-1] == 1:
            image = image[..., 0]
        Image.fromarray(image).save(directory / (filename + '.png'))
    Image.fromarray(uint8(orientation_rgb, np)).save(directory / 'orientation.png')
    _,raw_rgb=orientation_maps(arrays['rotation'],np.ones_like(arrays['mx']),np.zeros_like(arrays['my']),mask,np)
    Image.fromarray(uint8(raw_rgb,np)).save(directory/'orientation_raw.png')
    world_rgb=np.stack([projector[...,0,1],projector[...,0,2],projector[...,1,2]],-1)+.5
    # Fixed, display-only eligibility. Q itself is defined even at isotropy.
    world_rgb=np.where(gap<.1,.5,world_rgb)*mask[...,None]
    Image.fromarray(uint8(world_rgb,np)).save(directory/'orientation_world_projector.png')
    for i in range(3):
        for j in range(i,3):
            channel=Q[...,i,j]+(0 if i==j else .5)
            Image.fromarray(uint8(channel*mask,np)).save(directory/('Q_world_%d%d.png'%(i,j)))
    Image.fromarray(mask.astype(np.uint8) * 255).save(directory / 'mask.png')
    record = dict(identity, width_scale=[0,1], albedo_scale=[0,1], metallic_scale=[0,1],
        RGB='original color_map_backward, RGB uint8; no extra gamma; linear components in NPZ',
        orientation='major-width angle mod pi, hue=angle/pi; corrected local tangent chart; grey only exact isotropy',
        axes='mx/my are original raw predictor axes; major=max(mx,my), minor=min(mx,my); no training head reordering',
        frame='tangents/bitangents/normals are original encoded (vector+1)/2; position and normal are world coordinates',
        world_Q='mx^2*t*tT+my^2*b*bT; diagonal0..1, offdiagonal-.5...5 shifted+.5; no learned scale',
        world_orientation='Unoriented major-axis projector; RGB=Pxy,Pxz,Pyz shifted+.5; gap<.1 grey for display only; raw orientation retained',
        background='original mesh-hit mask; original test_step zeros RGB outside hits',
        files={p.name:dict(bytes=p.stat().st_size, sha256=digest(p)) for p in directory.iterdir() if p.is_file()})
    (directory / 'view.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def historical_comparison(output, historical):
    """Only saved images: never requery the shader or load historical checkpoints."""
    import numpy as np
    from PIL import Image, ImageDraw
    from skimage.metrics import structural_similarity
    from infrastructure.sphere_replay import split_ids
    output, historical = Path(output), Path(historical)
    path = output / 'comparison'
    path.mkdir(exist_ok=False)
    rows = []
    for camera in split_ids()['nvs']:
        old_path = historical / 'nvs_imgs/sphere_material' / ('test_' + camera + '.png')
        new_path = output / 'renders/step-100000/nvs' / ('camera-' + camera)
        old = np.asarray(Image.open(old_path).convert('RGB'))
        gt = np.asarray(Image.open(new_path / 'ground_truth.png').convert('RGB'))
        corrected = np.asarray(Image.open(new_path / 'reconstructed.png').convert('RGB'))
        require(old.shape == (2 * gt.shape[0], *gt.shape[1:]), 'Historical panel dimensions differ; no resizing')
        old_gt, original = np.split(old, 2, axis=0)
        delta = np.abs(old_gt.astype(int) - gt.astype(int))
        row = dict(camera_id=camera, corrected_completed_updates=100000, historical_completed_updates='unknown',
                   historical_gt_max_uint8_difference=int(delta.max()),
                   historical_gt_different_pixels=int(np.any(delta != 0, axis=-1).sum()),
                   paired_common_GT_valid=bool(np.array_equal(old_gt, gt)))
        # Each score uses its own preserved GT. Common-GT scores would be
        # misleading if geometry masks/calibration/quantization differ.
        for arm, reference, prediction in (('historical',old_gt,original),('corrected',gt,corrected)):
            mse = np.mean((reference.reshape(-1,3).astype(np.float32) - prediction.reshape(-1,3).astype(np.float32))**2, axis=0).mean()
            row[arm+'_psnr'] = float(10 * np.log10(255 * 255 / mse)) if mse else None
            row[arm+'_ssim'] = float(structural_similarity(reference,prediction,win_size=11,channel_axis=2,data_range=255))
        rows.append(row)
        panels = [gt, original, corrected] if row['paired_common_GT_valid'] else [old_gt, original, gt, corrected]
        labels = ['GT','Historical A (step unknown)','Corrected A (100000)'] if len(panels)==3 else [
            'Historical GT','Historical A (step unknown)','Corrected GT','Corrected A (100000)']
        sheet = Image.new('RGB',(gt.shape[1]*len(panels),gt.shape[0]+32),'white')
        draw = ImageDraw.Draw(sheet)
        for i,(panel,label) in enumerate(zip(panels,labels)):
            sheet.paste(Image.fromarray(panel),(i*gt.shape[1],32));draw.text((i*gt.shape[1]+5,8),label,fill='black')
        sheet.save(path / ('camera-'+camera+'.png'))
    with (path / 'scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (path / 'comparison.json').write_text(json.dumps(dict(status='complete',scores=rows,
        historical_training_step='unverified: no material checkpoint or producing config available',
        comparison_scope='Descriptive historical-image comparison; not a matched-training-budget causal ablation',
        original_logged_scores=(historical/'logs/sphere_material/log.txt').read_text()),indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    logs = [json.loads(line) for line in (output / 'training.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for prefix,axis in (('train',axes[0]),('val',axes[1])):
        keys=sorted({k for row in logs if row['prefix']==prefix for k in row if k not in ('prefix','completed_updates')})
        for key in keys:
            selected=[r for r in logs if r['prefix']==prefix and key in r]
            axis.plot([r['completed_updates'] for r in selected],[r[key] for r in selected],label=key)
        axis.set_title(prefix);axis.set_xlabel('Completed original Adam returns');axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(path/'training-curves.png',dpi=160);plt.close(fig)
    return rows
