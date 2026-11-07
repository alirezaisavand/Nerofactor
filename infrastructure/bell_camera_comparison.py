"""CPU saved-array/JSON diagnostic comparisons; no model imports or checkpoint loading."""

import importlib.metadata
from pathlib import Path

from infrastructure.bell_inference import THRESHOLDS, scalar_metric
from infrastructure.bell_comparison import artifact, metric_delta, rgb_metrics
from infrastructure.bell_camera_diagnostic import (PROTOCOL, ADMISSION_SHA256, CHECKPOINT_SHA256, admission, identity)
from infrastructure.signatures import read_report, require, write_report


def score_differences(metrics, old):
    psnr = metrics['psnr']['value'] if metrics['psnr']['status'] == 'finite' else None
    return {'stored_best_psnr': old['stored_full_precision_best_psnr'], 'logged_psnr': old['logged_psnr'],
            'logged_ssim': old['logged_ssim'], 'measured': metrics,
            'psnr_minus_stored_best': psnr-old['stored_full_precision_best_psnr'] if psnr is not None else None,
            'psnr_minus_logged': psnr-old['logged_psnr'] if psnr is not None else None,
            'ssim_minus_logged': metrics['ssim']-old['logged_ssim'],
            'absolute_psnr_difference_from_stored_best': abs(psnr-old['stored_full_precision_best_psnr']) if psnr is not None else None,
            'absolute_psnr_difference_from_logged': abs(psnr-old['logged_psnr']) if psnr is not None else None,
            'absolute_ssim_difference_from_logged': abs(metrics['ssim']-old['logged_ssim']),
            'same_five_decimal_psnr_display': format(psnr,'.5f') == '23.78712' if psnr is not None else False,
            'same_five_decimal_ssim_display': format(metrics['ssim'],'.5f') == '0.87727',
            'historical_metric_acceptance_tolerance': None}


def jpeg_statistics(signed):
    import numpy as np
    absolute = np.abs(signed)
    return {'pixel_exact': bool(not absolute.any()),
            'max_absolute_uint8_difference': int(absolute.max()),
            'mean_absolute_uint8_difference': float(absolute.mean()),
            'root_mean_square_uint8_difference': float(np.sqrt(np.mean(signed.astype(np.float64)**2))),
            'absolute_difference_percentiles_50_95_99': [float(x) for x in np.percentile(absolute,[50,95,99])]}


def historical_comparison(ground_truth_uint8, metrics, montage, run):
    """JPEG panel is an explicitly lossy diagnostic, never the scoring GT."""
    import numpy as np
    import PIL
    from PIL import Image
    approved, _ = admission()
    expected = approved['historical_montage']
    require(ground_truth_uint8.dtype == np.uint8 and ground_truth_uint8.shape == (200,200,3), 'Original RGB conversion required')
    from infrastructure.bell_inference import file_hash
    require(file_hash(montage) == expected['sha256'], 'Historical montage hash changed')
    with Image.open(montage) as image:
        require(image.format == 'JPEG' and image.mode == 'RGB' and list(image.size) == expected['dimensions_wh'],
                'Historical JPEG dimensions/mode differ; do not resize or reinterpret')
        panel = np.asarray(image.crop(tuple(expected['GT_panel_box_xyxy']))).copy()
    require(panel.shape == ground_truth_uint8.shape, 'Historical GT panel size mismatch')
    signed = ground_truth_uint8.astype(np.int16)-panel.astype(np.int16)
    np.save(str(run.output('arrays/current-GT-minus-JPEG-panel.npy')), signed, allow_pickle=False)
    Image.fromarray(panel).save(str(run.output('renders/historical-GT-panel-decoded.png')))
    return {'historical_montage_sha256': expected['sha256'], 'panel_box_xyxy': expected['GT_panel_box_xyxy'],
            'panel_interpretation': 'Inspected legacy layout plus researcher visual identity; producing layout/source remains unattributed.',
            **score_differences(metrics, approved['historical_metrics']),
            'jpeg': {**jpeg_statistics(signed),
                     'Pillow': PIL.__version__, 'libjpeg': getattr(Image.core,'jpeglib_version',None),
                     'acceptance_tolerance': None, 'resize_align_recompress_denoise': False,
                     'limit': 'JPEG compression and unknown producing preprocessing prevent pixel-exact GT identity claims; report differences without tuning.'},
            'origin_attribution': 'No metric/JPEG closeness establishes producing commit, training membership or split.'}


def load_diagnostic(path, digest, expected_step):
    import numpy as np
    report, info = read_report(path,digest)
    require(type(report.get('schema_version')) is int and report['schema_version']==1 and
            report.get('protocol')==PROTOCOL and report.get('status')==report.get('phase')=='complete' and
            type(report.get('evaluation_calls')) is int and report['evaluation_calls']==1,
            'Incomplete diagnostic report')
    require(report.get('source_id')=='0' and type(report.get('evaluation_step')) is int and report['evaluation_step']==expected_step and
            type(report.get('checkpoint_stored_step')) is int and report['checkpoint_stored_step']==150000 and
            report.get('checkpoint_sha256')==CHECKPOINT_SHA256 and report.get('diagnostic_admission_sha256')==ADMISSION_SHA256,
            'Wrong source camera, stored/evaluation step or checkpoint identity')
    require(report.get('implementation')==identity(), 'Current adapter/protocol differs from execution')
    selection=report.get('selection',{})
    require(selection.get('database_source_id')=='0' and type(selection.get('diagnostic_iterator_index')) is int and
            selection['diagnostic_iterator_index']==0 and
            selection.get('diagnostic_index_to_source_id')=={'0':'0'} and selection.get('original_A_validation_ids')==['58'] and
            selection.get('original_A_tables_unchanged') is True and report.get('weights_and_buffers_unchanged') is True and
            report.get('selection_rng_unchanged') is True, 'Unverified camera selection/immutability')
    require([r['method'] for r in selection.get('getter_reads',[])] == ['get_image','get_pose','get_K','get_mask','get_depth'] and
            all(r['source_id']=='0' for r in selection['getter_reads']), 'Actual selected getter IDs disagree')
    from infrastructure.bell_inference import proposed_split
    require(report.get('original_A_splits')==proposed_split(), 'Original A split changed')
    runtime=report['runtime']['packages']
    require(all(runtime[n]==importlib.metadata.version(n) for n in ('numpy','scikit-image')), 'Use the same metric library versions')
    hashes=report['output_hashes'];root=Path(path).resolve().parent
    required={'arrays/prediction.npy','arrays/ground_truth.npy','arrays/mask.npy',
              'arrays/current-GT-minus-JPEG-panel.npy','renders/prediction.png','renders/ground_truth.png',
              'renders/historical-GT-panel-decoded.png','metrics.json'}
    require(type(hashes) is dict and required <= hashes.keys(), 'Missing required diagnostic artifacts')
    for name in hashes:artifact(root,name,hashes)
    arrays={n:np.load(str(artifact(root,'arrays/'+n+'.npy',hashes)),allow_pickle=False)
            for n in ('prediction','ground_truth','mask')}
    p,g,m=arrays['prediction'],arrays['ground_truth'],arrays['mask']
    require(p.shape==g.shape==(200,200,3) and p.dtype==g.dtype==np.float32 and m.shape==(200,200,1) and m.dtype==np.int32 and
            all(np.isfinite(a).all() for a in arrays.values()), 'Invalid lossless diagnostic arrays')
    require(rgb_metrics(p,g)==report['metrics'], 'Stored metrics disagree with lossless arrays')
    metric,_=read_report(root/'metrics.json',hashes['metrics.json']);require(metric==report['metrics'],'Metric artifact mismatch')
    from PIL import Image
    for name,value in [('prediction',p),('ground_truth',g)]:
        with Image.open(root/'renders'/(name+'.png')) as image:
            require(image.mode=='RGB' and image.size==(200,200) and
                    np.array_equal(np.asarray(image),np.clip(value*255,0,255).astype(np.uint8)), 'Lossless PNG disagrees with float RGB')
    with Image.open(root/'renders/historical-GT-panel-decoded.png') as image:
        require(image.mode=='RGB' and image.size==(200,200), 'Invalid decoded JPEG panel artifact')
        panel=np.asarray(image).copy()
    signed=np.clip(g*255,0,255).astype(np.uint8).astype(np.int16)-panel.astype(np.int16)
    saved=np.load(str(root/'arrays/current-GT-minus-JPEG-panel.npy'),allow_pickle=False)
    require(saved.dtype==np.int16 and np.array_equal(saved,signed), 'Saved JPEG difference disagrees with GT/panel')
    approved,_=admission();h=report['historical_comparison']
    expected=approved['historical_montage']
    require(h.get('historical_montage_sha256')==expected['sha256'] and h.get('panel_box_xyxy')==expected['GT_panel_box_xyxy'] and
            all(h.get(k)==v for k,v in score_differences(report['metrics'],approved['historical_metrics']).items()),
            'Historical score/context fields disagree with admitted evidence and metrics')
    require(all(h['jpeg'].get(k)==v for k,v in jpeg_statistics(signed).items()) and
            h['jpeg'].get('acceptance_tolerance') is None and h['jpeg'].get('resize_align_recompress_denoise') is False,
            'JPEG diagnostic metadata disagrees with saved arrays/protocol')
    return report,arrays,info


def compare_steps(primary, control, output):
    import numpy as np
    from infrastructure.bell_inference import file_hash
    r,a,ri=primary;c,b,ci=control
    require(r['run_id']!=c['run_id'], 'Step control must be an independent fresh run')
    fields=('source_id','checkpoint_sha256','checkpoint_stored_step','evidence','diagnostic_admission_sha256',
            'implementation','declared_image_id','declared_source_commit','legacy_sha256','used_inputs','route','selection',
            'original_A_splits','effective_trainer','effective_renderer','effective_shader','network_state_fingerprints',
            'preprocessing','runtime','rng_before','rng_after')
    mismatch=[k for k in fields if r[k]!=c[k]]
    if not np.array_equal(a['ground_truth'],b['ground_truth']):mismatch.append('ground_truth')
    if not np.array_equal(a['mask'],b['mask']):mismatch.append('mask')
    left,right=a['prediction'].astype(np.float64),b['prediction'].astype(np.float64)
    signed=right-left;absolute=np.abs(signed)
    violations=absolute>(THRESHOLDS['float_atol']+THRESHOLDS['float_rtol']*np.abs(left))
    quantized=np.abs(np.clip(a['prediction']*255,0,255).astype(np.uint8).astype(np.int16)-
                     np.clip(b['prediction']*255,0,255).astype(np.uint8).astype(np.int16))
    psnr=metric_delta(r['metrics']['psnr'],c['metrics']['psnr']);ssim=abs(r['metrics']['ssim']-c['metrics']['ssim'])
    passed=not mismatch and not violations.any() and quantized.max()<=THRESHOLDS['uint8_max'] and \
           psnr is not None and psnr<=THRESHOLDS['psnr_db'] and ssim<=THRESHOLDS['ssim']
    hashes={}
    for name,value in [('step-control-minus-primary.npy',signed),('float-violation-mask.npy',violations)]:
        with (output/name).open('xb') as stream:np.save(stream,value,allow_pickle=False)
        hashes[name]=file_hash(output/name)
    return {'schema_version':1,'protocol':PROTOCOL,'kind':'fresh-process-step-control','primary':ri,'control':ci,
            'result':'within_frozen_step_comparison_bounds' if passed else 'step_pair_mismatch',
            'thresholds':dict(THRESHOLDS),'identity_mismatches':mismatch,'rgb_bitwise_equal':bool(np.array_equal(a['prediction'],b['prediction'])),
            'max_absolute_error':float(absolute.max()),'mean_absolute_error':float(absolute.mean()),
            'float_violation_count':int(violations.sum()),'uint8_max_error':int(quantized.max()),
            'uint8_mismatch_count':int((quantized>0).sum()),'psnr_delta_db':psnr,'ssim_delta':ssim,
            'primary_metrics':r['metrics'],'control_metrics':c['metrics'],
            'primary_historical_comparison':r['historical_comparison'],'control_historical_comparison':c['historical_comparison'],
            'auxiliary_outputs_identical':r['auxiliary_outputs']==c['auxiliary_outputs'],'output_hashes':hashes,
            'limits':['Same active step branches expected by source review; native determinism is not assumed.',
                      'Without an independent same-step repeat, a nonzero pair difference is not causal evidence of step sensitivity.',
                      'No historical producing commit/split conclusion follows from metric or JPEG agreement.']}
