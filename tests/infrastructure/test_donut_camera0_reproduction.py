"""Small CPU checks for16-bit metrics and the single-render execution gate."""
import ast
import json
from pathlib import Path
import numpy as np
import pytest

from experiments.donut_camera0_reproduction import comparison,authorize,LIMITS,PROTOCOL


def test_one_16bit_LSB_is_preserved_and_scored():
    ref=np.full((32,32,3),32000,np.uint16);pr=ref+np.uint16(1);mask=np.ones((32,32),bool)
    result,error,_=comparison(ref,pr,mask,mask)
    r=result['regions']['fixed_original_foreground']
    assert r['MSE']==pytest.approx((1/65535)**2,rel=1e-10)
    assert r['PSNR_dB']==pytest.approx(20*np.log10(65535),abs=1e-8)
    assert not r['exact_match'] and r['fraction_RGB_components_within_2_LSB']==1
    assert result['foreground_IoU']==1 and error.max()>0
    with pytest.raises(ValueError,match='16-bit'):comparison(ref.astype(np.uint8),pr.astype(np.uint8),mask,mask)


def test_foreground_disagreement_is_reported_without_registration():
    ref=np.full((32,32,3),32000,np.uint16);pr=ref.copy();fixed=np.ones((32,32),bool);render=fixed.copy();render[:2]=False
    r,_,_=comparison(ref,pr,fixed,render)
    assert r['original_only_pixels']==64 and r['render_only_pixels']==0
    assert r['foreground_IoU']==pytest.approx(30/32)
    assert r['regions']['common_foreground']['pixels']==960
    assert r['regions']['common_SSIM_window_interior']['pixels']<960


def test_authorization_binds_source_freeze_output_and_work(tmp_path):
    p=tmp_path/'a.json';f=dict(output_directory='/explicit/fresh')
    value=dict(approved=True,protocol=PROTOCOL,commit='c'*40,freeze_sha256='f'*64,output_directory=f['output_directory'],limits=LIMITS)
    p.write_text(json.dumps(value));assert authorize(p,f,'f'*64,'c'*40)==value
    for key,new in (('approved',False),('commit','d'*40),('freeze_sha256','e'*64),('output_directory','/other'),('limits',{})):
        changed=dict(value);changed[key]=new;p.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='authorization'):authorize(p,f,'f'*64,'c'*40)


def test_only_one_render_call_and_launcher_explicitly_executes_it():
    root=Path(__file__).resolve().parents[2]
    script=(root/'experiments/render_donut_camera0.py').read_text();tree=ast.parse(script)
    def dotted(n):
        if isinstance(n,ast.Name):return n.id
        if isinstance(n,ast.Attribute):return dotted(n.value)+'.'+n.attr
        return ''
    renders=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and dotted(n.func)=='bpy.ops.render.render']
    assert len(renders)==1
    assert 'if not args.execute:' in script and script.index('if not args.execute:')<script.index('returned=bpy.ops.render.render')
    launch=(root/'experiments/donut_camera0_reproduction.py').read_text()
    assert "'--','--execute'" in launch and "receipt['successful_render_returns']==1" in launch
    assert 'save_mainfile' not in script and 'save_as_mainfile' not in script
