#!/usr/bin/env python3
"""Authorized sidecar compilation, native BVH microtest, or offline comparison."""
import argparse,importlib.util,json,os,platform,re,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from infrastructure.ray_backend_e11c import (BASE,ORIGINAL_BINARY,SOURCE,LIMITS,TOL,authorize,freeze_check,
                                            file_record,json_new,compare,selected_binary)
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT,digest,baseline,PINS


def guarded(a):
    freeze=freeze_check(a.freeze,a.freeze_sha256)
    authorize(a.authorization,a.authorization_sha256,a.freeze_sha256,a.commit,a.host_root)
    for p in (Path('/'),ROOT,Path('/inputs'),a.authorization):
        require(os.statvfs(str(p)).f_flag & os.ST_RDONLY,'Read-only root/source/input/authorization required')
    require(os.geteuid()!=0 and {l.split(':',1)[0].strip() for l in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'Non-root/no network required')
    actual={p.relative_to('/inputs').as_posix() for p in Path('/inputs').rglob('*') if p.is_file()}
    require(actual==set(freeze['inputs']),'Unexpected or missing staged input files')
    require(not any(p.is_symlink() for p in Path('/inputs').rglob('*')),'Staged source symlinks forbidden')
    for name,sha in freeze['inputs'].items():require(digest(Path('/inputs')/name)==sha,'Staged input hash mismatch: '+name)
    return freeze


def build():
    import torch
    require(platform.python_version()=='3.8.10' and torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and
            not torch._C._GLIBCXX_USE_CXX11_ABI,'Reference Python/Torch/CUDA/ABI required')
    require(os.environ.get('TORCH_CUDA_ARCH_LIST')=='8.0' and os.environ.get('MAX_JOBS')=='4' and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='','GPU-free fixed architecture/build concurrency required')
    macros=Path('/usr/include/eigen3/Eigen/src/Core/util/Macros.h').read_text()
    eigen='.'.join(re.search(r'^#define EIGEN_'+part+r'_VERSION\s+(\d+)',macros,re.M).group(1) for part in ['WORLD','MAJOR','MINOR'])
    require(eigen=='3.3.7','Pinned installed Eigen required; no download fallback')
    manifest=json.loads(Path('/inputs/source-manifest.json').read_text());require(manifest['upstream_commit']==SOURCE,'Wrong upstream source')
    for n,sha in manifest['patched'].items():require(digest(Path('/inputs/source')/n)==sha,'Build source mismatch')
    original=Path(importlib.util.find_spec('_raytracing').origin);require(digest(original)==ORIGINAL_BINARY,'Historical backend differs')
    tools={name:subprocess.check_output(cmd,text=True).strip() for name,cmd in [
        ('gcc',['gcc','-dumpfullversion']),('g++',['g++','-dumpfullversion']),('nvcc',['nvcc','--version'])]}
    require(tools['gcc']==tools['g++']=='9.4.0' and 'V11.1.105' in tools['nvcc'],'Pinned compiler versions required')
    command=[sys.executable,'-B','/inputs/source/setup.py','build_ext','--build-lib','/outputs/corrected','--build-temp','/outputs/build']
    started=time.monotonic()
    with Path('/outputs/compile.log').open('x') as log:
        result=subprocess.run(command,cwd='/outputs',stdout=log,stderr=subprocess.STDOUT)
    require(result.returncode==0,'Native compilation failed; see compile.log (no retry)')
    binaries=list(Path('/outputs/corrected').glob('_raytracing*.so'));require(len(binaries)==1,'One sidecar binary required')
    for n,sha in manifest['patched'].items():require(digest(Path('/inputs/source')/n)==sha,'Source changed during build')
    require(digest(original)==ORIGINAL_BINARY,'Original extension changed during derivative build')
    return dict(status='complete',upstream_commit=SOURCE,source_manifest=manifest,command=command,elapsed_seconds=time.monotonic()-started,
        python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cxx11_abi=0,architecture='8.0',eigen=eigen,toolchain=tools,
        original_binary=file_record(original),corrected_binary=file_record(binaries[0]),compile_log=file_record(Path('/outputs/compile.log')),
        compiler_binaries={name:file_record(Path(__import__('shutil').which(name)).resolve()) for name in ('gcc','g++','nvcc')},
        eigen_macros=file_record(Path('/usr/include/eigen3/Eigen/src/Core/util/Macros.h')),
        build_files=[file_record(p) for p in sorted(Path('/outputs/build').rglob('*')) if p.is_file() and p.name in ('build.ninja','.ninja_log')],
        native_cuda_execution=False)


def probe(a):
    import torch
    from benchmarks.bell_geometry_e11 import mesh_arrays
    from benchmarks.generate_bell_e11 import run_with_cleanup
    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and torch.cuda.device_count()==1 and
            'A100' in torch.cuda.get_device_name(0),'Reference stack/one selected A100 required')
    from benchmarks.bell_e11 import MESH
    require(digest('/mesh.ply')==MESH,'Immutable Bell mesh mismatch')
    build_record=json.loads(Path('/opt/e11c/build.json').read_text());binary=selected_binary(build_record,a.backend)
    require('_raytracing' not in sys.modules,'Fresh extension namespace required')
    spec=importlib.util.spec_from_file_location('_raytracing',binary);extension=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension);sys.modules['_raytracing']=extension
    science,_=baseline('material',PINS['material']);wrapper_path=science/'raytracing/raytracer.py'
    spec=importlib.util.spec_from_file_location('_e11c_wrapper',wrapper_path);wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    refs=json.loads(Path('/inputs/references.json').read_text())
    with np.load('/inputs/panels.npz',allow_pickle=False) as z:panel={k:z[k] for k in z.files}
    def work():
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(LIMITS['cuda_allocator_fraction'])
        vertices,faces=mesh_arrays('/mesh.ply');tick=time.perf_counter();tracer=wrapper.RayTracer(vertices,faces);torch.cuda.synchronize()
        build_times={'bell_seconds':time.perf_counter()-tick};tick=time.perf_counter()
        cube_tracer=wrapper.RayTracer(panel['cube_v'],panel['cube_f']);torch.cuda.synchronize();build_times['cube_seconds']=time.perf_counter()-tick
        counts={'attempted_rays':0,'returned_rays':0,'attempted_calls':0,'returned_calls':0}
        def query(t,o,d):
            count=len(o);require(counts['attempted_rays']+count<=LIMITS['max_native_ray_queries']//2,'Per-backend query budget exceeded')
            counts['attempted_rays']+=count;counts['attempted_calls']+=1
            value=t.trace(o,d);counts['returned_rays']+=count;counts['returned_calls']+=1;return value
        results={}
        with torch.no_grad():
            for name,t in [('bell',tracer),('cube',cube_tracer)]:
                o,d=[torch.from_numpy(panel[name+'_'+key]).cuda() for key in ('o','d')]
                pos,raw,depth=query(t,o,d);torch.cuda.synchronize()
                pos,raw,depth=[x.cpu().numpy().copy() for x in (pos,raw,depth)]
                with Path('/outputs/'+name+'.npz').open('xb') as f:np.savez(f,positions=pos,raw_normals=raw,depth=depth)
                rows=compare(panel[name+'_o'],panel[name+'_d'],pos,raw,depth,refs[name]);results[name]=rows
            o,d=[torch.from_numpy(panel['perf_'+key]).cuda() for key in ('o','d')]
            for _ in range(LIMITS['warmups']):query(tracer,o,d)
            torch.cuda.synchronize();wall=[];gpu=[]
            for _ in range(LIMITS['timed_repeats']):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                tick=time.perf_counter();begin.record();pos,raw,depth=query(tracer,o,d);end.record();torch.cuda.synchronize()
                wall.append((time.perf_counter()-tick)*1000);gpu.append(begin.elapsed_time(end))
            with Path('/outputs/performance-last.npz').open('xb') as f:np.savez_compressed(f,positions=pos.cpu().numpy(),raw_normals=raw.cpu().numpy(),depth=depth.cpu().numpy())
        k=refs['known_failure_index'];target=results['bell'][k]
        baseline_reproduced=target['hit'] and abs(target['depth']-refs['saved_failure_depth'])<=TOL['depth_abs']
        passed=(all(x['passed'] for rows in results.values() for x in rows) if a.backend=='predicate_fp64' else baseline_reproduced)
        require(counts['returned_rays']==counts['attempted_rays']==LIMITS['max_native_ray_queries']//2,'Incomplete native query work')
        return dict(status='pass' if passed else 'fail',backend=a.backend,binary=file_record(binary),cases=results,counts=counts,
            all_cases_match_independent_reference=all(x['passed'] for rows in results.values() for x in rows),
            acceptance_role='all CPU references' if a.backend=='predicate_fp64' else 'historical failure reproduction (not reference correctness)',
            baseline_reproduced=bool(baseline_reproduced),bvh_construction=build_times,primitive_index_interface='Extension returns no primitive ID; original PLY reference identity checked through closest depth/position/raw-normal agreement',
            target_reference_face=refs['bell'][k]['face'],performance=dict(batch_size=len(o),warmups=LIMITS['warmups'],wall_ms=wall,cuda_event_ms=gpu),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(0),gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
            driver=Path('/proc/driver/nvidia/version').read_text() if Path('/proc/driver/nvidia/version').exists() else 'unavailable',torch=torch.__version__,cuda=torch.version.cuda,wrapper=file_record(wrapper_path))
    result=run_with_cleanup(work,torch);result['cleanup_verified']=True;return result


def analyze(root):
    # Versioned file-only replay; native compare/probe/build remain unchanged.
    from infrastructure.ray_backend_acceptance_e11c import analyze as offline_analyze
    return offline_analyze(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['build','probe','analyze'],required=True)
    p.add_argument('--backend',choices=['original','predicate_fp64']);p.add_argument('--root',type=Path)
    p.add_argument('--analysis-output',type=Path,help='New offline report path; never overwrites prior analysis')
    p.add_argument('--freeze',type=Path);p.add_argument('--freeze-sha256');p.add_argument('--authorization',type=Path)
    p.add_argument('--authorization-sha256');p.add_argument('--commit');p.add_argument('--host-root',type=Path);a=p.parse_args()
    if a.mode=='analyze':
        result=analyze(a.root);json_new(a.analysis_output or a.root/'comparison.json',result);return 0 if result['status']=='PROCEED' else 1
    frozen=guarded(a);started=time.monotonic()
    try:
        result=build() if a.mode=='build' else probe(a)
        freeze_check(a.freeze,a.freeze_sha256);result.update(commit=a.commit,freeze_sha256=a.freeze_sha256,elapsed_wall_seconds=time.monotonic()-started)
        require(sum(p.stat().st_size for p in Path('/outputs').rglob('*') if p.is_file())<=LIMITS['output_bytes'],'Output limit exceeded')
        result['input_sha256']=frozen['inputs']
        result['artifacts']=[file_record(p) for p in sorted(Path('/outputs').glob('*.npz'))]
        report=Path('/outputs/build.json' if a.mode=='build' else '/outputs/probe.json')
        json_new(report,result)
        json_new('/outputs/completion.json',dict(status=result['status'],report_sha256=digest(report)))
        return 0 if result['status'] in ('pass','complete') else 1
    except BaseException as exc:
        json_new('/outputs/failure.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc(),elapsed_wall_seconds=time.monotonic()-started));raise
if __name__=='__main__':sys.exit(main())
