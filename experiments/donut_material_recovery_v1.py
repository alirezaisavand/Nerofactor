"""Two separately authorized full-field arms; linear oracle observations only.

Frozen core owns BRDF, sampling, networks and existing material priors. This
small observation adapter deliberately does not invoke the historical loader,
tone-mapped RGB loss inputs, learned indirect network or automatic NVS.
"""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np

from benchmarks.donut_recovery_v1 import (ABLATION_ARMS, TRAINING, TRAIN_LIMITS as LIMITS, SHADER,
    POLICY, PANEL, SPLITS, SIZE, COLOR, METALLIC, WIDTHS, inputs, numeric, unit,
    physical_errors, surface_frames, runtime, verify_freeze, json_new)
from benchmarks.donut_tensor_d8 import ARMS as D8_ARMS
from benchmarks.donut_tensor_d9 import ARMS as D9_ARMS
from benchmarks import donut_log_tensor_d11 as d11
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest


def illumination_preview(path, light):
    """Display-only upper clipping; scientific HDR arrays are never modified."""
    from benchmarks.bell_e11 import linear_to_srgb
    from PIL import Image
    require(light.ndim==3 and light.shape[-1]==3 and light.size>0 and
        np.isfinite(light).all() and (light>=0).all(), 'Invalid illumination preview input')
    above=light>1
    Image.fromarray(linear_to_srgb(np.minimum(light,1))).save(path)
    return dict(encoding='standard-sRGB uint8',exposure=1.0,
        display_only_upper_clip=1.0,channels_above_one=int(above.sum()),
        pixels_with_channel_above_one=int(above.any(-1).sum()),
        scientific_linear_values='unchanged in illumination.npz and illumination metrics')


def shader_lighting(shader, original_shader, arm, calibrated=None):
    """Use the native direct light, with the D3 calibration in D3 and D4.

    Both arms retain the identical validated zero_bounce_v1 configuration.
    No modification to corrected-core, network capacity or BRDF is required.
    """
    require(arm in ('D1','D2','D3','D6','D7',*D8_ARMS,*D9_ARMS,*d11.ARMS,*ABLATION_ARMS) and shader.cfg['indirect_mode']=='zero_bounce_v1',
            'Matched direct-only acquisition required')
    if arm=='D1':return nullcontext()
    if arm in ('D3','D4'):
        require(calibrated is not None and shader.cfg['outer_light_version']=='direction',
                'Calibrated arm requires a calibrated directional environment')
        return shader.environment(calibrated, 'donut-D3-calibrated-directional-MLP-zero-indirect-v1')
    return shader.environment(lambda p,d:original_shader.predict_outer_lights(shader,p,d),
                              'donut-D2-native-directional-MLP-zero-indirect-v1')


def training_objective(photo, material_prior, diffuse_white, *, include_diffuse_white):
    loss = photo + material_prior
    return loss + diffuse_white if include_diffuse_white else loss


def tensor_pair(method, arms):
    """Select the other representation using the native method identity."""
    require(len(arms)==2 and len(set(arms))==2 and method in arms,
            'Two distinct tensor representations including the selected method are required')
    return method,next(arm for arm in arms if arm!=method)


def calibration_mean_record(grid_light, weights, definition):
    realized_mean = (grid_light * weights[..., None]).sum(axis=(0, 1))
    target = definition['mean_linear_RGB']
    mean_error = abs(realized_mean - target)
    require(np.isfinite(grid_light).all() and (grid_light >= 0).all() and
            (mean_error <= definition['absolute_mean_tolerance']).all(),
            'Exported environment violates the calibrated grid mean')
    return dict(target_mean_RGB=target, realized_mean_RGB=realized_mean.tolist(),
                absolute_error=mean_error.tolist())


def benchmark(root, expected):
    require(root.resolve()==root and digest(root/'generation.json')==expected,'Explicit admitted generation hash required')
    g=json.loads((root/'generation.json').read_text())
    from benchmarks.donut_recovery_v1 import REFERENCE, ENVIRONMENT
    from infrastructure.verified_ray_backend import identity
    require(g['status']=='complete_pending_researcher_admission' and g['protocol'] in (REFERENCE,'donut-kurt-reference-v2') and
        g['cleanup_verified'] and g['splits']==SPLITS and g['environment']==ENVIRONMENT and
        g['backend']==identity() and g['material']['color']==COLOR and g['material']['widths_major_minor']==WIDTHS and
        g['material']['metallic']==METALLIC and g['BRDF']=='Kurt-a0-metallic-additive-v1' and
        g['indirect']=='zero' and g['point_light']=='absent',
        'Incomplete/foreign reference dataset')
    for name,item in g['outputs'].items():
        p=root/name
        require(p.resolve()==p and root in p.parents and p.stat().st_size==item['bytes'] and
                digest(p)==item['sha256'],'Reference artifact changed: '+name)
    require(len(g['views'])==6 and all(r['radiance_convergence']['passed'] and r['energy_convergence']['passed'] for r in g['views']),
            'Reference scientific gates incomplete')
    if g['protocol']=='donut-kurt-reference-v2':
        require(g.get('reflectance_orders')==[64,128] and
            g.get('validation',{}).get('grazing',{}).get('slope_two_grid',{}).get('passed') is True and
            all(g.get('validation',{}).get('ordinary',{}).get(key,{}).get('passed') is True
                for key in ('slope_two_grid','old_incident_two_grid','slope_vs_independent_incident')) and
            g.get('camera0_reuse',{}).get('array_equality_verified') is True and
            g['camera0_reuse'].get('new_camera0_primary_rays')==g['camera0_reuse'].get('new_camera0_RGB_integrations')==0,
            'V2 slope validation/camera0 reuse evidence incomplete')
    return g,{camera:numeric(root/('view-%d.npz'%camera)) for camera in PANEL}


def train(args, *, heterogeneous=False, acquisition=None):
    heterogeneous = heterogeneous or acquisition is not None
    faceted = acquisition is not None and getattr(acquisition,'FACETED',True)
    if heterogeneous:
        if acquisition is None:
            from benchmarks import donut_heterogeneous_training_v1 as hetero
        else:
            hetero = acquisition
        freeze,auth=hetero.runtime(args)
    else:
        freeze,auth=runtime(args,args.phase)
    import os
    ablation=freeze.get('diffuse_white_ablation') if args.phase in ABLATION_ARMS else None
    chart=freeze.get('surface_chart_training') if args.phase=='D7' else None
    tensor_plan=freeze.get('tensor_training')
    log_tensor=args.phase in d11.ARMS
    spatial=freeze.get('surface_q_training') or chart or tensor_plan
    parity_plan=chart or tensor_plan or (spatial if heterogeneous else None)
    spatial_name='surface_log_q' if log_tensor else ('surface_chart' if chart else 'surface_q')
    bound=spatial or ablation or (freeze.get('calibrated_training') if args.phase=='D3' else None)
    if heterogeneous:
        bound=freeze['arm_training']
    weighting = spatial.get('distance_weighting') if spatial else None
    robust_delta = spatial.get('robust_delta') if spatial else None
    method = bound.get('learned_control',args.phase) if bound else args.phase
    known_l2 = bound is not None and bound.get('illumination')=='known_L2'
    lighting_arm = 'D1' if known_l2 else ('D6' if weighting is not None or robust_delta is not None else method)
    calibration=bound['calibration'] if args.phase in ('D3','D4') else None
    limits=ablation['limits'][args.phase] if ablation else (bound['limits'] if bound else LIMITS)
    protocol=bound['protocol'] if bound else TRAINING
    include_diffuse_white=ablation is None
    probe_diffuse_white=known_l2 or args.phase in ('D3',*ABLATION_ARMS)
    require(auth.get('researcher_admitted_dataset') is True and auth.get('benchmark_generation_sha256')==args.benchmark_sha256
        and os.statvfs(args.benchmark).f_flag&os.ST_RDONLY, 'Separate explicit dataset admission/training approval required')
    if heterogeneous:
        generation,data=hetero.dataset(freeze,args.benchmark,args.admission,args.verification)
    else:
        generation,data=benchmark(args.benchmark,args.benchmark_sha256)
    require(generation['freeze_sha256']==freeze.get('reference_freeze_sha256',args.freeze_sha256),
            'Reference does not match the explicitly bound generation freeze')
    panel, splits = (PANEL, SPLITS) if acquisition is None else (acquisition.PANEL, acquisition.SPLITS)

    def source_geometry():
        return acquisition.geometry(freeze,args.oracle) if faceted else inputs(args.oracle,args.cameras)

    geometry,_,records=source_geometry()
    expected_initial=None
    if known_l2 or lighting_arm in ('D2','D3','D6','D7',*D8_ARMS,*D9_ARMS,*d11.ARMS,*ABLATION_ARMS):
        require(args.initial is not None and digest(args.initial)==args.initial_sha256 and
            os.statvfs(args.initial).f_flag&os.ST_RDONLY,'Explicit immutable initialization anchor required')
        if bound:
            require(args.initial_sha256==bound['initial_sha256'] and
                    args.benchmark_sha256==bound['benchmark_generation_sha256'],
                    'Initialization or dataset differs from the freeze')
        expected_initial=json.loads(args.initial.read_text())
    if log_tensor:
        initial_controls=Path('/inputs/d11-initial')
        require(all(os.statvfs(initial_controls/name).f_flag & os.ST_RDONLY for name in d11.CONTROL_FILES),
                'Immutable D8-T initialization controls required')
        d11.verify_initial_controls(initial_controls, tensor_plan['D8_T_initial_controls'])
    args.output.mkdir();(args.output/'runtime/cache').mkdir(parents=True);started=time.monotonic()
    report=dict(schema_version=1,protocol=protocol,status='incomplete',arm=args.phase,
        commit=args.commit,freeze_sha256=args.freeze_sha256,benchmark_sha256=args.benchmark_sha256,
        input_identities=records,optimizer_returns=0,cleanup_verified=False,
        effective_shader=deepcopy(SHADER),policy=POLICY,saved_endpoints=[],
        effective_lighting=('known analytic L2' if lighting_arm=='D1' else 'original A directional outer_light MLP'),
        indirect_illumination='identically zero in D1/D2',research_G_handoff_admitted=False,
        counters=dict(shader_calls=0,shadow_rays=0,component_reverse_sweeps=0))
    if bound:
        report.update(indirect_illumination='identically zero', initialization_anchor_sha256=args.initial_sha256)
    known_audit = None
    light_handles = ExitStack()
    if known_l2:
        from infrastructure.known_l2_audit import KnownL2Audit
        known_audit = KnownL2Audit()
        report.update(learned_control=method,diffuse_white_in_objective=True)

    def light_path(name, dispatches=1):
        return known_audit.path(name,dispatches) if known_audit else nullcontext()
    if heterogeneous:
        report.update(heterogeneous_dataset=True,authorization_sha256=args.authorization_sha256,
                      dataset_admission_sha256=hetero.ADMISSION)
    if faceted:
        report['counters'].update(boundary_predictor_calls=0, boundary_predictor_points=0)
    if ablation:
        report.update(diffuse_white_in_objective=False, diffuse_white_diagnostic_coefficient=POLICY['diffuse_white_prior'],
                      optimization_terms=['RGB_loss','material_prior'], mean_calibration=calibration is not None)
    if spatial:
        spatial_policy=spatial['spatial_policy']
        report.update(diffuse_white_in_objective=True, mean_calibration=False, surface_q_policy=spatial_policy,
                      optimization_terms=['RGB_loss','material_prior','diffuse_white_prior',spatial_name+'_loss'],
                      surface_q_timing=dict(setup_wall_seconds=0., forward_CUDA_seconds=0.))
        report['counters'].update(surface_q_batches=0,surface_q_pairs=0,
                                  surface_q_predictor_calls=0,surface_q_predictor_points=0)
        if weighting is not None:
            report['surface_q_distance_weighting'] = weighting
            report['counters']['edge_diagnostic_batches'] = 0
        if robust_delta is not None:
            report['surface_q_penalty'] = dict(name='pseudo_huber_v1',delta=robust_delta)
            report['counters']['robust_diagnostic_batches'] = 0
    if tensor_plan:
        report.update(representation=tensor_plan['representation'],
                      effective_regularization='legacy_scalar_energy_specular_white_plus_surface_Q_v1',
                      tensor_training_forward_CUDA_seconds=0.)
    if log_tensor:
        report.update(effective_regularization='legacy_scalar_energy_specular_white_plus_surface_log_Q_v1',
                      spatial_regularizer=d11.LOSS_ID,coefficient_sha256=tensor_plan['coefficient_sha256'])
    if calibration:
        report.update(effective_lighting='original A directional outer_light MLP with fixed spherical mean RGB',
                      calibration=calibration)
        report['counters'].update(calibration_evaluations=0, calibration_directions=0)
        timing = report['calibration_timing'] = {
            stage: dict(setup_wall_seconds=0., grid_forward_and_reduction_CUDA_seconds=0.)
            for stage in ('training','evaluation')
        }
    def charge(key,count):
        require(report['counters'][key]+count<=limits[key], 'Native work cap: '+key)
        report['counters'][key]+=count
    def budget(key=None,count=0):
        require(time.monotonic()-started<limits['inner_seconds'],'Authorized arm runtime reached; preserve partials')
        if key: charge(key,count)
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<limits['output_bytes'],'Owned output cap reached')
    def progress():
        p=args.output/'progress.next';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(args.output/'progress.json')
    try:
        import torch
        import random
        from infrastructure.verified_ray_backend import load
        from benchmarks.generate_bell_e11 import run_with_cleanup
        from benchmarks.neural_observation_e16 import compatibility
        from experiments.bell_neural_material_e16 import state_hash
        from infrastructure.material_recovery_e13_evaluation import native_tensor_default
        from infrastructure.sphere_replay import evaluation_scope
        from infrastructure.sphere_visuals import write_view
        from infrastructure.surface_q_regularization import (surface_pairs,world_q,normal_transport,
            gradient_diagnostics,legacy_energy_specular)
        require(torch.__version__=='1.8.1+cu111' and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Reviewed native runtime required')
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
        report['runtime']=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
            CXX11_ABI=bool(torch._C._GLIBCXX_USE_CXX11_ABI),numpy=np.__version__,declared_image=args.image_id)
        report['backend']=load('predicate_fp64',args.image_id);activate('material',PINS['material'])
        import raytracing
        from network import field,renderer
        from train.lr_common_manager import WarmUpCosLR
        from scientific_variants.material_a_v1 import native_classes,orientation
        tracer=raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),geometry['triangles'].astype(np.int32))
        def trace(o,d):
            p,n,depth=tracer.trace(o,d)
            return p,n,depth.reshape(-1,1),depth.reshape(-1)<10
        random.seed(POLICY['seed']);np.random.seed(POLICY['seed'])
        torch.manual_seed(POLICY['seed']);torch.cuda.manual_seed_all(POLICY['seed'])
        shader_class,_=native_classes(field.MCShadingNetwork,renderer.NeROMaterialRenderer,
                                      'A_corrected_core_v1',field.linear_to_srgb)
        if tensor_plan:
            from scientific_variants import donut_tensor_d8 as tensor_math
            from scientific_variants.donut_tensor_shader_d8 import tensor_shader, paired_initial_gate
            shader_class=tensor_shader(shader_class,field.linear_to_srgb)
            if args.phase in D9_ARMS:
                from scientific_variants.donut_tensor_initialization_d9 import weak_start_shader, paired_initial_gate
                shader_class=weak_start_shader(shader_class)
        shader=shader_class(deepcopy(SHADER),trace).cuda()
        initial=dict(state_sha256=state_hash(shader),variant='A_corrected_core_v1',seed=POLICY['seed'])
        require(expected_initial is None or initial==expected_initial,'Full initial weights/buffers differ from the admitted anchor')
        sibling=None
        if tensor_plan:
            representation,counterpart=('D8-T','D8-C') if log_tensor else tensor_pair(method,tensor_plan['arms'])
            with torch.random.fork_rng(devices=[0]):
                shader.select_tensor_representation(representation)
                shader.cuda()
                torch.manual_seed(POLICY['seed']);torch.cuda.manual_seed_all(POLICY['seed'])
                sibling=shader_class(deepcopy(SHADER),trace).cuda()
                require(state_hash(sibling)==initial['state_sha256'],'Paired fresh base state differs')
                sibling.select_tensor_representation(counterpart)
                sibling.cuda()
            require((shader.tensor_arm,sibling.tensor_arm)==(representation,counterpart),
                    'Native tensor pair differs from its declared representations')
            initial=dict(common_base_state_sha256=initial['state_sha256'],state_sha256=state_hash(shader),
                         variant=shader.tensor_variant,seed=POLICY['seed'])
        if log_tensor:
            require(initial==json.loads((initial_controls/'initial.json').read_text()),
                    'Fresh D11 weights or buffers differ from accepted D8-T')
            report['D8_T_initial_state_verified']=True
        if heterogeneous:
            require(initial==json.loads(args.arm_initial.read_text()),
                    'Fresh arm weights or buffers differ from its accepted historical initialization')
            report['historical_arm_initial_parity_verified']=True
        json_new(args.output/'initial.json',initial);report['initial']=initial
        material_names=('feats_network','mx_predictor','my_predictor','rotation_predictor','kd_predictor','metallic_predictor')
        for name,p in shader.named_parameters():
            p.requires_grad_(name.split('.')[0] in material_names or (lighting_arm!='D1' and name.startswith('outer_light.')))
        report['trainable_parameters']={name:sum(p.numel() for n,p in shader.named_parameters() if p.requires_grad and n.startswith(name+'.'))
            for name in (*material_names,'outer_light','inner_light')}
        require(all(report['trainable_parameters'][name]>0 for name in material_names),'Full native material field required')
        expected_count=1485584+((2*(256+2)) if method in ('D8-T','D9-T',*d11.ARMS) else 0)
        require(sum(report['trainable_parameters'][name] for name in material_names)==expected_count,
                'Full material capacity differs from the frozen inventory')
        if known_audit:
            light_handles.enter_context(known_audit.attached(shader))
            lighting_anchor = known_audit.lighting_hash(shader)
        def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
        calibrated = None
        if calibration:
            from benchmarks.donut_calibration_d3 import CalibratedEnvironment, quadrature
            calibration_dirs, calibration_weights = quadrature()
            calibrated = CalibratedEnvironment(
                lambda points, directions: field.MCShadingNetwork.predict_outer_lights(shader, points, directions),
                tensor(calibration_dirs.reshape(-1,3)), tensor(calibration_weights.reshape(-1,1)),
                tensor(calibration['mean_linear_RGB']), torch)

        def calibration_scope(evaluation):
            if calibrated is None:
                return nullcontext()
            return calibrated.scope(budget, timing, torch, evaluation=evaluation)

        if tensor_plan:
            tensor_start=torch.cuda.Event(enable_timing=True)
            tensor_end=torch.cuda.Event(enable_timing=True)

        def shade(p,wo,n,step,training,seed=None,export=False):
            budget('shader_calls',1);budget('shadow_rays',len(p)*1536)
            with light_path('training' if training else 'evaluation'):
                if tensor_plan:
                    if training:
                        tensor_start.record()
                    tape=tensor_plan['representation']
                    tape_seed=tape['training_tape_seed_offset']+step if training else tape['evaluation_tape_seed_offset']+seed
                    o=shader(p,wo,n,None,step,training,tape_seed=tape_seed,export=export)
                    if training:
                        tensor_end.record()
                else:
                    _,o=shader(p,wo,n,None,step,training)
            require(all(bool(torch.isfinite(v).all()) for v in o.values() if isinstance(v,torch.Tensor)),'Nonfinite native output')
            if tensor_plan and training:
                report['tensor_training_forward_CUDA_seconds']+=tensor_start.elapsed_time(tensor_end)/1000
            return o
        # No training gradient or state update enters this independent parity gate.
        first=data[panel[0]];valid=np.flatnonzero(first['shader_valid'])
        ix=valid[np.linspace(0,len(valid)-1,24).astype(int)]
        p,n,wo=(tensor(first[k][ix]) for k in ('position','normals','outgoing'))
        T,B=shader.compute_tangent_bitangent_flat(n,None);axis=tensor(first['axis'][ix])
        angle=torch.atan2((axis*B).sum(-1),(axis*T).sum(-1))
        rotation=torch.stack([torch.cos(2*angle),torch.sin(2*angle)],-1)
        captures=[]
        with evaluation_scope(shader,torch,np),torch.no_grad(),native_tensor_default(torch),light_path('compatibility',32):
            for draw in range(32):
                torch.manual_seed(900000+draw);torch.cuda.manual_seed_all(900000+draw)
                budget('shader_calls',1);budget('shadow_rays',len(ix)*1536)
                if heterogeneous:
                    mx,my=tensor(first['widths_major_minor'][ix]).split(1,dim=1)
                    metallic,kd=(tensor(first[key][ix]) for key in ('metallic','color'))
                else:
                    mx,my=(n.new_full((24,1),v) for v in WIDTHS)
                    metallic,kd=n.new_full((24,1),METALLIC),n.new_tensor(COLOR).expand(24,3)
                if tensor_plan:
                    u=tensor_math.frame(n)
                    q=tensor_math.chart_q(mx,my,axis,u)
                    o=shader.shade_tensor(p,n,wo,q,u,kd,metallic,900000+draw,False)
                else:
                    _,o=shader.shade_anisotropic_mixed(p,n,wo,mx,my,metallic,kd,rotation,(rotation+1)/2,None,False)
                captures.append(o['linear_rgb'].cpu().numpy())
        reference=first['linear_components'][ix,2];coarse=first['coarse_components'][ix,2]
        gate=compatibility(np.stack(captures),reference,abs(reference-coarse),POLICY['compatibility_absolute'],POLICY['compatibility_relative'])
        np.savez_compressed(args.output/'native-reference-gate.npz',native_draws=np.stack(captures),reference=reference,coarse=coarse,pixel_indices=first['pixel_indices'][ix])
        report['native_reference_gate']=gate
        if bound:
            report['native_reference_gate_lighting']='known L2, same compatibility control as D1/D2'
        progress()
        require(gate['passed'],'Native/reference compatibility or precision gate failed before optimization')
        if heterogeneous:
            pool=hetero.fit_pool(data)
        else:
            pool={key:np.concatenate([data[i][key][data[i]['shader_valid']] for i in SPLITS['fitting']])
                  for key in ('position','normals','outgoing','linear_components')}
        require(len(pool['position'])>=512,'Insufficient original fitting observations')
        sample_rng=np.random.RandomState(POLICY['seed']);order=sample_rng.permutation(len(pool['position']));cursor=0
        schedule=WarmUpCosLR(dict(end_warm=1000,end_iter=100000))
        optimizer_source = SimpleNamespace(parameters=lambda: (p for p in shader.parameters() if p.requires_grad)) if known_l2 else shader
        optimizer=schedule.construct_optimizer(torch.optim.Adam,optimizer_source)
        # The historical diagnostic edges remain disjoint from D6 training edges.
        if not faceted:
            pairs=surface_pairs(geometry['vertices_native'],geometry['triangles'])
            pair_ix=np.linspace(0,len(pairs['points0'])-1,min(1024,len(pairs['points0']))).astype(int)
        else:
            pairs,pair_ix,boundary = acquisition.diagnostics(geometry, freeze)
        pair_points=[pairs[k][pair_ix] for k in ('points0','points1')]
        if not faceted:
            pair_frames=[surface_frames(geometry,v) for v in pair_points]
        else:
            pair_frames=acquisition.diagnostic_frames(geometry,pairs,pair_ix)
            np.savez_compressed(args.output/'boundary-probes.npz',**boundary)
        if heterogeneous and not faceted:
            from benchmarks.donut_heterogeneous_v1 import material_field
            for pp,ff in zip(pair_points,pair_frames):
                ff['Q_world']=material_field(pp,ff['normals'],generation['ring_radius_native'])['Q_world']
        if spatial:
            from benchmarks.donut_surface_q_d6 import (
                NeighborhoodSampler, coverage_summary, fitting_face_counts, neighborhoods, surface_q_term,
            )
            spatial_term=surface_q_term
            if parity_plan:
                from benchmarks.donut_surface_chart_d7 import array_hash
            if chart:
                from benchmarks.donut_surface_chart_d7 import surface_chart_term
                spatial_term=surface_chart_term
            elif log_tensor:
                from scientific_variants.donut_log_tensor_d11 import surface_log_q_term, world_log_tensor
                spatial_term=surface_log_q_term
            elif tensor_plan:
                from scientific_variants.donut_tensor_shader_d8 import tensor_surface_term
                spatial_term=tensor_surface_term
            setup_started=time.monotonic()
            domain=pairs if faceted else neighborhoods(geometry,spatial_policy)
            require(len(domain['face_area'])==spatial_policy['expected_mesh_faces'] and
                    len(domain['face_pairs'])==spatial_policy['expected_pairs'] and
                    np.array_equal(domain['face_pairs'],pairs['face_pairs']) and
                    np.array_equal(domain['diagnostic_ids'],pair_ix), 'Surface geometry or diagnostic identity differs')
            sampler=NeighborhoodSampler(domain['training_ids'],len(domain['face_pairs']),spatial_policy['rng_seed'])
            fitting_counts=fitting_face_counts(data,splits['fitting'],len(domain['face_area']))
            if parity_plan:
                hashes={k:array_hash(v) for k,v in dict(domain,fitting_camera_count=fitting_counts).items()}
                expected=parity_plan['D6_domain_array_sha256']
                matches = all(hashes[key]==value for key,value in expected.items()) if faceted else hashes==expected
                require(matches, 'Frozen neighborhood domain differs')
                report['D6_domain_parity_verified']=True
                if acquisition is not None:
                    report['surface_q_domain_hashes']=hashes
            surface_tensors={k:tensor(domain[k]) for k in ('points0','points1','normals0','normals1','distance')}
            np.savez_compressed(args.output/'surface-q-domain.npz',**domain,fitting_camera_count=fitting_counts)
            spatial_start=torch.cuda.Event(enable_timing=True)
            spatial_end=torch.cuda.Event(enable_timing=True)
            report['surface_q_timing']['setup_wall_seconds']=time.monotonic()-setup_started
            report['surface_q_domain']=dict(training_pairs=len(domain['training_ids']),diagnostic_pairs=len(pair_ix),
                mesh_faces=len(domain['face_area']),faces_without_fitting_samples=int((fitting_counts==0).sum()),
                fitting_proxy='a face has support when a valid sample from any fitting camera hits it')
        def evaluate(step):
            destination=args.output/('step-%05d'%step);destination.mkdir()
            metrics={};shader.eval()
            if known_audit:
                known_audit.checkpoint(step,shader,optimizer,lighting_anchor)
            with evaluation_scope(shader,torch,np),torch.no_grad(),native_tensor_default(torch),calibration_scope(True):
                if calibration:
                    grid_light = calibrated.grid_radiance().cpu().numpy().reshape(*calibration_dirs.shape)
                    calibration_record = calibration_mean_record(grid_light, calibration_weights, calibration)
                for camera,d in data.items():
                    indices=np.flatnonzero(d['shader_valid']);chunks=[]
                    for start in range(0,len(indices),512):
                        ids=indices[start:start+512];pp,nn,vv=(tensor(d[k][ids]) for k in ('position','normals','outgoing'))
                        draws=[];last=None
                        for draw in range(8):
                            seed=800000+camera*1000+start+draw
                            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                            last=shade(pp,vv,nn,step,False,seed,export=draw==7)
                            draws.append(np.stack([last[k].cpu().numpy() for k in ('linear_rgb','linear_diffuse','linear_specular')],1))
                        # Deterministic field maps from the same returned forward, no extra material query.
                        chunks.append((ids,{k:v.cpu().numpy() for k,v in last.items() if k in
                            ('kd','mx','my','metallic','rotation','rotation_raw','tangents','bitangents','normals','Q')},np.mean(draws,axis=0),np.std(draws,axis=0,ddof=1)/np.sqrt(8)))
                    arrays={k:np.zeros((SIZE,SIZE,width),np.float32) for k,width in
                        dict(kd=3,mx=1,my=1,metallic=1,rotation=2,rotation_raw=2,tangents=3,bitangents=3,normals=3,
                             linear_rgb=3,linear_diffuse=3,linear_specular=3).items()}
                    if tensor_plan:
                        arrays['Q']=np.zeros((SIZE,SIZE,9),np.float64)
                    standard_error=np.zeros((SIZE*SIZE,3,3),np.float32)
                    for ids,physical,render_mean,se in chunks:
                        pixels=d['pixel_indices'][ids]
                        for key,value in physical.items():arrays[key].reshape(SIZE*SIZE,-1)[pixels]=value.reshape(len(ids),-1)
                        for k,key in enumerate(('linear_rgb','linear_diffuse','linear_specular')):arrays[key].reshape(-1,3)[pixels]=render_mean[:,k]
                        standard_error[pixels]=se
                    # Preserve geometric mask separately from one-sided BRDF domain.
                    mask=np.zeros(SIZE*SIZE,bool);mask[d['pixel_indices'][indices]]=True;mask=mask.reshape(SIZE,SIZE)
                    truth=np.zeros((SIZE*SIZE,3));truth[d['pixel_indices']]=d['linear_components'][:,2]
                    arrays.update(hit_mask=mask,rgb_gt=tensor(truth).reshape(SIZE,SIZE,3).cpu().numpy(),
                        rgb_pr=field.linear_to_srgb(tensor(arrays['linear_rgb'])).cpu().numpy(),
                        diffuse_color=field.linear_to_srgb(tensor(arrays['linear_diffuse'])).cpu().numpy(),
                        specular_color=field.linear_to_srgb(tensor(arrays['linear_specular'])).cpu().numpy(),
                        position=np.zeros((SIZE,SIZE,3),np.float32),normal=np.zeros((SIZE,SIZE,3),np.float32))
                    # write_view expects display GT; numerical linear GT remains separate.
                    arrays['linear_gt']=arrays['rgb_gt'].copy();arrays['rgb_gt']=field.linear_to_srgb(tensor(arrays['linear_gt'])).cpu().numpy()
                    arrays['position'].reshape(-1,3)[d['pixel_indices']]=d['position']
                    arrays['normal'].reshape(-1,3)[d['pixel_indices']]=d['normals']
                    view_identity=dict(arm=args.phase,completed_updates=step,camera_id=camera)
                    if tensor_plan:
                        direct=arrays.pop('Q').reshape(SIZE,SIZE,3,3)
                        arrays['Q_tensor_direct']=direct
                        view_identity.update(tensor_variant=initial['variant'],
                            physical_Q_array='Q_tensor_direct',
                            exported_axes='Sorted evaluation eigensystem of Q, with presentation angles in the A display chart')
                    write_view(destination/('camera-%d'%camera),arrays,view_identity,np)
                    point_pred={k:arrays[k].reshape(-1,arrays[k].shape[-1])[d['pixel_indices'][indices]] for k in ('kd','mx','my','metallic')}
                    t=2*arrays['tangents'].reshape(-1,3)[d['pixel_indices'][indices]]-1
                    b=2*arrays['bitangents'].reshape(-1,3)[d['pixel_indices'][indices]]-1
                    point_pred['axis']=np.where(point_pred['mx']>=point_pred['my'],t,b)
                    point_pred['Q_world']=point_pred['mx'][:,:,None]**2*t[:,:,None]*t[:,None,:]+point_pred['my'][:,:,None]**2*b[:,:,None]*b[:,None,:]
                    if tensor_plan:
                        require(np.max(abs(point_pred['Q_world']-direct[mask]))<=
                                tensor_plan['representation']['export_world_Q_atol'],
                                'Exported axes do not reconstruct the evaluated world tensor')
                    if heterogeneous:
                        errors,physical=hetero.point_errors(point_pred,{k:d[k][indices] for k in hetero.TARGET_KEYS})
                    else:
                        errors,physical=physical_errors(point_pred,{k:d[k][indices] for k in ('normals','axis','Q_world')})
                    difference=arrays['linear_rgb'][mask]-arrays['linear_gt'][mask]
                    metrics[str(camera)]=dict(split='fitting' if camera in splits['fitting'] else 'held_out',physical=physical,
                        RGB_linear_RMSE=float(np.sqrt(np.mean(difference**2))),RGB_relative_L2=float(np.linalg.norm(difference)/max(np.linalg.norm(arrays['linear_gt'][mask]),1e-15)),
                        preview_RGB_channels_above_one=int((arrays['linear_rgb'][mask]>1).sum()),
                        native_standard_error_RMS=float(np.sqrt(np.mean(standard_error.reshape(SIZE,SIZE,3,3)[mask]**2))))
                    np.savez_compressed(destination/('camera-%d-errors.npz'%camera),**errors,native_component_SE=standard_error,geometry_hit_mask=d['hit_mask'])
                q=[]
                diagnostic_widths=[]
                diagnostic_log_q=[]
                for pp,ff in zip(pair_points,pair_frames):
                    pn=tensor(ff['normals'])
                    if tensor_plan:
                        local,u,_,_=shader.predict_tensor(tensor(pp),pn)
                        q.append(tensor_math.world_tensor(local,u).to(pn.dtype))
                        if log_tensor:
                            diagnostic_log_q.append(world_log_tensor(local,u))
                        widths=tensor_math.export_axes(local,u)[:2]
                    else:
                        a=shader.predict_anisotropic_components(tensor(pp))
                        q.append(world_q(a[0],a[1],a[4],pn,torch))
                        widths=a[:2]
                    if spatial:
                        diagnostic_widths.append(torch.cat(widths,dim=-1).cpu().numpy())
                R=normal_transport(tensor(pair_frames[1]['normals']),tensor(pair_frames[0]['normals']),torch)
                delta=q[0]-R@q[1]@R.transpose(-1,-2)
                gt0,gt1=(tensor(ff['Q_world']) for ff in pair_frames)
                truth_delta=gt0-R@gt1@R.transpose(-1,-2)
                metrics['surface_Q_smoothness_diagnostic']=dict(pairs=len(pair_ix),predicted_mean_squared=float(delta.square().sum((-1,-2)).mean()),
                    oracle_mean_squared=float(truth_delta.square().sum((-1,-2)).mean()),new_loss=False)
                if spatial:
                    if parity_plan:
                        require(array_hash(sampler.counts)==parity_plan['D6_sampler_snapshots'][str(step)],
                                'Sampled pair counts differ from accepted D6')
                    sampled=coverage_summary(domain,sampler.counts,fitting_counts)
                    metrics['surface_Q_training_coverage']=sampled
                    widths=np.stack(diagnostic_widths,axis=1)
                    squared=widths**2
                    gap=abs(squared[...,0]-squared[...,1])/squared.sum(axis=-1)
                    log_diagnostic={}
                    if log_tensor:
                        transport=R.double()
                        log_residual=diagnostic_log_q[0]-transport@diagnostic_log_q[1]@transport.transpose(-1,-2)
                        log_square=log_residual.square().sum((-1,-2))
                        require(bool(torch.isfinite(log_square).all()), 'Nonfinite reserved-edge log-Q')
                        log_diagnostic['log_transported_square']=log_square.cpu().numpy()
                        metrics['surface_log_Q_smoothness_diagnostic']=dict(pairs=len(pair_ix),
                            predicted_mean_squared=float(log_square.mean()),used_for_training=False)
                    np.savez_compressed(destination/'surface-q-diagnostic.npz',pair_ids=pair_ix,
                        transported_square=delta.square().sum((-1,-2)).cpu().numpy(),
                        oracle_transported_square=truth_delta.square().sum((-1,-2)).cpu().numpy(),
                        widths=widths,anisotropy_gap=gap,training_pair_counts=sampler.counts,**log_diagnostic)
                if faceted:
                    qu, wi = [], []
                    pn = tensor(boundary['normals'])
                    for pp in boundary['positions']:
                        budget('boundary_predictor_calls',1)
                        charge('boundary_predictor_points',len(pp))
                        if tensor_plan:
                            local,u,_,_=shader.predict_tensor(tensor(pp),pn)
                            qu.append(tensor_math.world_tensor(local,u).cpu().numpy())
                            widths=tensor_math.export_axes(local,u)[:2]
                        else:
                            physical=shader.predict_anisotropic_components(tensor(pp))
                            qu.append(world_q(physical[0],physical[1],physical[4],pn,torch).cpu().numpy())
                            widths=physical[:2]
                        wi.append(np.sort(torch.cat(widths,dim=-1).cpu().numpy(),axis=1)[:,::-1])
                    require(all(np.isfinite(values).all() for values in (*qu,*wi)), 'Nonfinite Bell boundary predictions')
                    np.savez_compressed(destination/'boundary-predictions.npz',
                                        Q_world=np.stack(qu),widths_major_minor=np.stack(wi))
                y,x=np.mgrid[:128,:256];phi=(x+.5)/256*2*np.pi;theta=(y+.5)/128*np.pi
                directions=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],-1)
                if calibration:
                    require(np.array_equal(directions, calibration_dirs), 'Export grid differs from calibration')
                    light = grid_light
                else:
                    with light_path('illumination_map'):
                        light=shader.predict_outer_lights(tensor(np.zeros((32768,3))),tensor(directions.reshape(-1,3))).cpu().numpy().reshape(128,256,3)
                require(np.isfinite(light).all() and (light>=0).all(),'Invalid illumination map')
                from benchmarks.bell_reference_e11 import environment
                from benchmarks.donut_recovery_v1 import ENVIRONMENT
                truth_light=environment(tensor(directions),ENVIRONMENT).cpu().numpy()
                np.savez_compressed(destination/'illumination.npz',directions=directions,linear_RGB=light,known_linear_RGB=truth_light)
                preview=illumination_preview(destination/'illumination-preview.png',light)
                weight=np.sin(theta)[...,None]
                metrics['illumination']=dict(minimum=light.min((0,1)).tolist(),maximum=light.max((0,1)).tolist(),
                    mean_spherical=((light*weight).sum((0,1))/weight.sum()).tolist(),
                    relative_L2=float(np.sqrt(((light-truth_light)**2*weight).sum()/max((truth_light**2*weight).sum(),1e-15))),
                    indirect='exactly zero',mode=report['effective_lighting'],preview=preview)
                if calibration:
                    metrics['illumination']['calibration'] = dict(calibration_record,
                        raw_mean_RGB=calibrated.raw_mean.cpu().tolist(),
                        scale_RGB=calibrated.scale.cpu().tolist())
                if known_audit:
                    metrics['illumination']['analytic_agreement'] = acquisition.validate_map(light,truth_light)
                json_new(destination/'metrics.json',metrics)
            shader.train();report['saved_endpoints'].append(dict(step=step,metrics=metrics))
            progress()
        def work():
            nonlocal cursor,order,sibling
            with shader_lighting(shader,field.MCShadingNetwork,lighting_arm,calibrated),native_tensor_default(torch):
                if tensor_plan:
                    sibling_attachment = known_audit.attached(sibling) if known_audit else nullcontext()
                    sibling_lighting = 'D1' if known_l2 else sibling.tensor_arm
                    with evaluation_scope(shader,torch,np),torch.no_grad(),shader_lighting(
                            sibling,field.MCShadingNetwork,sibling_lighting),sibling_attachment,light_path('paired',2):
                        budget('shader_calls',2);budget('shadow_rays',2*len(p)*1536)
                        gate,capture=paired_initial_gate(shader,sibling,p,n,wo,tensor_plan['representation'])
                        np.savez_compressed(args.output/'paired-initial-gate.npz',**capture)
                        report['paired_initial_gate']=gate
                        if log_tensor:
                            archived=numeric(initial_controls/'paired-initial-gate.npz')
                            require(capture.keys()==archived.keys() and
                                    all(v.dtype==archived[k].dtype and np.array_equal(v,archived[k])
                                        for k,v in capture.items()),
                                    'D11 initial native RGB or physical field differs from D8-T capture')
                            report['D8_T_initial_capture_verified']=True
                    progress()
                    require(gate['passed'],'Paired initial Q, scalar heads, light or RGB differ')
                    sibling=None
                evaluate(0)
                with (args.output/'training.jsonl').open('x') as log:
                    for step in range(10000):
                        budget();shader.train()
                        if cursor+512>len(order):order=sample_rng.permutation(len(order));cursor=0
                        selected=order[cursor:cursor+512];cursor+=512
                        pp,nn,vv=(tensor(pool[k][selected]) for k in ('position','normals','outgoing'))
                        optimizer.zero_grad();lr=schedule(optimizer,step)
                        with calibration_scope(False):
                            o=shade(pp,vv,nn,step,True)
                        photo=renderer.NeROMaterialRenderer.compute_rgb_loss(SimpleNamespace(cfg={'rgb_loss':'charbonier'}),o['linear_rgb'],tensor(pool['linear_components'][selected,2])).mean()
                        if tensor_plan:
                            scalar,energy,spec=shader.regularize_tensor(pp,nn,o)
                            reg=(scalar+energy+spec).mean()
                        else:
                            reg=shader.anisotropic_regularization(pp,nn,o['tangents'],o['bitangents'],o['mx'],o['my'],o['metallic'],o['kd'],o['f_d'],o['f_s_sum'],o['L_spec'],o['rotation'],o['rotation_raw'],step).mean()
                        white=renderer.NeROMaterialRenderer.compute_diffuse_light_regularization(SimpleNamespace(cfg={'reg_diffuse_light_lambda':.01}),o['diffuse_light']).mean()
                        loss=training_objective(photo,reg,white,include_diffuse_white=include_diffuse_white)
                        if spatial:
                            count=spatial_policy['batch_pairs']
                            budget('surface_q_batches',1)
                            charge('surface_q_pairs',count)
                            charge('surface_q_predictor_calls',2)
                            charge('surface_q_predictor_points',2*count)
                            indices=sampler.draw(count)
                            spatial_start.record()
                            index=torch.as_tensor(indices,dtype=torch.long,device='cuda')
                            neighborhood={k:v.index_select(0,index) for k,v in surface_tensors.items()}
                            if robust_delta is not None:
                                spatial_loss, quadratic_loss, edge_contributions, attenuation = spatial_term(
                                    shader,neighborhood,spatial_policy,torch,distance_weighting=weighting,robust_delta=robust_delta)
                            elif weighting is None:
                                spatial_loss=spatial_term(shader,neighborhood,spatial_policy,torch)
                            else:
                                spatial_loss, edge_contributions = spatial_term(
                                    shader,neighborhood,spatial_policy,torch,distance_weighting=weighting)
                            spatial_end.record()
                            loss=loss+spatial_loss
                        require(bool(torch.isfinite(loss) & torch.isfinite(white)),
                                'Nonfinite joint loss or diffuse-white diagnostic')
                        if tensor_plan:
                            q_widths=tensor_math.sorted_widths(o['Q_local'].detach())
                        row=dict(completed_updates=step+1,lr=lr,RGB_loss=float(photo.detach()),material_prior=float(reg.detach()),
                            diffuse_white_prior=float(white.detach()),kd_mean=o['kd'].detach().mean(0).cpu().tolist(),
                            metallic_mean=float(o['metallic'].detach().mean()),diffuse_allocation=o['f_d'].detach().mean(0).cpu().tolist(),
                            mx_mean=float((q_widths[:,0] if tensor_plan else o['mx']).detach().mean()),
                            my_mean=float((q_widths[:,1] if tensor_plan else o['my']).detach().mean()))
                        if ablation or spatial:
                            row['optimization_loss']=float(loss.detach())
                        if spatial:
                            row[spatial_name+'_loss']=float(spatial_loss.detach())
                            report['surface_q_timing']['forward_CUDA_seconds']+=spatial_start.elapsed_time(spatial_end)/1000
                            if weighting is not None:
                                charge('edge_diagnostic_batches',1)
                                row['edge_diagnostic']=acquisition.edge_diagnostic(indices,edge_contributions,o,weighting,torch)
                            if robust_delta is not None:
                                charge('robust_diagnostic_batches',1)
                                ratio,mean_slope,min_slope,above=attenuation.tolist()
                                row['robust_diagnostic']=dict(quadratic_loss=float(quadratic_loss.detach()),penalty_ratio=ratio,
                                    mean_gradient_multiplier=mean_slope,min_gradient_multiplier=min_slope,fraction_above_delta=above)
                        if step==0 or (step+1)%2000==0:
                            if not tensor_plan:
                                energy,spec=legacy_energy_specular(o['f_d'],o['f_s_sum'],o['L_spec'],.05,.05,torch)
                            terms=dict(photo=photo,material=reg,energy=energy,specular=spec)
                            if probe_diffuse_white:
                                terms['diffuse_white']=white
                            if spatial:
                                terms[spatial_name]=spatial_loss
                            if robust_delta is not None:
                                terms['surface_q_quadratic_counterfactual']=quadratic_loss
                            budget('component_reverse_sweeps',len(terms))
                            named=[(n,p) for n,p in shader.named_parameters() if p.requires_grad]
                            row['gradient_interactions']=gradient_diagnostics(terms,[p for _,p in named],torch,[n for n,_ in named])
                            if known_audit:
                                require(row['gradient_interactions']['terms']['diffuse_white']['gradient_l2']==0.,
                                        'Known L2 diffuse-white unexpectedly has a material derivative')
                        loss.backward()
                        active=[p for p in shader.parameters() if p.requires_grad and p.grad is not None]
                        require(active and all(bool(torch.isfinite(p.grad).all()) for p in active),'Missing/nonfinite native gradients')
                        row['gradient_groups']={group:sum(float(torch.sum(p.grad.detach()**2)) for name,p in shader.named_parameters()
                            if name.startswith(group+'.') and p.grad is not None)**.5 for group in (*material_names,'outer_light','inner_light')}
                        require(all(any(name.startswith(group+'.') and p.grad is not None for name,p in shader.named_parameters())
                            for group in material_names),'A material predictor has no gradient')
                        require(lighting_arm=='D1' or any(name.startswith('outer_light.') and p.grad is not None
                            for name,p in shader.named_parameters()),'Learned direct light is disconnected')
                        norm=torch.nn.utils.clip_grad_norm_(shader.parameters(),.7);row['preclip_gradient_l2']=float(norm)
                        require(bool(torch.isfinite(norm)), 'Nonfinite total gradient norm')
                        before=[p.detach().clone() for p in active] if step<2 else []
                        optimizer.step();report['optimizer_returns']+=1
                        if step<2:
                            displacement=sum(float(abs(p.detach()-q).sum()) for p,q in zip(active,before))
                            require(displacement==0 if step==0 else displacement>0,'Original warm-up parameter displacement failed')
                            require(all(int(optimizer.state[p]['step'])==step+1 and
                                bool(torch.isfinite(optimizer.state[p]['exp_avg']).all()) and
                                bool(torch.isfinite(optimizer.state[p]['exp_avg_sq']).all()) for p in active), 'Adam state did not advance finitely')
                            row['parameter_displacement_L1']=displacement
                        log.write(json.dumps(row,allow_nan=False)+'\n')
                        if (step+1)%2000==0:
                            log.flush()
                            checkpoint=args.output/('checkpoint-%05d.pth'%(step+1))
                            saved=dict(protocol=protocol,variant=initial['variant'],arm=args.phase,
                                benchmark_sha256=args.benchmark_sha256,freeze_sha256=args.freeze_sha256,
                                completed_updates=step+1,network_state_dict=shader.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                                torch_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),
                                sample_RNG=sample_rng.get_state(),sample_order=order,sample_cursor=cursor)
                            if spatial:
                                saved['surface_q_sampler']=sampler.checkpoint()
                            torch.save(saved,checkpoint)
                            # Trust is limited to this process's own newly written checkpoint.
                            loaded=torch.load(checkpoint,map_location='cpu')
                            require(all(loaded[k]==saved[k] for k in ('protocol','variant','arm','benchmark_sha256','freeze_sha256','completed_updates')), 'Checkpoint identity mismatch')
                            require(set(loaded['network_state_dict'])==set(shader.state_dict()) and all(torch.equal(v.cpu(),loaded['network_state_dict'][k]) for k,v in shader.state_dict().items()), 'Checkpoint round-trip differs')
                            shader.load_state_dict(loaded['network_state_dict'],strict=True)
                            del loaded,saved
                            evaluate(step+1)
                require(report['optimizer_returns']==10000,'Incomplete recovery history')
                if probe_diffuse_white:
                    measured=('component_reverse_sweeps',)
                    if calibration:
                        measured+=('calibration_evaluations','calibration_directions')
                    require(all(report['counters'][key]==limits[key] for key in measured),
                            'Incomplete calibration or diagnostic accounting')
                if spatial:
                    measured=('component_reverse_sweeps','surface_q_batches','surface_q_pairs',
                              'surface_q_predictor_calls','surface_q_predictor_points')
                    require(all(report['counters'][key]==limits[key] for key in measured) and
                            sampler.counts.sum()==limits['surface_q_pairs'] and
                            (sampler.counts[domain['training_ids']]>0).all() and
                            not sampler.counts[domain['diagnostic_ids']].any(),
                            'Incomplete surface work accounting, coverage or diagnostic isolation')
            return {}
        run_with_cleanup(work,torch)
        light_handles.close()
        if known_audit:
            report['known_lighting_audit']=known_audit.final(acquisition.route_plan(bound['work'],tensor_plan is not None),
                                                           freeze['steps'])
        report['cleanup_verified']=True
        report['cleanup_scope']='Native synchronization/cache cleanup and lighting/default/RNG scopes; process/container exit handled by foreground launcher'
        import resource
        report['resources']=dict(CUDA_peak_allocated=torch.cuda.max_memory_allocated(),
            CUDA_peak_reserved=torch.cuda.max_memory_reserved(),host_peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if heterogeneous:
            require(report['counters']==bound['work'], 'Heterogeneous work counters differ from fixed plan')
            hetero.freeze(args.freeze,args.freeze_sha256)
            hetero.dataset(freeze,args.benchmark,args.admission,args.verification)
        else:
            verify_freeze(args.freeze,args.freeze_sha256);benchmark(args.benchmark,args.benchmark_sha256)
        source_geometry()
        report.update(status='complete',scientific_work_complete=True)
        report['outputs']={p.relative_to(args.output).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
            for p in args.output.rglob('*') if p.is_file() and p.name not in ('progress.json','progress.next')}
    except BaseException as exc:
        report.update(status='failed',scientific_work_complete=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        if getattr(acquisition,'CAPTURE_ORIENTATION_FAILURE',False):
            from benchmarks.bell_robust_completion_v1 import orientation_failure
            try:
                witness=orientation_failure(exc,report['optimizer_returns']+1)
            except Exception as diagnostic_error:
                witness=dict(status='unavailable',reason=str(diagnostic_error))
            if witness is not None:
                report['orientation_failure']=witness
        raise
    finally:
        light_handles.close()
        report['elapsed_seconds']=time.monotonic()-started
        json_new(args.output/'recovery.json',report)
