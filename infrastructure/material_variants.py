"""Metadata/configuration admission for native A derivatives; no scientific imports."""
from copy import deepcopy
import math
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, PINS, digest

ORIGINAL='A_original'
ESTIMATOR='A_estimator_v1'
CORE='A_corrected_core_v1'
ENERGY='A_energy_budget_v1'
VARIANTS=(ORIGINAL,ESTIMATOR,CORE,ENERGY)
REGULARIZERS=('legacy_v1','disabled_v1','unconditional_v1')
SHADER_KEYS={'diffuse_sample_num','specular_sample_num','human_lights','light_exp_max','inner_light_exp_max',
 'outer_light_version','geometry_type','reg_change','change_eps','change_type','reg_lambda1','reg_min_max',
 'random_azimuth','is_real','anisotropy','max_n_exp','max_alpha_exp','reg_energy_loss','reg_energy_loss_lambda',
 'reg_spec_loss','reg_spec_loss_lambda','regularization_version'}
SHADER_KEYS |= {'illumination_mode','known_environment','indirect_mode'}
# These historical defaults have no active anisotropic consumer. Do not expose
# them as working numerical controls in a corrected configuration.
SHADER_KEYS -= {'geometry_type','reg_min_max','max_n_exp','max_alpha_exp'}


def lighting(shader):
    mode=shader.get('illumination_mode','learned_native')
    require(mode in ('learned_native','known_env_v1'),'Unknown illumination mode')
    if mode=='learned_native':
        require('known_environment' not in shader and shader.get('indirect_mode','native_learned')=='native_learned',
                'Learned native lighting cannot silently accept a known environment/zero-bounce override')
        return
    require(shader.get('human_lights') is False and shader.get('indirect_mode')=='zero_bounce_v1',
            'Known-light benchmark explicitly disables human light and indirect bounces')
    environment=shader.get('known_environment',{})
    require(type(environment) is dict and set(environment)=={'ambient','lobes'},'Explicit analytic environment required')
    rgb=lambda a:isinstance(a,list) and len(a)==3 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in a)
    require(rgb(environment['ambient']) and isinstance(environment['lobes'],list),'Invalid environment RGB/lobes')
    for lobe in environment['lobes']:
        require(type(lobe) is dict and set(lobe)=={'axis','RGB','power'} and rgb(lobe['RGB']) and
                isinstance(lobe['axis'],list) and len(lobe['axis'])==3 and
                all(type(v) in (int,float) and math.isfinite(v) for v in lobe['axis']) and
                abs(sum(v*v for v in lobe['axis'])-1)<1e-12 and type(lobe['power']) is int and lobe['power']>=1,
                'Environment lobe needs unit world axis, nonnegative RGB and positive integer power')


def configuration(cfg,variant):
    require(variant in VARIANTS,'Unknown material variant')
    require(cfg.get('material_variant',variant)==variant,'Configuration conflicts with explicit material variant')
    lighting(cfg.get('shader_cfg',{}))
    if variant==ORIGINAL:return deepcopy(cfg)
    out=deepcopy(cfg);shader=out.get('shader_cfg')
    require(out.get('network')=='material' and type(shader) is dict,'Explicit material shader_cfg required')
    require(not (set(shader)-SHADER_KEYS),'Unsupported corrected shader option')
    require(shader.get('anisotropy') is True and out.get('n_lobes',1)==1,'Corrected variant requires anisotropy and one lobe')
    if 'database_name' in out and 'is_real' in shader:
        require(shader['is_real']==out['database_name'].startswith('real'),'is_real conflicts with selected dataset')
    require(shader.get('regularization_version') in REGULARIZERS,'Explicit regularization_version required')
    require(variant!=ESTIMATOR or shader['regularization_version']=='legacy_v1','Estimator-only arm preserves legacy priors')
    require(variant!=ENERGY or shader['regularization_version']!='legacy_v1','Energy model cannot use conditional historical energy priors')
    for key in ('diffuse_sample_num','specular_sample_num'):
        require(type(shader.get(key)) is int and shader[key]>0,'Explicit positive sample count required: '+key)
    for key in ('reg_change','reg_energy_loss','reg_spec_loss','random_azimuth','human_lights'):
        require(type(shader.get(key)) is bool,'Explicit Boolean shader setting required: '+key)
    for key in ('reg_lambda1','reg_energy_loss_lambda','reg_spec_loss_lambda'):
        v=shader.get(key);require(type(v) in (int,float) and math.isfinite(v) and v>=0,'Explicit nonnegative coefficient required: '+key)
    require(shader.get('outer_light_version','direction') in ('direction','sphere_direction'),'Unsupported native illumination interface')
    # Corrected files have ONE owner. Old presets retain their ignored fields in A_original only.
    for key in ('reg_energy_loss','reg_energy_loss_lambda','reg_spec_loss','reg_spec_loss_lambda','anisotropy'):
        require(key not in out,'Corrected configuration must put '+key+' exclusively in shader_cfg')
    if shader['regularization_version']=='disabled_v1':
        require(not out.get('reg_mat',True) and not shader['reg_change'] and not shader['reg_energy_loss'] and not shader['reg_spec_loss'],
                'disabled_v1 requires material priors disabled explicitly')
    out['material_variant']=variant
    return out


def identity(variant,cfg):
    require(variant in VARIANTS,'Unknown material variant')
    if variant==ORIGINAL and cfg.get('shader_cfg',{}).get('illumination_mode','learned_native')=='learned_native':
        return {'variant':variant,'base_commit':PINS['material'],'algorithm':'anisotropic-material-A'}
    configuration(cfg,variant)
    paths=['scientific_variants/__init__.py','scientific_variants/material_a_v1.py','infrastructure/material_variants.py']
    return dict(variant=variant,base_commit=PINS['material'],algorithm='anisotropic-material-'+variant,
        derivative_files={n:digest(ROOT/n) for n in paths},regularization_version=cfg['shader_cfg'].get('regularization_version','original_effective'),
        lighting_mode=cfg['shader_cfg'].get('illumination_mode','learned_native'),
        ideal_directional_energy_bound_proved=(variant==ENERGY),energy_bound_scope='Normalized even Beckmann NDF/unit geometry/Fresnel in [0,1]; finite sampled radiance is not clamped',
        historical_checkpoint_equivalent=False)


def install(renderer_module,field_module,variant,cfg):
    """Install only the material registry in an already admitted cold A process."""
    configuration(cfg,variant)
    if variant==ORIGINAL and cfg.get('shader_cfg',{}).get('illumination_mode','learned_native')=='learned_native':return renderer_module.name2renderer['material']
    require(getattr(renderer_module,'_corrected_A_variant',None) is None,'One material variant per process')
    from scientific_variants.material_a_v1 import native_classes
    shader,renderer=native_classes(field_module.MCShadingNetwork,renderer_module.NeROMaterialRenderer,variant,field_module.linear_to_srgb)
    renderer_module.name2renderer['material']=renderer
    renderer_module._corrected_A_variant=variant
    return renderer


def admit_oracle(p,path):
    """Separate generated-benchmark admission; never manufacture a G-run handoff.

    File identities are verified; generation correctness remains researcher-reviewed
    evidence, not something a JSON declaration alone can establish.
    """
    from pathlib import Path
    from infrastructure.signatures import read_report
    from infrastructure.paths import InputRef
    from infrastructure.stage_workspace import source_records
    require(p.get('oracle_benchmark') is True and p['stage']=='material','Explicit oracle benchmark mode required')
    require(path is not None,'Generated oracle benchmark admission manifest required')
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and any(root in path.parents for root in p['resolver'].input_roots.values()),'Oracle admission must be in an explicit read-only input root')
    proof,record=read_report(path,digest(path))
    require(proof.get('schema_version')==1 and proof.get('protocol')=='paper-anisotropic-oracle-dataset-v1' and
            proof.get('status')=='complete' and proof.get('database_name')==p['configuration']['database_name'] and
            proof.get('mesh_sha256')==digest(p['selected'].mesh),'Oracle scene/mesh identity mismatch or incomplete evidence')
    for key in ('coordinates','normals','image_formation','illumination','BRDF','split_policy'):
        require(isinstance(proof.get(key),str) and bool(proof[key]),'Missing oracle benchmark declaration: '+key)
    require(proof.get('researcher_reviewed_generation') is True and proof.get('known_material_labels') is True,
            'Reviewed independently generated material/illumination evidence required')
    expected_law='Kurt-a0-convex-budget-v1' if p.get('material_variant')==ENERGY else 'Kurt-a0-metallic-additive-v1'
    require(proof['BRDF']==expected_law,'Target BRDF family differs; cross-family inverse comparisons need a separate reviewed protocol')
    for key in ('generation_report','material_labels'):
        item=proof.get(key,{})
        target=p['resolver'].input(InputRef(item.get('root'),item.get('path')))
        require(digest(target)==item.get('sha256'),'Oracle '+key+' hash mismatch')
        if key=='generation_report':
            generated,_=read_report(target,item['sha256'])
            require(generated.get('protocol')=='paper-benchmark-generation-v1' and generated.get('status')=='complete' and
                    generated.get('database_name')==proof['database_name'] and generated.get('mesh_sha256')==proof['mesh_sha256'] and
                    generated.get('BRDF')==proof['BRDF'] and
                    generated.get('independent_reference') is True and generated.get('dataset_files')==proof.get('dataset_files') and
                    generated.get('material_labels_sha256')==proof.get('material_labels',{}).get('sha256'),
                    'Incomplete/mismatched independent benchmark generation evidence')
            from infrastructure.verified_ray_backend import identity as ray_identity
            require(p.get('ray_backend')=='predicate_fp64' and generated.get('ray_backend')==ray_identity(),
                    'Oracle dataset and A/B/C must use the same admitted tracing backend')
            if p['configuration']['shader_cfg'].get('illumination_mode')=='known_env_v1':
                require(generated.get('known_environment')==p['configuration']['shader_cfg']['known_environment'] and
                        generated.get('indirect_mode')=='zero_bounce_v1',
                        'Dataset illumination/bounce model differs from selected known-light renderer')
    actual=source_records(p['selected'].dataset/p['scene'])
    require(proof.get('dataset_files')==actual,'Generated dataset differs from admitted manifest')
    names={r['path'] for r in actual};ids=[str(i) for i in range(128)]
    required={name for i in ids for name in (i+'-camera.pkl',i+'.png',i+'-depth.png','masks_train/'+i+'.png')}
    require(required<=names and {n for n in names if '/' not in n and n.endswith('.pkl')}=={i+'-camera.pkl' for i in ids},
            'Oracle dataset must contain exactly128 ordered cameras with RGB/depth/masks; no extra root pickles')
    return dict(admission=record,geometry_policy='benchmark-defined-oracle-surface-v1',research_G_handoff_admitted=False,
                evidence=proof,limitation='Generation review is explicit researcher evidence; hashes do not establish physical correctness.')
