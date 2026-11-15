"""Metadata-only E10 native selection contracts; no native science imports/execution."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest
import yaml
from infrastructure import stage_execution as stage
from infrastructure.material_variants import CORE,ESTIMATOR,ORIGINAL,identity
from infrastructure.stage_sources import ROOT,PINS


@pytest.fixture
def inputs(tmp_path):
    data=tmp_path/'datasets';thesis=tmp_path/'thesis';geometry=tmp_path/'geometry';out=tmp_path/'outputs'
    for p in (data/'GlossySynthetic/bell',data/'GlossyReal/bear',thesis,geometry):p.mkdir(parents=True)
    (geometry/'mesh.ply').write_bytes(b'synthetic path fixture only')
    return data,thesis,geometry,out


def planning(inputs,variant=CORE,scene='bell'):
    data,thesis,geometry,out=inputs
    name='material_a_'+scene+('_estimator_v1.yaml' if variant==ESTIMATOR else '_core_v1.yaml')
    return stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments'/name,ROOT,{'geometry':geometry},material_variant=variant)


@pytest.mark.parametrize('variant,scene',[(CORE,'bell'),(CORE,'bear'),(ESTIMATOR,'bell')])
def test_explicit_native_variant_plan_resolves_original_inputs_and_preserves_schedule(inputs,variant,scene):
    p=planning(inputs,variant,scene);cfg=p['configuration']
    assert cfg['material_variant']==variant and p['scientific_commit']==PINS['material'] and p['scene']==scene
    original=yaml.safe_load((p['science']/('configs/material/syn/bell.yaml' if scene=='bell' else 'configs/material/real/bear.yaml')).read_text())
    for key in ('optimizer_type','lr_type','lr_cfg','total_step','val_interval','save_interval','loss','val_metric','key_metric_name'):
        assert cfg[key]==original[key]
    assert cfg['shader_cfg']['diffuse_sample_num']==512 and cfg['shader_cfg']['specular_sample_num']==1024
    assert not p['resolver'].output_root.exists() and p['selected'].checkpoint is None
    assert identity(variant,cfg)['historical_checkpoint_equivalent'] is False


def test_original_A_plan_and_defaults_are_unchanged(inputs):
    data,thesis,geometry,out=inputs
    p=stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments/material_a_bear_v1.yaml',ROOT,{'geometry':geometry})
    original=yaml.safe_load((p['science']/'configs/material/real/bear.yaml').read_text())
    assert p['configuration']==original and p['material_variant']==ORIGINAL
    assert identity(ORIGINAL,original)['algorithm']=='anisotropic-material-A'


def test_variant_mismatch_geometry_and_research_handoff_fail_closed(inputs):
    data,thesis,geometry,out=inputs
    with pytest.raises(ValueError,match='mismatch'):
        stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments/material_a_bell_core_v1.yaml',ROOT,{'geometry':geometry},material_variant=ESTIMATOR)
    with pytest.raises(ValueError,match='geometry'):
        stage.plan('geometry',PINS['geometry'],'R',data,thesis,out,material_variant=CORE)
    with patch.object(stage,'native_guards'),patch.object(stage,'native_run') as run:
        with pytest.raises(ValueError,match='mesh-handoff'):stage.execute(planning(inputs),'unadmitted','image','0'*40,corrected_material_smoke=True)
        run.assert_not_called();assert not out.exists()


def test_cold_plan_does_not_import_torch_or_scientific_modules(inputs):
    args=[str(p) for p in inputs]
    script='''
import builtins,sys
from pathlib import Path
real=builtins.__import__
def blocked(name,*args,**kwargs):
 assert name.split('.')[0] not in ('torch','network','dataset','train','open3d','nvdiffrast','scientific_variants'),name
 return real(name,*args,**kwargs)
builtins.__import__=blocked
from infrastructure.stage_execution import plan
from infrastructure.stage_sources import ROOT,PINS
p=plan('material',PINS['material'],'C',Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[4]),ROOT/'configs/experiments/material_a_bell_core_v1.yaml',ROOT,{'geometry':Path(sys.argv[3])},material_variant='A_corrected_core_v1')
assert not p['resolver'].output_root.exists()
print('metadata-only corrected native plan passed')
'''
    result=subprocess.run(['python3','-B','-c',script,*args],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_oracle_benchmark_is_separate_from_G_handoff_and_requires_complete_generated_evidence(inputs):
    from infrastructure.material_variants import admit_oracle
    from infrastructure.stage_sources import digest
    from infrastructure.stage_workspace import source_records
    data,thesis,geometry,out=inputs;scene=data/'GlossySynthetic/bell'
    (scene/'masks_train').mkdir()
    for i in range(128):
        for name in (str(i)+'.png',str(i)+'-depth.png',str(i)+'-camera.pkl','masks_train/'+str(i)+'.png'):
            (scene/name).write_bytes(b'synthetic metadata fixture only; never decoded')
    labels=geometry/'labels.json';labels.write_text('{}\n')
    environment=yaml.safe_load((ROOT/'configs/material_corrected/bell_core_known_v1.yaml').read_text())['shader_cfg']['known_environment']
    generated=geometry/'generation.json';generated.write_text(json.dumps(dict(known_environment=environment,indirect_mode='zero_bounce_v1',protocol='paper-benchmark-generation-v1',status='complete',database_name='syn/bell',BRDF='Kurt-a0-metallic-additive-v1',mesh_sha256=digest(geometry/'mesh.ply'),independent_reference=True,dataset_files=source_records(scene),material_labels_sha256=digest(labels))))
    p=stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments/material_a_bell_core_known_v1.yaml',ROOT,{'geometry':geometry},material_variant=CORE,oracle_benchmark=True)
    proof=dict(schema_version=1,protocol='paper-anisotropic-oracle-dataset-v1',status='complete',database_name='syn/bell',
        mesh_sha256=digest(geometry/'mesh.ply'),coordinates='fixture world',normals='fixture outward',image_formation='fixture independent linear RGB',
        illumination='fixture known',BRDF='Kurt-a0-metallic-additive-v1',split_policy='fixture fixed IDs',researcher_reviewed_generation=True,known_material_labels=True,
        generation_report=dict(root='geometry',path='generation.json',sha256=digest(generated)),material_labels=dict(root='geometry',path='labels.json',sha256=digest(labels)),dataset_files=source_records(scene))
    path=geometry/'oracle.json';path.write_text(json.dumps(proof))
    v=admit_oracle(p,path);assert v['research_G_handoff_admitted'] is False
    for key,value in [('status','pending'),('mesh_sha256','0'*64),('known_material_labels',False),('dataset_files',[])]:
        path.write_text(json.dumps({**proof,key:value}))
        with pytest.raises(ValueError):admit_oracle(p,path)
    path.write_text(json.dumps(proof));labels.write_text('{"changed":true}\n')
    with pytest.raises(ValueError):admit_oracle(p,path)
    assert not out.exists()


def test_original_oracle_benchmark_does_not_install_corrected_mathematics(inputs):
    data,thesis,geometry,out=inputs
    p=stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments/material_a_bell_original_data_v1.yaml',ROOT,{'geometry':geometry},oracle_benchmark=True)
    assert p['material_variant']==ORIGINAL and not p['configuration']['reg_mat']
    assert 'regularization_version' not in p['configuration']['shader_cfg']


@pytest.mark.parametrize('variant,kind',[(ORIGINAL,'original'),(ESTIMATOR,'estimator'),(CORE,'core')])
def test_known_light_three_arm_benchmark_has_identical_data_objective_budget_and_light(inputs,variant,kind):
    data,thesis,geometry,out=inputs
    p=stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/('configs/experiments/material_a_bell_'+kind+'_known_v1.yaml'),ROOT,{'geometry':geometry},material_variant=variant,oracle_benchmark=True)
    cfg=p['configuration'];assert cfg['total_step']==2000 and not cfg['reg_mat'] and not cfg['reg_diffuse_light']
    shader=cfg['shader_cfg'];assert shader['illumination_mode']=='known_env_v1' and shader['indirect_mode']=='zero_bounce_v1' and not shader['human_lights']
    assert cfg['database_name']=='syn/bell' and p['selected'].dataset.name=='GlossySynthetic'


def test_known_light_cannot_train_on_historical_images_without_generated_oracle_admission(inputs):
    data,thesis,geometry,out=inputs
    with pytest.raises(ValueError,match='Known-light'):
        stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments/material_a_bell_core_known_v1.yaml',ROOT,{'geometry':geometry},material_variant=CORE)


def test_native_prior_candidate_is_explicit_and_independent_of_spatial_change_gate(inputs):
    data,thesis,geometry,out=inputs
    p=stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments/material_a_bell_core_prior_candidate_v1.yaml',ROOT,{'geometry':geometry},material_variant=CORE)
    s=p['configuration']['shader_cfg'];assert p['configuration']['reg_mat'] and not s['reg_change']
    assert s['regularization_version']=='unconditional_v1' and s['reg_energy_loss'] and s['reg_spec_loss']


def test_separate_energy_model_requires_matching_ground_truth_family_and_configuration(inputs):
    from infrastructure.material_variants import ENERGY
    data,thesis,geometry,out=inputs
    p=stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments/material_a_bell_energy_budget_known_v1.yaml',ROOT,{'geometry':geometry},material_variant=ENERGY,oracle_benchmark=True)
    assert p['material_variant']==ENERGY and identity(ENERGY,p['configuration'])['ideal_directional_energy_bound_proved'] is True
    with pytest.raises(ValueError):
        stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/'configs/experiments/material_a_bell_core_known_v1.yaml',ROOT,{'geometry':geometry},material_variant=ENERGY,oracle_benchmark=True)


def test_candidate_configs_fail_closed_in_legacy_dispatch_without_modifying_legacy_entry(inputs):
    import ast
    p=planning(inputs);assert p['configuration']['train_dataset_type']=='dummy'
    raw=yaml.safe_load(p['raw']);assert raw['train_dataset_type']=='stage_variant_dummy_v1'
    tree=ast.parse((p['science']/'dataset/name2dataset.py').read_text())
    mapping=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='name2dataset' for t in n.targets))
    names=[ast.literal_eval(k) for k in mapping.keys];assert 'dummy' in names and 'stage_variant_dummy_v1' not in names
    source=ast.parse((p['science']/'train/trainer.py').read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Trainer')
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assert run.body[0].value.func.attr=='_init_dataset' and run.body[1].value.func.attr=='_init_network'
