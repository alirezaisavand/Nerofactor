import importlib.abc,sys
forbidden=('network','dataset','train','nvdiffrast','open3d','scientific_sources','experiments.neural_material_field_v1','experiments.material_shader_v1')
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if any(name==p or name.startswith(p+'.') for p in forbidden):raise RuntimeError('Native model/shader import blocked: '+name)
sys.meta_path.insert(0,Block())
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','-o','junit_family=xunit1','tests/scientific/test_native_material_a_e10.py','tests/infrastructure/test_material_variants_e10.py','tests/infrastructure/test_material_smoke.py::PlanningTests::test_original_bear_configuration_no_overrides_and_cpu_only_preflight','tests/infrastructure/test_material_smoke.py::PlanningTests::test_default_research_execution_still_requires_G_handoff','tests/infrastructure/test_material_smoke.py::PlanningTests::test_bounded_orchestration_reuses_output_failure_and_cleanup_framework','--junitxml=/home/isavand/paper/evidence/material-e10-preparation-v6/pytest.xml'])
assert not any(any(name==p or name.startswith(p+'.') for p in forbidden) for name in sys.modules)
print('E10 native/full-N1/scientific-experiment import blocker verified; only synthetic CPU stand-ins and metadata tests executed')
sys.exit(code)
