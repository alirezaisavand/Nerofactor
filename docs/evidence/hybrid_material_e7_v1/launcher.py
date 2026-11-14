import importlib.abc,sys
forbidden=('experiments.material_shader_v1','experiments.neural_material_field_v1','experiments.neural_material_recovery_v1','experiments.run_known_material_recovery_v1','experiments.anisotropic_estimator_v1')
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.startswith(forbidden):raise RuntimeError('Scientific import blocked: '+name)
sys.meta_path.insert(0,Block())
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','tests/scientific/test_hybrid_material_e7_v1.py','--junitxml=/home/isavand/paper/evidence/material-e7-preparation-v3/pytest.xml'])
assert not any(n.startswith(forbidden) for n in sys.modules)
print('E7 scientific import blocker intact; no real shader/neural calls authorized or executed')
sys.exit(code)
