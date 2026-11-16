"""Explicit isolated E16 native paired-field entry point."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.bell_neural_material_e16 import main
if __name__=='__main__':main()
