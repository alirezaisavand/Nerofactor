"""Isolated explicit entry point for the authorized E15 physical experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.bell_material_inverse_e15 import main
if __name__=='__main__':main()
