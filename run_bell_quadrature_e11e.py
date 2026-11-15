"""Explicitly authorized,84-probe E11E reference check; no dataset or training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.bell_quadrature_e11e import main
if __name__=='__main__':main()
