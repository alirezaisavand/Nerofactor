"""Explicit isolated E17 retention entry point."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.bell_neural_retention_e17 import main
if __name__=='__main__':main()
