"""Separate explicitly authorized E11D pilot; never launches the full dataset."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.bell_highlight_pilot_e11d import main
if __name__=='__main__':main()
