"""Explicit isolated E14 entry point; importing performs no scientific rendering."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.bell_two_light_e14 import main
if __name__=='__main__':main()
