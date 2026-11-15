"""One separately authorized material-only experiment; cached visibility only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.bell_anisotropy_e11f import main
if __name__=='__main__':main()
