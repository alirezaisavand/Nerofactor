"""Explicit isolated E11G scientific entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.bell_multiview_e11g import main
if __name__ == '__main__':
    main()
