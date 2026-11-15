#!/usr/bin/env python3
"""Guarded Bell oracle generator; no generation occurs without explicit authorization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmarks.generate_bell_e11 import main
if __name__=='__main__':main()
