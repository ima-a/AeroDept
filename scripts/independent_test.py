"""Versioned reserved-test freeze and explicit one-time execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aerodepth_lite.independent_test import main
if __name__=='__main__':main()
