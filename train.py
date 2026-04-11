#!/usr/bin/env python3
"""
Wrapper to run training from outer directory.
Handles Python path setup for the nested project structure.
"""
import sys
import os
from pathlib import Path

# Add inner project to path
project_root = Path(__file__).parent / "Brain-Connectivity-GCN-main"
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# Now import and run
from brain_gcn.main import main

if __name__ == "__main__":
    main()
