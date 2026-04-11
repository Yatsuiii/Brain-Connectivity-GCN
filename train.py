#!/usr/bin/env python3
"""
Entry point for training. Run from the project root directory.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as a script
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from brain_gcn.main import main

if __name__ == "__main__":
    main()
