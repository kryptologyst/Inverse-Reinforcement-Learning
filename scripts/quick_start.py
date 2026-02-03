"""Scripts for running IRL experiments."""

#!/usr/bin/env python3
"""Quick start script for Maximum Entropy IRL."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train.train_irl import main

if __name__ == "__main__":
    main()
