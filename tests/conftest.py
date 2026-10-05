"""Put the repository root on sys.path so the experiment modules import cleanly.

The modules live in the repo root (one level up from this tests/ directory) and
are imported by name: `import zeta_zeros`, `import goldbach`, etc.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
