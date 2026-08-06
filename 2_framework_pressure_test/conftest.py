"""Make `core` (shared/) and this folder importable, matching repo convention.

The design doc's §8 notes these sys.path hacks should eventually be retired in
favour of a real package. Not done here -- that is a separate change and would
touch the existing simulator, which this work is not allowed to break.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
