"""Entry point: wires the reusable framework (core.py) to concrete Systems
(schwarzschild.py) and runs a batch of simulations from a YAML config.

Add a new System (e.g. a Kerr geodesic) by implementing the System interface
in its own module and registering it in SYSTEM_REGISTRY below.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),".","geodesics"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),".","radiation"))
from core import run_from_config
from schwarzschild import (
    SchwarzschildGeodesic,
    SchwarzschildGeodesicCoordTime,
    SchwarzschildPhotonGeodesic,
)
from inspiral import SchwarzschildAdiabaticInspiral

SYSTEM_REGISTRY = {
    "schwarzschild_massive": SchwarzschildGeodesic,
    "schwarzschild_massive_coord_time": SchwarzschildGeodesicCoordTime,
    "schwarzschild_photon": SchwarzschildPhotonGeodesic,
    "schwarzschild_adiabatic_inspiral": SchwarzschildAdiabaticInspiral,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_from_config(config_path, SYSTEM_REGISTRY)
