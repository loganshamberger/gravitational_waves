"""Reusable simulation framework: System, Runner, config-driven batch runs.

Runner drives a System: it validates that the supplied parameter dict
contains everything the system needs, then invokes simulate(). Systems
are pluggable — SchwarzschildGeodesic is one instance; a Kerr version can
implement the same interface later without touching this module.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class MissingParameterError(ValueError):
    """Raised when a System is run without all of its required parameters."""


class UnknownSimulationKindError(ValueError):
    """Raised when a config entry's `kind` has no matching System."""


class System(ABC):
    """Base class every simulatable physical system must implement."""

    @abstractmethod
    def validate(self, params: Dict[str, Any]) -> None:
        """Raise MissingParameterError (or similar) if params is unfit to simulate."""

    @abstractmethod
    def simulate(self, params: Dict[str, Any]) -> Any:
        """Run the simulation and return the result (e.g. a trajectory)."""

    @abstractmethod
    def visualize(self, result: Any) -> None:
        """Render the result of simulate()."""


class Runner:
    """Validates params against a System, then runs it."""

    def __init__(self, system: System):
        self.system = system

    def run(self, params: Dict[str, Any]) -> Any:
        self.system.validate(params)
        return self.system.simulate(params)


def run_from_config(config_path: str, registry: Dict[str, type]) -> None:
    """Read a YAML file of simulations and run + visualize each one.

    `registry` maps a `kind` string to a System subclass. Expected YAML shape:
        simulations:
          - kind: schwarzschild_massive
            params: {M: 1.0, E: 0.95, h: 4.0, r0: 10.0, phi0: 0.0, dr_dtau0: 0.0, tau_max: 500.0}
          - kind: schwarzschild_photon
            params: {M: 1.0, b: 6.0, r0: 30.0, phi0: 0.0, dr_dlambda0: -0.99, lambda_max: 80.0}
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for entry in config["simulations"]:
        kind = entry["kind"]
        system_cls = registry.get(kind)
        if system_cls is None:
            raise UnknownSimulationKindError(
                f"Unknown simulation kind {kind!r}; known kinds: {list(registry)}"
            )

        system = system_cls()
        runner = Runner(system)
        result = runner.run(entry["params"])
        system.visualize(result)
