"""Layer 1 -- shared kinds. Domain-neutral products reused by both scenarios.

This layer is optional and exists only because a SECOND consumer appeared:
both the oscillator scenario and the SDLC scenario need "a sampled series".
Per the design's L1 rule, it is defined now that it has two consumers, not
speculatively.

Nothing here is physics. `Signal` does not know what it carries -- that is the
free type parameter the scheduler grounds [6.4].
"""

from dataclasses import dataclass

import numpy as np

from core import DataProduct


@dataclass
class Signal(DataProduct):
    """A quantity sampled on a time base. The `values` dtype is the parameter.

    Under the batch driver `values` holds real time samples. A frequency-domain
    driver would ground the same port to complex phasors -- the node would not
    change, only the ground type would [6.4].
    """

    t: np.ndarray
    values: np.ndarray
    name: str = "signal"

    def __post_init__(self) -> None:
        if self.t.shape != self.values.shape:
            raise ValueError(
                f"{self.name}: t{self.t.shape} and values{self.values.shape} "
                "must have the same shape"
            )

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


@dataclass
class Table(DataProduct):
    """A list of records. The record type is the free parameter.

    Used by the SDLC scenario to carry work items; used by nothing in the
    oscillator scenario. Kept here rather than in the SDLC folder because it is
    domain-neutral -- it is the token-representation rung of the ladder.
    """

    rows: list

    def __len__(self) -> int:
        return len(self.rows)


# --- Real / Complex markers used to ground Signal's free parameter ----------
# These are ordinary types, not core machinery. The scheduler binds "F" to one
# of them; the core never knows what they mean.


class Real:
    """Ground type: a Signal of real samples (what the batch driver binds)."""


class Complex:
    """Ground type: a Signal of complex phasors.

    RESERVED. Nothing produces this yet -- it exists so the two-phase check has
    a second representation to be tested against, which is the only way to
    prove the check does anything.
    """
