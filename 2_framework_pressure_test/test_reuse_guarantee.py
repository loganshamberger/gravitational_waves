"""The reuse guarantee, made mechanical [§7].

Two checks the design says must exist as TESTS rather than conventions:
  1. no-leak: L0 imports nothing domain-specific and names no physics.
  2. canary: a completely non-physics graph runs on L0.

The canary here is a CSV -> features -> classifier -> report pipeline built
with numpy only (the doc names scikit-learn's LogisticRegression; the repo has
no sklearn dependency and adding one for a canary would be silly, so the same
shape is built by hand).
"""

import csv
import re

import numpy as np
import pytest

from core import BatchScheduler, Context, DataProduct, Graph, Process

CORE_PATH = None


def _core_source():
    import core

    global CORE_PATH
    CORE_PATH = core.__file__
    with open(core.__file__) as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. No-leak
# ---------------------------------------------------------------------------

PHYSICS_NOUNS = [
    "schwarzschild", "kerr", "geodesic", "quadrupole", "strain", "waveform",
    "inspiral", "spacetime", "metric", "horizon", "orbit", "photon",
    "gravitational", "detector", "psd", "antenna", "geometric units",
]


def test_core_mentions_no_physics_noun():
    src = _core_source().lower()
    # Strip the module docstring: it is allowed to reference the design doc.
    body = src.split('"""', 2)[-1]
    # Whole words only -- "metric" is a substring of "Parametric".
    hits = [n for n in PHYSICS_NOUNS if re.search(rf"\b{re.escape(n)}\b", body)]
    assert not hits, f"physics leaked into L0 ({CORE_PATH}): {hits}"


def test_core_imports_nothing_from_a_domain_package():
    src = _core_source()
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, re.MULTILINE)
    allowed = {
        "logging", "abc", "dataclasses", "pathlib", "typing", "numpy", "yaml",
    }
    leaked = [m for m in imports if m.split(".")[0] not in allowed]
    assert not leaked, f"L0 imports outside the allowed set: {leaked}"


def test_core_defines_no_oracle_port_type():
    """§9 reserved the shape and explicitly did NOT build it."""
    import core

    assert not hasattr(core, "Oracle"), (
        "an Oracle port type appeared in L0; §9 says reserve the shape, build "
        "nothing until a real consumer needs it"
    )


# ---------------------------------------------------------------------------
# 2. Canary: a non-physics graph on L0
# ---------------------------------------------------------------------------


class Dataset(DataProduct):
    def __init__(self, X, y):
        self.X, self.y = X, y


class Model(DataProduct):
    def __init__(self, w, b, accuracy):
        self.w, self.b, self.accuracy = w, b, accuracy


class LoadCSV(Process):
    inputs = {}
    outputs = {"data": Dataset}

    def __init__(self, path):
        self.path = path

    def validate(self, ctx, **kw):
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def run(self, ctx, **kw):
        rows = list(csv.reader(self.path.open()))[1:]
        arr = np.array(rows, dtype=float)
        return {"data": Dataset(arr[:, :-1], arr[:, -1])}


class FeatureEngineer(Process):
    inputs = {"data": Dataset}
    outputs = {"data": Dataset}

    def run(self, ctx, data):
        X = data.X
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        return {"data": Dataset(np.hstack([X, X**2]), data.y)}


class LogisticRegression(Process):
    inputs = {"data": Dataset}
    outputs = {"model": Model}

    def __init__(self, epochs=400, lr=0.3, test_frac=0.3):
        self.epochs, self.lr, self.test_frac = epochs, lr, test_frac

    def run(self, ctx, data):
        n = len(data.y)
        idx = ctx.rng.permutation(n)          # <-- exercises ctx.rng
        cut = int(n * (1 - self.test_frac))
        tr, te = idx[:cut], idx[cut:]
        w, b = np.zeros(data.X.shape[1]), 0.0
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-(data.X[tr] @ w + b)))
            err = p - data.y[tr]
            w -= self.lr * data.X[tr].T @ err / len(tr)
            b -= self.lr * err.mean()
        pred = (1 / (1 + np.exp(-(data.X[te] @ w + b))) > 0.5).astype(float)
        return {"model": Model(w, b, float((pred == data.y[te]).mean()))}


class Report(Process):
    inputs = {"model": Model}
    outputs = {}

    def run(self, ctx, model):
        # <-- exercises ctx.workdir
        (ctx.workdir / "canary_report.txt").write_text(
            f"accuracy={model.accuracy:.3f} n_features={len(model.w)}\n"
        )
        return {}


def test_canary_graph_runs_with_zero_physics(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    path = tmp_path / "data.csv"
    with path.open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["x1", "x2", "label"])
        wtr.writerows(np.column_stack([X, y]))

    g = Graph()
    g.add("load", LoadCSV(path))
    g.add("features", FeatureEngineer())
    g.add("fit", LogisticRegression())
    g.add("report", Report())
    g.connect("load.data", "features.data")
    g.connect("features.data", "fit.data")
    g.connect("fit.model", "report.model")

    ctx = Context(rng=np.random.default_rng(7), workdir=tmp_path)
    products = BatchScheduler().run(g, ctx)

    assert products["fit"]["model"].accuracy > 0.85
    assert (tmp_path / "canary_report.txt").exists()


def test_canary_is_reproducible_from_the_seed(tmp_path):
    """ctx.rng is the ONLY source of randomness -- same seed, same model."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(float)
    path = tmp_path / "d.csv"
    with path.open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["x1", "x2", "label"])
        wtr.writerows(np.column_stack([X, y]))

    def once():
        g = Graph()
        g.add("load", LoadCSV(path))
        g.add("fit", LogisticRegression())
        g.connect("load.data", "fit.data")
        out = BatchScheduler().run(
            g, Context(rng=np.random.default_rng(42), workdir=tmp_path)
        )
        return out["fit"]["model"].w

    assert np.allclose(once(), once())
