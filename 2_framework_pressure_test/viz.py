"""Visualization: plotting sinks, and a renderer for graph structure itself.

Two distinct jobs, both of which the design calls for:

  * §3.1 -- "visualization is a sink over any product; it is no longer a method
    welded to one System." `PlotSignal` below is one sink that serves all four
    scenarios, because they all speak the L1 `Signal` kind.
  * §3.3/§6 -- the structure is now a first-class object, so it can be DRAWN.
    `plot_graph` renders any Graph, marks cycle members, and draws back-edges
    distinctly. The old System catalog could not do this at all: there was no
    structure to render.

Palette: categorical slots 1-3 of the validated default (blue/orange/aqua),
which clear the all-pairs CVD and normal-vision gates in light mode. Every
series is directly labelled or legended, which satisfies the relief rule the
validator flags for aqua's sub-3:1 contrast.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from core import Context, Graph, Process
from kinds import Signal
from scenario_oscillator import SIGNAL

# Validated categorical slots (light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#fcfcfb"
CYCLE_RED = "#e34948"


def _style(ax, title=None, xlabel=None, ylabel=None, pad=8):
    """Recessive grid and axes; ink for text, never a series colour.

    `pad` needs raising to ~24 when the caller puts a legend above the axes,
    or the legend and the title collide.
    """
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=pad)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    return ax


def new_figure(nrows=1, ncols=1, figsize=(10, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    return fig, axes


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plotting sinks -- ordinary Processes [§3.1]
# ---------------------------------------------------------------------------


class PlotSignal(Process):
    """Sink over ANY `Signal`. One node, four scenarios.

    This is the design claim made concrete: the same sink plots a driving
    force, an energy trace, a detector strain and an SNR series, because they
    are all the same L1 kind.
    """

    inputs = {"signal": SIGNAL}
    outputs = {}

    def __init__(self, filename: str, title=None, xlabel="t", ylabel=None, logy=False):
        self.filename = filename
        self.title = title
        self.xlabel, self.ylabel, self.logy = xlabel, ylabel, logy

    def run(self, ctx: Context, signal: Signal):
        fig, ax = new_figure(figsize=(9, 3.6))
        ax.plot(signal.t, signal.values, color=BLUE, linewidth=2, label=signal.name)
        if self.logy:
            ax.set_yscale("log")
        _style(ax, self.title or signal.name, self.xlabel, self.ylabel or signal.name)
        save(fig, ctx.workdir / self.filename)
        ctx.logger.info("wrote %s", self.filename)
        return {}


# ---------------------------------------------------------------------------
# Rendering the STRUCTURE
# ---------------------------------------------------------------------------


def plot_graph(graph: Graph, path: Path, title: str = "", note: str = ""):
    """Draw a Graph as a layered diagram, marking cycles.

    Layout is by longest-path depth (columns) with nodes stacked within a
    column. Back-edges -- those pointing to an equal or shallower column -- are
    drawn dashed and red, which is exactly where a cycle shows up visually.
    """
    depth = graph.depths()
    cyc = graph.cycle_nodes()

    columns: dict[int, list[str]] = {}
    for n in sorted(graph.nodes, key=lambda n: (depth[n], n)):
        columns.setdefault(depth[n], []).append(n)

    pos, max_rows = {}, max(len(c) for c in columns.values())
    for d, names in columns.items():
        for i, n in enumerate(names):
            pos[n] = (d * 3.6, -(i - (len(names) - 1) / 2) * 1.6)

    fig, ax = new_figure(figsize=(2.6 * len(columns) + 3.0, 1.5 * max_rows + 2.4))
    ax.set_axis_off()
    ax.set_facecolor(SURFACE)

    box_w, box_h = 2.05, 0.78
    for n, (x, y) in pos.items():
        in_cycle = n in cyc
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.06,rounding_size=0.12",
                linewidth=2,
                edgecolor=CYCLE_RED if in_cycle else BLUE,
                facecolor="#fdecec" if in_cycle else "#eaf2fc",
                zorder=2,
            )
        )
        node = graph.nodes[n]
        ax.text(x, y + 0.12, n, ha="center", va="center",
                color=INK, fontsize=10, fontweight="bold", zorder=3)
        ax.text(x, y - 0.17, node.label, ha="center", va="center",
                color=INK_MUTED, fontsize=7.5, zorder=3)

    # Parallel edges between the same pair must be fanned apart, or they draw
    # exactly on top of each other and the diagram under-reports the wiring.
    parallel: dict[tuple[str, str], list[int]] = {}
    for i, e in enumerate(graph.edges):
        parallel.setdefault((e.src, e.dst), []).append(i)

    for i, e in enumerate(graph.edges):
        x0, y0 = pos[e.src]
        x1, y1 = pos[e.dst]
        back = depth[e.dst] <= depth[e.src]
        # A forward edge spanning more than one column would be drawn straight
        # THROUGH the nodes in between; route it under them instead.
        skip = not back and depth[e.dst] > depth[e.src] + 1

        siblings = parallel[(e.src, e.dst)]
        k, j = len(siblings), siblings.index(i)
        fan = (j - (k - 1) / 2) * 0.22          # 0 when there is only one edge

        span = abs(x1 - x0) or 1.0
        if back:
            # above the row
            start, end = (x0, y0 + box_h / 2), (x1, y1 + box_h / 2)
            rad = min(0.6, 2.6 / span) * (1.0 if x1 < x0 else -1.0)
        elif skip:
            # below the row
            start, end = (x0, y0 - box_h / 2), (x1, y1 - box_h / 2)
            rad = min(0.6, 2.6 / span) * (1.0 if x1 > x0 else -1.0)
        else:
            start = (x0 + box_w / 2, y0 + fan)
            end = (x1 - box_w / 2, y1 + fan)
            rad = 0.0 if abs(y1 - y0) < 1e-9 else 0.12
        ax.add_patch(
            FancyArrowPatch(
                start, end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>", mutation_scale=13,
                linewidth=1.6,
                linestyle="--" if back else "-",
                color=CYCLE_RED if back else INK_MUTED,
                zorder=1,
            )
        )
        # Skip-edge labels go below the arc; forward labels above the boxes.
        label_dy = 0.0 if (back or skip) else 0.52
        # Label at the Bezier apex. matplotlib's arc3 puts the control point at
        # mid + (rad*dy, -rad*dx), so the t=0.5 point is mid + half of that.
        dx, dy = end[0] - start[0], end[1] - start[1]
        mx = (start[0] + end[0]) / 2 + 0.5 * rad * dy
        # Forward labels must clear the box height, or they land on a node.
        my = (start[1] + end[1]) / 2 - 0.5 * rad * dx + label_dy
        ax.text(mx, my, f"{e.src_port}→{e.dst_port}", ha="center", va="center",
                color=CYCLE_RED if back else INK_MUTED, fontsize=7,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.4), zorder=4)

    subtitle = note or (
        f"cycle: {', '.join(sorted(cyc))} — batch scheduler refuses this"
        if cyc else "acyclic — runs under the batch scheduler"
    )
    ax.set_title(title or "graph", color=INK, fontsize=12, loc="left", pad=16)
    ax.text(0, 1.0, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            color=CYCLE_RED if cyc else INK_MUTED, fontsize=9)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    # Back-edges arc above the top row, skip-edges below the bottom one.
    has_back = any(depth[e.dst] <= depth[e.src] for e in graph.edges)
    has_skip = any(depth[e.dst] > depth[e.src] + 1 for e in graph.edges)
    ax.set_xlim(min(xs) - 1.8, max(xs) + 1.8)
    ax.set_ylim(min(ys) - (2.2 if has_skip else 1.2),
                max(ys) + (2.2 if has_back else 1.2))
    return save(fig, path)
