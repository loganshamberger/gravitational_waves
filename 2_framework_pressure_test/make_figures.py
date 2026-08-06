"""Render every scenario's structure and results to figures/.

Run:  python 2_framework_pressure_test/make_figures.py

Produces two families:
  graph_*.png    -- the wiring of each scenario, cycles marked
  result_*.png   -- what each simulation actually produced
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np

from core import BatchScheduler, Context, CycleError
from viz import (
    AQUA, BLUE, CYCLE_RED, INK, INK_MUTED, ORANGE,
    new_figure, plot_graph, save, _style,
)

import scenario_coupled_oscillators as coup
import scenario_gw_injection as gw
import scenario_oscillator as osc
import scenario_sdlc as sdlc

FIGS = HERE / "figures"


class Batch(BatchScheduler):
    ground_bindings = osc.BATCH_BINDINGS


class SdlcBatch(BatchScheduler):
    ground_bindings = sdlc.BATCH_BINDINGS


def ctx(seed=0):
    return Context(rng=np.random.default_rng(seed), workdir=FIGS)


written = []


def note(p):
    written.append(Path(p).name)
    return p


# ---------------------------------------------------------------------------
# Structure diagrams
# ---------------------------------------------------------------------------


def graphs():
    note(plot_graph(osc.build_driven_graph(), FIGS / "graph_01_driven_oscillator.png",
                    "Scenario 1 — driven oscillator (force wired in as an edge)"))
    note(plot_graph(sdlc.build_linear_graph(), FIGS / "graph_02_sdlc_linear.png",
                    "Scenario 2 — SDLC pipeline, rework buried inside `dev`"))
    note(plot_graph(sdlc.build_rework_graph(), FIGS / "graph_03_sdlc_rework.png",
                    "Scenario 2 — SDLC with the honest rework loop"))
    note(plot_graph(coup.build_cascade_graph(), FIGS / "graph_04_coupled_cascade.png",
                    "Scenario 3 — one-way cascade",
                    "acyclic — runs, but secularly divergent for equal masses"))
    note(plot_graph(coup.build_coupled_graph(), FIGS / "graph_05_coupled_cyclic.png",
                    "Scenario 3 — the honest bidirectional coupling",
                    "cycle — exact AND transparent, and the only one refused"))
    note(plot_graph(coup.build_monolith_graph(), FIGS / "graph_06_coupled_monolith.png",
                    "Scenario 3 — both masses in one node",
                    "acyclic — exact, but the coupling force has no port at all"))
    note(plot_graph(gw.build_injection_graph(), FIGS / "graph_07_gw_injection.png",
                    "Scenario 4 — GW injection and recovery"))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def oscillator_results():
    free = Batch().run(osc.build_free_graph(c=0.15, t_max=40.0), ctx())
    driven = Batch().run(
        osc.build_driven_graph(c=0.2, amplitude=1.0, omega_d=2.0, t_max=40.0), ctx()
    )
    fig, (a, b) = new_figure(2, 1, figsize=(9.5, 6.2))

    ft, dt_ = free["osc"]["trajectory"], driven["osc"]["trajectory"]
    a.plot(ft.t, ft.x, color=BLUE, lw=2, label="free (damped)")
    a.plot(dt_.t, dt_.x, color=ORANGE, lw=2, label="driven at resonance")
    a.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)
    _style(a, "Displacement", "t", "x(t)")

    a2, b2 = free["energy"]["energy"], driven["energy"]["energy"]
    b.plot(a2.t, a2.values, color=BLUE, lw=2, label="free — decays")
    b.plot(b2.t, b2.values, color=ORANGE, lw=2, label="driven — pumped to steady state")
    b.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)
    _style(b, "Energy", "t", "E(t)")
    note(save(fig, FIGS / "result_01_oscillator.png"))


def coupled_results():
    """The F5 money plot: the runnable transparent encoding diverges."""
    kc, m, k = 0.2, 1.0, 4.0
    fig, (a, b) = new_figure(2, 1, figsize=(9.5, 6.4))

    # One beat cycle only: over a long window the carrier fills in solid and
    # the envelope — the thing worth seeing — stops reading.
    mono = Batch().run(coup.build_monolith_graph(kc=kc, t_max=140.0, n=20001), ctx())
    t, x1, x2 = mono["pair"]["m1"].t, mono["pair"]["m1"].x, mono["pair"]["m2"].x
    a.plot(t, x1, color=BLUE, lw=1.0, label="mass 1")
    a.plot(t, x2, color=ORANGE, lw=1.0, label="mass 2")
    a.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, ncol=2,
             loc="lower left", bbox_to_anchor=(0, 1.0))
    _style(a, "Exact bidirectional coupling — energy beats between the masses",
           "t", "x(t)", pad=24)

    for tmax, alpha in ((480.0, 1.0),):
        casc = Batch().run(coup.build_cascade_graph(kc=kc, t_max=tmax, n=20001), ctx())
        ct, cx = casc["osc2"]["trajectory"].t, casc["osc2"]["trajectory"].x
        b.plot(ct, cx, color=CYCLE_RED, lw=1.0,
               label="one-way cascade (cycle removed) — diverges")
    mono2 = Batch().run(coup.build_monolith_graph(kc=kc, t_max=480.0, n=20001), ctx())
    b.plot(mono2["pair"]["m2"].t, mono2["pair"]["m2"].x, color=AQUA, lw=1.2,
           label="exact — bounded at |x| = 1")
    b.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, ncol=2,
             loc="lower left", bbox_to_anchor=(0, 1.0))
    _style(b, "Dropping the feedback edge changes the answer qualitatively",
           "t", "x₂(t)", pad=24)
    note(save(fig, FIGS / "result_02_coupled_divergence.png"))


def sdlc_results():
    g = sdlc.build_linear_graph(n_items=4000)
    g.nodes["dev"].capacity = 4000
    out = SdlcBatch().run(g, ctx(11))
    stages = [
        ("backlog", len(out["backlog"]["items"])),
        ("triaged", len(out["triage"]["ready"])),
        ("reviewed", len(out["dev"]["for_review"])),
        ("approved", len(out["review"]["approved"])),
        ("deployed", len(out["ci"]["passing"])),
    ]
    fig, (a, b) = new_figure(1, 2, figsize=(11, 4.2))
    names = [s for s, _ in stages]
    vals = [v for _, v in stages]
    a.barh(names[::-1], vals[::-1], color=BLUE, height=0.62, zorder=3)
    for i, v in enumerate(vals[::-1]):
        a.text(v + max(vals) * 0.015, i, f"{v:,}", va="center",
               color=INK, fontsize=9, zorder=4)
    a.set_xlim(0, max(vals) * 1.16)
    _style(a, "Token driver — items reaching each stage", "items", None)

    rates = sdlc.AnalyticQueueScheduler().run(sdlc.build_linear_graph(), ctx())
    sim = len(out["ci"]["passing"]) / len(out["dev"]["for_review"])
    ana = rates["ci"]["passing"].arrival_rate / rates["dev"]["for_review"].arrival_rate
    b.bar(["simulated\n(tokens)", "closed form\n(rates)"], [sim, ana],
          color=[BLUE, ORANGE], width=0.5, zorder=3)
    for i, v in enumerate((sim, ana)):
        b.text(i, v + 0.012, f"{v:.3f}", ha="center", color=INK, fontsize=10, zorder=4)
    b.set_ylim(0, max(sim, ana) * 1.25)
    _style(b, "Same graph, two drivers, two representations", None, "deployed fraction")
    note(save(fig, FIGS / "result_03_sdlc.png"))


def gw_results():
    out = Batch().run(gw.build_injection_graph(), ctx(3))
    wf = out["to_si"]["waveform"]
    data = out["noise"]["data"]
    clean = out["inject"]["data"]
    truth = out["inject"]["truth"]
    snr = out["filter"]["snr"]
    psd = out["noise"]["psd"]

    fig, axes = new_figure(2, 3, figsize=(16, 7))
    (a, e, c), (b, f_ax, d) = axes

    a.plot(wf.t, wf.h_plus, color=BLUE, lw=0.9, label="h+")
    a.plot(wf.t, wf.h_cross, color=ORANGE, lw=0.9, label="hx")
    a.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, ncol=2,
             loc="lower left", bbox_to_anchor=(0, 1.0))
    _style(a, "Injected chirp (SI strain)", "t [s]", "h", pad=32)

    ph = np.unwrap(np.angle(wf.h_plus + 1j * wf.h_cross))
    fr = np.gradient(ph, wf.t) / (2 * np.pi)
    e.plot(wf.t, fr, color=BLUE, lw=2)
    e.plot([wf.t[0], wf.t[-1]], [fr[0], fr[-1]], color=INK_MUTED, lw=1,
           ls="--", label="straight line, for reference")
    e.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED,
             loc="upper left")
    _style(e, f"Frequency sweep: {fr[0]:.0f} to {fr[-1]:.0f} Hz",
           "t [s]", "f_gw [Hz]")

    mask = (psd.t > 10) & (psd.t < 2000)
    c.loglog(psd.t[mask], np.sqrt(psd.values[mask]), color=BLUE, lw=2)
    c.text(0.97, 0.92, "toy aLIGO curve", transform=c.transAxes, ha="right",
           color=INK_MUTED, fontsize=9)
    _style(c, "Noise amplitude spectral density", "f [Hz]", "sqrt(S) [1/sqrt(Hz)]")

    def whiten(sig):
        n = len(sig.values)
        fq = np.fft.rfftfreq(n, sig.dt)
        sp = np.interp(fq, psd.t, psd.values)
        return np.fft.irfft(np.fft.rfft(sig.values) / np.sqrt(sp), n=n)

    wd = whiten(data)
    sigma = np.std(wd[len(wd) // 10 : -len(wd) // 10]) or 1.0
    b.plot(data.t, wd / sigma, color=INK_MUTED, lw=0.5, label="whitened data")
    b.plot(clean.t, whiten(clean) / sigma, color=ORANGE, lw=1.0,
           label="whitened signal")
    b.axvline(truth.t_inject, color=AQUA, lw=1.4, ls="--")
    b.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, ncol=2,
             loc="lower left", bbox_to_anchor=(0, 1.0))
    _style(b, "8 s segment, signal injected at a random time", "t [s]",
           "sigma", pad=30)

    zoom = (data.t > truth.t_inject - 0.3) & (data.t < truth.t_inject + 3.3)
    f_ax.plot(data.t[zoom], (wd / sigma)[zoom], color=INK_MUTED, lw=0.5)
    f_ax.plot(clean.t[zoom], (whiten(clean) / sigma)[zoom], color=ORANGE, lw=1.1)
    f_ax.axvline(truth.t_inject, color=AQUA, lw=1.4, ls="--")
    f_ax.text(truth.t_inject, f_ax.get_ylim()[1] * 0.82,
              f"  injected at {truth.t_inject:.3f} s", color=INK, fontsize=9)
    _style(f_ax, "Zoom on the injection", "t [s]", "sigma")

    i = int(np.argmax(snr.values))
    d.plot(snr.t, snr.values, color=AQUA, lw=1.0)
    d.axvline(truth.t_inject, color=INK_MUTED, lw=1.2, ls="--")
    d.plot([snr.t[i]], [snr.values[i]], "o", ms=9, color=AQUA, zorder=4)
    err_ms = 1e3 * (snr.t[i] - truth.t_inject)
    d.annotate(f"peak SNR {snr.values[i]:.1f}\nrecovered t = {snr.t[i]:.3f} s"
               f"\nerror {err_ms:+.2f} ms",
               (snr.t[i], snr.values[i]), textcoords="offset points",
               xytext=(-140, -30), color=INK, fontsize=9)
    _style(d, "Matched filter finds it", "lag [s]", "SNR")
    note(save(fig, FIGS / "result_04_gw_injection.png"))


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    graphs()
    oscillator_results()
    coupled_results()
    sdlc_results()
    gw_results()
    print(f"wrote {len(written)} figures to {FIGS}")
    for n in written:
        print("  ", n)
