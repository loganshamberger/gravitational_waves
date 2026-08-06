"""Scenario 4 tests -- the end-to-end injection.

These check that the PIPELINE is wired correctly and behaves the way the
physics demands qualitatively (scalings, monotonicity, unit guards). They do
NOT validate the waveform model, which is a deliberate toy -- see the fidelity
warning in scenario_gw_injection.py.
"""

import numpy as np
import pytest

from core import BatchScheduler, Context, Graph
from scenario_oscillator import BATCH_BINDINGS
from scenario_gw_injection import (
    ChirpSource,
    GeometricToSI,
    MatchedFilter,
    ProjectOntoDetector,
    Waveform,
    aligo_psd,
    build_injection_graph,
)


class Batch(BatchScheduler):
    ground_bindings = BATCH_BINDINGS


def ctx(tmp_path, seed=0):
    return Context(rng=np.random.default_rng(seed), workdir=tmp_path)


def peak_snr(tmp_path, **kw):
    out = Batch().run(build_injection_graph(**kw), ctx(tmp_path))
    return float(out["filter"]["snr"].values.max())


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_the_whole_pipeline_runs(tmp_path):
    out = Batch().run(build_injection_graph(), ctx(tmp_path))
    assert (tmp_path / "injection.txt").exists()
    assert set(out) == {
        "chirp", "to_si", "detector", "resample", "taper", "inject", "noise",
        "filter", "report",
    }


def test_strain_is_the_right_order_of_magnitude(tmp_path):
    """30+30 Msun at 400 Mpc should give h ~ 1e-21, like GW150914."""
    out = Batch().run(build_injection_graph(), ctx(tmp_path))
    h = np.max(np.abs(out["to_si"]["waveform"].h_plus))
    assert 1e-22 < h < 1e-20


def test_the_signal_is_recovered_above_threshold(tmp_path):
    assert peak_snr(tmp_path) > 8.0


def test_injection_is_reproducible_from_the_seed(tmp_path):
    a = Batch().run(build_injection_graph(), ctx(tmp_path, seed=7))
    b = Batch().run(build_injection_graph(), ctx(tmp_path, seed=7))
    assert np.array_equal(a["noise"]["data"].values, b["noise"]["data"].values)


def test_different_seeds_give_different_noise(tmp_path):
    a = Batch().run(build_injection_graph(), ctx(tmp_path, seed=1))
    b = Batch().run(build_injection_graph(), ctx(tmp_path, seed=2))
    assert not np.array_equal(a["noise"]["data"].values, b["noise"]["data"].values)


# ---------------------------------------------------------------------------
# Scalings that would catch a mis-wired pipeline
# ---------------------------------------------------------------------------


def test_snr_falls_as_one_over_distance(tmp_path):
    """Peak-over-lags is biased upward by the noise floor, and the bias grows
    as SNR falls -- so this checks the 1/D trend at loud distances with a
    tolerance that admits that bias, rather than pretending it is exact."""
    s100, s200, s400 = (peak_snr(tmp_path, distance_mpc=d) for d in (100., 200., 400.))
    assert s100 / s200 == pytest.approx(2.0, rel=0.06)
    assert s200 / s400 == pytest.approx(2.0, rel=0.06)


def test_snr_falls_as_one_over_noise_amplitude(tmp_path):
    """Regression guard: an earlier version reported the unscaled PSD, which
    made SNR independent of how noisy the data actually was."""
    quiet = peak_snr(tmp_path, noise_scale=0.5, distance_mpc=200.0)
    loud = peak_snr(tmp_path, noise_scale=4.0, distance_mpc=200.0)
    assert quiet / loud == pytest.approx(8.0, rel=0.15)


# ---------------------------------------------------------------------------
# Random injection time -- the filter has to SEARCH, not just confirm
# ---------------------------------------------------------------------------


def test_the_injection_time_is_random_and_reproducible(tmp_path):
    times = {
        Batch().run(build_injection_graph(), ctx(tmp_path, seed=s))["inject"][
            "truth"
        ].t_inject
        for s in range(5)
    }
    assert len(times) == 5, "injection time should vary with the seed"
    a = Batch().run(build_injection_graph(), ctx(tmp_path, seed=2))
    b = Batch().run(build_injection_graph(), ctx(tmp_path, seed=2))
    assert a["inject"]["truth"].t_inject == b["inject"]["truth"].t_inject


def test_the_signal_is_placed_inside_the_segment_with_margins(tmp_path):
    out = Batch().run(build_injection_graph(segment_s=8.0), ctx(tmp_path))
    truth = out["inject"]["truth"]
    chirp_len = out["taper"]["signal"].t[-1]
    assert 0.5 <= truth.t_inject <= 8.0 - chirp_len - 0.5


def test_the_matched_filter_recovers_the_injection_time(tmp_path):
    """The real measurement: the peak lag must land on where it was injected."""
    for seed in range(5):
        out = Batch().run(build_injection_graph(), ctx(tmp_path, seed=seed))
        snr, truth = out["filter"]["snr"], out["inject"]["truth"]
        recovered = snr.t[int(np.argmax(snr.values))]
        assert abs(recovered - truth.t_inject) < 0.005, f"seed {seed}"


def test_recovery_degrades_when_the_signal_is_too_quiet(tmp_path):
    """A control: at 20x the distance the peak is noise, not signal."""
    out = Batch().run(build_injection_graph(distance_mpc=8000.0), ctx(tmp_path, seed=1))
    snr, truth = out["filter"]["snr"], out["inject"]["truth"]
    recovered = snr.t[int(np.argmax(snr.values))]
    assert snr.values.max() < 8.0
    assert abs(recovered - truth.t_inject) > 0.005


def test_a_segment_too_short_for_the_signal_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot hold"):
        Batch().run(build_injection_graph(segment_s=1.0), ctx(tmp_path))


def test_edge_on_source_is_quieter_than_face_on(tmp_path):
    face = peak_snr(tmp_path, inclination=0.0)
    edge = peak_snr(tmp_path, inclination=np.pi / 2)
    assert face > 2 * edge


def test_the_chirp_frequency_rises_monotonically(tmp_path):
    """It is a chirp, not a sinusoid."""
    out = Batch().run(build_injection_graph(), ctx(tmp_path))
    wf = out["to_si"]["waveform"]
    phase = np.unwrap(np.angle(wf.h_plus + 1j * wf.h_cross))
    f = np.gradient(phase, wf.t) / (2 * np.pi)

    # An inspiral spends most of its time at low frequency, so quartile
    # medians understate the sweep -- test the sweep itself, and monotonicity.
    assert np.all(np.diff(f) > -1e-9), "frequency must not decrease"
    assert f[0] > 0
    assert f[-1] > 3 * f[0]


# ---------------------------------------------------------------------------
# Antenna pattern
# ---------------------------------------------------------------------------


def test_antenna_pattern_is_maximal_overhead():
    fp, fc = ProjectOntoDetector(theta=0.0, phi=0.0, psi=0.0).antenna_pattern()
    assert fp == pytest.approx(1.0)
    assert fc == pytest.approx(0.0, abs=1e-12)


def test_antenna_pattern_has_a_null():
    fp, fc = ProjectOntoDetector(
        theta=np.pi / 2, phi=np.pi / 4, psi=0.0
    ).antenna_pattern()
    assert abs(fp) < 1e-12 and abs(fc) < 1e-12


def test_a_source_in_the_null_produces_no_strain(tmp_path):
    """The physical statement. (Deliberately NOT phrased as 'SNR is small' --
    see the next test for why that would be a trap.)"""
    out = Batch().run(
        build_injection_graph(theta=np.pi / 2, phi=np.pi / 4, psi=0.0), ctx(tmp_path)
    )
    nulled = np.max(np.abs(out["detector"]["strain"].values))
    normal = np.max(
        np.abs(Batch().run(build_injection_graph(), ctx(tmp_path))["detector"]["strain"].values)
    )
    assert nulled < 1e-15 * normal


def test_matched_filter_snr_is_meaningless_for_a_null_template(tmp_path):
    """A trap this toy contains, documented rather than hidden.

    The graph taps its template off the same projected strain it injects, so in
    an antenna null BOTH are ~zero and SNR degenerates to 0/0 -- which comes
    back as a finite, entirely spurious number. Nothing here is wrong per
    stage; the composition is what makes it meaningless. A real search uses an
    independent template bank, which does not have this degeneracy.
    """
    snr = peak_snr(tmp_path, theta=np.pi / 2, phi=np.pi / 4, psi=0.0)
    assert snr > 1.0        # NOT small -- this is the point
    assert np.isfinite(snr)


# ---------------------------------------------------------------------------
# Units travel on the wire, and a NODE enforces them -- not the core
# ---------------------------------------------------------------------------


def test_units_are_carried_by_the_product(tmp_path):
    out = Batch().run(build_injection_graph(), ctx(tmp_path))
    assert out["chirp"]["waveform"].units == "geometric"
    assert out["to_si"]["waveform"].units == "SI"


def test_the_detector_refuses_geometric_input(tmp_path):
    """The core type-check passes -- both ports are `Waveform`. The refusal
    comes from the domain node's validate(), exactly as §3.3 requires.
    """
    g = Graph()
    g.add("chirp", ChirpSource(n=2000))
    g.add("detector", ProjectOntoDetector())
    g.connect("chirp.waveform", "detector.waveform")   # wires fine

    with pytest.raises(ValueError, match="needs SI strain"):
        Batch().run(g, ctx(tmp_path))


def test_the_converter_refuses_already_converted_input(tmp_path):
    node = GeometricToSI(60.0, 400.0)
    with pytest.raises(ValueError, match="needs geometric input"):
        node.validate(
            ctx(tmp_path),
            waveform=Waveform(np.zeros(2), np.zeros(2), np.zeros(2), units="SI"),
        )


def test_unknown_unit_system_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown unit system"):
        Waveform(np.zeros(2), np.zeros(2), np.zeros(2), units="cgs")


# ---------------------------------------------------------------------------
# L1 reuse
# ---------------------------------------------------------------------------


def test_resample_hits_the_requested_rate(tmp_path):
    out = Batch().run(build_injection_graph(sample_rate=2048.0), ctx(tmp_path))
    assert out["resample"]["signal"].dt == pytest.approx(1 / 2048.0)


def test_the_same_l1_resample_serves_this_and_the_oscillator(tmp_path):
    """The L1 rule: define it when it has two consumers. It has two."""
    from l1_nodes import Resample
    from kinds import Signal

    t = np.linspace(0, 1, 101)
    out = Resample(50.0).run(ctx(tmp_path), signal=Signal(t=t, values=np.sin(t)))
    assert out["signal"].dt == pytest.approx(1 / 50.0)


def test_psd_is_positive_and_bottoms_out_in_the_bucket():
    f = np.logspace(1, 3.5, 500)
    s = aligo_psd(f)
    assert np.all(s > 0)
    assert 50 < f[np.argmin(s)] < 400        # sensitivity bucket
