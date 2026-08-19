"""GPU smoke: num_el_samples > 1 must not illegal-address after mean_planes.

The historical failure was cudaErrorIllegalAddress when
elevational_height_mm > 0 and num_elevational_samples > 1, because
post-Hilbert kernels still launched with size.z = N over a 1-plane
buffer. Collapse now runs even when conv_psf is false.

Skipped on hosts without a CUDA-built raysim / device (typical CI).
"""

from __future__ import annotations

import numpy as np
import pytest


def _cuda_rs():
    pytest.importorskip("raysim", reason="raysim package required")
    try:
        import raysim.cuda as rs
    except Exception as exc:  # noqa: BLE001 — skip if extension missing
        pytest.skip(f"raysim.cuda unavailable: {exc}")
    return rs


@pytest.mark.parametrize("conv_psf", [True, False])
def test_elevational_samples_simulate_without_illegal_address(conv_psf: bool) -> None:
    rs = _cuda_rs()
    try:
        materials = rs.Materials()
        world = rs.World("water")
        sim = rs.RaytracingUltrasoundSimulator(world, materials)
    except RuntimeError as exc:
        pytest.skip(f"no CUDA device?: {exc}")

    probe = rs.IVUSProbe(
        rs.Pose(position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0]),
        64,
        10.0,
        1.5,
        8,
        1.0,
        1.54,
        2.0,
    )
    p = rs.SimParams()
    p.conv_psf = conv_psf
    p.buffer_size = 512
    p.t_far = 8.0
    p.b_mode_size = (64, 64)
    p.noise_sigma = 0.0

    img = sim.simulate(probe, p)
    assert img is not None
    arr = np.asarray(img)
    assert arr.size > 0
    assert np.isfinite(arr).all()
