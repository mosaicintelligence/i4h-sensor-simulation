"""CUDA smoke: num_el_samples > 1 must collapse to one RF plane.

The historical failure was cudaErrorIllegalAddress when
elevational_height_mm > 0 and num_elevational_samples > 1, because
post-Hilbert kernels still launched with size.z = N over a 1-plane
buffer. Collapse now runs even when conv_psf is false.

Skipped when the CUDA extension is missing or no GPU is visible.
Not collected by `pytest vessel-generator/tests/`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

N_EL = 8
N_SCANLINES = 64
BUFFER_SIZE = 4096  # Hilbert FFT length is a compile-time constant.
B_MODE_SIZE = (64, 64)  # (width, height) as SimParams.b_mode_size


def _cuda_rs():
    pytest.importorskip("raysim", reason="raysim package required")
    try:
        import raysim.cuda as rs
    except ImportError as exc:
        pytest.skip(f"raysim CUDA extension not built: {exc}")
    try:
        rs.Materials()
    except RuntimeError as exc:
        msg = str(exc)
        if "cudaErrorNoDevice" in msg or "no CUDA-capable device" in msg:
            pytest.skip(f"no CUDA device: {exc}")
        raise
    return rs


@pytest.mark.parametrize("conv_psf", [True, False])
def test_elevational_samples_collapse_to_one_plane(conv_psf: bool) -> None:
    rs = _cuda_rs()
    materials = rs.Materials()
    world = rs.World("water")
    # OptiX GAS build requires at least one primitive.
    world.add(
        rs.Sphere(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            0.4,
            materials.get_index("vessel_wall"),
        )
    )
    sim = rs.RaytracingUltrasoundSimulator(world, materials)

    probe = rs.IVUSProbe(
        rs.Pose(position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0]),
        N_SCANLINES,
        10.0,
        1.5,
        N_EL,
        1.0,
        1.54,
        2.0,
    )
    p = rs.SimParams()
    p.conv_psf = conv_psf
    p.buffer_size = BUFFER_SIZE
    p.t_far = 8.0
    p.b_mode_size = B_MODE_SIZE
    p.noise_sigma = 0.0

    img = np.asarray(sim.simulate(probe, p))
    # Python simulate() returns B-mode only. Collapse is the 3D→2D RF contract
    # (length buffer_size * num_elements). A 2D B-mode of the requested size
    # is the equivalent observable: a leftover elevational axis would be N
    # times larger or would illegal-address in Hilbert / envelope LPF.
    ny, nx = int(B_MODE_SIZE[1]), int(B_MODE_SIZE[0])
    assert img.ndim == 2
    assert img.shape == (ny, nx)
    assert np.isfinite(img).all()
