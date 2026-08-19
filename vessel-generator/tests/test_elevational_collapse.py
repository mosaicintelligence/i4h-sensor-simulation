"""GPU smoke: num_el_samples > 1 must not illegal-address after mean_planes.

The historical failure was cudaErrorIllegalAddress when
elevational_height_mm > 0 and num_elevational_samples > 1, because
post-Hilbert kernels still launched with size.z = N over a 1-plane
buffer. Collapse now runs even when conv_psf is false.

Skipped only when raysim was not built with the CUDA extension.
"""

from __future__ import annotations

import numpy as np
import pytest


def _cuda_rs():
    pytest.importorskip("raysim", reason="raysim package required")
    try:
        import raysim.cuda as rs
    except ImportError as exc:
        pytest.skip(f"raysim CUDA extension not built: {exc}")
    return rs


def _require_cuda(exc: BaseException) -> None:
    msg = str(exc)
    if "cudaErrorNoDevice" in msg or "no CUDA-capable device" in msg:
        pytest.fail(
            "raysim is built but CUDA sees no device. Run `nvidia-smi` in this "
            "shell without sudo; if that fails, `newgrp vglusers` then "
            f"`conda activate ultrasound`. Original error: {exc}"
        )
    raise exc


@pytest.mark.parametrize("conv_psf", [True, False])
def test_elevational_samples_simulate_without_illegal_address(conv_psf: bool) -> None:
    rs = _cuda_rs()
    try:
        materials = rs.Materials()
        world = rs.World("water")
        # OptiX GAS build requires at least one primitive; an empty world
        # fails with OPTIX_ERROR_INVALID_VALUE / "buildInputs is null".
        world.add(
            rs.Sphere(
                np.array([2.0, 0.0, 0.0], dtype=np.float32),
                0.4,
                materials.get_index("vessel_wall"),
            )
        )
        sim = rs.RaytracingUltrasoundSimulator(world, materials)
    except RuntimeError as exc:
        _require_cuda(exc)

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
    # Hilbert FFT size is a compile-time constant (4096 samples).
    p.buffer_size = 4096
    p.t_far = 8.0
    p.b_mode_size = (64, 64)
    p.noise_sigma = 0.0

    img = sim.simulate(probe, p)
    assert img is not None
    arr = np.asarray(img)
    assert arr.size > 0
    assert np.isfinite(arr).all()
