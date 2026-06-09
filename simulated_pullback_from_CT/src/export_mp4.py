from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def polar_to_cart_display(arr_theta_r: np.ndarray, *, t_far_mm: float, cart_size: int = 512) -> np.ndarray:
    n_theta, n_r = arr_theta_r.shape
    half = float(t_far_mm)
    xs = np.linspace(-half, half, cart_size)
    ys = np.linspace(-half, half, cart_size)
    x, y = np.meshgrid(xs, ys)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(x, y) % (2.0 * np.pi)  # 0 at north, clockwise
    r_idx = (r / half) * (n_r - 1)
    t_idx = (theta / (2.0 * np.pi)) * n_theta
    r_idx = np.clip(r_idx, 0.0, n_r - 1.001)
    t_idx = t_idx % n_theta

    r0 = np.floor(r_idx).astype(np.int64)
    r1 = r0 + 1
    t0 = np.floor(t_idx).astype(np.int64) % n_theta
    t1 = (t0 + 1) % n_theta

    wr = r_idx - r0
    wt = t_idx - np.floor(t_idx)
    out = (
        arr_theta_r[t0, r0] * (1.0 - wt) * (1.0 - wr)
        + arr_theta_r[t1, r0] * wt * (1.0 - wr)
        + arr_theta_r[t0, r1] * (1.0 - wt) * wr
        + arr_theta_r[t1, r1] * wt * wr
    )
    out[r > half] = 0.0
    return out.astype(np.float32)


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float32)
    lo = float(np.nanpercentile(f, 1.0))
    hi = float(np.nanpercentile(f, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def write_polar_mp4(
    polar_stack: np.ndarray,
    t_far_mm: float,
    fps: int,
    out_path: str | Path,
    cart_size: int = 512,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out), fps=int(fps), codec="libx264", quality=8) as w:
        for frame in polar_stack:
            cart = polar_to_cart_display(frame, t_far_mm=t_far_mm, cart_size=cart_size)
            gray = _to_uint8(cart)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            w.append_data(rgb)
