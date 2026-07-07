"""Parametric centerlines with local frames.

A :class:`Centerline` knows its position, tangent, normal and binormal as
continuous functions of arclength. The downstream sweep code uses these to
build a tube; the sampling code uses them to define the local "vessel
axis" so that probe-tilt sampling is well-defined at any sample point.

v1 implements straight centerlines only. The class exposes ``position``,
``tangent``, ``normal``, ``binormal`` and ``frame`` as functions of
arclength so v2 can drop in a curved centerline (smooth 3D B-spline,
random-walk tortuosity, etc.) without changing any caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vesselgen.config import CenterlineConfig


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """Right-handed local frame at a station along a centerline.

    All three vectors are unit-length. ``tangent`` is the direction of
    increasing arclength along the centerline; ``normal`` and ``binormal``
    span the cross-section plane.
    """

    position: np.ndarray  # shape (3,)
    tangent: np.ndarray  # shape (3,)
    normal: np.ndarray  # shape (3,)
    binormal: np.ndarray  # shape (3,)

    def to_world(self, x_local: float, y_local: float) -> np.ndarray:
        """Map a 2D point in the cross-section plane to 3D world coords.

        ``x_local`` is along ``normal``, ``y_local`` is along ``binormal``.
        """
        return self.position + x_local * self.normal + y_local * self.binormal


# ---------------------------------------------------------------------------
# Centerline
# ---------------------------------------------------------------------------


def _orthonormal_basis_for(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pick a deterministic (normal, binormal) basis perpendicular to ``direction``.

    For a straight centerline the local frame is constant. We pick a
    reproducible basis: project the world ``+x`` axis into the plane
    orthogonal to ``direction``; if that is degenerate (centerline is along
    ``+x``), fall back to ``+z``.
    """
    t = direction / np.linalg.norm(direction)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(t, candidate)) > 0.95:
        candidate = np.array([0.0, 0.0, 1.0])
    n = candidate - np.dot(candidate, t) * t
    n = n / np.linalg.norm(n)
    b = np.cross(t, n)
    return n, b


class Centerline:
    """A parametric 3D centerline with a local frame at every arclength.

    Public read-only attributes:
        config:            the :class:`CenterlineConfig` used to build it.
        length_mm:         total arclength.
        stations:          (N,) arclengths at which cross-sections are sampled.
        positions:         (N, 3) positions at each station.
        tangents:          (N, 3) tangent vectors at each station.
        normals:           (N, 3) normal vectors at each station.
        binormals:         (N, 3) binormal vectors at each station.
        origin:            shape (3,), proximal endpoint.
        end:               shape (3,), distal endpoint.
        direction:         shape (3,), unit tangent (constant for v1).
    """

    def __init__(self, config: CenterlineConfig):
        self.config = config
        self.length_mm = float(config.length_mm)
        self.origin = np.asarray(config.origin, dtype=float)
        self.direction = np.asarray(config.direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.end = self.origin + self.length_mm * self.direction

        normal, binormal = _orthonormal_basis_for(self.direction)
        self._normal = normal
        self._binormal = binormal

        self.stations = np.linspace(0.0, self.length_mm, config.n_stations)
        self.positions = self.origin[None, :] + self.stations[:, None] * self.direction[None, :]
        self.tangents = np.tile(self.direction, (config.n_stations, 1))
        self.normals = np.tile(normal, (config.n_stations, 1))
        self.binormals = np.tile(binormal, (config.n_stations, 1))

    # -----------------------------------------------------------------
    # Continuous queries (used by the sampling API)
    # -----------------------------------------------------------------

    def position(self, s: float) -> np.ndarray:
        return self.origin + float(np.clip(s, 0.0, self.length_mm)) * self.direction

    def tangent(self, s: float) -> np.ndarray:  # noqa: ARG002 - constant for v1
        return self.direction.copy()

    def frame(self, s: float) -> Frame:
        """Local right-handed frame at arclength ``s``."""
        return Frame(
            position=self.position(s),
            tangent=self.tangent(s),
            normal=self._normal.copy(),
            binormal=self._binormal.copy(),
        )

    def project(self, world_point: np.ndarray) -> tuple[float, np.ndarray]:
        """Project a 3D point onto the centerline.

        Returns (arclength_along_centerline, perpendicular_offset_3d). The
        arclength is clipped to the centerline range; the offset is the
        component of the point relative to the centerline that lies in the
        plane orthogonal to the local tangent (always orthogonal to
        ``direction`` for a straight centerline).
        """
        rel = np.asarray(world_point, dtype=float) - self.origin
        s = float(np.clip(np.dot(rel, self.direction), 0.0, self.length_mm))
        offset = rel - np.dot(rel, self.direction) * self.direction
        return s, offset

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "length_mm": self.length_mm,
            "origin": self.origin.tolist(),
            "direction": self.direction.tolist(),
            "n_stations": int(self.config.n_stations),
            "frame_normal": self._normal.tolist(),
            "frame_binormal": self._binormal.tolist(),
        }
