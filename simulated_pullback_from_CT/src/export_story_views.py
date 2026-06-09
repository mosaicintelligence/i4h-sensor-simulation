from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[:, :, :3].copy()


def _back_view_angles(forward: np.ndarray) -> tuple[float, float]:
    """Return matplotlib (elev, azim) so the camera sits behind the probe and looks forward.

    matplotlib's 3D view places the viewer in the direction (elev, azim) and looks
    toward the data box center. To look forward along ``forward`` from behind the
    probe, we place the viewer along ``-forward``.
    """
    f = np.asarray(forward, dtype=np.float64)
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return 10.0, -60.0
    f = f / n
    back = -f
    azim = float(np.degrees(np.arctan2(back[1], back[0])))
    elev = float(np.degrees(np.arctan2(back[2], np.hypot(back[0], back[1]))))
    return elev, azim


def write_story_views(
    mesh_path: str | Path,
    poses_payload: dict,
    fps: int,
    camera_out_path: str | Path,
    god_out_path: str | Path,
    god_n_mesh_points: int = 20000,
    camera_local_radius_mm: float = 22.0,
    camera_view_radius_mm: float = 18.0,
    camera_back_margin_mm: float = 3.0,
    image_size_px: tuple[int, int] = (960, 960),
) -> None:
    mesh = trimesh.load_mesh(str(mesh_path), process=True)
    verts_all = np.asarray(mesh.vertices, dtype=np.float64)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    if faces_all.ndim != 2 or faces_all.shape[1] != 3:
        raise ValueError(f"Mesh at {mesh_path} is not triangular.")

    # Precompute per-triangle centroid and geometric normal once.
    tri_verts = verts_all[faces_all]  # (F, 3, 3)
    centroids = tri_verts.mean(axis=1)  # (F, 3)
    e1 = tri_verts[:, 1] - tri_verts[:, 0]
    e2 = tri_verts[:, 2] - tri_verts[:, 0]
    normals = np.cross(e1, e2)
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(n_norm, 1e-12)

    # Subsample vertices only for the god's-eye scatter density.
    if verts_all.shape[0] > god_n_mesh_points:
        rng = np.random.default_rng(123)
        gids = rng.choice(verts_all.shape[0], size=god_n_mesh_points, replace=False)
        god_verts = verts_all[gids]
    else:
        god_verts = verts_all

    poses = poses_payload["poses"]
    path_xyz = np.asarray([p["position_mm"] for p in poses], dtype=np.float64)

    camera_out = Path(camera_out_path)
    god_out = Path(god_out_path)
    camera_out.parent.mkdir(parents=True, exist_ok=True)
    god_out.parent.mkdir(parents=True, exist_ok=True)

    xz = god_verts[:, [0, 2]]
    pad = 10.0
    xlim = (float(xz[:, 0].min() - pad), float(xz[:, 0].max() + pad))
    zlim = (float(xz[:, 1].min() - pad), float(xz[:, 1].max() + pad))

    with imageio.get_writer(
        str(camera_out), fps=int(fps), codec="libx264", quality=8
    ) as cam_writer, imageio.get_writer(
        str(god_out), fps=int(fps), codec="libx264", quality=8
    ) as god_writer:
        for i, rec in enumerate(poses):
            probe = np.asarray(rec["position_mm"], dtype=np.float64)
            fwd = np.asarray(rec["tangent"], dtype=np.float64)
            fnorm = float(np.linalg.norm(fwd))
            if fnorm > 1e-9:
                fwd = fwd / fnorm

            # Filter triangles for the camera view:
            #   - within a local sphere of the probe (cull distant clutter)
            #   - in front of the probe along the tangent (cull "behind camera" clutter)
            #   - back-facing relative to the camera (we are INSIDE the lumen; visible
            #     wall triangles have outward normals pointing away from us, so
            #     dot(normal, centroid - probe) > 0)
            rel = centroids - probe
            dist = np.linalg.norm(rel, axis=1)
            fwd_proj = rel @ fwd
            in_sphere = dist <= float(camera_local_radius_mm)
            in_front = fwd_proj >= -float(camera_back_margin_mm)
            facing_inward = np.einsum("ij,ij->i", normals, rel) > 0.0
            keep = in_sphere & in_front & facing_inward
            if int(keep.sum()) < 64:
                # Fallback: if back-face cull was too aggressive (mesh winding
                # not as expected), keep all sphere+front triangles.
                keep = in_sphere & in_front

            local_face_idx = np.flatnonzero(keep)

            fig = plt.figure(
                figsize=(image_size_px[0] / 100, image_size_px[1] / 100), dpi=100
            )
            ax = fig.add_subplot(111, projection="3d")
            if local_face_idx.size > 0:
                ax.plot_trisurf(
                    verts_all[:, 0],
                    verts_all[:, 1],
                    verts_all[:, 2],
                    triangles=faces_all[local_face_idx],
                    color="#c9816a",
                    edgecolor="none",
                    linewidth=0.0,
                    antialiased=True,
                    shade=True,
                    alpha=1.0,
                )
            ax.scatter([probe[0]], [probe[1]], [probe[2]], s=40, c="red", depthshade=False)
            look = probe + fwd * 10.0
            ax.plot(
                [probe[0], look[0]],
                [probe[1], look[1]],
                [probe[2], look[2]],
                c="yellow",
                linewidth=2.0,
            )

            half = float(camera_view_radius_mm)
            ax.set_xlim(probe[0] - half, probe[0] + half)
            ax.set_ylim(probe[1] - half, probe[1] + half)
            ax.set_zlim(probe[2] - half, probe[2] + half)
            elev, azim = _back_view_angles(fwd)
            ax.view_init(elev=elev, azim=azim)
            ax.grid(False)
            ax.set_axis_off()
            ax.set_title(f"Camera-eye view | frame {i:03d}", fontsize=12)
            fig.tight_layout()
            cam_writer.append_data(_fig_to_rgb(fig))
            plt.close(fig)

            fig2 = plt.figure(
                figsize=(image_size_px[0] / 100, image_size_px[1] / 100), dpi=100
            )
            ax2 = fig2.add_subplot(111)
            ax2.scatter(god_verts[:, 0], god_verts[:, 2], s=0.9, c="#c9816a", alpha=1.0)
            ax2.plot(path_xyz[:, 0], path_xyz[:, 2], c="#2759d9", linewidth=1.8, alpha=0.95)
            ax2.scatter(path_xyz[: i + 1, 0], path_xyz[: i + 1, 2], s=5, c="#2759d9", alpha=0.9)
            ax2.scatter([probe[0]], [probe[2]], s=44, c="red")
            ax2.set_xlim(*xlim)
            ax2.set_ylim(*zlim)
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_title(f"God's-eye view (AP projection) | frame {i:03d}", fontsize=12)
            ax2.set_xlabel("left-right axis (x, mm)")
            ax2.set_ylabel("superior-inferior axis (z, mm)")
            ax2.grid(False)
            fig2.tight_layout()
            god_writer.append_data(_fig_to_rgb(fig2))
            plt.close(fig2)
