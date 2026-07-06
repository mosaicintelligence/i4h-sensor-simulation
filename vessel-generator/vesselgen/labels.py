"""Stable segmentation-label IDs and material-name -> label mapping.

The IDs below are the canonical per-pixel labels for paired-dataset
segmentation masks. They cover every distinct material the vessel
generator can emit (lumen + each wall layer + each in-wall lesion kind +
guidewire), plus the surrounding extravascular space and an explicit
background class for pixels outside the imaging FOV.

Conventions:

* The IDs are dense, monotone, and never re-used: future additions
  should append at the end, never re-number existing classes. Saved
  paired-dataset manifests record the integer IDs by name so old
  manifests stay valid even as the table grows.

* ``material_to_label`` maps the simulator material strings produced by
  :class:`vesselgen.io` (the manifest's ``surfaces[*].material``,
  ``lesions[*].material``, ``guidewire.material``) to label IDs. Use it
  when rasterising any closed mesh into the paired-dataset segmentation
  image.

* ``LABEL_WALL_LIKE`` is a convenience alias used to compute the legacy
  "wall" mask (union of the three named tissue layers + the legacy
  single-slab ``vessel_wall`` material). Existing models that train
  against the legacy 4-class (background, lumen, wall, extravascular)
  scheme can collapse the new fine-grained labels with one np.isin call.
"""

from __future__ import annotations

LABEL_BACKGROUND: int = 0
LABEL_LUMEN: int = 1
LABEL_INTIMA: int = 2
LABEL_MEDIA: int = 3
LABEL_ADVENTITIA: int = 4
LABEL_PERI_ADVENTITIA: int = 5
"""Peri-adventitia / extravascular space outside the modelled wall.

Legacy renders (pre 2026-06 world-model change) emitted this for any
in-FOV pixel outside the outer mesh. The new pipeline drops the outer
mesh and considers everything beyond ``interface_02`` to be
``LABEL_ADVENTITIA``, so this id is retained only for backward
compatibility -- new datasets never emit it.
"""
LABEL_CALCIFIED_PLAQUE: int = 6
LABEL_LIPID_POOL: int = 7
LABEL_FIBROUS_PLAQUE: int = 8
LABEL_THROMBUS: int = 9
LABEL_GUIDEWIRE: int = 10
LABEL_VESSEL_WALL: int = 11  # legacy single-slab wall (no layered breakdown)


# Backward-compatible alias for any caller that still imports the old name.
LABEL_EXTRAVASCULAR: int = LABEL_PERI_ADVENTITIA


LABEL_NAMES: dict[str, int] = {
    "background": LABEL_BACKGROUND,
    "lumen": LABEL_LUMEN,
    "intima": LABEL_INTIMA,
    "media": LABEL_MEDIA,
    "adventitia": LABEL_ADVENTITIA,
    "peri_adventitia": LABEL_PERI_ADVENTITIA,
    "extravascular": LABEL_PERI_ADVENTITIA,  # legacy alias
    "calcified_plaque": LABEL_CALCIFIED_PLAQUE,
    "lipid_pool": LABEL_LIPID_POOL,
    "fibrous_plaque": LABEL_FIBROUS_PLAQUE,
    "thrombus": LABEL_THROMBUS,
    "guidewire": LABEL_GUIDEWIRE,
    "vessel_wall": LABEL_VESSEL_WALL,
}


material_to_label: dict[str, int] = {
    "lumen": LABEL_LUMEN,
    "intima": LABEL_INTIMA,
    "media": LABEL_MEDIA,
    "adventitia": LABEL_ADVENTITIA,
    "vessel_wall": LABEL_VESSEL_WALL,
    "peri_adventitia": LABEL_PERI_ADVENTITIA,
    "extravascular": LABEL_PERI_ADVENTITIA,  # legacy material name still maps to id 5
    "calcified_plaque": LABEL_CALCIFIED_PLAQUE,
    "lipid_pool": LABEL_LIPID_POOL,
    "fibrous_plaque": LABEL_FIBROUS_PLAQUE,
    "thrombus": LABEL_THROMBUS,
    "tungsten": LABEL_GUIDEWIRE,
}


LABEL_WALL_LIKE: tuple[int, ...] = (
    LABEL_INTIMA,
    LABEL_MEDIA,
    LABEL_ADVENTITIA,
    LABEL_VESSEL_WALL,
)
"""Labels that collapse to the legacy 4-class "wall" mask."""


LABEL_LESIONS: tuple[int, ...] = (
    LABEL_CALCIFIED_PLAQUE,
    LABEL_LIPID_POOL,
    LABEL_FIBROUS_PLAQUE,
    LABEL_THROMBUS,
)
"""All in-wall lesion / inclusion labels."""


# Display palette used by the paired-dataset overlay PNGs. Background +
# guidewire intentionally use distinct, saturated hues so they pop on the
# log-compressed B-mode; all wall layers share a warm-red family so the
# legacy "wall" mask still reads as a single colour at a glance.
LABEL_RGBA: dict[int, tuple[float, float, float, float]] = {
    LABEL_LUMEN: (0.20, 0.85, 0.30, 0.35),
    LABEL_INTIMA: (1.00, 0.55, 0.55, 0.45),
    LABEL_MEDIA: (1.00, 0.25, 0.25, 0.45),
    LABEL_ADVENTITIA: (0.75, 0.10, 0.10, 0.50),
    LABEL_VESSEL_WALL: (1.00, 0.25, 0.25, 0.45),
    LABEL_PERI_ADVENTITIA: (0.35, 0.55, 1.00, 0.30),
    LABEL_CALCIFIED_PLAQUE: (1.00, 1.00, 0.20, 0.65),
    LABEL_LIPID_POOL: (0.90, 0.55, 0.10, 0.55),
    LABEL_FIBROUS_PLAQUE: (0.95, 0.80, 0.40, 0.55),
    LABEL_THROMBUS: (0.55, 0.30, 0.85, 0.55),
    LABEL_GUIDEWIRE: (0.10, 0.95, 0.95, 0.85),
}


def manifest_label_dict() -> dict[str, int]:
    """Return the canonical name -> ID mapping for paired-dataset manifests."""
    return dict(LABEL_NAMES)
