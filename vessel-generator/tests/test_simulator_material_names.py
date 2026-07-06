"""Smoke test: every vessel-generator material name resolves in raysim.

The vessel-generator manifest references named simulator materials. If the
simulator's ``Materials()`` table is rebuilt without one of them, downstream
``Materials::get_index(name)`` raises and the whole pullback breaks. This
test fails fast if any required material disappears from the simulator
build.

The test is skipped automatically when raysim isn't installed (e.g. CPU-only
CI workers) so it doesn't gate development on a CUDA build.
"""

from __future__ import annotations

import pytest

# These are the simulator material names the upgraded vessel-generator
# can emit. Kept here (not imported from the generator) so the test
# fails loudly even if a refactor accidentally drops one from the
# generator side too.
REQUIRED_MATERIAL_NAMES = (
    # Legacy / background materials still used by single-layer vessels
    # and the existing pullback config.
    "lumen",
    "vessel_wall",
    "extravascular",
    # Trilaminar wall layers (new).
    "intima",
    "media",
    "adventitia",
    # Atherosclerotic plaque components (new).
    "calcified_plaque",
    "lipid_pool",
    "fibrous_plaque",
    "thrombus",
    # Guidewire material (already in the table; this test guards it
    # against accidental removal because the generator now depends on
    # it for the guidewire mesh).
    "tungsten",
)


@pytest.fixture(scope="module")
def materials():
    rs = pytest.importorskip(
        "raysim",
        reason="raysim native extension required for material-name smoke test",
    )
    if not hasattr(rs, "Materials"):
        pytest.skip("raysim built without Materials (CUDA extension missing)")
    try:
        return rs.Materials()
    except RuntimeError as exc:
        pytest.skip(f"raysim.Materials() unavailable (no CUDA device?): {exc}")


@pytest.mark.parametrize("name", REQUIRED_MATERIAL_NAMES)
def test_material_name_resolves(materials, name):
    idx = materials.get_index(name)
    assert isinstance(idx, int)
    assert idx >= 0
