"""Smoke test: every vessel-generator material name is registered in raysim.

The vessel-generator manifest references named simulator materials. If the
simulator's ``Materials()`` table is rebuilt without one of them, downstream
``Materials::get_index(name)`` raises and the whole pullback breaks.

``Materials()`` uploads the table to the GPU, so a pure runtime check would
skip on CPU-only hosts (including typical CI). The always-on assertion below
parses ``material.cpp`` so missing names fail even without CUDA. When a CUDA
device is available, a second check exercises the live ``Materials`` table.
"""

from __future__ import annotations

import re
from pathlib import Path

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

_MATERIAL_ENTRY_RE = re.compile(
    r'\{\s*"(?P<name>[^"]+)"\s*,\s*Material\s*\(',
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATERIAL_CPP = (
    _REPO_ROOT
    / "i4h-sensor-simulation"
    / "ultrasound-raytracing"
    / "csrc"
    / "core"
    / "material.cpp"
)


def _material_names_in_cpp() -> set[str]:
    text = _MATERIAL_CPP.read_text(encoding="utf-8")
    return {match.group("name") for match in _MATERIAL_ENTRY_RE.finditer(text)}


@pytest.mark.parametrize("name", REQUIRED_MATERIAL_NAMES)
def test_material_name_present_in_material_cpp(name: str) -> None:
    assert _MATERIAL_CPP.is_file(), f"missing material table source: {_MATERIAL_CPP}"
    registered = _material_names_in_cpp()
    assert name in registered, (
        f"material {name!r} is required by vessel-generator but not registered in "
        f"{_MATERIAL_CPP.relative_to(_REPO_ROOT)}"
    )


@pytest.fixture(scope="module")
def materials():
    rs = pytest.importorskip(
        "raysim",
        reason="raysim native extension required for live Materials() smoke test",
    )
    if not hasattr(rs, "Materials"):
        pytest.skip("raysim built without Materials (CUDA extension missing)")
    try:
        return rs.Materials()
    except RuntimeError as exc:
        pytest.skip(f"raysim.Materials() unavailable (no CUDA device?): {exc}")


@pytest.mark.parametrize("name", REQUIRED_MATERIAL_NAMES)
def test_material_name_resolves_at_runtime(materials, name: str) -> None:
    idx = materials.get_index(name)
    assert isinstance(idx, int)
    assert idx >= 0
