"""Vessel save / load roundtrip including layers, lesions, guidewire."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vesselgen.config import (
    BranchConfig,
    CalcificationLesionConfig,
    CenterlineConfig,
    CrossSectionConfig,
    GuidewireConfig,
    LayeredWallConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.vessel import Vessel


def _vessel_with_layers_and_lesions(seed: int = 0) -> Vessel:
    parent = BranchConfig(
        centerline=CenterlineConfig(length_mm=40.0, n_stations=24),
        cross_section=CrossSectionConfig(mean_radius_mm=4.5, distal_radius_mm=4.2),
        wall=LayeredWallConfig.trilaminar(total_thickness_mm=0.9),
        name="parent",
        seed=seed,
    )
    cfg = VesselConfig(
        parent=parent,
        side_branches=[],
        seed=seed,
        lesions=[
            CalcificationLesionConfig(
                arclength_frac=0.5, azimuth_deg=45.0, kind="hard",
                arc_extent_deg=55.0, axial_extent_mm=7.0, seed=10,
            ),
            CalcificationLesionConfig(
                arclength_frac=0.65, azimuth_deg=80.0, kind="soft_lipid",
                arc_extent_deg=45.0, axial_extent_mm=6.0,
                inner_offset_frac=0.25, outer_offset_frac=0.7, seed=11,
            ),
        ],
        guidewire=GuidewireConfig(
            diameter_mm=0.46, lateral_offset_mm=1.5, offset_azimuth_deg=20.0,
        ),
    )
    return Vessel.from_config(cfg)


def test_manifest_lists_all_surfaces_lesions_and_guidewire(tmp_path: Path):
    vessel = _vessel_with_layers_and_lesions()
    out_dir = vessel.save(tmp_path / "vessel_a")
    manifest = json.loads((out_dir / "vessel.json").read_text())

    surface_names = [s["name"] for s in manifest["surfaces"]]
    # 3-layer trilaminar wall: lumen + 2 interior interfaces. The
    # outermost adventitia boundary is no longer emitted (rays just
    # stay in adventitia past interface_02 until the FOV).
    assert surface_names == ["lumen", "interface_01", "interface_02"]
    surface_materials = [s["material"] for s in manifest["surfaces"]]
    assert surface_materials == ["intima", "media", "adventitia"]
    for s in manifest["surfaces"]:
        assert (out_dir / s["obj"]).exists()

    assert len(manifest["lesions"]) == 2
    assert {l["kind"] for l in manifest["lesions"]} == {"hard", "soft_lipid"}
    assert manifest["lesions"][0]["material"] == "calcified_plaque"
    assert manifest["lesions"][1]["material"] == "lipid_pool"
    for lesion in manifest["lesions"]:
        assert (out_dir / lesion["obj"]).exists()

    assert manifest["guidewire"] is not None
    assert manifest["guidewire"]["material"] == "tungsten"
    assert (out_dir / manifest["guidewire"]["obj"]).exists()
    assert manifest["world"]["background_material"] == "lumen"


def test_load_vessel_restores_layers_lesions_and_guidewire(tmp_path: Path):
    vessel = _vessel_with_layers_and_lesions(seed=2)
    out_dir = vessel.save(tmp_path / "vessel_b")
    loaded = Vessel.load(out_dir)

    assert len(loaded.surfaces) == len(vessel.surfaces)
    assert [s.name for s in loaded.surfaces] == [s.name for s in vessel.surfaces]
    assert [s.material_name for s in loaded.surfaces] == [
        s.material_name for s in vessel.surfaces
    ]

    assert len(loaded.lesions) == len(vessel.lesions)
    assert [l.material_name for l in loaded.lesions] == [
        l.material_name for l in vessel.lesions
    ]
    for orig, restored in zip(vessel.lesions, loaded.lesions):
        # Volumes should match within mesh-export rounding.
        np.testing.assert_allclose(restored.mesh.volume, orig.mesh.volume, rtol=0.05)

    assert loaded.guidewire is not None
    assert loaded.guidewire.material_name == vessel.guidewire.material_name
    np.testing.assert_allclose(
        loaded.guidewire.in_plane_position_mm, vessel.guidewire.in_plane_position_mm
    )


def test_legacy_single_layer_vessel_still_roundtrips(tmp_path: Path):
    """A vessel without layers/lesions/guidewire still works."""
    parent = BranchConfig(
        centerline=CenterlineConfig(length_mm=40.0, n_stations=24),
        cross_section=CrossSectionConfig(mean_radius_mm=4.5, distal_radius_mm=4.2),
        wall=WallConfig(mean_thickness_mm=0.8, max_perturbation_frac=0.4),
        name="parent",
        seed=0,
    )
    cfg = VesselConfig(parent=parent, side_branches=[], seed=0)
    vessel = Vessel.from_config(cfg)
    out_dir = vessel.save(tmp_path / "vessel_legacy")

    manifest = json.loads((out_dir / "vessel.json").read_text())
    assert [s["name"] for s in manifest["surfaces"]] == ["lumen", "outer"]
    assert manifest["lesions"] == []
    assert manifest["guidewire"] is None

    loaded = Vessel.load(out_dir)
    assert len(loaded.surfaces) == 2
    assert loaded.guidewire is None
    assert loaded.lesions == []


def test_layered_vessel_writes_interior_interfaces_under_surfaces(tmp_path: Path):
    """A 3-layer wall writes lumen.obj + surfaces/interface_NN.obj only."""
    vessel = _vessel_with_layers_and_lesions()
    out_dir = vessel.save(tmp_path / "vessel_compat")
    assert (out_dir / "lumen.obj").exists()
    assert (out_dir / "surfaces" / "interface_01.obj").exists()
    assert (out_dir / "surfaces" / "interface_02.obj").exists()
    # The outer adventitia boundary is no longer emitted.
    assert not (out_dir / "outer.obj").exists()


def test_legacy_single_slab_still_emits_outer_obj(tmp_path: Path):
    """Backward compat: legacy single-slab walls keep lumen.obj + outer.obj."""
    parent = BranchConfig(
        centerline=CenterlineConfig(length_mm=40.0, n_stations=24),
        cross_section=CrossSectionConfig(mean_radius_mm=4.5, distal_radius_mm=4.2),
        wall=WallConfig(mean_thickness_mm=0.8, max_perturbation_frac=0.4),
        name="parent",
        seed=0,
    )
    cfg = VesselConfig(parent=parent, side_branches=[], seed=0)
    vessel = Vessel.from_config(cfg)
    out_dir = vessel.save(tmp_path / "vessel_legacy_compat")
    assert (out_dir / "lumen.obj").exists()
    assert (out_dir / "outer.obj").exists()
