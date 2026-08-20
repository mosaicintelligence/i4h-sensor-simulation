"""CPU-only: Twin demo-app YAML overlays stock names, it does not replace them."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_RAYSIM_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__name__ != "ScikitBuildRedirectingFinder"
]
if str(_RAYSIM_ROOT) in sys.path:
    sys.path.remove(str(_RAYSIM_ROOT))
sys.path.insert(0, str(_RAYSIM_ROOT))

from raysim.config import IvusSimConfig

REPO = Path(__file__).resolve().parents[3]
TWIN_DEMO = REPO / "instrument-calibration" / "p035_visions" / "twin_demo.yaml"

REQUIRED = (
    "intima",
    "media",
    "adventitia",
    "calcified_plaque",
    "lipid_pool",
    "fibrous_plaque",
    "thrombus",
)


def test_twin_demo_yaml_is_overlay_only():
    assert TWIN_DEMO.is_file()
    text = TWIN_DEMO.read_text(encoding="utf-8")
    assert "Stock acoustics stay" in text
    assert "volcano_s5i.yaml" in text
    cfg = IvusSimConfig.from_yaml(TWIN_DEMO)
    names = [row.name for row in cfg.materials]
    assert names == list(REQUIRED)
    for row in cfg.materials:
        assert row.specularity >= 1.0, row.name
    intima = next(row for row in cfg.materials if row.name == "intima")
    assert intima.impedance_mrayl == pytest.approx(1.87)
    calc = next(row for row in cfg.materials if row.name == "calcified_plaque")
    assert calc.attenuation_db_per_cm_mhz == pytest.approx(12.0)
    assert calc.sigma == pytest.approx(6.0)


def test_apply_materials_calls_update_not_add():
    cfg = IvusSimConfig.from_yaml(TWIN_DEMO)
    mats = MagicMock()
    n = cfg.apply_materials(mats)
    assert n == len(REQUIRED)
    assert mats.update_material.call_count == len(REQUIRED)
    first = mats.update_material.call_args_list[0]
    assert first.args[0] == "intima"
    assert first.kwargs["impedance"] == pytest.approx(1.87)
    assert first.kwargs["specularity"] == pytest.approx(1.0)


def test_apply_materials_requires_update_binding():
    cfg = IvusSimConfig.from_yaml(TWIN_DEMO)
    with pytest.raises(TypeError, match="update_material"):
        cfg.apply_materials(object())
