"""Pass 26 -- Tier 1 metric self-validation harness.

The goal is to subject every Tier 1 metric to a forward-modeling sanity
check before we let it gate a sim parameter:

    1. Monotonicity      -- sweep the parameter, confirm metric tracks it.
    2. Inversion         -- given measured metric values, recover the
                            parameter to within tolerance via a linear /
                            affine fit.
    3. Confounder        -- overlay a synthetic un-modeled coherent
                            pattern (analogous to the bench's reverb /
                            wall-reflection signal that broke `ff_std`),
                            confirm the metric shifts by <=5% of the
                            parameter-induced range.

If a metric passes all three, we trust it as a gate.  If it fails (1) it
is broken; if it fails (2) the recovery is too noisy to gate; if it fails
(3) it conflates the parameter with an un-modeled component and must not
gate (this is precisely what `ff_std_palette` did under Pass 22).

Phase 1 covers ``processing.noise.sigma`` only, via the envelope-domain
residual metric ``_envelope_residual_std``.  Subsequent phases add
parameters as their tests are reformulated.

Usage::

    python tier1_self_validate.py --param noise_sigma
                                  [--n-frames 12] [--slider 68]
                                  [--no-confounder] [--out OUT_DIR]

Outputs::

    tier1_results/self_validate/<param>.json   per-grid metric values
    tier1_results/self_validate/<param>.png    summary figure
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import tier1_evaluation as t1  # noqa: E402

WORKSPACE_ROOT = _SCRIPT_DIR.parents[1]
OUT_DIR_DEFAULT = (WORKSPACE_ROOT / "instrument-calibration"
                   / "p035_visions" / "tier1_results" / "self_validate")


@dataclass
class SweepResult:
    param_name: str
    param_values: list[float]
    metric_values: list[float]
    metric_p25: list[float]
    metric_p75: list[float]
    palette_means: list[float]
    n_valid_pixels: list[int]
    confounder_metric_values: list[float] | None
    confounder_palette_delta: float | None


def _render_anechoic_milk_polar(cfg, sim_params, materials,
                                slider: float, ref_slider: float,
                                n_frames: int,
                                world_kind: str = "milk") -> np.ndarray:
    """Render ``n_frames`` of milk (or water) and return a polar palette
    stack of shape ``(n_frames, n_theta, n_r)``.

    The slider->dB bump uses the Pass 25 linear ``slider_to_db`` so the
    operating point is consistent with the Tier 1 milk anchors.
    """
    if world_kind == "milk":
        world = t1.build_uniform_milk_world(materials)
    elif world_kind == "water":
        world = t1.build_anechoic_world(materials)
    else:
        raise ValueError(f"unknown world_kind: {world_kind}")
    gain_bump_db = t1.slider_to_db(slider, ref_slider)
    saved_gain_db = float(sim_params.gain_db)
    sim_params.gain_db = saved_gain_db + gain_bump_db
    try:
        frames = t1.render_frames(cfg, world, materials, n_frames, sim_params)
    finally:
        sim_params.gain_db = saved_gain_db
    stk = np.stack([t1.b_mode_to_theta_r(f, cfg) for f in frames])
    return stk.astype(np.float64)


def _band_mask(cfg, r_lo_mm: float = 5.0, r_hi_mm: float = 20.0):
    """Return a band mask over the (theta, r) grid for the chosen depth
    range.  Matches the Pass 25 milk anchors' r-band."""
    _, r_mm, _, _ = t1.polar_axes(cfg)
    return (r_mm >= r_lo_mm) & (r_mm <= r_hi_mm), r_mm


def sweep_noise_sigma(slider: float = 68.0,
                      ref_slider: float = 54.0,
                      n_frames: int = 12,
                      sigma_grid: tuple[float, ...] = (
                          1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3),
                      run_confounder: bool = True,
                      confounder_envelope_amplitude: float = 0.5,
                      confounder_decay_mm: float = 3.0,
                      ) -> SweepResult:
    """Sweep ``processing.noise.sigma`` and measure the envelope-residual
    metric.  Optionally overlays a synthetic coherent palette pattern
    (reverb-like) and re-measures the metric to test confounder
    resistance.
    """
    cfg, materials, sim_params = t1.load_calibrated_config()
    log_mult = float(cfg.processing.log_multiplier)
    log_floor = float(cfg.processing.log_floor)
    reject = float(cfg.processing.reject_palette)
    saturation = float(cfg.processing.saturation_palette)
    softness = float(getattr(cfg.processing, "reject_palette_softness", 0.0))

    band_mask, r_mm = _band_mask(cfg)

    metric_vals = []
    metric_p25 = []
    metric_p75 = []
    palette_means = []
    n_valid = []
    conf_metric_vals: list[float] = []
    palette_delta_observed: list[float] = []

    base_sigma = float(sim_params.noise_sigma)
    try:
        for sigma in sigma_grid:
            sim_params.noise_sigma = float(sigma)
            print(f"[self_validate] noise.sigma = {sigma:.3e} -> "
                  f"rendering {n_frames} milk frames @ slider {slider:.0f}...",
                  flush=True)
            stk = _render_anechoic_milk_polar(
                cfg, sim_params, materials, slider, ref_slider, n_frames)
            band_stk = stk[:, :, band_mask]  # (n_frames, n_theta, n_r_band)
            stats = t1._envelope_residual_std(
                band_stk, log_multiplier=log_mult, log_floor=log_floor,
                reject_palette=reject, saturation_palette=saturation,
                reject_softness=softness,
            )
            metric_vals.append(stats["envelope_residual_std"])
            metric_p25.append(stats["envelope_residual_std_p25"])
            metric_p75.append(stats["envelope_residual_std_p75"])
            palette_means.append(stats["palette_mean"])
            n_valid.append(stats["n_valid_pixels"])
            print(f"            envelope_residual_std = "
                  f"{stats['envelope_residual_std']:.4g} "
                  f"(palette_mean {stats['palette_mean']:.1f}, "
                  f"valid {stats['n_valid_pixels']} px)", flush=True)
            if run_confounder:
                # Add a per-pixel coherent contamination IN ENVELOPE DOMAIN
                # (constant across frames -- like a wall/catheter
                # reflection adding a fixed envelope amplitude at each
                # pixel).  The metric MUST be invariant to such an overlay
                # in the high-SNR limit, because per-pixel temporal std in
                # envelope domain is preserved under per-pixel constant
                # envelope shifts.  The overlay is shaped exponentially in
                # r (decays with depth, mimicking the bench's
                # catheter-origin coherent signature).
                #
                # Implementation: we work in palette domain throughout, so
                # we (a) invert each pixel's mean palette to its envelope,
                # (b) add the envelope-domain overlay, (c) forward the new
                # mean palette, (d) apply the per-pixel mean shift to every
                # frame.  This is equivalent to having rendered the sim
                # with envelope_true += overlay everywhere.
                r_band = r_mm[band_mask]
                env_overlay = confounder_envelope_amplitude * np.exp(
                    -(r_band - r_band[0]) / confounder_decay_mm)
                # Pixel-mean envelope from palette_mean.
                env_mean_pix = t1._palette_to_envelope(
                    band_stk.mean(axis=0),
                    log_multiplier=log_mult, log_floor=log_floor,
                    reject_palette=reject, reject_softness=softness,
                )
                env_mean_new = env_mean_pix + env_overlay[None, :]
                # Forward back to palette (linear-LUT regime).
                palette_pre_new = log_mult * np.log10(
                    np.maximum(env_mean_new, log_floor) / log_floor)
                if softness > 0.0:
                    z = (palette_pre_new - reject) / softness
                    palette_mean_new = reject + softness * np.log1p(np.exp(z))
                else:
                    palette_mean_new = palette_pre_new
                palette_mean_new = np.clip(palette_mean_new, reject,
                                           saturation - 1.0)
                shift = palette_mean_new - band_stk.mean(axis=0)
                conf_stk = band_stk + shift[None]
                palette_delta_observed.append(float(np.nanmean(shift)))
                conf_stats = t1._envelope_residual_std(
                    conf_stk, log_multiplier=log_mult, log_floor=log_floor,
                    reject_palette=reject, saturation_palette=saturation,
                    reject_softness=softness,
                )
                conf_metric_vals.append(conf_stats["envelope_residual_std"])
                print(f"            confounder (env-overlay): "
                      f"envelope_residual_std = "
                      f"{conf_stats['envelope_residual_std']:.4g}"
                      f" (env shift {confounder_envelope_amplitude:.2f} -> "
                      f"palette delta mean {float(np.nanmean(shift)):+.1f})",
                      flush=True)
    finally:
        sim_params.noise_sigma = base_sigma

    avg_palette_delta = (float(np.mean(palette_delta_observed))
                        if palette_delta_observed else None)
    return SweepResult(
        param_name="noise_sigma",
        param_values=[float(s) for s in sigma_grid],
        metric_values=metric_vals,
        metric_p25=metric_p25,
        metric_p75=metric_p75,
        palette_means=palette_means,
        n_valid_pixels=n_valid,
        confounder_metric_values=(conf_metric_vals if run_confounder
                                  else None),
        confounder_palette_delta=avg_palette_delta,
    )


def evaluate_sweep(res: SweepResult) -> dict:
    """Score a SweepResult against the three Phase-1 criteria.

    The metric model is variance-additive::

        metric^2 = floor^2 + (slope * sigma)^2

    where ``floor`` captures the irreducible per-frame variation present
    even at sigma -> 0 (procedural milk speckle re-rolled per frame in
    the sim; physically analogous to bench's catheter-rotation-driven
    speckle decorrelation).  Above the floor (sigma >= floor / (2 *
    slope)) the metric is dominated by the noise term and is invertible
    to sigma.  Below the floor it returns ~floor regardless of sigma.

    Returns a dict with per-criterion ``ok`` flags and diagnostic values.
    """
    x = np.asarray(res.param_values, dtype=float)
    y = np.asarray(res.metric_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x_f = x[finite]
    y_f = y[finite]

    # Variance-additive model fit on (sigma^2, metric^2).
    x2 = x_f ** 2
    y2 = y_f ** 2
    if x2.size >= 2 and np.ptp(y2) > 0:
        b, a = np.polyfit(x2, y2, 1)  # y2 = b*x2 + a
        speckle_floor = float(np.sqrt(max(a, 0.0)))
        slope = float(np.sqrt(max(b, 0.0)))
    else:
        speckle_floor = float("nan")
        slope = float("nan")
        a = float("nan"); b = float("nan")

    # "Above floor" for the inversion gate means the noise contribution to
    # variance is at least 4x the floor variance (i.e. noise std >= 2x
    # floor std).  In this regime the metric is unambiguously
    # noise-dominated and the variance-additive inversion is precise;
    # closer to the floor the metric loses precision because small
    # variance fluctuations translate to large recovered-sigma swings.
    # Monotonicity is checked over a broader range (noise variance >=
    # floor variance) since monotonicity does not require precise
    # inversion.
    if np.isfinite(slope) and slope > 0 and np.isfinite(speckle_floor):
        sigma_above_inversion = 2.0 * speckle_floor / slope
        sigma_above_monotonic = speckle_floor / slope
    else:
        sigma_above_inversion = float("nan")
        sigma_above_monotonic = float("nan")
    above_inversion = (x_f >= sigma_above_inversion) if np.isfinite(sigma_above_inversion) else np.zeros_like(x_f, dtype=bool)
    above_monotonic = (x_f >= sigma_above_monotonic) if np.isfinite(sigma_above_monotonic) else np.zeros_like(x_f, dtype=bool)
    # Keep the old name for backward-compatibility of the figure caption.
    above = above_inversion
    sigma_above = sigma_above_inversion

    # (1) Monotonicity ABOVE the (looser) monotonic threshold: pairwise
    # diffs should have a consistent sign once noise dominates floor.
    if above_monotonic.sum() >= 2:
        dy_above = np.diff(y_f[above_monotonic])
        monotonic_ok = bool(np.all(dy_above > 0)) or bool(np.all(dy_above < 0))
        monotonic_sign = int(np.sign(dy_above.mean()))
    elif y_f.size >= 2:
        dy = np.diff(y_f)
        monotonic_ok = bool(np.all(dy > 0)) or bool(np.all(dy < 0))
        monotonic_sign = int(np.sign(dy.mean()))
    else:
        monotonic_ok = False
        monotonic_sign = 0

    # (2) Inversion error: recover sigma from sqrt((y^2 - floor^2)) / slope.
    # Gated only in the noise-dominated regime (above_inversion).
    if np.isfinite(slope) and slope > 0 and np.isfinite(speckle_floor):
        with np.errstate(invalid="ignore"):
            sigma_recovered = np.sqrt(np.maximum(y_f ** 2 - speckle_floor ** 2, 0.0)) / slope
        if above_inversion.sum() > 0:
            rel_err = np.abs(sigma_recovered[above_inversion] - x_f[above_inversion]) / np.maximum(x_f[above_inversion], 1e-12)
            max_rel_err = float(rel_err.max())
            inversion_ok = bool(max_rel_err <= 0.10)
        else:
            max_rel_err = float("nan")
            inversion_ok = False
    else:
        sigma_recovered = np.full_like(y_f, np.nan)
        max_rel_err = float("nan")
        inversion_ok = False

    # (3) Confounder: with the corrected envelope-domain overlay, the
    # metric should shift <=5% relative to the unconfounded metric at
    # the same sigma (pointwise, NOT normalised by parameter range --
    # that earlier normalisation was misleading for a heteroscedastic
    # metric).  We gate on the MAXIMUM relative shift over all sigma
    # points so a single misbehaving point fails the test.
    conf_ok: bool | None = None
    conf_max_rel_shift: float | None = None
    if res.confounder_metric_values is not None:
        conf = np.asarray(res.confounder_metric_values, dtype=float)
        rel_shift = np.abs(conf - y) / np.maximum(np.abs(y), 1e-12)
        rel_shift_valid = rel_shift[finite]
        if rel_shift_valid.size:
            conf_max_rel_shift = float(rel_shift_valid.max())
            conf_ok = bool(conf_max_rel_shift <= 0.05)
        else:
            conf_max_rel_shift = float("nan")
            conf_ok = False

    return {
        "model": "metric^2 = floor^2 + (slope * sigma)^2",
        "speckle_floor": float(speckle_floor),
        "slope_metric_per_param": float(slope),
        "sigma_above_floor_inversion_gate": float(sigma_above_inversion),
        "sigma_above_floor_monotonic_gate": float(sigma_above_monotonic),
        "above_floor_inversion_mask": [bool(x) for x in above_inversion.tolist()],
        "above_floor_monotonic_mask": [bool(x) for x in above_monotonic.tolist()],
        "monotonic_ok": monotonic_ok,
        "monotonic_sign": monotonic_sign,
        "inversion_ok": inversion_ok,
        "inversion_max_rel_err_above_floor": max_rel_err,
        "sigma_recovered": sigma_recovered.tolist(),
        "confounder_ok": conf_ok,
        "confounder_max_rel_shift": conf_max_rel_shift,
    }


def write_outputs(res: SweepResult, eval_summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{res.param_name}.json"
    payload = {
        "param_name": res.param_name,
        "param_values": res.param_values,
        "metric_values": res.metric_values,
        "metric_p25": res.metric_p25,
        "metric_p75": res.metric_p75,
        "palette_means": res.palette_means,
        "n_valid_pixels": res.n_valid_pixels,
        "confounder_metric_values": res.confounder_metric_values,
        "confounder_palette_delta": res.confounder_palette_delta,
        "evaluation": eval_summary,
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {json_path}")

    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); skipping figure.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))
    x = np.asarray(res.param_values)
    y = np.asarray(res.metric_values)
    p25 = np.asarray(res.metric_p25)
    p75 = np.asarray(res.metric_p75)
    ax.errorbar(x, y, yerr=[y - p25, p75 - y], fmt="o-", color="C0",
                label="envelope_residual_std (median ± IQR)")
    if res.confounder_metric_values is not None:
        ax.plot(x, res.confounder_metric_values, "s--", color="C3",
                label="with env-domain coh overlay (should be invariant)")
    # Variance-additive model overlay: metric^2 = floor^2 + (slope*sigma)^2
    floor = eval_summary["speckle_floor"]
    slope = eval_summary["slope_metric_per_param"]
    if np.isfinite(floor) and np.isfinite(slope) and slope > 0:
        xx = np.geomspace(min(x) * 0.5, max(x) * 2.0, 400)
        yy_model = np.sqrt(floor ** 2 + (slope * xx) ** 2)
        ax.plot(xx, yy_model, "k:", lw=1.0, alpha=0.6,
                label=f"model: sqrt(floor^2 + (slope*σ)^2)\n"
                      f"floor={floor:.3f}, slope={slope:.0f}")
        ax.axhline(floor, color="k", ls="--", lw=0.6, alpha=0.4,
                   label=f"speckle floor = {floor:.3f}")
        sigma_above_inv = eval_summary["sigma_above_floor_inversion_gate"]
        if np.isfinite(sigma_above_inv):
            ax.axvline(sigma_above_inv, color="0.4", ls="-.", lw=0.6,
                       alpha=0.5,
                       label=f"σ above-floor (inv gate): {sigma_above_inv:.2e}")
    ax.set_xlabel("noise.sigma (RF units, YAML)")
    ax.set_ylabel("envelope_residual_std (envelope units, median per pixel)")
    title_lines = [
        f"Tier 1 self-validate -- {res.param_name}",
        (f"monotonic={eval_summary['monotonic_ok']} "
         f"inversion_ok={eval_summary['inversion_ok']} "
         f"max_rel_err_above_floor={eval_summary['inversion_max_rel_err_above_floor']:.1%}"),
    ]
    if eval_summary["confounder_ok"] is not None:
        title_lines.append(
            f"confounder_ok={eval_summary['confounder_ok']} "
            f"max_rel_shift={eval_summary['confounder_max_rel_shift']:.1%}")
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    png_path = out_dir / f"{res.param_name}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {png_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param", default="noise_sigma",
                        choices=["noise_sigma"],
                        help="parameter to self-validate (Phase 1: noise_sigma only).")
    parser.add_argument("--n-frames", type=int, default=12,
                        help="frames per sweep point.")
    parser.add_argument("--slider", type=float, default=68.0,
                        help="bench slider to render at.")
    parser.add_argument("--ref-slider", type=float, default=54.0,
                        help="YAML reference slider (gain_db corresponds to this).")
    parser.add_argument("--sigma-grid", default="1e-4,2.5e-4,5e-4,1e-3,2.5e-3,5e-3",
                        help="comma-separated noise.sigma values to sweep.")
    parser.add_argument("--no-confounder", action="store_true",
                        help="skip the synthetic coh-overlay test.")
    parser.add_argument("--confounder-amplitude", type=float, default=0.5,
                        help="peak ENVELOPE amplitude of synthetic coh overlay "
                             "(constant across frames, decays exponentially in r); "
                             "test asserts the metric is invariant to this overlay.")
    parser.add_argument("--out", default=str(OUT_DIR_DEFAULT),
                        help="output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.param == "noise_sigma":
        sigma_grid = tuple(float(s) for s in args.sigma_grid.split(","))
        res = sweep_noise_sigma(
            slider=args.slider, ref_slider=args.ref_slider,
            n_frames=args.n_frames, sigma_grid=sigma_grid,
            run_confounder=not args.no_confounder,
            confounder_envelope_amplitude=args.confounder_amplitude,
        )
        eval_summary = evaluate_sweep(res)
        write_outputs(res, eval_summary, out_dir)
        print("\n=== Self-validate summary ===")
        for k, v in eval_summary.items():
            if isinstance(v, float):
                print(f"  {k:50s} {v:.4g}")
            else:
                print(f"  {k:50s} {v}")
        return 0 if (
            eval_summary["monotonic_ok"]
            and eval_summary["inversion_ok"]
            and (eval_summary["confounder_ok"] is None
                 or eval_summary["confounder_ok"])
        ) else 1
    else:
        raise NotImplementedError(args.param)


if __name__ == "__main__":
    sys.exit(main())
