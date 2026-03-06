#!/usr/bin/env python3
"""
SU(3) Phase Diagram Analysis Suite
====================================
Comprehensive analysis of beta-scans and kappa-scans for SU(3) gauge theory.

Generates:
  - beta-dependence plots (R vs beta, plaq vs beta) for multiple L
  - kappa-dependence plots (R vs kappa) at fixed beta
  - Finite-size scaling analysis
  - Comparison with SU(2) baseline
  - Threshold analysis (Omega_7 vs beta)
  - Publication-ready figures with error bars

Usage:
  python scripts/analyze_su3_scans.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import json
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size":       12,
    "axes.labelsize":  14,
    "axes.titlesize":  16,
    "legend.fontsize": 12,
    "errorbar.capsize": 3,
    "lines.linewidth":  1.5,
})

# SU(2) hard-coded references for overlay
SU2_REFERENCE = {
    4: {"R": 0.3417, "R_err": 0.0036, "plaq": 0.7720, "plaq_err": 0.0040},
}
BIOLOGICAL_LO, BIOLOGICAL_HI = 0.35, 0.48


# ---------------------------------------------------------------------------
# Data loading  (filename-agnostic: classify by dict content)
# ---------------------------------------------------------------------------

def _flatten(obj):
    """Flatten a JSON value that may be a list or a single dict."""
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, list):
                out.extend(item)
            else:
                out.append(item)
        return out
    return [obj]


def load_all_results(data_dir: str, su3_only: bool = True) -> List[Dict]:
    """Load every *.json file in data_dir and return flat list of result dicts.
    
    With su3_only=True (default) skips any file whose name starts with 'su2_'
    so SU(2) reference data does not pollute SU(3) phase-diagram analysis.
    """
    records = []
    for fname in sorted(glob.glob(f"{data_dir}/*.json")):
        if su3_only and Path(fname).name.startswith("su2_"):
            continue
        try:
            with open(fname, "r") as f:
                raw = json.load(f)
            records.extend(_flatten(raw))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not load {fname}: {e}")
    return records


def load_beta_scans(data_dir: str, kappa: float = 0.3) -> Dict[int, List[Dict]]:
    """Return {L: [sorted-by-beta dicts]} for gauge_group='SU3' and given kappa."""
    results: Dict[int, List[Dict]] = {}
    for d in load_all_results(data_dir):
        if abs(d.get("kappa", -1) - kappa) > 0.01:
            continue
        L = int(d.get("L", 0))
        if L == 0:
            continue
        results.setdefault(L, []).append(d)
    for L in results:
        results[L].sort(key=lambda x: x["beta_g"])
    return results


def load_kappa_scans(data_dir: str, beta: float) -> Dict[int, List[Dict]]:
    """Return {L: [sorted-by-kappa dicts]} for given beta."""
    results: Dict[int, List[Dict]] = {}
    for d in load_all_results(data_dir):
        if abs(d.get("beta_g", -1) - beta) > 0.5:
            continue
        L = int(d.get("L", 0))
        if L == 0:
            continue
        results.setdefault(L, []).append(d)
    for L in results:
        results[L].sort(key=lambda x: x["kappa"])
    return results


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_finite_size_scaling(
    data: Dict[int, List[Dict]],
    beta_fixed: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Weighted linear fit  R(L) = R_inf + a/L.
    Returns (L_vals, R_vals, R_errs, fit_params).
    """
    L_list, R_list, R_err_list = [], [], []
    for L in sorted(data.keys()):
        closest = min(data[L], key=lambda x: abs(x["beta_g"] - beta_fixed))
        if abs(closest["beta_g"] - beta_fixed) < 1.0:
            L_list.append(L)
            R_list.append(closest["R_mean"])
            R_err_list.append(max(closest["R_err"], 1e-6))

    L_arr = np.array(L_list)
    R_arr = np.array(R_list)
    E_arr = np.array(R_err_list)

    if len(L_arr) < 2:
        return L_arr, R_arr, E_arr, {}

    x = 1.0 / L_arr
    w = 1.0 / E_arr**2
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    try:
        cov    = np.linalg.inv(A.T @ W @ A)
        params = cov @ (A.T @ W @ R_arr)
        R_inf, a = params
        R_inf_err = np.sqrt(cov[0, 0])
        chi2 = float(np.sum(w * (R_arr - (R_inf + a * x))**2)) / max(len(L_arr) - 2, 1)
        fit = {"R_inf": R_inf, "R_inf_err": R_inf_err, "a": a, "chi2": chi2}
    except np.linalg.LinAlgError:
        fit = {}

    return L_arr, R_arr, E_arr, fit


def find_optimal_kappa(points: List[Dict]) -> Dict:
    """Fit quadratic around R minimum; return optimal kappa and R value."""
    kappas = np.array([d["kappa"] for d in points])
    R_vals = np.array([d["R_mean"] for d in points])
    R_errs = np.array([max(d["R_err"], 1e-6) for d in points])

    idx_min = int(np.argmin(R_vals))
    base = {"kappa_min": kappas[idx_min], "R_min": R_vals[idx_min],
            "kappa_opt": kappas[idx_min], "R_opt": R_vals[idx_min]}

    if len(points) < 3:
        return base

    start = max(0, idx_min - 1)
    end   = min(len(points), idx_min + 2)
    if end - start < 3:
        return base

    xf = kappas[start:end]
    yf = R_vals[start:end]
    wf = 1.0 / R_errs[start:end]**2
    A  = np.vstack([np.ones_like(xf), xf, xf**2]).T
    W  = np.diag(wf)
    try:
        params = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ yf)
        a, b, c = params
        if c > 0:
            kappa_opt = -b / (2 * c)
            R_opt     = a + b * kappa_opt + c * kappa_opt**2
            base.update({"kappa_opt": float(kappa_opt), "R_opt": float(R_opt)})
    except np.linalg.LinAlgError:
        pass
    return base


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_beta_dependence(
    data: Dict[int, List[Dict]],
    output_dir: str,
):
    """R vs beta and plaquette vs beta for all available L."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, max(len(data), 1)))

    for (L, color) in zip(sorted(data.keys()), colors):
        pts   = data[L]
        betas = [p["beta_g"]    for p in pts]
        R     = [p["R_mean"]    for p in pts]
        R_err = [p["R_err"]     for p in pts]
        plaq  = [p["plaq_mean"] for p in pts]
        pe    = [p["plaq_err"]  for p in pts]

        ax1.errorbar(betas, R,    yerr=R_err, fmt="o-", color=color,
                     label=f"SU(3) L={L}", capsize=3, markersize=6)
        ax2.errorbar(betas, plaq, yerr=pe,    fmt="o-", color=color,
                     label=f"L={L}",   capsize=3, markersize=6)

    # SU(2) reference band at hard-coded point
    for i, (L, color) in enumerate(zip(sorted(data.keys()), colors)):
        if L in SU2_REFERENCE:
            ref = SU2_REFERENCE[L]
            ax1.axhline(ref["R"], linestyle="--", color=color, alpha=0.5,
                        label=f"SU(2) L={L}")
            all_betas = [p["beta_g"] for p in data[L]]
            ax1.fill_between([min(all_betas), max(all_betas)],
                             ref["R"] - ref["R_err"],
                             ref["R"] + ref["R_err"],
                             color=color, alpha=0.08)

    ax1.axhspan(BIOLOGICAL_LO, BIOLOGICAL_HI, color="green", alpha=0.10,
                label="biological range")
    ax1.set_xlabel("beta (gauge coupling)")
    ax1.set_ylabel("R  (MIXED fraction)")
    ax1.set_title("MIXED Fraction vs Gauge Coupling")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("beta (gauge coupling)")
    ax2.set_ylabel("(1/N) Re Tr U_p")
    ax2.set_title("Plaquette Average vs Gauge Coupling")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{output_dir}/su3_beta_dependence.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir}/su3_beta_dependence.png")
    plt.close(fig)


def plot_kappa_dependence(
    data: Dict[int, List[Dict]],
    beta_fixed: float,
    output_dir: str,
):
    """R vs kappa for all available L at fixed beta."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, max(len(data), 1)))

    for (L, color) in zip(sorted(data.keys()), colors):
        pts    = data[L]
        kappas = [p["kappa"]  for p in pts]
        R      = [p["R_mean"] for p in pts]
        R_err  = [p["R_err"]  for p in pts]

        ax.errorbar(kappas, R, yerr=R_err, fmt="o-", color=color,
                    label=f"L={L}", capsize=3, markersize=6)

        opt = find_optimal_kappa(pts)
        ax.plot(opt["kappa_opt"], opt["R_opt"], "s", color=color,
                markersize=9, markeredgecolor="black",
                label=f"min L={L} at k={opt['kappa_opt']:.3f}")

    ax.axhspan(BIOLOGICAL_LO, BIOLOGICAL_HI, color="green", alpha=0.10,
               label="biological range")
    ax.set_xlabel("kappa (hopping parameter)")
    ax.set_ylabel("R  (MIXED fraction)")
    ax.set_title(f"Hopping Dependence at beta={beta_fixed:.1f}")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{output_dir}/su3_kappa_dependence.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir}/su3_kappa_dependence.png")
    plt.close(fig)


def plot_finite_size_scaling(
    beta_scan_data: Dict[int, List[Dict]],
    output_dir: str,
    beta_choice: float = 8.0,
) -> Dict:
    """1/L scaling for R and linear extrapolation to thermodynamic limit."""
    fig, ax = plt.subplots(figsize=(8, 6))

    L_vals, R_vals, R_errs, fit = compute_finite_size_scaling(
        beta_scan_data, beta_choice
    )

    if len(L_vals) >= 2:
        x = 1.0 / L_vals
        ax.errorbar(x, R_vals, yerr=R_errs, fmt="o", markersize=8, capsize=3,
                    label="SU(3) data")

        if fit:
            x_fit = np.linspace(0, max(x) * 1.1, 200)
            y_fit = fit["R_inf"] + fit["a"] * x_fit
            label  = f"fit: R_inf = {fit['R_inf']:.4f} +/- {fit['R_inf_err']:.4f}"
            ax.plot(x_fit, y_fit, "--", color="red", alpha=0.8, label=label)
            ax.plot(0, fit["R_inf"], "s", color="red",
                    markersize=10, markeredgecolor="black")
            # Annotate L labels
            for L, xv, yv in zip(L_vals, x, R_vals):
                ax.annotate(f"L={L}", (xv, yv),
                            textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.axhspan(BIOLOGICAL_LO, BIOLOGICAL_HI, color="green", alpha=0.12,
               label="biological range")
    ax.set_xlabel("1/L")
    ax.set_ylabel("R  (MIXED fraction)")
    ax.set_title(f"Finite-Size Scaling at beta={beta_choice:.1f}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{output_dir}/su3_finite_size_scaling.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir}/su3_finite_size_scaling.png")
    plt.close(fig)
    return fit


def plot_omega7_threshold(
    beta_scan_data: Dict[int, List[Dict]],
    output_dir: str,
):
    """Omega_7 vs beta for all L."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, max(len(beta_scan_data), 1)))

    all_betas = []
    for L, color in zip(sorted(beta_scan_data.keys()), colors):
        pts   = beta_scan_data[L]
        betas = [p["beta_g"] for p in pts]
        om7   = [p.get("omega_7_mean", float("nan")) for p in pts]
        om7e  = [p.get("omega_7_err",  0.0)          for p in pts]
        all_betas.extend(betas)

        ax.errorbar(betas, om7, yerr=om7e, fmt="o-", color=color,
                    label=f"L={L}", capsize=3, markersize=6)

    OMEGA7_STAR = 0.5236  # pi/6
    if all_betas:
        bmin, bmax = min(all_betas), max(all_betas)
        ax.axhline(OMEGA7_STAR, linestyle="--", color="black",
                   linewidth=2, label="Omega_7* = pi/6")
        ax.fill_between([bmin, bmax],
                        OMEGA7_STAR - 0.05, OMEGA7_STAR + 0.05,
                        color="gray", alpha=0.15)

    ax.set_xlabel("beta (gauge coupling)")
    ax.set_ylabel("Omega_7 (curvature order parameter)")
    ax.set_title("E4 Order Parameter vs Gauge Coupling")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{output_dir}/su3_omega7_threshold.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir}/su3_omega7_threshold.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary_report(
    beta_data:  Dict[int, List[Dict]],
    kappa_data: Dict[int, List[Dict]],
    fss_fit:    Dict,
    output_dir: str,
):
    lines = []
    lines.append("=" * 70)
    lines.append("SU(3) PHASE DIAGRAM ANALYSIS - SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("BETA-SCAN RESULTS (kappa=0.3):")
    lines.append("-" * 40)
    for L in sorted(beta_data.keys()):
        pts = beta_data[L]
        if not pts:
            continue
        R_min_pt = min(pts, key=lambda x: x["R_mean"])
        R_max_pt = max(pts, key=lambda x: x["R_mean"])
        lines.append(f"  L={L}: R in [{R_min_pt['R_mean']:.4f}, {R_max_pt['R_mean']:.4f}] "
                     f"over beta in [{pts[0]['beta_g']:.1f}, {pts[-1]['beta_g']:.1f}]")
        high = [p for p in pts if p["beta_g"] >= 6.0]
        if high:
            R_hi = np.mean([p["R_mean"] for p in high])
            R_hi_err = np.sqrt(sum(p["R_err"]**2 for p in high)) / len(high)
            lines.append(f"      High-beta plateau (beta>=6): R = {R_hi:.4f} +/- {R_hi_err:.4f}")
    lines.append("")

    lines.append("KAPPA-SWEEP RESULTS:")
    lines.append("-" * 40)
    for L in sorted(kappa_data.keys()):
        pts = kappa_data[L]
        if not pts:
            continue
        beta_val = pts[0]["beta_g"]
        opt = find_optimal_kappa(pts)
        lines.append(f"  L={L}, beta={beta_val:.1f}:")
        lines.append(f"      Grid minimum:  R = {opt['R_min']:.4f} at kappa = {opt['kappa_min']:.2f}")
        lines.append(f"      Interpolated:  R = {opt['R_opt']:.4f} at kappa = {opt['kappa_opt']:.3f}")
    lines.append("")

    if fss_fit:
        lines.append("FINITE-SIZE SCALING (R = R_inf + a/L):")
        lines.append("-" * 40)
        lines.append(f"  R_inf = {fss_fit['R_inf']:.4f} +/- {fss_fit['R_inf_err']:.4f}")
        lines.append(f"  a     = {fss_fit['a']:.4f}  (1/L coefficient)")
        lines.append(f"  chi2/dof = {fss_fit['chi2']:.2f}")
        lines.append("")

    lines.append("BIOLOGICAL RANGE CHECK (0.35 <= R <= 0.48):")
    lines.append("-" * 40)
    for L in sorted(beta_data.keys()):
        pts = beta_data[L]
        if not pts:
            continue
        R_max = max(p["R_mean"] for p in pts)
        sym = "OK" if R_max >= BIOLOGICAL_LO else "NO"
        lines.append(f"  [{sym}] L={L}: R_max = {R_max:.4f}")
    if fss_fit:
        R_inf = fss_fit["R_inf"]
        sym = "OK" if BIOLOGICAL_LO <= R_inf <= BIOLOGICAL_HI else "NO"
        lines.append(f"  [{sym}] R_inf = {R_inf:.4f} (thermodynamic limit)")
    lines.append("")

    # Print & save
    text = "\n".join(lines)
    print(text)
    path = Path(output_dir) / "summary_report.txt"
    path.write_text(text, encoding="utf-8")
    print(f"  Saved: {path}")
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="SU(3) phase diagram analysis")
    p.add_argument("--data-dir",    default="data",
                   help="Directory containing JSON result files")
    p.add_argument("--output-dir",  default="analysis_output",
                   help="Directory for output plots and reports")
    p.add_argument("--kappa-fixed", type=float, default=0.3)
    p.add_argument("--beta-fixed",  type=float, default=8.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("SU(3) PHASE DIAGRAM ANALYSIS")
    print("=" * 70)
    print(f"Data dir:   {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70 + "\n")

    print("Loading data ...")
    beta_data  = load_beta_scans(args.data_dir,  args.kappa_fixed)
    kappa_data = load_kappa_scans(args.data_dir, args.beta_fixed)

    print(f"  beta-scans  -> L = {sorted(beta_data.keys())}")
    print(f"  kappa-scans -> L = {sorted(kappa_data.keys())}")
    print()

    print("Generating plots ...")
    if beta_data:
        plot_beta_dependence(beta_data, args.output_dir)
        plot_omega7_threshold(beta_data, args.output_dir)
        fss_fit = plot_finite_size_scaling(
            beta_data, args.output_dir, args.beta_fixed
        )
    else:
        fss_fit = {}

    if kappa_data:
        plot_kappa_dependence(kappa_data, args.beta_fixed, args.output_dir)

    print("\nGenerating summary report ...")
    generate_summary_report(beta_data, kappa_data, fss_fit, args.output_dir)

    print("\n" + "=" * 70)
    print(f"Analysis complete. Results in: {args.output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
