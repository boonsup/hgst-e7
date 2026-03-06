#!/usr/bin/env python3
"""
Direct comparison between SU(2) and SU(3) at matching (L, beta, kappa).

Usage:
  python scripts/compare_su2_su3.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]

Looks for:
  su2_*.json  — SU(2) results (produced by running with --gauge-group SU2)
  su3_*.json  — SU(3) results (produced by running with --gauge-group SU3)

Falls back to the hard-coded values from implementation14 if no files found.
"""

import json
import glob
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (12, 5),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "errorbar.capsize": 3,
})

# Hard-coded baseline from implementation14 (L=4, beta=6.0, kappa=0.3)
SU2_BASELINE = {"L": 4, "beta_g": 6.0, "kappa": 0.3,
                "R_mean": 0.3417, "R_err": 0.0036,
                "plaq_mean": 0.7720, "plaq_err": 0.0040}
SU3_BASELINE = {"L": 4, "beta_g": 6.0, "kappa": 0.3,
                "R_mean": 0.4038, "R_err": 0.0026,
                "plaq_mean": 0.4307, "plaq_err": 0.0067}


def _flatten(obj):
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(item if isinstance(item, list) else [item])
        return out
    return [obj]


def load_group_data(data_dir: str, group: str):
    """Load all result dicts for the given gauge group from data_dir JSON files.
    The gauge group is inferred from the 'gauge_group' key if present,
    else from the filename prefix."""
    records = []
    for fname in sorted(glob.glob(f"{data_dir}/*.json")):
        try:
            with open(fname) as f:
                raw = json.load(f)
            for d in _flatten(raw):
                g = d.get("gauge_group", "")
                if not g:
                    # Infer from filename
                    base = os.path.basename(fname).lower()
                    g = "SU3" if "su3" in base else "SU2"
                if g.upper() == group.upper():
                    records.append(d)
        except (json.JSONDecodeError, OSError):
            continue
    return records


def find_matching_pairs(su2_recs, su3_recs, tol_beta=0.1, tol_kappa=0.02):
    """Match SU(2) and SU(3) records with the same (L, beta, kappa)."""
    pairs = []
    for d2 in su2_recs:
        for d3 in su3_recs:
            if (d2["L"] == d3["L"]
                    and abs(d2["beta_g"] - d3["beta_g"]) < tol_beta
                    and abs(d2["kappa"]  - d3["kappa"])  < tol_kappa):
                pairs.append({"L": d2["L"], "beta": d2["beta_g"],
                               "kappa": d2["kappa"], "SU2": d2, "SU3": d3})
    # De-duplicate
    seen, unique = set(), []
    for p in pairs:
        key = (p["L"], round(p["beta"], 2), round(p["kappa"], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return sorted(unique, key=lambda x: (x["L"], x["beta"]))


def plot_comparison(pairs, output_dir):
    if not pairs:
        print("  No matched pairs — using hard-coded L=4 baseline only.")
        pairs = [{"L": 4, "beta": 6.0, "kappa": 0.3,
                  "SU2": SU2_BASELINE, "SU3": SU3_BASELINE}]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    L_vals    = [p["L"]             for p in pairs]
    R_su2     = [p["SU2"]["R_mean"] for p in pairs]
    R_su2_err = [p["SU2"]["R_err"]  for p in pairs]
    R_su3     = [p["SU3"]["R_mean"] for p in pairs]
    R_su3_err = [p["SU3"]["R_err"]  for p in pairs]

    x     = np.arange(len(L_vals))
    width = 0.35

    ax1.bar(x - width/2, R_su2, width, yerr=R_su2_err,
            capsize=3, label="SU(2)", color="steelblue")
    ax1.bar(x + width/2, R_su3, width, yerr=R_su3_err,
            capsize=3, label="SU(3)", color="tomato")
    ax1.axhspan(0.35, 0.48, color="green", alpha=0.10, label="biological range")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L={p['L']}\nb={p['beta']:.1f}" for p in pairs])
    ax1.set_ylabel("R  (MIXED fraction)")
    ax1.set_title("SU(2) vs SU(3): MIXED Fraction")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Ratio SU3/SU2
    R2 = np.array(R_su2)
    R3 = np.array(R_su3)
    ratios = R3 / np.where(R2 > 0, R2, 1e-9)
    ratio_err = ratios * np.sqrt(
        (np.array(R_su3_err)/np.where(R3 > 0, R3, 1e-9))**2 +
        (np.array(R_su2_err)/np.where(R2 > 0, R2, 1e-9))**2
    )

    ax2.errorbar(L_vals, ratios, yerr=ratio_err,
                 fmt="o-", capsize=3, color="purple", markersize=7)
    ax2.axhline(1.0, linestyle="--", color="black", alpha=0.5, label="ratio = 1")
    ax2.set_xlabel("Lattice size L")
    ax2.set_ylabel("R(SU3) / R(SU2)")
    ax2.set_title("Relative Frustration Enhancement")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{output_dir}/su2_vs_su3_comparison.{ext}",
                    dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir}/su2_vs_su3_comparison.png")
    plt.close(fig)

    # Text summary
    print("\nSU(2) vs SU(3) Comparison:")
    print("-" * 58)
    print(f"  {'L':>3}  {'beta':>5}  {'R_SU2':>8}  {'R_SU3':>8}  {'Ratio':>7}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}")
    for i, p in enumerate(pairs):
        print(f"  {p['L']:>3}  {p['beta']:>5.1f}  "
              f"{R_su2[i]:>8.4f}  {R_su3[i]:>8.4f}  {ratios[i]:>7.3f}")
    print(f"\n  Mean ratio: {np.mean(ratios):.3f} "
          f"+/- {np.std(ratios)/max(np.sqrt(len(ratios)),1):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--output-dir", default="analysis_output")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading SU(2) data from {args.data_dir} ...")
    su2_data = load_group_data(args.data_dir, "SU2")
    su3_data = load_group_data(args.data_dir, "SU3")
    print(f"  SU(2) records: {len(su2_data)}")
    print(f"  SU(3) records: {len(su3_data)}")

    pairs = find_matching_pairs(su2_data, su3_data)
    print(f"  Matched pairs: {len(pairs)}")

    plot_comparison(pairs, args.output_dir)


if __name__ == "__main__":
    main()
