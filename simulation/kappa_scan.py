#!/usr/bin/env python3
"""
kappa_scan.py — κ (matter hopping) parameter sweep.
====================================================

Implements implementation_extended.md Step 1.3 using the real API.
Scans κ ∈ kappa_list at fixed (L, β) to map the (β, κ) phase diagram
for R(MIXED) and order parameters.

Usage
-----
# Default: L=6, β=4.0, κ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
python kappa_scan.py

# Custom κ range
python kappa_scan.py --L 6 --beta 4.0 --kappa 0.05 0.1 0.2 0.3 0.5 1.0 2.0 \
    --n-therm 1000 --n-meas 500 --out results_kappa.json

# Also scan beta to build full 2D phase diagram
python kappa_scan.py --L 6 --beta 1.0 2.0 4.0 \
    --kappa 0.1 0.2 0.3 0.4 0.5 0.6 --out results_2D.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simulation import SimConfig, run_point


# ──────────────────────────────────────────────────────────────────────────────
# κ scan (single β)
# ──────────────────────────────────────────────────────────────────────────────

def kappa_scan(
    L:        int,
    beta_g:   float,
    kappa_list: List[float],
    n_therm:  int  = 1000,
    n_measure: int = 500,
    n_skip:   int  = 2,
    base_seed: int = 200,
    verbose:  bool = True,
) -> List[Dict]:
    """
    Scan over κ values at fixed (L, β_g).

    Returns a list of per-point result dicts, one per κ value.
    """
    results = []

    print(f"\n{'='*64}")
    print(f"  κ-scan: L={L}, β={beta_g}, {len(kappa_list)} points")
    print(f"  n_therm={n_therm}, n_measure={n_measure}, n_skip={n_skip}")
    print(f"{'='*64}")
    print(f"  {'κ':>6}  {'plaq':>8}  {'R':>8}  {'R_err':>8}  "
          f"{'Ω₇':>8}  {'t_s':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

    for idx, kappa in enumerate(kappa_list):
        cfg = SimConfig(
            L=L,
            beta_g=beta_g,
            kappa=kappa,
            n_therm=n_therm,
            n_measure=n_measure,
            n_skip=n_skip,
            seed=base_seed + idx,
        )
        res = run_point(cfg, verbose=False)
        res["kappa"] = kappa      # ensure stored even if run_point omits it
        res["beta_g"] = beta_g
        results.append(res)

        om7 = res.get("omega_7_mean", float("nan"))
        print(f"  {kappa:>6.3f}  "
              f"{res['plaq_mean']:>+8.4f}  "
              f"{res['R_mean']:>8.4f}  "
              f"{res['R_err']:>8.4f}  "
              f"{om7:>8.4f}  "
              f"{res['t_meas_s']:>6.1f}s")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2-D (β × κ) scan
# ──────────────────────────────────────────────────────────────────────────────

def kappa_beta_2d(
    L:           int,
    beta_list:   List[float],
    kappa_list:  List[float],
    n_therm:     int  = 1000,
    n_measure:   int  = 500,
    n_skip:      int  = 2,
    base_seed:   int  = 200,
) -> Dict:
    """
    Full (β, κ) grid scan.

    Returns {"beta_list": [...], "kappa_list": [...], "grid": {beta: [kappa results]}}
    """
    grid: Dict[str, List[Dict]] = {}
    for bi, beta in enumerate(beta_list):
        grid[str(beta)] = kappa_scan(
            L=L, beta_g=beta, kappa_list=kappa_list,
            n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
            base_seed=base_seed + bi * len(kappa_list),
        )

    return {
        "L": L,
        "beta_list": beta_list,
        "kappa_list": kappa_list,
        "grid": grid,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HGST SU(2) κ-sweep and (β,κ) phase diagram",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--L", type=int, default=6)
    p.add_argument("--beta", type=float, nargs="+", default=[4.0],
                   help="β value(s). Multiple values → 2-D grid scan.")
    p.add_argument("--kappa", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                   help="κ values to scan.")
    p.add_argument("--n-therm", type=int, default=1000)
    p.add_argument("--n-meas",  type=int, default=500)
    p.add_argument("--n-skip",  type=int, default=2)
    p.add_argument("--seed",    type=int, default=200)
    p.add_argument("--out",     type=str, default=None)
    return p.parse_args()


def _save(data, outfile: Optional[str]) -> None:
    if outfile is None:
        return

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)
    print(f"\nResults saved → {path}")


if __name__ == "__main__":
    args = parse_args()

    if len(args.beta) == 1:
        # Single-β κ scan
        results = kappa_scan(
            L=args.L,
            beta_g=args.beta[0],
            kappa_list=args.kappa,
            n_therm=args.n_therm,
            n_measure=args.n_meas,
            n_skip=args.n_skip,
            base_seed=args.seed,
        )
        output = {"L": args.L, "beta_g": args.beta[0], "kappa_scan": results}
    else:
        # 2-D (β × κ) grid
        output = kappa_beta_2d(
            L=args.L,
            beta_list=args.beta,
            kappa_list=args.kappa,
            n_therm=args.n_therm,
            n_measure=args.n_meas,
            n_skip=args.n_skip,
            base_seed=args.seed,
        )

    _save(output, args.out)
