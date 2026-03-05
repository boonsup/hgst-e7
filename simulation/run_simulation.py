#!/usr/bin/env python3
"""
run_simulation.py — Unified CLI runner for HGST lattice gauge theory.
======================================================================
# Gauge groups: SU2 (default), SU3, SM (SU(3)×SU(2)×U(1))
#
# Added SM support in implementation16:
#   --gauge-group SM  dispatches to run_sm_point.SMConfig / run_sm_point.run_sm_point
#   --beta-3, --beta-2, --beta-1   independent SM couplings
#   --kappa-q, --kappa-l           quark / lepton hopping

Wraps simulation.py's production_scan / beta_scan / run_point using the
actual tested API (MetropolisUpdater, ObservableAccumulator, etc.).

Replaces the template in implementation_extended.md Step 1.4.

Usage
-----
# Single β-point
python run_simulation.py --mode point --L 4 --beta 4.0 --kappa 0.3

# Full β-scan for one L
python run_simulation.py --mode scan --L 8 --n-beta 10 --beta-min 0.5 --beta-max 8.0 \
    --n-therm 2000 --n-meas 1000 --n-skip 2 --out data/L8_scan.json

# Multi-L production scan (mirrors the CLI in simulation.py)
python run_simulation.py --mode production --L 4 6 8 --n-beta 12 \
    --beta-min 0.5 --beta-max 8.0 --kappa 0.3 \
    --n-therm 2000 --n-meas 1000 --out results_L468.json

# Quick smoke test
python run_simulation.py --mode test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Force UTF-8 stdout so Unicode (β, κ, Ω …) survives Windows cp874 terminals / pipes
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Our tested modules
from simulation import SimConfig, run_point, beta_scan, kappa_scan, production_scan, smoke_test
from run_sm_point import SMConfig, run_sm_point


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HGST SU(2) lattice simulation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mode", choices=["test", "point", "scan", "production"],
                   default="test",
                   help="Execution mode")

    # Lattice / physics
    p.add_argument("--L", type=int, nargs="+", default=[4],
                   help="Lattice size(s). Use multiple values in production mode.")
    p.add_argument("--beta", type=float, default=4.0,
                   help="Gauge coupling β  (point mode only)")
    p.add_argument("--kappa", type=float, nargs="+", default=[0.3],
                   help="Matter hopping κ. Multiple values run a κ-scan in scan mode.")
    p.add_argument("--beta-min", type=float, default=0.5,
                   help="β scan start (scan / production modes)")
    p.add_argument("--beta-max", type=float, default=4.0,
                   help="β scan end   (scan / production modes)")
    p.add_argument("--n-beta", type=int, default=8,
                   help="Number of β scan points")

    # Monte Carlo parameters
    p.add_argument("--n-therm", type=int, default=500,
                   help="Thermalisation sweeps")
    p.add_argument("--n-meas", type=int, default=300,
                   help="Measurement sweeps")
    p.add_argument("--n-skip", type=int, default=2,
                   help="Sweeps between measurements (decorrelation)")
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed")
    p.add_argument("--cold", action="store_true",
                   help="Cold start (ordered) instead of hot (random)")
    p.add_argument("--gauge-group", default="SU2", choices=["SU2", "SU3", "SM"],
                   help="Gauge group: SU2 (default), SU3, or SM (SU(3)×SU(2)×U(1))")

    # SM-specific couplings (only used when --gauge-group SM)
    p.add_argument("--beta-3",  type=float, default=6.0, dest="beta_3",
                   help="SU(3) gauge coupling β₃ (SM only)")
    p.add_argument("--beta-2",  type=float, default=4.0, dest="beta_2",
                   help="SU(2) gauge coupling β₂ (SM only)")
    p.add_argument("--beta-1",  type=float, default=2.0, dest="beta_1",
                   help="U(1) gauge coupling β₁ (SM only)")
    p.add_argument("--kappa-q", type=float, default=0.0, dest="kappa_q",
                   help="Quark hopping κ_q (SM only)")
    p.add_argument("--kappa-l", type=float, default=0.0, dest="kappa_l",
                   help="Lepton hopping κ_l (SM only)")

    # Output
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON file (optional)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-measurement progress")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Mode dispatchers
# ──────────────────────────────────────────────────────────────────────────────

def mode_test() -> None:
    print("Running smoke test via simulation.smoke_test()…")
    ok = smoke_test()
    if ok:
        print("✓ All checks passed — pipeline is healthy.")
    else:
        print("✗ Smoke test FAILED — check errors above.")
        sys.exit(1)


def mode_point(args: argparse.Namespace) -> None:
    # ── SM dispatch ──────────────────────────────────────────────────────────
    if args.gauge_group == "SM":
        L = args.L[0]
        cfg = SMConfig(
            L=L,
            beta_3=args.beta_3, beta_2=args.beta_2, beta_1=args.beta_1,
            kappa_q=args.kappa_q, kappa_l=args.kappa_l,
            n_therm=args.n_therm, n_measure=args.n_meas, n_skip=args.n_skip,
            hot_start=not args.cold, seed=args.seed,
        )
        print(f"SM single-point run: L={L}, β₃={cfg.beta_3}, β₂={cfg.beta_2}, "
              f"β₁={cfg.beta_1}, κq={cfg.kappa_q}, κl={cfg.kappa_l}")
        t0 = time.perf_counter()
        res = run_sm_point(cfg, verbose=True)
        elapsed = time.perf_counter() - t0
        print(f"  Total time: {elapsed:.1f}s")
        _save([res], args.out)
        return

    # ── SU2 / SU3 dispatch ───────────────────────────────────────────────────
    L = args.L[0]
    kappa = args.kappa[0] if isinstance(args.kappa, list) else args.kappa
    cfg = SimConfig(
        L=L,
        beta_g=args.beta,
        kappa=kappa,
        n_therm=args.n_therm,
        n_measure=args.n_meas,
        n_skip=args.n_skip,
        seed=args.seed,
        hot_start=not args.cold,
        gauge_group=args.gauge_group,
    )
    print(f"Single-point run: L={L}, β={args.beta}, κ={kappa}")
    t0 = time.perf_counter()
    res = run_point(cfg, verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    print(f"\nResults (L={L}, β={args.beta}):")
    print(f"  plaq_avg  = {res['plaq_mean']:+.6f} ± {res['plaq_err']:.6f}")
    print(f"  R (MIXED) = {res['R_mean']:.6f} ± {res['R_err']:.6f}")
    if "omega_7_mean" in res:
        print(f"  Ω₇        = {res['omega_7_mean']:.6f} ± {res['omega_7_err']:.6f}")
    print(f"  Total time: {elapsed:.1f}s")

    _save([res], args.out)


def mode_scan(args: argparse.Namespace) -> None:
    L = args.L[0]
    kappas = args.kappa if isinstance(args.kappa, list) else [args.kappa]

    # Multiple κ values → κ-scan at fixed β
    if len(kappas) > 1:
        results = kappa_scan(
            L=L, beta_g=args.beta, kappa_list=kappas,
            n_therm=args.n_therm, n_measure=args.n_meas, n_skip=args.n_skip,
            base_seed=args.seed, verbose=args.verbose,
            gauge_group=args.gauge_group,
        )
    else:
        # Single κ → β-scan
        betas = list(np.linspace(args.beta_min, args.beta_max, args.n_beta))
        results = beta_scan(
            L=L, beta_list=betas, kappa=kappas[0],
            n_therm=args.n_therm, n_measure=args.n_meas, n_skip=args.n_skip,
            base_seed=args.seed, verbose=args.verbose,
            gauge_group=args.gauge_group,
        )
    _save(results, args.out)


def mode_production(args: argparse.Namespace) -> None:
    all_results = production_scan(
        L_list=args.L,
        n_beta=args.n_beta,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        kappa=args.kappa,
        n_therm=args.n_therm,
        n_measure=args.n_meas,
        n_skip=args.n_skip,
        outfile=args.out,
    )
    if args.out is None:
        _save(all_results, None)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    dispatch = {
        "test":       lambda: mode_test(),
        "point":      lambda: mode_point(args),
        "scan":       lambda: mode_scan(args),
        "production": lambda: mode_production(args),
    }
    dispatch[args.mode]()
