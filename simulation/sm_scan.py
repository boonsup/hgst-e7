#!/usr/bin/env python3
"""
sm_scan.py — Systematic phase-diagram exploration for SU(3)×SU(2)×U(1).
=========================================================================

Scan modes
----------
  beta3   β₃-scan at fixed β₂, β₁, κ (vary SU(3) coupling)
  kappa   κ-scan at fixed β₃, β₂, β₁ (vary matter hopping, same κq=κl)
  fss     Finite-size scaling at a fixed (β₃, κ) over multiple L values
  qvsl    Quark-vs-lepton scan: run κ-scan with SU(3) decoupled (β₃→0)

Usage examples
--------------
  python sm_scan.py beta3  --out sm_beta3_scan_L4.json
  python sm_scan.py kappa  --out sm_kappa_scan_L4.json
  python sm_scan.py fss    --beta-3 6.0 --kappa 0.4 --L 4 6 8 --out sm_fss.json
  python sm_scan.py qvsl   --out sm_qvsl_L4.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_sm_point import SMConfig, run_sm_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(data: list, outfile: str) -> None:
    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)
    print(f"\nResults saved → {path}")


def _header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def _row(res: dict) -> str:
    return (
        f"  β₃={res['beta_3']:5.1f}  β₂={res['beta_2']:4.1f}  β₁={res['beta_1']:4.1f}"
        f"  κq={res['kappa_q']:4.2f}  κl={res['kappa_l']:4.2f}  L={res['L']:2d}"
        f"  plaq3={res['plaq3_mean']:.4f}±{res['plaq3_err']:.4f}"
        f"  plaq2={res['plaq2_mean']:.4f}±{res['plaq2_err']:.4f}"
        f"  plaq1={res['plaq1_mean']:.4f}±{res['plaq1_err']:.4f}"
        f"  Rq={res['R_quark_mean']:.4f}±{res['R_quark_err']:.4f}"
        f"  Rl={res['R_lepton_mean']:.4f}±{res['R_lepton_err']:.4f}"
        f"  acc_lnk={res['link_acc']:.3f}"
    )


def _run_point(cfg: SMConfig, label: str) -> dict:
    print(f"\n[{label}]")
    t0 = time.perf_counter()
    res = run_sm_point(cfg, verbose=False)
    elapsed = time.perf_counter() - t0
    print(f"  done ({elapsed:.1f}s)  " + _row(res)[2:])
    return res


# ---------------------------------------------------------------------------
# Scan 1: β₃-scan
# ---------------------------------------------------------------------------

def scan_beta3(args: argparse.Namespace) -> None:
    beta3_list   = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    beta2        = args.beta_2
    beta1        = args.beta_1
    kappa_q      = args.kappa_q
    kappa_l      = args.kappa_l
    L            = args.L[0]

    _header(f"β₃-scan  β₂={beta2} β₁={beta1} κq={kappa_q} κl={kappa_l} L={L}")

    results = []
    for i, b3 in enumerate(beta3_list):
        cfg = SMConfig(
            L=L, beta_3=b3, beta_2=beta2, beta_1=beta1,
            kappa_q=kappa_q, kappa_l=kappa_l,
            n_therm=args.n_therm, n_measure=args.n_meas,
            n_skip=args.n_skip, hot_start=not args.cold,
            seed=args.seed + i,
        )
        res = _run_point(cfg, f"β₃={b3}")
        results.append(res)

    print("\n" + "─" * 72)
    print("  β₃-scan summary:")
    for r in results:
        print(f"    β₃={r['beta_3']:5.1f}  "
              f"Rq={r['R_quark_mean']:.4f}±{r['R_quark_err']:.4f}  "
              f"Rl={r['R_lepton_mean']:.4f}±{r['R_lepton_err']:.4f}  "
              f"plaq3={r['plaq3_mean']:.4f}  plaq2={r['plaq2_mean']:.4f}  plaq1={r['plaq1_mean']:.4f}")

    _save(results, args.out)


# ---------------------------------------------------------------------------
# Scan 2: κ-scan  (κq = κl = κ)
# ---------------------------------------------------------------------------

def scan_kappa(args: argparse.Namespace) -> None:
    kappa_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    beta3      = args.beta_3
    beta2      = args.beta_2
    beta1      = args.beta_1
    L          = args.L[0]

    _header(f"κ-scan  β₃={beta3} β₂={beta2} β₁={beta1} L={L}")

    results = []
    for i, k in enumerate(kappa_list):
        cfg = SMConfig(
            L=L, beta_3=beta3, beta_2=beta2, beta_1=beta1,
            kappa_q=k, kappa_l=k,
            n_therm=args.n_therm, n_measure=args.n_meas,
            n_skip=args.n_skip, hot_start=not args.cold,
            seed=args.seed + i,
        )
        res = _run_point(cfg, f"κ={k:.2f}")
        results.append(res)

    # Find best-R point
    best_q = max(results, key=lambda r: r['R_quark_mean'])
    best_l = max(results, key=lambda r: r['R_lepton_mean'])

    print("\n" + "─" * 72)
    print("  κ-scan summary:")
    for r in results:
        print(f"    κ={r['kappa_q']:.2f}  "
              f"Rq={r['R_quark_mean']:.4f}±{r['R_quark_err']:.4f}  "
              f"Rl={r['R_lepton_mean']:.4f}±{r['R_lepton_err']:.4f}  "
              f"plaq3={r['plaq3_mean']:.4f}  plaq2={r['plaq2_mean']:.4f}  plaq1={r['plaq1_mean']:.4f}")
    print(f"\n  Max R_quark  at κ={best_q['kappa_q']:.2f}  →  Rq={best_q['R_quark_mean']:.4f}")
    print(f"  Max R_lepton at κ={best_l['kappa_l']:.2f}  →  Rl={best_l['R_lepton_mean']:.4f}")

    _save(results, args.out)


# ---------------------------------------------------------------------------
# Scan 3: Finite-size scaling
# ---------------------------------------------------------------------------

def scan_fss(args: argparse.Namespace) -> None:
    L_list  = args.L
    beta3   = args.beta_3
    beta2   = args.beta_2
    beta1   = args.beta_1
    kappa_q = args.kappa_q
    kappa_l = args.kappa_l

    _header(f"FSS  β₃={beta3} β₂={beta2} β₁={beta1} κq={kappa_q} κl={kappa_l}  L={L_list}")

    results = []
    for i, L in enumerate(L_list):
        cfg = SMConfig(
            L=L, beta_3=beta3, beta_2=beta2, beta_1=beta1,
            kappa_q=kappa_q, kappa_l=kappa_l,
            n_therm=args.n_therm, n_measure=args.n_meas,
            n_skip=args.n_skip, hot_start=not args.cold,
            seed=args.seed + i,
        )
        res = _run_point(cfg, f"L={L}")
        results.append(res)

    print("\n" + "─" * 72)
    print("  FSS summary:")
    for r in results:
        print(f"    L={r['L']:2d}  "
              f"Rq={r['R_quark_mean']:.4f}±{r['R_quark_err']:.4f}  "
              f"Rl={r['R_lepton_mean']:.4f}±{r['R_lepton_err']:.4f}  "
              f"plaq3={r['plaq3_mean']:.4f}  plaq2={r['plaq2_mean']:.4f}  plaq1={r['plaq1_mean']:.4f}")

    _save(results, args.out)


# ---------------------------------------------------------------------------
# Scan 4: Quark vs Lepton — EW-only (β₃ → 0) vs full SM
# ---------------------------------------------------------------------------

def scan_qvsl(args: argparse.Namespace) -> None:
    """
    Run κ-scan twice:
      (a) Full SM:           β₃=6, β₂=4, β₁=2
      (b) EW-only (β₃=0):   β₃=0, β₂=4, β₁=2  (no SU(3) colour)
    Compare R_quark vs R_lepton in both cases.
    """
    kappa_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    beta2 = args.beta_2
    beta1 = args.beta_1
    L     = args.L[0]

    results = []
    for scenario, b3 in [("full_SM", args.beta_3), ("EW_only", 0.0)]:
        _header(f"Quark-vs-Lepton  scenario={scenario}  β₃={b3}  L={L}")
        for i, k in enumerate(kappa_list):
            cfg = SMConfig(
                L=L, beta_3=b3, beta_2=beta2, beta_1=beta1,
                kappa_q=k, kappa_l=k,
                n_therm=args.n_therm, n_measure=args.n_meas,
                n_skip=args.n_skip, hot_start=not args.cold,
                seed=args.seed + i + (100 if scenario == "EW_only" else 0),
            )
            res = _run_point(cfg, f"{scenario} κ={k:.2f}")
            res['scenario'] = scenario
            results.append(res)

    print("\n" + "─" * 72)
    print("  Quark-vs-Lepton summary:")
    for sc in ["full_SM", "EW_only"]:
        print(f"  [{sc}]  β₃={'6.0' if sc=='full_SM' else '0.0'}")
        for r in [x for x in results if x['scenario'] == sc]:
            delta_R = r['R_quark_mean'] - r['R_lepton_mean']
            print(f"    κ={r['kappa_q']:.2f}  "
                  f"Rq={r['R_quark_mean']:.4f}  Rl={r['R_lepton_mean']:.4f}  "
                  f"ΔR(q-l)={delta_R:+.4f}")

    _save(results, args.out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SM systematic scans",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", choices=["beta3", "kappa", "fss", "qvsl"])

    # SM couplings (defaults match spec point)
    p.add_argument("--beta-3",  type=float, default=6.0,  dest="beta_3")
    p.add_argument("--beta-2",  type=float, default=4.0,  dest="beta_2")
    p.add_argument("--beta-1",  type=float, default=2.0,  dest="beta_1")
    p.add_argument("--kappa-q", type=float, default=0.3,  dest="kappa_q")
    p.add_argument("--kappa-l", type=float, default=0.3,  dest="kappa_l")
    p.add_argument("--kappa",   type=float, default=None,
                   help="Set kappa_q = kappa_l together")

    p.add_argument("--L",       type=int, nargs="+", default=[4])
    p.add_argument("--n-therm", type=int, default=500,  dest="n_therm")
    p.add_argument("--n-meas",  type=int, default=200,  dest="n_meas")
    p.add_argument("--n-skip",  type=int, default=2,    dest="n_skip")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--cold",    action="store_true")
    p.add_argument("--out",     type=str, default=None)

    args = p.parse_args()
    if args.kappa is not None:
        args.kappa_q = args.kappa
        args.kappa_l = args.kappa
    if args.out is None:
        args.out = f"sm_{args.mode}_scan_L{args.L[0]}.json"
    return args


if __name__ == "__main__":
    args = parse_args()
    dispatch = {
        "beta3": scan_beta3,
        "kappa": scan_kappa,
        "fss":   scan_fss,
        "qvsl":  scan_qvsl,
    }
    dispatch[args.mode](args)
