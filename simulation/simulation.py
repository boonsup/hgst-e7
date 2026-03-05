"""
simulation.py — Main Thermalize → Measure → Scan Loop
======================================================
Layer 6 Integration Module: Full HGST E7/E10 simulation

Orchestrates the full Monte Carlo experiment:

  1.  Initialize fields (hot or cold start)
  2.  Thermalize with MetropolisUpdater (discard n_therm sweeps)
  3.  Measure ObservableSet every n_skip sweeps  (n_measure times)
  4.  Collect means + std errors via ObservableAccumulator
  5.  Repeat over parameter grid (β_g, κ, L)

PRIMARY SCIENTIFIC QUESTION (E7/E10):
  Does R(MIXED) → 0 under SU(2) gauge dynamics (same as U(1) E7-falsification)?
  Or does non-Abelian frustration hold R > 0 across the phase transition?

  Expected signatures:
    R large β (ordered phase)  > R small β (disordered phase)  → SU(2) rescues E7
    R large β → 0              → SU(2) behaves like U(1), E7 still falsified

QUICK SELF-TEST mode (--test / default when __main__):
  Runs a short burn-in + measure on a small lattice to verify the pipeline.
  Use --scan for the full production parameter scan.

Usage::

    python simulation.py             # quick smoke-test (default)
    python simulation.py --test      # same as above, explicit
    python simulation.py --scan      # full β scan, L=4,6  (several minutes)
    python simulation.py --scan --L 4 --n-beta 5 --n-therm 200 --n-meas 100

Dependencies: lattice.py, fields.py, action.py, updates.py, observables.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from lattice import Lattice2D
from fields import initialize_links, initialize_matter
from updates import MetropolisUpdater, SweepStats
from observables import ObservableAccumulator, measure

__all__ = [
    "SimConfig", "run_point", "beta_scan", "kappa_scan",
    "production_scan", "smoke_test",
]


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SimConfig:
    """
    All parameters for one simulation run.

    Defaults are tuned for a medium-quality production scan (~10 min on L=6).
    For quick testing use n_therm=200, n_measure=100, n_skip=1.
    """
    L:          int   = 4        # linear lattice size (N = L²)
    beta_g:     float = 2.0      # inverse gauge coupling
    kappa:      float = 0.3      # matter hopping parameter
    n_therm:    int   = 500      # thermalisation sweeps (discarded)
    n_measure:  int   = 200      # measurement sweeps
    n_skip:     int   = 2        # sweeps between measurements (decorrelation)
    hot_start:  bool  = True     # True = random, False = cold start
    seed:       int   = 0        # RNG seed
    eps_link:   float = 0.3      # initial link step size
    eps_matter: float = 0.5      # initial matter step size
    tune_every: int   = 50       # auto-tune interval during thermalisation
    skip_R_therm: bool = True    # skip R during thermalisation (expensive)
    gauge_group: str   = 'SU2'   # 'SU2' or 'SU3'


# ──────────────────────────────────────────────────────────────────────────────
# Single-point run
# ──────────────────────────────────────────────────────────────────────────────

def run_point(cfg: SimConfig, verbose: bool = False) -> Dict[str, float]:
    """
    Run a complete simulation for one parameter point.

    Returns a flat dict of mean ± err for all observables plus metadata.
    """
    lat    = Lattice2D(cfg.L)
    links  = initialize_links(lat,  group=cfg.gauge_group.lower(), random=cfg.hot_start)
    matter = initialize_matter(lat, group=cfg.gauge_group.lower(), random=cfg.hot_start)

    upd = MetropolisUpdater(
        lat, links, matter,
        beta_g      = cfg.beta_g,
        kappa       = cfg.kappa,
        gauge_group = cfg.gauge_group,
        eps_link    = cfg.eps_link,
        eps_matter  = cfg.eps_matter,
        seed        = cfg.seed,
    )

    # ── Thermalisation ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    therm_stats = upd.thermalize(
        cfg.n_therm,
        tune_every = cfg.tune_every,
    )
    t_therm = time.perf_counter() - t0

    if verbose:
        print(f"  Therm done ({cfg.n_therm} sweeps, {t_therm:.1f}s)  "
              f"link_acc={therm_stats.link_rate:.3f}  "
              f"matter_acc={therm_stats.matter_rate:.3f}  "
              f"eps_link={upd.eps_link:.4f}  eps_matter={upd.eps_matter:.4f}")

    # ── Measurement loop ────────────────────────────────────────────────
    acc   = ObservableAccumulator()
    t1    = time.perf_counter()

    for step in range(cfg.n_measure):
        for _ in range(cfg.n_skip):
            upd.sweep()
        obs = measure(links, matter, lat, skip_R=False)
        acc.add(obs)

        if verbose and (step + 1) % max(1, cfg.n_measure // 5) == 0:
            print(f"    [{step+1}/{cfg.n_measure}]  "
                  f"plaq={obs.plaq_avg:+.4f}  R={obs.R:.4f}  "
                  f"elapsed={time.perf_counter()-t1:.1f}s")

    t_meas = time.perf_counter() - t1

    stats = acc.finalize()
    stats.update({
        "L":        float(cfg.L),
        "N":        float(lat.N),
        "beta_g":   cfg.beta_g,
        "kappa":    cfg.kappa,
        "n_therm":  float(cfg.n_therm),
        "n_meas":   float(cfg.n_measure),
        "eps_link_final":   upd.eps_link,
        "eps_matter_final": upd.eps_matter,
        "t_therm_s":  t_therm,
        "t_meas_s":   t_meas,
        "link_acc_therm":   therm_stats.link_rate,
        "matter_acc_therm": therm_stats.matter_rate,
    })
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Beta scan
# ──────────────────────────────────────────────────────────────────────────────

def beta_scan(
    L:           int,
    beta_list:   List[float],
    kappa:       float     = 0.3,
    n_therm:     int       = 500,
    n_measure:   int       = 200,
    n_skip:      int       = 2,
    base_seed:   int       = 0,
    verbose:     bool      = True,
    gauge_group: str       = 'SU2',
) -> List[Dict[str, float]]:
    """
    Scan over a list of β_g values for fixed (L, κ, gauge_group).

    Returns a list of per-point result dicts, one per β value.
    Seeds are offset by point index for statistical independence.
    """
    results = []
    print(f"\n{'='*60}")
    print(f"  beta-scan [{gauge_group}]: L={L}, kappa={kappa}, {len(beta_list)} points")
    print(f"  n_therm={n_therm}, n_measure={n_measure}, n_skip={n_skip}")
    print(f"{'='*60}")
    print(f"  {'beta_g':>6}  {'plaq':>8}  {'R':>8}  {'R_err':>8}  "
          f"{'Omega_7':>9}  {'t_s':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*6}")

    for idx, beta_g in enumerate(beta_list):
        cfg = SimConfig(
            L=L, beta_g=beta_g, kappa=kappa,
            n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
            seed=base_seed + idx,
            gauge_group=gauge_group,
        )
        res = run_point(cfg, verbose=False)
        results.append(res)

        om7 = res.get("omega_7_mean", float("nan"))
        print(f"  {beta_g:>6.2f}  "
              f"{res['plaq_mean']:>+8.4f}  "
              f"{res['R_mean']:>8.4f}  "
              f"{res['R_err']:>8.4f}  "
              f"{om7:>9.4f}  "
              f"{res['t_meas_s']:>6.1f}s")

    return results


def kappa_scan(
    L:           int,
    beta_g:      float,
    kappa_list:  List[float],
    n_therm:     int       = 500,
    n_measure:   int       = 200,
    n_skip:      int       = 2,
    base_seed:   int       = 0,
    verbose:     bool      = True,
    gauge_group: str       = 'SU2',
) -> List[Dict[str, float]]:
    """
    Scan over a list of κ values for fixed (L, β_g, gauge_group).

    Returns a list of per-point result dicts, one per κ value.
    """
    results = []
    print(f"\n{'='*60}")
    print(f"  kappa-scan [{gauge_group}]: L={L}, beta={beta_g}, {len(kappa_list)} points")
    print(f"  n_therm={n_therm}, n_measure={n_measure}, n_skip={n_skip}")
    print(f"{'='*60}")
    print(f"  {'kappa':>6}  {'plaq':>8}  {'R':>8}  {'R_err':>8}  "
          f"{'Omega_7':>9}  {'t_s':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*6}")

    for idx, kappa in enumerate(kappa_list):
        cfg = SimConfig(
            L=L, beta_g=beta_g, kappa=kappa,
            n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
            seed=base_seed + idx,
            gauge_group=gauge_group,
        )
        res = run_point(cfg, verbose=False)
        results.append(res)

        om7 = res.get("omega_7_mean", float("nan"))
        print(f"  {kappa:>6.3f}  "
              f"{res['plaq_mean']:>+8.4f}  "
              f"{res['R_mean']:>8.4f}  "
              f"{res['R_err']:>8.4f}  "
              f"{om7:>9.4f}  "
              f"{res['t_meas_s']:>6.1f}s")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Full production scan
# ──────────────────────────────────────────────────────────────────────────────

def production_scan(
    L_list:    List[int]   = None,
    n_beta:    int         = 8,
    beta_min:  float       = 0.5,
    beta_max:  float       = 4.0,
    kappa:     float       = 0.3,
    n_therm:   int         = 2000,
    n_measure: int         = 1000,
    n_skip:    int         = 2,
    outfile:   Optional[str] = None,
) -> Dict[str, list]:
    """
    Full E7/E10 production scan.

    Scans β_g ∈ [beta_min, beta_max] for each L in L_list.
    Saves results to JSON if outfile is given.

    Returns dict {"L{L}": [per-beta result dicts], ...}
    """
    if L_list is None:
        L_list = [4, 6]

    betas = list(np.linspace(beta_min, beta_max, n_beta))
    all_results = {}

    for L in L_list:
        res_list = beta_scan(
            L=L, beta_list=betas, kappa=kappa,
            n_therm=n_therm, n_measure=n_measure, n_skip=n_skip,
        )
        all_results[f"L{L}"] = res_list

    # Print E7 summary
    print(f"\n{'='*60}")
    print("  E7 SUMMARY: R(β_g) by L")
    print(f"  {'β_g':>6}", end="")
    for L in L_list:
        print(f"  {'R(L='+str(L)+')':>10}  {'err':>6}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in L_list:
        print(f"  {'-'*10}  {'-'*6}", end="")
    print()

    for i, beta in enumerate(betas):
        print(f"  {beta:>6.2f}", end="")
        for L in L_list:
            r  = all_results[f"L{L}"][i]
            print(f"  {r['R_mean']:>10.4f}  {r['R_err']:>6.4f}", end="")
        print()

    if outfile:
        # Serialise numpy floats to Python float
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return str(obj)
            return obj

        with open(outfile, "w") as f:
            json.dump(_clean(all_results), f, indent=2)
        print(f"\n  Results saved to: {outfile}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

def smoke_test() -> bool:
    """
    Short sanity check: run a minimal simulation and verify output shapes.
    Completes in ~5 seconds on L=3.
    """
    PASS, FAIL = "PASS", "FAIL"
    results: list[tuple[str, bool, str]] = []

    print("=" * 60)
    print("simulation.py — Smoke Test  (L=3, β=2.0, κ=0.3)")
    print("  n_therm=100, n_measure=30, n_skip=1")
    print("=" * 60)

    cfg = SimConfig(L=3, beta_g=2.0, kappa=0.3,
                    n_therm=100, n_measure=30, n_skip=1, seed=77)
    t0  = time.perf_counter()
    res = run_point(cfg, verbose=True)
    elapsed = time.perf_counter() - t0

    # ── T1: result dict has expected keys ────────────────────────────────
    required = {"R_mean", "R_err", "plaq_mean", "plaq_err",
                "n_samples", "beta_g", "L", "eps_link_final"}
    ok1 = required.issubset(res.keys())
    results.append(("T1  result dict has required keys", ok1,
                    f"found {len(res)} keys"))

    # ── T2: n_samples = n_measure ────────────────────────────────────────
    ok2 = res["n_samples"] == cfg.n_measure
    results.append(("T2  n_samples == n_measure", ok2,
                    f"n_samples={res['n_samples']:.0f}, n_measure={cfg.n_measure}"))

    # ── T3: R ∈ [0, 1] ───────────────────────────────────────────────────
    ok3 = 0.0 <= res["R_mean"] <= 1.0 and res["R_err"] >= 0.0
    results.append(("T3  R_mean ∈ [0,1], R_err ≥ 0", ok3,
                    f"R_mean={res['R_mean']:.4f} ± {res['R_err']:.4f}"))

    # ── T4: plaq_avg ∈ [−1, 1] ───────────────────────────────────────────
    ok4 = -1.0 <= res["plaq_mean"] <= 1.0
    results.append(("T4  plaq_mean ∈ [−1, 1]", ok4,
                    f"plaq_mean={res['plaq_mean']:+.4f} ± {res['plaq_err']:.4f}"))

    # ── T5: auto-tune produced reasonable eps ────────────────────────────
    ok5 = 0.01 <= res["eps_link_final"] <= np.pi
    results.append(("T5  eps_link_final ∈ [0.01, π]", ok5,
                    f"eps_link_final={res['eps_link_final']:.4f}"))

    # ── T6: thermalisation acceptance rates ──────────────────────────────
    ok6 = (0.05 <= res["link_acc_therm"]   <= 0.99 and
           0.05 <= res["matter_acc_therm"] <= 0.99)
    results.append(("T6  therm acceptance rates in (0.05, 0.99)", ok6,
                    f"link_acc={res['link_acc_therm']:.3f}, "
                    f"matter_acc={res['matter_acc_therm']:.3f}"))

    # ── T7: beta scan runs and returns len(betas) results ────────────────
    betas_small = [0.5, 2.0, 4.0]
    cfg_scan = dict(L=3, kappa=0.1, n_therm=50, n_measure=20, n_skip=1)
    scan_res = beta_scan(beta_list=betas_small, **cfg_scan, verbose=False)
    ok7 = (len(scan_res) == len(betas_small) and
           all("R_mean" in r for r in scan_res))
    results.append(("T7  beta_scan returns correct length", ok7,
                    f"n_points={len(scan_res)}"))

    # ── T8: R decreases at high β vs low β (E7 β-dependence) ─────────────
    R_low  = scan_res[0]["R_mean"]
    R_high = scan_res[-1]["R_mean"]
    # At higher β gauge ordering should shift R (direction validates dynamics)
    ok8 = R_low != R_high   # they just must differ; direction tested in E10
    results.append(("T8  R(β=0.5) ≠ R(β=4.0)  (β-dependence present)", ok8,
                    f"R(0.5)={R_low:.4f}, R(4.0)={R_high:.4f}  "
                    f"Δ={R_high-R_low:+.4f}"))

    # ── Print ─────────────────────────────────────────────────────────────
    print()
    all_pass = True
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        print(f"         {detail}")
        if not ok:
            all_pass = False

    print("-" * 60)
    if all_pass:
        print(f"  All {len(results)} smoke-test checks PASSED.  ({elapsed:.1f}s total)")
        print()
        print("  Final observables (L=3, β=2.0, κ=0.3):")
        print(f"    plaq_avg = {res['plaq_mean']:+.4f} ± {res['plaq_err']:.4f}")
        print(f"    R(MIXED) = {res['R_mean']:.4f} ± {res['R_err']:.4f}")
        print()
        print("  β-scan R values (L=3, κ=0.1):")
        for b, r in zip(betas_small, scan_res):
            print(f"    β={b:.1f}  R={r['R_mean']:.4f} ± {r['R_err']:.4f}")
        print()
        print("  To run full production scan:")
        print("    python simulation.py --scan")
        print("    python simulation.py --scan --L 4 6 --n-beta 10 --n-therm 2000"
              " --n-meas 1000")
    else:
        print("  SOME CHECKS FAILED — review before running full scan.")
    print("=" * 60)
    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HGST E7/E10 SU(2) simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test",    action="store_true", help="Run smoke test (default)")
    p.add_argument("--scan",    action="store_true", help="Run production β-scan")
    p.add_argument("--L",       nargs="+", type=int, default=[4, 6],
                   help="Lattice sizes for scan")
    p.add_argument("--n-beta",  type=int,   default=8,
                   help="Number of β points in scan")
    p.add_argument("--beta-min", type=float, default=0.5)
    p.add_argument("--beta-max", type=float, default=4.0)
    p.add_argument("--kappa",   type=float, default=0.3)
    p.add_argument("--n-therm", type=int,   default=2000,
                   help="Thermalisation sweeps")
    p.add_argument("--n-meas",  type=int,   default=1000,
                   help="Measurement sweeps")
    p.add_argument("--n-skip",  type=int,   default=2)
    p.add_argument("--out",     type=str,   default=None,
                   help="JSON output file for scan results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.scan:
        production_scan(
            L_list    = args.L,
            n_beta    = args.n_beta,
            beta_min  = args.beta_min,
            beta_max  = args.beta_max,
            kappa     = args.kappa,
            n_therm   = args.n_therm,
            n_measure = args.n_meas,
            n_skip    = args.n_skip,
            outfile   = args.out,
        )
    else:
        # Default: smoke test
        ok = smoke_test()
        sys.exit(0 if ok else 1)
