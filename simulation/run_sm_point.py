#!/usr/bin/env python3
"""
run_sm_point.py — Standalone runner for a single SM (SU(3)×SU(2)×U(1)) simulation point.
===========================================================================================

Usage:
    python run_sm_point.py [--L 4] [--beta-3 6.0] [--beta-2 4.0] [--beta-1 2.0]
                           [--kappa-q 0.0] [--kappa-l 0.0]
                           [--n-therm 500] [--n-meas 200] [--n-skip 2]
                           [--seed 123] [--hot] [--out sm_L4_pg.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from lattice import Lattice2D
from sm_fields import initialize_sm_links, initialize_quarks, initialize_leptons
from sm_updates import SMUpdater
from sm_observables import measure


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SMConfig:
    L:          int   = 4
    beta_3:     float = 6.0
    beta_2:     float = 4.0
    beta_1:     float = 2.0
    kappa_q:    float = 0.0
    kappa_l:    float = 0.0
    n_therm:    int   = 500
    n_measure:  int   = 200
    n_skip:     int   = 2
    hot_start:  bool  = True
    seed:       int   = 0
    eps_link:   float = 0.3
    eps_quark:  float = 0.3
    eps_lepton: float = 0.3
    tune_every: int   = 50


# ---------------------------------------------------------------------------
# Single-point runner
# ---------------------------------------------------------------------------

def run_sm_point(cfg: SMConfig, verbose: bool = True) -> Dict:
    lat = Lattice2D(cfg.L)

    rng_links   = np.random.default_rng(cfg.seed)
    rng_quarks  = np.random.default_rng(cfg.seed + 1)
    rng_leptons = np.random.default_rng(cfg.seed + 2)

    links   = initialize_sm_links(lat,   random=cfg.hot_start, rng=rng_links)
    quarks  = initialize_quarks(lat,     random=cfg.hot_start, rng=rng_quarks)
    leptons = initialize_leptons(lat,    random=cfg.hot_start, rng=rng_leptons)

    updater = SMUpdater(
        lat, links, quarks, leptons,
        beta_3=cfg.beta_3, beta_2=cfg.beta_2, beta_1=cfg.beta_1,
        kappa_q=cfg.kappa_q, kappa_l=cfg.kappa_l,
        eps_link=cfg.eps_link, eps_quark=cfg.eps_quark, eps_lepton=cfg.eps_lepton,
        seed=cfg.seed + 3,
        target_rate=0.5
    )

    # --------------- Thermalization ---------------
    t0 = time.perf_counter()
    if verbose:
        print(f"\nThermalizing {cfg.n_therm} sweeps "
              f"(β₃={cfg.beta_3}, β₂={cfg.beta_2}, β₁={cfg.beta_1}, "
              f"κq={cfg.kappa_q}, κl={cfg.kappa_l}, L={cfg.L}) …")
    # Pure gauge point: skip quark/lepton updates if kappa = 0
    do_q = cfg.kappa_q != 0.0
    do_l = cfg.kappa_l != 0.0
    therm_stats = updater.thermalize(
        cfg.n_therm,
        update_quarks=do_q, update_leptons=do_l,
        tune_every=cfg.tune_every
    )
    t_therm = time.perf_counter() - t0
    if verbose:
        print(f"  Thermalization done ({t_therm:.1f}s)  "
              f"accept: link={therm_stats.link_rate:.3f}, "
              f"quark={therm_stats.quark_rate:.3f}, "
              f"lepton={therm_stats.lepton_rate:.3f}")

    # --------------- Measurement loop ---------------
    if verbose:
        print(f"\nMeasuring {cfg.n_measure} samples (skip={cfg.n_skip}) …")
    acc = {k: [] for k in ('plaq_3', 'plaq_2', 'plaq_1', 'R_quark', 'R_lepton')}

    t1 = time.perf_counter()
    pure_gauge = (cfg.kappa_q == 0.0 and cfg.kappa_l == 0.0)
    for i in range(cfg.n_measure):
        for _ in range(cfg.n_skip):
            updater.sweep(update_links=True, update_quarks=do_q, update_leptons=do_l)
        obs = measure(links, quarks, leptons, lat, skip_R=pure_gauge)
        acc['plaq_3'].append(obs.plaq_3)
        acc['plaq_2'].append(obs.plaq_2)
        acc['plaq_1'].append(obs.plaq_1)
        acc['R_quark'].append(obs.R_quark)
        acc['R_lepton'].append(obs.R_lepton)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1:4d}/{cfg.n_measure}] "
                  f"plaq3={obs.plaq_3:.4f}  plaq2={obs.plaq_2:.4f}  plaq1={obs.plaq_1:.4f}")
    t_meas = time.perf_counter() - t1

    N = cfg.n_measure

    def stats(k):
        a = np.array(acc[k])
        return float(np.mean(a)), float(np.std(a) / np.sqrt(N))

    pm3, pe3 = stats('plaq_3')
    pm2, pe2 = stats('plaq_2')
    pm1, pe1 = stats('plaq_1')
    Rq_m, Rq_e = stats('R_quark')
    Rl_m, Rl_e = stats('R_lepton')

    results = {
        'L': cfg.L,
        'beta_3': cfg.beta_3, 'beta_2': cfg.beta_2, 'beta_1': cfg.beta_1,
        'kappa_q': cfg.kappa_q, 'kappa_l': cfg.kappa_l,
        'n_therm': cfg.n_therm, 'n_measure': cfg.n_measure, 'n_skip': cfg.n_skip,
        'seed': cfg.seed,
        'plaq3_mean': pm3, 'plaq3_err': pe3,
        'plaq2_mean': pm2, 'plaq2_err': pe2,
        'plaq1_mean': pm1, 'plaq1_err': pe1,
        'R_quark_mean': Rq_m, 'R_quark_err': Rq_e,
        'R_lepton_mean': Rl_m, 'R_lepton_err': Rl_e,
        'link_acc': therm_stats.link_rate,
        'quark_acc': therm_stats.quark_rate,
        'lepton_acc': therm_stats.lepton_rate,
        't_therm_s': round(t_therm, 2),
        't_meas_s': round(t_meas, 2),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SM pure-gauge L={cfg.L} results")
        print(f"{'='*60}")
        print(f"  plaq_3  = {pm3:.6f} ± {pe3:.6f}")
        print(f"  plaq_2  = {pm2:.6f} ± {pe2:.6f}")
        print(f"  plaq_1  = {pm1:.6f} ± {pe1:.6f}")
        if not pure_gauge:
            print(f"  R_quark = {Rq_m:.6f} ± {Rq_e:.6f}")
            print(f"  R_lepton= {Rl_m:.6f} ± {Rl_e:.6f}")
        print(f"  accept(link)={therm_stats.link_rate:.3f}  "
              f"quark={therm_stats.quark_rate:.3f}  "
              f"lepton={therm_stats.lepton_rate:.3f}")
        print(f"  timing: therm={t_therm:.1f}s  meas={t_meas:.1f}s")
        print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SM L=4 pure-gauge (or full) point run")
    p.add_argument("--L",       type=int,   default=4)
    p.add_argument("--beta-3",  type=float, default=6.0, dest="beta_3")
    p.add_argument("--beta-2",  type=float, default=4.0, dest="beta_2")
    p.add_argument("--beta-1",  type=float, default=2.0, dest="beta_1")
    p.add_argument("--kappa-q", type=float, default=0.0, dest="kappa_q")
    p.add_argument("--kappa-l", type=float, default=0.0, dest="kappa_l")
    p.add_argument("--n-therm", type=int,   default=500, dest="n_therm")
    p.add_argument("--n-meas",  type=int,   default=200, dest="n_measure")
    p.add_argument("--n-skip",  type=int,   default=2,   dest="n_skip")
    p.add_argument("--seed",    type=int,   default=123)
    p.add_argument("--hot",     action="store_true", dest="hot_start", default=True)
    p.add_argument("--cold",    action="store_false",dest="hot_start")
    p.add_argument("--eps-link",   type=float, default=0.3, dest="eps_link")
    p.add_argument("--eps-quark",  type=float, default=0.3, dest="eps_quark")
    p.add_argument("--eps-lepton", type=float, default=0.3, dest="eps_lepton")
    p.add_argument("--tune-every", type=int, default=50, dest="tune_every")
    p.add_argument("--out",     type=str,   default="sm_pure_gauge_L4.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SMConfig(
        L=args.L,
        beta_3=args.beta_3, beta_2=args.beta_2, beta_1=args.beta_1,
        kappa_q=args.kappa_q, kappa_l=args.kappa_l,
        n_therm=args.n_therm, n_measure=args.n_measure, n_skip=args.n_skip,
        hot_start=args.hot_start, seed=args.seed,
        eps_link=args.eps_link, eps_quark=args.eps_quark, eps_lepton=args.eps_lepton,
        tune_every=args.tune_every,
    )
    results = run_sm_point(cfg, verbose=True)

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out}")
