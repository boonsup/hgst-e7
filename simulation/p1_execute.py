#!/usr/bin/env python3
"""
p1_execute.py — REVISION P1: Autocorrelation + L=10 FSS + Multi-Ansatz Fits
=============================================================================
Tasks executed:
  Task A — Add L=10 FSS point for SM and SU(3); fit three ansätze
  Task B — Compute τ_int (Madras-Sokal windowing); correct all σ_R
  Task C — Document corrected triad-count formula; output σ_correct

Outputs:
  p1_sm_fss_corrected.json   — SM FSS L=4,6,8,10 with τ_int and corrected σ
  p1_su3_fss_corrected.json  — SU(3) FSS L=4,6,8,10 with τ_int and corrected σ
  p1_fss_fits.json           — Three-ansatz FSS fit results for SM and SU(3)
  p1_summary.txt             — Human-readable summary for updating main.tex

Usage:
    python p1_execute.py [--fast] [--skip-su3-l10]

    --fast      : use n_meas=500 at all sizes (for testing; default uses
                  n_meas=2000 for L≤8, n_meas=500 for L=10)
    --skip-su3-l10 : skip SU(3) L=10 (SM L=10 still runs); saves ~10 min
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import simulation modules
sys.path.insert(0, str(Path(__file__).parent))
from lattice import Lattice2D
from sm_fields import initialize_sm_links, initialize_quarks, initialize_leptons
from sm_updates import SMUpdater
from sm_observables import measure as sm_measure, _bfs_paths as sm_bfs_paths
from simulation import SimConfig, run_point


# ===========================================================================
# Madras-Sokal integrated autocorrelation time
# ===========================================================================

def madras_sokal_tauint(
    ts: np.ndarray,
    c: float = 6.0,
    max_window: int = 500,
) -> Tuple[float, np.ndarray]:
    """
    Estimate the integrated autocorrelation time τ_int using the
    Madras-Sokal windowing algorithm.

    Reference: Madras & Sokal (1988), J. Stat. Phys. 50, 109-186.

    Algorithm:
      1. Compute normalised ACF Γ(t) = C(t) / C(0).
      2. Increase window W until W >= c * τ_int(W); use that τ_int.

    Returns
    -------
    tau_int : float    Integrated autocorrelation time (in units of measurements)
    gamma   : ndarray  Normalised ACF Γ(t) up to window (diagnostic)
    """
    N = len(ts)
    mu = np.mean(ts)
    x = ts - mu

    # Compute ACF up to max_window
    W_max = min(max_window, N // 4)
    acf = np.array([
        np.dot(x[:N-t], x[t:]) / (N - t) for t in range(W_max + 1)
    ])
    if acf[0] <= 0:
        return 0.5, acf / max(abs(acf[0]), 1e-15)

    gamma = acf / acf[0]  # normalised ACF

    # Sokal's windowing: increase W until W >= c * tau_int(W)
    tau = 0.5
    for W in range(1, W_max + 1):
        tau += gamma[W]
        if W >= c * tau:
            return max(tau, 0.5), gamma[:W+1]

    # If window never closed, return estimate at W_max (conservative)
    return max(tau, 0.5), gamma


def corrected_stats(
    ts: np.ndarray,
    c: float = 6.0,
) -> Tuple[float, float, float, float]:
    """
    Returns (mean, sigma_naive, sigma_corrected, tau_int) for a time series.

    sigma_corrected = sigma_naive * sqrt(2 * tau_int)
    sigma_naive     = std(ts) / sqrt(N)
    """
    N = len(ts)
    mean = float(np.mean(ts))
    sigma_naive = float(np.std(ts, ddof=1) / np.sqrt(N))
    tau_int, _ = madras_sokal_tauint(ts, c=c)
    sigma_corrected = float(sigma_naive * np.sqrt(2.0 * tau_int))
    return mean, sigma_naive, sigma_corrected, float(tau_int)


# ===========================================================================
# FSS multi-ansatz fitter
# ===========================================================================

def fss_fit_one_ansatz(
    Ls: np.ndarray,
    Rs: np.ndarray,
    sigmas: np.ndarray,
    ansatz: str,
) -> Dict:
    """
    Fit R(L) = R_inf + ... to data {L, R, sigma}.

    ansatz in {"1/L", "1/L+1/L2", "1/L2"}

    Returns dict with keys: ansatz, R_inf, R_inf_err, params, chi2dof, dof, converged
    """
    from scipy.optimize import curve_fit

    w = 1.0 / sigmas**2  # weights

    if ansatz == "1/L":
        def model(L, Rinf, a):
            return Rinf + a / L
        p0 = [Rs[-1], 0.1]
        labels = ["R_inf", "a"]
    elif ansatz == "1/L+1/L2":
        def model(L, Rinf, a, b):
            return Rinf + a / L + b / L**2
        p0 = [Rs[-1], 0.1, 0.1]
        labels = ["R_inf", "a", "b"]
    elif ansatz == "1/L2":
        def model(L, Rinf, a):
            return Rinf + a / L**2
        p0 = [Rs[-1], 0.1]
        labels = ["R_inf", "a"]
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")

    try:
        popt, pcov = curve_fit(
            model, Ls, Rs,
            p0=p0, sigma=sigmas, absolute_sigma=True,
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        residuals = (Rs - model(Ls, *popt)) / sigmas
        chi2 = float(np.sum(residuals**2))
        dof = len(Ls) - len(popt)
        chi2dof = chi2 / dof if dof > 0 else np.nan
        return {
            "ansatz":    ansatz,
            "R_inf":     float(popt[0]),
            "R_inf_err": float(perr[0]),
            "params":    {l: float(v) for l, v in zip(labels, popt)},
            "param_errs": {l: float(v) for l, v in zip(labels, perr)},
            "chi2":      chi2,
            "chi2dof":   chi2dof,
            "dof":       dof,
            "converged": True,
        }
    except Exception as exc:
        return {
            "ansatz":    ansatz,
            "R_inf":     np.nan,
            "R_inf_err": np.nan,
            "params":    {},
            "chi2":      np.nan,
            "chi2dof":   np.nan,
            "dof":       len(Ls) - len(p0),
            "converged": False,
            "error":     str(exc),
        }


def fss_multi_ansatz(
    Ls: List[int],
    Rs: List[float],
    sigmas: List[float],
    label: str = "",
) -> Dict:
    """
    Fit three FSS ansätze and report R_inf ± stat ± syst.

    Returns dict with:
      fits        : list of fit dicts for each ansatz
      R_inf_best  : R_inf from best-fit ansatz (lowest chi2dof)
      R_inf_stat  : statistical error from best fit
      R_inf_syst  : 0.5 * spread of R_inf across converged fits
      ansatz_best : name of best-fit ansatz
    """
    Larr = np.array(Ls, dtype=float)
    Rarr = np.array(Rs, dtype=float)
    sarr = np.array(sigmas, dtype=float)

    ansätze = ["1/L", "1/L+1/L2", "1/L2"]
    fits = [fss_fit_one_ansatz(Larr, Rarr, sarr, a) for a in ansätze]

    converged = [f for f in fits if f["converged"]]
    if not converged:
        return {"label": label, "fits": fits,
                "R_inf_best": np.nan, "R_inf_stat": np.nan,
                "R_inf_syst": np.nan, "ansatz_best": "none"}

    # Best fit = lowest chi2dof among converged
    best = min(converged, key=lambda f: abs(f["chi2dof"] - 1.0)
                if not np.isnan(f["chi2dof"]) else 1e9)
    rinf_vals = [f["R_inf"] for f in converged]
    syst = 0.5 * (max(rinf_vals) - min(rinf_vals)) if len(rinf_vals) > 1 else 0.0

    return {
        "label":       label,
        "Ls":          Ls,
        "Rs":          Rs,
        "sigmas":      sigmas,
        "fits":        fits,
        "R_inf_best":  best["R_inf"],
        "R_inf_stat":  best["R_inf_err"],
        "R_inf_syst":  syst,
        "ansatz_best": best["ansatz"],
        "chi2dof_best": best["chi2dof"],
    }


# ===========================================================================
# SM FSS runner (saves full time series)
# ===========================================================================

@dataclass
class SMTimeSeries:
    L: int
    seed: int
    n_therm: int
    n_meas: int
    n_skip: int
    R_quark: np.ndarray = field(default_factory=lambda: np.array([]))
    R_lepton: np.ndarray = field(default_factory=lambda: np.array([]))
    plaq3: np.ndarray = field(default_factory=lambda: np.array([]))
    plaq2: np.ndarray = field(default_factory=lambda: np.array([]))
    plaq1: np.ndarray = field(default_factory=lambda: np.array([]))
    link_acc: float = 0.0
    quark_acc: float = 0.0
    lepton_acc: float = 0.0
    t_therm_s: float = 0.0
    t_meas_s: float = 0.0


def run_sm_fss_timeseries(
    L: int,
    seed: int,
    n_therm: int = 1000,
    n_meas: int = 2000,
    n_skip: int = 1,
    beta_3: float = 6.0,
    beta_2: float = 4.0,
    beta_1: float = 2.0,
    kappa_q: float = 0.2,
    kappa_l: float = 0.2,
    verbose: bool = True,
) -> SMTimeSeries:
    """Run SM FSS point and return full time series for autocorrelation analysis."""

    lat = Lattice2D(L)
    rng_links   = np.random.default_rng(seed)
    rng_quarks  = np.random.default_rng(seed + 1)
    rng_leptons = np.random.default_rng(seed + 2)

    links   = initialize_sm_links(lat,   random=True, rng=rng_links)
    quarks  = initialize_quarks(lat,     random=True, rng=rng_quarks)
    leptons = initialize_leptons(lat,    random=True, rng=rng_leptons)

    updater = SMUpdater(
        lat, links, quarks, leptons,
        beta_3=beta_3, beta_2=beta_2, beta_1=beta_1,
        kappa_q=kappa_q, kappa_l=kappa_l,
        eps_link=0.3, eps_quark=0.3, eps_lepton=0.3,
        seed=seed + 3,
        target_rate=0.5,
    )

    if verbose:
        print(f"\n[SM L={L}]  Thermalizing {n_therm} sweeps "
              f"(β₃={beta_3}, β₂={beta_2}, β₁={beta_1}, κ={kappa_q}) …")
    t0 = time.perf_counter()
    stats = updater.thermalize(
        n_therm,
        update_quarks=True, update_leptons=True,
        tune_every=100,
    )
    t_therm = time.perf_counter() - t0
    if verbose:
        print(f"  done ({t_therm:.1f}s)  "
              f"acc: link={stats.link_rate:.3f} q={stats.quark_rate:.3f} l={stats.lepton_rate:.3f}")

    if verbose:
        print(f"[SM L={L}]  Pre-computing BFS paths …")
    all_paths = sm_bfs_paths(lat)  # compute once, reuse every measurement

    if verbose:
        print(f"[SM L={L}]  Measuring {n_meas} samples (skip={n_skip}) …")

    rq_ts = np.empty(n_meas)
    rl_ts = np.empty(n_meas)
    p3_ts = np.empty(n_meas)
    p2_ts = np.empty(n_meas)
    p1_ts = np.empty(n_meas)

    t1 = time.perf_counter()
    for i in range(n_meas):
        for _ in range(n_skip):
            updater.sweep(update_links=True, update_quarks=True, update_leptons=True)
        obs = sm_measure(links, quarks, leptons, lat, skip_R=False,
                         all_paths=all_paths)
        rq_ts[i] = obs.R_quark
        rl_ts[i] = obs.R_lepton
        p3_ts[i] = obs.plaq_3
        p2_ts[i] = obs.plaq_2
        p1_ts[i] = obs.plaq_1
        if verbose and (i + 1) % 200 == 0:
            print(f"  [{i+1:4d}/{n_meas}]  R_q={rq_ts[i]:.4f}  R_l={rl_ts[i]:.4f}")
    t_meas = time.perf_counter() - t1

    ts = SMTimeSeries(
        L=L, seed=seed, n_therm=n_therm, n_meas=n_meas, n_skip=n_skip,
        R_quark=rq_ts, R_lepton=rl_ts,
        plaq3=p3_ts, plaq2=p2_ts, plaq1=p1_ts,
        link_acc=stats.link_rate,
        quark_acc=stats.quark_rate, lepton_acc=stats.lepton_rate,
        t_therm_s=round(t_therm, 2), t_meas_s=round(t_meas, 2),
    )
    if verbose:
        mu_q, s_naive_q, s_corr_q, tau_q = corrected_stats(rq_ts)
        print(f"  R_quark = {mu_q:.5f}  σ_naive={s_naive_q:.5f}  "
              f"σ_corr={s_corr_q:.5f}  τ_int={tau_q:.2f}")
    return ts


# ===========================================================================
# SU(3) FSS runner (saves full time series)
# ===========================================================================

@dataclass
class SU3TimeSeries:
    L: int
    beta: float
    kappa: float
    seed: int
    n_therm: int
    n_meas: int
    n_skip: int
    R: np.ndarray = field(default_factory=lambda: np.array([]))
    plaq: np.ndarray = field(default_factory=lambda: np.array([]))
    link_acc: float = 0.0
    matter_acc: float = 0.0
    t_therm_s: float = 0.0
    t_meas_s: float = 0.0


def run_su3_fss_timeseries(
    L: int,
    seed: int,
    beta: float = 8.0,
    kappa: float = 0.3,
    n_therm: int = 2000,
    n_meas: int = 2000,
    n_skip: int = 1,
    verbose: bool = True,
) -> SU3TimeSeries:
    """
    Run SU(3) FSS point using simulation.py's run_point and extract time series.
    We call run_point in streaming mode or use the internal accumulator.

    Since run_point doesn't expose individual measurements, we call the
    lower-level API: SimConfig → initialize → thermalize → measure loop.
    """
    from lattice import Lattice2D as _Lat
    from fields import initialize_links, initialize_matter
    from updates import MetropolisUpdater
    from observables import measure as obs_measure, ObservableAccumulator

    lat = _Lat(L)
    rng = np.random.default_rng(seed)

    links  = initialize_links(lat, group="su3", random=True, rng=np.random.default_rng(seed))
    matter = initialize_matter(lat, group="su3", random=True, rng=np.random.default_rng(seed+1))

    updater = MetropolisUpdater(
        lat, links, matter,
        beta_g=beta, kappa=kappa,
        gauge_group='SU3',
        eps_link=0.2, eps_matter=2.0,
        seed=seed+2,
        target_rate=0.5,
    )

    if verbose:
        print(f"\n[SU3 L={L}]  Thermalizing {n_therm} sweeps (β={beta}, κ={kappa}) …")
    t0 = time.perf_counter()
    therm_stats = updater.thermalize(n_therm, tune_every=200)
    t_therm = time.perf_counter() - t0
    if verbose:
        print(f"  done ({t_therm:.1f}s)  "
              f"acc: link={therm_stats.link_rate:.3f} matter={therm_stats.matter_rate:.3f}")

    if verbose:
        print(f"[SU3 L={L}]  Measuring {n_meas} samples (skip={n_skip}) …")

    R_ts   = np.empty(n_meas)
    plq_ts = np.empty(n_meas)

    t1 = time.perf_counter()
    for i in range(n_meas):
        for _ in range(n_skip):
            updater.sweep()
        obs = obs_measure(links, matter, lat)
        R_ts[i]   = obs.R
        plq_ts[i] = obs.plaq_avg
        if verbose and (i + 1) % 400 == 0:
            print(f"  [{i+1:4d}/{n_meas}]  R={R_ts[i]:.5f}  plaq={plq_ts[i]:.5f}")
    t_meas = time.perf_counter() - t1

    ts = SU3TimeSeries(
        L=L, beta=beta, kappa=kappa, seed=seed,
        n_therm=n_therm, n_meas=n_meas, n_skip=n_skip,
        R=R_ts, plaq=plq_ts,
        link_acc=float(therm_stats.link_rate), matter_acc=float(therm_stats.matter_rate),
        t_therm_s=round(t_therm, 2), t_meas_s=round(t_meas, 2),
    )
    if verbose:
        mu, s_naive, s_corr, tau = corrected_stats(R_ts)
        print(f"  R = {mu:.5f}  σ_naive={s_naive:.5f}  "
              f"σ_corr={s_corr:.5f}  τ_int={tau:.2f}")
    return ts


# ===========================================================================
# Analyse time series → corrected JSON record
# ===========================================================================

def analyse_sm_ts(ts: SMTimeSeries, beta_3=6.0, beta_2=4.0, beta_1=2.0,
                  kappa_q=0.2, kappa_l=0.2) -> Dict:
    mq, snq, scq, tauq = corrected_stats(ts.R_quark)
    ml, snl, scl, taul = corrected_stats(ts.R_lepton)
    mp3, _, _, _ = corrected_stats(ts.plaq3)
    mp2, _, _, _ = corrected_stats(ts.plaq2)
    mp1, _, _, _ = corrected_stats(ts.plaq1)
    return {
        "L": ts.L,
        "beta_3": beta_3, "beta_2": beta_2, "beta_1": beta_1,
        "kappa_q": kappa_q, "kappa_l": kappa_l,
        "n_therm": ts.n_therm, "n_meas": ts.n_meas, "n_skip": ts.n_skip,
        "seed": ts.seed,
        # Plaquette (not critical for FSS but reported)
        "plaq3_mean": float(mp3),
        "plaq2_mean": float(mp2),
        "plaq1_mean": float(mp1),
        # R_quark
        "R_quark_mean":         float(mq),
        "R_quark_err_naive":    float(snq),
        "R_quark_err_corrected": float(scq),
        "R_quark_tau_int":      float(tauq),
        # R_lepton
        "R_lepton_mean":         float(ml),
        "R_lepton_err_naive":    float(snl),
        "R_lepton_err_corrected": float(scl),
        "R_lepton_tau_int":      float(taul),
        # acceptance
        "link_acc":   ts.link_acc,
        "quark_acc":  ts.quark_acc,
        "lepton_acc": ts.lepton_acc,
        "t_therm_s": ts.t_therm_s,
        "t_meas_s":  ts.t_meas_s,
    }


def analyse_su3_ts(ts: SU3TimeSeries) -> Dict:
    mu, sn, sc, tau = corrected_stats(ts.R)
    mp, _, _, _ = corrected_stats(ts.plaq)
    return {
        "L":    ts.L,
        "beta": ts.beta,
        "kappa": ts.kappa,
        "n_therm": ts.n_therm, "n_meas": ts.n_meas, "n_skip": ts.n_skip,
        "seed": ts.seed,
        "plaq_mean": float(mp),
        "R_mean":         float(mu),
        "R_err_naive":    float(sn),
        "R_err_corrected": float(sc),
        "R_tau_int":      float(tau),
        "link_acc":   ts.link_acc,
        "matter_acc": ts.matter_acc,
        "t_therm_s": ts.t_therm_s,
        "t_meas_s":  ts.t_meas_s,
    }


# ===========================================================================
# Summary text for main.tex update
# ===========================================================================

def format_summary(sm_fss: Dict, su3_fss: Dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("P1 REVISION SUMMARY — Values for updating preprint/main.tex")
    lines.append("=" * 70)
    lines.append("")
    lines.append("SECTION B+C: CORRECTED ERROR BARS (autocorrelation-aware)")
    lines.append("")
    lines.append("SM FSS (β₃=6, β₂=4, β₁=2, κ=0.2):")
    lines.append(f"  {'L':>4}  {'R_q':>8}  {'σ_naive':>10}  {'σ_corr':>10}  {'τ_int':>7}  {'N_meas':>6}")
    for rec in sm_fss["points"]:
        lines.append(f"  {rec['L']:>4}  {rec['R_quark_mean']:>8.5f}  "
                     f"{rec['R_quark_err_naive']:>10.6f}  "
                     f"{rec['R_quark_err_corrected']:>10.6f}  "
                     f"{rec['R_quark_tau_int']:>7.2f}  "
                     f"{rec['n_meas']:>6}")
    lines.append("")
    lines.append("SU(3) FSS (β=8.0, κ=0.3):")
    lines.append(f"  {'L':>4}  {'R':>8}  {'σ_naive':>10}  {'σ_corr':>10}  {'τ_int':>7}  {'N_meas':>6}")
    for rec in su3_fss["points"]:
        lines.append(f"  {rec['L']:>4}  {rec['R_mean']:>8.5f}  "
                     f"{rec['R_err_naive']:>10.6f}  "
                     f"{rec['R_err_corrected']:>10.6f}  "
                     f"{rec['R_tau_int']:>7.2f}  "
                     f"{rec['n_meas']:>6}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("")
    lines.append("SECTION A: FSS MULTI-ANSATZ FITS")
    lines.append("")

    for label, fss_results in [("SM R_quark", sm_fss["fss_quark"]),
                                 ("SM R_lepton", sm_fss["fss_lepton"]),
                                 ("SU(3) R", su3_fss["fss_R"])]:
        lines.append(f"{label}:")
        for f in fss_results["fits"]:
            if f["converged"]:
                lines.append(f"  ansatz={f['ansatz']:<14}  R_inf={f['R_inf']:.5f}±{f['R_inf_err']:.5f}"
                              f"  χ²/dof={f['chi2dof']:.3f} (dof={f['dof']})")
            else:
                lines.append(f"  ansatz={f['ansatz']:<14}  DID NOT CONVERGE: {f.get('error','')}")
        best = fss_results['ansatz_best']
        Ri   = fss_results['R_inf_best']
        Ri_stat = fss_results['R_inf_stat']
        Ri_syst = fss_results['R_inf_syst']
        lines.append(f"  BEST ({best}):  R_inf = {Ri:.5f} ± {Ri_stat:.5f}(stat) ± {Ri_syst:.5f}(syst)")
        lines.append(f"  → Combined:  R_inf = {Ri:.4f} ± {(Ri_stat**2+Ri_syst**2)**0.5:.4f}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("")
    lines.append("SECTION C: CORRECTED TRIAD-COUNT FORMULA (for §3.5 in main.tex)")
    lines.append("")
    lines.append("The error on R should be reported as:")
    lines.append("  σ_R = sqrt(R(1-R) / N_eff)  where  N_eff = N_meas / (2τ_int)")
    lines.append("OR equivalently:")
    lines.append("  σ_R = (std(R_ts) / sqrt(N_meas)) × sqrt(2τ_int)")
    lines.append("The factor sqrt(2τ_int) corrects for inter-configuration autocorrelation.")
    lines.append("Individual triads within one configuration are NOT independent.")
    lines.append("")
    lines.append("Observed correction factors (sqrt(2τ_int)):")
    for rec in sm_fss["points"]:
        if rec['R_quark_tau_int'] > 0:
            cf = (2 * rec['R_quark_tau_int']) ** 0.5
            lines.append(f"  SM L={rec['L']:>2}: τ_int={rec['R_quark_tau_int']:.2f}  "
                         f"correction factor={cf:.2f}×")
    for rec in su3_fss["points"]:
        if rec['R_tau_int'] > 0:
            cf = (2 * rec['R_tau_int']) ** 0.5
            lines.append(f"  SU3 L={rec['L']:>2}: τ_int={rec['R_tau_int']:.2f}  "
                         f"correction factor={cf:.2f}×")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ===========================================================================
# Main execution
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="P1 revision: autocorrelation + L=10 FSS")
    p.add_argument("--fast",         action="store_true",
                   help="Use n_meas=500 at all sizes (quick test)")
    p.add_argument("--skip-su3-l10", action="store_true",
                   help="Skip SU(3) L=10 run (saves ~10 min)")
    p.add_argument("--skip-sm",      action="store_true",
                   help="Skip SM FSS runs (resume SU(3) only, load existing SM JSON)")
    p.add_argument("--outdir", type=str, default=".",
                   help="Output directory for JSON and summary files")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Simulation parameters
    # -----------------------------------------------------------------------
    SM_PARAMS = dict(beta_3=6.0, beta_2=4.0, beta_1=2.0, kappa_q=0.2, kappa_l=0.2)
    SM_FSS_SIZES   = [4, 6, 8, 10]
    SM_FSS_SEEDS   = {4: 99, 6: 100, 8: 101, 10: 102}
    SM_FSS_NTHERM  = {4: 500, 6: 800, 8: 1000, 10: 1200}
    if args.fast:
        SM_FSS_NMEAS = {L: 500 for L in SM_FSS_SIZES}
    else:
        SM_FSS_NMEAS = {4: 2000, 6: 2000, 8: 2000, 10: 500}

    SU3_BETA  = 8.0
    SU3_KAPPA = 0.3
    SU3_FSS_SIZES   = [4, 6, 8, 10]
    SU3_FSS_SEEDS   = {4: 200, 6: 201, 8: 202, 10: 203}
    SU3_FSS_NTHERM  = {4: 1000, 6: 1500, 8: 2000, 10: 2500}
    if args.fast:
        SU3_FSS_NMEAS = {L: 500 for L in SU3_FSS_SIZES}
    else:
        SU3_FSS_NMEAS = {4: 2000, 6: 2000, 8: 2000, 10: 500}

    # -----------------------------------------------------------------------
    # SM FSS runs
    # -----------------------------------------------------------------------
    out_sm = outdir / "p1_sm_fss_corrected.json"
    if args.skip_sm and out_sm.exists():
        print(f"\n✓ SM FSS: loading existing {out_sm}")
        sm_fss_result = json.loads(out_sm.read_text())
        sm_records    = sm_fss_result["points"]
        fss_q         = sm_fss_result["fss_quark"]
        fss_l         = sm_fss_result["fss_lepton"]
    else:
        print("\n" + "=" * 70)
        print("P1 TASK A+B+C: SM FSS  (L = 4, 6, 8, 10)")
        print("=" * 70)

        sm_records = []
        for L in SM_FSS_SIZES:
            ts = run_sm_fss_timeseries(
                L=L, seed=SM_FSS_SEEDS[L],
                n_therm=SM_FSS_NTHERM[L],
                n_meas=SM_FSS_NMEAS[L],
                n_skip=1,
                **SM_PARAMS,
                verbose=True,
            )
            rec = analyse_sm_ts(ts, **SM_PARAMS)
            sm_records.append(rec)

        # FSS fits using corrected σ
        Ls_sm = [r["L"]                       for r in sm_records]
        Rq    = [r["R_quark_mean"]            for r in sm_records]
        sq    = [r["R_quark_err_corrected"]   for r in sm_records]
        Rl    = [r["R_lepton_mean"]           for r in sm_records]
        sl    = [r["R_lepton_err_corrected"]  for r in sm_records]

        fss_q = fss_multi_ansatz(Ls_sm, Rq, sq, label="SM_R_quark")
        fss_l = fss_multi_ansatz(Ls_sm, Rl, sl, label="SM_R_lepton")

        sm_fss_result = {
            "group": "SM",
            "beta_3": SM_PARAMS["beta_3"],
            "beta_2": SM_PARAMS["beta_2"],
            "beta_1": SM_PARAMS["beta_1"],
            "kappa_q": SM_PARAMS["kappa_q"],
            "kappa_l": SM_PARAMS["kappa_l"],
            "points": sm_records,
            "fss_quark":  fss_q,
            "fss_lepton": fss_l,
        }
        out_sm.write_text(json.dumps(sm_fss_result, indent=2, default=str))
        print(f"\n✓ SM FSS written to {out_sm}")

    # -----------------------------------------------------------------------
    # SU(3) FSS runs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("P1 TASK A+B+C: SU(3) FSS  (L = 4, 6, 8, 10)")
    print("=" * 70)

    su3_records = []
    for L in SU3_FSS_SIZES:
        if L == 10 and args.skip_su3_l10:
            print(f"\n[SU3 L=10] SKIPPED (--skip-su3-l10)")
            continue
        ts = run_su3_fss_timeseries(
            L=L, seed=SU3_FSS_SEEDS[L],
            beta=SU3_BETA, kappa=SU3_KAPPA,
            n_therm=SU3_FSS_NTHERM[L],
            n_meas=SU3_FSS_NMEAS[L],
            n_skip=1,
            verbose=True,
        )
        rec = analyse_su3_ts(ts)
        su3_records.append(rec)

    Ls_su3 = [r["L"]               for r in su3_records]
    R_su3  = [r["R_mean"]          for r in su3_records]
    s_su3  = [r["R_err_corrected"] for r in su3_records]

    fss_su3 = fss_multi_ansatz(Ls_su3, R_su3, s_su3, label="SU3_R")

    su3_fss_result = {
        "group": "SU3",
        "beta":  SU3_BETA,
        "kappa": SU3_KAPPA,
        "points":  su3_records,
        "fss_R":   fss_su3,
    }
    out_su3 = outdir / "p1_su3_fss_corrected.json"
    out_su3.write_text(json.dumps(su3_fss_result, indent=2, default=str))
    print(f"\n✓ SU(3) FSS written to {out_su3}")

    # -----------------------------------------------------------------------
    # Combined FSS fit summary
    # -----------------------------------------------------------------------
    out_fits = outdir / "p1_fss_fits.json"
    fits_summary = {
        "SM_R_quark":  fss_q,
        "SM_R_lepton": fss_l,
        "SU3_R":       fss_su3,
    }
    out_fits.write_text(json.dumps(fits_summary, indent=2, default=str))
    print(f"✓ FSS fit results written to {out_fits}")

    # -----------------------------------------------------------------------
    # Human-readable summary
    # -----------------------------------------------------------------------
    summary = format_summary(sm_fss_result, su3_fss_result)
    out_summary = outdir / "p1_summary.txt"
    out_summary.write_text(summary, encoding="utf-8")
    print(f"✓ Summary written to {out_summary}\n")

    print(summary)

    t_total = time.perf_counter() - t_start
    print(f"\nTotal wall time: {t_total/60:.1f} min")


if __name__ == "__main__":
    main()
