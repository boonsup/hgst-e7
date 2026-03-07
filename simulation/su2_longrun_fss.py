#!/usr/bin/env python3
"""
su2_longrun_fss.py — Long SU(2) runs with proper autocorrelation + FSS analysis.

Reviewer Response (Issue 2):
  - N_meas = 2000, n_skip = 5, n_therm = 5000 for each L ∈ {4,6,8,10,12}
  - Jackknife blocking with block_size > tau_int for error estimation
  - Integrated autocorrelation time via window method (Madras & Sokal 1988)
  - FSS model selection: AIC/BIC over 1/L, 1/L², 1/L + 1/L²
  - Bootstrap on fit parameters for R_inf uncertainty

Output: su3_scans/data/su2_longrun_fss.json
"""

from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulation import SimConfig, run_point
from fields import initialize_links, initialize_matter
from lattice import Lattice2D
from updates import MetropolisUpdater
from observables import measure, ObservableAccumulator

# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
BETA   = 8.0
KAPPA  = 0.3
N_THERM = 5000
N_MEAS  = 2000
N_SKIP  = 5
SIZES   = [4, 6, 8, 10, 12]
BASE_SEED = 42

OUT_FILE = Path(__file__).parent.parent / "su3_scans" / "data" / "su2_longrun_fss.json"

# ─────────────────────────────────────────────────────────────────────────────
# Autocorrelation time (windowed method, Madras & Sokal 1988)
# ─────────────────────────────────────────────────────────────────────────────
def integrated_autocorr(x: np.ndarray, c: float = 5.0) -> float:
    """Estimate integrated autocorrelation time using windowing."""
    n = len(x)
    mu = np.mean(x)
    xc = x - mu
    # Normalized autocorrelation
    acf = np.correlate(xc, xc, mode='full')[n-1:]
    if acf[0] <= 0:
        return 0.5
    acf = acf / acf[0]
    tau = 0.5
    for t in range(1, n // 2):
        tau += acf[t]
        if t >= c * tau:  # window condition
            break
    return max(tau, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Jackknife blocking
# ─────────────────────────────────────────────────────────────────────────────
def jackknife_blocked(x: np.ndarray, block_size: int) -> tuple[float, float]:
    """Return (mean, jackknife std error) with given block size."""
    n = len(x)
    n_blocks = n // block_size
    if n_blocks < 4:
        return float(np.mean(x)), float(np.std(x) / np.sqrt(n))
    blocks = np.array([np.mean(x[i*block_size:(i+1)*block_size])
                       for i in range(n_blocks)])
    mean = np.mean(blocks)
    jk_means = np.array([(np.sum(blocks) - b) / (n_blocks - 1) for b in blocks])
    jk_var = (n_blocks - 1) / n_blocks * np.sum((jk_means - mean)**2)
    return float(mean), float(np.sqrt(jk_var))


# ─────────────────────────────────────────────────────────────────────────────
# Run one lattice size and collect full R time series
# ─────────────────────────────────────────────────────────────────────────────
def run_and_collect(L: int, seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  SU(2) L={L}, β={BETA}, N_therm={N_THERM}, N_meas={N_MEAS}")
    print(f"{'='*60}")

    lat    = Lattice2D(L)
    links  = initialize_links(lat, group='su2', random=True)
    matter = initialize_matter(lat, group='su2', random=True)

    upd = MetropolisUpdater(
        lat, links, matter,
        beta_g='su2', kappa=KAPPA, gauge_group='SU2',
        eps_link=0.6, eps_matter=2.0, seed=seed,
    )
    # Override beta_g properly
    upd.beta_g = BETA

    # Thermalize
    t0 = time.perf_counter()
    therm_stats = upd.thermalize(N_THERM, tune_every=500)
    t_therm = time.perf_counter() - t0
    print(f"  Therm done ({t_therm:.1f}s)  link_acc={therm_stats.link_rate:.3f}")

    # Measure
    R_series = []
    t1 = time.perf_counter()
    for step in range(N_MEAS):
        for _ in range(N_SKIP):
            upd.sweep()
        obs = measure(links, matter, lat, skip_R=False)
        R_series.append(float(obs.R))
        if (step + 1) % 500 == 0:
            elapsed = time.perf_counter() - t1
            print(f"    [{step+1}/{N_MEAS}]  R={obs.R:.4f}  t={elapsed:.0f}s")
    t_meas = time.perf_counter() - t1

    R_arr = np.array(R_series)
    tau = integrated_autocorr(R_arr)
    n_eff = int(N_MEAS / (2 * tau))
    block_size = max(1, int(2 * tau))
    R_mean, R_jk_err = jackknife_blocked(R_arr, block_size)

    print(f"  R = {R_mean:.5f} ± {R_jk_err:.5f}")
    print(f"  tau_int = {tau:.2f}, N_eff = {n_eff}, block_size = {block_size}")

    return {
        "L": L,
        "beta": BETA,
        "kappa": KAPPA,
        "seed": seed,
        "N_therm": N_THERM,
        "N_meas": N_MEAS,
        "N_skip": N_SKIP,
        "R_mean": R_mean,
        "R_err_jk": R_jk_err,
        "tau_int": float(tau),
        "N_eff": n_eff,
        "block_size": block_size,
        "t_therm_s": t_therm,
        "t_meas_s": t_meas,
        "link_acc": therm_stats.link_rate,
        "R_series": R_series,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FSS analysis with model selection
# ─────────────────────────────────────────────────────────────────────────────
def fss_analysis(data: list[dict]) -> dict:
    """Fit R(L) with multiple FSS ansätze; select by AIC."""
    Ls = np.array([d["L"] for d in data], dtype=float)
    Rs = np.array([d["R_mean"] for d in data])
    Es = np.array([d["R_err_jk"] for d in data])

    models = {
        "1/L":      (lambda L, Ri, a:        Ri + a/L,           2),
        "1/L2":     (lambda L, Ri, a:        Ri + a/L**2,        2),
        "1/L+1/L2": (lambda L, Ri, a, b:    Ri + a/L + b/L**2,   3),
    }

    results = {}
    n = len(Ls)

    for name, (func, n_params) in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(func, Ls, Rs, sigma=Es, absolute_sigma=True, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            R_pred = func(Ls, *popt)
            residuals = (Rs - R_pred) / Es
            chi2 = float(np.sum(residuals**2))
            dof = n - n_params
            aic = chi2 + 2 * n_params
            bic = chi2 + n_params * np.log(n)
            results[name] = {
                "R_inf": float(popt[0]),
                "R_inf_err": float(perr[0]),
                "params": popt.tolist(),
                "params_err": perr.tolist(),
                "chi2": chi2,
                "dof": max(dof, 1),
                "chi2dof": chi2 / max(dof, 1),
                "AIC": aic,
                "BIC": bic,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    # Select best by AIC
    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        best_name = min(valid, key=lambda k: valid[k]["AIC"])
    else:
        best_name = None

    # Bootstrap uncertainty on R_inf for best model
    R_inf_bootstrap = []
    if best_name and best_name in valid:
        func, n_params = models[best_name][:2], models[best_name][1]
        func_f = models[best_name][0]
        rng = np.random.default_rng(999)
        for _ in range(1000):
            Rs_boot = Rs + rng.normal(0, Es)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt_b, _ = curve_fit(func_f, Ls, Rs_boot, sigma=Es, absolute_sigma=True, maxfev=5000)
                R_inf_bootstrap.append(float(popt_b[0]))
            except Exception:
                pass
        if R_inf_bootstrap:
            valid[best_name]["R_inf_bootstrap_err"] = float(np.std(R_inf_bootstrap))

    return {
        "models": results,
        "best_model": best_name,
        "L_values": Ls.tolist(),
        "R_values": Rs.tolist(),
        "R_errors": Es.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    all_runs = []
    t_total = time.perf_counter()

    for i, L in enumerate(SIZES):
        seed = BASE_SEED + i * 17
        try:
            result = run_and_collect(L, seed)
            all_runs.append(result)
            # Save incrementally (no time series in incremental save)
            incremental = [{k: v for k, v in r.items() if k != "R_series"}
                           for r in all_runs]
            tmp = OUT_FILE.with_suffix(".tmp.json")
            with open(tmp, "w") as f:
                json.dump(incremental, f, indent=2)
            print(f"  Saved incremental to {tmp}")
        except Exception as e:
            print(f"  ERROR for L={L}: {e}")
            import traceback; traceback.print_exc()

    # FSS analysis (L>=6 only for fitting)
    fss_data = [r for r in all_runs if r["L"] >= 6]
    fss = fss_analysis(fss_data)

    total_time = time.perf_counter() - t_total

    # Full output
    output = {
        "metadata": {
            "beta": BETA,
            "kappa": KAPPA,
            "N_therm": N_THERM,
            "N_meas": N_MEAS,
            "N_skip": N_SKIP,
            "sizes": SIZES,
            "total_time_s": total_time,
            "generated": "2026-03-07",
        },
        "runs": [{k: v for k, v in r.items() if k != "R_series"} for r in all_runs],
        "R_series": {str(r["L"]): r["R_series"] for r in all_runs},
        "fss": fss,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Saved to {OUT_FILE}")
    print(f"Total time: {total_time:.0f}s")
    if fss["best_model"]:
        bm = fss["models"][fss["best_model"]]
        print(f"Best FSS: {fss['best_model']}  R_inf = {bm['R_inf']:.5f} ± {bm['R_inf_err']:.5f}  chi2/dof = {bm['chi2dof']:.3f}  AIC={bm['AIC']:.2f}")


if __name__ == "__main__":
    main()
