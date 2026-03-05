"""
P3-A: SU(2) L=10 Finite-Size Scaling run
=========================================
Protocol mirrors the SU(3) P1 FSS run (n_commutator.py, p_null_distribution.py).
Parameters: beta=4.0, kappa=0.3, L=10, N_therm=4000, N_meas=500, seed=203

After this run, update main.tex:
  1. Add L=10 row to SU(2) FSS table (tab:su2_bscan or create tab:su2_fss)
  2. Fit all three ansatze (1/L, 1/L+1/L^2, 1/L^2) to R(4,6,8,10)
  3. Update R_infty(SU2) estimate and uncertainty
  4. Update Table 6 L=inf row: R(SU2) -> new R_infty, ratio -> 0.3539/new_R_infty
  5. Update ordering claim in eq:hierarchy and discussion

Run:
  python su2_l10_run.py

Output:
  su2_l10_results.json
"""

import json, time, sys
import numpy as np
from pathlib import Path

# ── locate simulation.py ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
SU2_PATH   = SCRIPT_DIR          # simulation.py lives alongside this script
sys.path.insert(0, str(SU2_PATH))

try:
    from simulation import SimConfig, run_point   # type: ignore
    USE_MODULE = True
except ImportError:
    USE_MODULE = False
    print("WARNING: simulation.py not found on path; falling back to stub.")

# ── parameters ────────────────────────────────────────────────────────────────
SEED    = 203
RNG     = np.random.default_rng(SEED)
L       = 10
BETA    = 4.0
KAPPA   = 0.3
N_THERM = 4000    # thermalisation sweeps
N_MEAS  = 500     # measurement sweeps (same as SU(3) P1)
N_SKIP  = 5       # sweeps between measurements

# Known FSS data (from Table 5 / §4.3):
SU2_KNOWN = {4: (0.347, 0.005), 6: (0.360, 0.003), 8: (0.360, 0.002)}


def fit_fss(L_arr, R_arr, sigma_arr):
    """Fit three FSS ansatze; return (R_inf, sigma_stat, sigma_syst)."""
    L_arr    = np.array(L_arr, dtype=float)
    R_arr    = np.array(R_arr, dtype=float)
    sigma_arr = np.array(sigma_arr, dtype=float)

    results = {}
    for name, feats in [
        ("1/L",       lambda L: np.column_stack([np.ones_like(L), 1/L])),
        ("1/L+1/L2",  lambda L: np.column_stack([np.ones_like(L), 1/L, 1/L**2])),
        ("1/L2",      lambda L: np.column_stack([np.ones_like(L), 1/L**2])),
    ]:
        A = feats(L_arr)
        W = np.diag(1 / sigma_arr**2)
        AtWA = A.T @ W @ A
        AtWb = A.T @ (W @ R_arr)
        try:
            coeffs = np.linalg.solve(AtWA, AtWb)
            R_inf  = coeffs[0]
            res    = R_arr - A @ coeffs
            chi2   = float(res @ W @ res)
            dof    = len(L_arr) - A.shape[1]
            results[name] = {"R_inf": float(R_inf), "chi2": chi2,
                             "dof": dof, "chi2dof": chi2/dof if dof>0 else np.nan}
        except np.linalg.LinAlgError:
            results[name] = {"R_inf": np.nan, "chi2": np.nan, "dof": 0, "chi2dof": np.nan}

    valid = [v["R_inf"] for v in results.values() if np.isfinite(v["R_inf"])]
    best_key = min(results, key=lambda k: results[k]["chi2dof"]
                   if np.isfinite(results[k]["chi2dof"]) else 1e9)
    R_inf_best = results[best_key]["R_inf"]
    spread     = max(valid) - min(valid) if valid else 0.0
    sigma_syst = spread / 2
    return R_inf_best, results, sigma_syst


def main():
    t0 = time.time()
    print(f"SU(2) L=10 FSS run  seed={SEED}  beta={BETA}  kappa={KAPPA}")
    print(f"N_therm={N_THERM}  N_meas={N_MEAS}  N_skip={N_SKIP}")

    if USE_MODULE:
        # ── Real SU(2) simulation via SimConfig / run_point ───────────────────
        cfg = SimConfig(
            L          = L,
            beta_g     = BETA,
            kappa      = KAPPA,
            n_therm    = N_THERM,
            n_measure  = N_MEAS,
            n_skip     = N_SKIP,
            seed       = SEED,
            gauge_group= "SU2",
            hot_start  = True,
        )
        print("Running simulation (thermalise + measure)...")
        res = run_point(cfg, verbose=True)
        R_mean  = float(res["R_mean"])
        sigma_R = float(res["R_err"])
        tau_int = float(res.get("tau_int", 0.5))
        R_naive = sigma_R  # run_point already applies autocorr correction
        R_samples = None   # run_point doesn't expose raw samples
    else:
        # ── Stub: placeholder value for testing purposes ─────────────────────
        print("STUB mode: using placeholder R=0.362 (replace with real run)")
        R_samples = RNG.normal(0.362, 0.003, N_MEAS)
        R_mean  = float(np.mean(R_samples))
        R_naive = float(np.std(R_samples, ddof=1) / np.sqrt(len(R_samples)))

        # ── Madras-Sokal autocorrelation correction ───────────────────────────
        N = len(R_samples)
        Rc = R_samples - R_mean
        gamma0 = float(np.var(R_samples, ddof=0))
        tau_int = 0.5
        W = 6  # windowed sum
        for lag in range(1, min(W*10, N//2)):
            gamma_t = float(np.dot(Rc[:-lag], Rc[lag:]) / N)
            if gamma_t <= 0:
                break
            tau_int += gamma_t / gamma0
            if lag >= W * tau_int:
                break
        sigma_R = R_naive * float(np.sqrt(2 * tau_int))

    print(f"\nL=10 result: R = {R_mean:.4f} ± {sigma_R:.5f}  "
          f"(tau_int={tau_int:.2f})")

    # ── Four-point FSS fit ────────────────────────────────────────────────────
    L_arr     = [4, 6, 8, 10]
    R_arr     = [SU2_KNOWN[4][0], SU2_KNOWN[6][0], SU2_KNOWN[8][0], R_mean]
    sigma_arr = [SU2_KNOWN[4][1], SU2_KNOWN[6][1], SU2_KNOWN[8][1], sigma_R]

    R_inf_best, fit_results, sigma_syst = fit_fss(L_arr, R_arr, sigma_arr)
    sigma_stat = min(sigma_arr)  # representative; propagate properly in paper

    print(f"\nFSS fits (four-point, L=4,6,8,10):")
    for name, r in fit_results.items():
        print(f"  {name:12s}  R_inf={r['R_inf']:.4f}  chi2/dof={r['chi2dof']:.2f}")
    print(f"\nBest R_inf(SU2) = {R_inf_best:.4f} ± {sigma_stat:.4f}(stat) "
          f"± {sigma_syst:.4f}(syst)")
    print(f"Combined: {R_inf_best:.4f} ± {np.sqrt(sigma_stat**2+sigma_syst**2):.4f}")

    # ── Ordering check ────────────────────────────────────────────────────────
    R_inf_SU3 = 0.3539
    ratio_inf = R_inf_SU3 / R_inf_best if R_inf_best > 0 else np.nan
    print(f"\nR_inf(SU3)/R_inf(SU2) = {ratio_inf:.3f}  "
          f"({'SU3>SU2' if ratio_inf>1 else 'SU3<SU2'} at L=inf)")

    result = {
        "seed": SEED, "L": L, "beta": BETA, "kappa": KAPPA,
        "N_therm": N_THERM, "N_meas": N_MEAS, "N_skip": N_SKIP,
        "R_mean": R_mean, "R_naive": R_naive, "tau_int": tau_int,
        "sigma_R": sigma_R,
        "fss_inputs": {"L": L_arr, "R": R_arr, "sigma": sigma_arr},
        "fss_fits": fit_results,
        "R_inf_best": R_inf_best, "sigma_stat": sigma_stat,
        "sigma_syst": sigma_syst,
        "R_inf_SU3": R_inf_SU3,
        "ratio_SU3_over_SU2": ratio_inf,
        "runtime_s": time.time() - t0,
        "stub_mode": not USE_MODULE,
    }

    out = SCRIPT_DIR / "su2_l10_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {out}")
    return result


if __name__ == "__main__":
    main()
