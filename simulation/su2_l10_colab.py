"""
su2_l10_colab.py — SU(2) L=10 Finite-Size Scaling (Colab-ready, zero dependencies)
=====================================================================================
Task: P3-A from REVISION2_PLAN.md
Paper: HGST E7 / main.tex (preprint/revision)
Purpose: measure R(SU2, L=10, beta=4.0, kappa=0.3) to complete the four-point FSS

This file is 100% self-contained — no local module imports.
Paste the entire file into a single Colab cell (or run as a .py script).

REQUIREMENTS (Colab runtime: any Python 3.10+ kernel):
  - numpy  (pre-installed in Colab)
  - scipy  (pre-installed in Colab)
  The two lines below install them if somehow missing.

OUTPUT FILES (written to current directory):
  su2_l10_runlog.json   — full per-measurement R time series + metadata
  su2_l10_summary.json  — import-ready dict for updating main.tex

EXPECTED RESULTS (from stub run with R≈0.362):
  R(L=10) ≈ 0.360–0.365
  R_inf(SU2, 4-pt FSS) ≈ 0.355–0.370 (all three ansatze + spread)
  Runtime: ~30–90 min on Colab CPU (T4 GPU not used — pure numpy)

HOW TO IMPORT BACK:
  import json
  with open("su2_l10_summary.json") as f:
      result = json.load(f)
  # result["R_mean"], result["sigma_R"], result["fss"]["R_inf_best"], etc.
"""

# ── Colab install guard ────────────────────────────────────────────────────────
import subprocess, sys
for _pkg in ["numpy", "scipy"]:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "-q"])

# ── Stdlib + numpy ─────────────────────────────────────────────────────────────
import gc
import json
import time
from itertools import combinations
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — RUN CONFIGURATION  (edit this section)
# ═══════════════════════════════════════════════════════════════════════════════

L       = 10       # lattice size (THIS run)
BETA    = 4.0      # inverse gauge coupling
KAPPA   = 0.3      # matter hopping
N_THERM = 4000     # thermalisation sweeps (discarded)
N_MEAS  = 500      # production measurements
N_SKIP  = 5        # sweeps between measurements
SEED    = 203      # RNG seed
EPS_LINK   = 0.30  # Metropolis link step
EPS_MATTER = 0.50  # Metropolis matter step
TUNE_EVERY = 200   # re-tune eps every N sweeps during thermalisation
TARGET_RATE = 0.50 # target Metropolis acceptance rate

# Known SU(2) points from existing paper data (Table 5 / §4.3 of main.tex)
# These feed the four-point FSS fit at the end.
SU2_KNOWN = {
    4: (0.347, 0.005),   # (R_mean, sigma_R) from production run
    6: (0.360, 0.003),
    8: (0.360, 0.002),
}
R_INF_SU3 = 0.3539   # four-point FSS result for SU(3) from main.tex

OUTLOG = "su2_l10_runlog.json"
OUTSUM = "su2_l10_summary.json"

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SU(2) ALGEBRA
# (inlined from su2.py; Haar measure, Metropolis proposal, group checks)
# ═══════════════════════════════════════════════════════════════════════════════

def _random_su2(rng: np.random.Generator) -> np.ndarray:
    """Haar-random SU(2) matrix via quaternion parameterisation."""
    z = rng.standard_normal(4)
    a = z / np.linalg.norm(z)
    return np.array([
        [a[0] + 1j*a[3],   a[2] + 1j*a[1]],
        [-a[2] + 1j*a[1],  a[0] - 1j*a[3]],
    ], dtype=np.complex128)


def _small_su2(eps: float, rng: np.random.Generator) -> np.ndarray:
    """Near-identity SU(2) with rotation angle ~ eps."""
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    theta = rng.uniform(-eps, eps)
    c, s = np.cos(theta), np.sin(theta)
    nx, ny, nz = axis
    return np.array([
        [c + 1j*nz*s,          ny*s + 1j*nx*s],
        [-ny*s + 1j*nx*s,      c - 1j*nz*s   ],
    ], dtype=np.complex128)


def _identity_su2() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)


def _dagger(U: np.ndarray) -> np.ndarray:
    return U.conj().T


def _random_doublet(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform unit vector in ℂ²."""
    v = rng.standard_normal(4).view(np.complex128)
    return v / np.linalg.norm(v)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LATTICE GEOMETRY
# (inlined from lattice.py; L×L open-boundary grade lattice)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Lattice:
    L: int
    N: int                              = field(init=False)
    edges: List[Tuple[int,int]]         = field(init=False)
    plaquettes: List[Tuple[int,int,int,int]] = field(init=False)
    r: np.ndarray                       = field(init=False)   # grade magnitudes
    bfs_paths: Dict[int, Dict[int, List[int]]] = field(init=False)

    def __post_init__(self):
        L = self.L
        self.N = L * L
        # edges
        edges = []
        for m in range(L):
            for n in range(L):
                i = n + m*L
                if n+1 < L: edges.append((i, i+1))
                if m+1 < L: edges.append((i, i+L))
        self.edges = edges
        # plaquettes
        plaq = []
        for m in range(L-1):
            for n in range(L-1):
                i0 = n     + m*L
                i1 = (n+1) + m*L
                i2 = (n+1) + (m+1)*L
                i3 = n     + (m+1)*L
                plaq.append((i0, i1, i2, i3))
        self.plaquettes = plaq
        # grade magnitudes r[i] = 2^(n-m), 1-indexed
        r = np.empty(self.N, dtype=float)
        for idx in range(self.N):
            n_idx = idx % L + 1   # 1-indexed
            m_idx = idx // L + 1
            r[idx] = 2.0 ** (n_idx - m_idx)
        self.r = r
        # precompute all BFS shortest paths (topology-fixed, done once)
        self.bfs_paths = self._precompute_bfs()

    def _precompute_bfs(self) -> Dict[int, Dict[int, List[int]]]:
        """BFS from every source; returns src → {dst: path} for all pairs."""
        edge_set = set(self.edges)
        adj: Dict[int, List[int]] = {i: [] for i in range(self.N)}
        for (i, j) in self.edges:
            adj[i].append(j); adj[j].append(i)
        all_paths: Dict[int, Dict[int, List[int]]] = {}
        for src in range(self.N):
            paths: Dict[int, List[int]] = {src: [src]}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in adj[cur]:
                    if nxt not in paths:
                        paths[nxt] = paths[cur] + [nxt]
                        queue.append(nxt)
            all_paths[src] = paths
        return all_paths


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIELD INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_links(lat: Lattice, rng: np.random.Generator, cold: bool = False):
    """links: dict (i,j) → 2×2 complex SU(2) matrix."""
    if cold:
        return {e: _identity_su2() for e in lat.edges}
    return {e: _random_su2(rng) for e in lat.edges}


def init_matter(lat: Lattice, rng: np.random.Generator, cold: bool = False):
    """matter: dict site → ψ = r_i · χ_i  (unit doublet × grade)."""
    if cold:
        chi0 = np.array([1.0+0j, 0.0+0j], dtype=np.complex128)
        return {s: lat.r[s] * chi0 for s in range(lat.N)}
    return {s: lat.r[s] * _random_doublet(rng) for s in range(lat.N)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ACTION (Wilson gauge + minimal hopping)
# (inlined from action.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _plaq_trace(links: dict, plaq: Tuple) -> float:
    """½ Re Tr U_plaquette for plaquette (i0,i1,i2,i3)."""
    i0, i1, i2, i3 = plaq
    edge_set = set(links.keys())
    U = _identity_su2()
    for a, b in [(i0,i1),(i1,i2),(i2,i3),(i3,i0)]:
        if (a,b) in links:    U = U @ links[(a,b)]
        elif (b,a) in links:  U = U @ _dagger(links[(b,a)])
    return 0.5 * float(np.real(np.trace(U)))


def _plaq_trace_fast(U01, U12, U23, U30) -> float:
    """½ Re Tr(U01 U12 U23† U30†)  for plaquette."""
    return 0.5 * float(np.real(np.trace(U01 @ U12 @ _dagger(U23) @ _dagger(U30))))
    # NOTE: stored as edges i0→i1, i1→i2, i2→i3 (up→right in CCW square)
    # plaquette = U(i0→i1) @ U(i1→i2) @ U†(i3→i2) @ U†(i0→i3)


def plaquette_average(links: dict, lat: Lattice) -> float:
    """Mean plaquette ½Re Tr over all plaquettes."""
    return float(np.mean([_plaq_trace(links, p) for p in lat.plaquettes]))


def _delta_action_link(links: dict, matter: dict, lat: Lattice,
                        edge: Tuple, U_new: np.ndarray,
                        beta: float, kappa: float) -> float:
    """Change in action when links[edge] → U_new."""
    i, j = edge
    U_old = links[edge]

    # Gauge part: sum over plaquettes containing this edge
    dS_gauge = 0.0
    for p_idx, plaq in enumerate(lat.plaquettes):
        i0,i1,i2,i3 = plaq
        plaq_edges = [(i0,i1),(i1,i2),(i2,i3),(i3,i0)]
        if edge not in plaq_edges and (j,i) not in plaq_edges:
            continue
        tr_old = _plaq_trace(links, plaq)
        links[edge] = U_new
        tr_new = _plaq_trace(links, plaq)
        links[edge] = U_old
        dS_gauge += -beta * (tr_new - tr_old)

    # Matter part: hopping terms i⇄j
    psi_i = matter[i]; psi_j = matter[j]
    hop_old = float(np.real(psi_i.conj() @ U_old          @ psi_j))
    hop_new = float(np.real(psi_i.conj() @ U_new          @ psi_j))
    hop_back_old = float(np.real(psi_j.conj() @ _dagger(U_old) @ psi_i))
    hop_back_new = float(np.real(psi_j.conj() @ _dagger(U_new) @ psi_i))
    dS_matter = -kappa * ((hop_new - hop_old) + (hop_back_new - hop_back_old))

    return dS_gauge + dS_matter


def _delta_action_matter(links: dict, matter: dict, lat: Lattice,
                          site: int, psi_new: np.ndarray,
                          kappa: float) -> float:
    """Change in action when matter[site] → psi_new."""
    psi_old = matter[site]
    dS = 0.0
    for j in _neighbours(lat, site):
        e_fwd = (site, j) if (site,j) in links else None
        e_bwd = (j, site) if (j,site) in links else None
        if e_fwd:
            U = links[e_fwd]
            dS += -kappa * (float(np.real(psi_new.conj() @ U          @ matter[j]))
                          - float(np.real(psi_old.conj() @ U          @ matter[j])))
            dS += -kappa * (float(np.real(matter[j].conj() @ _dagger(U) @ psi_new))
                          - float(np.real(matter[j].conj() @ _dagger(U) @ psi_old)))
        elif e_bwd:
            U = links[e_bwd]
            dS += -kappa * (float(np.real(psi_new.conj() @ _dagger(U) @ matter[j]))
                          - float(np.real(psi_old.conj() @ _dagger(U) @ matter[j])))
            dS += -kappa * (float(np.real(matter[j].conj() @ U @ psi_new))
                          - float(np.real(matter[j].conj() @ U @ psi_old)))
    return dS


def _neighbours(lat: Lattice, site: int) -> List[int]:
    """All lattice neighbours of site (both directions)."""
    nbs = []
    for (i, j) in lat.edges:
        if i == site: nbs.append(j)
        elif j == site: nbs.append(i)
    return nbs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — METROPOLIS UPDATES
# (inlined from updates.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _sweep(links: dict, matter: dict, lat: Lattice,
           beta: float, kappa: float,
           eps_link: float, eps_matter: float,
           rng: np.random.Generator) -> Tuple[float, float]:
    """One full Metropolis sweep. Returns (link_acc_rate, matter_acc_rate)."""
    n_link_acc = 0
    for edge in lat.edges:
        dU = _small_su2(eps_link, rng) @ links[edge]
        dS = _delta_action_link(links, matter, lat, edge, dU, beta, kappa)
        if dS <= 0.0 or rng.random() < np.exp(-dS):
            links[edge] = dU
            n_link_acc += 1

    n_mat_acc = 0
    for site in range(lat.N):
        r_i = lat.r[site]
        psi_old = matter[site]
        chi_old = psi_old / r_i if r_i > 1e-15 else psi_old
        eta = rng.standard_normal(4).view(np.complex128)
        eta /= np.linalg.norm(eta)
        chi_new = chi_old + eps_matter * eta
        chi_new /= np.linalg.norm(chi_new)
        psi_new = r_i * chi_new
        dS = _delta_action_matter(links, matter, lat, site, psi_new, kappa)
        if dS <= 0.0 or rng.random() < np.exp(-dS):
            matter[site] = psi_new
            n_mat_acc += 1

    n_e = len(lat.edges)
    return n_link_acc / n_e, n_mat_acc / lat.N


def _tune_eps(acc_rate: float, eps: float,
              target: float = TARGET_RATE,
              lo: float = 0.01, hi: float = 2.0) -> float:
    """Adaptive step-size tuning."""
    if acc_rate > target + 0.05:
        eps = min(eps * 1.15, hi)
    elif acc_rate < target - 0.05:
        eps = max(eps * 0.87, lo)
    return eps


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MIXED-FRACTION OBSERVABLE R
# (inlined + optimised from observables.py + positive_control.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_sign_matrix(links: dict, matter: dict, lat: Lattice) -> np.ndarray:
    """
    Build N×N sign matrix S where S[i,j] ∈ {-1, 0, +1}.
    S[i,j] = sign(Re[ψ_i† · U_path(i→j) · ψ_j])
    Uses precomputed BFS paths (lat.bfs_paths).
    """
    N = lat.N
    S = np.zeros((N, N), dtype=np.int8)
    for src in range(N):
        paths = lat.bfs_paths[src]
        psi_src = matter[src]
        for dst, path in paths.items():
            if dst == src:
                continue
            # path holonomy
            dim = 2
            U = np.eye(dim, dtype=np.complex128)
            for k in range(len(path) - 1):
                a, b = path[k], path[k+1]
                if (a,b) in links:    U = U @ links[(a,b)]
                elif (b,a) in links:  U = U @ _dagger(links[(b,a)])
            val = float(np.real(psi_src.conj() @ U @ matter[dst]))
            if   val > 0: S[src, dst] =  1
            elif val < 0: S[src, dst] = -1
    return S


def _mixed_R_from_sign_matrix(S: np.ndarray) -> float:
    """
    Vectorized MIXED-fraction R from sign matrix.

    Triple (i<j, k distinct): valid if S[i,j]≠0, S[i,k]≠0, S[k,j]≠0.
    MIXED if S[i,j] ≠ S[i,k]*S[k,j].
    """
    N = S.shape[0]
    nz = (S != 0)                          # (N,N) bool

    n_valid = 0
    n_mixed = 0

    for i in range(N):
        for j in range(i+1, N):
            if not nz[i, j]:
                continue
            s_ij = int(S[i, j])
            # valid mediators k: nz[i,k] AND nz[k,j], k≠i, k≠j
            valid_k = nz[i, :] & nz[:, j]
            valid_k[i] = False; valid_k[j] = False
            k_count = int(valid_k.sum())
            if k_count == 0:
                continue
            n_valid += k_count
            # mediated sign: S[i,k]*S[k,j]
            med = (S[i, :] * S[:, j]).astype(np.int16)   # (N,)
            mixed_k = valid_k & (med != s_ij)
            n_mixed += int(mixed_k.sum())

    return n_mixed / n_valid if n_valid > 0 else 0.0


def measure_R(links: dict, matter: dict, lat: Lattice) -> Tuple[float, float]:
    """
    Returns (R, plaq_avg).
    R = MIXED fraction; plaq_avg = mean plaquette ½Re Tr.
    """
    S = _compute_sign_matrix(links, matter, lat)
    R = _mixed_R_from_sign_matrix(S)
    plaq = plaquette_average(links, lat)
    return R, plaq


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — AUTOCORRELATION (Madras-Sokal windowing)
# ═══════════════════════════════════════════════════════════════════════════════

def madras_sokal_tau(ts: np.ndarray, c_factor: float = 6.0) -> float:
    """
    Integrated autocorrelation time τ_int via Madras-Sokal windowing.
    Window W = c_factor × current τ_int estimate.
    Returns τ_int ≥ 0.5.
    """
    N = len(ts)
    mu = ts.mean()
    tc = ts - mu
    gamma0 = float(np.dot(tc, tc) / N)
    if gamma0 < 1e-30:
        return 0.5
    tau = 0.5
    W_est = max(1, int(c_factor * tau))
    for lag in range(1, N // 2):
        gamma_t = float(np.dot(tc[:-lag], tc[lag:]) / N)
        if gamma_t <= 0:
            break
        tau += gamma_t / gamma0
        if lag >= int(c_factor * tau):
            break
    return max(0.5, tau)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FSS FITTING (three ansatze)
# ═══════════════════════════════════════════════════════════════════════════════

def fss_fit(L_arr, R_arr, sigma_arr) -> dict:
    """
    Fit three FSS ansatze; return best R_inf + errors.
    Ansatze:
      A1: R(L) = R_inf + a/L
      A2: R(L) = R_inf + a/L + b/L²
      A3: R(L) = R_inf + a/L²

    Returns dict with 'best', 'ansatze', 'sigma_stat', 'sigma_syst', 'R_inf_best'.
    """
    L_arr = np.array(L_arr, dtype=float)
    R_arr = np.array(R_arr, dtype=float)
    s_arr = np.array(sigma_arr, dtype=float)
    W = np.diag(1.0 / s_arr**2)

    def _fit(feats_fn):
        A = feats_fn(L_arr)
        try:
            AtWA = A.T @ W @ A
            AtWb = A.T @ (W @ R_arr)
            coeffs = np.linalg.solve(AtWA, AtWb)
            res = R_arr - A @ coeffs
            chi2 = float(res @ W @ res)
            dof  = len(L_arr) - A.shape[1]
            return float(coeffs[0]), chi2, dof
        except np.linalg.LinAlgError:
            return np.nan, np.nan, 0

    results = {}
    for name, fn in [
        ("1/L",      lambda L: np.column_stack([np.ones_like(L), 1/L])),
        ("1/L+1/L2", lambda L: np.column_stack([np.ones_like(L), 1/L, 1/L**2])),
        ("1/L2",     lambda L: np.column_stack([np.ones_like(L), 1/L**2])),
    ]:
        R_inf, chi2, dof = _fit(fn)
        results[name] = {
            "R_inf":    round(float(R_inf), 6) if np.isfinite(R_inf) else None,
            "chi2":     round(float(chi2),  4) if np.isfinite(chi2)  else None,
            "dof":      dof,
            "chi2dof":  round(float(chi2/dof), 4) if (dof>0 and np.isfinite(chi2)) else None,
        }

    valid = [v["R_inf"] for v in results.values() if v["R_inf"] is not None and np.isfinite(v["R_inf"])]
    if not valid:
        return {"ansatze": results, "R_inf_best": None,
                "sigma_stat": None, "sigma_syst": None}

    best_key = min(
        (k for k in results if results[k]["chi2dof"] is not None),
        key=lambda k: results[k]["chi2dof"] if results[k]["chi2dof"] is not None else 1e9
    )
    R_inf_best = results[best_key]["R_inf"]
    sigma_syst = (max(valid) - min(valid)) / 2
    sigma_stat = float(np.min(s_arr))   # conservative: min sigma from inputs

    return {
        "ansatze": results,
        "best_ansatz": best_key,
        "R_inf_best": round(R_inf_best, 6),
        "sigma_stat":  round(sigma_stat, 6),
        "sigma_syst":  round(sigma_syst, 6),
        "sigma_combined": round(float(np.sqrt(sigma_stat**2 + sigma_syst**2)), 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_su2_l10():
    t_start = time.time()

    print("=" * 65)
    print(f"SU(2) L={L} FSS run")
    print(f"  beta={BETA}  kappa={KAPPA}  seed={SEED}")
    print(f"  N_therm={N_THERM}  N_meas={N_MEAS}  N_skip={N_SKIP}")
    print("=" * 65)

    rng = np.random.default_rng(SEED)

    # ── Build lattice (precomputes BFS paths) ─────────────────────────────
    print("Building lattice and precomputing BFS paths...")
    lat = Lattice(L=L)
    print(f"  sites={lat.N}  edges={len(lat.edges)}  "
          f"plaquettes={len(lat.plaquettes)}  "
          f"BFS paths={sum(len(v) for v in lat.bfs_paths.values())}")

    # ── Initialise fields (hot start) ────────────────────────────────────
    links  = init_links(lat,  rng, cold=False)
    matter = init_matter(lat, rng, cold=False)

    eps_link   = EPS_LINK
    eps_matter = EPS_MATTER

    # ── Thermalisation ────────────────────────────────────────────────────
    print(f"\nThermalising ({N_THERM} sweeps)...")
    t_therm0 = time.time()
    for sw in range(N_THERM):
        link_acc, mat_acc = _sweep(links, matter, lat, BETA, KAPPA,
                                    eps_link, eps_matter, rng)
        if (sw + 1) % TUNE_EVERY == 0:
            eps_link   = _tune_eps(link_acc,   eps_link)
            eps_matter = _tune_eps(mat_acc, eps_matter)
        if (sw + 1) % 500 == 0:
            elapsed = time.time() - t_therm0
            eta = elapsed / (sw+1) * (N_THERM - sw - 1)
            print(f"  [{sw+1:4d}/{N_THERM}]  "
                  f"link_acc={link_acc:.3f}  mat_acc={mat_acc:.3f}  "
                  f"eps_L={eps_link:.4f}  eps_M={eps_matter:.4f}  "
                  f"ETA {eta/60:.1f} min")

    t_therm_done = time.time()
    print(f"Thermalisation done in {(t_therm_done-t_therm0)/60:.1f} min")

    # ── Production measurements ───────────────────────────────────────────
    print(f"\nMeasuring ({N_MEAS} measurements, every {N_SKIP} sweeps)...")
    R_series     = []
    plaq_series  = []
    t_meas0 = time.time()

    for meas in range(N_MEAS):
        for _ in range(N_SKIP):
            _sweep(links, matter, lat, BETA, KAPPA, eps_link, eps_matter, rng)
        R_val, plaq_val = measure_R(links, matter, lat)
        R_series.append(float(R_val))
        plaq_series.append(float(plaq_val))

        if (meas + 1) % 50 == 0:
            elapsed = time.time() - t_meas0
            remaining = elapsed / (meas+1) * (N_MEAS - meas - 1)
            print(f"  [{meas+1:3d}/{N_MEAS}]  "
                  f"R={np.mean(R_series):.4f}±{np.std(R_series)/np.sqrt(meas+1):.5f}  "
                  f"plaq={np.mean(plaq_series):.4f}  "
                  f"ETA {remaining/60:.1f} min")
        gc.collect()

    t_meas_done = time.time()
    print(f"\nMeasurements done in {(t_meas_done-t_meas0)/60:.1f} min")

    # ── Statistics ────────────────────────────────────────────────────────
    R_arr   = np.array(R_series)
    R_mean  = float(R_arr.mean())
    R_naive = float(R_arr.std(ddof=1) / np.sqrt(len(R_arr)))
    tau_int = madras_sokal_tau(R_arr)
    sigma_R = R_naive * float(np.sqrt(2.0 * tau_int))

    plaq_mean = float(np.mean(plaq_series))
    plaq_std  = float(np.std(plaq_series, ddof=1))

    print(f"\n{'='*65}")
    print(f"L={L} result:")
    print(f"  R      = {R_mean:.4f} ± {sigma_R:.5f}  (tau_int={tau_int:.2f})")
    print(f"  plaq   = {plaq_mean:.4f} ± {plaq_std:.4f}")
    print(f"  eps_link={eps_link:.4f}  eps_matter={eps_matter:.4f}")
    print(f"  total runtime = {(time.time()-t_start)/60:.1f} min")

    # ── Four-point FSS fit ────────────────────────────────────────────────
    L_fss     = [4, 6, 8, L]
    R_fss     = [SU2_KNOWN[4][0], SU2_KNOWN[6][0], SU2_KNOWN[8][0], R_mean]
    sigma_fss = [SU2_KNOWN[4][1], SU2_KNOWN[6][1], SU2_KNOWN[8][1], sigma_R]

    fss = fss_fit(L_fss, R_fss, sigma_fss)

    print(f"\nFSS fits (L={L_fss[0]},{L_fss[1]},{L_fss[2]},{L_fss[3]}):")
    for name, r in fss["ansatze"].items():
        chi2dof_str = f"{r['chi2dof']:.2f}" if r["chi2dof"] is not None else "n/a"
        print(f"  {name:12s}  R_inf={r['R_inf']}  χ²/dof={chi2dof_str}")
    print(f"\nBest ({fss.get('best_ansatz','?')}):")
    print(f"  R_inf(SU2) = {fss['R_inf_best']} "
          f"± {fss['sigma_stat']}(stat) ± {fss['sigma_syst']}(syst)")
    print(f"  combined   = {fss['R_inf_best']} ± {fss['sigma_combined']}")

    ratio = R_INF_SU3 / fss["R_inf_best"] if fss["R_inf_best"] else None
    print(f"\nR_inf(SU3)/R_inf(SU2) = {ratio:.4f}  "
          f"({'SU3>SU2' if ratio and ratio>1 else 'SU3<SU2 — ordering inverts at L=inf!'} at L→∞)")

    # ── Write outputs ─────────────────────────────────────────────────────
    runlog = {
        "meta": {
            "script": "su2_l10_colab.py",
            "date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "L": L, "beta": BETA, "kappa": KAPPA, "seed": SEED,
            "N_therm": N_THERM, "N_meas": N_MEAS, "N_skip": N_SKIP,
            "eps_link_final": round(eps_link, 5),
            "eps_matter_final": round(eps_matter, 5),
            "runtime_s": round(time.time() - t_start, 1),
        },
        "R_series":    [round(r, 6) for r in R_series],
        "plaq_series": [round(p, 6) for p in plaq_series],
    }

    summary = {
        "L": L, "beta": BETA, "kappa": KAPPA, "seed": SEED,
        "N_meas": N_MEAS, "N_skip": N_SKIP,
        "R_mean":    round(R_mean, 6),
        "R_naive":   round(R_naive, 6),
        "tau_int":   round(tau_int, 4),
        "sigma_R":   round(sigma_R, 6),
        "plaq_mean": round(plaq_mean, 6),
        "known_su2": {str(k): list(v) for k, v in SU2_KNOWN.items()},
        "fss_L":     L_fss,
        "fss_R":     [round(r, 6) for r in R_fss],
        "fss_sigma": [round(s, 6) for s in sigma_fss],
        "fss":       fss,
        "R_inf_SU3": R_INF_SU3,
        "ratio_SU3_over_SU2": round(ratio, 5) if ratio else None,
        "ordering_holds_at_Linf": (ratio > 1.0) if ratio else None,
        "runtime_s": round(time.time() - t_start, 1),
        "stub_mode": False,
    }

    with open(OUTLOG, "w") as f:
        json.dump(runlog, f, indent=2)
    with open(OUTSUM, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {OUTLOG}")
    print(f"Saved: {OUTSUM}")
    print("\n── SUMMARY FOR main.tex UPDATE ──────────────────────────────────")
    print(f"Table 5 / SU(2) FSS — add row:")
    print(f"  L=10  R={R_mean:.4f}  σ={sigma_R:.5f}  τ_int={tau_int:.2f}")
    print(f"Table 6 — update L=∞ SU(2) row:")
    R_inf = fss["R_inf_best"]
    R_inf_err = fss["sigma_combined"]
    print(f"  R_inf(SU2) = {R_inf} ± {R_inf_err}")
    print(f"  ratio at L=∞: {ratio:.3f}  ({'ordering holds' if ratio and ratio>1 else '⚠ ordering inverts'})")
    print(f"\nSu next step: run  python update_tex_after_su2l10.py  to patch main.tex")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_su2_l10()
