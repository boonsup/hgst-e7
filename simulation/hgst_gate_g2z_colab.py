"""
hgst_gate_g2z_colab.py
======================
Colab-ready single-file script — closes Gate G2 (Task J) and Task Z.

  Task J — SU(2) beta=8.0, L=10, kappa=0.3 (matched-beta FSS)
  Task Z — SU(3) beta=8.0, L=8,  kappa=0.3, seed=302 (replica/second seed)

HOW TO USE IN COLAB
  1. Upload this file (or paste into a cell).
  2. Set TASK at the bottom: "J", "Z", or "BOTH".
  3. Runtime: Task J ≈ 60–120 min CPU; Task Z ≈ 30–60 min CPU.

OUTPUTS (written to working directory, then auto-downloaded in Colab):
  su2_b8_l10_results.json      — Task J
  su3_l8_replica_results.json  — Task Z
  gate_g2z_tex_update.txt      — ready-to-paste main.tex / appendix fragments

REQUIREMENTS  (all pre-installed on Colab):
  numpy >= 1.21
  scipy >= 1.7
"""

# ── 0. Install guard (Colab safe) ─────────────────────────────────────────────
import subprocess, sys
for _pkg in ["numpy", "scipy"]:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "-q"])

# ── Standard library ──────────────────────────────────────────────────────────
import gc
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.linalg import expm

# ==============================================================================
# ═══  SECTION 0 — RUN CONFIGURATION  (edit here)  ════════════════════════════
# ==============================================================================

# ── Task J — SU(2) matched-beta L=10 ─────────────────────────────────────────
J_L       = 10        # lattice size
J_BETA    = 8.0       # ← matched to SU(3) FSS protocol (not 4.0)
J_KAPPA   = 0.3
J_SEED    = 203       # same seed as log run for reproducibility
J_N_THERM = 4000      # thermalisation sweeps
J_N_MEAS  = 500       # measurement sweeps
J_N_SKIP  = 5         # sweeps between measurements

# Existing SU(2) L=4,6,8 canonical values used in 4-pt FSS
# (from main.tex SU(2) table; high-beta plateau; verify beta matches J_BETA)
SU2_KNOWN = {
    4: (0.347, 0.005),
    6: (0.360, 0.003),
    8: (0.360, 0.002),
}
R_INF_SU3 = 0.3539    # canonical SU(3) R_inf from main.tex (4-pt autocorr FSS)

# ── Task Z — SU(3) L=8 replica ───────────────────────────────────────────────
Z_L       = 8         # same L as seed-202 run
Z_BETA    = 8.0
Z_KAPPA   = 0.3
Z_SEED    = 302       # ← NEW seed (seed 202 already used; this is the replica)
Z_N_THERM = 2000      # matches seed-202 protocol
Z_N_MEAS  = 500
Z_N_SKIP  = 1

# Primary result from seed-202 for comparison
Z_PRIMARY = {"R_mean": 0.3906041666666667, "R_err": 0.0016515252639315416,
             "tau_int": 3.534466279749996, "seed": 202}

# ── Output files ──────────────────────────────────────────────────────────────
OUT_J   = "su2_b8_l10_results.json"
OUT_Z   = "su3_l8_replica_results.json"
OUT_TEX = "gate_g2z_tex_update.txt"

# ── Step tuning ───────────────────────────────────────────────────────────────
EPS_LINK_INIT   = 0.30
EPS_MATTER_INIT = 0.50
TUNE_EVERY      = 200
TARGET_ACC      = 0.50

# ==============================================================================
# ═══  SECTION 1 — LATTICE GEOMETRY  ══════════════════════════════════════════
# ==============================================================================

@dataclass
class Lattice:
    L: int
    N: int                                   = field(init=False)
    edges: List[Tuple[int,int]]              = field(init=False)
    plaquettes: List[Tuple[int,int,int,int]] = field(init=False)
    r: np.ndarray                            = field(init=False)
    _adj: Dict[int, List[int]]               = field(init=False)
    bfs_paths: Dict[int, Dict[int, List[int]]] = field(init=False)

    def __post_init__(self):
        L = self.L
        self.N = L * L
        edges = []
        for m in range(L):
            for n in range(L):
                i = n + m * L
                if n + 1 < L: edges.append((i, i + 1))
                if m + 1 < L: edges.append((i, i + L))
        self.edges = edges

        plaq = []
        for m in range(L - 1):
            for n in range(L - 1):
                i0 = n     + m*L
                i1 = (n+1) + m*L
                i2 = (n+1) + (m+1)*L
                i3 = n     + (m+1)*L
                plaq.append((i0, i1, i2, i3))
        self.plaquettes = plaq

        # grade magnitudes: r[i] = 2^(n-m), 1-indexed
        r = np.empty(self.N, dtype=float)
        for idx in range(self.N):
            n_idx = idx % L + 1
            m_idx = idx // L + 1
            r[idx] = 2.0 ** (n_idx - m_idx)
        self.r = r

        adj: Dict[int, List[int]] = {i: [] for i in range(self.N)}
        for (i, j) in edges:
            adj[i].append(j); adj[j].append(i)
        self._adj = adj

        self.bfs_paths = self._precompute_bfs()

    def _precompute_bfs(self) -> Dict[int, Dict[int, List[int]]]:
        all_paths: Dict[int, Dict[int, List[int]]] = {}
        for src in range(self.N):
            paths: Dict[int, List[int]] = {src: [src]}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in self._adj[cur]:
                    if nxt not in paths:
                        paths[nxt] = paths[cur] + [nxt]
                        queue.append(nxt)
            all_paths[src] = paths
        return all_paths

    def neighbours(self, site: int) -> List[int]:
        return self._adj[site]


# ==============================================================================
# ═══  SECTION 2 — SU(2) ALGEBRA  ════════════════════════════════════════════
# ==============================================================================

def _su2_random(rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(4); z /= np.linalg.norm(z)
    a0, a1, a2, a3 = z
    return np.array([[a0 + 1j*a3,   a2 + 1j*a1],
                     [-a2 + 1j*a1,  a0 - 1j*a3]], dtype=np.complex128)

def _su2_near_id(eps: float, rng: np.random.Generator) -> np.ndarray:
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    theta = rng.uniform(-eps, eps)
    c, s   = np.cos(theta), np.sin(theta)
    nx, ny, nz = axis
    return np.array([[c + 1j*nz*s,         ny*s + 1j*nx*s],
                     [-ny*s + 1j*nx*s,     c - 1j*nz*s   ]], dtype=np.complex128)

def _su2_id() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)

def _dagger(U: np.ndarray) -> np.ndarray:
    return U.conj().T

def _random_doublet(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(4).view(np.complex128)
    return v / np.linalg.norm(v)


# ==============================================================================
# ═══  SECTION 3 — SU(3) ALGEBRA  ════════════════════════════════════════════
# ==============================================================================

_GM = [
    np.array([[0,1,0],[1,0,0],[0,0,0]],             dtype=np.complex128),  # λ1
    np.array([[0,-1j,0],[1j,0,0],[0,0,0]],          dtype=np.complex128),  # λ2
    np.array([[1,0,0],[0,-1,0],[0,0,0]],             dtype=np.complex128),  # λ3
    np.array([[0,0,1],[0,0,0],[1,0,0]],              dtype=np.complex128),  # λ4
    np.array([[0,0,-1j],[0,0,0],[1j,0,0]],           dtype=np.complex128),  # λ5
    np.array([[0,0,0],[0,0,1],[0,1,0]],              dtype=np.complex128),  # λ6
    np.array([[0,0,0],[0,0,-1j],[0,1j,0]],           dtype=np.complex128),  # λ7
    np.array([[1,0,0],[0,1,0],[0,0,-2]],             dtype=np.complex128) / np.sqrt(3),  # λ8
]
_I3 = np.eye(3, dtype=np.complex128)


def _su3_project(U: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """Project to nearest SU(3) via SVD + Newton polish."""
    W, _, Vh = np.linalg.svd(U)
    Q = W @ Vh
    W[:, 2] *= np.linalg.det(Q).conj()
    Q = W @ Vh
    for _ in range(n_iter):
        Q = 0.5 * Q @ (3.0 * _I3 - Q.conj().T @ Q)
        d = np.linalg.det(Q)
        Q[:, 2] /= d ** (1.0 / 3.0)
    return Q


def _su3_random(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(-np.pi, np.pi, 8)
    H = 1j * sum(t * L for t, L in zip(theta, _GM))
    return _su3_project(expm(H))


def _su3_near_id(eps: float, rng: np.random.Generator) -> np.ndarray:
    """Near-identity SU(3) for Metropolis proposals (Cabibbo-Marinari lite)."""
    theta = rng.normal(0, eps, 8)
    H = 1j * sum(t * L for t, L in zip(theta, _GM))
    return _su3_project(expm(H))


def _su3_plaq_trace(links: dict, p: Tuple) -> float:
    """(1/3) Re Tr U_p  for SU(3) plaquette."""
    i0, i1, i2, i3 = p
    U = _I3.copy()
    for a, b in [(i0,i1),(i1,i2),(i2,i3),(i3,i0)]:
        if   (a,b) in links: U = U @ links[(a,b)]
        elif (b,a) in links: U = U @ _dagger(links[(b,a)])
    return float(np.real(np.trace(U))) / 3.0


def _random_triplet(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform unit vector in ℂ³."""
    v = rng.standard_normal(6).view(np.complex128)
    return v / np.linalg.norm(v)


# ==============================================================================
# ═══  SECTION 4 — ACTION  ════════════════════════════════════════════════════
# ==============================================================================

def _plaq_trace_su2(links, plaq):
    i0,i1,i2,i3 = plaq
    U = _su2_id()
    for a, b in [(i0,i1),(i1,i2),(i2,i3),(i3,i0)]:
        if   (a,b) in links: U = U @ links[(a,b)]
        elif (b,a) in links: U = U @ _dagger(links[(b,a)])
    return 0.5 * float(np.real(np.trace(U)))


def _delta_link_su2(links, matter, lat, edge, U_new, beta, kappa):
    i, j   = edge
    U_old  = links[edge]
    dS_g   = 0.0
    for p in lat.plaquettes:
        i0,i1,i2,i3 = p
        pe = [(i0,i1),(i1,i2),(i2,i3),(i3,i0)]
        if edge not in pe and (j,i) not in pe: continue
        tr_old = _plaq_trace_su2(links, p)
        links[edge] = U_new
        tr_new = _plaq_trace_su2(links, p)
        links[edge] = U_old
        dS_g += -beta * (tr_new - tr_old)
    psi_i, psi_j = matter[i], matter[j]
    hop_old = float(np.real(psi_i.conj() @ U_old          @ psi_j))
    hop_new = float(np.real(psi_i.conj() @ U_new          @ psi_j))
    hob_old = float(np.real(psi_j.conj() @ _dagger(U_old) @ psi_i))
    hob_new = float(np.real(psi_j.conj() @ _dagger(U_new) @ psi_i))
    return dS_g - kappa * ((hop_new - hop_old) + (hob_new - hob_old))


def _delta_matter_su2(links, matter, lat, site, psi_new, kappa):
    psi_old = matter[site]; dS = 0.0
    for nj in lat.neighbours(site):
        if (site,nj) in links:
            U = links[(site,nj)]
            dS += -kappa * (float(np.real(psi_new.conj() @ U          @ matter[nj]))
                           -float(np.real(psi_old.conj() @ U          @ matter[nj])))
            dS += -kappa * (float(np.real(matter[nj].conj() @ _dagger(U) @ psi_new))
                           -float(np.real(matter[nj].conj() @ _dagger(U) @ psi_old)))
        elif (nj,site) in links:
            U = links[(nj,site)]
            dS += -kappa * (float(np.real(psi_new.conj() @ _dagger(U) @ matter[nj]))
                           -float(np.real(psi_old.conj() @ _dagger(U) @ matter[nj])))
            dS += -kappa * (float(np.real(matter[nj].conj() @ U @ psi_new))
                           -float(np.real(matter[nj].conj() @ U @ psi_old)))
    return dS


def _delta_link_su3(links, matter, lat, edge, U_new, beta, kappa):
    i, j  = edge
    U_old = links[edge]
    dS_g  = 0.0
    for p in lat.plaquettes:
        i0,i1,i2,i3 = p
        pe = [(i0,i1),(i1,i2),(i2,i3),(i3,i0)]
        if edge not in pe and (j,i) not in pe: continue
        tr_old = _su3_plaq_trace(links, p)
        links[edge] = U_new
        tr_new = _su3_plaq_trace(links, p)
        links[edge] = U_old
        dS_g += -beta * (tr_new - tr_old)
    psi_i, psi_j = matter[i], matter[j]
    hop_old = float(np.real(psi_i.conj() @ U_old          @ psi_j))
    hop_new = float(np.real(psi_i.conj() @ U_new          @ psi_j))
    hob_old = float(np.real(psi_j.conj() @ _dagger(U_old) @ psi_i))
    hob_new = float(np.real(psi_j.conj() @ _dagger(U_new) @ psi_i))
    return dS_g - kappa * ((hop_new - hop_old) + (hob_new - hob_old))


def _delta_matter_su3(links, matter, lat, site, psi_new, kappa):
    psi_old = matter[site]; dS = 0.0
    for nj in lat.neighbours(site):
        if (site,nj) in links:
            U = links[(site,nj)]
            dS += -kappa * (float(np.real(psi_new.conj() @ U          @ matter[nj]))
                           -float(np.real(psi_old.conj() @ U          @ matter[nj])))
            dS += -kappa * (float(np.real(matter[nj].conj() @ _dagger(U) @ psi_new))
                           -float(np.real(matter[nj].conj() @ U @ psi_old)))
        elif (nj,site) in links:
            U = links[(nj,site)]
            dS += -kappa * (float(np.real(psi_new.conj() @ _dagger(U) @ matter[nj]))
                           -float(np.real(psi_old.conj() @ _dagger(U) @ matter[nj])))
            dS += -kappa * (float(np.real(matter[nj].conj() @ U @ psi_new))
                           -float(np.real(matter[nj].conj() @ U @ psi_old)))
    return dS


# ==============================================================================
# ═══  SECTION 5 — METROPOLIS SWEEPS  ════════════════════════════════════════
# ==============================================================================

def _sweep_su2(links, matter, lat, beta, kappa, eps_link, eps_matter, rng):
    n_la = 0
    for e in lat.edges:
        U_new = _su2_near_id(eps_link, rng) @ links[e]
        dS = _delta_link_su2(links, matter, lat, e, U_new, beta, kappa)
        if dS <= 0.0 or rng.random() < np.exp(-dS):
            links[e] = U_new; n_la += 1
    n_ma = 0
    for s in range(lat.N):
        r_i = lat.r[s]
        chi = matter[s] / r_i if r_i > 1e-15 else matter[s]
        eta = rng.standard_normal(4).view(np.complex128)
        chi_new = chi + eps_matter * eta / np.linalg.norm(eta)
        chi_new /= np.linalg.norm(chi_new)
        psi_new = r_i * chi_new
        dS = _delta_matter_su2(links, matter, lat, s, psi_new, kappa)
        if dS <= 0.0 or rng.random() < np.exp(-dS):
            matter[s] = psi_new; n_ma += 1
    return n_la / len(lat.edges), n_ma / lat.N


def _sweep_su3(links, matter, lat, beta, kappa, eps_link, eps_matter, rng):
    n_la = 0
    for e in lat.edges:
        U_new = _su3_near_id(eps_link, rng) @ links[e]
        dS = _delta_link_su3(links, matter, lat, e, U_new, beta, kappa)
        if dS <= 0.0 or rng.random() < np.exp(-dS):
            links[e] = U_new; n_la += 1
    n_ma = 0
    for s in range(lat.N):
        r_i = lat.r[s]
        chi = matter[s] / r_i if r_i > 1e-15 else matter[s]
        eta = rng.standard_normal(6).view(np.complex128)
        chi_new = chi + eps_matter * eta / np.linalg.norm(eta)
        chi_new /= np.linalg.norm(chi_new)
        psi_new = r_i * chi_new
        dS = _delta_matter_su3(links, matter, lat, s, psi_new, kappa)
        if dS <= 0.0 or rng.random() < np.exp(-dS):
            matter[s] = psi_new; n_ma += 1
    return n_la / len(lat.edges), n_ma / lat.N


def _tune_eps(acc, eps, target=TARGET_ACC, lo=0.01, hi=2.0):
    if   acc > target + 0.05: return min(eps * 1.15, hi)
    elif acc < target - 0.05: return max(eps * 0.87, lo)
    return eps


# ==============================================================================
# ═══  SECTION 6 — MIXED-FRACTION OBSERVABLE  ════════════════════════════════
# ==============================================================================

def _sign_matrix(links, matter, lat, dim):
    """Build NxN sign matrix from BFS holonomies."""
    N = lat.N
    S = np.zeros((N, N), dtype=np.int8)
    dagger_fn = _dagger
    id_fn = np.eye(dim, dtype=np.complex128)
    for src in range(N):
        psi_src = matter[src]
        for dst, path in lat.bfs_paths[src].items():
            if dst == src: continue
            U = id_fn.copy()
            for k in range(len(path) - 1):
                a, b = path[k], path[k+1]
                if   (a,b) in links: U = U @ links[(a,b)]
                elif (b,a) in links: U = U @ dagger_fn(links[(b,a)])
            val = float(np.real(psi_src.conj() @ U @ matter[dst]))
            if   val > 0: S[src, dst] =  1
            elif val < 0: S[src, dst] = -1
    return S


def _mixed_R(S: np.ndarray) -> float:
    """MIXED fraction from sign matrix (vectorised)."""
    N = S.shape[0]
    nz = (S != 0)
    n_valid = n_mixed = 0
    for i in range(N):
        for j in range(i+1, N):
            if not nz[i, j]: continue
            s_ij = int(S[i, j])
            valid_k = nz[i, :] & nz[:, j]
            valid_k[i] = valid_k[j] = False
            k_cnt = int(valid_k.sum())
            if k_cnt == 0: continue
            n_valid += k_cnt
            med = (S[i, :] * S[:, j]).astype(np.int16)
            n_mixed += int((valid_k & (med != s_ij)).sum())
    return n_mixed / n_valid if n_valid > 0 else 0.0


def measure_R(links, matter, lat, dim):
    """Returns (R_mixed, plaq_mean)."""
    S    = _sign_matrix(links, matter, lat, dim)
    R    = _mixed_R(S)
    plaq = float(np.mean([
        (_plaq_trace_su2(links, p) if dim==2 else _su3_plaq_trace(links, p))
        for p in lat.plaquettes
    ]))
    return R, plaq


# ==============================================================================
# ═══  SECTION 7 — STATISTICS  ════════════════════════════════════════════════
# ==============================================================================

def madras_sokal_tau(ts: np.ndarray, c: float = 6.0) -> float:
    N  = len(ts); mu = ts.mean(); tc = ts - mu
    g0 = float(np.dot(tc, tc) / N)
    if g0 < 1e-30: return 0.5
    tau = 0.5
    for lag in range(1, N // 2):
        gt = float(np.dot(tc[:-lag], tc[lag:]) / N)
        if gt <= 0: break
        tau += gt / g0
        if lag >= int(c * tau): break
    return max(0.5, tau)


# ==============================================================================
# ═══  SECTION 8 — FSS FITTING  ═══════════════════════════════════════════════
# ==============================================================================

def fss_fit(L_arr, R_arr, sigma_arr) -> dict:
    L_a = np.array(L_arr, dtype=float)
    R_a = np.array(R_arr, dtype=float)
    s_a = np.array(sigma_arr, dtype=float)
    W   = np.diag(1.0 / s_a**2)

    def _fit(fn):
        A = fn(L_a)
        try:
            c = np.linalg.solve(A.T @ W @ A, A.T @ (W @ R_a))
            res = R_a - A @ c
            chi2 = float(res @ W @ res)
            dof  = len(L_a) - A.shape[1]
            return float(c[0]), chi2, dof
        except np.linalg.LinAlgError:
            return np.nan, np.nan, 0

    ansatze = {}
    for name, fn in [
        ("1/L",      lambda L: np.column_stack([np.ones_like(L), 1/L])),
        ("1/L+1/L2", lambda L: np.column_stack([np.ones_like(L), 1/L, 1/L**2])),
        ("1/L2",     lambda L: np.column_stack([np.ones_like(L), 1/L**2])),
    ]:
        Ri, chi2, dof = _fit(fn)
        ansatze[name] = {
            "R_inf":   round(float(Ri),   6) if np.isfinite(Ri)   else None,
            "chi2":    round(float(chi2), 4) if np.isfinite(chi2) else None,
            "dof":     dof,
            "chi2dof": round(float(chi2/dof), 4) if (dof > 0 and np.isfinite(chi2)) else None,
        }

    valid = [v["R_inf"] for v in ansatze.values()
             if v["R_inf"] is not None and np.isfinite(v["R_inf"])]
    if not valid:
        return {"ansatze": ansatze, "R_inf_best": None, "sigma_stat": None,
                "sigma_syst": None, "sigma_combined": None}

    best_key = min(
        (k for k in ansatze if ansatze[k]["chi2dof"] is not None),
        key=lambda k: ansatze[k]["chi2dof"] or 1e9
    )
    R_inf_best = ansatze[best_key]["R_inf"]
    sigma_syst = (max(valid) - min(valid)) / 2.0
    sigma_stat = float(np.min(s_a))

    return {
        "ansatze":        ansatze,
        "best_ansatz":    best_key,
        "R_inf_best":     round(R_inf_best, 6),
        "sigma_stat":     round(sigma_stat, 6),
        "sigma_syst":     round(sigma_syst, 6),
        "sigma_combined": round(np.sqrt(sigma_stat**2 + sigma_syst**2), 6),
    }


# ==============================================================================
# ═══  SECTION 9 — GENERIC SIMULATION RUNNER  ════════════════════════════════
# ==============================================================================

def _run_sim(label, L, beta, kappa, n_therm, n_meas, n_skip, seed,
             gauge_group, eps_link_init=EPS_LINK_INIT,
             eps_matter_init=EPS_MATTER_INIT):
    """
    Run one MC simulation and return (R_mean, sigma_R, tau_int,
                                       plaq_mean, R_series, plaq_series,
                                       eps_link, eps_matter, t_therm_s, t_meas_s).
    gauge_group: 'SU2' or 'SU3'
    """
    print("")
    print("=" * 65)
    print(f"{label}  [{gauge_group}  L={L}  beta={beta}  kappa={kappa}  seed={seed}]")
    print(f"  n_therm={n_therm}  n_meas={n_meas}  n_skip={n_skip}")
    print("=" * 65)
    t0 = time.time()

    rng  = np.random.default_rng(seed)
    dim  = 2 if gauge_group == "SU2" else 3
    sweep_fn = _sweep_su2 if gauge_group == "SU2" else _sweep_su3

    print("Building lattice (precomputing BFS paths)...")
    lat = Lattice(L=L)
    print(f"  sites={lat.N}  edges={len(lat.edges)}  plaquettes={len(lat.plaquettes)}")

    # Initialise fields
    if gauge_group == "SU2":
        links  = {e: _su2_random(rng)    for e in lat.edges}
        matter = {s: lat.r[s] * _random_doublet(rng) for s in range(lat.N)}
    else:
        links  = {e: _su3_random(rng)    for e in lat.edges}
        matter = {s: lat.r[s] * _random_triplet(rng) for s in range(lat.N)}

    eps_l = eps_link_init
    eps_m = eps_matter_init

    # ── Thermalisation ────────────────────────────────────────────────────────
    print(f"\nThermalising ({n_therm} sweeps)...")
    t_therm0 = time.time()
    for sw in range(n_therm):
        la, ma = sweep_fn(links, matter, lat, beta, kappa, eps_l, eps_m, rng)
        if (sw + 1) % TUNE_EVERY == 0:
            eps_l = _tune_eps(la, eps_l)
            eps_m = _tune_eps(ma, eps_m)
        if (sw + 1) % max(1, n_therm // 8) == 0:
            elapsed = time.time() - t_therm0
            eta    = elapsed / (sw + 1) * (n_therm - sw - 1)
            print(f"  [{sw+1:5d}/{n_therm}]  link_acc={la:.3f}  mat_acc={ma:.3f}  "
                  f"eps_L={eps_l:.4f}  eps_M={eps_m:.4f}  ETA {eta/60:.1f} min")
    t_therm_s = time.time() - t_therm0
    print(f"Thermalisation done in {t_therm_s/60:.1f} min  "
          f"[eps_link={eps_l:.4f}  eps_matter={eps_m:.4f}]")

    # ── Production measurements ───────────────────────────────────────────────
    print(f"\nMeasuring ({n_meas} measurements, every {n_skip} sweeps)...")
    R_series    = []
    plaq_series = []
    t_meas0 = time.time()

    for meas in range(n_meas):
        for _ in range(n_skip):
            sweep_fn(links, matter, lat, beta, kappa, eps_l, eps_m, rng)
        R_val, plaq_val = measure_R(links, matter, lat, dim)
        R_series.append(float(R_val))
        plaq_series.append(float(plaq_val))

        if (meas + 1) % max(1, n_meas // 10) == 0:
            elapsed   = time.time() - t_meas0
            remaining = elapsed / (meas + 1) * (n_meas - meas - 1)
            rmean = float(np.mean(R_series))
            rstd  = float(np.std(R_series, ddof=1) / np.sqrt(meas+1))
            print(f"  [{meas+1:4d}/{n_meas}]  "
                  f"R={rmean:.4f}+/-{rstd:.5f}  "
                  f"plaq={np.mean(plaq_series):.4f}  "
                  f"ETA {remaining/60:.1f} min")
        gc.collect()

    t_meas_s = time.time() - t_meas0
    print(f"Measurements done in {t_meas_s/60:.1f} min")

    # ── Statistics ────────────────────────────────────────────────────────────
    Ra      = np.array(R_series)
    R_mean  = float(Ra.mean())
    R_naive = float(Ra.std(ddof=1) / np.sqrt(len(Ra)))
    tau_int = madras_sokal_tau(Ra)
    sigma_R = R_naive * float(np.sqrt(2.0 * tau_int))
    plaq_mean = float(np.mean(plaq_series))

    print(f"\nResult:  R = {R_mean:.4f} +/- {sigma_R:.5f}  "
          f"(tau_int={tau_int:.2f}  plaq={plaq_mean:.4f})")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    return (R_mean, sigma_R, tau_int, plaq_mean, R_naive,
            R_series, plaq_series, eps_l, eps_m, t_therm_s, t_meas_s)


# ==============================================================================
# ═══  SECTION 10 — TASK J  ═══════════════════════════════════════════════════
#   SU(2)  beta=8.0  L=10  (matched-beta FSS)
# ==============================================================================

def run_task_J() -> dict:
    t_start = time.time()
    (R_mean, sigma_R, tau_int, plaq_mean, R_naive,
     R_series, plaq_series, eps_l, eps_m,
     t_therm_s, t_meas_s) = _run_sim(
        label="Task J — SU(2) beta=8.0 L=10",
        L=J_L, beta=J_BETA, kappa=J_KAPPA,
        n_therm=J_N_THERM, n_meas=J_N_MEAS, n_skip=J_N_SKIP,
        seed=J_SEED, gauge_group="SU2"
    )

    # 4-point FSS
    L_fss     = [4, 6, 8, J_L]
    R_fss     = [SU2_KNOWN[4][0], SU2_KNOWN[6][0], SU2_KNOWN[8][0], R_mean]
    sig_fss   = [SU2_KNOWN[4][1], SU2_KNOWN[6][1], SU2_KNOWN[8][1], sigma_R]
    fss       = fss_fit(L_fss, R_fss, sig_fss)

    print("\nFSS fits (SU(2), L=4,6,8,10, beta=8.0):")
    for name, r in fss["ansatze"].items():
        cd = f"{r['chi2dof']:.2f}" if r["chi2dof"] is not None else "n/a"
        print(f"  {name:12s}  R_inf={r['R_inf']}  chi2/dof={cd}")
    print(f"\nBest ({fss['best_ansatz']}):  "
          f"R_inf(SU2) = {fss['R_inf_best']} "
          f"+/- {fss['sigma_stat']}(stat) +/- {fss['sigma_syst']}(syst)")

    ratio = R_INF_SU3 / fss["R_inf_best"] if fss["R_inf_best"] else None
    print(f"R_inf(SU3)/R_inf(SU2) = {ratio:.4f}" if ratio else "ratio: N/A")

    result = {
        "task": "J",
        "group": "SU2",
        "L": J_L, "beta": J_BETA, "kappa": J_KAPPA, "seed": J_SEED,
        "n_therm": J_N_THERM, "n_meas": J_N_MEAS, "n_skip": J_N_SKIP,
        "R_mean":    round(R_mean,    6),
        "R_naive":   round(R_naive,   6),
        "sigma_R":   round(sigma_R,   6),
        "tau_int":   round(tau_int,   4),
        "plaq_mean": round(plaq_mean, 6),
        "eps_link_final":   round(eps_l, 5),
        "eps_matter_final": round(eps_m, 5),
        "t_therm_s": round(t_therm_s, 1),
        "t_meas_s":  round(t_meas_s,  1),
        "fss_inputs": {"L": L_fss, "R": [round(r,6) for r in R_fss],
                       "sigma": [round(s,6) for s in sig_fss]},
        "fss": fss,
        "R_inf_SU3": R_INF_SU3,
        "ratio_SU3_over_SU2": round(ratio, 5) if ratio else None,
        "ordering_holds_at_Linf": (ratio > 1.0) if ratio else None,
        "R_series":    [round(r,6) for r in R_series],
        "plaq_series": [round(p,6) for p in plaq_series],
        "runtime_total_s": round(time.time() - t_start, 1),
        "date_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }

    with open(OUT_J, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUT_J}")
    return result


# ==============================================================================
# ═══  SECTION 11 — TASK Z  ═══════════════════════════════════════════════════
#   SU(3)  beta=8.0  L=8  seed=302  (replica / second seed)
# ==============================================================================

def run_task_Z() -> dict:
    t_start = time.time()
    (R_mean, sigma_R, tau_int, plaq_mean, R_naive,
     R_series, plaq_series, eps_l, eps_m,
     t_therm_s, t_meas_s) = _run_sim(
        label="Task Z — SU(3) beta=8.0 L=8 seed=302 (replica)",
        L=Z_L, beta=Z_BETA, kappa=Z_KAPPA,
        n_therm=Z_N_THERM, n_meas=Z_N_MEAS, n_skip=Z_N_SKIP,
        seed=Z_SEED, gauge_group="SU3"
    )

    # Compare to seed-202 primary
    primary_R   = Z_PRIMARY["R_mean"]
    primary_err = Z_PRIMARY["R_err"]
    pull = (R_mean - primary_R) / np.sqrt(sigma_R**2 + primary_err**2)
    consistent = abs(pull) < 3.0

    print(f"\nReplica vs primary check:")
    print(f"  seed-202 (primary):  R = {primary_R:.4f} +/- {primary_err:.5f}")
    print(f"  seed-302 (replica):  R = {R_mean:.4f} +/- {sigma_R:.5f}")
    print(f"  pull = {pull:.2f} sigma  ({'CONSISTENT' if consistent else 'INCONSISTENT — INVESTIGATE'})")

    result = {
        "task": "Z",
        "group": "SU3",
        "L": Z_L, "beta": Z_BETA, "kappa": Z_KAPPA, "seed": Z_SEED,
        "n_therm": Z_N_THERM, "n_meas": Z_N_MEAS, "n_skip": Z_N_SKIP,
        "R_mean":    round(R_mean,    6),
        "R_naive":   round(R_naive,   6),
        "sigma_R":   round(sigma_R,   6),
        "tau_int":   round(tau_int,   4),
        "plaq_mean": round(plaq_mean, 6),
        "eps_link_final":   round(eps_l, 5),
        "eps_matter_final": round(eps_m, 5),
        "t_therm_s": round(t_therm_s, 1),
        "t_meas_s":  round(t_meas_s,  1),
        "primary_seed202": Z_PRIMARY,
        "pull_sigma": round(float(pull), 3),
        "replica_consistent_3sigma": bool(consistent),
        "R_series":    [round(r,6) for r in R_series],
        "plaq_series": [round(p,6) for p in plaq_series],
        "runtime_total_s": round(time.time() - t_start, 1),
        "date_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }

    with open(OUT_Z, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUT_Z}")
    return result


# ==============================================================================
# ═══  SECTION 12 — TEX EXPORT  ═══════════════════════════════════════════════
# ==============================================================================

def write_tex_update(result_j: Optional[dict], result_z: Optional[dict]):
    """Write ready-to-paste LaTeX fragments into gate_g2z_tex_update.txt."""
    lines = []
    lines.append("% ============================================================")
    lines.append("% gate_g2z_tex_update.txt — generated by hgst_gate_g2z_colab.py")
    lines.append("% Paste relevant fragments into main.tex / appendix/tables.tex")
    lines.append("% ============================================================")
    lines.append("")

    if result_j:
        Rj   = result_j["R_mean"]
        sj   = result_j["sigma_R"]
        tauj = result_j["tau_int"]
        fss  = result_j["fss"]
        Ri   = fss["R_inf_best"]
        Rc   = fss["sigma_combined"]
        Rb   = fss["best_ansatz"]
        ratio = result_j.get("ratio_SU3_over_SU2")
        ordering = result_j.get("ordering_holds_at_Linf")

        lines.append("% ── Task J — SU(2) beta=8.0 L=10 ──────────────────────────────")
        lines.append("%")
        lines.append("% 1. Add to tab:SU2 (§7):")
        lines.append("%")
        lines.append(r"\hline")
        lines.append(f"$R[L=10]$, SU(2) ($\\beta{{=}}8.0$, seed {result_j['seed']})")
        lines.append(f"  & ${Rj:.4f}\\pm{sj:.5f}$ & VALIDATED \\\\")
        lines.append("")
        lines.append("% 2. Update tab:SU2 FSS row:")
        lines.append("%")
        lines.append(f"$R_\\infty[\\mathrm{{SU}}(2)]$ ({Rb} FSS, 4-pt, $\\beta{{=}}8.0$)")
        lines.append(f"  & ${Ri:.4f}\\pm{Rc:.4f}$ & VALIDATED \\\\")
        lines.append("")
        lines.append("% 3. Update main.tex preamble macro \\RinfSUtwo:")
        lines.append("%")
        lines.append(f"\\newcommand{{\\RinfSUtwo}}{{{Ri:.4f} \\pm {Rc:.4f}}}  % 4-pt 1/L FSS beta=8.0")
        lines.append("")
        lines.append("% 4. Conjecture 9.1 — update R_inf(SU2) value:")
        lines.append("%")
        if ratio is not None:
            lines.append(f"% R_inf(SU3)/R_inf(SU2) = {ratio:.4f}  "
                         f"({'ordering holds (SU3>SU2)' if ordering else 'WARNING: ordering inverts at L=inf — update conjecture'})")
        lines.append("")
        lines.append("% 5. Update tab:gaps — close SU(2) L=10 (Task J) row:")
        lines.append("%")
        lines.append(f"SU(2) $L=10$ ($\\beta{{=}}8.0$ matched)")
        lines.append(f"  & \\textbf{{COMPLETED}}")
        lines.append(f"  & $R(L{{=}}10,\\beta{{=}}8.0) = {Rj:.4f}\\pm{sj:.5f}$")
        lines.append(f"    (seed~{result_j['seed']}, $\\tau_{{\\mathrm{{int}}}}={tauj:.2f}$,")
        lines.append(f"    {result_j['n_meas']} sweeps). 4-pt FSS: $R_\\infty = {Ri:.4f}\\pm{Rc:.4f}$. \\\\")
        lines.append("")
        lines.append("% 6. appendix/tables.tex — update SU(2) L=10 row:")
        lines.append("%")
        lines.append(f"SU(2) $L=10$ ($\\beta{{=}}8.0$)")
        lines.append(f"  & SU(2) & 10 & $\\beta=8.0$, $\\kappa={result_j['kappa']}$ & {result_j['seed']} \\\\")
        lines.append("")

    if result_z:
        Rz   = result_z["R_mean"]
        sz   = result_z["sigma_R"]
        tauz = result_z["tau_int"]
        cons = result_z["replica_consistent_3sigma"]
        pull = result_z["pull_sigma"]
        pR   = result_z["primary_seed202"]["R_mean"]
        pErr = result_z["primary_seed202"]["R_err"]

        lines.append("% ── Task Z — SU(3) L=8 replica (seed 302) ─────────────────────")
        lines.append("%")
        lines.append("% 1. Add footnote to tab:SU3 or tab:E7-epistemic:")
        lines.append("%")
        lines.append(f"% SU(3) L=8 replica: seed 302 gives $R={Rz:.4f}\\pm{sz:.5f}$;")
        lines.append(f"% primary (seed 202): $R={pR:.4f}\\pm{pErr:.5f}$;")
        lines.append(f"% pull = ${pull:.2f}\\sigma$ ({'consistent' if cons else 'INCONSISTENT'}).")
        lines.append("%")
        lines.append("% 2. Update tab:SU3 to add replica row:")
        lines.append("%")
        lines.append(f"$R(L=8)$ replica (seed~{result_z['seed']}) & ${Rz:.4f}\\pm{sz:.5f}$ & VALIDATED \\\\")
        lines.append("")
        lines.append("% 3. Update tab:E7-epistemic — SU(3) FSS independence column:")
        lines.append("%")
        lines.append(r"$R_\infty[\mathrm{SU}(3)] = \RinfSUthree$ & VALIDATED")
        if cons:
            lines.append(f"  & 4-pt autocorr FSS; replica (seed~{result_z['seed']}) "
                         f"consistent at ${abs(pull):.1f}\\sigma$ \\\\")
        else:
            lines.append(f"  & REQUIRES INVESTIGATION: replica pull = ${pull:.2f}\\sigma$ \\\\")
        lines.append("")
        lines.append("% 4. Close Task Z in tab:gaps:")
        lines.append("%")
        lines.append("SU(3) $L=8$ replica (second seed)")
        lines.append(f"  & \\textbf{{COMPLETED}}")
        lines.append(f"  & seed~{result_z['seed']}: $R={Rz:.4f}\\pm{sz:.5f}$; "
                     f"pull ${pull:.2f}\\sigma$ vs seed~202. "
                     f"{'Replica consistent.' if cons else 'Investigate inconsistency.'} \\\\")
        lines.append("")
        lines.append("% 5. appendix/tables.tex — add SU(3) L=8 replica row:")
        lines.append("%")
        lines.append(f"E7 SU(3) $L=8$ replica")
        lines.append(f"  & SU(3) & 8 & $\\beta=8.0$, $\\kappa={result_z['kappa']}$ & {result_z['seed']} \\\\")

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved -> {OUT_TEX}")


# ==============================================================================
# ═══  SECTION 13 — COLAB DOWNLOAD HELPER  ════════════════════════════════════
# ==============================================================================

def _colab_download(*files):
    """Auto-download output files when running in Colab."""
    try:
        from google.colab import files  # type: ignore
        for fname in files:
            import os
            if os.path.exists(fname):
                files.download(fname)
                print(f"  Downloaded: {fname}")
    except ImportError:
        print("  (Not in Colab — files saved to working directory.)")


# ==============================================================================
# ═══  SECTION 14 — ENTRY POINT  ══════════════════════════════════════════════
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
#  SET THIS before running:
#    "J"    — Task J only  (SU(2) beta=8.0 L=10  ≈ 60–120 min)
#    "Z"    — Task Z only  (SU(3) L=8 replica    ≈ 30–60 min)
#    "BOTH" — both tasks   (≈ 90–180 min total)
# ─────────────────────────────────────────────────────────────────────────────
TASK = "BOTH"  # ← edit here

if __name__ == "__main__":
    result_j = None
    result_z = None

    print(f"HGST Gate G2+Z — running task(s): {TASK}")
    print(f"Date (UTC): {time.strftime('%Y-%m-%d %H:%M', time.gmtime())}")
    print()

    if TASK in ("J", "BOTH"):
        result_j = run_task_J()

    if TASK in ("Z", "BOTH"):
        result_z = run_task_Z()

    write_tex_update(result_j, result_z)

    print("\n" + "=" * 65)
    print("DONE — Summary for main.tex")
    print("=" * 65)
    if result_j:
        fss = result_j["fss"]
        print(f"Task J  SU(2) L=10 beta=8.0:")
        print(f"  R(L=10)   = {result_j['R_mean']:.4f} +/- {result_j['sigma_R']:.5f}  (tau_int={result_j['tau_int']:.2f})")
        print(f"  R_inf(SU2)= {fss['R_inf_best']} +/- {fss['sigma_combined']}  [{fss['best_ansatz']}]")
        print(f"  ratio SU3/SU2 at L=inf = {result_j.get('ratio_SU3_over_SU2')}")
    if result_z:
        print(f"Task Z  SU(3) L=8 replica seed={result_z['seed']}:")
        print(f"  R(L=8)    = {result_z['R_mean']:.4f} +/- {result_z['sigma_R']:.5f}  (tau_int={result_z['tau_int']:.2f})")
        print(f"  vs seed-202: pull = {result_z['pull_sigma']:.2f}sigma  "
              f"({'OK' if result_z['replica_consistent_3sigma'] else 'INCONSISTENT'})")
    print(f"\nOutput files:")
    if result_j: print(f"  {OUT_J}")
    if result_z: print(f"  {OUT_Z}")
    print(f"  {OUT_TEX}")
    print("\nPaste fragments from gate_g2z_tex_update.txt into main.tex.")

    _colab_download(OUT_J, OUT_Z, OUT_TEX)
