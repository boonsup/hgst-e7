"""
observables.py — Gauge-Invariant Observables
=============================================
Layer 5 Measurement Module

Computes all observables used to test HGST extensions E1, E4, E7:

  E1 (U(1) gauge VALIDATED):
      plaq_avg    = ⟨½ Re Tr U_p⟩         ∈ [−1, 1]
      poly_loop   = |mean_{p} Tr U_p_path| (Polyakov-like, disorder param)

  E4 (Order parameters VALIDATED):
      omega[k]    = grade-k hopping order parameter, k=1..max_grade
                  = mean_{(i,j): grade_sum=k} 2 Re(ψ_i† U_ij ψ_j) / (r_i r_j)
      thresholds: Ω_2* = 0.5,  Ω_4* = 0.0,  Ω_7* = π/6

  E7 (MIXED triad prediction — KEY TEST):
      R           = MIXED-triad fraction over all site pairs
                  = n_MIXED / n_valid_triads

      Computed via gauge-covariant correlator signs:
          s(i,j) = sign( Re[ ψ_i† · U_{BFS-path(i→j)} · ψ_j ] )

      Baseline (random gauge, no dynamics): R ≈ 0.465 (from positive_control.py)
      U(1) prediction (E7 FALSIFIED):       R → 0  as β → ∞
      SU(2) hypothesis (E10 candidate):     R > 0  (non-Abelian frustration)

All observables are gauge-invariant.

Dependencies: su2.py, lattice.py, fields.py, action.py, positive_control.py
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

import su2
from lattice import Lattice2D
from fields import LinkDict, MatterDict
from action import plaquette_average
from positive_control import count_mixed_triads


# ──────────────────────────────────────────────────────────────────────────────
# BFS nearest-path finder (for gauge-correlator extension)
# ──────────────────────────────────────────────────────────────────────────────

def _bfs_shortest_path(
    lattice: Lattice2D,
    src: int,
) -> Dict[int, List[int]]:
    """
    BFS from `src` on the lattice graph.  Returns a dict mapping each
    reachable site to the shortest path (list of site indices) from src.

    The lattice adjacency uses both directions of canonical edges.
    """
    adj: Dict[int, List[int]] = {i: [] for i in range(lattice.N)}
    for (i, j) in lattice.edges():
        adj[i].append(j)
        adj[j].append(i)

    paths: Dict[int, List[int]] = {src: [src]}
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        for nxt in adj[cur]:
            if nxt not in paths:
                paths[nxt] = paths[cur] + [nxt]
                queue.append(nxt)
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# All-pair gauge-covariant sign matrix (E7 input)
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_pair_signs(
    links:   LinkDict,
    matter:  MatterDict,
    lattice: Lattice2D,
) -> Dict[Tuple[int, int], int]:
    """
    Build the full directed sign dict for all reachable site pairs.

        s(i, j) = sign( Re[ ψ_i† · U_{path(i→j)} · ψ_j ] )

    where U_{path} is the ordered product of link matrices along the BFS
    shortest path from i to j.

    This is the gauge-covariant analogue of the path-product sign extension
    validated in positive_control.py Control 5.

    Returns
    -------
    signs : dict { (i, j) → +1 | −1 | 0 }
        Contains entries for all ordered pairs (i ≠ j) reachable on the lattice.
        For a connected lattice all N(N−1) pairs are present.
    """
    N = lattice.N
    signs: Dict[Tuple[int, int], int] = {}

    for src in range(N):
        paths_from_src = _bfs_shortest_path(lattice, src)
        psi_src = matter[src]

        for dst, path in paths_from_src.items():
            if dst == src:
                continue

            # Compute ordered product U = U_{s0,s1} · U_{s1,s2} · ... · U_{sk-1,sk}
            _dim = next(iter(links.values())).shape[0]
            U = np.eye(_dim, dtype=complex)
            for k in range(len(path) - 1):
                a, b = path[k], path[k + 1]
                if (a, b) in links:
                    U = U @ links[(a, b)]
                elif (b, a) in links:
                    U = U @ links[(b, a)].conj().T
                else:
                    raise KeyError(f"Edge ({a},{b}) not found in links.")

            psi_dst = matter[dst]
            val = float(np.real(psi_src.conj() @ U @ psi_dst))

            if val > 0:
                signs[(src, dst)] = +1
            elif val < 0:
                signs[(src, dst)] = -1
            else:
                signs[(src, dst)] = 0

    return signs


# ──────────────────────────────────────────────────────────────────────────────
# E7: MIXED-triad fraction R
# ──────────────────────────────────────────────────────────────────────────────

def mixed_triad_R(
    links:   LinkDict,
    matter:  MatterDict,
    lattice: Lattice2D,
) -> Tuple[float, int, int]:
    """
    Compute the MIXED-triad fraction R for the current gauge+matter configuration.

    Steps:
      1. Build all-pair gauge-covariant sign dict via BFS path holonomies.
      2. Pass to count_mixed_triads() (validated in positive_control.py).

    Returns
    -------
    (R, n_mixed, n_valid) where
      R       = n_mixed / n_valid  (0 if n_valid == 0)
      n_mixed = number of MIXED triads
      n_valid = number of valid triads (all three directed pairs present)

    E7 predictions:
      U(1)  gauge (E1, NEGATIVE control): R → 0 as β → ∞
      SU(2) gauge (E10 candidate):       R > 0  (frustration survives)
      Random (baseline):                 R ≈ 0.465  (Control 5 baseline)
    """
    signs = compute_all_pair_signs(links, matter, lattice)
    n_mixed, n_valid = count_mixed_triads(signs, lattice.N)
    R = n_mixed / n_valid if n_valid > 0 else 0.0
    return float(R), int(n_mixed), int(n_valid)


# ──────────────────────────────────────────────────────────────────────────────
# E4: Grade-k hopping order parameters Ω_k
# ──────────────────────────────────────────────────────────────────────────────

def omega_k(
    links:   LinkDict,
    matter:  MatterDict,
    lattice: Lattice2D,
) -> Dict[int, float]:
    """
    Compute grade-k normalised hopping order parameters.

        Ω_k = mean_{(i,j) canonical: grade_sum(i,j) = k}
                  2 Re(ψ_i† U_{ij} ψ_j) / (r_i · r_j)

    where grade_sum(i,j) = (n_i + m_i) + (n_j + m_j).

    The denominator r_i · r_j removes the grade-magnitude weighting so Ω_k
    measures the *angular alignment* of the doublets through the gauge link.
    Ω_k ∈ [−2, 2].

    E4 thresholds (from HGST validation):
        Ω_2* = 0.5   → grade-2 order emerges
        Ω_4* = 0.0   → grade-4 boundary
        Ω_7* = π/6   → grade-7 critical angle (≈ 0.524)

    Returns
    -------
    dict { k → Ω_k }  for all grade sums k present in the lattice.
    """
    # Accumulate hopping per grade_sum
    sums:   Dict[int, float] = {}
    counts: Dict[int, int]   = {}

    for (i, j), Uij in links.items():
        ri = lattice.r[i]
        rj = lattice.r[j]
        if ri < 1e-15 or rj < 1e-15:
            continue
        psi_i = matter[i]
        psi_j = matter[j]
        hop_norm = float(np.real(psi_i.conj() @ Uij @ psi_j)) / (ri * rj)

        ni, mi = lattice.grade_indices(i)
        nj, mj = lattice.grade_indices(j)
        k = (ni + mi) + (nj + mj)

        sums[k]   = sums.get(k, 0.0)   + 2.0 * hop_norm
        counts[k] = counts.get(k, 0)   + 1

    return {k: sums[k] / counts[k] for k in sorted(sums)}


# ──────────────────────────────────────────────────────────────────────────────
# E1: Plaquette and Polyakov-like disorder observable
# ──────────────────────────────────────────────────────────────────────────────

def polyakov_disorder(links: LinkDict, lattice: Lattice2D) -> float:
    """
    Polyakov-loop-inspired disorder parameter.

    For each row m=1..L, compute the product of horizontal links along the row
    (a 'Polyakov path' around the horizontal direction with open boundary).
    Return the mean |½ Tr P| over all rows — measures long-range order.

    Close to 0 in the disordered phase, close to 1 in the ordered phase.
    (True Polyakov loop requires periodic b.c.; this is an open-path analogue.)
    """
    L    = lattice.L
    vals = []
    _dim = next(iter(links.values())).shape[0]
    for m in range(1, L + 1):
        U = np.eye(_dim, dtype=complex)
        for n in range(1, L):
            i = lattice.site_index(n,   m)
            j = lattice.site_index(n+1, m)
            if (i, j) in links:
                U = U @ links[(i, j)]
            elif (j, i) in links:
                U = U @ links[(j, i)].conj().T
        vals.append(abs(float(np.real(np.trace(U))) / _dim))
    return float(np.mean(vals))


# ──────────────────────────────────────────────────────────────────────────────
# Composite measurement dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ObservableSet:
    """
    All observables measured at one Monte Carlo sample.

    Fields
    ------
    plaq_avg   : ⟨½ Re Tr U_p⟩              (E1 diagnostic)
    poly_dis   : mean |½ Tr P_row|           (long-range order)
    R          : MIXED-triad fraction        (E7 KEY test)
    n_mixed    : raw count of MIXED triads
    n_valid    : raw count of valid triads
    omega      : dict k → Ω_k               (E4 order params)
    """
    plaq_avg:  float
    poly_dis:  float
    R:         float
    n_mixed:   int
    n_valid:   int
    omega:     Dict[int, float]

    def omega_at(self, k: int) -> Optional[float]:
        """Return Ω_k or None if grade level k not present."""
        return self.omega.get(k)

    def summary(self) -> str:
        lines = [
            f"  plaq_avg = {self.plaq_avg:+.6f}",
            f"  poly_dis = {self.poly_dis:.6f}",
            f"  R(MIXED) = {self.R:.6f}  "
            f"(n_mixed={self.n_mixed}, n_valid={self.n_valid})",
        ]
        if self.omega:
            ostr = "  Ω_k      = " + "  ".join(
                f"k{k}={v:+.4f}" for k, v in sorted(self.omega.items())
            )
            lines.append(ostr)
            # E4 threshold markers
            thresholds = {2: 0.5, 4: 0.0, 7: np.pi / 6}
            for k, thr in thresholds.items():
                if k in self.omega:
                    above = "above" if self.omega[k] >= thr else "below"
                    lines.append(f"            Ω_{k}={self.omega[k]:+.4f} "
                                 f"{'≥' if self.omega[k]>=thr else '<'}"
                                 f" Ω_{k}*={thr:.4f}  ({above} threshold)")
        return "\n".join(lines)


def measure(
    links:        LinkDict,
    matter:       MatterDict,
    lattice:      Lattice2D,
    skip_R:       bool = False,
) -> ObservableSet:
    """
    Measure all observables for the current configuration.

    Parameters
    ----------
    links, matter, lattice : current configuration
    skip_R : if True, skip the O(N³) MIXED-triad computation
             (useful during thermalisation when R measurement is not needed)

    Returns
    -------
    ObservableSet
    """
    pa  = plaquette_average(links, lattice)
    pd  = polyakov_disorder(links, lattice)
    wk  = omega_k(links, matter, lattice)

    if skip_R:
        return ObservableSet(plaq_avg=pa, poly_dis=pd,
                             R=float('nan'), n_mixed=0, n_valid=0,
                             omega=wk)

    R, nm, nv = mixed_triad_R(links, matter, lattice)
    return ObservableSet(plaq_avg=pa, poly_dis=pd,
                         R=R, n_mixed=nm, n_valid=nv,
                         omega=wk)


# ──────────────────────────────────────────────────────────────────────────────
# Accumulator for time-series statistics
# ──────────────────────────────────────────────────────────────────────────────

class ObservableAccumulator:
    """
    Accumulates ObservableSet measurements and computes means + std errors.

    Usage::

        acc = ObservableAccumulator()
        for _ in range(n_measure):
            updater.sweep()
            acc.add(measure(links, matter, lat))
        stats = acc.finalize()
        print(stats["R_mean"], stats["R_err"])
    """

    def __init__(self) -> None:
        self._R:        List[float] = []
        self._plaq:     List[float] = []
        self._poly:     List[float] = []
        self._omega:    Dict[int, List[float]] = {}

    def add(self, obs: ObservableSet) -> None:
        """Append one measurement."""
        if not np.isnan(obs.R):
            self._R.append(obs.R)
        self._plaq.append(obs.plaq_avg)
        self._poly.append(obs.poly_dis)
        for k, v in obs.omega.items():
            self._omega.setdefault(k, []).append(v)

    def finalize(self) -> Dict[str, float]:
        """
        Return dict of mean ± std-error for each accumulated observable.

        Keys: 'R_mean', 'R_err', 'plaq_mean', 'plaq_err',
              'poly_mean', 'poly_err',
              'omega_{k}_mean', 'omega_{k}_err' for each grade k.
        """
        def _stats(arr: List[float], key: str) -> Dict[str, float]:
            if not arr:
                return {f"{key}_mean": float("nan"), f"{key}_err": float("nan")}
            a = np.array(arr)
            return {
                f"{key}_mean": float(np.mean(a)),
                f"{key}_err":  float(np.std(a) / np.sqrt(len(a))),
            }

        out: Dict[str, float] = {}
        out.update(_stats(self._R,    "R"))
        out.update(_stats(self._plaq, "plaq"))
        out.update(_stats(self._poly, "poly"))
        for k, vals in self._omega.items():
            out.update(_stats(vals, f"omega_{k}"))
        out["n_samples"] = float(len(self._plaq))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    from fields import initialize_links, initialize_matter
    from updates import MetropolisUpdater

    rng = np.random.default_rng(999)
    PASS, FAIL = "PASS", "FAIL"
    results: list[tuple[str, bool, str]] = []

    lat = Lattice2D(4)

    # ── T1: Cold start plaq_avg = 1 ──────────────────────────────────────
    lk_c = initialize_links(lat, random=False)
    mt_c = initialize_matter(lat, random=False)
    obs_c = measure(lk_c, mt_c, lat)
    ok1 = abs(obs_c.plaq_avg - 1.0) < 1e-12
    results.append(("T1  cold plaq_avg = 1.0", ok1,
                    f"plaq_avg = {obs_c.plaq_avg:.10f}"))

    # ── T2: Cold start omega_k — all edges identity → hops = r_i·r_j·1 ──
    # For cold start: U=I, ψ_i=[r_i,0], ψ_j=[r_j,0]
    # 2 Re(ψ_i† I ψ_j)/(r_i r_j) = 2 Re([r_i,0]·[r_j,0])/(r_i r_j) = 2·1 = 2? No...
    # ψ_i=[r_i,0], so ψ_i†=conj([r_i,0])= [r_i,0] (real).
    # ψ_i† I ψ_j = r_i * r_j (dot product of [1,0]·[1,0] = 1) → 2*1 = 2 but
    # divided by r_i*r_j... wait, the matter is r_i * chi_i where chi_i=[1,0].
    # So ψ_i† ψ_j = r_i * conj([1,0]) · r_j * [1,0] = r_i*r_j*1.
    # normalized: 2*Re(r_i*r_j)/(r_i*r_j) = 2. So omega_k(cold) = 2 for all k.
    wk_c = obs_c.omega
    ok2 = all(abs(v - 2.0) < 1e-10 for v in wk_c.values()) and len(wk_c) > 0
    results.append(("T2  cold omega_k = 2.0 for all k (perfectly aligned)", ok2,
                    f"omega values: {list(wk_c.values())[:3]}..."))

    # ── T3: hot start R in plausible range ────────────────────────────────
    lk_h = initialize_links(lat, random=True)
    mt_h = initialize_matter(lat, random=True)
    obs_h = measure(lk_h, mt_h, lat)
    ok3 = 0.0 <= obs_h.R <= 1.0 and obs_h.n_valid > 0
    results.append(("T3  hot R ∈ [0,1], n_valid > 0", ok3,
                    f"R={obs_h.R:.4f}, n_mixed={obs_h.n_mixed}, "
                    f"n_valid={obs_h.n_valid}"))

    # ── T4: gauge invariance of R ─────────────────────────────────────────
    from fields import gauge_transform, random_gauge
    V = random_gauge(lat)
    lk_t, mt_t = gauge_transform(lk_h, mt_h, lat, V)
    obs_t = measure(lk_t, mt_t, lat)
    # R is gauge-invariant (ψ_i† U_path ψ_j transforms as scalar under V_i,V_j)
    ok4 = abs(obs_t.R - obs_h.R) < 1e-6
    results.append(("T4  R gauge invariant", ok4,
                    f"R(orig)={obs_h.R:.6f}, R(gauged)={obs_t.R:.6f}, "
                    f"diff={abs(obs_t.R-obs_h.R):.2e}"))

    # ── T5: gauge invariance of omega_k ──────────────────────────────────
    max_omega_diff = max(
        abs(obs_t.omega.get(k, 0) - obs_h.omega.get(k, 0))
        for k in obs_h.omega
    )
    ok5 = max_omega_diff < 1e-8
    results.append(("T5  omega_k gauge invariant", ok5,
                    f"max |Δω_k| = {max_omega_diff:.2e}"))

    # ── T6: skip_R flag works ────────────────────────────────────────────
    obs_nr = measure(lk_h, mt_h, lat, skip_R=True)
    ok6 = np.isnan(obs_nr.R) and obs_nr.n_valid == 0
    results.append(("T6  skip_R=True → R=nan, n_valid=0", ok6,
                    f"R={obs_nr.R}, n_valid={obs_nr.n_valid}"))

    # ── T7: ObservableAccumulator statistics ─────────────────────────────
    acc = ObservableAccumulator()
    for _ in range(50):
        acc.add(obs_h)            # same obs repeated → std = 0
    stats = acc.finalize()
    ok7 = (abs(stats["R_mean"] - obs_h.R) < 1e-12 and
           stats["R_err"] < 1e-12 and
           stats["n_samples"] == 50.0)
    results.append(("T7  Accumulator mean/err (50 identical samples)", ok7,
                    f"R_mean={stats['R_mean']:.4f}, R_err={stats['R_err']:.2e}, "
                    f"n={stats['n_samples']:.0f}"))

    # ── T8: R changes as beta increases (E7 dynamics test) ───────────────
    # After thermalisation at high beta, R should change from the hot baseline
    lat8  = Lattice2D(4)
    lk8   = initialize_links(lat8, random=True)
    mt8   = initialize_matter(lat8, random=True)
    R_hot = measure(lk8, mt8, lat8).R
    upd   = MetropolisUpdater(lat8, lk8, mt8, beta_g=6.0, kappa=0.0, seed=42)
    upd.thermalize(300, update_matter=False)
    R_warm = measure(lk8, mt8, lat8).R
    # R should change (direction depends on group, but it must change at β=6)
    ok8 = abs(R_warm - R_hot) > 0.01
    results.append(("T8  R changes after thermalisation at β=6", ok8,
                    f"R: hot={R_hot:.4f} → warm={R_warm:.4f}  "
                    f"Δ={abs(R_warm-R_hot):.4f}"))

    # ── T9: summary() produces non-empty string ──────────────────────────
    s = obs_h.summary()
    ok9 = "plaq_avg" in s and "R(MIXED)" in s and "Ω_k" in s
    results.append(("T9  ObservableSet.summary() contains key fields", ok9,
                    f"length={len(s)} chars"))

    # ── T10: ObservableAccumulator with varying R values ─────────────────
    acc2 = ObservableAccumulator()
    vals = [0.3, 0.4, 0.5, 0.6, 0.7]
    for v in vals:
        o = ObservableSet(plaq_avg=0.5, poly_dis=0.5, R=v,
                          n_mixed=3, n_valid=6, omega={4: 1.0})
        acc2.add(o)
    s2 = acc2.finalize()
    ok10 = (abs(s2["R_mean"] - 0.5) < 1e-10 and
            abs(s2["R_err"]  - np.std(vals)/np.sqrt(5)) < 1e-10)
    results.append(("T10 Accumulator mean=0.5, err=σ/√5 for vals [0.3..0.7]", ok10,
                    f"R_mean={s2['R_mean']:.4f}, R_err={s2['R_err']:.6f}"))

    # ── Print ─────────────────────────────────────────────────────────────
    print("=" * 66)
    print("observables.py — Gauge-Invariant Observables Self-Test")
    print("=" * 66)
    all_pass = True
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        print(f"         {detail}")
        if not ok:
            all_pass = False
    print("-" * 66)
    if all_pass:
        print(f"  All {len(results)} tests PASSED.")
        print()
        print("  HOT-START OBSERVABLES (L=4):")
        print(obs_h.summary())
        print()
        print("  E7 DYNAMICS TEST:")
        print(f"    R(hot)  = {R_hot:.4f}")
        print(f"    R(β=6)  = {R_warm:.4f}  ({'+' if R_warm>R_hot else ''}"
              f"{R_warm-R_hot:+.4f})")
        print()
        print("  E4 THRESHOLDS CHECK (hot start):")
        for k, thr in {2: 0.5, 4: 0.0, 7: np.pi/6}.items():
            if k in obs_h.omega:
                v = obs_h.omega[k]
                print(f"    Ω_{k} = {v:+.4f}  (Ω_{k}* = {thr:.4f}  "
                      f"{'OK: above' if v >= thr else 'below'} threshold)")
    else:
        print("  SOME TESTS FAILED — review before proceeding to simulation.py")
    print("=" * 66)


if __name__ == "__main__":
    _run_tests()
