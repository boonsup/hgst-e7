#!/usr/bin/env python3
"""
sm_observables.py - Gauge-invariant observables for SU(3)×SU(2)×U(1) gauge theory.
=====================================================================================
Measures:
  • Plaquette averages for each factor: ⟨plaq_3⟩, ⟨plaq_2⟩, ⟨plaq_1⟩
  • MIXED triad fraction R for quark sector (colour+weak doublets)
  • MIXED triad fraction R for lepton sector (weak doublets + hypercharge)
  • SMObservables dataclass bundling all measurements

The sign of a pair (i,j) is computed from the gauge-invariant inner product:
    s(i,j) = sign Re⟨ψ_i | P_{i→j} | ψ_j⟩
where P_{i→j} is the BFS shortest-path-ordered product of link matrices.
This is gauge-invariant: the sign is unchanged under site-dependent
SU(3)×SU(2)×U(1) transformations.

Epistemic status: VALIDATED after test suite passes.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import dataclasses
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lattice import Lattice2D
from sm_gauge import SMGaugeElement, sm_plaquette_average
from sm_fields import QuarkDoublet, LeptonDoublet
from positive_control import count_mixed_triads


# ---------------------------------------------------------------------------
# BFS path-ordered transport
# ---------------------------------------------------------------------------

def _bfs_paths(lattice: Lattice2D) -> Dict[int, Dict[int, list]]:
    """
    Compute BFS shortest paths from every source to every other site.

    Returns paths[src][dst] = list of site indices (BFS path, including src and dst).
    Uses the undirected graph (both orientations of each canonical edge).
    """
    N = lattice.N
    adj = {i: [] for i in range(N)}
    for (i, j) in lattice.edges():
        adj[i].append(j)
        adj[j].append(i)

    all_paths = {}
    for src in range(N):
        paths = {src: [src]}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt not in paths:
                    paths[nxt] = paths[cur] + [nxt]
                    queue.append(nxt)
        all_paths[src] = paths
    return all_paths


def _path_ordered_transport(links: dict, path: list) -> SMGaugeElement:
    """
    Compute U = U_{s0,s1} @ U_{s1,s2} @ … along a BFS path.
    Handles reverse edges via dagger.
    """
    U = SMGaugeElement.identity()
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        if (a, b) in links:
            U = U @ links[(a, b)]
        elif (b, a) in links:
            U = U @ links[(b, a)].dagger()
        else:
            raise KeyError(f"Edge ({a},{b}) not in links")
    return U


# ---------------------------------------------------------------------------
# Sign dictionaries
# ---------------------------------------------------------------------------

def compute_sign_dict_quark(
    links: Dict[Tuple[int, int], SMGaugeElement],
    quarks: Dict[int, QuarkDoublet],
    lattice: Lattice2D,
    all_paths: Optional[Dict[int, Dict[int, list]]] = None
) -> Dict[Tuple[int, int], int]:
    """
    Build sign dict for all site pairs using quark doublets.

    s(i,j) = sign Re⟨ψ_i | U_{path}(i→j) | ψ_j⟩

    where the action of U on a quark:  SU(2) on the doublet, SU(3) on colour.
    """
    if all_paths is None:
        all_paths = _bfs_paths(lattice)

    signs = {}
    N = lattice.N

    for src in range(N):
        psi_src = quarks[src]
        for dst, path in all_paths[src].items():
            if dst == src:
                continue
            U = _path_ordered_transport(links, path)
            # Apply U to psi_dst
            psi_dst = quarks[dst]
            u_temp = U.su2[0, 0] * psi_dst.up + U.su2[0, 1] * psi_dst.down
            d_temp = U.su2[1, 0] * psi_dst.up + U.su2[1, 1] * psi_dst.down
            U3_u = U.su3 @ u_temp
            U3_d = U.su3 @ d_temp
            prod_re = (np.vdot(psi_src.up, U3_u) +
                       np.vdot(psi_src.down, U3_d)).real
            if prod_re > 1e-15:
                signs[(src, dst)] = 1
            elif prod_re < -1e-15:
                signs[(src, dst)] = -1
            # zero: omit (not counted in n_valid)

    return signs


def compute_sign_dict_lepton(
    links: Dict[Tuple[int, int], SMGaugeElement],
    leptons: Dict[int, LeptonDoublet],
    lattice: Lattice2D,
    all_paths: Optional[Dict[int, Dict[int, list]]] = None
) -> Dict[Tuple[int, int], int]:
    """
    Build sign dict for all site pairs using lepton doublets.

    s(i,j) = sign Re⟨l_i | U_{path}(i→j) | l_j⟩

    U acts via SU(2) rotation + U(1) hypercharge Y=−1/2.
    """
    if all_paths is None:
        all_paths = _bfs_paths(lattice)

    signs = {}
    N = lattice.N

    for src in range(N):
        l_src = leptons[src]
        for dst, path in all_paths[src].items():
            if dst == src:
                continue
            U = _path_ordered_transport(links, path)
            l_dst = leptons[dst]
            u1_fac = np.exp(-0.5j * np.angle(U.u1))
            nu_temp = U.su2[0, 0] * l_dst.neutrino + U.su2[0, 1] * l_dst.electron
            e_temp  = U.su2[1, 0] * l_dst.neutrino + U.su2[1, 1] * l_dst.electron
            U_nu = u1_fac * nu_temp
            U_e  = u1_fac * e_temp
            prod_re = (l_src.neutrino.conjugate() * U_nu +
                       l_src.electron.conjugate() * U_e).real
            if prod_re > 1e-15:
                signs[(src, dst)] = 1
            elif prod_re < -1e-15:
                signs[(src, dst)] = -1

    return signs


# ---------------------------------------------------------------------------
# R measurement
# ---------------------------------------------------------------------------

def mixed_triad_R(
    signs: Dict[Tuple[int, int], int],
    N: int
) -> Tuple[float, int, int]:
    """Compute MIXED fraction R from sign dict using positive_control kernel."""
    n_mixed, n_valid = count_mixed_triads(signs, N)
    R = n_mixed / n_valid if n_valid > 0 else 0.0
    return R, n_mixed, n_valid


# ---------------------------------------------------------------------------
# Observables dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SMObservables:
    """All observables at one Monte Carlo sample."""
    plaq_3:          float
    plaq_2:          float
    plaq_1:          float
    R_quark:         float
    R_lepton:        float
    n_mixed_quark:   int
    n_valid_quark:   int
    n_mixed_lepton:  int
    n_valid_lepton:  int

    def summary(self) -> str:
        return (
            f"  plaq_3  = {self.plaq_3:.6f}\n"
            f"  plaq_2  = {self.plaq_2:.6f}\n"
            f"  plaq_1  = {self.plaq_1:.6f}\n"
            f"  R_quark = {self.R_quark:.6f}  "
            f"(n_mixed={self.n_mixed_quark}, n_valid={self.n_valid_quark})\n"
            f"  R_lepton= {self.R_lepton:.6f}  "
            f"(n_mixed={self.n_mixed_lepton}, n_valid={self.n_valid_lepton})"
        )


def measure(
    links: Dict[Tuple[int, int], SMGaugeElement],
    quarks: Dict[int, QuarkDoublet],
    leptons: Dict[int, LeptonDoublet],
    lattice: Lattice2D,
    skip_R: bool = False,
    all_paths: Optional[Dict[int, Dict[int, list]]] = None
) -> SMObservables:
    """
    Measure all observables for the current field configuration.

    Parameters
    ----------
    skip_R : if True, skip expensive R calculation (plaquettes only)
    all_paths : pre-computed BFS paths (pass to avoid recomputation per sample)
    """
    plaq3, plaq2, plaq1 = sm_plaquette_average(links, lattice)

    if skip_R:
        return SMObservables(
            plaq_3=plaq3, plaq_2=plaq2, plaq_1=plaq1,
            R_quark=0.0, R_lepton=0.0,
            n_mixed_quark=0, n_valid_quark=0,
            n_mixed_lepton=0, n_valid_lepton=0
        )

    if all_paths is None:
        all_paths = _bfs_paths(lattice)

    signs_q = compute_sign_dict_quark(links, quarks, lattice, all_paths)
    Rq, nm_q, nv_q = mixed_triad_R(signs_q, lattice.N)

    signs_l = compute_sign_dict_lepton(links, leptons, lattice, all_paths)
    Rl, nm_l, nv_l = mixed_triad_R(signs_l, lattice.N)

    return SMObservables(
        plaq_3=plaq3, plaq_2=plaq2, plaq_1=plaq1,
        R_quark=Rq,   R_lepton=Rl,
        n_mixed_quark=nm_q,   n_valid_quark=nv_q,
        n_mixed_lepton=nm_l,  n_valid_lepton=nv_l
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> bool:
    from sm_fields import initialize_sm_links, initialize_quarks, initialize_leptons

    PASS, FAIL = "PASS", "FAIL"
    results = []
    rng = np.random.default_rng(42)
    lat = Lattice2D(4)

    links  = initialize_sm_links(lat, random=True, rng=rng)
    quarks = initialize_quarks(lat, random=True, rng=rng)
    leptons= initialize_leptons(lat, random=True, rng=rng)

    # Pre-compute BFS paths once
    all_paths = _bfs_paths(lat)

    obs = measure(links, quarks, leptons, lat, all_paths=all_paths)

    # T1: Plaquette averages in [−1, 1]
    ok1 = (-1.0 <= obs.plaq_3 <= 1.0 and
           -1.0 <= obs.plaq_2 <= 1.0 and
           -1.0 <= obs.plaq_1 <= 1.0)
    results.append(("T1 plaquette averages in [−1,1]", ok1,
                   f"plaq3={obs.plaq_3:.4f}, plaq2={obs.plaq_2:.4f}, "
                   f"plaq1={obs.plaq_1:.4f}"))

    # T2: R values in [0, 1]
    ok2 = (0.0 <= obs.R_quark <= 1.0 and 0.0 <= obs.R_lepton <= 1.0)
    results.append(("T2 R in [0,1]", ok2,
                   f"R_quark={obs.R_quark:.4f}, R_lepton={obs.R_lepton:.4f}"))

    # T3: skip_R=True returns zero R
    obs_skip = measure(links, quarks, leptons, lat, skip_R=True)
    ok3 = (obs_skip.R_quark == 0.0 and obs_skip.R_lepton == 0.0 and
           obs_skip.n_valid_quark == 0 and obs_skip.n_valid_lepton == 0)
    results.append(("T3 skip_R=True zeroes R", ok3, ""))

    # T4: R values are finite
    ok4 = np.isfinite(obs.R_quark) and np.isfinite(obs.R_lepton)
    results.append(("T4 R finite", ok4, ""))

    print("=" * 66)
    print("sm_observables.py — Observables Self-Test")
    print("=" * 66)
    all_pass = True
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if detail:
            print(f"         {detail}")
        if not ok:
            all_pass = False
    print("-" * 66)
    if all_pass:
        print(f"  All {len(results)} tests PASSED.")
        print("\n  Observables module for SM is ready.")
        print("\n  Sample measurement (hot start, L=4):")
        print(obs.summary())
    else:
        print("  SOME TESTS FAILED — debug before proceeding.")
    print("=" * 66)
    return all_pass


if __name__ == "__main__":
    import sys as _sys
    success = _run_tests()
    _sys.exit(0 if success else 1)
