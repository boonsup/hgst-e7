"""
fields.py — Field Initialization and Gauge Transformations
==========================================================
Layer 2 Operations Module: Field configuration management

Defines the two field types in the SU(2) gauge + matter system:

  GAUGE LINKS:  U_{ij} ∈ SU(2),  one per canonical lattice edge (i,j)
  MATTER:       ψ_i = r_i · χ_i,  one per lattice site
                  r_i = lat.r[i] = 2^(n_i - m_i)   (HGST grade magnitude)
                  χ_i ∈ ℂ²,  |χ_i| = 1             (SU(2) doublet)

Gauge transformation V : sites → SU(2):
  U_{ij}  →  V_i · U_{ij} · V_j†    (adjoint rep on links)
  ψ_i     →  V_i · ψ_i              (fundamental rep on matter)
  r_i     →  r_i                     (gauge-invariant grade magnitude)

This module is group-agnostic in principle; the group is passed via the
`group` parameter ('su2' or 'u1') and resolved to the appropriate algebra
module.  For the main SU(2) simulation, leave `group='su2'`.

Dependencies: su2.py, u1.py, lattice.py
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple, Union

import numpy as np

from lattice import Lattice2D
import su2
import su3
import u1 as u1mod

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

# SU(2) link: 2×2 complex numpy array
SU2Link  = np.ndarray            # shape (2,2), dtype complex128
# Matter field at one site: complex 2-vector (NOT yet scaled by r)
Doublet  = np.ndarray            # shape (2,), dtype complex128

# Link and matter dicts
LinkDict   = Dict[Tuple[int, int], SU2Link]
MatterDict = Dict[int, np.ndarray]   # site → ψ = r·χ  (shape (2,))


# ──────────────────────────────────────────────────────────────────────────────
# Doublet helpers
# ──────────────────────────────────────────────────────────────────────────────

def random_doublet(rng: np.random.Generator) -> Doublet:
    """
    Sample a Haar-uniform SU(2) doublet: a unit vector in ℂ².

    Method: draw four independent Gaussians, form a complex pair, normalise.
    This is the correct Haar measure on S³ ≅ SU(2).
    """
    v = rng.standard_normal(4).view(np.complex128)  # shape (2,)
    return v / np.linalg.norm(v)


def identity_doublet() -> Doublet:
    """Return the 'cold' doublet [1, 0]."""
    return np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)


def apply_su2_to_doublet(U: SU2Link, chi: Doublet) -> Doublet:
    """Return U · χ  (standard SU(2) fundamental representation action)."""
    return U @ chi

def random_triplet(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-uniform SU(3) triplet: unit vector in ℂ³."""
    v = rng.standard_normal(6).view(np.complex128)  # shape (3,)
    return v / np.linalg.norm(v)


def identity_triplet() -> np.ndarray:
    """Return the 'cold' SU(3) triplet [1, 0, 0]."""
    return np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)

# ──────────────────────────────────────────────────────────────────────────────
# Link initialization
# ──────────────────────────────────────────────────────────────────────────────

def initialize_links(
    lattice: Lattice2D,
    *,
    group: str = "su2",
    random: bool = True,
    rng: np.random.Generator | None = None,
) -> LinkDict:
    """
    Create a link configuration for every canonical edge in the lattice.

    Parameters
    ----------
    lattice : Lattice2D
        Grade lattice providing the edge list.
    group : {'su2', 'u1'}
        Gauge group.  'su2' returns 2×2 complex arrays wrapped in a 1×1
        array for u1 (stored as complex scalar inside a (1,1) ndarray so
        that action.py can treat both uniformly via np.trace).
        Default: 'su2'.
    random : bool
        True  → Haar-random links  (hot start)
        False → identity links     (cold start, all plaquettes = 1)
    rng : np.random.Generator, optional
        RNG instance.  Created fresh if None.

    Returns
    -------
    links : dict { (i,j) → U }
        Keys are exactly lattice.edges() in canonical order.
    """
    if rng is None:
        rng = np.random.default_rng()

    links: LinkDict = {}
    for edge in lattice.edges():
        if group == "su2":
            links[edge] = su2.random_su2() if random else su2.identity_su2()
        elif group == "su3":
            links[edge] = su3.random_su3(rng) if random else su3.identity_su3()
        elif group == "u1":
            z = u1mod.random_u1(rng) if random else u1mod.identity_u1()
            # Store U(1) element as a (1,1) complex array so action.py can use
            # np.trace uniformly.  The scalar value is accessible as links[e][0,0].
            links[edge] = np.array([[z]], dtype=np.complex128)
        else:
            raise ValueError(f"Unknown group '{group}'.  Use 'su2', 'su3', or 'u1'.")

    return links


# ──────────────────────────────────────────────────────────────────────────────
# Matter initialization
# ──────────────────────────────────────────────────────────────────────────────

def initialize_matter(
    lattice: Lattice2D,
    *,
    group: str = "su2",
    random: bool = True,
    rng: np.random.Generator | None = None,
) -> MatterDict:
    """
    Create a matter-field configuration for every site in the lattice.

    Each site i carries ψ_i = r_i · χ_i where χ_i is a unit vector in
    the fundamental representation: ℂ² (SU2 doublet) or ℂ³ (SU3 triplet).

    Parameters
    ----------
    lattice : Lattice2D
    group   : {'su2', 'su3'}
    random  : bool  —  True → Haar-random; False → cold (reference) vector
    rng     : np.random.Generator, optional

    Returns
    -------
    matter : dict { site_index → ψ_i }  with |ψ_i| = r_i
    """
    if rng is None:
        rng = np.random.default_rng()

    matter: MatterDict = {}
    for site in range(lattice.N):
        r_i = lattice.r[site]                            # grade magnitude ≥ 0
        if group == "su2":
            chi = random_doublet(rng) if random else identity_doublet()
        elif group == "su3":
            chi = random_triplet(rng) if random else identity_triplet()
        else:
            raise ValueError(f"Unknown group '{group}'.  Use 'su2' or 'su3'.")
        matter[site] = r_i * chi

    return matter


# ──────────────────────────────────────────────────────────────────────────────
# Gauge transformation
# ──────────────────────────────────────────────────────────────────────────────

def gauge_transform(
    links: LinkDict,
    matter: MatterDict,
    lattice: Lattice2D,
    V: Dict[int, SU2Link],
) -> Tuple[LinkDict, MatterDict]:
    """
    Apply a gauge transformation V to both the link and matter configurations.

    Transformation rules:
        U_{ij}  →  V_i · U_{ij} · V_j†          (adjoint / bifundamental)
        ψ_i     →  V_i · ψ_i                     (fundamental)
        r_i     →  r_i         (unchanged; scalar grade, not in SU(2))

    Parameters
    ----------
    links  : LinkDict   canonical-edge → SU(2) link matrix
    matter : MatterDict site → ψ_i
    lattice: Lattice2D
    V      : dict { site → SU(2) matrix }
             Must have entries for every site 0..N-1.

    Returns
    -------
    (links_t, matter_t) : new dicts (inputs are not mutated)

    Notes
    -----
    The grade magnitude |ψ_i| = r_i · |χ_i| is preserved because SU(2)
    matrices are unitary → |V_i χ_i| = |χ_i| = 1 → |V_i ψ_i| = r_i.
    """
    links_t  = {}
    for (i, j), U in links.items():
        Vi  = V[i]
        Vj  = V[j]
        links_t[(i, j)] = Vi @ U @ su2.dagger(Vj)

    matter_t = {}
    for site, psi in matter.items():
        matter_t[site] = V[site] @ psi

    return links_t, matter_t


def identity_gauge(lattice: Lattice2D) -> Dict[int, SU2Link]:
    """Return V_i = 𝟙 for every site (trivial gauge transform)."""
    return {s: su2.identity_su2() for s in range(lattice.N)}


def random_gauge(
    lattice: Lattice2D,
    rng: np.random.Generator | None = None,
) -> Dict[int, SU2Link]:
    """Return a Haar-random V_i for every site."""
    if rng is None:
        rng = np.random.default_rng()
    return {s: su2.random_su2() for s in range(lattice.N)}


# ──────────────────────────────────────────────────────────────────────────────
# Deep-copy helpers (needed by updates.py for Metropolis accept/reject)
# ──────────────────────────────────────────────────────────────────────────────

def copy_links(links: LinkDict) -> LinkDict:
    """Return a deep copy of the link dict."""
    return {e: U.copy() for e, U in links.items()}


def copy_matter(matter: MatterDict) -> MatterDict:
    """Return a deep copy of the matter dict."""
    return {s: psi.copy() for s, psi in matter.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Gauge-covariant correlator helper (used by observables.py)
# ──────────────────────────────────────────────────────────────────────────────

def path_holonomy(
    links: LinkDict,
    path: list[int],
) -> SU2Link:
    """
    Compute the ordered product of links along a path of sites.

    U_{path} = U_{s0,s1} · U_{s1,s2} · ... · U_{s_{k-1},s_k}

    Each link is retrieved via canonical-edge lookup (with dagger for reverse
    direction, exactly as in u1._get_link but for SU(2)).

    Parameters
    ----------
    links : LinkDict
    path  : list of site indices [s0, s1, ..., sk]   (k ≥ 1)

    Returns
    -------
    2×2 SU(2) matrix (approximately; rounding errors accumulate for long paths)
    """
    result = su2.identity_su2()
    for k in range(len(path) - 1):
        i, j = path[k], path[k + 1]
        if (i, j) in links:
            result = result @ links[(i, j)]
        elif (j, i) in links:
            result = result @ su2.dagger(links[(j, i)])
        else:
            raise KeyError(f"Edge ({i},{j}) not found in links (neither direction).")
    return result


def gauge_correlator_sign(
    links: LinkDict,
    matter: MatterDict,
    path: list[int],
) -> int:
    """
    Compute the sign of the gauge-covariant correlator along a path.

        sign_c(i,j) = sign( Re[ ψ_i† · U_path · ψ_j ] )

    This is the gauge-invariant sign used by observables.py for the
    MIXED-triad fraction R(N).

    Returns +1, −1, or 0 (if the correlator vanishes exactly).
    """
    src, dst = path[0], path[-1]
    psi_i = matter[src]
    psi_j = matter[dst]
    U     = path_holonomy(links, path)
    val   = float(np.real(psi_i.conj() @ U @ psi_j))
    if val > 0:
        return +1
    elif val < 0:
        return -1
    else:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    rng = np.random.default_rng(2025)
    PASS, FAIL = "PASS", "FAIL"
    results: list[tuple[str, bool, str]] = []

    # ── T1: SU(2) link initialization — all links are SU(2) ──────────────
    lat = Lattice2D(4)
    links_r = initialize_links(lat, random=True,  rng=rng)
    links_c = initialize_links(lat, random=False, rng=rng)

    ok1a = all(su2.is_su2(U) for U in links_r.values())
    ok1b = all(su2.is_su2(U) for U in links_c.values())
    ok1  = ok1a and ok1b
    results.append(("T1  init_links: all U ∈ SU(2)", ok1,
                    f"n_edges={len(links_r)}, hot={ok1a}, cold={ok1b}"))

    # ── T2: Cold start → all links = identity ────────────────────────────
    id_err = max(np.linalg.norm(U - su2.identity_su2()) for U in links_c.values())
    ok2 = id_err < 1e-14
    results.append(("T2  cold links = identity", ok2,
                    f"max ‖U−I‖ = {id_err:.2e}"))

    # ── T3: Matter field magnitudes |ψ_i| = r_i ──────────────────────────
    matter_r = initialize_matter(lat, random=True,  rng=rng)
    matter_c = initialize_matter(lat, random=False, rng=rng)

    norm_errs = [abs(np.linalg.norm(matter_r[s]) - lat.r[s]) for s in range(lat.N)]
    ok3a = max(norm_errs) < 1e-12
    norm_errs_c = [abs(np.linalg.norm(matter_c[s]) - lat.r[s]) for s in range(lat.N)]
    ok3b = max(norm_errs_c) < 1e-12
    ok3  = ok3a and ok3b
    results.append(("T3  |ψ_i| = r_i for all sites", ok3,
                    f"max err hot={max(norm_errs):.2e}, cold={max(norm_errs_c):.2e}"))

    # ── T4: Gauge transform — links remain SU(2) ─────────────────────────
    V = random_gauge(lat, rng)
    links_t, matter_t = gauge_transform(links_r, matter_r, lat, V)
    ok4 = all(su2.is_su2(U) for U in links_t.values())
    results.append(("T4  gauge transform: U_t ∈ SU(2)", ok4,
                    f"n_links={len(links_t)}"))

    # ── T5: Gauge invariance of plaquette Re(Tr U_p) ─────────────────────
    from lattice import Lattice2D as _L2D

    def _plaq_retrace(lnks, plqs):
        vals = []
        for p in plqs:
            i0, i1, i2, i3 = p
            def gl(a, b):
                if (a,b) in lnks: return lnks[(a,b)]
                return su2.dagger(lnks[(b,a)])
            U = gl(i0,i1) @ gl(i1,i2) @ gl(i2,i3) @ gl(i3,i0)
            vals.append(0.5 * np.real(np.trace(U)))
        return vals

    plqs = lat.plaquettes()
    re_orig = _plaq_retrace(links_r, plqs)
    re_tran = _plaq_retrace(links_t, plqs)
    max_diff = max(abs(a - b) for a, b in zip(re_orig, re_tran))
    ok5 = max_diff < 1e-10
    results.append(("T5  plaquette Re½Tr invariant under gauge transform", ok5,
                    f"max |Tr_orig−Tr_gauged| = {max_diff:.2e}"))

    # ── T6: Gauge covariance of matter — |ψ_t| = r_i preserved ──────────
    norm_errs_t = [abs(np.linalg.norm(matter_t[s]) - lat.r[s]) for s in range(lat.N)]
    ok6 = max(norm_errs_t) < 1e-12
    results.append(("T6  |V_i ψ_i| = r_i (SU(2) unitary → norm preserved)", ok6,
                    f"max err = {max(norm_errs_t):.2e}"))

    # ── T7: Identity gauge transform → fields unchanged ──────────────────
    V_id = identity_gauge(lat)
    links_id, matter_id = gauge_transform(links_r, matter_r, lat, V_id)
    err_links  = max(np.linalg.norm(links_id[e] - links_r[e]) for e in lat.edges())
    err_matter = max(np.linalg.norm(matter_id[s] - matter_r[s]) for s in range(lat.N))
    ok7 = err_links < 1e-14 and err_matter < 1e-14
    results.append(("T7  identity gauge → fields unchanged", ok7,
                    f"ΔU_max={err_links:.2e}, Δψ_max={err_matter:.2e}"))

    # ── T8: path_holonomy — single hop equals link ────────────────────────
    test_edge = lat.edges()[0]
    i0, i1 = test_edge
    hop = path_holonomy(links_r, [i0, i1])
    err_hop = np.linalg.norm(hop - links_r[test_edge])
    # Also test reverse hop
    hop_rev = path_holonomy(links_r, [i1, i0])
    err_rev = np.linalg.norm(hop_rev - su2.dagger(links_r[test_edge]))
    ok8 = err_hop < 1e-14 and err_rev < 1e-14
    results.append(("T8  path_holonomy: single hop = link; reverse = U†", ok8,
                    f"forward err={err_hop:.2e}, reverse err={err_rev:.2e}"))

    # ── T9: gauge_correlator_sign returns ±1 or 0 ────────────────────────
    signs_ok = True
    for _ in range(200):
        e = lat.edges()[int(rng.integers(len(lat.edges())))]
        sgn = gauge_correlator_sign(links_r, matter_r, list(e))
        if sgn not in (-1, 0, 1):
            signs_ok = False
    ok9 = signs_ok
    results.append(("T9  gauge_correlator_sign ∈ {-1, 0, +1}", ok9,
                    "200 random single-hop checks"))

    # ── T10: U(1) links stored as (1,1) array ────────────────────────────
    links_u1 = initialize_links(lat, group='u1', random=True, rng=rng)
    ok10a = all(v.shape == (1, 1) for v in links_u1.values())
    ok10b = all(abs(abs(v[0, 0]) - 1.0) < 1e-12 for v in links_u1.values())
    ok10  = ok10a and ok10b
    results.append(("T10 U(1) links: shape (1,1), |z|=1", ok10,
                    f"all (1,1)={ok10a}, all |z|=1={ok10b}"))

    # ── Print ─────────────────────────────────────────────────────────────
    print("=" * 66)
    print("fields.py — Field Initialization & Gauge Transform Self-Test")
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
        print("  FIELD INVENTORY (L=4 lattice):")
        print(f"    Sites    : {lat.N}")
        print(f"    Edges    : {lat.n_edges}")
        print(f"    Links    : {lat.n_edges} × 2×2 SU(2) matrices")
        print(f"    Matter   : {lat.N} × ℂ² doublets, |ψ_i| = r_i")
        r_vals = sorted(set(f"{lat.r[s]:.3f}" for s in range(lat.N)))
        print(f"    r values : {r_vals}")
    else:
        print("  SOME TESTS FAILED — review before proceeding to action.py")
    print("=" * 66)


if __name__ == "__main__":
    _run_tests()
