"""
action.py — Wilson Gauge Action and Matter Hopping Term
=======================================================
Layer 3 Operations Module: Energy functionals

Two action contributions:

  GAUGE (Wilson):
      S_g = β Σ_p  (1 − ½ Re Tr U_p)
      U_p = U_{i0,i1} · U_{i1,i2} · U_{i2,i3} · U_{i3,i0}   (loop product)
      = 0 for cold start (all U = 𝟙 → ½ Tr U_p = 1)
      → ∞ weight suppresses large plaquette phases at large β

  MATTER (hopping / scalar QED analogue):
      S_m = −κ Σ_{(i,j) canonical}  2 Re( ψ_i† U_{ij} ψ_j )
      ψ_i = r_i · χ_i   (HGST grade magnitude × doublet)
      = −κ Σ [forward hopping + hermitian conjugate]

  TOTAL:
      S = S_g(β) + S_m(κ)

Local (delta) versions for Metropolis accept/reject:
      ΔS_g(link e, proposal U')  — changes only plaquettes containing e
      ΔS_m(link e, proposal U')  — changes only hops through e

Dependencies: su2.py, lattice.py, fields.py
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import su2
from lattice import Lattice2D
from fields import LinkDict, MatterDict


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_link(links: LinkDict, i: int, j: int) -> np.ndarray:
    """
    Retrieve U_{ij} handling canonical vs reverse direction.

    If (i,j) is canonical: return links[(i,j)]
    If (j,i) is canonical: return U_{ji}† = dagger(links[(j,i)])
    """
    if (i, j) in links:
        return links[(i, j)]
    elif (j, i) in links:
        return links[(j, i)].conj().T   # dagger; works for any unitary (SU2, SU3)
    else:
        raise KeyError(f"Edge ({i},{j}) not found in links (neither direction).")


def _plaquette_matrix(links: LinkDict, p: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Compute the ordered plaquette loop matrix U_p.

        U_p = U_{i0,i1} · U_{i1,i2} · U_{i2,i3} · U_{i3,i0}

    Each factor uses canonical lookup with automatic dagger for reverse edges.
    """
    i0, i1, i2, i3 = p
    return (
        _get_link(links, i0, i1)
        @ _get_link(links, i1, i2)
        @ _get_link(links, i2, i3)
        @ _get_link(links, i3, i0)
    )


def _plaquette_retrace(links: LinkDict, p: Tuple[int, int, int, int]) -> float:
    """(1/N) Re Tr U_p  ∈ [-1, 1].  Equals 1 for U_p = ퟙ.
    N is auto-detected from matrix shape (2 for SU2, 3 for SU3)."""
    Up = _plaquette_matrix(links, p)
    n  = Up.shape[0]
    return float(np.real(np.trace(Up))) / n


# ──────────────────────────────────────────────────────────────────────────────
# Full-action functions
# ──────────────────────────────────────────────────────────────────────────────

def gauge_action(
    links: LinkDict,
    lattice: Lattice2D,
    beta_g: float,
) -> float:
    """
    Wilson gauge action.

        S_g = β Σ_p (1 − ½ Re Tr U_p)

    Each of the N_p plaquettes contributes 0 (cold/ordered) to 2 (maximally
    disordered).  Hot-start average ≈ β × N_p.

    Parameters
    ----------
    links   : canonical-edge → SU(2) link matrix
    lattice : Lattice2D (provides plaquette list)
    beta_g  : inverse gauge coupling  (β > 0 favours ordered links)

    Returns
    -------
    S_g : float ≥ 0
    """
    total = 0.0
    for p in lattice.plaquettes():
        total += 1.0 - _plaquette_retrace(links, p)
    return beta_g * total


def matter_action(
    links: LinkDict,
    matter: MatterDict,
    lattice: Lattice2D,
    kappa: float,
) -> float:
    """
    Matter hopping action.

        S_m = −κ Σ_{(i,j) canonical}  2 Re( ψ_i† U_{ij} ψ_j )

    Summing over canonical edges automatically captures both hopping
    directions because 2 Re(z) = z + z*.

    ψ_i = r_i · χ_i  carries the HGST grade magnitude, so the hopping
    amplitude is proportional to  r_i · r_j  between neighbouring grades.

    Parameters
    ----------
    links   : canonical-edge → SU(2) link
    matter  : site → ψ_i  (shape (2,) complex array, |ψ_i| = r_i)
    lattice : Lattice2D
    kappa   : hopping parameter (κ > 0 favours aligned matter fields)

    Returns
    -------
    S_m : float  (negative when fields are aligned — energetically favoured)
    """
    total = 0.0
    for (i, j), Uij in links.items():
        psi_i = matter[i]
        psi_j = matter[j]
        hop = float(np.real(psi_i.conj() @ Uij @ psi_j))
        total += 2.0 * hop
    return -kappa * total


def total_action(
    links: LinkDict,
    matter: MatterDict,
    lattice: Lattice2D,
    beta_g: float,
    kappa: float,
) -> float:
    """
    S = S_g(β) + S_m(κ).

    Convenience wrapper used by simulation.py for debugging / energy tracking.
    The Metropolis updater uses the cheaper local (delta) variants.
    """
    return gauge_action(links, lattice, beta_g) + matter_action(links, matter, lattice, kappa)


# ──────────────────────────────────────────────────────────────────────────────
# Local (delta) functions for Metropolis
# ──────────────────────────────────────────────────────────────────────────────

def delta_gauge_action(
    links: LinkDict,
    lattice: Lattice2D,
    edge: Tuple[int, int],
    U_new: np.ndarray,
    beta_g: float,
) -> float:
    """
    Change in gauge action when link `edge` is replaced by `U_new`.

        ΔS_g = β Σ_{p ∋ edge} [ retrace(U_p_new) - retrace(U_p_old) ] × (−1)
             = −β Σ_{p ∋ edge} [ Δ retrace(p) ]

    Only plaquettes that share `edge` contribute.  Uses Lattice2D.plaquettes_of_edge.

    Parameters
    ----------
    links   : current link configuration (NOT mutated)
    lattice : Lattice2D
    edge    : canonical edge (i, j) being proposed for update
    U_new   : proposed SU(2) matrix for this link
    beta_g  : inverse gauge coupling

    Returns
    -------
    ΔS_g : float   (negative → accept proposal lowers action)
    """
    affected_idx = lattice.plaquettes_of_edge(edge)
    all_plaquettes = lattice.plaquettes()
    affected = [all_plaquettes[idx] for idx in affected_idx]

    # Sum re-traces with old and new link
    retrace_old = sum(_plaquette_retrace(links, p) for p in affected)

    # Temporarily substitute U_new
    links_tmp = dict(links)      # shallow copy — only modifies one entry
    links_tmp[edge] = U_new
    retrace_new = sum(_plaquette_retrace(links_tmp, p) for p in affected)

    return -beta_g * (retrace_new - retrace_old)


def delta_matter_action(
    links: LinkDict,
    matter: MatterDict,
    edge: Tuple[int, int],
    U_new: np.ndarray,
    kappa: float,
) -> float:
    """
    Change in matter hopping action when link `edge` is replaced by `U_new`.

    Only the term for `edge` itself changes:
        ΔS_m = −κ · 2 Re( ψ_i† U_new ψ_j )  −  (−κ · 2 Re( ψ_i† U_old ψ_j ))
             = −2κ Re( ψ_i† (U_new − U_old) ψ_j )

    Parameters
    ----------
    links   : current configuration (NOT mutated)
    matter  : matter fields
    edge    : canonical edge (i, j) being proposed
    U_new   : proposed SU(2) matrix
    kappa   : hopping parameter

    Returns
    -------
    ΔS_m : float
    """
    i, j = edge
    psi_i = matter[i]
    psi_j = matter[j]
    hop_old = float(np.real(psi_i.conj() @ links[edge] @ psi_j))
    hop_new = float(np.real(psi_i.conj() @ U_new       @ psi_j))
    return -kappa * 2.0 * (hop_new - hop_old)


def delta_action_link(
    links: LinkDict,
    matter: MatterDict,
    lattice: Lattice2D,
    edge: Tuple[int, int],
    U_new: np.ndarray,
    beta_g: float,
    kappa: float,
) -> float:
    """
    Combined ΔS = ΔS_g + ΔS_m for a single link update.
    Convenience wrapper used by updates.py MetropolisUpdater.
    """
    dg = delta_gauge_action(links, lattice, edge, U_new, beta_g)
    dm = delta_matter_action(links, matter, edge, U_new, kappa)
    return dg + dm


def delta_matter_action_site(
    links: LinkDict,
    matter: MatterDict,
    lattice: Lattice2D,
    site: int,
    psi_new: np.ndarray,
    kappa: float,
) -> float:
    """
    Change in matter action when site `site` matter field is replaced by `psi_new`.

    Affected terms: all canonical edges adjacent to `site`.

        ΔS_m = −κ Σ_{j ~ site} 2 Re( ψ_new† U_{site,j} ψ_j )
                               − 2 Re( ψ_old† U_{site,j} ψ_j )

    Parameters
    ----------
    links   : current link configuration
    matter  : current matter fields  (NOT mutated)
    lattice : Lattice2D (provides neighbour list)
    site    : site index being proposed for update
    psi_new : proposed matter field ψ'_site
    kappa   : hopping parameter

    Returns
    -------
    ΔS_m : float
    """
    psi_old = matter[site]
    delta = 0.0

    for (i, j), Uij in links.items():
        if i == site:
            # canonical edge goes site→j: term = 2 Re(ψ_site† U_{site,j} ψ_j)
            psi_nb = matter[j]
            hop_old = float(np.real(psi_old.conj() @ Uij @ psi_nb))
            hop_new = float(np.real(psi_new.conj() @ Uij @ psi_nb))
            delta += 2.0 * (hop_new - hop_old)
        elif j == site:
            # canonical edge goes i→site: term = 2 Re(ψ_i† U_{i,site} ψ_site)
            psi_nb = matter[i]
            hop_old = float(np.real(psi_nb.conj() @ Uij @ psi_old))
            hop_new = float(np.real(psi_nb.conj() @ Uij @ psi_new))
            delta += 2.0 * (hop_new - hop_old)

    return -kappa * delta


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def plaquette_average(links: LinkDict, lattice: Lattice2D) -> float:
    """
    Mean plaquette  ⟨½ Re Tr U_p⟩  ∈ [−1, 1].

    Useful thermalization diagnostic:
      cold start  → 1.0
      hot start   → ~0.0  (random SU(2), E[½ Tr] = 0)
      ordered phase → approaches 1.0 as β → ∞
    """
    vals = [_plaquette_retrace(links, p) for p in lattice.plaquettes()]
    return float(np.mean(vals))


def action_density(
    links: LinkDict,
    matter: MatterDict,
    lattice: Lattice2D,
    beta_g: float,
    kappa: float,
) -> dict:
    """
    Return a dict of per-component action densities (per site).

        s_g = S_g / N_p
        s_m = S_m / N_edges
        s   = S   / N_sites
    """
    Sg = gauge_action(links, lattice, beta_g)
    Sm = matter_action(links, matter, lattice, kappa)
    return {
        "S_g":           Sg,
        "S_m":           Sm,
        "S_total":       Sg + Sm,
        "s_g_per_plaq":  Sg / max(len(lattice.plaquettes()), 1),
        "s_m_per_edge":  Sm / max(lattice.n_edges, 1),
        "s_per_site":    (Sg + Sm) / max(lattice.N, 1),
        "plaq_avg":      plaquette_average(links, lattice),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    from fields import initialize_links, initialize_matter, gauge_transform, random_gauge

    rng  = np.random.default_rng(1234)
    PASS, FAIL = "PASS", "FAIL"
    results: list[tuple[str, bool, str]] = []

    lat = Lattice2D(4)
    beta_g = 1.5
    kappa  = 0.5

    # ── T1: Cold start → S_g = 0 ─────────────────────────────────────────
    links_cold   = initialize_links(lat, random=False)
    matter_cold  = initialize_matter(lat, random=False)
    Sg_cold = gauge_action(links_cold, lat, beta_g)
    ok1 = abs(Sg_cold) < 1e-12
    results.append(("T1  cold start → S_g = 0", ok1,
                    f"S_g(cold) = {Sg_cold:.4e}"))

    # ── T2: Cold start plaquette average = 1 ─────────────────────────────
    p_avg_cold = plaquette_average(links_cold, lat)
    ok2 = abs(p_avg_cold - 1.0) < 1e-12
    results.append(("T2  cold start → ⟨½TrU_p⟩ = 1", ok2,
                    f"plaq_avg = {p_avg_cold:.10f}"))

    # ── T3: Hot start → S_g > 0, plaquette avg ≈ 0 ───────────────────────
    links_hot   = initialize_links(lat, random=True)
    matter_hot  = initialize_matter(lat, random=True)
    Sg_hot  = gauge_action(links_hot, lat, beta_g)
    p_avg_h = plaquette_average(links_hot, lat)
    ok3 = Sg_hot > 0 and abs(p_avg_h) < 0.6
    results.append(("T3  hot start → S_g > 0, ⟨½TrU_p⟩ ≈ 0", ok3,
                    f"S_g={Sg_hot:.4f}, plaq_avg={p_avg_h:.4f}"))

    # ── T4: Gauge invariance of S_g ───────────────────────────────────────
    V = random_gauge(lat)
    links_t, matter_t = gauge_transform(links_hot, matter_hot, lat, V)
    Sg_t = gauge_action(links_t, lat, beta_g)
    ok4 = abs(Sg_t - Sg_hot) < 1e-8
    results.append(("T4  S_g gauge invariant", ok4,
                    f"|ΔS_g| = {abs(Sg_t - Sg_hot):.2e}"))

    # ── T5: Gauge invariance of S_m ───────────────────────────────────────
    Sm_orig = matter_action(links_hot, matter_hot, lat, kappa)
    Sm_t    = matter_action(links_t,   matter_t,   lat, kappa)
    ok5 = abs(Sm_t - Sm_orig) < 1e-8
    results.append(("T5  S_m gauge invariant", ok5,
                    f"|ΔS_m| = {abs(Sm_t - Sm_orig):.2e}"))

    # ── T6: ΔS_g matches full recompute ──────────────────────────────────
    test_edge = lat.edges()[3]
    U_new = su2.random_su2()
    Sg_before = gauge_action(links_hot, lat, beta_g)
    dSg_local = delta_gauge_action(links_hot, lat, test_edge, U_new, beta_g)
    links_updated = dict(links_hot)
    links_updated[test_edge] = U_new
    Sg_after = gauge_action(links_updated, lat, beta_g)
    dSg_full = Sg_after - Sg_before
    ok6 = abs(dSg_local - dSg_full) < 1e-10
    results.append(("T6  ΔS_g(local) matches full recompute", ok6,
                    f"local={dSg_local:.8f}, full={dSg_full:.8f}, "
                    f"err={abs(dSg_local-dSg_full):.2e}"))

    # ── T7: ΔS_m(link) matches full recompute ────────────────────────────
    Sm_before = matter_action(links_hot, matter_hot, lat, kappa)
    dSm_local = delta_matter_action(links_hot, matter_hot, test_edge, U_new, kappa)
    # links_updated already has U_new for test_edge
    Sm_after  = matter_action(links_updated, matter_hot, lat, kappa)
    dSm_full  = Sm_after - Sm_before
    ok7 = abs(dSm_local - dSm_full) < 1e-10
    results.append(("T7  ΔS_m(link) matches full recompute", ok7,
                    f"local={dSm_local:.8f}, full={dSm_full:.8f}, "
                    f"err={abs(dSm_local-dSm_full):.2e}"))

    # ── T8: ΔS_m(site) matches full recompute ────────────────────────────
    test_site = 7
    r_site    = lat.r[test_site]
    # Random new doublet, scale by r
    chi_new = rng.standard_normal(4).view(np.complex128)
    chi_new /= np.linalg.norm(chi_new)
    psi_new = r_site * chi_new
    Sm_before2 = matter_action(links_hot, matter_hot, lat, kappa)
    dSm_site_local = delta_matter_action_site(links_hot, matter_hot, lat,
                                              test_site, psi_new, kappa)
    matter_updated = dict(matter_hot)
    matter_updated[test_site] = psi_new
    Sm_after2 = matter_action(links_hot, matter_updated, lat, kappa)
    dSm_site_full = Sm_after2 - Sm_before2
    ok8 = abs(dSm_site_local - dSm_site_full) < 1e-10
    results.append(("T8  ΔS_m(site) matches full recompute", ok8,
                    f"local={dSm_site_local:.8f}, full={dSm_site_full:.8f}, "
                    f"err={abs(dSm_site_local-dSm_site_full):.2e}"))

    # ── T9: total_action = S_g + S_m ─────────────────────────────────────
    S_tot = total_action(links_hot, matter_hot, lat, beta_g, kappa)
    S_sum = gauge_action(links_hot, lat, beta_g) + matter_action(links_hot, matter_hot, lat, kappa)
    ok9 = abs(S_tot - S_sum) < 1e-14
    results.append(("T9  total_action = S_g + S_m", ok9,
                    f"diff = {abs(S_tot - S_sum):.2e}"))

    # ── T10: action_density keys and plaq_avg consistency ────────────────
    ad = action_density(links_hot, matter_hot, lat, beta_g, kappa)
    keys_ok  = {"S_g", "S_m", "S_total", "plaq_avg"}.issubset(ad.keys())
    plaq_ok  = abs(ad["plaq_avg"] - p_avg_h) < 1e-12
    ok10 = keys_ok and plaq_ok
    results.append(("T10 action_density: keys and plaq_avg consistent", ok10,
                    f"plaq_avg diff={abs(ad['plaq_avg']-p_avg_h):.2e}"))

    # ── Print ─────────────────────────────────────────────────────────────
    print("=" * 66)
    print("action.py — Wilson Gauge + Matter Hopping Self-Test")
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
        ad_c = action_density(links_cold, matter_cold, lat, beta_g, kappa)
        ad_h = action_density(links_hot,  matter_hot,  lat, beta_g, kappa)
        print("  ACTION SUMMARY (L=4, β=1.5, κ=0.5):")
        print(f"    Cold: S_g={ad_c['S_g']:.4f}  S_m={ad_c['S_m']:.4f}"
              f"  plaq_avg={ad_c['plaq_avg']:.4f}")
        print(f"    Hot:  S_g={ad_h['S_g']:.4f}  S_m={ad_h['S_m']:.4f}"
              f"  plaq_avg={ad_h['plaq_avg']:.4f}")
        print()
        print("  S_g ≈ 0 for cold (all plaquettes = 1)")
        print("  S_g ≈ β × N_p for random links (plaq_avg ≈ 0)")
        print(f"  N_plaquettes = {len(lat.plaquettes())},  β × N_p = {beta_g*len(lat.plaquettes()):.1f}")
    else:
        print("  SOME TESTS FAILED — review before proceeding to updates.py")
    print("=" * 66)


if __name__ == "__main__":
    _run_tests()
