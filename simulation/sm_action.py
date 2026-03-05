#!/usr/bin/env python3
"""
sm_action.py - Gauge and matter actions for SU(3)×SU(2)×U(1) gauge theory on HGST lattice.
============================================================================================
Provides:
  • sm_gauge_action          — full Wilson gauge action (re-exported from sm_gauge)
  • sm_delta_action_link     — local link change (re-exported from sm_gauge)
  • sm_quark_action          — quark kinetic action
  • sm_lepton_action         — lepton kinetic action
  • sm_matter_action         — total matter action = quark + lepton
  • sm_total_action          — S_gauge + S_matter
  • sm_delta_action_quark    — local matter action change for quark update
  • sm_delta_action_lepton   — local matter action change for lepton update

All actions are gauge-invariant.

Epistemic status: VALIDATED after test suite passes.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lattice import Lattice2D
from sm_gauge import (SMGaugeElement, sm_gauge_action, sm_delta_action_link,
                      sm_plaquette_average)
from sm_fields import QuarkDoublet, LeptonDoublet


# ---------------------------------------------------------------------------
# Helper: apply U ~ SU(3)×SU(2)×U(1) to a quark doublet
# ---------------------------------------------------------------------------

def _apply_link_to_quark(U: SMGaugeElement, q: QuarkDoublet) -> QuarkDoublet:
    """Compute U|q⟩: SU(2) acts on (up,down) doublet, SU(3) acts on colour."""
    u_temp = U.su2[0, 0] * q.up + U.su2[0, 1] * q.down
    d_temp = U.su2[1, 0] * q.up + U.su2[1, 1] * q.down
    return QuarkDoublet(up=U.su3 @ u_temp, down=U.su3 @ d_temp)


def _apply_link_to_lepton(U: SMGaugeElement, l: LeptonDoublet) -> LeptonDoublet:
    """Compute U|l⟩: SU(2) acts on (ν,e) doublet; U(1) hypercharge Y=−1/2."""
    u1_fac = np.exp(-0.5j * np.angle(U.u1))
    nu_temp = U.su2[0, 0] * l.neutrino + U.su2[0, 1] * l.electron
    e_temp  = U.su2[1, 0] * l.neutrino + U.su2[1, 1] * l.electron
    return LeptonDoublet(neutrino=complex(u1_fac * nu_temp),
                         electron=complex(u1_fac * e_temp))


def _quark_inner(qi: QuarkDoublet, qj: QuarkDoublet) -> complex:
    """⟨ψ_i|ψ_j⟩ = ψ_i.up† · ψ_j.up + ψ_i.down† · ψ_j.down"""
    return (np.vdot(qi.up, qj.up) + np.vdot(qi.down, qj.down))


def _lepton_inner(li: LeptonDoublet, lj: LeptonDoublet) -> complex:
    """⟨l_i|l_j⟩ = ν_i* ν_j + e_i* e_j"""
    return (li.neutrino.conjugate() * lj.neutrino +
            li.electron.conjugate() * lj.electron)


# ---------------------------------------------------------------------------
# Matter kinetic actions
# ---------------------------------------------------------------------------

def sm_quark_action(
    links: Dict[Tuple[int, int], SMGaugeElement],
    quarks: Dict[int, QuarkDoublet],
    lattice: Lattice2D,
    kappa_q: float
) -> float:
    """
    Kinetic action for left-handed quark doublets.

        S_q = −κ_q Σ_{edges (i,j)} 2 Re⟨ψ_i | U_ij | ψ_j⟩

    The sum is over canonical directed edges only; reverse edges contribute
    via the Hermitian conjugate.
    """
    if kappa_q == 0.0:
        return 0.0
    action = 0.0
    for (i, j), U in links.items():
        qi = quarks[i]
        qj_trans = _apply_link_to_quark(U, quarks[j])
        action -= kappa_q * 2.0 * _quark_inner(qi, qj_trans).real
    return action


def sm_lepton_action(
    links: Dict[Tuple[int, int], SMGaugeElement],
    leptons: Dict[int, LeptonDoublet],
    lattice: Lattice2D,
    kappa_l: float
) -> float:
    """
    Kinetic action for left-handed lepton doublets.

        S_l = −κ_l Σ_{edges (i,j)} 2 Re⟨l_i | U_ij | l_j⟩

    U acts via SU(2) rotation plus U(1) hypercharge factor Y=−1/2.
    """
    if kappa_l == 0.0:
        return 0.0
    action = 0.0
    for (i, j), U in links.items():
        li = leptons[i]
        lj_trans = _apply_link_to_lepton(U, leptons[j])
        action -= kappa_l * 2.0 * _lepton_inner(li, lj_trans).real
    return action


def sm_matter_action(
    links: Dict[Tuple[int, int], SMGaugeElement],
    quarks: Dict[int, QuarkDoublet],
    leptons: Dict[int, LeptonDoublet],
    lattice: Lattice2D,
    kappa_q: float,
    kappa_l: float
) -> float:
    """Total matter action = S_quark + S_lepton."""
    return (sm_quark_action(links, quarks, lattice, kappa_q) +
            sm_lepton_action(links, leptons, lattice, kappa_l))


def sm_total_action(
    links: Dict[Tuple[int, int], SMGaugeElement],
    quarks: Dict[int, QuarkDoublet],
    leptons: Dict[int, LeptonDoublet],
    lattice: Lattice2D,
    beta_3: float,
    beta_2: float,
    beta_1: float,
    kappa_q: float,
    kappa_l: float
) -> float:
    """S = S_gauge + S_matter."""
    return (sm_gauge_action(links, lattice, beta_3, beta_2, beta_1) +
            sm_matter_action(links, quarks, leptons, lattice, kappa_q, kappa_l))


# ---------------------------------------------------------------------------
# Local action changes for Metropolis updates
# ---------------------------------------------------------------------------

def sm_delta_action_quark(
    links: Dict[Tuple[int, int], SMGaugeElement],
    quarks: Dict[int, QuarkDoublet],
    lattice: Lattice2D,
    site: int,
    quark_new: QuarkDoublet,
    kappa_q: float
) -> float:
    """
    Change in matter action when quark at `site` is replaced by quark_new.

    Only edges incident on `site` contribute.
    Positive return → action increases → proposal tends to be rejected.
    """
    if kappa_q == 0.0:
        return 0.0

    delta = 0.0
    quark_old = quarks[site]

    for nbr in lattice.neighbors(site):
        # Canonical edge site → nbr
        if (site, nbr) in links:
            U = links[(site, nbr)]
            qnbr_trans = _apply_link_to_quark(U, quarks[nbr])
            delta -= kappa_q * 2.0 * (
                _quark_inner(quark_new, qnbr_trans).real -
                _quark_inner(quark_old, qnbr_trans).real
            )

    # Reverse edges: nbr → site (where nbr is a canonical source)
    for (i, j), U in links.items():
        if j == site:
            # site is the target; contribution is ψ_i† U ψ_site
            qi = quarks[i]
            old_trans = _apply_link_to_quark(U, quark_old)
            new_trans = _apply_link_to_quark(U, quark_new)
            delta -= kappa_q * 2.0 * (
                _quark_inner(qi, new_trans).real -
                _quark_inner(qi, old_trans).real
            )

    return delta


def sm_delta_action_lepton(
    links: Dict[Tuple[int, int], SMGaugeElement],
    leptons: Dict[int, LeptonDoublet],
    lattice: Lattice2D,
    site: int,
    lepton_new: LeptonDoublet,
    kappa_l: float
) -> float:
    """
    Change in matter action when lepton at `site` is replaced by lepton_new.
    """
    if kappa_l == 0.0:
        return 0.0

    delta = 0.0
    lepton_old = leptons[site]

    # Forward edges: site → nbr
    for nbr in lattice.neighbors(site):
        if (site, nbr) in links:
            U = links[(site, nbr)]
            lnbr_trans = _apply_link_to_lepton(U, leptons[nbr])
            delta -= kappa_l * 2.0 * (
                _lepton_inner(lepton_new, lnbr_trans).real -
                _lepton_inner(lepton_old, lnbr_trans).real
            )

    # Reverse edges: nbr → site
    for (i, j), U in links.items():
        if j == site:
            li = leptons[i]
            old_trans = _apply_link_to_lepton(U, lepton_old)
            new_trans = _apply_link_to_lepton(U, lepton_new)
            delta -= kappa_l * 2.0 * (
                _lepton_inner(li, new_trans).real -
                _lepton_inner(li, old_trans).real
            )

    return delta


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> bool:
    from sm_fields import (initialize_sm_links, initialize_quarks,
                           initialize_leptons)

    PASS, FAIL = "PASS", "FAIL"
    results = []
    rng = np.random.default_rng(42)
    lat = Lattice2D(4)

    links  = initialize_sm_links(lat, random=True, rng=rng)
    quarks = initialize_quarks(lat, random=True, rng=rng)
    leptons= initialize_leptons(lat, random=True, rng=rng)

    b3, b2, b1  = 6.0, 4.0, 2.0
    kq, kl = 0.3, 0.3

    # T1: Gauge action finite
    Sg = sm_gauge_action(links, lat, b3, b2, b1)
    ok1 = np.isfinite(Sg)
    results.append(("T1 gauge action finite", ok1, f"S_g={Sg:.4f}"))

    # T2: Matter action finite
    Sm = sm_matter_action(links, quarks, leptons, lat, kq, kl)
    ok2 = np.isfinite(Sm)
    results.append(("T2 matter action finite", ok2, f"S_m={Sm:.4f}"))

    # T3: Total action finite
    St = sm_total_action(links, quarks, leptons, lat, b3, b2, b1, kq, kl)
    ok3 = np.isfinite(St) and abs(St - (Sg + Sm)) < 1e-10
    results.append(("T3 total = gauge + matter", ok3, f"S_tot={St:.4f}"))

    # T4: Delta link matches finite difference
    edge = list(lat.edges())[3]
    dU = SMGaugeElement.small_random(0.1, rng)
    U_new = dU @ links[edge]
    delta_ded = sm_delta_action_link(links, lat, edge, U_new, b3, b2, b1)
    links_t = dict(links); links_t[edge] = U_new
    Sg_old = sm_gauge_action(links, lat, b3, b2, b1)
    Sg_new = sm_gauge_action(links_t, lat, b3, b2, b1)
    delta_full = Sg_new - Sg_old
    ok4 = abs(delta_ded - delta_full) < 1e-9 * max(1.0, abs(delta_full))
    results.append(("T4 delta_action_link accuracy", ok4,
                   f"Δ_ded={delta_ded:.6f}, Δ_full={delta_full:.6f}"))

    # T5: Delta quark matches finite difference
    site = 2
    r_i = lat.r[site]
    chi6_old = quarks[site].chi(r_i)
    delta_chi = rng.normal(0, 0.1, 12).view(np.complex128)
    chi6_new = chi6_old + delta_chi
    chi6_new /= np.linalg.norm(chi6_new)
    from sm_fields import QuarkDoublet as QD
    q_new = QD.from_chi(chi6_new, r_i)
    delta_ded_q = sm_delta_action_quark(links, quarks, lat, site, q_new, kq)
    quarks_t = dict(quarks); quarks_t[site] = q_new
    Sm_old = sm_matter_action(links, quarks, leptons, lat, kq, kl)
    Sm_new = sm_matter_action(links, quarks_t, leptons, lat, kq, kl)
    delta_full_q = Sm_new - Sm_old
    ok5 = abs(delta_ded_q - delta_full_q) < 1e-9 * max(1.0, abs(delta_full_q))
    results.append(("T5 delta_action_quark accuracy", ok5,
                   f"Δ_ded={delta_ded_q:.6f}, Δ_full={delta_full_q:.6f}"))

    print("=" * 66)
    print("sm_action.py — Action Computations Self-Test")
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
        print("\n  Action module for SM is ready.")
    else:
        print("  SOME TESTS FAILED — debug before proceeding.")
    print("=" * 66)
    return all_pass


if __name__ == "__main__":
    import sys as _sys
    success = _run_tests()
    _sys.exit(0 if success else 1)
