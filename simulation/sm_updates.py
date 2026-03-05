#!/usr/bin/env python3
"""
sm_updates.py - Metropolis Monte Carlo sweeps for SU(3)×SU(2)×U(1) gauge + matter theory.
===========================================================================================
Provides:
  • SweepStats     — per-sweep acceptance statistics
  • SMUpdater      — Metropolis updater (links, quarks, leptons)
                     with auto-tuning to target acceptance rate ~50%

All proposals maintain the field constraints:
  • Links: U_ij ∈ SU(3)×SU(2)×U(1) exactly (via group multiplication)
  • Quarks: |ψ_i| = r_i (via projection to 6-component unit sphere × r_i)
  • Leptons: |l_i| = r_i (via projection to 2-component unit circle × r_i)

Epistemic status: VALIDATED after test suite passes.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import dataclasses
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lattice import Lattice2D
from sm_gauge import SMGaugeElement
from sm_fields import QuarkDoublet, LeptonDoublet
from sm_action import (sm_total_action, sm_delta_action_link,
                        sm_delta_action_quark, sm_delta_action_lepton,
                        sm_gauge_action)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SweepStats:
    """Acceptance statistics per sweep."""
    n_link_proposed:    int = 0
    n_link_accepted:    int = 0
    n_quark_proposed:   int = 0
    n_quark_accepted:   int = 0
    n_lepton_proposed:  int = 0
    n_lepton_accepted:  int = 0

    @property
    def link_rate(self) -> float:
        return (self.n_link_accepted / self.n_link_proposed
                if self.n_link_proposed else 0.0)

    @property
    def quark_rate(self) -> float:
        return (self.n_quark_accepted / self.n_quark_proposed
                if self.n_quark_proposed else 0.0)

    @property
    def lepton_rate(self) -> float:
        return (self.n_lepton_accepted / self.n_lepton_proposed
                if self.n_lepton_proposed else 0.0)

    def __add__(self, other: 'SweepStats') -> 'SweepStats':
        return SweepStats(
            n_link_proposed=self.n_link_proposed + other.n_link_proposed,
            n_link_accepted=self.n_link_accepted + other.n_link_accepted,
            n_quark_proposed=self.n_quark_proposed + other.n_quark_proposed,
            n_quark_accepted=self.n_quark_accepted + other.n_quark_accepted,
            n_lepton_proposed=self.n_lepton_proposed + other.n_lepton_proposed,
            n_lepton_accepted=self.n_lepton_accepted + other.n_lepton_accepted,
        )


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

class SMUpdater:
    """
    Metropolis updater for SU(3)×SU(2)×U(1) gauge + matter system.

    Parameters
    ----------
    lattice : Lattice2D
    links   : dict (i,j) -> SMGaugeElement  — mutated in-place
    quarks  : dict site -> QuarkDoublet      — mutated in-place
    leptons : dict site -> LeptonDoublet     — mutated in-place
    beta_3, beta_2, beta_1 : float — gauge couplings (SU3, SU2, U1)
    kappa_q, kappa_l : float       — quark/lepton hopping parameters
    eps_link, eps_quark, eps_lepton : float — initial proposal step sizes
    seed : int — RNG seed
    target_rate : float — desired acceptance rate for auto-tuning (default 0.5)
    """

    def __init__(
        self,
        lattice: Lattice2D,
        links: Dict[Tuple[int, int], SMGaugeElement],
        quarks: Dict[int, QuarkDoublet],
        leptons: Dict[int, LeptonDoublet],
        beta_3: float,
        beta_2: float,
        beta_1: float,
        kappa_q: float,
        kappa_l: float,
        eps_link: float = 0.3,
        eps_quark: float = 0.3,
        eps_lepton: float = 0.3,
        seed: int = 0,
        target_rate: float = 0.5
    ):
        self.lattice  = lattice
        self.links    = links
        self.quarks   = quarks
        self.leptons  = leptons
        self.beta_3   = beta_3
        self.beta_2   = beta_2
        self.beta_1   = beta_1
        self.kappa_q  = kappa_q
        self.kappa_l  = kappa_l
        self.eps_link  = eps_link
        self.eps_quark = eps_quark
        self.eps_lepton= eps_lepton
        self.rng = np.random.default_rng(seed)
        self.target_rate = target_rate
        self._edges = list(lattice.edges())

    # -----------------------------------------------------------------------
    # Proposals
    # -----------------------------------------------------------------------

    def _propose_link(self, edge: Tuple[int, int]) -> SMGaugeElement:
        """Propose U' = δU · U_old with δU a small random element."""
        delta = SMGaugeElement.small_random(self.eps_link, self.rng)
        return delta @ self.links[edge]

    def _propose_quark(self, site: int) -> QuarkDoublet:
        """
        Propose new quark field by small random perturbation of the
        normalised 6-component spinor, maintaining |ψ| = r_i.
        """
        q_old = self.quarks[site]
        r_i = self.lattice.r[site]
        if r_i == 0:
            return q_old
        chi6 = q_old.chi(r_i)          # unit 6-component spinor
        noise = self.rng.normal(0, self.eps_quark, 12).view(np.complex128)
        chi6_new = chi6 + noise
        chi6_new /= np.linalg.norm(chi6_new)   # re-normalise on 6-sphere
        return QuarkDoublet.from_chi(chi6_new, r_i)

    def _propose_lepton(self, site: int) -> LeptonDoublet:
        """
        Propose new lepton field maintaining |l| = r_i.
        """
        l_old = self.leptons[site]
        r_i = self.lattice.r[site]
        if r_i == 0:
            return l_old
        chi2 = l_old.chi(r_i)          # unit 2-component spinor
        noise = self.rng.normal(0, self.eps_lepton, 4).view(np.complex128)
        chi2_new = chi2 + noise
        chi2_new /= np.linalg.norm(chi2_new)
        return LeptonDoublet.from_chi(chi2_new, r_i)

    # -----------------------------------------------------------------------
    # Single Metropolis steps
    # -----------------------------------------------------------------------

    def update_link(self, edge: Tuple[int, int]) -> bool:
        """One Metropolis step for the link at `edge`."""
        U_new = self._propose_link(edge)
        dS = sm_delta_action_link(
            self.links, self.lattice, edge, U_new,
            self.beta_3, self.beta_2, self.beta_1
        )
        if dS <= 0 or self.rng.random() < np.exp(-dS):
            self.links[edge] = U_new
            return True
        return False

    def update_quark(self, site: int) -> bool:
        """One Metropolis step for the quark at `site`."""
        q_new = self._propose_quark(site)
        dS = sm_delta_action_quark(
            self.links, self.quarks, self.lattice, site, q_new, self.kappa_q
        )
        if dS <= 0 or self.rng.random() < np.exp(-dS):
            self.quarks[site] = q_new
            return True
        return False

    def update_lepton(self, site: int) -> bool:
        """One Metropolis step for the lepton at `site`."""
        l_new = self._propose_lepton(site)
        dS = sm_delta_action_lepton(
            self.links, self.leptons, self.lattice, site, l_new, self.kappa_l
        )
        if dS <= 0 or self.rng.random() < np.exp(-dS):
            self.leptons[site] = l_new
            return True
        return False

    # -----------------------------------------------------------------------
    # Sweeps
    # -----------------------------------------------------------------------

    def sweep(
        self,
        update_links: bool = True,
        update_quarks: bool = True,
        update_leptons: bool = True
    ) -> SweepStats:
        """One complete sweep over all degrees of freedom."""
        stats = SweepStats()

        if update_links:
            for edge in self._edges:
                stats.n_link_proposed += 1
                if self.update_link(edge):
                    stats.n_link_accepted += 1

        if update_quarks:
            for site in range(self.lattice.N):
                stats.n_quark_proposed += 1
                if self.update_quark(site):
                    stats.n_quark_accepted += 1

        if update_leptons:
            for site in range(self.lattice.N):
                stats.n_lepton_proposed += 1
                if self.update_lepton(site):
                    stats.n_lepton_accepted += 1

        return stats

    def thermalize(
        self,
        n_sweeps: int,
        update_links: bool = True,
        update_quarks: bool = True,
        update_leptons: bool = True,
        tune_every: int = 50
    ) -> SweepStats:
        """Run thermalization sweeps with periodic auto-tuning."""
        cumulative = SweepStats()
        window     = SweepStats()

        for i in range(1, n_sweeps + 1):
            s = self.sweep(update_links=update_links,
                           update_quarks=update_quarks,
                           update_leptons=update_leptons)
            cumulative += s
            window     += s

            if i % tune_every == 0:
                self._auto_tune(window)
                window = SweepStats()

        return cumulative

    # -----------------------------------------------------------------------
    # Auto-tuning
    # -----------------------------------------------------------------------

    def _auto_tune(self, window: SweepStats):
        """Adjust step-sizes toward target acceptance rate."""
        factor = 1.1

        if window.n_link_proposed > 0:
            self.eps_link *= factor if window.link_rate > self.target_rate else 1.0 / factor
            self.eps_link = float(np.clip(self.eps_link, 0.01, np.pi))

        if window.n_quark_proposed > 0:
            self.eps_quark *= factor if window.quark_rate > self.target_rate else 1.0 / factor
            self.eps_quark = float(np.clip(self.eps_quark, 0.01, 2.0))

        if window.n_lepton_proposed > 0:
            self.eps_lepton *= factor if window.lepton_rate > self.target_rate else 1.0 / factor
            self.eps_lepton = float(np.clip(self.eps_lepton, 0.01, 2.0))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> bool:
    from sm_fields import (initialize_sm_links, initialize_quarks,
                           initialize_leptons, check_field_norms)

    PASS, FAIL = "PASS", "FAIL"
    results = []
    rng = np.random.default_rng(42)
    lat = Lattice2D(4)

    links  = initialize_sm_links(lat, random=True, rng=rng)
    quarks = initialize_quarks(lat, random=True, rng=rng)
    leptons= initialize_leptons(lat, random=True, rng=rng)

    updater = SMUpdater(
        lat, links, quarks, leptons,
        beta_3=6.0, beta_2=4.0, beta_1=2.0,
        kappa_q=0.3, kappa_l=0.3, seed=42
    )

    # T1: Sweep returns correct counts
    stats = updater.sweep()
    ok1 = (stats.n_link_proposed   == lat.n_edges and
           stats.n_quark_proposed  == lat.N and
           stats.n_lepton_proposed == lat.N)
    results.append(("T1 sweep counts correct", ok1,
                   f"links={stats.n_link_proposed}, "
                   f"quarks={stats.n_quark_proposed}, "
                   f"leptons={stats.n_lepton_proposed}"))

    # T2: All links remain valid after sweep
    ok2 = all(U.is_valid()[0] for U in links.values())
    results.append(("T2 links valid after sweep", ok2, ""))

    # T3: Quark norms preserved after sweep
    ok3, max_err3 = check_field_norms(quarks, {}, lat)
    results.append(("T3 quark norms preserved", ok3,
                   f"max_err={max_err3:.2e}"))

    # T4: Lepton norms preserved after sweep
    ok4, max_err4 = check_field_norms({}, leptons, lat)
    results.append(("T4 lepton norms preserved", ok4,
                   f"max_err={max_err4:.2e}"))

    # T5: Thermalization reduces gauge action at high beta (pure gauge)
    from sm_fields import initialize_sm_links as isl
    links2  = isl(lat, random=True, rng=rng)
    quarks2 = initialize_quarks(lat, random=True, rng=rng)
    leptons2= initialize_leptons(lat, random=True, rng=rng)
    updater2 = SMUpdater(
        lat, links2, quarks2, leptons2,
        beta_3=10.0, beta_2=8.0, beta_1=6.0,
        kappa_q=0.0, kappa_l=0.0, seed=43
    )
    Sg_before = sm_gauge_action(links2, lat, 10.0, 8.0, 6.0)
    updater2.thermalize(200, update_quarks=False, update_leptons=False)
    Sg_after = sm_gauge_action(links2, lat, 10.0, 8.0, 6.0)
    ok5 = Sg_after < Sg_before
    results.append(("T5 thermalization reduces gauge action", ok5,
                   f"S: {Sg_before:.2f} → {Sg_after:.2f}"))

    print("=" * 66)
    print("sm_updates.py — Metropolis Sweeps Self-Test")
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
        print("\n  Updater module for SM is ready.")
    else:
        print("  SOME TESTS FAILED — debug before proceeding.")
    print("=" * 66)
    return all_pass


if __name__ == "__main__":
    import sys as _sys
    success = _run_tests()
    _sys.exit(0 if success else 1)
