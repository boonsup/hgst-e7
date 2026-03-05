"""
updates.py — Metropolis Monte Carlo Sweeps
==========================================
Layer 4 Operations Module: MCMC engine

Implements the standard Metropolis algorithm for:
  1. LINK updates    — propose U' = small_su2(ε) · U_old  (left-multiply by
                       SU(2) near-identity); accept if exp(−ΔS) > uniform[0,1)
  2. MATTER updates  — propose χ'_i = normalise(χ_i + ε·δ) where δ is a
                       small random complex 2-vector; scale by r_i

One "sweep" = one attempted update per link + one per matter site.
The class auto-tunes epsilon to maintain target acceptance rates after
each tuning sweep.

Design principles:
  - MetropolisUpdater holds refs to links and matter (mutates in-place)
  - Returns acceptance statistics each sweep for monitoring and auto-tune
  - RNG state is fully reproducible given a seed

Dependencies: su2.py, lattice.py, fields.py, action.py
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import numpy as np

import su2
import su3
from lattice import Lattice2D
from fields import LinkDict, MatterDict, random_doublet
from action import delta_action_link, delta_matter_action_site


# ──────────────────────────────────────────────────────────────────────────────
# Acceptance statistics dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SweepStats:
    """Per-sweep acceptance statistics."""
    n_link_proposed:   int = 0
    n_link_accepted:   int = 0
    n_matter_proposed: int = 0
    n_matter_accepted: int = 0

    @property
    def link_rate(self) -> float:
        if self.n_link_proposed == 0:
            return 0.0
        return self.n_link_accepted / self.n_link_proposed

    @property
    def matter_rate(self) -> float:
        if self.n_matter_proposed == 0:
            return 0.0
        return self.n_matter_accepted / self.n_matter_proposed

    def __add__(self, other: "SweepStats") -> "SweepStats":
        return SweepStats(
            n_link_proposed   = self.n_link_proposed   + other.n_link_proposed,
            n_link_accepted   = self.n_link_accepted   + other.n_link_accepted,
            n_matter_proposed = self.n_matter_proposed + other.n_matter_proposed,
            n_matter_accepted = self.n_matter_accepted + other.n_matter_accepted,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Metropolis updater class
# ──────────────────────────────────────────────────────────────────────────────

class MetropolisUpdater:
    """
    Metropolis–Hastings updater for SU(2) or SU(3) gauge + matter system.

    The updater holds *references* to the link and matter dicts and mutates
    them in-place.  No copies are made during production sweeps (only a
    shallow dict copy per-step for delta computation in action.py).

    Parameters
    ----------
    lattice     : Lattice2D
    links       : LinkDict  (mutated in-place)
    matter      : MatterDict (mutated in-place)
    beta_g      : inverse gauge coupling
    kappa       : matter hopping parameter
    gauge_group : 'SU2' (default) or 'SU3'
    eps_link    : initial step size for link proposals (default 0.3)
    eps_matter  : initial step size for matter proposals (default 0.5)
    seed        : RNG seed for reproducibility
    target_rate : target acceptance rate for auto-tune (default 0.5)
    """

    def __init__(
        self,
        lattice:    Lattice2D,
        links:      LinkDict,
        matter:     MatterDict,
        beta_g:     float,
        kappa:      float,
        gauge_group: str   = 'SU2',
        eps_link:   float = 0.3,
        eps_matter: float = 0.5,
        seed:       int   = 0,
        target_rate: float = 0.5,
    ) -> None:
        self.lattice    = lattice
        self.links      = links
        self.matter     = matter
        self.beta_g     = beta_g
        self.kappa      = kappa
        self.gauge_group = gauge_group.upper()
        self.dim        = 3 if self.gauge_group == 'SU3' else 2
        self.eps_link   = eps_link
        self.eps_matter = eps_matter
        self.rng        = np.random.default_rng(seed)
        self.target_rate = target_rate

        # Precompute canonical edge list for sweep ordering
        self._edges = list(lattice.edges())

    # ── Single-step updaters ─────────────────────────────────────────────

    def _propose_link(self, edge: Tuple[int, int]) -> np.ndarray:
        """
        Propose U'_{ij} = δU · U_{ij}  where δU is a small near-identity
        gauge-group element (SU2 or SU3, dispatched from self.gauge_group).
        """
        if self.gauge_group == 'SU3':
            delta_U = su3.small_random_su3(self.eps_link, self.rng)
        else:
            delta_U = su2.small_random_su2(self.eps_link)
        return delta_U @ self.links[edge]

    def _propose_matter(self, site: int) -> np.ndarray:
        """
        Propose ψ'_i = r_i · normalise( χ_i + ε · η )
        where η is a small random Gaussian complex 2-vector.

        The normalisation ensures |ψ'_i| = r_i (unit doublet scaled by grade).
        """
        psi_old = self.matter[site]
        r_i     = self.lattice.r[site]
        # Avoid division by zero for zero-grade sites (shouldn't occur for n≥1,m≥1)
        if r_i < 1e-15:
            return psi_old.copy()

        chi_old = psi_old / r_i                          # normalised spinor
        eta     = self.rng.standard_normal(2 * self.dim).view(np.complex128)  # shape (dim,)
        eta    /= np.linalg.norm(eta)
        chi_new = chi_old + self.eps_matter * eta
        chi_new /= np.linalg.norm(chi_new)
        return r_i * chi_new

    def update_link(self, edge: Tuple[int, int]) -> bool:
        """
        One Metropolis step for a single link.

        Returns True if the proposal was accepted.
        """
        U_new = self._propose_link(edge)
        dS = delta_action_link(
            self.links, self.matter, self.lattice,
            edge, U_new, self.beta_g, self.kappa,
        )
        if dS <= 0.0 or self.rng.random() < np.exp(-dS):
            self.links[edge] = U_new
            return True
        return False

    def update_site(self, site: int) -> bool:
        """
        One Metropolis step for a single matter site.

        Returns True if the proposal was accepted.
        """
        psi_new = self._propose_matter(site)
        dS = delta_matter_action_site(
            self.links, self.matter, self.lattice,
            site, psi_new, self.kappa,
        )
        if dS <= 0.0 or self.rng.random() < np.exp(-dS):
            self.matter[site] = psi_new
            return True
        return False

    # ── Full sweep ──────────────────────────────────────────────────────

    def sweep(
        self,
        update_links:  bool = True,
        update_matter: bool = True,
    ) -> SweepStats:
        """
        Execute one complete sweep: visit each link and each site once.

        Link order: canonical edge ordering (deterministic).
        Matter order: sequential site 0..N-1.

        Parameters
        ----------
        update_links  : if False, skip all link updates
        update_matter : if False, skip all matter updates

        Returns
        -------
        SweepStats with acceptance counts for this sweep.
        """
        stats = SweepStats()

        if update_links:
            for edge in self._edges:
                stats.n_link_proposed += 1
                if self.update_link(edge):
                    stats.n_link_accepted += 1

        if update_matter:
            for site in range(self.lattice.N):
                stats.n_matter_proposed += 1
                if self.update_site(site):
                    stats.n_matter_accepted += 1

        return stats

    def thermalize(
        self,
        n_sweeps: int,
        update_links:  bool = True,
        update_matter: bool = True,
        tune_every:    int  = 50,
    ) -> SweepStats:
        """
        Run n_sweeps thermalization sweeps discarding measurements.

        Auto-tunes eps_link and eps_matter every `tune_every` sweeps to
        maintain target acceptance rate.

        Returns cumulative SweepStats over all thermalization sweeps.
        """
        cumulative = SweepStats()
        window     = SweepStats()

        for i in range(1, n_sweeps + 1):
            s = self.sweep(update_links=update_links, update_matter=update_matter)
            cumulative = cumulative + s
            window     = window     + s

            if i % tune_every == 0:
                self._auto_tune(window)
                window = SweepStats()  # reset window

        return cumulative

    # ── Auto-tune ────────────────────────────────────────────────────────

    def _auto_tune(self, window: SweepStats) -> None:
        """
        Adjust eps_link and eps_matter to move acceptance rates toward target.

        Rule:
          acceptance > target  → increase ε by factor 1.1  (larger steps)
          acceptance < target  → decrease ε by factor 0.9  (smaller steps)

        Clamps: eps_link ∈ [0.01, π],  eps_matter ∈ [0.01, 2.0]
        """
        if window.n_link_proposed > 0:
            rate = window.link_rate
            if rate > self.target_rate:
                self.eps_link *= 1.1
            else:
                self.eps_link *= 0.9
            self.eps_link = float(np.clip(self.eps_link, 0.01, np.pi))

        if window.n_matter_proposed > 0:
            rate = window.matter_rate
            if rate > self.target_rate:
                self.eps_matter *= 1.1
            else:
                self.eps_matter *= 0.9
            self.eps_matter = float(np.clip(self.eps_matter, 0.01, 2.0))


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    from fields import initialize_links, initialize_matter
    from action import gauge_action, total_action, plaquette_average

    rng_seed = 42
    PASS, FAIL = "PASS", "FAIL"
    results: list[tuple[str, bool, str]] = []

    lat    = Lattice2D(4)
    beta_g = 2.0
    kappa  = 0.3

    # ── T1: SweepStats arithmetic and properties ─────────────────────────
    s1 = SweepStats(10, 5, 8, 4)
    s2 = SweepStats(10, 7, 8, 6)
    s3 = s1 + s2
    ok1 = (s3.n_link_proposed == 20 and
           s3.link_rate == pytest_approx(0.6) and
           s3.matter_rate == pytest_approx(0.625))
    results.append(("T1  SweepStats arithmetic", ok1,
                    f"link_rate={s3.link_rate:.3f}, matter_rate={s3.matter_rate:.3f}"))

    # ── T2: Single sweep runs without error; stats sane ──────────────────
    links  = initialize_links(lat, random=True)
    matter = initialize_matter(lat, random=True)
    upd = MetropolisUpdater(lat, links, matter, beta_g, kappa, seed=rng_seed)
    stats = upd.sweep()
    ok2 = (stats.n_link_proposed   == lat.n_edges and
           stats.n_matter_proposed == lat.N and
           0 <= stats.link_rate   <= 1.0 and
           0 <= stats.matter_rate <= 1.0)
    results.append(("T2  sweep() stat counts correct", ok2,
                    f"links_prop={stats.n_link_proposed}, "
                    f"matter_prop={stats.n_matter_proposed}, "
                    f"link_acc={stats.link_rate:.3f}, "
                    f"matter_acc={stats.matter_rate:.3f}"))

    # ── T3: All links remain SU(2) after sweep ───────────────────────────
    ok3 = all(su2.is_su2(U) for U in links.values())
    results.append(("T3  all links ∈ SU(2) after sweep", ok3,
                    f"n_links={len(links)}"))

    # ── T4: All matter norms preserved |ψ_i| = r_i ──────────────────────
    norm_errs = [abs(np.linalg.norm(matter[s]) - lat.r[s]) for s in range(lat.N)]
    ok4 = max(norm_errs) < 1e-12
    results.append(("T4  |ψ_i| = r_i after sweep", ok4,
                    f"max norm error = {max(norm_errs):.2e}"))

    # ── T5: Thermalisation reduces action (hot→cold with large β) ────────
    # Use a high beta to strongly prefer ordered phase
    lat5   = Lattice2D(4)
    beta5  = 8.0
    links5 = initialize_links(lat5, random=True)
    mat5   = initialize_matter(lat5, random=True)
    Sg_hot = gauge_action(links5, lat5, beta5)
    upd5 = MetropolisUpdater(lat5, links5, mat5, beta5, kappa=0.0, seed=7)
    upd5.thermalize(500, update_matter=False)
    Sg_warm = gauge_action(links5, lat5, beta5)
    ok5 = Sg_warm < Sg_hot
    results.append(("T5  thermalisation reduces S_g at β=8 (links only)", ok5,
                    f"S_g: hot={Sg_hot:.2f} → warm={Sg_warm:.2f}  "
                    f"Δ={Sg_hot-Sg_warm:.2f}"))

    # ── T6: Plaquette average increases toward 1 during thermalisation ───
    p_hot  = plaquette_average(
        initialize_links(lat5, random=True), lat5
    )
    p_warm = plaquette_average(links5, lat5)
    ok6 = p_warm > p_hot
    results.append(("T6  ⟨½TrU_p⟩ increases after thermalisation at β=8", ok6,
                    f"plaq_avg: before≈{p_hot:.3f}, after={p_warm:.3f}"))

    # ── T7: Auto-tune adjusts eps ────────────────────────────────────────
    lat7 = Lattice2D(4)
    lk7  = initialize_links(lat7, random=True)
    mt7  = initialize_matter(lat7, random=True)
    upd7 = MetropolisUpdater(lat7, lk7, mt7, beta_g=0.5, kappa=0.1,
                             eps_link=0.01, eps_matter=0.01, seed=99)
    eps_before = upd7.eps_link
    upd7.thermalize(200, tune_every=50)
    eps_after = upd7.eps_link
    # At β=0.5 (weakly coupled) with tiny ε, acceptance is high → ε should grow
    ok7 = eps_after > eps_before
    results.append(("T7  auto-tune grows eps_link for high acceptance", ok7,
                    f"eps_link: {eps_before:.4f} → {eps_after:.4f}"))

    # ── T8: link-only sweep → matter fields unchanged ────────────────────
    lat8  = Lattice2D(3)
    lk8   = initialize_links(lat8, random=True)
    mt8   = initialize_matter(lat8, random=True)
    mt8_copy = {s: v.copy() for s, v in mt8.items()}
    upd8  = MetropolisUpdater(lat8, lk8, mt8, beta_g=1.0, kappa=0.2, seed=5)
    upd8.sweep(update_links=True, update_matter=False)
    matter_unchanged = all(
        np.allclose(mt8[s], mt8_copy[s]) for s in range(lat8.N)
    )
    ok8 = matter_unchanged
    results.append(("T8  link-only sweep: matter unchanged", ok8,
                    "update_matter=False → no matter touched"))

    # ── T9: matter-only sweep → links unchanged ──────────────────────────
    lk8_copy = {e: U.copy() for e, U in lk8.items()}
    upd8b = MetropolisUpdater(lat8, lk8, mt8, beta_g=1.0, kappa=0.2, seed=6)
    upd8b.sweep(update_links=False, update_matter=True)
    links_unchanged = all(
        np.allclose(lk8[e], lk8_copy[e]) for e in lat8.edges()
    )
    ok9 = links_unchanged
    results.append(("T9  matter-only sweep: links unchanged", ok9,
                    "update_links=False → no links touched"))

    # ── T10: thermalize returns cumulative stats ─────────────────────────
    lat10 = Lattice2D(3)
    lk10  = initialize_links(lat10, random=True)
    mt10  = initialize_matter(lat10, random=True)
    upd10 = MetropolisUpdater(lat10, lk10, mt10, beta_g=1.0, kappa=0.2, seed=3)
    n_therm = 20
    cum = upd10.thermalize(n_therm)
    expected_link_props   = n_therm * lat10.n_edges
    expected_matter_props = n_therm * lat10.N
    ok10 = (cum.n_link_proposed   == expected_link_props and
            cum.n_matter_proposed == expected_matter_props)
    results.append(("T10 thermalize cumulative proposal counts", ok10,
                    f"link={cum.n_link_proposed} (exp {expected_link_props}), "
                    f"matter={cum.n_matter_proposed} (exp {expected_matter_props})"))

    # ── Print ─────────────────────────────────────────────────────────────
    print("=" * 66)
    print("updates.py — Metropolis Sweeps Self-Test")
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
        print("  UPDATER SUMMARY (L=4, β=2.0, κ=0.3, 1 sweep):")
        print(f"    Link acceptance rate:   {stats.link_rate:.3f}")
        print(f"    Matter acceptance rate: {stats.matter_rate:.3f}")
        print(f"    eps_link:   {upd.eps_link:.4f}")
        print(f"    eps_matter: {upd.eps_matter:.4f}")
        print()
        print("  THERMALISATION (L=4, β=8.0, 500 sweeps, links only):")
        print(f"    plaq_avg after: {plaquette_average(links5, lat5):.4f}")
        print(f"    (approaches 1 as β→∞ → ordered phase)")
    else:
        print("  SOME TESTS FAILED — review before proceeding to observables.py")
    print("=" * 66)


def pytest_approx(x: float, rel: float = 1e-3) -> "_Approx":
    class _Approx:
        def __init__(self, v, r):
            self.v = v
            self.r = r
        def __eq__(self, other):
            return abs(other - self.v) <= self.r * abs(self.v) + 1e-10
    return _Approx(x, rel)


if __name__ == "__main__":
    _run_tests()
