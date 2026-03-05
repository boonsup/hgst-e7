#!/usr/bin/env python3
"""
sm_fields.py - Field initialization for SU(3)×SU(2)×U(1) gauge theory on the HGST lattice.
============================================================================================
Provides:
  • QuarkDoublet   — left-handed (u_L, d_L): color triplet × weak doublet
  • LeptonDoublet  — left-handed (ν_L, e_L): color singlet × weak doublet
  • initialize_sm_links   — SMGaugeElement on every directed edge
  • initialize_quarks     — QuarkDoublet at each site, |ψ| = r_i
  • initialize_leptons    — LeptonDoublet at each site, |ψ| = r_i
  • random_gauge          — random gauge transformation matrices at each site
  • gauge_transform_links / _quarks / _leptons — apply gauge transformation
  • check_field_norms     — verify |ψ| = r_i for all matter fields

Field norm convention: the combined doublet satisfies |ψ| = r_i where
r_i = 2^{n-m} is the HGST grade magnitude at site i.  For quarks,
|ψ|² = |up|² + |down|² = r_i²; for leptons, |ψ|² = |ν|² + |e|² = r_i².

Epistemic status: VALIDATED after test suite passes.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lattice import Lattice2D
from sm_gauge import SMGaugeElement
import su3
import su2
import u1


# ---------------------------------------------------------------------------
# Matter field dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QuarkDoublet:
    """
    Left-handed quark doublet (u_L, d_L).

    Each component is an SU(3) colour triplet.
    Combined norm: |up|² + |down|² = r_i² (HGST grade magnitude).
    """
    up:   np.ndarray   # shape (3,), complex — scaled by r_i
    down: np.ndarray   # shape (3,), complex — scaled by r_i

    def __post_init__(self):
        if self.up.shape != (3,):
            raise ValueError(f"up must be shape (3,), got {self.up.shape}")
        if self.down.shape != (3,):
            raise ValueError(f"down must be shape (3,), got {self.down.shape}")

    def norm(self) -> float:
        """Combined doublet norm √(|up|²+|down|²). Should equal r_i."""
        return np.sqrt(float(np.linalg.norm(self.up)**2 +
                             np.linalg.norm(self.down)**2))

    def chi(self, r: float) -> np.ndarray:
        """Return normalised 6-component spinor (divide by r)."""
        if r == 0:
            return np.concatenate([self.up, self.down])
        return np.concatenate([self.up, self.down]) / r

    @staticmethod
    def from_chi(chi6: np.ndarray, r: float) -> 'QuarkDoublet':
        """Construct from a unit 6-component spinor and grade magnitude r."""
        scaled = chi6 * r
        return QuarkDoublet(up=scaled[:3].copy(), down=scaled[3:].copy())


@dataclass
class LeptonDoublet:
    """
    Left-handed lepton doublet (ν_L, e_L).

    Colour singlets; transforms under SU(2) and U(1) (hypercharge Y=−1/2).
    Combined norm: |ν|² + |e|² = r_i².
    """
    neutrino: complex
    electron: complex

    def norm(self) -> float:
        """Combined doublet norm. Should equal r_i."""
        return float(np.sqrt(abs(self.neutrino)**2 + abs(self.electron)**2))

    def chi(self, r: float) -> np.ndarray:
        """Return unit 2-component spinor."""
        vec = np.array([self.neutrino, self.electron])
        return vec / r if r != 0 else vec

    @staticmethod
    def from_chi(chi2: np.ndarray, r: float) -> 'LeptonDoublet':
        scaled = chi2 * r
        return LeptonDoublet(neutrino=complex(scaled[0]),
                             electron=complex(scaled[1]))


# ---------------------------------------------------------------------------
# Link initialization
# ---------------------------------------------------------------------------

def initialize_sm_links(
    lattice: Lattice2D,
    random: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Dict[Tuple[int, int], SMGaugeElement]:
    """
    Initialise link variables on all directed canonical edges.

    Parameters
    ----------
    lattice : Lattice2D
    random  : if True, random Haar; if False, identity
    rng     : random number generator

    Returns
    -------
    links : dict (i,j) -> SMGaugeElement
    """
    if rng is None:
        rng = np.random.default_rng()
    links = {}
    for (i, j) in lattice.edges():
        links[(i, j)] = SMGaugeElement.random(rng) if random else SMGaugeElement.identity()
    return links


# ---------------------------------------------------------------------------
# Matter field initialization
# ---------------------------------------------------------------------------

def initialize_quarks(
    lattice: Lattice2D,
    random: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Dict[int, QuarkDoublet]:
    """
    Initialise left-handed quark doublets at each site.

    The combined norm satisfies |ψ| = r_i (HGST grade magnitude).
    Random initial state draws (up, down) from a uniform distribution on
    the 6-complex-component sphere scaled by r_i.
    """
    if rng is None:
        rng = np.random.default_rng()
    quarks = {}
    for site in range(lattice.N):
        r_i = lattice.r[site]
        if random:
            # 6-component complex unit vector (up||down)
            raw = rng.normal(0, 1, 12).view(np.complex128)  # 6 complex numbers
            norm = np.linalg.norm(raw)
            chi6 = raw / norm
        else:
            chi6 = np.array([1, 0, 0, 0, 1, 0], dtype=complex) / np.sqrt(2)
        quarks[site] = QuarkDoublet.from_chi(chi6, r_i)
    return quarks


def initialize_leptons(
    lattice: Lattice2D,
    random: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Dict[int, LeptonDoublet]:
    """
    Initialise left-handed lepton doublets at each site.

    The combined norm satisfies |ψ| = r_i.
    """
    if rng is None:
        rng = np.random.default_rng()
    leptons = {}
    for site in range(lattice.N):
        r_i = lattice.r[site]
        if random:
            z = rng.normal(0, 1, 4).view(np.complex128)  # 2 complex numbers
            norm = np.sqrt(abs(z[0])**2 + abs(z[1])**2)
            chi2 = z / norm
        else:
            chi2 = np.array([1.0 + 0j, 0j])
        leptons[site] = LeptonDoublet.from_chi(chi2, r_i)
    return leptons


# ---------------------------------------------------------------------------
# Gauge transformation utilities
# ---------------------------------------------------------------------------

def random_gauge(
    lattice: Lattice2D,
    rng: Optional[np.random.Generator] = None
) -> Dict[int, SMGaugeElement]:
    """Generate a random gauge transformation V_i ∈ SU(3)×SU(2)×U(1) at each site."""
    if rng is None:
        rng = np.random.default_rng()
    return {site: SMGaugeElement.random(rng) for site in range(lattice.N)}


def gauge_transform_links(
    links: Dict[Tuple[int, int], SMGaugeElement],
    V: Dict[int, SMGaugeElement]
) -> Dict[Tuple[int, int], SMGaugeElement]:
    """
    Transform links: U_ij → V_i @ U_ij @ V_j†
    """
    return {(i, j): V[i] @ U @ V[j].dagger()
            for (i, j), U in links.items()}


def gauge_transform_quarks(
    quarks: Dict[int, QuarkDoublet],
    V: Dict[int, SMGaugeElement]
) -> Dict[int, QuarkDoublet]:
    """
    Transform quark doublets under SU(3)×SU(2).

    (u_L, d_L) → V_i:
      u' = V3 @ (V2[0,0]*u + V2[0,1]*d)
      d' = V3 @ (V2[1,0]*u + V2[1,1]*d)
    """
    new_quarks = {}
    for site, q in quarks.items():
        Vi = V[site]
        u_temp = Vi.su2[0, 0] * q.up   + Vi.su2[0, 1] * q.down
        d_temp = Vi.su2[1, 0] * q.up   + Vi.su2[1, 1] * q.down
        new_quarks[site] = QuarkDoublet(
            up=Vi.su3 @ u_temp,
            down=Vi.su3 @ d_temp
        )
    return new_quarks


def gauge_transform_leptons(
    leptons: Dict[int, LeptonDoublet],
    V: Dict[int, SMGaugeElement]
) -> Dict[int, LeptonDoublet]:
    """
    Transform lepton doublets under SU(2) × U(1) (hypercharge Y=−1/2).

    (ν_L, e_L) → V_i:
      ν' = e^{−iθ/2} (V2[0,0]*ν + V2[0,1]*e)
      e' = e^{−iθ/2} (V2[1,0]*ν + V2[1,1]*e)
    where e^{iθ} = V_i.u1.
    """
    new_leptons = {}
    for site, l in leptons.items():
        Vi = V[site]
        u1_factor = np.exp(-0.5j * np.angle(Vi.u1))
        nu_temp = Vi.su2[0, 0] * l.neutrino + Vi.su2[0, 1] * l.electron
        e_temp  = Vi.su2[1, 0] * l.neutrino + Vi.su2[1, 1] * l.electron
        new_leptons[site] = LeptonDoublet(
            neutrino=complex(u1_factor * nu_temp),
            electron=complex(u1_factor * e_temp)
        )
    return new_leptons


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def check_field_norms(
    quarks: Dict[int, QuarkDoublet],
    leptons: Dict[int, LeptonDoublet],
    lattice: Lattice2D,
    tol: float = 1e-10
) -> Tuple[bool, float]:
    """
    Verify that |ψ| = r_i for all matter fields.

    Returns (all_ok, max_error).
    """
    max_err = 0.0
    for site, q in quarks.items():
        err = abs(q.norm() - lattice.r[site])
        max_err = max(max_err, err)
    for site, lep in leptons.items():
        err = abs(lep.norm() - lattice.r[site])
        max_err = max(max_err, err)
    return max_err < tol, max_err


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> bool:
    PASS, FAIL = "PASS", "FAIL"
    results = []
    rng = np.random.default_rng(42)
    lat = Lattice2D(4)

    # T1: Link initialization
    links = initialize_sm_links(lat, random=True, rng=rng)
    ok1 = (len(links) == lat.n_edges and
           all(isinstance(U, SMGaugeElement) for U in links.values()))
    results.append(("T1 link initialization", ok1, f"{len(links)} links"))

    # T2: Cold links are identity
    I_elem = SMGaugeElement.identity()
    links_cold = initialize_sm_links(lat, random=False)
    ok2 = all(
        np.allclose(U.su3, I_elem.su3) and
        np.allclose(U.su2, I_elem.su2) and
        abs(U.u1 - I_elem.u1) < 1e-12
        for U in links_cold.values()
    )
    results.append(("T2 cold links identity", ok2, ""))

    # T3: Quark initialization + norm constraint
    quarks = initialize_quarks(lat, random=True, rng=rng)
    ok3a = all(isinstance(q, QuarkDoublet) for q in quarks.values())
    ok3b, max_err3 = check_field_norms(quarks, {}, lat)
    results.append(("T3 quark initialization + |ψ|=r_i", ok3a and ok3b,
                   f"max norm error={max_err3:.2e}"))

    # T4: Lepton initialization + norm constraint
    leptons = initialize_leptons(lat, random=True, rng=rng)
    ok4a = all(isinstance(l, LeptonDoublet) for l in leptons.values())
    ok4b, max_err4 = check_field_norms({}, leptons, lat)
    results.append(("T4 lepton initialization + |ψ|=r_i", ok4a and ok4b,
                   f"max norm error={max_err4:.2e}"))

    # T5: Gauge-transformed links remain valid SU(3)×SU(2)×U(1)
    V = random_gauge(lat, rng)
    links_t = gauge_transform_links(links, V)
    ok5 = all(U.is_valid()[0] for U in links_t.values())
    results.append(("T5 gauge-transformed links valid", ok5, ""))

    # T6: Quark gauge transform preserves norm, is non-trivial
    quarks_t = gauge_transform_quarks(quarks, V)
    ok6a, max_err6 = check_field_norms(quarks_t, {}, lat)
    diff6 = sum(np.linalg.norm(quarks_t[s].up - quarks[s].up)
                for s in range(lat.N))
    ok6b = diff6 > 1e-10
    results.append(("T6 quark gauge transform norm invariant", ok6a and ok6b,
                   f"max_err={max_err6:.2e}, diff={diff6:.2e}"))

    # T7: Lepton gauge transform preserves norm, is non-trivial
    leptons_t = gauge_transform_leptons(leptons, V)
    ok7a, max_err7 = check_field_norms({}, leptons_t, lat)
    diff7 = sum(abs(leptons_t[s].neutrino - leptons[s].neutrino) +
                abs(leptons_t[s].electron - leptons[s].electron)
                for s in range(lat.N))
    ok7b = diff7 > 1e-10
    results.append(("T7 lepton gauge transform norm invariant", ok7a and ok7b,
                   f"max_err={max_err7:.2e}, diff={diff7:.2e}"))

    print("=" * 66)
    print("sm_fields.py — Field Initialization Self-Test")
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
        print("\n  Field initialization for SM is ready.")
    else:
        print("  SOME TESTS FAILED — debug before proceeding.")
    print("=" * 66)
    return all_pass


if __name__ == "__main__":
    import sys as _sys
    success = _run_tests()
    _sys.exit(0 if success else 1)
