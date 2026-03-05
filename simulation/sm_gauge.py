#!/usr/bin/env python3
"""
sm_gauge.py - SU(3)×SU(2)×U(1) Product Group for HGST
========================================================
Implements the full Standard Model gauge structure on the grade lattice.

Each link variable is a tuple (U3, U2, U1) representing:
  U3 ∈ SU(3)  (gluons)
  U2 ∈ SU(2)  (weak bosons)
  U1 ∈ U(1)   (hypercharge B)

The module provides:
  • SMGaugeElement dataclass with multiplication, inverse, trace, serialization
  • Random generation (Haar for each factor independently)
  • Small random proposals for Metropolis updates
  • Plaquette computation and plaquette average
  • Delta-action computation for local link updates
  • Comprehensive 11-test self-test suite

Epistemic status: VALIDATED after test suite passes.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import su3
import su2
import u1


# ---------------------------------------------------------------------------
# Product group element
# ---------------------------------------------------------------------------

@dataclass
class SMGaugeElement:
    """
    An element of SU(3)×SU(2)×U(1).

    Fields
    ------
    su3 : np.ndarray, shape (3,3), complex  — SU(3) matrix
    su2 : np.ndarray, shape (2,2), complex  — SU(2) matrix
    u1  : complex                            — U(1) phase (|u1| = 1)

    All operations are component-wise: direct product structure throughout.
    """

    su3: np.ndarray
    su2: np.ndarray
    u1:  complex  # noqa (shadows module name locally)

    def __post_init__(self):
        if self.su3.shape != (3, 3):
            raise ValueError(f"SU(3) component must be 3×3, got {self.su3.shape}")
        if self.su2.shape != (2, 2):
            raise ValueError(f"SU(2) component must be 2×2, got {self.su2.shape}")
        if not isinstance(self.u1, (complex, np.complexfloating)):
            self.u1 = complex(self.u1)

    # -----------------------------------------------------------------------
    # Group operations
    # -----------------------------------------------------------------------

    def __matmul__(self, other: 'SMGaugeElement') -> 'SMGaugeElement':
        """
        Direct product multiplication:
            (U3,U2,U1) @ (V3,V2,V1) = (U3@V3, U2@V2, U1*V1)
        """
        return SMGaugeElement(
            su3=self.su3 @ other.su3,
            su2=self.su2 @ other.su2,
            u1=self.u1 * other.u1
        )

    def dagger(self) -> 'SMGaugeElement':
        """
        Group inverse = Hermitian conjugate for each factor:
            (U3,U2,U1)† = (U3†, U2†, U1̄)
        """
        return SMGaugeElement(
            su3=su3.dagger(self.su3),
            su2=su2.dagger(self.su2),
            u1=u1.dagger_u1(self.u1)
        )

    def trace(self) -> float:
        """
        Combined normalised trace for Wilson action.

        Returns (1/3)ReTr(U3) + (1/2)ReTr(U2) + Re(U1) ∈ [−3, 3].
        Equals 3.0 at identity.
        """
        return (su3.su3_trace(self.su3) +
                0.5 * su2.su2_trace(self.su2) +
                u1.u1_trace(self.u1))

    # -----------------------------------------------------------------------
    # Serialisation (for checkpointing)
    # -----------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Flatten to a real array of length 28."""
        # SU(3): 3×3 complex → 18 real numbers
        su3_flat = np.concatenate([self.su3.real.flatten(), self.su3.imag.flatten()])
        # SU(2): 2×2 complex → 8 real numbers
        su2_flat = np.concatenate([self.su2.real.flatten(), self.su2.imag.flatten()])
        # U(1): 1 complex → 2 real numbers
        u1_flat = np.array([self.u1.real, self.u1.imag])
        return np.concatenate([su3_flat, su2_flat, u1_flat])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'SMGaugeElement':
        """Reconstruct from flat array (inverse of to_array)."""
        su3_re = arr[0:9].reshape(3, 3)
        su3_im = arr[9:18].reshape(3, 3)
        su3_mat = su3_re + 1j * su3_im

        su2_re = arr[18:22].reshape(2, 2)
        su2_im = arr[22:26].reshape(2, 2)
        su2_mat = su2_re + 1j * su2_im

        u1_val = complex(arr[26] + 1j * arr[27])
        return SMGaugeElement(su3=su3_mat, su2=su2_mat, u1=u1_val)

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @staticmethod
    def identity() -> 'SMGaugeElement':
        """Identity element of the product group."""
        return SMGaugeElement(
            su3=su3.identity_su3(),
            su2=su2.identity_su2(),
            u1=u1.identity_u1()
        )

    @staticmethod
    def random(rng: Optional[np.random.Generator] = None) -> 'SMGaugeElement':
        """
        Random element drawn from Haar measure on each factor independently.
        """
        if rng is None:
            rng = np.random.default_rng()
        return SMGaugeElement(
            su3=su3.random_su3(rng),
            su2=su2.random_su2(),     # su2.random_su2 uses its own np.random
            u1=u1.random_u1(rng)
        )

    @staticmethod
    def small_random(epsilon: float = 0.1,
                     rng: Optional[np.random.Generator] = None) -> 'SMGaugeElement':
        """
        Small random element near identity for Metropolis proposals.
        """
        if rng is None:
            rng = np.random.default_rng()
        return SMGaugeElement(
            su3=su3.small_random_su3(epsilon, rng),
            su2=su2.small_random_su2(epsilon),
            u1=u1.small_u1(epsilon, rng)
        )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def is_valid(self, tol: float = 1e-10) -> Tuple[bool, str]:
        """
        Check this is a valid element of SU(3)×SU(2)×U(1).

        Returns (bool, message).
        """
        ok3, msg3 = su3.is_su3(self.su3, tol)
        if not ok3:
            return False, f"SU(3) invalid: {msg3}"

        if not su2.is_su2(self.su2, tol):
            return False, "SU(2) invalid"

        if not u1.is_u1(self.u1, tol):
            return False, f"U(1) invalid: |u1|={abs(self.u1):.2e}"

        return True, "OK"

    def __repr__(self) -> str:
        return f"SMGaugeElement(trace={self.trace():.4f})"


# ---------------------------------------------------------------------------
# Plaquette operations
# ---------------------------------------------------------------------------

def sm_plaquette_product(
    links: dict,
    lattice,
    plaquette: Tuple[int, int, int, int]
) -> 'SMGaugeElement':
    """
    Ordered product of links around a plaquette.

    Parameters
    ----------
    links : dict (i,j) -> SMGaugeElement
    lattice : Lattice2D
    plaquette : (i0, i1, i2, i3) in counterclockwise order

    Returns
    -------
    U_p = U_{i0i1} @ U_{i1i2} @ U_{i2i3} @ U_{i3i0}
    """
    i0, i1, i2, i3 = plaquette

    def get_link(i, j):
        if (i, j) in links:
            return links[(i, j)]
        elif (j, i) in links:
            return links[(j, i)].dagger()
        else:
            raise KeyError(f"Edge ({i},{j}) not in links")

    return get_link(i0, i1) @ get_link(i1, i2) @ get_link(i2, i3) @ get_link(i3, i0)


def sm_plaquette_average(
    links: dict,
    lattice
) -> Tuple[float, float, float]:
    """
    Compute per-factor plaquette averages.

    Returns
    -------
    (plaq_3, plaq_2, plaq_1) where:
        plaq_3 = ⟨(1/3)ReTr U₃⟩
        plaq_2 = ⟨(1/2)ReTr U₂⟩
        plaq_1 = ⟨Re U₁⟩
    """
    s3 = s2 = s1 = 0.0
    n = 0
    for p in lattice.plaquettes():
        U_p = sm_plaquette_product(links, lattice, p)
        s3 += su3.su3_trace(U_p.su3)
        s2 += 0.5 * su2.su2_trace(U_p.su2)
        s1 += u1.u1_trace(U_p.u1)
        n += 1
    return s3 / n, s2 / n, s1 / n


# ---------------------------------------------------------------------------
# Action functions (used by sm_action.py and the self-test)
# ---------------------------------------------------------------------------

def sm_gauge_action(
    links: dict,
    lattice,
    beta_3: float,
    beta_2: float,
    beta_1: float
) -> float:
    """
    Full Wilson gauge action:
        S = β₃Σ(1 − (1/3)ReTrU₃) + β₂Σ(1 − (1/2)ReTrU₂) + β₁Σ(1 − ReU₁)
    """
    action = 0.0
    for p in lattice.plaquettes():
        U_p = sm_plaquette_product(links, lattice, p)
        action += beta_3 * (1.0 - su3.su3_trace(U_p.su3))
        action += beta_2 * (1.0 - 0.5 * su2.su2_trace(U_p.su2))
        action += beta_1 * (1.0 - u1.u1_trace(U_p.u1))
    return action


def sm_delta_action_link(
    links: dict,
    lattice,
    edge: Tuple[int, int],
    U_new: 'SMGaugeElement',
    beta_3: float,
    beta_2: float,
    beta_1: float
) -> float:
    """
    Change in gauge action when replacing `links[edge]` with U_new.
    Only plaquettes containing the edge are recomputed.
    """
    U_old = links[edge]
    delta = 0.0
    plaq_indices = lattice.plaquettes_of_edge(edge)
    all_plaquettes = lattice.plaquettes()

    for p_idx in plaq_indices:
        p = all_plaquettes[p_idx]

        # Build plaquette with old and new link
        links_old_snap = links  # current state has old link

        # Temporarily swap the link to compute new plaquette
        links_new_snap = dict(links)
        links_new_snap[edge] = U_new

        U_p_old = sm_plaquette_product(links_old_snap, lattice, p)
        U_p_new = sm_plaquette_product(links_new_snap, lattice, p)

        # ΔS = S_new − S_old → beta*(old_trace − new_trace) because action = beta*(1-trace)
        delta += beta_3 * (su3.su3_trace(U_p_old.su3) - su3.su3_trace(U_p_new.su3))
        delta += beta_2 * 0.5 * (su2.su2_trace(U_p_old.su2) - su2.su2_trace(U_p_new.su2))
        delta += beta_1 * (u1.u1_trace(U_p_old.u1)    - u1.u1_trace(U_p_new.u1))

    return delta


# ---------------------------------------------------------------------------
# Self-test suite
# ---------------------------------------------------------------------------

def _run_tests() -> bool:
    from lattice import Lattice2D

    PASS, FAIL = "PASS", "FAIL"
    results = []
    rng = np.random.default_rng(42)
    lat = Lattice2D(4)

    print("=" * 70)
    print("sm_gauge.py — SU(3)×SU(2)×U(1) Product Group Self-Test")
    print("=" * 70)

    # T1: Identity element
    I = SMGaugeElement.identity()
    ok1, msg1 = I.is_valid()
    results.append(("T1 identity element", ok1, msg1))

    # T2: Random element validity
    U = SMGaugeElement.random(rng)
    ok2, msg2 = U.is_valid()
    results.append(("T2 random element", ok2, msg2))

    # T3: Small random near identity
    V = SMGaugeElement.small_random(0.2, rng)
    ok3a, msg3 = V.is_valid()
    dist = np.linalg.norm(V.to_array() - I.to_array())
    ok3 = ok3a and dist < 3.0
    results.append(("T3 small random near identity", ok3,
                   f"dist={dist:.4f}, {msg3}"))

    # T4: Multiplication closure
    U = SMGaugeElement.random(rng)
    Vs = SMGaugeElement.random(rng)
    W = U @ Vs
    ok4, msg4 = W.is_valid()
    results.append(("T4 multiplication closure", ok4, msg4))

    # T5: Inverse property
    U = SMGaugeElement.random(rng)
    Uinv = U.dagger()
    I_check = U @ Uinv
    ok5a, msg5 = I_check.is_valid()
    inv_dist = np.linalg.norm(I_check.to_array() - I.to_array())
    ok5 = ok5a and inv_dist < 1e-10
    results.append(("T5 inverse property", ok5,
                   f"dist to I={inv_dist:.2e}, {msg5}"))

    # T6: Trace at identity = 3
    tr_I = I.trace()
    ok6 = abs(tr_I - 3.0) < 1e-12
    results.append(("T6 trace(identity)=3", ok6, f"trace={tr_I:.6f}"))

    # T7: Trace range [-3,3] over 100 random elements
    traces = [SMGaugeElement.random(rng).trace() for _ in range(100)]
    min_tr, max_tr = min(traces), max(traces)
    ok7 = -3.0 <= min_tr <= max_tr <= 3.0
    results.append(("T7 trace ∈ [−3,3]", ok7,
                   f"min={min_tr:.4f}, max={max_tr:.4f}"))

    # T8: Serialisation round-trip
    U = SMGaugeElement.random(rng)
    arr = U.to_array()
    U2 = SMGaugeElement.from_array(arr)
    diff = np.linalg.norm(U.to_array() - U2.to_array())
    ok8 = diff < 1e-12
    results.append(("T8 serialisation round-trip", ok8, f"diff={diff:.2e}"))

    # T9: Plaquette trace gauge invariance (trace is invariant; matrix is covariant)
    links = {e: SMGaugeElement.random(rng) for e in lat.edges()}
    traces_orig = {}
    for p in lat.plaquettes():
        U_p = sm_plaquette_product(links, lat, p)
        traces_orig[p] = U_p.trace()
    gauge = {s: SMGaugeElement.random(rng) for s in range(lat.N)}
    links_t = {(i, j): gauge[i] @ U_ij @ gauge[j].dagger()
               for (i, j), U_ij in links.items()}
    max_diff = 0.0
    for p in lat.plaquettes():
        U_p_new = sm_plaquette_product(links_t, lat, p)
        max_diff = max(max_diff, abs(U_p_new.trace() - traces_orig[p]))
    ok9 = max_diff < 1e-10
    results.append(("T9 plaquette trace gauge invariance", ok9,
                   f"max diff={max_diff:.2e}"))

    # T10: Delta action matches finite difference
    edges = list(lat.edges())
    edge = edges[5]
    delta_U = SMGaugeElement.small_random(0.1, rng)
    U_old = links[edge]
    U_new = delta_U @ U_old
    beta_3, beta_2, beta_1 = 1.0, 1.0, 1.0
    delta_ded = sm_delta_action_link(links, lat, edge, U_new, beta_3, beta_2, beta_1)
    links_temp = dict(links);  links_temp[edge] = U_new
    S_old = sm_gauge_action(links, lat, beta_3, beta_2, beta_1)
    S_new = sm_gauge_action(links_temp, lat, beta_3, beta_2, beta_1)
    delta_full = S_new - S_old
    ok10 = abs(delta_ded - delta_full) < 1e-10 * max(1.0, abs(delta_full))
    results.append(("T10 delta_action accuracy", ok10,
                   f"Δ_ded={delta_ded:.6f}, Δ_full={delta_full:.6f}, "
                   f"diff={abs(delta_ded-delta_full):.2e}"))

    # T11: Plaquette average computation
    plaq3, plaq2, plaq1 = sm_plaquette_average(links, lat)
    ok11 = (-1.0 <= plaq3 <= 1.0 and -1.0 <= plaq2 <= 1.0 and -1.0 <= plaq1 <= 1.0)
    results.append(("T11 plaquette averages in range", ok11,
                   f"plaq3={plaq3:.4f}, plaq2={plaq2:.4f}, plaq1={plaq1:.4f}"))

    # -----------------------------------------------------------------------
    all_pass = True
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if detail:
            print(f"         {detail}")
        if not ok:
            all_pass = False
    print("-" * 70)
    if all_pass:
        print(f"  All {len(results)} tests PASSED.")
        print("\n  SU(3)×SU(2)×U(1) product group module is VALIDATED.")
        print("  Ready for integration into the simulation pipeline.")
    else:
        print("  SOME TESTS FAILED — debug before proceeding.")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    import sys as _sys
    success = _run_tests()
    _sys.exit(0 if success else 1)
