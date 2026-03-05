"""
u1.py — U(1) Phase Algebra
==========================
Layer 1 Substrate Module: E1_U1_Gauge (Negative Control)

U(1) = { e^{iθ} | θ ∈ [0, 2π) } ⊂ ℂ

Interface mirrors su2.py for group-swap modularity:

    su2.py          u1.py             Role
    ─────────────   ───────────────   ─────────────────────────────
    random_su2()    random_u1()       Haar-uniform group element
    small_su2(ε)    small_u1(ε)       Metropolis proposal near identity
    identity_su2()  identity_u1()     Group identity
    is_su2(U)       is_u1(z)          Membership test
    dagger(U)       dagger_u1(z)      Group inverse
    commutator(U,V) commutator_u1(z,w) Lie bracket (always 0 for U(1))
    su2_trace(U)    u1_trace(z)       Re(z) ~ ½Tr_SU(2)
    su2_plaquette   u1_plaquette      Ordered link product around plaquette

Epistemic role
──────────────
U(1) is the NEGATIVE CONTROL for the E7 MIXED-triad prediction.
E1 proved that U(1) gauge action drives plaquette holonomies → 0, which
forces gauge correlators to become uniform (all same sign), suppressing
MIXED triads:  R → 0  as β → ∞.

The SU(2) module (su2.py) is the POSITIVE candidate: non-Abelian
non-commutativity may introduce sign frustration and rescue R > 0.

Key difference from SU(2):
  U(1) is Abelian → commutator always zero → no frustration mechanism.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────
# Type alias for clarity
# ──────────────────────────────────────────────
# U(1) element: a Python complex number with |z| = 1
U1Element = complex


# ──────────────────────────────────────────────
# Core group operations
# ──────────────────────────────────────────────

def random_u1(rng: np.random.Generator | None = None) -> U1Element:
    """
    Sample a Haar-uniform U(1) element.

    θ ~ Uniform[0, 2π)  →  z = e^{iθ}

    The Haar measure on U(1) is dθ/2π, so uniform θ is exact.
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.uniform(0.0, 2.0 * np.pi)
    return complex(np.cos(theta), np.sin(theta))


def small_u1(epsilon: float, rng: np.random.Generator | None = None) -> U1Element:
    """
    Sample a U(1) element near the identity for Metropolis link proposals.

    δ ~ Uniform[−ε, ε]  →  z = e^{iδ}

    Analogue of small_random_su2(epsilon) in su2.py.
    """
    if rng is None:
        rng = np.random.default_rng()
    delta = rng.uniform(-epsilon, epsilon)
    return complex(np.cos(delta), np.sin(delta))


def identity_u1() -> U1Element:
    """Return the U(1) identity element: 1 + 0j."""
    return complex(1.0, 0.0)


def dagger_u1(z: U1Element) -> U1Element:
    """
    Return the group inverse (conjugate) of z.

    For z = e^{iθ}:  z† = e^{-iθ} = conj(z)
    """
    return z.conjugate()


def multiply_u1(z1: U1Element, z2: U1Element) -> U1Element:
    """Multiply two U(1) elements: e^{iθ₁} · e^{iθ₂} = e^{i(θ₁+θ₂)}."""
    return z1 * z2


def is_u1(z: U1Element, tol: float = 1e-10) -> bool:
    """
    Test whether z lies on the unit circle: |z| = 1 within tolerance.

    Analogue of is_su2(U) in su2.py.
    """
    return abs(abs(z) - 1.0) < tol


def commutator_u1(z1: U1Element, z2: U1Element) -> U1Element:
    """
    U(1) commutator: z1·z2·z1†·z2† = 1  (U(1) is Abelian).

    Returns (commutator - identity) so the result measures deviation from
    commutativity.  For U(1) this is always exactly 0+0j.

    Compare: commutator(U,V) in su2.py returns ‖UV−VU‖ ≈ 2.3  (non-zero).
    The zero commutator here is the defining Abelian property that
    PREVENTS sign frustration in the E7 negative-control experiment.
    """
    comm = z1 * z2 * z1.conjugate() * z2.conjugate()
    return comm - identity_u1()


def u1_trace(z: U1Element) -> float:
    """
    Real part of z — analogue of ½ Tr(U) for SU(2).

    For z = e^{iθ}:  Re(z) = cos θ ∈ [−1, 1]
    At identity: cos 0 = 1 (maximum)
    Used in the Wilson gauge action: S_g = β Σ (1 − Re z_p)
    """
    return z.real


def _get_link(
    links: Dict[Tuple[int, int], U1Element],
    i: int,
    j: int,
) -> U1Element:
    """
    Retrieve U_{ij} from the links dict, handling canonical edge direction.

    Lattice edges are stored only in canonical direction (lower→higher index
    for horizontal/right, lower→higher for vertical/up as built by Lattice2D).
    If (i,j) is absent but (j,i) is present, return conj(U_{j,i}).
    """
    if (i, j) in links:
        return links[(i, j)]
    elif (j, i) in links:
        return links[(j, i)].conjugate()
    else:
        raise KeyError(f"Edge ({i},{j}) not found in links dict (neither direction).")


def u1_plaquette(
    links: Dict[Tuple[int, int], U1Element],
    plaquette: Tuple[int, int, int, int],
) -> U1Element:
    """
    Compute the ordered U(1) plaquette holonomy.

    plaquette = (i0, i1, i2, i3) counterclockwise, as returned by
    Lattice2D.plaquettes().

    U_p = U_{i0,i1} · U_{i1,i2} · U_{i2,i3} · U_{i3,i0}

    where each factor is retrieved via _get_link(), which automatically
    returns the conjugate when traversing a canonical edge in the reverse
    direction.  No additional conjugation is applied here.

    For U(1) the ordering does not matter (Abelian), but we follow the
    same four-link loop convention as su2_plaquette() in su2.py for
    interface parity.  Gauge invariance (T9) is verified in the self-test.
    """
    i0, i1, i2, i3 = plaquette
    U01 = _get_link(links, i0, i1)
    U12 = _get_link(links, i1, i2)
    U23 = _get_link(links, i2, i3)
    U30 = _get_link(links, i3, i0)
    return U01 * U12 * U23 * U30


def u1_staple(
    links: Dict[Tuple[int, int], U1Element],
    edge: Tuple[int, int],
    plaquettes_of_edge: List[Tuple[int, int, int, int]],
) -> U1Element:
    """
    Sum of staples around a given link (i, j).

    Used in the gauge update to compute ΔS without recomputing the full action.
    s_total = Σ_{p ∋ (i,j)}  [ product of other 3 links in plaquette p ]

    Analogue of the staple sum in standard lattice gauge theory.
    """
    i, j = edge
    staple_sum = complex(0.0, 0.0)
    for p in plaquettes_of_edge:
        # Product of all links in the plaquette except (i, j) and (j, i)
        prod = complex(1.0, 0.0)
        verts = list(p)
        n = len(verts)
        for k in range(n):
            a = verts[k]
            b = verts[(k + 1) % n]
            if (a, b) == (i, j):
                # forward edge — skip (this is the link being updated)
                continue
            elif (b, a) == (i, j):
                # reverse edge — skip
                continue
            else:
                prod *= _get_link(links, a, b)
        staple_sum += prod
    return staple_sum


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def u1_to_angle(z: U1Element) -> float:
    """Extract angle θ ∈ (−π, π] from z = e^{iθ}."""
    return float(np.angle(z))


def angle_to_u1(theta: float) -> U1Element:
    """Convert angle θ to U(1) element e^{iθ}."""
    return complex(np.cos(theta), np.sin(theta))


def normalize_u1(z: complex) -> U1Element:
    """Project an arbitrary complex number onto the unit circle."""
    r = abs(z)
    if r < 1e-15:
        return identity_u1()
    return z / r


# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────

def _run_tests() -> None:
    rng = np.random.default_rng(42)
    PASS = "PASS"
    FAIL = "FAIL"
    results: list[Tuple[str, bool, str]] = []

    # ── Test 1: random_u1 lives on unit circle ──────────────────────────
    N = 10_000
    zs = [random_u1(rng) for _ in range(N)]
    magnitudes = [abs(z) for z in zs]
    ok1 = all(abs(m - 1.0) < 1e-12 for m in magnitudes)
    results.append(("T1  random_u1 ∈ U(1)", ok1,
                    f"max|r−1| = {max(abs(m-1) for m in magnitudes):.2e}"))

    # ── Test 2: uniform angle distribution ──────────────────────────────
    angles = [u1_to_angle(z) for z in zs]
    mean_cos = float(np.mean([np.cos(a) for a in angles]))
    mean_sin = float(np.mean([np.sin(a) for a in angles]))
    ok2 = abs(mean_cos) < 0.05 and abs(mean_sin) < 0.05
    results.append(("T2  Haar uniformity <cos θ>≈0, <sin θ>≈0", ok2,
                    f"<cos θ>={mean_cos:.4f}, <sin θ>={mean_sin:.4f}"))

    # ── Test 3: small_u1 stays near identity ────────────────────────────
    epsilon = 0.3
    zs_small = [small_u1(epsilon, rng) for _ in range(N)]
    angles_small = [abs(u1_to_angle(z)) for z in zs_small]
    ok3 = max(angles_small) <= epsilon + 1e-12
    results.append(("T3  small_u1 confined to [−ε, ε]", ok3,
                    f"max|δ| = {max(angles_small):.6f}, ε = {epsilon}"))

    # ── Test 4: group closure ────────────────────────────────────────────
    z1 = random_u1(rng)
    z2 = random_u1(rng)
    prod = multiply_u1(z1, z2)
    ok4 = is_u1(prod)
    results.append(("T4  group closure: z1·z2 ∈ U(1)", ok4,
                    f"|z1·z2| = {abs(prod):.15f}"))

    # ── Test 5: inverse / dagger ─────────────────────────────────────────
    z = random_u1(rng)
    z_inv = dagger_u1(z)
    prod_inv = multiply_u1(z, z_inv)
    ok5 = abs(prod_inv - identity_u1()) < 1e-14
    results.append(("T5  z · z† = 1", ok5,
                    f"|z·z†−1| = {abs(prod_inv - 1):.2e}"))

    # ── Test 6: Abelian commutator = 0 (contrast with SU(2)) ────────────
    z1, z2 = random_u1(rng), random_u1(rng)
    comm = commutator_u1(z1, z2)
    ok6 = abs(comm) < 1e-14
    results.append(("T6  Abelian: commutator ≡ 0", ok6,
                    f"|[z1,z2]| = {abs(comm):.2e}  (SU(2) gives ~2.3)"))

    # ── Test 7: u1_trace ∈ [−1, 1] ──────────────────────────────────────
    traces = [u1_trace(z) for z in [random_u1(rng) for _ in range(1000)]]
    ok7 = all(-1.0 <= t <= 1.0 for t in traces)
    ok7 &= abs(u1_trace(identity_u1()) - 1.0) < 1e-15
    results.append(("T7  u1_trace ∈ [−1,1]; identity→1", ok7,
                    f"Re(id)={u1_trace(identity_u1())}, min={min(traces):.4f}"))

    # ── Test 8: plaquette holonomy ───────────────────────────────────────
    # Manually set up a 4-link plaquette; commute them (should be trivial for U(1))
    from lattice import Lattice2D
    lat = Lattice2D(4)
    links = {e: random_u1(rng) for e in lat.edges()}
    # All plaquettes should be on unit circle
    p_vals = [u1_plaquette(links, p) for p in lat.plaquettes()]
    ok8 = all(is_u1(pv) for pv in p_vals)
    results.append(("T8  all plaquette holonomies ∈ U(1)", ok8,
                    f"n_plaquettes={len(p_vals)}, max|r−1|="
                    f"{max(abs(abs(v)-1) for v in p_vals):.2e}"))

    # ── Test 9: gauge invariance of plaquette (U(1)) ────────────────────
    # Gauge transform: U_{ij} → Λ_i · U_{ij} · Λ_j†
    # Plaquette product is gauge-invariant (all Λs cancel)
    lat2 = Lattice2D(4)
    links2 = {e: random_u1(rng) for e in lat2.edges()}
    # Original plaquette values
    p_orig = {p: u1_plaquette(links2, p) for p in lat2.plaquettes()}
    # Apply random gauge transform
    gauge = {s: random_u1(rng) for s in range(lat2.N)}
    links2_t = {}
    for (i, j) in lat2.edges():
        links2_t[(i, j)] = gauge[i] * links2[(i, j)] * dagger_u1(gauge[j])
    p_new = {p: u1_plaquette(links2_t, p) for p in lat2.plaquettes()}
    max_err = max(abs(p_new[p] - p_orig[p]) for p in lat2.plaquettes())
    ok9 = max_err < 1e-12
    results.append(("T9  plaquette gauge invariance", ok9,
                    f"max |U_p(orig)−U_p(gauged)| = {max_err:.2e}"))

    # ── Print results ────────────────────────────────────────────────────
    print("=" * 66)
    print("u1.py — U(1) Algebra Self-Test")
    print("Epistemic role: NEGATIVE CONTROL (Abelian → R → 0 under gauge)")
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
        print("  KEY DIFFERENCE FROM SU(2):")
        z1, z2 = complex(0, 1), complex(-1, 0)   # i and -1 in U(1)
        su2_note = "‖[U,V]‖ ≈ 2.3 (from su2.py T3)"
        u1_note  = f"|[z1,z2]| = {abs(commutator_u1(z1,z2)):.2e} (always 0)"
        print(f"  SU(2): {su2_note}")
        print(f"  U(1):  {u1_note}")
        print()
        print("  IMPLICATION: U(1) gauge action drive all plaquettes → 0")
        print("   → all gauge correlators → same sign → MIXED R → 0.")
        print("   SU(2) non-commutativity is the candidate frustration mechanism.")
    else:
        print("  SOME TESTS FAILED — review before proceeding to fields.py")
    print("=" * 66)


if __name__ == "__main__":
    _run_tests()
