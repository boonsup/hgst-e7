"""
su3.py - SU(3) matrix operations for HGST lattice simulations.
===============================================================================
All matrices are represented as 3x3 complex numpy arrays.
Interface mirrors su2.py for drop-in replacement in the HGST pipeline.

Epistemic status: UNDER CONSTRUCTION → will be VALIDATED after test suite passes.

Key differences from SU(2):
  • 8 generators (Gell-Mann matrices) vs. 3 Pauli matrices
  • Trace/3 convention for Wilson action (matches ½Tr for SU(2))
  • Determinant must be exactly 1 (enforced via projection)
  • No closed-form Haar measure — use Cabibbo-Marinari or heat bath

HGST Framework: Layer 1 Substrate Module — E10_SU3_Gauge (extension candidate)
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Gell-Mann matrices (generators of SU(3))
# ---------------------------------------------------------------------------

def _gm1() -> np.ndarray:
    """λ₁ = [[0,1,0],[1,0,0],[0,0,0]]"""
    return np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 0]], dtype=complex)

def _gm2() -> np.ndarray:
    """λ₂ = [[0,-i,0],[i,0,0],[0,0,0]]"""
    return np.array([[0, -1j, 0],
                     [1j, 0, 0],
                     [0, 0, 0]], dtype=complex)

def _gm3() -> np.ndarray:
    """λ₃ = [[1,0,0],[0,-1,0],[0,0,0]]"""
    return np.array([[1, 0, 0],
                     [0, -1, 0],
                     [0, 0, 0]], dtype=complex)

def _gm4() -> np.ndarray:
    """λ₄ = [[0,0,1],[0,0,0],[1,0,0]]"""
    return np.array([[0, 0, 1],
                     [0, 0, 0],
                     [1, 0, 0]], dtype=complex)

def _gm5() -> np.ndarray:
    """λ₅ = [[0,0,-i],[0,0,0],[i,0,0]]"""
    return np.array([[0, 0, -1j],
                     [0, 0, 0],
                     [1j, 0, 0]], dtype=complex)

def _gm6() -> np.ndarray:
    """λ₆ = [[0,0,0],[0,0,1],[0,1,0]]"""
    return np.array([[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0]], dtype=complex)

def _gm7() -> np.ndarray:
    """λ₇ = [[0,0,0],[0,0,-i],[0,i,0]]"""
    return np.array([[0, 0, 0],
                     [0, 0, -1j],
                     [0, 1j, 0]], dtype=complex)

def _gm8() -> np.ndarray:
    """λ₈ = (1/√3) [[1,0,0],[0,1,0],[0,0,-2]]"""
    return np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, -2]], dtype=complex) / np.sqrt(3)


# List of all Gell-Mann matrices (as a tuple for immutability)
GELL_MANN = (
    _gm1(), _gm2(), _gm3(), _gm4(),
    _gm5(), _gm6(), _gm7(), _gm8()
)

IDENTITY_3 = np.eye(3, dtype=complex)


# ---------------------------------------------------------------------------
# Core SU(3) properties
# ---------------------------------------------------------------------------

def structure_constants() -> np.ndarray:
    """
    Return the SU(3) structure constants f_abc where [λ_a, λ_b] = 2i f_abc λ_c.

    Returns
    -------
    f : ndarray, shape (8,8,8), dtype float
        f[a,b,c] = f_abc  (fully antisymmetric)
    """
    f = np.zeros((8, 8, 8), dtype=float)
    for a in range(8):
        for b in range(8):
            comm = GELL_MANN[a] @ GELL_MANN[b] - GELL_MANN[b] @ GELL_MANN[a]
            for c in range(8):
                # Tr(comm · λ_c) = 4i f_abc
                val = np.trace(comm @ GELL_MANN[c]).imag / 4.0
                if abs(val) > 1e-12:
                    f[a, b, c] = val
    return f


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def random_su3(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a random SU(3) matrix, approximately Haar-distributed.

    Method: exponential of random Lie algebra element, then projected via QR.

    Parameters
    ----------
    rng : np.random.Generator, optional

    Returns
    -------
    U : ndarray, shape (3,3), dtype complex
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = rng.uniform(-np.pi, np.pi, 8)
    H = 1j * sum(t * L for t, L in zip(theta, GELL_MANN))
    U = expm(H)
    return project_to_su3(U)


def project_to_su3(U: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """
    Project a matrix to the nearest SU(3) element.

    Algorithm:
      1. SVD: U = W S V†  => Q = W V†  (nearest unitary, Frobenius-optimal).
         Stable for any input including near-singular matrices.
      2. Fix det(Q) = 1 by absorbing the unit phase into the last column of W.
      3. Newton polish: Q <- Q (3I - Q†Q) / 2, then re-fix det (n_iter rounds).
    """
    # Step 1: Nearest unitary via SVD (numerically stable for any U)
    W, _, Vh = np.linalg.svd(U)
    Q = W @ Vh

    # Step 2: Fix det -> +1  (det(Q) is a unit complex number after SVD unitary)
    det = np.linalg.det(Q)
    W[:, 2] *= det.conj()   # absorb the phase into the last column of W
    Q = W @ Vh

    # Step 3: Newton unitarity refinement
    for _ in range(n_iter):
        Q = 0.5 * Q @ (3.0 * IDENTITY_3 - Q.conj().T @ Q)
        det = np.linalg.det(Q)
        Q[:, 2] /= det ** (1.0 / 3.0)

    return Q


def identity_su3() -> np.ndarray:
    """Return the 3×3 identity matrix (SU(3) identity)."""
    return IDENTITY_3.copy()


def small_random_su3(epsilon: float = 0.1,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a small random SU(3) matrix near identity for Metropolis proposals.

    Parameters
    ----------
    epsilon : float
        Typical size of algebra coefficients.
    rng : np.random.Generator, optional

    Returns
    -------
    delta_U : ndarray, shape (3,3)
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = rng.normal(0, epsilon, 8)
    H = 1j * sum(t * L for t, L in zip(theta, GELL_MANN))
    return project_to_su3(expm(H))


def random_su2_embedding(subgroup: str = '12',
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a random SU(2) element embedded in an SU(3) subgroup.

    Useful for Cabibbo-Marinari updates.

    Parameters
    ----------
    subgroup : str, {'12', '23', '13'}
    rng : np.random.Generator, optional

    Returns
    -------
    U_emb : ndarray, shape (3,3)
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = rng.uniform(0, 2 * np.pi)
    n = rng.normal(0, 1, 3)
    n = n / np.linalg.norm(n)

    c, s = np.cos(theta), np.sin(theta)
    nx, ny, nz = n
    su2 = np.array([[c + 1j * nz * s,  ny * s + 1j * nx * s],
                    [-ny * s + 1j * nx * s, c - 1j * nz * s]], dtype=complex)

    U_emb = IDENTITY_3.copy()
    if subgroup == '12':
        U_emb[0:2, 0:2] = su2
    elif subgroup == '23':
        U_emb[1:3, 1:3] = su2
    elif subgroup == '13':
        U_emb[0, 0] = su2[0, 0]
        U_emb[0, 2] = su2[0, 1]
        U_emb[2, 0] = su2[1, 0]
        U_emb[2, 2] = su2[1, 1]
    else:
        raise ValueError(f"Unknown subgroup: {subgroup!r}. Choose '12', '23', or '13'.")

    return U_emb


# ---------------------------------------------------------------------------
# Algebra operations
# ---------------------------------------------------------------------------

def dagger(U: np.ndarray) -> np.ndarray:
    """Hermitian conjugate (inverse for unitary matrices)."""
    return U.conj().T


def multiply(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Matrix product U @ V."""
    return U @ V


def su3_trace(U: np.ndarray) -> float:
    """
    Real part of Tr(U)/3.

    For SU(3) Wilson action: plaquette term = 1 - (1/3) Re Tr U_p.
    This normalisation matches the SU(2) convention 1 - (1/2) Tr U_p.
    """
    return np.trace(U).real / 3.0


def su3_det(U: np.ndarray) -> complex:
    """Determinant of U."""
    return np.linalg.det(U)


def commutator(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """[U, V] = U@V - V@U."""
    return U @ V - V @ U


def commutator_norm(U: np.ndarray, V: np.ndarray) -> float:
    """Frobenius norm of [U, V]. Measures degree of non-Abelianness."""
    return np.linalg.norm(commutator(U, V), 'fro')


def structure_constant_norm() -> float:
    """
    RMS of nonzero SU(3) structure constants f_abc.
    For SU(2): sqrt(6) ~ 2.449.
    """
    f = structure_constants()
    nonzero = f[f != 0]
    if len(nonzero) == 0:
        return 0.0
    return float(np.sqrt(np.mean(nonzero ** 2)))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_su3(U: np.ndarray, tol: float = 1e-10) -> Tuple[bool, str]:
    """
    Check whether U is a valid SU(3) matrix.

    Returns
    -------
    (valid, message) : bool, str
    """
    if U.shape != (3, 3):
        return False, f"Wrong shape: {U.shape}"

    uerr = np.linalg.norm(U @ dagger(U) - IDENTITY_3, 'fro')
    if uerr > tol:
        return False, f"Unitarity error {uerr:.2e} > {tol:.0e}"

    det = su3_det(U)
    if abs(det - 1.0) > tol:
        return False, f"|det-1| = {abs(det - 1):.2e} > {tol:.0e}"

    return True, "OK"


def unitarity_error(U: np.ndarray) -> float:
    """Frobenius norm of U@U† - I."""
    return np.linalg.norm(U @ dagger(U) - IDENTITY_3, 'fro')


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Internal validation suite. Run with: python su3.py"""
    PASS, FAIL = "PASS", "FAIL"
    results: list = []
    rng = np.random.default_rng(42)

    # T1: Gell-Mann matrices are traceless and Hermitian
    for i, L in enumerate(GELL_MANN):
        ok1a = abs(np.trace(L)) < 1e-12
        ok1b = np.allclose(L, L.conj().T, atol=1e-12)
        results.append((f"T1.{i+1} Gell-Mann lam_{i+1} (traceless+Hermitian)",
                        ok1a and ok1b,
                        f"tr={np.trace(L):.2e}, Hermitian={ok1b}"))

    # T2: structure constants antisymmetry
    f = structure_constants()
    ok2 = True
    for a in range(8):
        for b in range(8):
            for c in range(8):
                if abs(f[a, b, c] + f[b, a, c]) > 1e-10:
                    ok2 = False
                    break
    results.append(("T2 f_abc antisymmetric in (a,b)", ok2, ""))

    # T3: random_su3 produces valid SU(3) (100 trials)
    n_trials = 100
    valid, max_uerr = 0, 0.0
    for _ in range(n_trials):
        U = random_su3(rng)
        ok, _ = is_su3(U, tol=1e-8)
        if ok:
            valid += 1
        max_uerr = max(max_uerr, unitarity_error(U))
    ok3 = (valid == n_trials)
    results.append(("T3 random_su3 in SU(3) [100 trials]", ok3,
                    f"{valid}/{n_trials} valid, max uerr={max_uerr:.2e}"))

    # T4: small_random_su3 near identity
    # For theta_a ~ N(0, eps), H = sum_a theta_a lambda_a.
    # ||H||_F ~ eps * sqrt(sum_a ||lambda_a||_F^2) = eps * sqrt(8*2) = eps*4.
    # ||exp(iH) - I||_F ≈ ||H||_F ~ 4*eps; use 10*eps as generous upper bound.
    eps = 0.2
    U_small = small_random_su3(eps, rng)
    ok4a, msg4 = is_su3(U_small, tol=1e-8)
    dist = np.linalg.norm(U_small - IDENTITY_3, 'fro')
    ok4b = dist < 10.0 * eps     # generous Frobenius bound for 3x3 algebra
    ok4c = dist > 0.0            # moved from I
    results.append(("T4 small_random_su3 near I", ok4a and ok4b and ok4c,
                    f"dist={dist:.4f} (bound < {10.0*eps:.2f}), {msg4}"))

    # T5: identity_su3
    I = identity_su3()
    ok5, msg5 = is_su3(I)
    results.append(("T5 identity_su3", ok5, msg5))

    # T6: unitarity preserved
    U = random_su3(rng)
    err = unitarity_error(U)
    ok6 = err < 1e-10
    results.append(("T6 unitarity error < 1e-10", ok6, f"error={err:.2e}"))

    # T7: group closure (product is SU(3))
    U1, U2 = random_su3(rng), random_su3(rng)
    prod = multiply(U1, U2)
    ok7, msg7 = is_su3(prod, tol=1e-8)
    det_prod = su3_det(prod)
    results.append(("T7 group closure", ok7,
                    f"det(prod)={det_prod:.6f}, {msg7}"))

    # T8: non-commutativity
    cnorm = commutator_norm(U1, U2)
    ok8 = cnorm > 1e-6
    results.append(("T8 non-commutativity ||[U1,U2]|| > 0", ok8,
                    f"||[U1,U2]||_F = {cnorm:.4f}"))

    # T9: SU(2) embeddings are valid SU(3)
    for sub in ['12', '23', '13']:
        U_emb = random_su2_embedding(sub, rng)
        ok, msg = is_su3(U_emb, tol=1e-8)
        results.append((f"T9 SU(2) embedding [{sub}]", ok, msg))

    # T10: trace/3 convention
    U_rand = random_su3(rng)
    tr3 = su3_trace(U_rand)
    full_tr = np.trace(U_rand).real
    ok10 = abs(3 * tr3 - full_tr) < 1e-12
    results.append(("T10 trace/3 convention", ok10,
                    f"3*su3_trace={3*tr3:.6f}, Re Tr={full_tr:.6f}"))

    # T11: projector idempotence (project-of-projected == projected)
    U_raw = (np.random.default_rng(99).normal(0, 1, (3, 3))
             + 1j * np.random.default_rng(199).normal(0, 1, (3, 3)))
    U_p1 = project_to_su3(U_raw)
    ok11a, msg11 = is_su3(U_p1, tol=1e-8)
    U_p2 = project_to_su3(U_p1)
    re_proj_diff = np.linalg.norm(U_p1 - U_p2, 'fro')
    ok11b = re_proj_diff < 1e-10
    results.append(("T11 projector idempotence", ok11a and ok11b,
                    f"{msg11}, re-proj diff={re_proj_diff:.2e}"))

    # ── Print summary ───────────────────────────────────────────────────
    print("=" * 70)
    print("su3.py  --  SU(3) Algebra Self-Test")
    print("=" * 70)
    all_pass = True
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if detail:
            print(f"         {detail}")
        if not ok:
            all_pass = False
    print("-" * 70)

    n_tests = len(results)
    if all_pass:
        print(f"  All {n_tests} tests PASSED.")
        print()
        print("  SU(3) PROPERTIES:")
        f_norm = structure_constant_norm()
        print(f"    RMS structure constant f_abc  : {f_norm:.4f}  "
              f"(SU(2) ref: sqrt(6) ~ 2.449)")
        print(f"    Typical ||[U1,U2]||_F         : {commutator_norm(U1, U2):.4f}")
        print()
        print("  NEXT STEPS:")
        print("    1. Add su3.py to fields.py (triplet matter, gauge_group='SU3')")
        print("    2. Update action.py  (Wilson term: 1 - 1/3 Re Tr U_p)")
        print("    3. Modify updates.py (small_random_su3 for proposals)")
        print("    4. Run L=4 comparison with SU(2) at beta=6.0, kappa=0.3")
    else:
        n_fail = sum(1 for _, ok, _ in results if not ok)
        print(f"  {n_fail}/{n_tests} tests FAILED -- debug before integrating.")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
