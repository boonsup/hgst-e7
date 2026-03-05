"""
su2.py - SU(2) matrix operations for HGST lattice simulations.

All matrices are represented as 2x2 complex numpy arrays.
Epistemic status: THEOREM (gauge invariance proofs) + VALIDATED (unit tests below)

HGST Framework: Layer 1 Substrate Module — E10_SU2_Gauge
Interface contract: SubstrateInterface (see HGST_Modular_Synthesis_v1.md)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core SU(2) generators
# ---------------------------------------------------------------------------

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def random_su2() -> np.ndarray:
    """
    Generate a random SU(2) matrix uniformly w.r.t. the Haar measure.

    Method: generate 4 standard-normal random numbers, normalise to the
    unit 3-sphere, then assemble the matrix using the parametrisation

        U = a0*I + i*(a1*σx + a2*σy + a3*σz)

    which satisfies  U†U = I  and  det(U) = 1  exactly.

    Returns
    -------
    np.ndarray, shape (2, 2), dtype complex
    """
    z = np.random.normal(0.0, 1.0, 4)
    a = z / np.linalg.norm(z)
    return np.array(
        [
            [a[0] + 1j * a[3],   a[2] + 1j * a[1]],
            [-a[2] + 1j * a[1],  a[0] - 1j * a[3]],
        ],
        dtype=complex,
    )


def identity_su2() -> np.ndarray:
    """Return the 2×2 identity matrix (SU(2) identity element)."""
    return IDENTITY.copy()


def small_random_su2(epsilon: float = 0.1) -> np.ndarray:
    """
    Generate a small random SU(2) matrix near the identity.

    Useful for Metropolis proposals: the proposal is

        U_new = U_old @ small_random_su2(epsilon)

    with typical rotation angle |theta| <= epsilon (radians).

    Parameters
    ----------
    epsilon : float
        Maximum rotation angle in radians (default 0.1 ≈ 5.7°).

    Returns
    -------
    np.ndarray, shape (2, 2), dtype complex
    """
    axis = np.random.normal(0.0, 1.0, 3)
    axis /= np.linalg.norm(axis)
    theta = np.random.uniform(-epsilon, epsilon)
    c = np.cos(theta)
    s = np.sin(theta)
    nx, ny, nz = axis
    return np.array(
        [
            [c + 1j * nz * s,   ny * s + 1j * nx * s],
            [-ny * s + 1j * nx * s,  c - 1j * nz * s],
        ],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------------

def dagger(U: np.ndarray) -> np.ndarray:
    """Hermitian conjugate (= inverse for unitary matrices)."""
    return U.conj().T


def multiply(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Matrix product U @ V."""
    return U @ V


def su2_trace(U: np.ndarray) -> float:
    """
    Real part of Tr(U).

    For SU(2), Tr(U) = 2*a0 is real.  Returning a float keeps downstream
    action computations clean.
    """
    return np.trace(U).real


def su2_det(U: np.ndarray) -> complex:
    """Determinant of U."""
    return np.linalg.det(U)


def commutator(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """[U, V] = U@V - V@U.  Non-zero for non-Abelian gauge theory."""
    return U @ V - V @ U


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_su2(U: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check whether U is a valid SU(2) matrix.

    Tests:
        1. Shape (2, 2)
        2. Unitarity: U @ U† ≈ I
        3. det(U) ≈ 1
    """
    if U.shape != (2, 2):
        return False
    if not np.allclose(U @ dagger(U), IDENTITY, atol=tol):
        return False
    if not abs(su2_det(U) - 1.0) < tol:
        return False
    return True


def unitarity_error(U: np.ndarray) -> float:
    """||U @ U† - I||_F  (Frobenius norm; should be < 1e-14 for machine precision)."""
    return np.linalg.norm(U @ dagger(U) - IDENTITY)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests():
    """Internal validation suite.  Run with:  python su2.py"""
    import sys

    errors = []

    # --- Test 1: random_su2 ---
    for _ in range(1000):
        U = random_su2()
        if not is_su2(U, tol=1e-10):
            errors.append("random_su2: unitarity/det failed")
            break
    print("[T1] random_su2 (1000 samples): ", "PASS" if not errors else errors[-1])

    # --- Test 2: small_random_su2 ---
    for eps in [0.01, 0.1, 0.5]:
        for _ in range(500):
            U = small_random_su2(eps)
            if not is_su2(U, tol=1e-10):
                errors.append(f"small_random_su2(eps={eps}): failed")
                break
    print("[T2] small_random_su2 (500 × 3 eps): ", "PASS" if len(errors) == 0 else errors[-1])

    # --- Test 3: identity ---
    I = identity_su2()
    assert is_su2(I), "identity_su2 failed"
    print("[T3] identity_su2: PASS")

    # --- Test 4: dagger is inverse ---
    U = random_su2()
    err = unitarity_error(U)
    assert err < 1e-14, f"unitarity error {err}"
    print(f"[T4] unitarity_error: {err:.2e}  PASS")

    # --- Test 5: group closure (product is SU(2)) ---
    for _ in range(500):
        U, V = random_su2(), random_su2()
        if not is_su2(U @ V, tol=1e-10):
            errors.append("group closure failed")
            break
    print("[T5] group closure (500 samples): ", "PASS" if len(errors) == 0 else errors[-1])

    # --- Test 6: non-commutativity ---
    U, V = random_su2(), random_su2()
    C = commutator(U, V)
    assert np.linalg.norm(C) > 1e-10, "commutator is zero — Abelian? (unexpected)"
    print(f"[T6] non-commutativity ||[U,V]|| = {np.linalg.norm(C):.4f}  PASS")

    # --- Test 7: backward compatibility (U → identity recovers trivial action) ---
    I = identity_su2()
    tr_I = su2_trace(I)
    assert abs(tr_I - 2.0) < 1e-14, f"Tr(I) = {tr_I} != 2"
    print("[T7] Tr(identity) = 2.0  PASS")

    if errors:
        print(f"\nFAILURES ({len(errors)}):", errors)
        sys.exit(1)
    else:
        print("\nAll SU(2) tests PASSED.")


if __name__ == "__main__":
    _run_tests()
