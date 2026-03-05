"""
Task N: Compute C(G) = E[||[U1,U2]||_F] for G = SU(2) and SU(3)
by Monte Carlo integration over the Haar measure.

Haar-random SU(N) matrices are generated via the QR method:
  1. Draw X ~ GinOE (matrix of iid standard complex Gaussians).
  2. QR-decompose X = QR.
  3. Normalise phases: Q <- Q @ diag(R_ii / |R_ii|).
  Result Q is Haar-distributed over U(N); enforce det=1 for SU(N).

N_SAMPLES = 200_000 pairs per group.
"""

import numpy as np
import json, os

RNG_SEED   = 314
N_SAMPLES  = 200_000   # pairs (U1, U2) per group
OUT_FILE   = os.path.join(os.path.dirname(__file__), "n_commutator_results.json")


# ──────────────────────────────────────────────────────────────────
# Haar-random SU(N) sampler
# ──────────────────────────────────────────────────────────────────

def haar_sun_batch(rng, n, k):
    """Return (k, n, n) array of k independent Haar SU(n) matrices."""
    X = (rng.standard_normal((k, n, n)) + 1j * rng.standard_normal((k, n, n))) / np.sqrt(2)
    Q, R = np.linalg.qr(X)                         # Q: (k,n,n), R: (k,n,n)
    # Canonical form: multiply columns of Q by sign of diagonal of R
    phases = np.exp(-1j * np.angle(np.diagonal(R, axis1=-2, axis2=-1)))   # (k,n)
    Q = Q * phases[:, np.newaxis, :]                # broadcast over rows
    # Project to SU(n): divide by det^(1/n)
    dets = np.linalg.det(Q)                         # (k,)
    Q = Q / (dets ** (1.0 / n))[:, np.newaxis, np.newaxis]
    return Q


# ──────────────────────────────────────────────────────────────────
# Frobenius commutator norm
# ──────────────────────────────────────────────────────────────────

def commutator_norms_batch(U1, U2):
    """
    Given (k,n,n) U1 and U2, return (k,) array of ||[U1,U2]||_F = ||U1 U2 - U2 U1||_F.
    """
    C = U1 @ U2 - U2 @ U1                          # (k,n,n) complex
    # Frobenius norm: sqrt(Tr(C† C)) = sqrt(sum |C_ij|²)
    norms = np.sqrt(np.sum(np.abs(C) ** 2, axis=(-2, -1)))   # (k,)
    return norms.real


# ──────────────────────────────────────────────────────────────────
# Main computation
# ──────────────────────────────────────────────────────────────────

def compute_C(group_name, n, rng, n_samples=N_SAMPLES, batch=10_000):
    print(f"\n  Computing C({group_name})  n_samples={n_samples:,}  batch={batch:,}")
    norms = []
    done = 0
    while done < n_samples:
        k = min(batch, n_samples - done)
        U1 = haar_sun_batch(rng, n, k)
        U2 = haar_sun_batch(rng, n, k)
        norms.append(commutator_norms_batch(U1, U2))
        done += k
        if done % 50_000 == 0:
            partial = np.concatenate(norms)
            print(f"    {done:7,}/{n_samples:,}  running mean = {partial.mean():.4f}")
    all_norms = np.concatenate(norms)
    mean  = float(all_norms.mean())
    std   = float(all_norms.std())
    sem   = float(std / np.sqrt(n_samples))
    ci95  = [round(mean - 1.96 * sem, 4), round(mean + 1.96 * sem, 4)]
    # Bootstrap CI on the mean (fast: just via CLT)
    print(f"  C({group_name}) = {mean:.4f} ± {std:.4f}  (mean ± s.d.)")
    print(f"  95% CI on mean: [{ci95[0]:.4f}, {ci95[1]:.4f}]")
    return {"group": group_name, "n": n, "n_samples": n_samples,
            "mean": round(mean, 5), "std": round(std, 5),
            "sem": round(sem, 6), "ci95_mean": ci95}


def main():
    rng = np.random.default_rng(RNG_SEED)

    # U(1): trivially 0
    print("  C(U(1)) = 0.0000 (Abelian, exact)")
    results = [{"group": "U(1)", "n": 1, "n_samples": 0,
                "mean": 0.0, "std": 0.0, "sem": 0.0, "ci95_mean": [0.0, 0.0]}]

    # SU(2)
    r2 = compute_C("SU(2)", n=2, rng=rng)
    results.append(r2)

    # SU(3)
    r3 = compute_C("SU(3)", n=3, rng=rng)
    results.append(r3)

    with open(OUT_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n✓ Results written to {OUT_FILE}")

    # Paper-ready
    print("\n" + "="*56)
    print("PAPER-READY VALUES")
    print("="*56)
    for r in results:
        if r["n_samples"] == 0:
            print(f"  C({r['group']}) = 0  (exact, Abelian)")
        else:
            print(f"  C({r['group']}) = {r['mean']:.4f} ± {r['sem']:.4f}  "
                  f"(Monte Carlo, N={r['n_samples']:,}, 95% CI {r['ci95_mean']})")


if __name__ == "__main__":
    main()
