"""
Task P: Null distribution for R on Erdős–Rényi random signed graphs.

Matches the positive-control gene networks (Appendix E):
  - N ∈ {16, 32, 64} nodes
  - Mean undirected degree ≈ 2  →  edge probability p = 2 / (N-1)
  - Signs: Bernoulli(0.35) activation (+1), else repression (-1)
  - No self-loops (directed → undirected: union of forward/backward edges)

Protocol: 10,000 networks per size; for each, compute
  R = #MIXED_triangles / #total_triangles
using the same E7 protocol (sign product over all 3-cliques in
the undirected signed graph after conflict resolution).

Reports:
  mean(R), std(R), P(R ∈ [0.32, 0.48]) for each N and pooled.
"""

import numpy as np
import json, os
from itertools import combinations

RNG_SEED   = 271828
N_NETS     = 10_000        # networks per size
SIZES      = [16, 32, 64]
P_ACTIV    = 0.35          # Bernoulli activation probability (matches Appendix E)
OUT_FILE   = os.path.join(os.path.dirname(__file__), "p_null_distribution_results.json")


# ──────────────────────────────────────────────────────────────────
# Graph generation
# ──────────────────────────────────────────────────────────────────

def random_signed_graph(rng, N):
    """
    Returns dict: frozenset({i,j}) -> sign ∈ {+1,-1}.
    Edge prob = 2/(N-1) to give mean undirected degree ≈ 2.
    """
    p_edge = 2.0 / (N - 1)
    edges = {}
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                sign = +1 if rng.random() < P_ACTIV else -1
                edges[frozenset({i, j})] = sign
    return edges


# ──────────────────────────────────────────────────────────────────
# MIXED fraction  (same algorithm as h_regulondb.py)
# ──────────────────────────────────────────────────────────────────

def mixed_fraction(edges):
    if not edges:
        return float("nan")
    # Adjacency
    adj = {}
    for pair in edges:
        a, b = tuple(pair)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    n_mixed = 0
    n_total = 0
    visited = set()
    for pair, s_ab in edges.items():
        a, b = tuple(pair)
        common = adj.get(a, set()) & adj.get(b, set())
        for c in common:
            tri = frozenset({a, b, c})
            if tri in visited:
                continue
            visited.add(tri)
            s_bc = edges.get(frozenset({b, c}))
            s_ca = edges.get(frozenset({c, a}))
            if s_bc is None or s_ca is None:
                continue
            n_total += 1
            if s_ab * s_bc * s_ca == -1:
                n_mixed += 1
    if n_total == 0:
        return float("nan")
    return n_mixed / n_total


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def run_for_size(N, rng, n_nets=N_NETS):
    R_vals = []
    no_tri = 0
    for k in range(n_nets):
        edges = random_signed_graph(rng, N)
        r = mixed_fraction(edges)
        if np.isnan(r):
            no_tri += 1
        else:
            R_vals.append(r)
        if (k + 1) % 2000 == 0:
            valid = np.array(R_vals)
            print(f"    N={N:3d}  {k+1:5d}/{n_nets}  "
                  f"mean={valid.mean():.3f}  no-tri={no_tri}")
    R_arr = np.array(R_vals)
    in_range = np.sum((R_arr >= 0.32) & (R_arr <= 0.48)) / len(R_arr)
    result = {
        "N": N, "n_nets": n_nets, "n_no_triangles": no_tri,
        "n_valid": len(R_arr),
        "R_mean": round(float(R_arr.mean()), 4),
        "R_std":  round(float(R_arr.std()),  4),
        "R_p2":   round(float(np.percentile(R_arr, 2.5)), 4),
        "R_p97":  round(float(np.percentile(R_arr, 97.5)), 4),
        "P_in_range_0.32_0.48": round(float(in_range), 4),
    }
    print(f"  N={N}: mean={result['R_mean']:.4f}  std={result['R_std']:.4f}  "
          f"P([0.32,0.48])={in_range:.3f}  no-tri={no_tri}/{n_nets}")
    return result, R_arr


def main():
    rng = np.random.default_rng(RNG_SEED)
    results = []
    all_R = []

    for N in SIZES:
        print(f"\n{'='*56}")
        print(f"  N = {N} nodes  (p_edge = {2/(N-1):.4f},  n_nets = {N_NETS:,})")
        print(f"{'='*56}")
        res, R_arr = run_for_size(N, rng)
        results.append(res)
        all_R.append(R_arr)

    # Pooled
    pooled = np.concatenate(all_R)
    in_range_pool = float(np.mean((pooled >= 0.32) & (pooled <= 0.48)))
    pooled_res = {
        "N": "pooled (16+32+64)",
        "n_valid": len(pooled),
        "R_mean": round(float(pooled.mean()), 4),
        "R_std":  round(float(pooled.std()),  4),
        "R_p2":   round(float(np.percentile(pooled, 2.5)), 4),
        "R_p97":  round(float(np.percentile(pooled, 97.5)), 4),
        "P_in_range_0.32_0.48": round(in_range_pool, 4),
    }
    results.append(pooled_res)

    with open(OUT_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n✓ Results written to {OUT_FILE}")

    print("\n" + "="*56)
    print("PAPER-READY SUMMARY")
    print("="*56)
    for r in results:
        print(f"  N={r['N']:>20s}:  R={r['R_mean']:.3f}±{r['R_std']:.3f}  "
              f"95%[{r['R_p2']:.3f},{r['R_p97']:.3f}]  "
              f"P([0.32,0.48])={r['P_in_range_0.32_0.48']:.3f}")


if __name__ == "__main__":
    main()
