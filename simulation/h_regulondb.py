"""
Task H: RegulonDB R computation using the E7 HGST protocol.

Network source: RegulonDB 11.2 (E. coli K-12)
Files:
  data/NetWorkTFGene.txt  -- TF -> gene interactions
  data/NetWorkTFTF.txt    -- TF -> TF interactions

Protocol (Appendix E of paper):
  1. Build undirected signed graph from all signed (+/-) interactions.
     Confidence subsets: (a) all, (b) Confirmed+Strong only.
     Conflict resolution for same node-pair with opposing signs: skip pair.
  2. Find all triangles (3-cliques in undirected graph).
  3. For each triangle, compute sign product s = sign(e12 * e23 * e31).
     Triangle is MIXED if s = -1.
  4. R = #MIXED / #triangles_total.
  5. Bootstrap 1000 random sign re-labellings to build null distribution.
  6. Report R_obs, R_null mean+/-std, p-value, 95% CI.
"""

import os, sys, time, json
import numpy as np
from collections import defaultdict
from itertools import combinations

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TFGENE   = os.path.join(DATA_DIR, "NetWorkTFGene.txt")
TFTF     = os.path.join(DATA_DIR, "NetWorkTFTF.txt")
N_BOOT   = 2000
RNG_SEED = 777


# ─────────────────────────────────────────────────────────────────
# 1. Parse RegulonDB files
# ─────────────────────────────────────────────────────────────────

def parse_file(path, high_conf_only=False):
    """
    Returns list of (nodeA, nodeB, sign) tuples where sign ∈ {+1, -1}.
    Drops '?' and dual-regulation entries.
    If high_conf_only: keep only Confirmed + Strong rows.
    Node labels are gene names (col 3 / col 5, lower-cased for consistency).
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            reg_gene   = parts[2].strip().lower()   # regulator gene name
            tgt_name   = parts[4].strip().lower()   # regulated name
            func       = parts[5].strip()
            conf       = parts[6].strip()

            if func not in ("+", "-"):
                continue
            if high_conf_only and conf not in ("Confirmed", "Strong"):
                continue

            sign = +1 if func == "+" else -1
            rows.append((reg_gene, tgt_name, sign))
    return rows


def build_graph(rows):
    """
    Build undirected signed graph from interaction rows.
    - Collect all (sign) values for each unordered node pair.
    - If all same sign: keep that sign.
    - If conflicting signs: SKIP that pair (conservative).
    Returns: {frozenset({a,b}) -> sign}
    """
    pair_signs = defaultdict(list)
    for a, b, s in rows:
        if a == b:          # skip self-loops
            continue
        pair_signs[frozenset({a, b})].append(s)

    edges = {}
    n_conflict = 0
    for pair, signs in pair_signs.items():
        pos = signs.count(+1)
        neg = signs.count(-1)
        if pos > 0 and neg > 0:
            n_conflict += 1
            continue            # conflicted pair: skip
        edges[pair] = +1 if pos > 0 else -1

    return edges, n_conflict


def graph_stats(edges):
    nodes = set()
    for pair in edges:
        nodes.update(pair)
    n_pos = sum(1 for s in edges.values() if s == +1)
    n_neg = sum(1 for s in edges.values() if s == -1)
    return len(nodes), len(edges), n_pos, n_neg


# ─────────────────────────────────────────────────────────────────
# 2. Triangle enumeration and MIXED fraction
# ─────────────────────────────────────────────────────────────────

def find_triangles(edges):
    """
    Return list of (nodeA, nodeB, nodeC, sign_product) for all triangles.
    sign_product = e_AB * e_BC * e_CA ∈ {+1, -1}.
    """
    # Adjacency: node -> set of neighbours
    adj = defaultdict(set)
    for pair in edges:
        a, b = tuple(pair)
        adj[a].add(b)
        adj[b].add(a)

    nodes = list(adj.keys())
    triangles = []

    # For efficiency: iterate over edges; for each edge find common neighbours
    for pair, s_ab in edges.items():
        a, b = tuple(pair)
        common = adj[a] & adj[b]
        for c in common:
            if c <= a or c <= b:   # string comparison to avoid double-counting
                continue
            s_bc = edges.get(frozenset({b, c}))
            s_ca = edges.get(frozenset({c, a}))
            if s_bc is None or s_ca is None:
                continue
            prod = s_ab * s_bc * s_ca
            triangles.append((a, b, c, prod))
    return triangles


def mixed_fraction(triangles):
    if not triangles:
        return float("nan"), 0, 0
    mixed = sum(1 for *_, p in triangles if p == -1)
    total = len(triangles)
    return mixed / total, mixed, total


# ─────────────────────────────────────────────────────────────────
# 3. Bootstrap null distribution
# ─────────────────────────────────────────────────────────────────

def bootstrap_obs_ci(triangles, n_boot=N_BOOT, seed=RNG_SEED):
    """
    Bootstrap confidence interval for R_obs by resampling triangles with replacement.
    Returns (mean, std, lo_2.5, hi_97.5).
    """
    rng = np.random.default_rng(seed + 1)
    prods = np.array([p for *_, p in triangles])
    n = len(prods)
    R_boot = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        R_boot[k] = np.sum(prods[idx] == -1) / n
    return (float(np.mean(R_boot)), float(np.std(R_boot)),
            float(np.percentile(R_boot, 2.5)), float(np.percentile(R_boot, 97.5)))


def bootstrap_null(edges, triangles, n_boot=N_BOOT, seed=RNG_SEED):
    """
    For each bootstrap replicate: randomly re-assign signs to all edges
    (preserving edge count and +/- ratio globally), re-compute R.
    Returns array of length n_boot.
    """
    rng = np.random.default_rng(seed)
    edge_list = list(edges.keys())
    n_edges   = len(edge_list)
    signs_arr = np.array(list(edges.values()))   # +1 / -1 array

    # Pre-compute triangle edge indices for fast look-up
    # Map pair -> index in edge_list
    pair_to_idx = {pair: i for i, pair in enumerate(edge_list)}

    tri_indices = []  # list of (idx_ab, idx_bc, idx_ca)
    for a, b, c, _ in triangles:
        ia = pair_to_idx[frozenset({a, b})]
        ib = pair_to_idx[frozenset({b, c})]
        ic = pair_to_idx[frozenset({c, a})]
        tri_indices.append((ia, ib, ic))

    tri_indices = np.array(tri_indices, dtype=np.int32)  # shape (T,3)
    n_tri = len(tri_indices)

    null_R = np.empty(n_boot)
    for k in range(n_boot):
        shuf = rng.permutation(signs_arr)
        # Vectorised product across triangle edges
        prods = shuf[tri_indices[:, 0]] * shuf[tri_indices[:, 1]] * shuf[tri_indices[:, 2]]
        null_R[k] = np.sum(prods == -1) / n_tri

    return null_R


# ─────────────────────────────────────────────────────────────────
# 4. Full analysis for one confidence subset
# ─────────────────────────────────────────────────────────────────

def run_analysis(label, high_conf_only):
    print(f"\n{'='*64}")
    print(f"SUBSET: {label}")
    print(f"{'='*64}")

    # Parse
    rows_gene = parse_file(TFGENE, high_conf_only=high_conf_only)
    rows_tf   = parse_file(TFTF,   high_conf_only=high_conf_only)
    all_rows  = rows_gene + rows_tf
    print(f"  Parsed {len(rows_gene)} TF-gene interactions, {len(rows_tf)} TF-TF interactions")
    print(f"  Combined signed rows: {len(all_rows)}")

    # Build graph
    edges, n_conflict = build_graph(all_rows)
    n_nodes, n_edges, n_pos, n_neg = graph_stats(edges)
    print(f"  Graph: {n_nodes} nodes, {n_edges} edges  "
          f"(+: {n_pos}, -: {n_neg}, conflict-skipped: {n_conflict})")
    frac_neg = n_neg / n_edges if n_edges else float("nan")
    print(f"  Negative edge fraction: {frac_neg:.3f}")

    # Triangles
    t0 = time.perf_counter()
    triangles = find_triangles(edges)
    t_tri = time.perf_counter() - t0
    R_obs, n_mixed, n_total = mixed_fraction(triangles)
    print(f"  Triangles found: {n_total}  (MIXED: {n_mixed})  [{t_tri:.1f}s]")
    print(f"  R_obs = {R_obs:.4f}  (MIXED fraction)")

    # Bootstrap null
    print(f"  Running {N_BOOT} bootstrap permutations …")
    t0 = time.perf_counter()
    null_R = bootstrap_null(edges, triangles, n_boot=N_BOOT, seed=RNG_SEED)
    t_boot = time.perf_counter() - t0
    R_null_mean = float(np.mean(null_R))
    R_null_std  = float(np.std(null_R))
    R_null_lo   = float(np.percentile(null_R, 2.5))
    R_null_hi   = float(np.percentile(null_R, 97.5))
    # One-sided lower tail: fraction of null <= R_obs (R_obs is significantly LOW)
    p_lower     = float(np.mean(null_R <= R_obs))
    # Two-tailed: symmetrise
    p_two       = float(2 * min(np.mean(null_R <= R_obs), np.mean(null_R >= R_obs)))
    z_score     = (R_obs - R_null_mean) / R_null_std if R_null_std > 0 else float("nan")
    print(f"  [{t_boot:.1f}s]")
    print(f"  R_null = {R_null_mean:.4f} ± {R_null_std:.4f}  "
          f"95% CI = [{R_null_lo:.4f}, {R_null_hi:.4f}]")
    print(f"  p-value (one-sided lower, R_null <= R_obs) = {p_lower:.4f}")
    print(f"  p-value (two-sided) = {p_two:.4f}")
    print(f"  z-score = {z_score:+.2f}σ")

    # Bootstrap CI on R_obs
    R_obs_mean, R_obs_ci_std, R_obs_lo, R_obs_hi = bootstrap_obs_ci(triangles)
    print(f"  R_obs bootstrap: {R_obs_mean:.4f} ± {R_obs_ci_std:.4f}  "
          f"95% CI = [{R_obs_lo:.4f}, {R_obs_hi:.4f}]")

    result = {
        "label": label,
        "high_conf_only": high_conf_only,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_conflict_dropped": n_conflict,
        "neg_edge_frac": round(frac_neg, 4),
        "n_triangles": n_total,
        "n_mixed": n_mixed,
        "R_obs": round(R_obs, 5),
        "R_obs_bootstrap_std": round(R_obs_ci_std, 5),
        "R_obs_ci95": [round(R_obs_lo, 5), round(R_obs_hi, 5)],
        "R_null_mean": round(R_null_mean, 5),
        "R_null_std": round(R_null_std, 5),
        "R_null_ci95": [round(R_null_lo, 5), round(R_null_hi, 5)],
        "p_lower": round(p_lower, 4),
        "p_two": round(p_two, 4),
        "z_score": round(z_score, 2),
        "n_boot": N_BOOT,
    }
    return result


# ─────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    results = []
    results.append(run_analysis("All signed (Confirmed+Strong+Weak)", high_conf_only=False))
    results.append(run_analysis("High-confidence only (Confirmed+Strong)", high_conf_only=True))

    # Save
    out_json = os.path.join(args.outdir, "h_regulondb_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n✓ Results written to {out_json}")

    # Summary for paper
    print("\n" + "="*64)
    print("PAPER-READY SUMMARY")
    print("="*64)
    for r in results:
        print(f"\n[{r['label']}]")
        print(f"  Network: {r['n_nodes']} nodes, {r['n_edges']} signed edges "
              f"({r['neg_edge_frac']*100:.1f}% repression)")
        print(f"  Triangles: {r['n_triangles']}  (MIXED: {r['n_mixed']})")
        print(f"  R_obs   = {r['R_obs']:.4f} ± {r['R_obs_bootstrap_std']:.4f}  "
              f"[{r['R_obs_ci95'][0]:.4f}, {r['R_obs_ci95'][1]:.4f}]")
        print(f"  R_null  = {r['R_null_mean']:.4f} ± {r['R_null_std']:.4f}")
        p2 = r['p_two']
        sig = "p<0.001***" if p2 < 0.001 else "p<0.01**" if p2 < 0.01 else f"p={p2:.4f}"
        print(f"  z = {r['z_score']:+.2f}σ   {sig}")


if __name__ == "__main__":
    main()
