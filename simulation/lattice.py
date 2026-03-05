"""
lattice.py - 2D grade lattice (n,m) ∈ ℕ² for HGST simulations.

HGST Framework: Layer 3 Structure Module — GradeLattice_NxN
Interface contract: StructureInterface (see HGST_Modular_Synthesis_v1.md)

Key properties (from HGST Canonical v5):
  • Sites:      (n,m) with n ∈ [1,L], m ∈ [1,L]
  • Grade mag:  r_{n,m} = 2^(n-m)   (alpha/beta tower ratio)
  • Closure:    r_{n,m} * r_{m,n} = 1  (field closure theorem)
  • Edges:      directed, nearest-neighbour only (open boundaries default)
  • Plaquettes: minimal closed loops of 4 sites (counterclockwise)

Epistemic status: VALIDATED — grade structure follows from HGST THEOREM 1-2.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict


# Type aliases
Site    = int                      # linear site index
Edge    = Tuple[Site, Site]        # directed edge (source, target)
Plaquette = Tuple[Site, Site, Site, Site]  # (i, j, k, l) counterclockwise


class Lattice2D:
    """
    Square grade lattice of size L×L with open boundaries.

    Site indexing (column-major, matching HGST (n,m) convention):
        site_index = (n-1) + (m-1)*L   →   n = site%L + 1,  m = site//L + 1

    Edges are directed; only right (+n) and up (+m) neighbours are stored as
    canonical edges.  Both orientations are accessible via the edge_index dict.

    Parameters
    ----------
    L : int
        Lattice side length.  Total sites = L².
    """

    def __init__(self, L: int) -> None:
        if L < 2:
            raise ValueError("L must be >= 2 for plaquettes to exist.")
        self.L = L
        self.N = L * L

        self._build_adjacency()
        self._precompute_grade_magnitudes()

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> None:
        """
        Build site neighbour lists, directed edge list, and plaquette list.

        Edges stored: canonical (right, up) directions only.
        Plaquettes: oriented counterclockwise:
            i0 (bottom-left) → i1 (bottom-right) → i2 (top-right) → i3 (top-left)
        """
        L = self.L

        # --- edges ---
        self._edges: List[Edge] = []
        self._neighbors: List[List[Site]] = [[] for _ in range(self.N)]

        for m_idx in range(L):          # m_idx = m - 1
            for n_idx in range(L):      # n_idx = n - 1
                i = n_idx + m_idx * L

                # Right neighbour: (n+1, m)
                if n_idx + 1 < L:
                    j = (n_idx + 1) + m_idx * L
                    self._edges.append((i, j))
                    self._neighbors[i].append(j)

                # Up neighbour: (n, m+1)
                if m_idx + 1 < L:
                    j = n_idx + (m_idx + 1) * L
                    self._edges.append((i, j))
                    self._neighbors[i].append(j)

        # Build a set of all canonical edges for fast lookup
        self._edge_set: set = set(self._edges)

        # --- plaquettes (counterclockwise) ---
        self._plaquettes: List[Plaquette] = []
        for m_idx in range(L - 1):
            for n_idx in range(L - 1):
                i0 = n_idx       + m_idx       * L   # bottom-left
                i1 = (n_idx + 1) + m_idx       * L   # bottom-right
                i2 = (n_idx + 1) + (m_idx + 1) * L   # top-right
                i3 = n_idx       + (m_idx + 1) * L   # top-left
                self._plaquettes.append((i0, i1, i2, i3))

        # --- precompute which edges belong to each plaquette ---
        # For each canonical edge, store list of plaquette indices and the
        # four-edge sequence (for fast local action change computation later).
        self._edge_to_plaquettes: Dict[Edge, List[int]] = {e: [] for e in self._edges}
        for p_idx, (i0, i1, i2, i3) in enumerate(self._plaquettes):
            # Plaquette traversal: i0→i1 → i1→i2 → i2→i3 → i3→i0
            for e in [(i0, i1), (i1, i2), (i2, i3), (i3, i0)]:
                canon = e if e in self._edge_set else (e[1], e[0])
                if canon in self._edge_to_plaquettes:
                    self._edge_to_plaquettes[canon].append(p_idx)

    def _precompute_grade_magnitudes(self) -> None:
        """
        Precompute  r_{n,m} = 2^(n-m)  for each site.

        From HGST THEOREM 2 (field closure):
            r_{n,m} × r_{m,n} = 2^(n-m) × 2^(m-n) = 1  ✓
        """
        self.r = np.empty(self.N, dtype=float)
        for idx in range(self.N):
            n, m = self.grade_indices(idx)
            self.r[idx] = 2.0 ** (n - m)

    # ------------------------------------------------------------------
    # Public interface (StructureInterface contract)
    # ------------------------------------------------------------------

    def sites(self) -> List[Site]:
        """All site indices [0 .. N-1]."""
        return list(range(self.N))

    def edges(self) -> List[Edge]:
        """All directed canonical edges (i→j with i < j in layout order)."""
        return list(self._edges)

    def plaquettes(self) -> List[Plaquette]:
        """All minimal plaquettes (i0, i1, i2, i3) counterclockwise."""
        return list(self._plaquettes)

    def plaquettes_of_edge(self, edge: Edge) -> List[int]:
        """Indices of plaquettes that contain the given canonical edge."""
        canon = edge if edge in self._edge_set else (edge[1], edge[0])
        return self._edge_to_plaquettes.get(canon, [])

    def lattice_spacing(self) -> float:
        """
        Current lattice spacing (implicit unit; Δn = Δm = 1).
        Returns 1.0 for the discrete grade lattice.
        """
        return 1.0

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def site_index(self, n: int, m: int) -> Site:
        """Convert 1-indexed (n, m) to linear site index."""
        if not (1 <= n <= self.L and 1 <= m <= self.L):
            raise IndexError(f"(n={n}, m={m}) out of range for L={self.L}")
        return (n - 1) + (m - 1) * self.L

    def grade_indices(self, idx: Site) -> Tuple[int, int]:
        """Return 1-indexed (n, m) for a linear site index."""
        n = idx % self.L + 1
        m = idx // self.L + 1
        return n, m

    def neighbors(self, site: Site) -> List[Site]:
        """Outgoing neighbour indices for a site."""
        return self._neighbors[site]

    def has_edge(self, i: Site, j: Site) -> bool:
        """True if directed edge (i,j) is a canonical edge of the lattice."""
        return (i, j) in self._edge_set

    # ------------------------------------------------------------------
    # Derived counts (for sanity checks)
    # ------------------------------------------------------------------

    @property
    def n_sites(self) -> int:
        return self.N

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @property
    def n_plaquettes(self) -> int:
        return len(self._plaquettes)

    def __repr__(self) -> str:
        return (
            f"Lattice2D(L={self.L}, sites={self.N}, "
            f"edges={self.n_edges}, plaquettes={self.n_plaquettes})"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests():
    import sys

    errors = []

    for L in [2, 3, 4, 6, 8]:
        lat = Lattice2D(L)

        # T1: site count
        assert lat.N == L * L, f"L={L}: N mismatch"

        # T2: edge count for open square lattice = 2*L*(L-1)
        expected_edges = 2 * L * (L - 1)
        assert lat.n_edges == expected_edges, \
            f"L={L}: edges {lat.n_edges} != {expected_edges}"

        # T3: plaquette count = (L-1)^2
        expected_plaq = (L - 1) ** 2
        assert lat.n_plaquettes == expected_plaq, \
            f"L={L}: plaquettes {lat.n_plaquettes} != {expected_plaq}"

        # T4: grade magnitudes -- r_{n,m} = 2^(n-m)
        for idx in range(lat.N):
            n, m = lat.grade_indices(idx)
            expected_r = 2.0 ** (n - m)
            assert abs(lat.r[idx] - expected_r) < 1e-14, \
                f"L={L}: r mismatch at ({n},{m})"

        # T5: field closure theorem -- r * r_inv = 1
        for idx in range(lat.N):
            n, m = lat.grade_indices(idx)
            inv_idx = lat.site_index(m, n) if (1 <= m <= L and 1 <= n <= L) else None
            if inv_idx is not None:
                assert abs(lat.r[idx] * lat.r[inv_idx] - 1.0) < 1e-14, \
                    f"L={L}: field closure failed at ({n},{m})"

        # T6: site_index / grade_indices round-trip
        for n in range(1, L + 1):
            for m in range(1, L + 1):
                idx = lat.site_index(n, m)
                n2, m2 = lat.grade_indices(idx)
                assert (n, m) == (n2, m2), f"Round-trip failed: ({n},{m})"

        # T7: plaquettes contain only canonical edges
        edge_set = set(lat.edges())
        for (i0, i1, i2, i3) in lat.plaquettes():
            for src, tgt in [(i0, i1), (i1, i2), (i2, i3), (i3, i0)]:
                # Each plaquette edge must exist as either (src,tgt) or (tgt,src)
                assert (src, tgt) in edge_set or (tgt, src) in edge_set, \
                    f"L={L}: plaquette edge ({src},{tgt}) not in edge set"

        print(f"[L={L}] {lat}  PASS")

    # T8: explicit L=2 geometry verification
    lat2 = Lattice2D(2)
    # Site layout: (1,1)=0, (2,1)=1, (1,2)=2, (2,2)=3
    assert lat2.site_index(1, 1) == 0
    assert lat2.site_index(2, 1) == 1
    assert lat2.site_index(1, 2) == 2
    assert lat2.site_index(2, 2) == 3
    # Grade magnitudes for L=2
    expected = {0: 1.0, 1: 2.0, 2: 0.5, 3: 1.0}
    for idx, val in expected.items():
        assert abs(lat2.r[idx] - val) < 1e-14, f"L=2 r[{idx}]={lat2.r[idx]} expected {val}"
    # Single plaquette at L=2
    assert lat2.n_plaquettes == 1
    print(f"[T8] L=2 explicit geometry  PASS")

    if errors:
        print("FAILURES:", errors)
        sys.exit(1)
    else:
        print("\nAll Lattice2D tests PASSED.")


if __name__ == "__main__":
    _run_tests()
