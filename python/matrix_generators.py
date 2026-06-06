"""
matrix_generators.py
====================
Generates test sparse matrices in SciPy CSR format.

All matrices are symmetric positive (semi-)definite unless noted.
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional


# ─── 2D Laplacian ─────────────────────────────────────────────────────────────

def make_laplacian_2d(L: int) -> sp.csr_matrix:
    """
    Finite-difference 5-point Laplacian on an L×L grid.
    N = L², nnz ≈ 5N, condition number O(L²).

    Ordering: row-major (x changes fastest).
    """
    N = L * L
    diag     = np.full(N,  4.0)
    off_horiz = np.full(N - 1, -1.0)
    off_vert  = np.full(N - L, -1.0)

    # Remove horizontal connections that wrap around rows
    for i in range(1, L):
        off_horiz[i * L - 1] = 0.0

    A = (sp.diags(diag,       0,  shape=(N, N))
       + sp.diags(off_horiz,  1,  shape=(N, N))
       + sp.diags(off_horiz, -1,  shape=(N, N))
       + sp.diags(off_vert,   L,  shape=(N, N))
       + sp.diags(off_vert,  -L,  shape=(N, N)))
    return A.tocsr().astype(np.float64)


# ─── 3D Laplacian ─────────────────────────────────────────────────────────────

def make_laplacian_3d(L: int) -> sp.csr_matrix:
    """
    Finite-difference 7-point Laplacian on an L×L×L grid.
    N = L³, nnz ≈ 7N, condition number O(L²).
    """
    N = L * L * L
    Lsq = L * L

    # Self and 6 neighbours
    diag = np.full(N, 6.0)
    A = sp.diags(diag, 0, shape=(N, N), format='lil')

    for idx in range(N):
        z =  idx // Lsq
        y = (idx % Lsq) // L
        x =  idx % L

        for (dz, dy, dx) in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < L and 0 <= ny < L and 0 <= nz < L:
                nidx = nz * Lsq + ny * L + nx
                A[idx, nidx] = -1.0

    return A.tocsr().astype(np.float64)


def make_laplacian_3d_fast(L: int) -> sp.csr_matrix:
    """
    Kronecker-product construction of the 3D Laplacian — much faster
    than the loop-based version for large L.
    """
    I  = sp.eye(L, format='csr')
    T1 = sp.diags([-1., 2., -1.], [-1, 0, 1], shape=(L, L), format='csr')

    # 3D Laplacian = T⊗I⊗I + I⊗T⊗I + I⊗I⊗T
    A = (sp.kron(sp.kron(T1, I), I)
       + sp.kron(sp.kron(I, T1), I)
       + sp.kron(sp.kron(I, I), T1))
    return A.tocsr().astype(np.float64)


# ─── Lattice gauge (Dirac-like) ───────────────────────────────────────────────

def make_lattice_gauge(L: int, mass: float = 0.1,
                       rng: Optional[np.random.Generator] = None) -> sp.csr_matrix:
    """
    Simplified 2D lattice gauge Dirac operator on an L×L torus.
    N = L², nnz ≈ 9N (nearest-neighbour + self), *not* symmetric.

    In real QCD the Dirac operator acts on spinor-colour space, but this
    scalar toy version captures the key sparsity structure (fixed nnz/row,
    structured but non-trivial hopping, periodic boundary).

    Uses random U(1) gauge links to break translation invariance.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = L * L
    # Gauge links: random phases ∈ [0, 2π) for each direction and site
    phi_x = rng.uniform(0, 2 * np.pi, (L, L))   # links in +x direction
    phi_y = rng.uniform(0, 2 * np.pi, (L, L))

    rows, cols, vals = [], [], []

    def idx(x, y):
        return (y % L) * L + (x % L)

    for y in range(L):
        for x in range(L):
            i = idx(x, y)
            # Self (mass term)
            rows.append(i); cols.append(i); vals.append(4.0 + mass)
            # ±x hopping
            j_px = idx(x + 1, y)
            rows.append(i); cols.append(j_px)
            vals.append(-0.5 * np.exp( 1j * phi_x[y, x]).real)

            j_mx = idx(x - 1, y)
            rows.append(i); cols.append(j_mx)
            vals.append(-0.5 * np.exp(-1j * phi_x[y, (x - 1) % L]).real)

            # ±y hopping
            j_py = idx(x, y + 1)
            rows.append(i); cols.append(j_py)
            vals.append(-0.5 * np.exp( 1j * phi_y[y, x]).real)

            j_my = idx(x, y - 1)
            rows.append(i); cols.append(j_my)
            vals.append(-0.5 * np.exp(-1j * phi_y[(y - 1) % L, x]).real)

    A = sp.csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(N, N))
    # Make symmetric (Hermitian) for CG compatibility
    A = 0.5 * (A + A.T)
    return A.tocsr()


# ─── Random sparse ────────────────────────────────────────────────────────────

def make_random_sparse(N: int, nnz_per_row: int = 10,
                       rng: Optional[np.random.Generator] = None,
                       symmetric: bool = True) -> sp.csr_matrix:
    """
    Random sparse SPD matrix with approximately `nnz_per_row` non-zeros per row.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Build lower triangle + diagonal
    rows, cols, vals = [], [], []
    for i in range(N):
        # Random off-diagonal non-zeros in [0, i)
        n_off = min(nnz_per_row - 1, i)
        if n_off > 0:
            js = rng.choice(i, size=n_off, replace=False)
            for j in js:
                v = rng.uniform(0.1, 1.0)
                rows.append(i); cols.append(j); vals.append(-v)
                if symmetric:
                    rows.append(j); cols.append(i); vals.append(-v)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
    # Diagonal dominance ensures SPD
    diag_vals = np.abs(A).sum(axis=1).A1 + 1.0
    A = A + sp.diags(diag_vals)
    return A.tocsr()


# ─── Power-law (GNN-like) ─────────────────────────────────────────────────────

def make_power_law(N: int, m: int = 3,
                   rng: Optional[np.random.Generator] = None) -> sp.csr_matrix:
    """
    Barabási–Albert preferential attachment graph (power-law degree distribution).
    Mimics social network / GNN adjacency matrices.
    Returns the graph Laplacian (SPD).

    N : number of nodes
    m : edges added per node in BA construction
    """
    if rng is None:
        rng = np.random.default_rng(1)

    # Simple BA construction
    edges = set()
    degrees = np.zeros(N, dtype=np.int64)

    # Seed with a small clique
    seed = min(m + 1, N)
    for i in range(seed):
        for j in range(i):
            edges.add((i, j))
            degrees[i] += 1
            degrees[j] += 1

    for node in range(seed, N):
        deg_sum = degrees[:node].sum()
        if deg_sum == 0:
            targets = rng.choice(node, size=min(m, node), replace=False)
        else:
            p = degrees[:node] / deg_sum
            targets = rng.choice(node, size=min(m, node), replace=False, p=p)
        for t in targets:
            edges.add((node, t))
            degrees[node] += 1
            degrees[t]    += 1

    rows, cols = zip(*[(u, v) for u, v in edges]) if edges else ([], [])
    rows = list(rows) + list(cols)
    cols = list(cols) + list(rows[:len(edges)])
    vals = [-1.0] * len(rows)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
    # Graph Laplacian: D - A
    d = np.abs(A).sum(axis=1).A1
    L = sp.diags(d) - A
    return L.tocsr()


# ─── Registry ─────────────────────────────────────────────────────────────────

MATRIX_REGISTRY = {
    'laplacian_2d':   lambda N: make_laplacian_2d(int(N**0.5)),
    'laplacian_3d':   lambda N: make_laplacian_3d_fast(int(round(N**(1/3)))),
    'lattice_gauge':  lambda N: make_lattice_gauge(int(N**0.5)),
    'random_sparse':  lambda N: make_random_sparse(N),
    'power_law':      lambda N: make_power_law(N),
}


def get_matrix(name: str, N: int) -> sp.csr_matrix:
    """
    Convenience factory.

    Parameters
    ----------
    name : one of 'laplacian_2d', 'laplacian_3d', 'lattice_gauge',
                  'random_sparse', 'power_law'
    N    : approximate number of rows (rounded to nearest valid size)
    """
    if name not in MATRIX_REGISTRY:
        raise ValueError(f"Unknown matrix '{name}'. "
                         f"Choose from: {list(MATRIX_REGISTRY)}")
    return MATRIX_REGISTRY[name](N)


if __name__ == '__main__':
    import sys

    for name, factory in MATRIX_REGISTRY.items():
        try:
            A = factory(10_000)
            print(f"  {name:20s}  shape={A.shape}  nnz={A.nnz:>10,}"
                  f"  nnz/row={A.nnz/A.shape[0]:.1f}")
        except Exception as e:
            print(f"  {name:20s}  ERROR: {e}", file=sys.stderr)
