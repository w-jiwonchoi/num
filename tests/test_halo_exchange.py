"""
test_halo_exchange.py
=====================
Python-level tests for the halo exchange / distributed SpMV communication layer.

These tests focus specifically on the correctness of rank-to-rank data exchange,
ghost buffer population, and the row-partition logic — separately from the SpMV
arithmetic itself.

Run:
    mpiexec -np 2 python3 tests/test_halo_exchange.py
    mpiexec -np 4 python3 tests/test_halo_exchange.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import kokkos_spmv as kspmv
    HAS_KSPMV = True
except ImportError:
    HAS_KSPMV = False

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# ── Test bookkeeping ──────────────────────────────────────────────────────────
_pass = 0
_fail = 0
PASS_MARK = "\u2713"
FAIL_MARK = "\u2717"


def _check(name: str, cond: bool, detail: str = "") -> bool:
    global _pass, _fail
    mark = PASS_MARK if cond else FAIL_MARK
    suffix = f"  [{detail}]" if (detail and not cond) else ""
    print(f"  {mark}  {name}{suffix}")
    if cond:
        _pass += 1
    else:
        _fail += 1
    return cond


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denom) if denom != 0 else float(np.linalg.norm(a))


# ── Helpers: build small structured test matrices ─────────────────────────────

def _make_1d_laplacian(N: int) -> sp.csr_matrix:
    """Tridiagonal [-1, 2, -1] 1-D Laplacian, known ghost pattern."""
    diag     = np.full(N,  2.0)
    off_diag = np.full(N - 1, -1.0)
    A = (sp.diags(diag, 0, shape=(N, N))
       + sp.diags(off_diag,  1, shape=(N, N))
       + sp.diags(off_diag, -1, shape=(N, N)))
    return A.tocsr().astype(np.float64)


def _make_2d_laplacian(L: int) -> sp.csr_matrix:
    """5-point 2D Laplacian on L×L grid."""
    N = L * L
    I = sp.eye(L, format='csr')
    T = sp.diags([-1., 2., -1.], [-1, 0, 1], shape=(L, L), format='csr')
    A = sp.kron(T, I) + sp.kron(I, T)
    return A.tocsr().astype(np.float64)


# ── Test 1: Row partition correctness ────────────────────────────────────────

def test_row_partition() -> bool:
    """
    Check that each rank's local_row_start / local_row_end partitions
    [0, global_nrows) exactly, with no gaps or overlaps.
    """
    if not (HAS_CUPY and HAS_KSPMV and HAS_MPI):
        print("  [SKIP] test_row_partition")
        return True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 127  # intentionally not divisible by common GPU counts
    A = _make_1d_laplacian(N)

    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        A.shape[0], A.shape[1])

    my_start = A_dist.local_row_start
    my_end   = A_dist.local_row_end

    # Gather all [start, end) from every rank
    all_starts = comm.allgather(my_start)
    all_ends   = comm.allgather(my_end)

    all_ok = True
    if rank == 0:
        # Starts must be sorted and strictly increasing
        ok_order = all(all_starts[i] < all_starts[i+1] for i in range(size-1))
        all_ok &= _check("row_partition: starts are ordered", ok_order,
                          str(all_starts))

        # Contiguous: end[i] == start[i+1]
        ok_cont = all(all_ends[i] == all_starts[i+1] for i in range(size-1))
        all_ok &= _check("row_partition: ranges are contiguous", ok_cont,
                          f"ends={all_ends} starts={all_starts}")

        # Full coverage
        ok_full = (all_starts[0] == 0 and all_ends[-1] == N)
        all_ok &= _check(f"row_partition: covers [0, {N})", ok_full,
                          f"first_start={all_starts[0]} last_end={all_ends[-1]}")

        # Near-even load balance: max - min <= 1
        local_sizes = [all_ends[r] - all_starts[r] for r in range(size)]
        ok_balance = max(local_sizes) - min(local_sizes) <= 1
        all_ok &= _check("row_partition: load balanced (max-min <= 1)", ok_balance,
                          str(local_sizes))

    return all_ok


# ── Test 2: Ghost buffer population — 1D tridiagonal ─────────────────────────

def test_ghost_1d_tridiagonal() -> bool:
    """
    For the 1D Laplacian, rank r needs exactly two ghost values:
      - the last element of rank r-1 (left neighbour)
      - the first element of rank r+1 (right neighbour)
    Verify by comparing distributed SpMV with global SciPy result.
    """
    if not (HAS_CUPY and HAS_KSPMV and HAS_MPI):
        print("  [SKIP] test_ghost_1d_tridiagonal")
        return True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("  [SKIP] test_ghost_1d_tridiagonal (need >= 2 ranks)")
        return True

    N = 200
    A = _make_1d_laplacian(N)
    rng = np.random.default_rng(10)
    x_global = rng.standard_normal(N)
    y_ref = np.asarray(A @ x_global, dtype=np.float64)

    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        N, N)

    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end
    x_local = cp.asarray(x_global[local_start:local_end])
    y_local = cp.zeros(local_end - local_start, dtype=cp.float64)

    kspmv.dist_spmv(A_dist, x_local, y_local)
    y_local_np = cp.asnumpy(y_local)

    counts = np.array(comm.allgather(y_local_np.size))
    displacements = np.concatenate([[0], np.cumsum(counts[:-1])])
    y_gathered = None
    if rank == 0:
        y_gathered = np.empty(N, dtype=np.float64)

    comm.Gatherv(y_local_np,
                 [y_gathered, counts, displacements, MPI.DOUBLE],
                 root=0)

    all_ok = True
    if rank == 0:
        err = _rel_err(y_gathered, y_ref)
        all_ok = _check(f"ghost_1d_tridiagonal N={N} np={size}  rel_err={err:.2e}",
                         err < 1e-10)
    return all_ok


# ── Test 3: Ghost buffer population — 2D Laplacian ───────────────────────────

def test_ghost_2d_laplacian() -> bool:
    """
    The 2D Laplacian has a richer ghost pattern (full rows of ghost data
    from neighbouring ranks).  Verify distributed SpMV correctness.
    """
    if not (HAS_CUPY and HAS_KSPMV and HAS_MPI):
        print("  [SKIP] test_ghost_2d_laplacian")
        return True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("  [SKIP] test_ghost_2d_laplacian (need >= 2 ranks)")
        return True

    L = 16  # 16×16 = 256 rows
    A = _make_2d_laplacian(L)
    N = A.shape[0]
    rng = np.random.default_rng(11)
    x_global = rng.standard_normal(N)
    y_ref = np.asarray(A @ x_global, dtype=np.float64)

    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        N, N)

    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end
    x_local = cp.asarray(x_global[local_start:local_end])
    y_local = cp.zeros(local_end - local_start, dtype=cp.float64)

    kspmv.dist_spmv(A_dist, x_local, y_local)
    y_local_np = cp.asnumpy(y_local)

    counts = np.array(comm.allgather(y_local_np.size))
    displacements = np.concatenate([[0], np.cumsum(counts[:-1])])
    y_gathered = None
    if rank == 0:
        y_gathered = np.empty(N, dtype=np.float64)

    comm.Gatherv(y_local_np,
                 [y_gathered, counts, displacements, MPI.DOUBLE],
                 root=0)

    all_ok = True
    if rank == 0:
        err = _rel_err(y_gathered, y_ref)
        all_ok = _check(f"ghost_2d_laplacian L={L} N={N} np={size}  rel_err={err:.2e}",
                         err < 1e-10)
    return all_ok


# ── Test 4: Repeated spmv calls — ghost consistency ──────────────────────────

def test_repeated_spmv_consistency() -> bool:
    """
    Call dist_spmv multiple times with the same data.
    Results must be bit-identical (deterministic ghost exchange).
    """
    if not (HAS_CUPY and HAS_KSPMV and HAS_MPI):
        print("  [SKIP] test_repeated_spmv_consistency")
        return True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("  [SKIP] test_repeated_spmv_consistency (need >= 2 ranks)")
        return True

    N = 100
    A = _make_1d_laplacian(N)
    rng = np.random.default_rng(42)
    x_global = rng.standard_normal(N)

    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        N, N)

    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end
    x_local = cp.asarray(x_global[local_start:local_end])

    results = []
    for _ in range(5):
        y_local = cp.zeros(local_end - local_start, dtype=cp.float64)
        kspmv.dist_spmv(A_dist, x_local, y_local)
        results.append(cp.asnumpy(y_local))

    all_ok = True
    for i in range(1, 5):
        diff = float(np.max(np.abs(results[i] - results[0])))
        ok = (diff == 0.0)
        if rank == 0:
            all_ok &= _check(f"repeated_spmv run #{i} bit-identical",
                              ok, f"max_diff={diff:.2e}")

    return all_ok


# ── Test 5: Zero-vector SpMV ──────────────────────────────────────────────────

def test_zero_vector() -> bool:
    """Distributed SpMV on x=0 must return y=0 exactly."""
    if not (HAS_CUPY and HAS_KSPMV and HAS_MPI):
        print("  [SKIP] test_zero_vector")
        return True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("  [SKIP] test_zero_vector (need >= 2 ranks)")
        return True

    N = 64
    A = _make_1d_laplacian(N)

    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        N, N)

    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end
    x_local = cp.zeros(local_end - local_start, dtype=cp.float64)
    y_local = cp.zeros(local_end - local_start, dtype=cp.float64)

    kspmv.dist_spmv(A_dist, x_local, y_local)
    y_np = cp.asnumpy(y_local)

    max_abs = float(np.max(np.abs(y_np)))
    ok = (max_abs == 0.0)
    if rank == 0:
        return _check(f"zero_vector: y=0 everywhere (max|y|={max_abs:.2e})", ok)
    return ok


# ── Test 6: Standard basis vector — selects a matrix column ──────────────────

def test_basis_vector() -> bool:
    """
    x = e_j (unit vector in column j).
    dist_spmv result should equal column j of A.
    Test a few columns near partition boundaries.
    """
    if not (HAS_CUPY and HAS_KSPMV and HAS_MPI):
        print("  [SKIP] test_basis_vector")
        return True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("  [SKIP] test_basis_vector (need >= 2 ranks)")
        return True

    N = 60
    A = _make_1d_laplacian(N)
    A_dense = A.toarray()  # for reference

    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        N, N)

    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end

    # Test columns 0, N//size-1, N//size, N-1 (boundary columns)
    test_cols = sorted({0, N // size - 1, N // size, N - 1})
    all_ok = True

    for col in test_cols:
        x_global = np.zeros(N, dtype=np.float64)
        x_global[col] = 1.0
        y_ref = A_dense[:, col]

        x_local = cp.asarray(x_global[local_start:local_end])
        y_local = cp.zeros(local_end - local_start, dtype=cp.float64)

        kspmv.dist_spmv(A_dist, x_local, y_local)
        y_local_np = cp.asnumpy(y_local)

        counts = np.array(comm.allgather(y_local_np.size))
        displacements = np.concatenate([[0], np.cumsum(counts[:-1])])
        y_gathered = None
        if rank == 0:
            y_gathered = np.empty(N, dtype=np.float64)

        comm.Gatherv(y_local_np,
                     [y_gathered, counts, displacements, MPI.DOUBLE],
                     root=0)

        if rank == 0:
            err = _rel_err(y_gathered, y_ref) if np.any(y_ref != 0) else float(np.max(np.abs(y_gathered)))
            ok = err < 1e-10
            all_ok &= _check(f"basis_vector col={col:3d} np={size}  rel_err={err:.2e}", ok)

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _pass, _fail

    if not HAS_MPI:
        print("mpi4py not available — all distributed tests skipped.")
        sys.exit(0)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if HAS_KSPMV and HAS_CUPY:
        n_dev = max(1, cp.cuda.runtime.getDeviceCount())
        kspmv.init(device_id=rank % n_dev)

    if rank == 0:
        print("=" * 72)
        print("  KokkosSpMV Halo Exchange / Distributed SpMV Tests")
        print(f"  Running on {size} rank(s)")
        print("=" * 72)

    results = []

    if rank == 0:
        print("\n[1] Row partition correctness")
    results.append(test_row_partition())

    if rank == 0:
        print("\n[2] Ghost buffer — 1D tridiagonal (nearest-neighbour comm)")
    results.append(test_ghost_1d_tridiagonal())

    if rank == 0:
        print("\n[3] Ghost buffer — 2D Laplacian (multi-row ghost bands)")
    results.append(test_ghost_2d_laplacian())

    if rank == 0:
        print("\n[4] Repeated SpMV determinism (ghost consistency)")
    results.append(test_repeated_spmv_consistency())

    if rank == 0:
        print("\n[5] Zero-vector SpMV")
    results.append(test_zero_vector())

    if rank == 0:
        print("\n[6] Basis-vector SpMV (column selection)")
    results.append(test_basis_vector())

    if rank == 0:
        print()
        print("=" * 72)
        if all(results):
            print(f"  ALL {len(results)} test groups PASSED  "
                  f"({_pass} checks passed, {_fail} failed)")
        else:
            n_ok = sum(results)
            print(f"  {n_ok}/{len(results)} test groups PASSED  "
                  f"({_pass} checks passed, {_fail} failed)")
        print("=" * 72)
        sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
