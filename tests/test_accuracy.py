"""
test_accuracy.py
================
Python-level accuracy tests for kokkos_spmv.

Tests:
  1. Single-GPU SpMV  vs. SciPy reference (rel_err < 1e-10)
  2. Batch SpMV       vs. SciPy reference (rel_err < 1e-10 per column)
  3. Local CG solver  vs. SciPy CG        (solution rel_err < 1e-6)
  4. Distributed SpMV (multi-rank, requires mpiexec -np >=2)

Run:
    python3 tests/test_accuracy.py               # single GPU
    mpiexec -np 4 python3 tests/test_accuracy.py # distributed
"""

import sys
import os

# Allow running from repo root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("[WARN] CuPy not found — GPU tests will be skipped", file=sys.stderr)

try:
    import kokkos_spmv as kspmv
    HAS_KSPMV = True
except ImportError:
    HAS_KSPMV = False
    print("[WARN] kokkos_spmv not found — only SciPy self-checks will run",
          file=sys.stderr)

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from matrix_generators import get_matrix

# ── Test bookkeeping ──────────────────────────────────────────────────────────
_pass = 0
_fail = 0

PASS_MARK = "\u2713"
FAIL_MARK = "\u2717"


def _check(name: str, err: float, tol: float = 1e-10) -> bool:
    global _pass, _fail
    ok = err < tol
    mark = PASS_MARK if ok else FAIL_MARK
    suffix = "" if ok else f"  [tol={tol:.0e}]"
    print(f"  {mark}  {name:<60s}  rel_err={err:.3e}{suffix}")
    if ok:
        _pass += 1
    else:
        _fail += 1
    return ok


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom)


def _scipy_spmv(A: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    return np.asarray(A @ x, dtype=np.float64)


# ── 1. Single-GPU SpMV ────────────────────────────────────────────────────────

def test_spmv_single_gpu(N: int = 50_000) -> bool:
    """Compare kokkos_spmv.spmv against SciPy for multiple matrix types."""
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_spmv_single_gpu — cupy or kokkos_spmv unavailable")
        return True

    all_ok = True
    rng = np.random.default_rng(0)

    for mat_name in ('laplacian_2d', 'laplacian_3d', 'random_sparse', 'lattice_gauge'):
        A = get_matrix(mat_name, N)
        x = rng.standard_normal(A.shape[1])
        y_ref = _scipy_spmv(A, x)

        # Upload to GPU
        A_gpu = kspmv.upload_csr(
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(np.float64),
            A.shape[0], A.shape[1])
        x_gpu = cp.asarray(x)
        y_gpu = cp.zeros(A.shape[0], dtype=cp.float64)

        # KokkosKernels backend
        kspmv.spmv(A_gpu, x_gpu, y_gpu, kokkoskernels=True)
        err_kk = _rel_err(cp.asnumpy(y_gpu), y_ref)
        all_ok &= _check(f"spmv(kk)   {mat_name:<20} N={N:>7,}", err_kk)

        # Custom TeamPolicy backend
        y_gpu[:] = 0.0
        kspmv.spmv(A_gpu, x_gpu, y_gpu, kokkoskernels=False)
        err_custom = _rel_err(cp.asnumpy(y_gpu), y_ref)
        all_ok &= _check(f"spmv(cust) {mat_name:<20} N={N:>7,}", err_custom)

    return all_ok


# ── 2. Batch SpMV ─────────────────────────────────────────────────────────────

def test_batch_spmv(N: int = 20_000, k: int = 64) -> bool:
    """Batch SpMV: Y = A @ X.  Compare column-by-column with SciPy."""
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_batch_spmv — cupy or kokkos_spmv unavailable")
        return True

    all_ok = True
    rng = np.random.default_rng(1)

    for mat_name in ('laplacian_2d', 'laplacian_3d', 'random_sparse'):
        A = get_matrix(mat_name, N)
        # X shape: (A.shape[1], k) — k right-hand-side vectors
        X = rng.standard_normal((A.shape[1], k))
        Y_ref = np.asarray(A @ X, dtype=np.float64)   # shape (N, k)

        A_gpu = kspmv.upload_csr(
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(np.float64),
            A.shape[0], A.shape[1])

        # batch_spmv expects X of shape (ncols, k) C-contiguous
        X_gpu = cp.ascontiguousarray(cp.asarray(X))   # (ncols, k)
        Y_gpu = cp.zeros((A.shape[0], k), dtype=cp.float64)

        kspmv.batch_spmv(A_gpu, X_gpu, Y_gpu)
        Y_got = cp.asnumpy(Y_gpu)

        err = _rel_err(Y_got, Y_ref)
        all_ok &= _check(f"batch_spmv k={k:3d} {mat_name:<20} N={N:>7,}", err)

    return all_ok


# ── 3. Local CG solver ────────────────────────────────────────────────────────

def test_cg_local(N: int = 10_000) -> bool:
    """CG solver: compare solution with SciPy CG on same matrix."""
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_cg_local — cupy or kokkos_spmv unavailable")
        return True

    all_ok = True

    for mat_name in ('laplacian_3d', 'random_sparse'):
        A = get_matrix(mat_name, N)
        b = np.ones(A.shape[0], dtype=np.float64)

        # SciPy reference
        x_ref, info = spla.cg(A, b, tol=1e-12, maxiter=5000)
        if info != 0:
            print(f"    [WARN] SciPy CG did not converge for {mat_name} (info={info})",
                  file=sys.stderr)

        A_gpu = kspmv.upload_csr(
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(np.float64),
            A.shape[0], A.shape[1])
        b_gpu = cp.asarray(b)
        x_gpu = cp.zeros(A.shape[0], dtype=cp.float64)

        result = kspmv.cg_solve_local(A_gpu, b_gpu, x_gpu,
                                       tol=1e-10, max_iter=5000)
        x_got = cp.asnumpy(x_gpu)

        sol_err = _rel_err(x_got, x_ref)
        label = (f"cg_local {mat_name:<20} N={N:>7,} "
                 f"iters={result['iters']:4d} conv={result['converged']}")
        all_ok &= _check(label, sol_err, tol=1e-6)

    return all_ok


# ── 4. Distributed SpMV ───────────────────────────────────────────────────────

def test_distributed_spmv(N: int = 40_000) -> bool:
    """Distributed SpMV: gather results on rank 0 and compare with SciPy."""
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_distributed_spmv — cupy or kokkos_spmv unavailable")
        return True
    if not HAS_MPI:
        print("  [SKIP] test_distributed_spmv — mpi4py not found")
        return True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("  [SKIP] test_distributed_spmv — run with mpiexec -np >= 2")
        return True

    n_dev = cp.cuda.runtime.getDeviceCount()
    kspmv.init(device_id=rank % n_dev)

    all_ok = True

    for mat_name in ('laplacian_2d', 'laplacian_3d'):
        A = get_matrix(mat_name, N)
        rng = np.random.default_rng(2)
        x_global = rng.standard_normal(A.shape[1])
        y_ref_global = _scipy_spmv(A, x_global)

        A_dist = kspmv.distribute_csr(
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(np.float64),
            A.shape[0], A.shape[1])

        local_start = A_dist.local_row_start
        local_end   = A_dist.local_row_end
        x_local = cp.asarray(x_global[local_start:local_end])
        y_local = cp.zeros(local_end - local_start, dtype=cp.float64)

        kspmv.dist_spmv(A_dist, x_local, y_local)
        y_local_np = cp.asnumpy(y_local)

        # Gather to rank 0 for comparison
        counts = np.array(comm.allgather(y_local_np.size))
        displacements = np.concatenate([[0], np.cumsum(counts[:-1])])
        y_gathered = None
        if rank == 0:
            y_gathered = np.empty(A.shape[0], dtype=np.float64)

        comm.Gatherv(
            y_local_np,
            [y_gathered, counts, displacements, MPI.DOUBLE],
            root=0)

        if rank == 0:
            err = _rel_err(y_gathered, y_ref_global)
            label = f"dist_spmv {mat_name:<20} N={N:>7,} np={size}"
            ok = _check(label, err)
            all_ok = all_ok and ok

    return all_ok


# ── 5. Distributed CG solver ─────────────────────────────────────────────────

def test_distributed_cg(N: int = 20_000) -> bool:
    """Distributed CG: compare solution with SciPy CG on rank 0."""
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_distributed_cg — cupy or kokkos_spmv unavailable")
        return True
    if not HAS_MPI:
        print("  [SKIP] test_distributed_cg — mpi4py not found")
        return True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("  [SKIP] test_distributed_cg — run with mpiexec -np >= 2")
        return True

    n_dev = cp.cuda.runtime.getDeviceCount()
    kspmv.init(device_id=rank % n_dev)

    all_ok = True

    A = get_matrix('laplacian_3d', N)
    b = np.ones(A.shape[0], dtype=np.float64)

    x_ref, info = spla.cg(A, b, tol=1e-12, maxiter=5000)
    if rank == 0 and info != 0:
        print(f"    [WARN] SciPy CG did not converge (info={info})", file=sys.stderr)

    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        A.shape[0], A.shape[1])

    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end
    b_local = cp.asarray(b[local_start:local_end])
    x_local = cp.zeros(local_end - local_start, dtype=cp.float64)

    result = kspmv.cg_solve(A_dist, b_local, x_local, tol=1e-10, max_iter=5000)
    x_local_np = cp.asnumpy(x_local)

    counts = np.array(comm.allgather(x_local_np.size))
    displacements = np.concatenate([[0], np.cumsum(counts[:-1])])
    x_gathered = None
    if rank == 0:
        x_gathered = np.empty(A.shape[0], dtype=np.float64)

    comm.Gatherv(
        x_local_np,
        [x_gathered, counts, displacements, MPI.DOUBLE],
        root=0)

    if rank == 0:
        err = _rel_err(x_gathered, x_ref)
        label = (f"dist_cg laplacian_3d N={N:>7,} "
                 f"iters={result['iters']:4d} conv={result['converged']}")
        ok = _check(label, err, tol=1e-6)
        all_ok = all_ok and ok

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _pass, _fail

    rank, size = 0, 1
    if HAS_MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()

    if HAS_KSPMV:
        n_dev = 1
        if HAS_CUPY:
            n_dev = max(1, cp.cuda.runtime.getDeviceCount())
        kspmv.init(device_id=rank % n_dev)

    results = []

    if rank == 0:
        print("=" * 72)
        print("  KokkosSpMV Python Accuracy Tests")
        print("=" * 72)

        print("\n[1] Single-GPU SpMV")
    results.append(test_spmv_single_gpu(N=50_000))

    if rank == 0:
        print("\n[2] Batch SpMV (k=64)")
    results.append(test_batch_spmv(N=20_000, k=64))

    if rank == 0:
        print("\n[3] Local CG solver")
    results.append(test_cg_local(N=10_000))

    if size >= 2:
        if rank == 0:
            print("\n[4] Distributed SpMV")
        results.append(test_distributed_spmv(N=40_000))

        if rank == 0:
            print("\n[5] Distributed CG solver")
        results.append(test_distributed_cg(N=20_000))

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
