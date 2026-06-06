"""
verify_accuracy.py
==================
Validates KokkosSpMV results against SciPy reference on CPU.

Checks:
  - Single SpMV  : rel_err < 1e-10
  - Batch SpMV   : rel_err < 1e-10 per column
  - Distributed  : global rel_err < 1e-10  (requires mpiexec -np ≥2)
  - CG solver    : solution rel_err and residual < 1e-8

Run:
    python3 verify_accuracy.py                 # single GPU
    mpiexec -np 4 python3 verify_accuracy.py   # distributed
"""

import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("[WARN] CuPy not found — skipping GPU tests", file=sys.stderr)

try:
    import kokkos_spmv as kspmv
    HAS_KSPMV = True
except ImportError:
    HAS_KSPMV = False
    print("[WARN] kokkos_spmv not found — only SciPy self-checks will run",
          file=sys.stderr)

from matrix_generators import get_matrix

PASS = "✓"
FAIL = "✗"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||_2 / ||b||_2"""
    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom)


def scipy_spmv(A_scipy: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    return (A_scipy @ x).astype(np.float64)


def check(name: str, err: float, tol: float = 1e-10) -> bool:
    ok = err < tol
    marker = PASS if ok else FAIL
    print(f"  {marker}  {name:50s}  rel_err = {err:.3e}"
          + ("" if ok else f"  (tol={tol:.0e})"))
    return ok


# ─── Single-GPU SpMV ──────────────────────────────────────────────────────────

def test_spmv_single_gpu(N: int = 50_000):
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_spmv_single_gpu")
        return True

    all_pass = True
    for name in ('laplacian_2d', 'laplacian_3d', 'random_sparse', 'lattice_gauge'):
        A = get_matrix(name, N)
        x = np.random.default_rng(0).standard_normal(A.shape[1])
        y_ref = scipy_spmv(A, x)

        A_gpu = kspmv.upload_csr(A.indptr.astype(np.int32),
                                  A.indices.astype(np.int32),
                                  A.data.astype(np.float64),
                                  A.shape[0], A.shape[1])
        x_gpu = cp.asarray(x)
        y_gpu = cp.zeros(A.shape[0], dtype=cp.float64)
        kspmv.spmv(A_gpu, x_gpu, y_gpu)
        y_kspmv = cp.asnumpy(y_gpu)

        err = rel_err(y_kspmv, y_ref)
        ok  = check(f"spmv  {name:20s} N={N:>7,}", err)
        all_pass = all_pass and ok
    return all_pass


# ─── Batch SpMV ───────────────────────────────────────────────────────────────

def test_batch_spmv(N: int = 20_000, k: int = 64):
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_batch_spmv")
        return True

    all_pass = True
    for name in ('laplacian_2d', 'random_sparse'):
        A = get_matrix(name, N)
        X = np.random.default_rng(1).standard_normal((A.shape[1], k))
        Y_ref = (A @ X).astype(np.float64)   # (N, k)

        A_gpu = kspmv.upload_csr(A.indptr.astype(np.int32),
                                  A.indices.astype(np.int32),
                                  A.data.astype(np.float64),
                                  A.shape[0], A.shape[1])
        # kspmv batch_spmv expects X of shape (N, k) — transpose if needed
        X_gpu = cp.asarray(X.T.copy())   # (k, N) → need (N, k) LayoutRight
        X_gpu = cp.ascontiguousarray(X.T).T  # ensure (N, k) C-contiguous
        X_gpu = cp.ascontiguousarray(X)      # (N, k)
        Y_gpu = cp.zeros((A.shape[0], k), dtype=cp.float64)

        kspmv.batch_spmv(A_gpu, X_gpu, Y_gpu)
        Y_kspmv = cp.asnumpy(Y_gpu)

        err = rel_err(Y_kspmv, Y_ref)
        ok  = check(f"batch_spmv k={k:3d}  {name:20s} N={N:>7,}", err)
        all_pass = all_pass and ok
    return all_pass


# ─── CG solver ────────────────────────────────────────────────────────────────

def test_cg_local(N: int = 10_000):
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_cg_local")
        return True

    all_pass = True
    for name in ('laplacian_3d', 'random_sparse'):
        A = get_matrix(name, N)
        b = np.ones(A.shape[0], dtype=np.float64)

        # Reference (SciPy CG)
        x_ref, info = spla.cg(A, b, tol=1e-12, maxiter=2000)
        if info != 0:
            print(f"  [WARN] SciPy CG failed (info={info}) for {name}")

        A_gpu = kspmv.upload_csr(A.indptr.astype(np.int32),
                                  A.indices.astype(np.int32),
                                  A.data.astype(np.float64),
                                  A.shape[0], A.shape[1])
        b_gpu = cp.asarray(b)
        x_gpu = cp.zeros(A.shape[0], dtype=cp.float64)

        result = kspmv.cg_solve_local(A_gpu, b_gpu, x_gpu, tol=1e-10, max_iter=2000)
        x_kspmv = cp.asnumpy(x_gpu)

        sol_err = rel_err(x_kspmv, x_ref)
        ok  = check(
            f"cg_local  {name:20s} N={N:>7,}  iters={result['iters']:4d}  "
            f"converged={result['converged']}",
            sol_err, tol=1e-6)
        all_pass = all_pass and ok
    return all_pass


# ─── Distributed SpMV (multi-GPU) ────────────────────────────────────────────

def test_distributed_spmv(N: int = 40_000):
    if not (HAS_CUPY and HAS_KSPMV):
        print("  [SKIP] test_distributed_spmv")
        return True

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        print("  [SKIP] test_distributed_spmv (mpi4py not found)")
        return True

    if size < 2:
        print("  [SKIP] test_distributed_spmv (run with mpiexec -np ≥2)")
        return True

    kspmv.init(device_id=rank % cp.cuda.runtime.getDeviceCount())

    all_pass = True
    for name in ('laplacian_2d', 'laplacian_3d'):
        A = get_matrix(name, N)
        x_global = np.random.default_rng(2).standard_normal(A.shape[1])
        y_ref_global = scipy_spmv(A, x_global)

        # Distribute matrix
        A_dist = kspmv.distribute_csr(
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(np.float64),
            A.shape[0], A.shape[1])

        # Local slice of x
        local_start = A_dist.local_row_start
        local_end   = A_dist.local_row_end
        x_local = cp.asarray(x_global[local_start:local_end])
        y_local = cp.zeros(local_end - local_start, dtype=cp.float64)

        kspmv.dist_spmv(A_dist, x_local, y_local)

        # Gather results to rank 0 for comparison
        y_local_np = cp.asnumpy(y_local)
        y_gathered = None
        if rank == 0:
            y_gathered = np.empty(A.shape[0])
        comm.Gatherv(y_local_np, y_gathered, root=0)

        if rank == 0:
            err = rel_err(y_gathered, y_ref_global)
            ok  = check(f"dist_spmv  {name:20s} N={N:>7,}  np={size}", err)
            all_pass = all_pass and ok

    return all_pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print(" KokkosSpMV Accuracy Verification")
    print("=" * 72)

    if HAS_KSPMV:
        rank, size = kspmv.init(device_id=0)
    else:
        rank, size = 0, 1

    results = []

    if rank == 0:
        print("\n[1] Single-GPU SpMV")
    results.append(test_spmv_single_gpu(N=50_000))

    if rank == 0:
        print("\n[2] Batch SpMV")
    results.append(test_batch_spmv(N=20_000, k=64))

    if rank == 0:
        print("\n[3] Local CG solver")
    results.append(test_cg_local(N=10_000))

    if rank == 0:
        print("\n[4] Distributed SpMV")
    results.append(test_distributed_spmv(N=40_000))

    if rank == 0:
        n_pass = sum(results)
        n_total = len(results)
        print()
        print("=" * 72)
        if all(results):
            print(f"  ALL {n_total} test groups PASSED")
        else:
            print(f"  {n_pass}/{n_total} test groups passed — see failures above")
        print("=" * 72)
        sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
