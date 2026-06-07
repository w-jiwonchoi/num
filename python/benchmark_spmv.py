"""
benchmark_spmv.py
=================
Measures SpMV and Batch-SpMV throughput for all backends.

Usage:
    # Single GPU, all backends:
    python3 benchmark_spmv.py --N 1000000 --batch 128

    # Distributed (4 GPUs):
    mpiexec -np 4 python3 benchmark_spmv.py --N 4000000 --backend kokkos_dist

Outputs CSV to ../benchmark/results/spmv_<timestamp>.csv
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import jax
    import jax.experimental.sparse as jsparse
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

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

from matrix_generators import get_matrix

RESULTS_DIR = Path(__file__).parent.parent / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Timer ────────────────────────────────────────────────────────────────────

class Timer:
    """GPU-accurate timer using CUDA events when CuPy is available."""

    def __enter__(self):
        if HAS_CUPY:
            self._start = cp.cuda.Event()
            self._end   = cp.cuda.Event()
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if HAS_CUPY:
            self._end.record()
            self._end.synchronize()
            self.elapsed = cp.cuda.get_elapsed_time(self._start, self._end) * 1e-3
        else:
            self.elapsed = time.perf_counter() - self._t0


def measure(fn, warmup: int = 20, repeat: int = 100) -> float:
    """Returns median wall-time (seconds) for fn()."""
    for _ in range(warmup):
        fn()
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(repeat):
        with Timer() as t:
            fn()
        times.append(t.elapsed)
    return float(np.median(times))


# ─── Metrics helper ───────────────────────────────────────────────────────────

def spmv_metrics(elapsed_s: float, N: int, nnz: int, k: int = 1) -> dict:
    """Compute bandwidth and throughput from elapsed time."""
    # Memory traffic model:
    #   values: nnz * 8B, colind: nnz * 4B, rowptr: (N+1)*4B ≈ N*4B
    #   x: N * 8B (per RHS), y: N * 8B (per RHS)
    mem_bytes = nnz * 12 + k * N * 8 + N * 8
    return {
        'elapsed_us': elapsed_s * 1e6,
        'bw_GB_s':    mem_bytes / elapsed_s / 1e9,
        'gflops':     2 * nnz * k / elapsed_s / 1e9,
    }


# ─── Backend runners ──────────────────────────────────────────────────────────

def run_scipy_cpu(A: sp.csr_matrix, x: np.ndarray, repeat: int) -> dict:
    t = measure(lambda: A @ x, warmup=5, repeat=repeat)
    return spmv_metrics(t, A.shape[0], A.nnz)


def run_jax_1gpu(A: sp.csr_matrix, x: np.ndarray, repeat: int) -> dict:
    if not HAS_JAX:
        return {}
    try:
        A_bcoo = jsparse.BCOO.from_scipy_sparse(A)
        x_jax  = jnp.array(x)
        A_bcoo.block_until_ready()
        x_jax.block_until_ready()

        @jax.jit
        def fn():
            return A_bcoo @ x_jax

        fn().block_until_ready()  # warmup compile

        t = measure(lambda: fn().block_until_ready(), warmup=10, repeat=repeat)
        return spmv_metrics(t, A.shape[0], A.nnz)
    except Exception as e:
        print(f"  [SKIP] jax_1gpu: {type(e).__name__}: {e}", file=sys.stderr)
        return {}


def run_kokkos_1gpu(A: sp.csr_matrix, x: np.ndarray, repeat: int,
                    kokkoskernels: bool = True) -> dict:
    if not (HAS_CUPY and HAS_KSPMV):
        return {}
    A_gpu = kspmv.upload_csr(A.indptr.astype(np.int32),
                              A.indices.astype(np.int32),
                              A.data.astype(np.float64),
                              A.shape[0], A.shape[1])
    x_gpu = cp.asarray(x)
    y_gpu = cp.zeros(A.shape[0], dtype=cp.float64)

    t = measure(lambda: kspmv.spmv(A_gpu, x_gpu, y_gpu,
                                   kokkoskernels=kokkoskernels),
                warmup=20, repeat=repeat)
    return spmv_metrics(t, A.shape[0], A.nnz)


def run_batch(A: sp.csr_matrix, k: int, repeat: int) -> dict:
    """Batch SpMV: Y = A @ X where X is (N, k)."""
    if not (HAS_CUPY and HAS_KSPMV):
        return {}
    N = A.shape[1]
    A_gpu = kspmv.upload_csr(A.indptr.astype(np.int32),
                              A.indices.astype(np.int32),
                              A.data.astype(np.float64),
                              A.shape[0], A.shape[1])
    X_gpu = cp.random.standard_normal((N, k), dtype=cp.float64)
    Y_gpu = cp.zeros((A.shape[0], k), dtype=cp.float64)

    t = measure(lambda: kspmv.batch_spmv(A_gpu, X_gpu, Y_gpu),
                warmup=20, repeat=repeat)
    metrics = spmv_metrics(t, A.shape[0], A.nnz, k=k)
    metrics['k'] = k
    return metrics


def run_batch_jax_loop(A: sp.csr_matrix, k: int, repeat: int) -> dict:
    if not HAS_JAX:
        return {}
    try:
        A_bcoo = jsparse.BCOO.from_scipy_sparse(A)
        X_jax  = jnp.ones((A.shape[1], k))

        @jax.jit
        def fn():
            return jnp.stack([A_bcoo @ X_jax[:, i] for i in range(k)], axis=1)

        fn().block_until_ready()  # compile

        t = measure(lambda: fn().block_until_ready(), warmup=5, repeat=repeat)
        metrics = spmv_metrics(t, A.shape[0], A.nnz, k=k)
        metrics['k'] = k
        return metrics
    except Exception as e:
        print(f"  [SKIP] jax_loop_k{k}: {type(e).__name__}: {e}", file=sys.stderr)
        return {}



# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SpMV benchmark")
    p.add_argument('--N',        type=int, default=100_000)
    p.add_argument('--matrix',   default='laplacian_3d',
                   choices=['laplacian_2d', 'laplacian_3d',
                             'lattice_gauge', 'random_sparse', 'power_law'])
    p.add_argument('--batch',    type=int, default=0,
                   help='If > 0, also run batch SpMV with this many RHS vectors')
    p.add_argument('--repeat',   type=int, default=100)
    p.add_argument('--backend',  nargs='+',
                   default=['scipy', 'jax', 'kokkos', 'kokkos_custom'],
                   help='Backends to run')
    p.add_argument('--no-csv',   action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    rank = 0
    if HAS_MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    if HAS_KSPMV:
        kspmv.init(device_id=rank)

    A = get_matrix(args.matrix, args.N)
    N, nnz = A.shape[0], A.nnz
    x = np.random.default_rng(42).standard_normal(A.shape[1])

    if rank == 0:
        print(f"\nSpMV Benchmark — {args.matrix}  N={N:,}  nnz={nnz:,}  "
              f"nnz/row={nnz/N:.1f}")
        print("-" * 78)
        print(f"{'Backend':<30} {'Time(μs)':>10} {'BW(GB/s)':>10} {'GFLOPS':>8}")
        print("-" * 78)

    rows = []

    def report(tag, m):
        if not m or rank != 0:
            return
        print(f"  {tag:<28} {m['elapsed_us']:>10.1f} "
              f"{m['bw_GB_s']:>10.2f} {m['gflops']:>8.2f}")
        rows.append({'backend': tag, 'N': N, 'nnz': nnz, **m})

    if 'scipy' in args.backend:
        report('scipy_cpu',       run_scipy_cpu(A, x, args.repeat))

    if 'jax' in args.backend:
        report('jax_1gpu',        run_jax_1gpu(A, x, args.repeat))

    if 'kokkos' in args.backend:
        report('kokkos_kk_1gpu',  run_kokkos_1gpu(A, x, args.repeat, True))

    if 'kokkos_custom' in args.backend:
        report('kokkos_custom_1gpu', run_kokkos_1gpu(A, x, args.repeat, False))

    if args.batch > 0:
        k = args.batch
        if rank == 0:
            print(f"\n  --- Batch SpMV (k={k}) ---")
        report(f'jax_loop_k{k}',    run_batch_jax_loop(A, k, max(args.repeat//4, 10)))
        report(f'kokkos_batch_k{k}', run_batch(A, k, args.repeat))

    if not args.no_csv and rank == 0 and rows:
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = RESULTS_DIR / f"spmv_{ts}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results saved to {csv_path}")


if __name__ == '__main__':
    main()
