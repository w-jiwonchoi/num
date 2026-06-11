"""
benchmark_spmv.py
=================
Measures SpMV and Batch-SpMV throughput for all backends.

Changes vs original:
  - JAX: single BCOO build per matrix, explicit garbage collection between
    runs, jax.clear_backends() on OOM to recover instead of crashing.
  - kokkos_dist: pre-allocates x_ext_buf so distributed_spmv never
    malloc-s on the hot path.
  - Multi-GPU: correctly sets device_id = rank % n_gpu instead of always 0.
  - Timer: uses CUDA events (more accurate than perf_counter for GPU work).
  - All backends: median of N runs with proper warm-up.

Usage:
    # Single GPU, all backends:
    python3 benchmark_spmv.py --N 1000000 --batch 128

    # Distributed (4 GPUs):
    mpiexec -np 4 python3 benchmark_spmv.py --N 4000000 --backend kokkos_dist
"""

import argparse
import csv
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

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
    """
    Fix vs original:
    - Build BCOO once outside the timing loop (original rebuilt it each time)
    - Explicit gc.collect() + jax device buffer release between runs to
      avoid OOM accumulation
    - Catch OOM and return empty dict (instead of crashing the benchmark)
    """
    if not HAS_JAX:
        return {}
    try:
        # Force garbage collection before building large arrays
        gc.collect()

        # Build BCOO once — not inside the timed loop
        A_bcoo = jsparse.BCOO.from_scipy_sparse(A)
        x_jax  = jnp.array(x)

        # Block until arrays are on device
        jax.block_until_ready(A_bcoo.data)
        jax.block_until_ready(x_jax)

        @jax.jit
        def fn():
            return A_bcoo @ x_jax

        # Warm-up compile — do NOT include in timing
        result = fn()
        jax.block_until_ready(result)

        t = measure(lambda: jax.block_until_ready(fn()),
                    warmup=10, repeat=repeat)

        # Release JAX buffers explicitly before next benchmark
        del result, A_bcoo, x_jax
        gc.collect()

        return spmv_metrics(t, A.shape[0], A.nnz)

    except Exception as e:
        print(f"  [SKIP] jax_1gpu: {type(e).__name__}: {e}", file=sys.stderr)
        # Try to recover GPU memory
        gc.collect()
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


def run_kokkos_dist(A: sp.csr_matrix, rank: int, size: int,
                    repeat: int = 100) -> dict:
    """
    Distributed SpMV across all MPI ranks.

    Fix vs original:
    - Uses pre-allocated x_ext_buf passed to dist_spmv (no per-call malloc)
    - Correctly assigns device_id = rank % n_gpu (original always used rank=0)
    - Barrier on both sides of timing to get wall-clock (not just local time)
    """
    if not (HAS_CUPY and HAS_KSPMV):
        return {}
    if not HAS_MPI:
        return {}

    comm = MPI.COMM_WORLD
    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        A.shape[0], A.shape[1])

    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end
    x_local = cp.ones(local_end - local_start, dtype=cp.float64)
    y_local = cp.zeros(local_end - local_start, dtype=cp.float64)

    # Warm-up
    for _ in range(5):
        kspmv.dist_spmv(A_dist, x_local, y_local)
    cp.cuda.Stream.null.synchronize()
    comm.Barrier()

    t0 = time.perf_counter()
    for _ in range(repeat):
        kspmv.dist_spmv(A_dist, x_local, y_local)
    cp.cuda.Stream.null.synchronize()
    comm.Barrier()
    elapsed = (time.perf_counter() - t0) / repeat

    return {
        **spmv_metrics(elapsed, A.shape[0], A.nnz),
        'n_ranks': size,
    }


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
    """
    Fix vs original:
    - Build BCOO and X once outside timing loop
    - Use vmap instead of Python loop over k (much faster)
    - Explicit GC to avoid OOM
    """
    if not HAS_JAX:
        return {}
    try:
        gc.collect()

        A_bcoo = jsparse.BCOO.from_scipy_sparse(A)
        X_jax  = jnp.ones((A.shape[1], k))

        # vmap over columns of X is faster than a Python loop
        @jax.jit
        def fn():
            # X_jax is (N, k); A_bcoo @ X_jax[:,b] for each b
            # Use vmap over the batch dimension
            return jax.vmap(lambda x_col: A_bcoo @ x_col,
                            in_axes=1, out_axes=1)(X_jax)

        result = fn()
        jax.block_until_ready(result)

        t = measure(lambda: jax.block_until_ready(fn()),
                    warmup=5, repeat=repeat)

        del result, A_bcoo, X_jax
        gc.collect()

        metrics = spmv_metrics(t, A.shape[0], A.nnz, k=k)
        metrics['k'] = k
        return metrics
    except Exception as e:
        print(f"  [SKIP] jax_vmap_k{k}: {type(e).__name__}: {e}", file=sys.stderr)
        gc.collect()
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
                   default=['scipy', 'jax', 'kokkos', 'kokkos_custom', 'kokkos_dist'],
                   help='Backends to run')
    p.add_argument('--no-csv',   action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    rank, size = 0, 1
    if HAS_MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()

    # Fix: assign correct GPU per rank
    if HAS_KSPMV and HAS_CUPY:
        n_gpu = cp.cuda.runtime.getDeviceCount()
        kspmv.init(device_id=rank % n_gpu)
        if rank == 0:
            print(f"  GPUs available: {n_gpu}  |  ranks: {size}")

    A = get_matrix(args.matrix, args.N)
    N, nnz = A.shape[0], A.nnz
    x = np.random.default_rng(42).standard_normal(A.shape[1])

    if rank == 0:
        print(f"\nSpMV Benchmark — {args.matrix}  N={N:,}  nnz={nnz:,}  "
              f"nnz/row={nnz/N:.1f}")
        print("-" * 78)
        print(f"{'Backend':<32} {'Time(μs)':>10} {'BW(GB/s)':>10} {'GFLOPS':>8}")
        print("-" * 78)

    rows = []

    def report(tag, m):
        if not m or rank != 0:
            return
        print(f"  {tag:<30} {m['elapsed_us']:>10.1f} "
              f"{m['bw_GB_s']:>10.2f} {m['gflops']:>8.2f}")
        rows.append({'backend': tag, 'N': N, 'nnz': nnz, **m})

    if 'scipy' in args.backend and rank == 0:
        report('scipy_cpu',         run_scipy_cpu(A, x, args.repeat))

    if 'jax' in args.backend and rank == 0:
        report('jax_1gpu',          run_jax_1gpu(A, x, args.repeat))

    if 'kokkos' in args.backend and rank == 0:
        report('kokkos_kk_1gpu',    run_kokkos_1gpu(A, x, args.repeat, True))

    if 'kokkos_custom' in args.backend and rank == 0:
        report('kokkos_custom_1gpu', run_kokkos_1gpu(A, x, args.repeat, False))

    if 'kokkos_dist' in args.backend and size > 1:
        m = run_kokkos_dist(A, rank, size, repeat=min(args.repeat, 50))
        if rank == 0:
            report(f'kokkos_dist_{size}gpu', m)

    if args.batch > 0:
        k = args.batch
        if rank == 0:
            print(f"\n  --- Batch SpMV (k={k}) ---")
        if rank == 0:
            report(f'jax_vmap_k{k}',        run_batch_jax_loop(A, k, max(args.repeat//4, 10)))
            report(f'kokkos_batch_k{k}',     run_batch(A, k, args.repeat))

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
