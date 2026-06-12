"""
benchmark_spmv.py  (v0.2)
=========================
SpMV / Batch-SpMV / Distributed-SpMV throughput, all backends.

New vs v0.1:
  - Distributed runs measure BOTH local kernels →  kk_{np}gpu / custom_{np}gpu
    (fills the kk_2gpu / kk_4gpu gap from the slide deck)
  - --no-overlap to quantify comm/compute overlap gain (dist_noovl_{np}gpu)
  - --batch-sweep for k ∈ {8,32,128,...}
  - all 5 sparsity patterns supported, --matrix all
  - CSV rows tagged with matrix, n_ranks, k, overlap, cuda_aware

Usage:
    python3 benchmark_spmv.py --N 1000000 --matrix all --batch-sweep 8 32 128
    mpiexec -np 4 python3 benchmark_spmv.py --N 4000000 --backend kokkos_dist
"""

import argparse
import csv
import gc
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

ALL_MATRICES = ['laplacian_2d', 'laplacian_3d', 'lattice_gauge',
                'random_sparse', 'power_law']


class Timer:
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


def measure(fn, warmup=20, repeat=100):
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


def spmv_metrics(elapsed_s, N, nnz, k=1):
    mem_bytes = nnz * 12 + k * N * 8 + N * 8
    return {
        'elapsed_us': elapsed_s * 1e6,
        'elapsed_s':  elapsed_s,
        'bw_GB_s':    mem_bytes / elapsed_s / 1e9,
        'gflops':     2 * nnz * k / elapsed_s / 1e9,
    }


# ─── Single-GPU backends ──────────────────────────────────────────────────────

def run_scipy_cpu(A, x, repeat):
    t = measure(lambda: A @ x, warmup=3, repeat=min(repeat, 20))
    return spmv_metrics(t, A.shape[0], A.nnz)


def run_jax_1gpu(A, x, repeat):
    if not HAS_JAX:
        return {}
    try:
        gc.collect()
        A_bcoo = jsparse.BCOO.from_scipy_sparse(A)
        x_jax  = jnp.array(x)
        jax.block_until_ready(A_bcoo.data)
        jax.block_until_ready(x_jax)

        @jax.jit
        def fn():
            return A_bcoo @ x_jax

        # warmup을 충분히 — 첫 호출은 컴파일, 이후 캐시 히트 구분
        for _ in range(5):
            jax.block_until_ready(fn())

        t = measure(lambda: jax.block_until_ready(fn()),
                    warmup=10, repeat=repeat)
        del A_bcoo, x_jax
        gc.collect()
        return spmv_metrics(t, A.shape[0], A.nnz)
    except Exception as e:
        print(f"  [SKIP] jax_1gpu: {type(e).__name__}: {e}", file=sys.stderr)
        gc.collect()
        return {}

def run_kokkos_1gpu(A, x, repeat, kokkoskernels=True):
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


# ─── Distributed backends ─────────────────────────────────────────────────────

def run_kokkos_dist(A, rank, size, repeat=50,
                    kokkoskernels=True, overlap=True):
    if not (HAS_CUPY and HAS_KSPMV and HAS_MPI):
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

    def call():
        kspmv.dist_spmv(A_dist, x_local, y_local,
                        kokkoskernels=kokkoskernels, overlap=overlap)

    for _ in range(10):
        call()
    cp.cuda.Stream.null.synchronize()
    comm.Barrier()

    t0 = time.perf_counter()
    for _ in range(repeat):
        call()
    cp.cuda.Stream.null.synchronize()
    comm.Barrier()
    elapsed = (time.perf_counter() - t0) / repeat

    m = spmv_metrics(elapsed, A.shape[0], A.nnz)
    m['n_ranks'] = size
    m['overlap'] = int(overlap)
    return m


# ─── Batch backends ───────────────────────────────────────────────────────────

def run_batch(A, k, repeat):
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
                warmup=10, repeat=repeat)
    m = spmv_metrics(t, A.shape[0], A.nnz, k=k)
    m['k'] = k
    return m


def run_batch_jax(A, k, repeat):
    if not HAS_JAX:
        return {}
    try:
        gc.collect()
        A_bcoo = jsparse.BCOO.from_scipy_sparse(A)
        X_jax  = jnp.ones((A.shape[1], k))

        @jax.jit
        def fn():
            return jax.vmap(lambda x_col: A_bcoo @ x_col,
                            in_axes=1, out_axes=1)(X_jax)

        jax.block_until_ready(fn())
        t = measure(lambda: jax.block_until_ready(fn()),
                    warmup=5, repeat=repeat)
        del A_bcoo, X_jax
        gc.collect()
        m = spmv_metrics(t, A.shape[0], A.nnz, k=k)
        m['k'] = k
        return m
    except Exception as e:
        print(f"  [SKIP] jax_vmap_k{k}: {type(e).__name__}: {e}", file=sys.stderr)
        gc.collect()
        return {}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SpMV benchmark v0.2")
    p.add_argument('--N',      type=int, default=100_000)
    p.add_argument('--matrix', default='laplacian_3d',
                   choices=ALL_MATRICES + ['all'])
    p.add_argument('--batch',  type=int, default=0)
    p.add_argument('--batch-sweep', type=int, nargs='*', default=None,
                   help='Run batch SpMV for each k in this list')
    p.add_argument('--repeat', type=int, default=100)
    p.add_argument('--backend', nargs='+',
                   default=['scipy', 'jax', 'kokkos', 'kokkos_custom',
                            'kokkos_dist'])
    p.add_argument('--no-overlap', action='store_true',
                   help='Also benchmark dist SpMV with overlap disabled')
    p.add_argument('--no-csv', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    rank, size = 0, 1
    if HAS_MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()

    cuda_aware = False
    if HAS_KSPMV and HAS_CUPY:
        n_gpu = cp.cuda.runtime.getDeviceCount()
        kspmv.init(device_id=rank % n_gpu)
        cuda_aware = bool(getattr(kspmv, 'cuda_aware_mpi', lambda: False)())
        if rank == 0:
            print(f"  GPUs: {n_gpu} | ranks: {size} | "
                  f"CUDA-aware MPI: {'ON' if cuda_aware else 'OFF (host staging)'}")

    matrices = ALL_MATRICES if args.matrix == 'all' else [args.matrix]
    rows = []

    for mat_name in matrices:
        A = get_matrix(mat_name, args.N)
        N, nnz = A.shape[0], A.nnz
        x = np.random.default_rng(42).standard_normal(A.shape[1])

        if rank == 0:
            print(f"\nSpMV — {mat_name}  N={N:,}  nnz={nnz:,}  "
                  f"nnz/row={nnz/N:.1f}")
            print("-" * 78)
            print(f"{'Backend':<32} {'Time(μs)':>10} {'BW(GB/s)':>10} {'GFLOPS':>8}")
            print("-" * 78)

        def report(tag, m):
            if not m or rank != 0:
                return
            print(f"  {tag:<30} {m['elapsed_us']:>10.1f} "
                  f"{m['bw_GB_s']:>10.2f} {m['gflops']:>8.2f}")
            rows.append({'backend': tag, 'matrix': mat_name,
                         'N': N, 'nnz': nnz,
                         'cuda_aware': int(cuda_aware), **m})

        if size == 1:
            if 'scipy' in args.backend:
                report('scipy_cpu', run_scipy_cpu(A, x, args.repeat))
            if 'jax' in args.backend:
                report('jax_1gpu', run_jax_1gpu(A, x, args.repeat))
            if 'kokkos' in args.backend:
                report('kk_1gpu', run_kokkos_1gpu(A, x, args.repeat, True))
            if 'kokkos_custom' in args.backend:
                report('custom_1gpu', run_kokkos_1gpu(A, x, args.repeat, False))
        else:
            # Distributed: BOTH local kernels → kk_{np}gpu / custom_{np}gpu
            rep = min(args.repeat, 50)
            report(f'kk_{size}gpu',
                   run_kokkos_dist(A, rank, size, rep, True,  True))
            report(f'custom_{size}gpu',
                   run_kokkos_dist(A, rank, size, rep, False, True))
            if args.no_overlap:
                report(f'kk_noovl_{size}gpu',
                       run_kokkos_dist(A, rank, size, rep, True, False))

        # Batch sweep (single rank only)
        ks = args.batch_sweep if args.batch_sweep else \
             ([args.batch] if args.batch > 0 else [])
        if ks and size == 1:
            for k in ks:
                if rank == 0:
                    print(f"  --- Batch SpMV (k={k}) ---")
                report(f'jax_vmap',     run_batch_jax(A, k, max(args.repeat // 4, 10)))
                report(f'kokkos_batch', run_batch(A, k, max(args.repeat // 2, 20)))

    if not args.no_csv and rank == 0 and rows:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        csv_path = RESULTS_DIR / f"spmv_{ts}.csv"
        all_keys, seen = [], set()
        for r in rows:
            for k_ in r:
                if k_ not in seen:
                    all_keys.append(k_); seen.add(k_)
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Results saved to {csv_path}")


if __name__ == '__main__':
    main()
