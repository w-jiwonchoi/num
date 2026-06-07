"""
benchmark_cg.py
===============
Benchmarks the CG solver and measures scaling efficiency.

Produces data for:
  Fig 4 — CG scaling vs SpMV scaling
  Fig 5 — allreduce fraction of CG iteration time

Usage:
    python3 benchmark_cg.py --matrix laplacian_3d --N 1000000
    mpiexec -np 4 python3 benchmark_cg.py --matrix laplacian_3d --N 4000000
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


# ─── Timing helpers ──────────────────────────────────────────────────────────

def cuda_sync():
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()


def wall_time(fn, repeat: int = 5) -> float:
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    cuda_sync()
    return (time.perf_counter() - t0) / repeat


# ─── SciPy CG baseline ───────────────────────────────────────────────────────

def bench_scipy_cg(A: sp.csr_matrix, b: np.ndarray,
                   tol: float, max_iter: int) -> dict:
    iters_box = [0]
    calls = [0]

    def callback(xk):
        iters_box[0] += 1

    t0 = time.perf_counter()
    x, info = spla.cg(A, b, rtol=tol, maxiter=max_iter, callback=callback)
    elapsed = time.perf_counter() - t0
    res = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    return {
        'backend':   'scipy_cg',
        'elapsed_s': elapsed,
        'iters':     iters_box[0],
        'residual':  float(res),
        'converged': info == 0,
    }


# ─── KokkosSpMV local CG ─────────────────────────────────────────────────────

def bench_kspmv_cg_local(A: sp.csr_matrix, b: np.ndarray,
                          tol: float, max_iter: int) -> dict:
    if not (HAS_CUPY and HAS_KSPMV):
        return {}

    A_gpu = kspmv.upload_csr(A.indptr.astype(np.int32),
                              A.indices.astype(np.int32),
                              A.data.astype(np.float64),
                              A.shape[0], A.shape[1])
    b_gpu = cp.asarray(b)

    def run():
        x_gpu = cp.zeros(A.shape[0], dtype=cp.float64)
        return kspmv.cg_solve_local(A_gpu, b_gpu, x_gpu, tol=tol, max_iter=max_iter)

    # Warmup
    result = run()
    cuda_sync()

    t0 = time.perf_counter()
    result = run()
    cuda_sync()
    elapsed = time.perf_counter() - t0

    return {
        'backend':   'kspmv_cg_1gpu',
        'elapsed_s': elapsed,
        'iters':     result['iters'],
        'residual':  result['residual'],
        'converged': result['converged'],
    }


# ─── KokkosSpMV distributed CG ───────────────────────────────────────────────

def bench_kspmv_cg_dist(A: sp.csr_matrix, b_global: np.ndarray,
                         rank: int, size: int,
                         tol: float, max_iter: int) -> dict:
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
    b_local     = cp.asarray(b_global[local_start:local_end])

    def run():
        x_local = cp.zeros(local_end - local_start, dtype=cp.float64)
        return kspmv.cg_solve(A_dist, b_local, x_local, tol=tol, max_iter=max_iter)

    result = run()   # warmup
    cuda_sync()
    comm.Barrier()

    t0 = time.perf_counter()
    result = run()
    cuda_sync()
    comm.Barrier()
    elapsed = time.perf_counter() - t0

    return {
        'backend':   f'kspmv_cg_{size}gpu',
        'elapsed_s': elapsed,
        'iters':     result['iters'],
        'residual':  result['residual'],
        'converged': result['converged'],
        'n_ranks':   size,
    }


# ─── SpMV-only timing (to compare scaling to CG) ─────────────────────────────

def bench_spmv_dist(A: sp.csr_matrix, rank: int, size: int,
                    repeat: int = 20) -> dict:
    if not (HAS_CUPY and HAS_KSPMV):
        return {}

    comm = MPI.COMM_WORLD if HAS_MPI else None
    A_dist = kspmv.distribute_csr(
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
        A.shape[0], A.shape[1])

    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end
    x_local = cp.ones(local_end - local_start, dtype=cp.float64)
    y_local = cp.zeros(local_end - local_start, dtype=cp.float64)

    for _ in range(5):
        kspmv.dist_spmv(A_dist, x_local, y_local)
    cuda_sync()
    if comm: comm.Barrier()

    t0 = time.perf_counter()
    for _ in range(repeat):
        kspmv.dist_spmv(A_dist, x_local, y_local)
    cuda_sync()
    if comm: comm.Barrier()
    elapsed = (time.perf_counter() - t0) / repeat

    return {
        'backend':   f'kspmv_spmv_{size}gpu',
        'elapsed_s': elapsed,
        'n_ranks':   size,
        'N':         A.shape[0],
        'nnz':       A.nnz,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CG benchmark")
    p.add_argument('--N',       type=int, default=100_000)
    p.add_argument('--matrix',  default='laplacian_3d',
                   choices=['laplacian_2d', 'laplacian_3d',
                             'lattice_gauge', 'random_sparse'])
    p.add_argument('--tol',     type=float, default=1e-8)
    p.add_argument('--maxiter', type=int,   default=2000)
    p.add_argument('--no-csv',  action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    rank, size = 0, 1
    if HAS_MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
    if HAS_KSPMV:
        dev = rank % (cp.cuda.runtime.getDeviceCount() if HAS_CUPY else 1)
        kspmv.init(device_id=dev)

    A = get_matrix(args.matrix, args.N)
    N, nnz = A.shape[0], A.nnz
    b = np.ones(N, dtype=np.float64)

    if rank == 0:
        print(f"\nCG Benchmark — {args.matrix}  N={N:,}  nnz={nnz:,}"
              f"  tol={args.tol:.0e}  np={size}")
        print("-" * 78)
        fmt = f"  {'Backend':<30} {'Time(s)':>9} {'Iters':>7} {'Residual':>12} Converged"
        print(fmt)
        print("-" * 78)

    rows = []

    def report(m):
        if not m or rank != 0:
            return
        conv = "YES" if m.get('converged') else "NO "
        print(f"  {m['backend']:<30} {m['elapsed_s']:>9.3f} "
              f"{m.get('iters', '-'):>7} {m.get('residual', float('nan')):>12.3e} "
              f"{conv}")
        rows.append({'N': N, 'nnz': nnz, **m})

    if rank == 0:
        report(bench_scipy_cg(A, b, args.tol, args.maxiter))
        report(bench_kspmv_cg_local(A, b, args.tol, args.maxiter))

    if size > 1:
        report(bench_kspmv_cg_dist(A, b, rank, size, args.tol, args.maxiter))

    # SpMV alone (for scaling comparison)
    spmv_m = bench_spmv_dist(A, rank, size)
    if rank == 0 and spmv_m:
        spmv_m['backend'] = f'spmv_only_{size}gpu'
        report(spmv_m)

    if not args.no_csv and rank == 0 and rows:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = RESULTS_DIR / f"cg_{ts}.csv"
        all_keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results saved to {csv_path}")


if __name__ == '__main__':
    main()
