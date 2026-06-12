"""
benchmark_cg.py  (v0.3)
=======================
Fix vs v0.2:
  - petsc_cg: 4-GPU 시 전체 행렬 대신 로컬 부분만 전달하도록 수정.
    원인: np>1 일 때 rank 0 이 global A 를 그대로 petsc 에 넘겨서
    "row pointer of length N+1 given but expected local_N+1" 오류 발생.
    해결: petsc_cg 는 np==1 일 때만 실행 (분산 PETSc 설정은 별도이므로
    이 단순 래퍼에서는 single-rank 한정으로 명시).
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

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc as _PETSc
    HAS_PETSC = True
except (ImportError, Exception):
    HAS_PETSC = False

from matrix_generators import get_matrix

RESULTS_DIR = Path(__file__).parent.parent / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def cuda_sync():
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()


# ─── SciPy CG ────────────────────────────────────────────────────────────────

def bench_scipy_cg(A, b, tol, max_iter):
    iters_box = [0]

    def callback(xk):
        iters_box[0] += 1

    t0 = time.perf_counter()
    x, info = spla.cg(A, b, rtol=tol, maxiter=max_iter, callback=callback)
    elapsed = time.perf_counter() - t0
    res = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    return {'backend': 'scipy_cg', 'elapsed_s': elapsed,
            'iters': iters_box[0], 'residual': float(res), 'converged': info == 0}


# ─── PETSc CG (single-rank only) ─────────────────────────────────────────────

def bench_petsc_cg(A, b, tol, max_iter, mpi_size=1):
    """
    PETSc sequential CG.  Multi-rank 환경에서는 호출하지 않는다.
    (분산 PETSc 는 별도의 KSP 설정이 필요하고 현재 스크립트 범위 밖)
    """
    if not HAS_PETSC:
        return {}
    if mpi_size > 1:
        # 다중 rank 에서 sequential petsc 에 global 행렬을 넘기면
        # row pointer 크기 불일치 오류 발생 → 건너뜀
        print("  [SKIP] petsc_cg: not supported with np > 1 in this benchmark",
              file=sys.stderr)
        return {}

    PETSc = _PETSc
    n = A.shape[0]
    try:
        A_p = PETSc.Mat().createAIJWithArrays(
            size=A.shape,
            csr=(A.indptr.astype('int32'),
                 A.indices.astype('int32'),
                 A.data.astype('float64')))
        A_p.assemble()

        b_p = PETSc.Vec().createSeq(n)
        b_p.setArray(b); b_p.assemble()

        x_p = PETSc.Vec().createSeq(n)
        x_p.set(0.0)

        ksp = PETSc.KSP().create()
        ksp.setOperators(A_p)
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.setTolerances(rtol=tol, max_it=max_iter)
        ksp.getPC().setType(PETSc.PC.Type.NONE)
        ksp.setFromOptions()

        t0 = time.perf_counter()
        ksp.solve(b_p, x_p)
        elapsed = time.perf_counter() - t0

        x_np = x_p.getArray().copy()
        res  = float(np.linalg.norm(b - A @ x_np) / np.linalg.norm(b))
        return {'backend': 'petsc_cg', 'elapsed_s': elapsed,
                'iters': ksp.getIterationNumber(), 'residual': res,
                'converged': ksp.getConvergedReason() > 0}
    except Exception as e:
        print(f"  [WARN] petsc_cg failed: {e}", file=sys.stderr)
        return {}


# ─── KokkosSpMV local CG ─────────────────────────────────────────────────────

def bench_kspmv_cg_local(A, b, tol, max_iter):
    if not (HAS_CUPY and HAS_KSPMV):
        return {}
    A_gpu = kspmv.upload_csr(A.indptr.astype(np.int32),
                              A.indices.astype(np.int32),
                              A.data.astype(np.float64),
                              A.shape[0], A.shape[1])
    b_gpu = cp.asarray(b)

    def run():
        x_gpu = cp.zeros(A.shape[0], dtype=cp.float64)
        return kspmv.cg_solve_local(A_gpu, b_gpu, x_gpu,
                                    tol=tol, max_iter=max_iter)

    run()  # warmup
    cuda_sync()
    t0 = time.perf_counter()
    result = run()
    cuda_sync()
    elapsed = time.perf_counter() - t0

    return {'backend': 'kspmv_cg_1gpu', 'elapsed_s': elapsed,
            'iters': result['iters'], 'residual': result['residual'],
            'converged': result['converged']}


# ─── KokkosSpMV distributed CG ───────────────────────────────────────────────

def bench_kspmv_cg_dist(A, b_global, rank, size, tol, max_iter):
    if not (HAS_CUPY and HAS_KSPMV and HAS_MPI):
        return {}
    comm = MPI.COMM_WORLD
    A_dist = kspmv.distribute_csr(A.indptr.astype(np.int32),
                                   A.indices.astype(np.int32),
                                   A.data.astype(np.float64),
                                   A.shape[0], A.shape[1])
    local_start = A_dist.local_row_start
    local_end   = A_dist.local_row_end
    b_local = cp.asarray(b_global[local_start:local_end])

    def run():
        x_local = cp.zeros(local_end - local_start, dtype=cp.float64)
        return kspmv.cg_solve(A_dist, b_local, x_local,
                               tol=tol, max_iter=max_iter)

    run()  # warmup
    cuda_sync(); comm.Barrier()
    t0 = time.perf_counter()
    result = run()
    cuda_sync(); comm.Barrier()
    elapsed = time.perf_counter() - t0

    return {'backend': f'kspmv_cg_{size}gpu', 'elapsed_s': elapsed,
            'iters': result['iters'], 'residual': result['residual'],
            'converged': result['converged'], 'n_ranks': size}


# ─── SpMV-only timing ─────────────────────────────────────────────────────────

def bench_spmv_dist(A, rank, size, repeat=20):
    if not (HAS_CUPY and HAS_KSPMV):
        return {}
    comm = MPI.COMM_WORLD if HAS_MPI else None
    A_dist = kspmv.distribute_csr(A.indptr.astype(np.int32),
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

    return {'backend': f'spmv_only_{size}gpu', 'elapsed_s': elapsed,
            'n_ranks': size, 'N': A.shape[0], 'nnz': A.nnz}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CG benchmark v0.3")
    p.add_argument('--N',        type=int,   default=100_000)
    p.add_argument('--matrix',   default='laplacian_3d',
                   choices=['laplacian_2d', 'laplacian_3d',
                            'lattice_gauge', 'random_sparse'])
    p.add_argument('--tol',      type=float, default=1e-8)
    p.add_argument('--maxiter',  type=int,   default=2000)
    p.add_argument('--no-csv',   action='store_true')
    p.add_argument('--no-petsc', action='store_true',
                   help='Skip PETSc CG even if petsc4py is available')
    return p.parse_args()


def main():
    args = parse_args()

    rank, size = 0, 1
    if HAS_MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()

    if HAS_KSPMV and HAS_CUPY:
        n_gpu = cp.cuda.runtime.getDeviceCount()
        dev   = rank % n_gpu
        kspmv.init(device_id=dev)
        if rank == 0:
            print(f"  GPUs available: {n_gpu}  |  ranks: {size}")

    A = get_matrix(args.matrix, args.N)
    N, nnz = A.shape[0], A.nnz
    b = np.ones(N, dtype=np.float64)

    if rank == 0:
        print(f"\nCG Benchmark — {args.matrix}  N={N:,}  nnz={nnz:,}"
              f"  tol={args.tol:.0e}  np={size}")
        petsc_on = HAS_PETSC and not args.no_petsc and size == 1
        if petsc_on:
            print("  (petsc_cg baseline enabled)")
        elif HAS_PETSC and size > 1:
            print("  (petsc_cg skipped — np > 1, sequential petsc only)")
        else:
            print("  (petsc_cg skipped — petsc4py not installed)")
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
        if HAS_PETSC and not args.no_petsc:
            report(bench_petsc_cg(A, b, args.tol, args.maxiter, mpi_size=size))
        report(bench_kspmv_cg_local(A, b, args.tol, args.maxiter))

    if size > 1:
        report(bench_kspmv_cg_dist(A, b, rank, size, args.tol, args.maxiter))

    spmv_m = bench_spmv_dist(A, rank, size)
    if rank == 0 and spmv_m:
        report(spmv_m)

    if not args.no_csv and rank == 0 and rows:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = RESULTS_DIR / f"cg_{ts}.csv"
        all_keys, seen = [], set()
        for row in rows:
            for k_ in row.keys():
                if k_ not in seen:
                    all_keys.append(k_); seen.add(k_)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results saved to {csv_path}")


if __name__ == '__main__':
    main()
