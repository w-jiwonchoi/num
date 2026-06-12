#!/usr/bin/env bash
# run_all_benchmarks.sh — N × sparsity × task × batch 전체 스윕 + 시각화
set -e
REPO=${1:-/root/num}
cd "$REPO"

export OMP_PROC_BIND=false
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# CUDA-aware 빌드면 1, host-staging 빌드면 0과 일치시켜야 함
CUDA_AWARE=$(python3 -c "import sys; sys.path.insert(0,'python'); \
import kokkos_spmv as k; print(int(k.cuda_aware_mpi()))")
export OMPI_MCA_opal_cuda_support=$CUDA_AWARE
echo ">>> CUDA-aware MPI runtime flag: $CUDA_AWARE"

# power_law를 제외한 일반 행렬들
MATRICES="laplacian_2d laplacian_3d random_sparse lattice_gauge"
NS_SINGLE="100000 300000 1000000 3000000"

# Fix 1 반영 전까지 병목 방지를 위해 3M을 제외한 power_law 전용 N
NS_POWER="100000 300000 1000000"
NS_DIST="1000000 4000000 8000000"

# ── 1. Single-GPU: 일반 행렬 모든 sparsity × N, batch sweep 포함 ────────────
for M in $MATRICES; do
  for N in $NS_SINGLE; do
    echo ">>> [1-GPU] matrix=$M N=$N"
    python python/benchmark_spmv.py --N $N --matrix $M \
        --batch-sweep 8 32 128 --repeat 100
  done
done

# ── 1-1. Single-GPU: power_law 행렬은 N=1M까지만 실행 ───────────────────────
for N in $NS_POWER; do
  echo ">>> [1-GPU] matrix=power_law N=$N"
  python python/benchmark_spmv.py --N $N --matrix power_law \
      --batch-sweep 8 32 128 --repeat 100
done

# ── 2. Distributed: 2 GPU와 4 GPU, kk/custom 둘 다, overlap on/off ────────
for NP in 2 4; do
  for M in laplacian_3d random_sparse; do
    for N in $NS_DIST; do
      echo ">>> [${NP}-GPU dist] matrix=$M N=$N"
      mpiexec --allow-run-as-root -np $NP \
          python python/benchmark_spmv.py --N $N --matrix $M \
          --backend kokkos_dist --no-overlap --repeat 50
    done
  done
done

# ── 3. CG: 1 GPU vs 4 GPU vs scipy vs petsc ───────────────────────────────
for N in 1000000 4000000; do
  echo ">>> [CG] N=$N (1 GPU + baselines)"
  python python/benchmark_cg.py --matrix laplacian_3d --N $N
  echo ">>> [CG] N=$N (4 GPU)"
  mpiexec --allow-run-as-root -np 4 \
      python python/benchmark_cg.py --matrix laplacian_3d --N $N
done

# ── 4. 종합 시각화 ──────────────────────────────────────────────────────────
echo ">>> Generating comprehensive plots..."
python python/plot_comprehensive.py
echo "==== All benchmarks complete. Plots: benchmark/plots/ ===="
