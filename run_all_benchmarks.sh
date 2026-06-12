#!/usr/bin/env bash
# run_all_benchmarks.sh  (v0.3)
# ─────────────────────────────────────────────────────────────────────────────
# 변경 vs v0.2:
#   - dist 측정에서 --no-overlap 제거 (benchmark_spmv.py v0.3 에서 항상 측정)
#   - batch sweep 에 256, 512 추가
#   - power_law 를 3M 까지 확장 (이전엔 1M 로 제한)
#   - 4GPU 이득 분석을 위해 dist 크기를 16M 까지 확대
# ─────────────────────────────────────────────────────────────────────────────
set -e
REPO=${1:-/root/num}
cd "$REPO"

export OMP_PROC_BIND=false
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

CUDA_AWARE=$(python3 -c "import sys; sys.path.insert(0,'python'); \
import kokkos_spmv as k; print(int(k.cuda_aware_mpi()))")
export OMPI_MCA_opal_cuda_support=$CUDA_AWARE
echo ">>> CUDA-aware MPI runtime flag: $CUDA_AWARE"

MATRICES="laplacian_2d laplacian_3d random_sparse lattice_gauge"
NS_SINGLE="100000 300000 1000000 3000000"
NS_POWER="100000 300000 1000000 3000000"          # v0.3: 3M 추가
NS_DIST="1000000 4000000 8000000 16000000"        # v0.3: 16M 추가
BATCH_SWEEP="8 32 128 256 512"                    # v0.3: 256, 512 추가

# ── 1. Single-GPU: 일반 행렬 ─────────────────────────────────────────────────
for M in $MATRICES; do
  for N in $NS_SINGLE; do
    echo ">>> [1-GPU] matrix=$M N=$N"
    python python/benchmark_spmv.py --N $N --matrix $M \
        --batch-sweep $BATCH_SWEEP --repeat 100
  done
done

# ── 1-1. Single-GPU: power_law ───────────────────────────────────────────────
for N in $NS_POWER; do
  echo ">>> [1-GPU] matrix=power_law N=$N"
  python python/benchmark_spmv.py --N $N --matrix power_law \
      --batch-sweep $BATCH_SWEEP --repeat 100
done

# ── 2. Distributed: 2 GPU & 4 GPU
#       v0.3 변경: --no-overlap 제거 (noovl 은 benchmark_spmv.py 내에서 자동)
for NP in 2 4; do
  for M in laplacian_3d random_sparse; do
    for N in $NS_DIST; do
      echo ">>> [${NP}-GPU dist] matrix=$M N=$N"
      mpiexec --allow-run-as-root -np $NP \
          python python/benchmark_spmv.py --N $N --matrix $M \
          --backend kokkos_dist --repeat 50
    done
  done
done

# ── 3. CG: 1 GPU vs 4 GPU vs scipy vs petsc ──────────────────────────────────
for N in 1000000 4000000; do
  echo ">>> [CG] N=$N (1 GPU + baselines)"
  python python/benchmark_cg.py --matrix laplacian_3d --N $N
  echo ">>> [CG] N=$N (4 GPU)"
  mpiexec --allow-run-as-root -np 4 \
      python python/benchmark_cg.py --matrix laplacian_3d --N $N
done

# ── 4. 종합 시각화 ────────────────────────────────────────────────────────────
echo ">>> Generating comprehensive plots..."
python python/plot_comprehensive.py
echo "==== All benchmarks complete. Plots: benchmark/plots/ ===="
