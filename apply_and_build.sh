#!/usr/bin/env bash
# apply_and_build.sh — repo에 수정 파일 적용 후 CUDA-aware MPI로 재빌드
set -e
REPO=${1:-/root/num}
FIX=$(cd "$(dirname "$0")" && pwd)

echo "==> Applying v0.2 fixes to $REPO"
cp "$FIX/src/halo_exchange.hpp"  "$REPO/src/"
cp "$FIX/src/crs_matrix.hpp"     "$REPO/src/"
cp "$FIX/src/dist_spmv.hpp"      "$REPO/src/"
cp "$FIX/src/bindings.cpp"       "$REPO/src/"
cp "$FIX/CMakeLists.txt"         "$REPO/"
cp "$FIX/python/benchmark_spmv.py"      "$REPO/python/"
cp "$FIX/python/plot_comprehensive.py"  "$REPO/python/"

echo "==> Rebuilding (CUDA-aware MPI ON)"
cd "$REPO"
rm -rf build && mkdir build && cd build
CXX=nvcc_wrapper cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DKSPMV_BACKEND=CUDA \
    -DKOKKOS_ARCH=AMPERE80 \
    -DENABLE_CUDA_AWARE_MPI=ON \
    -DBUILD_TESTS=ON \
    -Dnanobind_DIR=$(python3 -c "import nanobind; print(nanobind.cmake_dir())")
make -j$(nproc)

cd "$REPO"
rm -f python/kokkos_spmv*.so
ln -s "$REPO"/build/kokkos_spmv*.so "$REPO"/python/ 2>/dev/null || true

echo "==> Sanity: distributed correctness test (4 ranks)"
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1
cd "$REPO"
mpiexec --allow-run-as-root -np 4 python3 tests/test_halo_exchange.py || {
    echo '!!! CUDA-aware path failed — rebuilding with host-staging fallback'
    cd build
    cmake .. -DENABLE_CUDA_AWARE_MPI=OFF && make -j$(nproc)
    cd ..
    export OMPI_MCA_opal_cuda_support=0
    mpiexec --allow-run-as-root -np 4 python3 tests/test_halo_exchange.py
}
echo "==== Build + verification complete ===="
