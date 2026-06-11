cmake_minimum_required(VERSION 3.22)
project(kokkos_spmv VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Backend selection (portable: CUDA | HIP | OPENMP) ───────────────────────
#   cmake .. -DKSPMV_BACKEND=CUDA   -DKOKKOS_ARCH=AMPERE80   (NVIDIA A100)
#   cmake .. -DKSPMV_BACKEND=HIP    -DKOKKOS_ARCH=VEGA90A    (AMD MI200)
#   cmake .. -DKSPMV_BACKEND=OPENMP                          (CPU only)
set(KSPMV_BACKEND "CUDA" CACHE STRING "Kokkos backend: CUDA | HIP | OPENMP")
set(KOKKOS_ARCH   ""     CACHE STRING "Kokkos architecture (GPU backends only)")

set(KOKKOS_ENABLE_CUDA   OFF)
set(KOKKOS_ENABLE_HIP    OFF)
set(KOKKOS_ENABLE_OPENMP OFF)
if(KSPMV_BACKEND STREQUAL "CUDA")
    set(KOKKOS_ENABLE_CUDA ON)
    if(KOKKOS_ARCH STREQUAL "")
        set(KOKKOS_ARCH "AMPERE80")
    endif()
elseif(KSPMV_BACKEND STREQUAL "HIP")
    set(KOKKOS_ENABLE_HIP ON)
    if(KOKKOS_ARCH STREQUAL "")
        set(KOKKOS_ARCH "VEGA90A")
    endif()
elseif(KSPMV_BACKEND STREQUAL "OPENMP")
    set(KOKKOS_ENABLE_OPENMP ON)
else()
    message(FATAL_ERROR "KSPMV_BACKEND must be CUDA, HIP, or OPENMP")
endif()

# ─── Kokkos ───────────────────────────────────────────────────────────────────
find_package(Kokkos 4.0 QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Kokkos not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.7.02
    )
    set(Kokkos_ENABLE_CUDA   ${KOKKOS_ENABLE_CUDA}   CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_HIP    ${KOKKOS_ENABLE_HIP}    CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP ${KOKKOS_ENABLE_OPENMP} CACHE BOOL "" FORCE)
    if(NOT KOKKOS_ARCH STREQUAL "")
        set(Kokkos_ARCH_${KOKKOS_ARCH} ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_MakeAvailable(kokkos)
endif()

# ─── KokkosKernels ────────────────────────────────────────────────────────────
find_package(KokkosKernels 4.0 QUIET)
if(NOT KokkosKernels_FOUND)
    message(STATUS "KokkosKernels not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        kokkos_kernels
        GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
        GIT_TAG        4.7.02
    )
    FetchContent_MakeAvailable(kokkos_kernels)
endif()

find_package(MPI REQUIRED)
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(kokkos_spmv NB_STATIC
    src/bindings.cpp
)

target_include_directories(kokkos_spmv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(kokkos_spmv PRIVATE
    Kokkos::kokkos
    Kokkos::kokkoskernels
    MPI::MPI_CXX
)

target_compile_options(kokkos_spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

# nvcc_wrapper rejects -Os which nanobind injects
if(CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    string(REPLACE "-Os" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Os" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "nvcc_wrapper detected: removed -Os flag")
endif()

# ─── CUDA-aware MPI (GPU-direct halo exchange) ────────────────────────────────
# Default ON for CUDA backend — the Vessel AI OpenMPI is built --with-cuda.
# Falls back automatically to the (now fixed) host-staging path when OFF.
if(KSPMV_BACKEND STREQUAL "CUDA")
    option(ENABLE_CUDA_AWARE_MPI "Pass GPU pointers directly to MPI" ON)
else()
    option(ENABLE_CUDA_AWARE_MPI "Pass GPU pointers directly to MPI" OFF)
endif()
if(ENABLE_CUDA_AWARE_MPI)
    target_compile_definitions(kokkos_spmv PRIVATE CUDA_AWARE_MPI=1)
    message(STATUS "CUDA-aware MPI: ENABLED (GPU-direct halo exchange)")
else()
    message(STATUS "CUDA-aware MPI: DISABLED (host-side pack/unpack)")
endif()

# ─── C++ unit tests ──────────────────────────────────────────────────────────
option(BUILD_TESTS "Build C++ unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    foreach(test_name test_spmv_kernel test_cg)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_include_directories(${test_name} PRIVATE src/)
        target_link_libraries(${test_name} PRIVATE
            Kokkos::kokkos
            Kokkos::kokkoskernels
            MPI::MPI_CXX
        )
        if(ENABLE_CUDA_AWARE_MPI)
            target_compile_definitions(${test_name} PRIVATE CUDA_AWARE_MPI=1)
        endif()
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O2>
        )
        add_test(NAME ${test_name}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                         --allow-run-as-root ./${test_name})
    endforeach()
endif()

install(TARGETS kokkos_spmv
        LIBRARY DESTINATION ${Python_SITEARCH})

message(STATUS "")
message(STATUS "kokkos_spmv v0.2 configuration summary")
message(STATUS "  Backend        : ${KSPMV_BACKEND}")
message(STATUS "  Architecture   : ${KOKKOS_ARCH}")
message(STATUS "  CUDA-aware MPI : ${ENABLE_CUDA_AWARE_MPI}")
message(STATUS "  Build tests    : ${BUILD_TESTS}")
message(STATUS "")
