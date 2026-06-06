cmake_minimum_required(VERSION 3.22)
project(kokkos_spmv VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Backend options ──────────────────────────────────────────────────────────
option(KOKKOS_ENABLE_CUDA   "Enable CUDA backend"   ON)
option(KOKKOS_ENABLE_HIP    "Enable HIP backend"    OFF)
option(KOKKOS_ENABLE_OPENMP "Enable OpenMP backend" OFF)

# Architecture: A100 → AMPERE80, V100 → VOLTA70, H100 → HOPPER90
set(KOKKOS_ARCH "AMPERE80" CACHE STRING "Kokkos architecture flag")

# ─── Kokkos ───────────────────────────────────────────────────────────────────
# Prefer an installed Kokkos (as in the Vessel AI Docker image); fall back to
# FetchContent.
find_package(Kokkos 4.0 QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Kokkos not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.7.02          # matches Vessel AI Docker image
    )
    set(Kokkos_ENABLE_CUDA   ${KOKKOS_ENABLE_CUDA}   CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_HIP    ${KOKKOS_ENABLE_HIP}    CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP ${KOKKOS_ENABLE_OPENMP} CACHE BOOL "" FORCE)
    set(Kokkos_ARCH_${KOKKOS_ARCH} ON CACHE BOOL "" FORCE)
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
        GIT_TAG        4.7.02          # matches Vessel AI Docker image
    )
    FetchContent_MakeAvailable(kokkos_kernels)
endif()

# ─── MPI ──────────────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)

# ─── Python + nanobind ────────────────────────────────────────────────────────
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# ─── Extension module ────────────────────────────────────────────────────────
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

# ─── nanobind -Os workaround ──────────────────────────────────────────────────
# nanobind injects -Os into compile flags.  nvcc_wrapper does NOT support -Os
# (it passes unknown flags straight to nvcc which rejects them).
# Override with -O3 for CXX (which nvcc_wrapper translates to --optimize 3).
# This matches the note in the project PPTX.
target_compile_options(kokkos_spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

# Strip -Os that nanobind may have added (CMake 3.25+ supports
# COMPILE_OPTIONS target property manipulation; for older versions we rely on
# the override above since -O3 appears later and wins on GCC/Clang/nvcc).
# For belt-and-suspenders, also set the global flag when using nvcc_wrapper:
if(CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    string(REPLACE "-Os" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Os" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "nvcc_wrapper detected: removed -Os flag")
endif()

# ─── CUDA-aware MPI ───────────────────────────────────────────────────────────
option(ENABLE_CUDA_AWARE_MPI "Pass GPU pointers directly to MPI" OFF)
if(ENABLE_CUDA_AWARE_MPI)
    target_compile_definitions(kokkos_spmv PRIVATE CUDA_AWARE_MPI=1)
    message(STATUS "CUDA-aware MPI: ENABLED")
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
        # Also apply the -Os workaround to test binaries
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O2>
        )
        add_test(NAME ${test_name}
                 COMMAND
                     ${MPIEXEC_EXECUTABLE}
                     ${MPIEXEC_NUMPROC_FLAG} 1
                     # Allow root execution (required in Vessel AI containers)
                     --allow-run-as-root
                     ./${test_name})
    endforeach()
endif()

# ─── Install ─────────────────────────────────────────────────────────────────
install(TARGETS kokkos_spmv
        LIBRARY DESTINATION ${Python_SITEARCH})

message(STATUS "")
message(STATUS "kokkos_spmv configuration summary")
message(STATUS "  CUDA backend  : ${KOKKOS_ENABLE_CUDA}")
message(STATUS "  HIP backend   : ${KOKKOS_ENABLE_HIP}")
message(STATUS "  OpenMP backend: ${KOKKOS_ENABLE_OPENMP}")
message(STATUS "  Architecture  : ${KOKKOS_ARCH}")
message(STATUS "  Build tests   : ${BUILD_TESTS}")
message(STATUS "")
