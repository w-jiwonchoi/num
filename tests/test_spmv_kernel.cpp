/**
 * test_spmv_kernel.cpp
 * ====================
 * C++ unit tests for SpMV and batch SpMV kernels.
 *
 * Compile and run via CMake:
 *   cmake --build build --target test_spmv_kernel
 *   mpiexec -np 1 ./build/test_spmv_kernel
 */

#include "spmv_kernel.hpp"
#include "batch_spmv_kernel.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>

// ─── Minimal test framework ───────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;

void check(const std::string& name, bool cond, const std::string& msg = "") {
    if (cond) {
        std::printf("  ✓  %s\n", name.c_str());
        ++g_pass;
    } else {
        std::printf("  ✗  %s  %s\n", name.c_str(), msg.c_str());
        ++g_fail;
    }
}

void check_close(const std::string& name, double got, double expected,
                 double tol = 1e-10) {
    double err = std::abs(got - expected);
    bool   ok  = err < tol;
    char   buf[64];
    std::snprintf(buf, sizeof(buf), "(got=%.6e  exp=%.6e  err=%.2e)", got, expected, err);
    check(name, ok, ok ? "" : buf);
}

// ─── Reference: dense SpMV on host ───────────────────────────────────────────

std::vector<double> dense_spmv(
    const std::vector<double>& values,
    const std::vector<int>&    colind,
    const std::vector<int>&    rowptr,
    const std::vector<double>& x,
    int nrows)
{
    std::vector<double> y(nrows, 0.0);
    for (int i = 0; i < nrows; ++i)
        for (int j = rowptr[i]; j < rowptr[i+1]; ++j)
            y[i] += values[j] * x[colind[j]];
    return y;
}

// ─── Build a simple 4×4 tridiagonal matrix ────────────────────────────────────
//
//  A = [ 2 -1  0  0 ]
//      [-1  2 -1  0 ]
//      [ 0 -1  2 -1 ]
//      [ 0  0 -1  2 ]
//
struct TriDiag4 {
    std::vector<double> values = { 2,-1, -1,2,-1, -1,2,-1, -1,2 };
    std::vector<int>    colind = { 0,1,  0,1,2,   1,2,3,   2,3 };
    std::vector<int>    rowptr = { 0, 2, 5, 8, 10 };
    int nrows = 4, ncols = 4;
};

// ─── Test 1: 4×4 tridiagonal SpMV ────────────────────────────────────────────

void test_tridiag_spmv() {
    std::puts("\n[1] Tridiagonal 4×4 SpMV");
    TriDiag4 td;
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};

    auto y_ref = dense_spmv(td.values, td.colind, td.rowptr, x, td.nrows);
    auto y_got = spmv_host_roundtrip(td.values, td.colind, td.rowptr, x,
                                     td.nrows, td.ncols, /*kokkoskernels=*/false);

    for (int i = 0; i < td.nrows; ++i) {
        double err = std::abs(y_got[i] - y_ref[i]);
        check("  row " + std::to_string(i), err < 1e-13,
              "(got=" + std::to_string(y_got[i]) +
              " ref=" + std::to_string(y_ref[i]) + ")");
    }
}

// ─── Test 2: KokkosKernels backend ───────────────────────────────────────────

void test_kokkoskernels_backend() {
    std::puts("\n[2] KokkosKernels backend");
    TriDiag4 td;
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};

    auto y_ref     = dense_spmv(td.values, td.colind, td.rowptr, x, td.nrows);
    auto y_custom  = spmv_host_roundtrip(td.values, td.colind, td.rowptr, x,
                                          td.nrows, td.ncols, false);
    auto y_kk      = spmv_host_roundtrip(td.values, td.colind, td.rowptr, x,
                                          td.nrows, td.ncols, true);

    for (int i = 0; i < td.nrows; ++i) {
        double err_custom = std::abs(y_custom[i] - y_ref[i]);
        double err_kk     = std::abs(y_kk[i]     - y_ref[i]);
        check("  custom row " + std::to_string(i), err_custom < 1e-13);
        check("  kk     row " + std::to_string(i), err_kk     < 1e-13);
    }
}

// ─── Test 3: Identity matrix SpMV → y = x ────────────────────────────────────

void test_identity() {
    std::puts("\n[3] Identity matrix (y must equal x)");
    int N = 100;
    std::vector<double> values(N, 1.0);
    std::vector<int>    colind(N), rowptr(N + 1);
    for (int i = 0; i < N; ++i) { colind[i] = i; rowptr[i] = i; }
    rowptr[N] = N;

    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = static_cast<double>(i) * 0.1;

    auto y = spmv_host_roundtrip(values, colind, rowptr, x, N, N, false);
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) max_err = std::max(max_err, std::abs(y[i] - x[i]));
    check("  max_err < 1e-14", max_err < 1e-14,
          "(max_err=" + std::to_string(max_err) + ")");
}

// ─── Test 4: Batch SpMV correctness ──────────────────────────────────────────

void test_batch_spmv() {
    std::puts("\n[4] Batch SpMV (k=4)");
    TriDiag4 td;
    int k = 4;

    // X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]  (identity, shape 4×4)
    std::vector<double> X_flat(td.nrows * k, 0.0);
    for (int i = 0; i < td.nrows && i < k; ++i)
        X_flat[i * k + i] = 1.0;

    auto Y_flat = batch_spmv_host_roundtrip(td.values, td.colind, td.rowptr,
                                             X_flat, td.nrows, td.ncols, k);

    // Y[:,b] should be A[:,b]  (= column b of A)
    // Column 0 of A: [2,-1,0,0]
    check_close("  Y[0,0] = 2",  Y_flat[0 * k + 0], 2.0);
    check_close("  Y[1,0] = -1", Y_flat[1 * k + 0], -1.0);
    check_close("  Y[2,0] = 0",  Y_flat[2 * k + 0], 0.0);
    check_close("  Y[3,0] = 0",  Y_flat[3 * k + 0], 0.0);

    // Column 1: [-1,2,-1,0]
    check_close("  Y[0,1] = -1", Y_flat[0 * k + 1], -1.0);
    check_close("  Y[1,1] = 2",  Y_flat[1 * k + 1], 2.0);
    check_close("  Y[2,1] = -1", Y_flat[2 * k + 1], -1.0);
    check_close("  Y[3,1] = 0",  Y_flat[3 * k + 1], 0.0);
}

// ─── Test 5: Large random matrix ─────────────────────────────────────────────

void test_large_random() {
    std::puts("\n[5] Large random sparse matrix (N=10000, nnz_per_row≈7)");
    int N = 10000, nnz_per_row = 7;
    unsigned seed = 42;

    std::vector<int>    rowptr(N + 1, 0);
    std::vector<int>    colind;
    std::vector<double> values;
    std::vector<double> x(N);

    // Simple seeded linear congruential generator
    auto lcg = [&]() -> unsigned {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };

    for (int i = 0; i < N; ++i) x[i] = (lcg() % 1000) * 0.001 - 0.5;

    for (int i = 0; i < N; ++i) {
        // Diagonal always present
        colind.push_back(i);
        values.push_back(static_cast<double>(nnz_per_row));

        // Random off-diagonals
        for (int t = 1; t < nnz_per_row; ++t) {
            int j = lcg() % N;
            if (j != i) {
                colind.push_back(j);
                values.push_back(-1.0);
            }
        }
        rowptr[i + 1] = colind.size();
    }

    auto y_ref = dense_spmv(values, colind, rowptr, x, N);
    auto y_got = spmv_host_roundtrip(values, colind, rowptr, x, N, N, false);

    double max_err = 0.0, rel_err = 0.0, ref_norm = 0.0;
    for (int i = 0; i < N; ++i) {
        max_err  = std::max(max_err, std::abs(y_got[i] - y_ref[i]));
        ref_norm = std::max(ref_norm, std::abs(y_ref[i]));
    }
    rel_err = max_err / (ref_norm + 1e-300);

    char buf[80];
    std::snprintf(buf, sizeof(buf), "(rel_err=%.2e)", rel_err);
    check("  rel_err < 1e-12", rel_err < 1e-12, buf);
}

// ─── Entry point ──────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Initialise Kokkos (and MPI if compiled in)
    Kokkos::InitializationSettings settings;
    settings.set_num_threads(1);
    Kokkos::initialize(settings);

    std::puts("=== KokkosSpMV Kernel Unit Tests ===");

    try {
        test_tridiag_spmv();
        test_kokkoskernels_backend();
        test_identity();
        test_batch_spmv();
        test_large_random();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nFATAL: %s\n", e.what());
        Kokkos::finalize();
        return 1;
    }

    Kokkos::finalize();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
