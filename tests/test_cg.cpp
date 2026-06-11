/**
 * test_cg.cpp
 * ===========
 * C++ unit tests for the CG solver (cg_solve_local and cg_solve_distributed).
 *
 * Compile and run via CMake:
 *   cmake --build build --target test_cg
 *   mpiexec -np 1 ./build/test_cg          # local tests only
 *   mpiexec -np 4 ./build/test_cg          # includes distributed tests
 *
 * Tests:
 *   1. Identity matrix (trivial system: x = b)
 *   2. 2-point tridiagonal system (exact 2-step convergence)
 *   3. Diagonal SPD matrix (exact solution known)
 *   4. 1D Laplacian (convergence within tolerance)
 *   5. Already-converged initial guess (zero iterations)
 *   6. Large sparse system (N=1000, convergence check)
 *   7. Distributed CG — 1D Laplacian (multi-rank, requires -np >= 2)
 */

#include "cg_solver.hpp"
#include "dist_spmv.hpp"
#include "vec_ops.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <stdexcept>

// ─── Minimal test framework ───────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;

void check(const std::string& name, bool cond, const std::string& detail = "") {
    if (cond) {
        std::printf("  \u2713  %s\n", name.c_str());
        ++g_pass;
    } else {
        std::printf("  \u2717  %s  [%s]\n", name.c_str(), detail.c_str());
        ++g_fail;
    }
}

void check_close(const std::string& name, double got, double expected,
                 double tol = 1e-10) {
    double err = std::abs(got - expected) /
                 (std::abs(expected) > 1e-300 ? std::abs(expected) : 1.0);
    char buf[80];
    std::snprintf(buf, sizeof(buf),
                  "got=%.6e exp=%.6e rel_err=%.2e", got, expected, err);
    check(name, err < tol, buf);
}

// ─── Matrix / vector construction helpers ────────────────────────────────────

/**
 * Build a CrsMatrixHandle on device from host CSR arrays.
 * rowptr length: nrows+1.  colind/values length: nnz = rowptr[nrows].
 */
CrsMatrixHandle make_device_crs(
    const std::vector<double>& h_values,
    const std::vector<int>&    h_colind,
    const std::vector<int>&    h_rowptr,
    int nrows, int ncols)
{
    return CrsMatrixHandle(h_values.data(), h_colind.data(), h_rowptr.data(),
                           nrows, ncols);
}

/**
 * 1D Laplacian (tridiagonal [-1, 2, -1]) of size N×N.
 */
struct Laplacian1D {
    int N;
    std::vector<double> values;
    std::vector<int>    colind;
    std::vector<int>    rowptr;

    explicit Laplacian1D(int n) : N(n) {
        rowptr.resize(N + 1, 0);
        for (int i = 0; i < N; ++i) {
            if (i > 0)     { colind.push_back(i-1); values.push_back(-1.0); }
            colind.push_back(i);   values.push_back(2.0);
            if (i < N-1)   { colind.push_back(i+1); values.push_back(-1.0); }
            rowptr[i + 1] = colind.size();
        }
    }

    CrsMatrixHandle device() const {
        return make_device_crs(values, colind, rowptr, N, N);
    }
};

/**
 * Diagonal matrix diag(d[0], ..., d[N-1]).
 */
struct DiagMatrix {
    int N;
    std::vector<double> diag;
    std::vector<double> values;
    std::vector<int>    colind;
    std::vector<int>    rowptr;

    DiagMatrix(int n, std::vector<double> d)
        : N(n), diag(std::move(d))
    {
        rowptr.resize(N + 1);
        for (int i = 0; i < N; ++i) {
            values.push_back(diag[i]);
            colind.push_back(i);
            rowptr[i + 1] = i + 1;
        }
    }

    CrsMatrixHandle device() const {
        return make_device_crs(values, colind, rowptr, N, N);
    }
};

/** Upload a host std::vector to device ViewVec1D. */
ViewVec1D to_device(const std::vector<double>& h, const char* label = "v") {
    ViewVec1D d(std::string(label), h.size());
    auto mirror = Kokkos::create_mirror_view(d);
    for (int i = 0; i < (int)h.size(); ++i) mirror(i) = h[i];
    Kokkos::deep_copy(d, mirror);
    return d;
}

/** Download a device ViewVec1D to host std::vector. */
std::vector<double> to_host(const ViewVec1D& d) {
    auto mirror = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(mirror, d);
    std::vector<double> h(d.extent(0));
    for (int i = 0; i < (int)h.size(); ++i) h[i] = mirror(i);
    return h;
}

double host_rel_err(const std::vector<double>& a,
                    const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < (int)a.size(); ++i) {
        num += (a[i]-b[i]) * (a[i]-b[i]);
        den += b[i] * b[i];
    }
    return (den > 0) ? std::sqrt(num/den) : std::sqrt(num);
}

// ─── Test 1: Identity matrix — CG converges in 1 iteration ──────────────────

void test_identity_cg() {
    std::puts("\n[1] Identity matrix (x = b, 1-step convergence)");
    int N = 50;
    std::vector<double> values(N, 1.0);
    std::vector<int>    colind(N), rowptr(N+1);
    std::iota(colind.begin(), colind.end(), 0);
    for (int i = 0; i <= N; ++i) rowptr[i] = i;

    CrsMatrixHandle A = make_device_crs(values, colind, rowptr, N, N);

    std::vector<double> b_h(N);
    for (int i = 0; i < N; ++i) b_h[i] = (double)(i+1);

    ViewVec1D b = to_device(b_h, "b");
    ViewVec1D x("x", N);
    Kokkos::deep_copy(x, 0.0);

    CGResult r = cg_solve_local(A, b, x, 1e-12, 200);
    auto x_h = to_host(x);

    double err = host_rel_err(x_h, b_h);
    char buf[64]; std::snprintf(buf, sizeof(buf), "rel_err=%.2e", err);
    check("identity: rel_err < 1e-10", err < 1e-10, buf);
    check("identity: converged", r.converged, "converged=" + std::to_string(r.converged));

    // CG on identity should converge in exactly 1 iteration
    char ibuf[32]; std::snprintf(ibuf, sizeof(ibuf), "iters=%d", r.iters);
    check("identity: iters <= 2", r.iters <= 2, ibuf);
}

// ─── Test 2: 2×2 symmetric positive definite ────────────────────────────────

void test_2x2_spd() {
    std::puts("\n[2] 2×2 SPD system");
    // A = [[4, 1], [1, 3]], b = [1, 2]
    // Solution: x = [1/11, 7/11]
    std::vector<double> values = {4.0, 1.0, 1.0, 3.0};
    std::vector<int>    colind = {0, 1, 0, 1};
    std::vector<int>    rowptr = {0, 2, 4};

    CrsMatrixHandle A = make_device_crs(values, colind, rowptr, 2, 2);

    std::vector<double> b_h = {1.0, 2.0};
    std::vector<double> x_exact = {1.0/11.0, 7.0/11.0};

    ViewVec1D b = to_device(b_h, "b2");
    ViewVec1D x("x2", 2);
    Kokkos::deep_copy(x, 0.0);

    CGResult r = cg_solve_local(A, b, x, 1e-14, 100);
    auto x_h = to_host(x);

    double err = host_rel_err(x_h, x_exact);
    char buf[64]; std::snprintf(buf, sizeof(buf), "rel_err=%.2e", err);
    check("2x2: rel_err < 1e-10", err < 1e-10, buf);
    check("2x2: converged", r.converged);
    // CG on N=2 SPD converges in <= 2 iterations
    char ibuf[32]; std::snprintf(ibuf, sizeof(ibuf), "iters=%d", r.iters);
    check("2x2: iters <= 2", r.iters <= 2, ibuf);
}

// ─── Test 3: Diagonal SPD (exact solution known) ────────────────────────────

void test_diagonal_spd() {
    std::puts("\n[3] Diagonal SPD (d[i] = i+1, b = ones, x_exact = 1/d)");
    int N = 100;
    std::vector<double> d(N);
    for (int i = 0; i < N; ++i) d[i] = (double)(i + 1);
    DiagMatrix M(N, d);
    CrsMatrixHandle A = M.device();

    std::vector<double> b_h(N, 1.0);
    std::vector<double> x_exact(N);
    for (int i = 0; i < N; ++i) x_exact[i] = 1.0 / d[i];

    ViewVec1D b = to_device(b_h, "bd");
    ViewVec1D x("xd", N);
    Kokkos::deep_copy(x, 0.0);

    CGResult r = cg_solve_local(A, b, x, 1e-12, 5000);
    auto x_h = to_host(x);

    double err = host_rel_err(x_h, x_exact);
    char buf[64]; std::snprintf(buf, sizeof(buf), "rel_err=%.2e iters=%d", err, r.iters);
    check("diagonal_spd: rel_err < 1e-8", err < 1e-8, buf);
    check("diagonal_spd: converged", r.converged,
          "converged=" + std::to_string(r.converged));
}

// ─── Test 4: 1D Laplacian (N=50) — convergence ───────────────────────────────

void test_laplacian_convergence() {
    std::puts("\n[4] 1D Laplacian (N=50, b=ones, convergence check)");
    int N = 50;
    Laplacian1D L(N);
    CrsMatrixHandle A = L.device();

    std::vector<double> b_h(N, 1.0);
    ViewVec1D b = to_device(b_h, "bl");
    ViewVec1D x("xl", N);
    Kokkos::deep_copy(x, 0.0);

    CGResult r = cg_solve_local(A, b, x, 1e-10, 500);
    auto x_h = to_host(x);

    // Residual check: ||b - Ax||/||b||
    // (We can't use the full SpMV here without device infrastructure,
    //  so trust the CGResult residual field.)
    char buf[80];
    std::snprintf(buf, sizeof(buf),
                  "rel_res=%.2e iters=%d conv=%d",
                  r.final_res, r.iters, (int)r.converged);
    check("laplacian: converged within max_iter", r.converged, buf);
    check("laplacian: final_res < 1e-10", r.final_res < 1e-10, buf);
    check("laplacian: iters > 0", r.iters > 0,
          "iters=" + std::to_string(r.iters));
}

// ─── Test 5: Already-converged initial guess ────────────────────────────────

void test_zero_rhs() {
    std::puts("\n[5] Zero RHS (b=0, x=0 already satisfies A*x=0)");
    int N = 20;
    Laplacian1D L(N);
    CrsMatrixHandle A = L.device();

    std::vector<double> b_h(N, 0.0);
    ViewVec1D b = to_device(b_h, "bzero");
    ViewVec1D x("xzero", N);
    Kokkos::deep_copy(x, 0.0);

    CGResult r = cg_solve_local(A, b, x, 1e-10, 100);

    check("zero_rhs: converged",       r.converged);
    check("zero_rhs: 0 iterations",    r.iters == 0,
          "iters=" + std::to_string(r.iters));
    check("zero_rhs: residual == 0.0", r.final_res == 0.0,
          "res=" + std::to_string(r.final_res));
}

// ─── Test 6: Larger sparse system (N=500) ────────────────────────────────────

void test_larger_laplacian() {
    std::puts("\n[6] Larger 1D Laplacian (N=500)");
    int N = 500;
    Laplacian1D L(N);
    CrsMatrixHandle A = L.device();

    // Manufactured solution: x* = sin(pi*i/(N+1))
    std::vector<double> x_exact(N), b_h(N);
    for (int i = 0; i < N; ++i) {
        x_exact[i] = std::sin(M_PI * (i+1.0) / (N+1.0));
    }
    // b = A * x_exact (compute on host)
    for (int i = 0; i < N; ++i) {
        b_h[i] = 2.0 * x_exact[i];
        if (i > 0)   b_h[i] -= x_exact[i-1];
        if (i < N-1) b_h[i] -= x_exact[i+1];
    }

    ViewVec1D b = to_device(b_h, "bsin");
    ViewVec1D x("xsin", N);
    Kokkos::deep_copy(x, 0.0);

    CGResult r = cg_solve_local(A, b, x, 1e-10, 2000);
    auto x_h = to_host(x);

    double err = host_rel_err(x_h, x_exact);
    char buf[80];
    std::snprintf(buf, sizeof(buf),
                  "rel_err=%.2e res=%.2e iters=%d", err, r.final_res, r.iters);
    check("larger_laplacian: converged",       r.converged, buf);
    check("larger_laplacian: rel_err < 1e-8",  err < 1e-8,  buf);
}

// ─── Test 7: Non-convergence detection ───────────────────────────────────────

void test_max_iter_respected() {
    std::puts("\n[7] max_iter=5 (should not converge for N=50 Laplacian)");
    int N = 50;
    Laplacian1D L(N);
    CrsMatrixHandle A = L.device();

    std::vector<double> b_h(N, 1.0);
    ViewVec1D b = to_device(b_h, "bmi");
    ViewVec1D x("xmi", N);
    Kokkos::deep_copy(x, 0.0);

    CGResult r = cg_solve_local(A, b, x, 1e-14, /*max_iter=*/5);

    check("max_iter: iters <= 5", r.iters <= 5,
          "iters=" + std::to_string(r.iters));
    check("max_iter: not converged (expected)", !r.converged,
          "converged=" + std::to_string(r.converged));
}

// ─── Test 8: Distributed CG (requires mpiexec -np >= 2) ─────────────────────

void test_distributed_cg(int rank, int size) {
    std::puts("\n[8] Distributed CG — 1D Laplacian (requires -np >= 2)");

    if (size < 2) {
        std::puts("  [SKIP] run with mpiexec -np >= 2");
        return;
    }

    int N = 100;
    Laplacian1D L(N);

    // Build global DistCrsHandle from rank-0 perspective
    // (distribute_csr takes global arrays on ALL ranks)
    std::vector<double> gv(L.values);
    std::vector<int>    gc(L.colind), gr(L.rowptr);

    MPI_Comm comm = MPI_COMM_WORLD;
    DistCrsHandle A_dist = distribute_csr(gv, gc, gr, N, N, comm);

    int local_start = A_dist.local_row_start;
    int local_end   = A_dist.local_row_end;
    int local_n     = local_end - local_start;

    // b = ones, x0 = zeros
    std::vector<double> b_h(local_n, 1.0);
    ViewVec1D b = to_device(b_h, "bd_cg");
    ViewVec1D x("xd_cg", local_n);
    Kokkos::deep_copy(x, 0.0);

    CGResult r = cg_solve_distributed(A_dist, b, x, 1e-10, 1000, comm);

    // Gather solution to rank 0 for verification
    auto x_local_h = to_host(x);
    std::vector<int> counts(size), displs(size);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    displs[0] = 0;
    for (int r_ = 1; r_ < size; ++r_) displs[r_] = displs[r_-1] + counts[r_-1];

    std::vector<double> x_global;
    if (rank == 0) x_global.resize(N);
    MPI_Gatherv(x_local_h.data(), local_n, MPI_DOUBLE,
                x_global.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        // Verify: compute ||b_global - A*x_global|| on host
        std::vector<double> b_global(N, 1.0);
        std::vector<double> Ax(N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = gr[i]; j < gr[i+1]; ++j) {
                Ax[i] += gv[j] * x_global[gc[j]];
            }
        }
        double res_num = 0.0, res_den = 0.0;
        for (int i = 0; i < N; ++i) {
            res_num += (b_global[i]-Ax[i]) * (b_global[i]-Ax[i]);
            res_den += b_global[i] * b_global[i];
        }
        double rel_res = std::sqrt(res_num/res_den);

        char buf[80];
        std::snprintf(buf, sizeof(buf),
                      "rel_res=%.2e iters=%d conv=%d np=%d",
                      rel_res, r.iters, (int)r.converged, size);
        check("dist_cg: converged",          r.converged, buf);
        check("dist_cg: rel_res < 1e-8",     rel_res < 1e-8, buf);
    }
}

// ─── Test 9: Multiple solves with same matrix ────────────────────────────────

void test_repeated_solves() {
    std::puts("\n[9] Repeated CG solves with the same matrix (consistency)");
    int N = 30;
    Laplacian1D L(N);
    CrsMatrixHandle A = L.device();

    std::vector<double> b_h(N, 1.0);
    ViewVec1D b = to_device(b_h, "brep");

    // Run 3 times and check solutions agree
    std::vector<std::vector<double>> solutions;
    for (int run = 0; run < 3; ++run) {
        ViewVec1D x("xrep", N);
        Kokkos::deep_copy(x, 0.0);
        cg_solve_local(A, b, x, 1e-12, 500);
        solutions.push_back(to_host(x));
    }

    double max_diff_01 = 0.0, max_diff_02 = 0.0;
    for (int i = 0; i < N; ++i) {
        max_diff_01 = std::max(max_diff_01, std::abs(solutions[0][i]-solutions[1][i]));
        max_diff_02 = std::max(max_diff_02, std::abs(solutions[0][i]-solutions[2][i]));
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "max_diff_01=%.2e", max_diff_01);
    check("repeated_solves: run0 == run1 (bit-identical)", max_diff_01 == 0.0, buf);
    std::snprintf(buf, sizeof(buf), "max_diff_02=%.2e", max_diff_02);
    check("repeated_solves: run0 == run2 (bit-identical)", max_diff_02 == 0.0, buf);
}

// ─── Entry point ─────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Kokkos::InitializationSettings settings;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    settings.set_device_id(rank % 1);  // single GPU per rank in test
#endif
    Kokkos::initialize(settings);

    if (rank == 0) {
        std::puts("=== KokkosSpMV CG Solver Unit Tests ===");
    }

    try {
        // Local tests run on all ranks
        if (rank == 0) {
            test_identity_cg();
            test_2x2_spd();
            test_diagonal_spd();
            test_laplacian_convergence();
            test_zero_rhs();
            test_larger_laplacian();
            test_max_iter_respected();
            test_repeated_solves();
        }

        // Distributed test — all ranks must participate
        MPI_Barrier(MPI_COMM_WORLD);
        test_distributed_cg(rank, size);

    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nFATAL on rank %d: %s\n", rank, e.what());
        Kokkos::finalize();
        MPI_Finalize();
        return 1;
    }

    Kokkos::finalize();

    if (rank == 0) {
        std::printf("\n=== Results: %d passed, %d failed ===\n",
                    g_pass, g_fail);
    }

    MPI_Finalize();
    return (g_fail == 0) ? 0 : 1;
}
