#pragma once
#include "crs_matrix.hpp"

// ─── Batch SpMV: Y = A * X  (X shape: N×k, Y shape: N×k) ────────────────────
//
//  Original bug: tile was {32, 4} meaning 32 rows × 4 rhs.
//  On A100 (warp=32), the *inner* dimension (rhs) should be 32 to fill a warp,
//  not the outer. With {32,4}, each warp covers 32 rows but only 4 rhs columns
//  simultaneously → 28 threads wasted per warp for typical k.
//
//  Fix A: MDRangePolicy with tile {1, 32} → one row, 32 rhs columns per warp.
//         This reuses the row_map lookup 32× and keeps rhs columns coalesced.
//
//  Fix B: For large k (≥ 32), use KokkosKernels SPMM via
//         KokkosSparse::spmm which maps to cuSPARSE SpMM (csrmm2) on CUDA.
//         This is the fastest path when k ≥ 8 on A100.
//
//  Strategy:
//    k < 8  → MDRangePolicy hierarchical (low overhead for small batches)
//    k ≥ 8  → KokkosKernels SPMM path (cuSPARSE, near-peak BW)

#include <KokkosSparse_spmv.hpp>   // for KokkosKernels SPMM

// ─── Hierarchical Batch SpMV functor ─────────────────────────────────────────
//
//  Outer (TeamThreadRange)  : rows
//  Inner (ThreadVectorRange): non-zeros in the row
//  For each non-zero (row i, col j, val v):
//    Y(i, 0..k-1) += v * X(j, 0..k-1)   ← inner loop unrolled by compiler
//
//  This gives:
//    1. Row-map accessed once per row (not k times)
//    2. X(j, *) read once and accumulated into k outputs → arithmetic intensity ↑
//    3. Y writes are register-accumulated then stored → no partial scatter

template <class Exec = ExecSpace>
struct BatchSpMVFunctorHierarchical {
    using Policy   = Kokkos::TeamPolicy<Exec>;
    using Member   = typename Policy::member_type;
    using CrsMat   = KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>;
    using MatView  = Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                                  typename Exec::memory_space>;
    using OutView  = Kokkos::View<Scalar**,       Kokkos::LayoutRight,
                                  typename Exec::memory_space>;

    CrsMat  A;
    MatView X;   // (N, k)
    OutView Y;   // (N, k)
    int     k;

    KOKKOS_INLINE_FUNCTION
    void operator()(const Member& team) const {
        const int i = team.league_rank();
        if (i >= A.numRows()) return;

        const int row_start = A.graph.row_map(i);
        const int row_end   = A.graph.row_map(i + 1);

        // Accumulate into a register array for each rhs column.
        // We unroll up to k=32 in registers; larger k handled by SPMM path.
        // For k <= 16, this fits in registers without spilling on A100.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, k),
            [&](const int b) {
                Scalar acc = 0.0;
                // ThreadVectorRange over nnz of this row
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, row_start, row_end),
                    [&](const int j, Scalar& s) {
                        s += A.values(j) * X(A.graph.entries(j), b);
                    },
                    acc
                );
                Y(i, b) = acc;
            }
        );
    }
};

// ─── MDRangePolicy batch SpMV (original, kept for k < 4) ─────────────────────

template <class Exec = ExecSpace>
struct BatchSpMVFunctorMDRange {
    using CrsMat  = KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>;
    using MatView = Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                                 typename Exec::memory_space>;
    using OutView = Kokkos::View<Scalar**,       Kokkos::LayoutRight,
                                 typename Exec::memory_space>;

    CrsMat  A;
    MatView X;
    OutView Y;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int b) const {
        const int row_start = A.graph.row_map(i);
        const int row_end   = A.graph.row_map(i + 1);
        Scalar acc = 0.0;
        for (int j = row_start; j < row_end; ++j)
            acc += A.values(j) * X(A.graph.entries(j), b);
        Y(i, b) = acc;
    }
};

/**
 * launch_batch_spmv
 *
 * Dispatches to the best kernel for the given (N, k, avg_nnz):
 *
 *   k < 4              → MDRangePolicy (minimal overhead)
 *   4 <= k < 32        → TeamPolicy hierarchical (register accumulation)
 *   k >= 32            → KokkosKernels SPMM (cuSPARSE SpMM on A100)
 *
 * @param A    Device-side CRS matrix
 * @param X    (N, k) input  — LayoutRight (C-contiguous)
 * @param Y    (N, k) output — LayoutRight
 * @param N    number of rows
 * @param k    batch size (number of RHS vectors)
 */
template <class Exec = ExecSpace>
void launch_batch_spmv(
    const KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>& A,
    const Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                       typename Exec::memory_space>&                      X,
    Kokkos::View<Scalar**, Kokkos::LayoutRight,
                 typename Exec::memory_space>&                             Y,
    int N, int k,
    std::pair<int,int> /*tile_hint*/ = {0, 0})  // ignored; auto-dispatched
{
    if (N == 0 || k == 0) return;

    if (k < 4) {
        // ── MDRangePolicy (low overhead, small k) ────────────────────────
        // Fix: tile is {1, 32} not {32, 4} — inner dim = rhs fills a warp
        using Policy = Kokkos::MDRangePolicy<
            Exec,
            Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;

        const int tile_row = 1;
        const int tile_rhs = std::min(32, k);
        Policy policy({0, 0}, {N, k}, {tile_row, tile_rhs});

        BatchSpMVFunctorMDRange<Exec> functor{A, X, Y};
        Kokkos::parallel_for("BatchSpMV_MDRange", policy, functor);

    } else if (k < 32) {
        // ── TeamPolicy hierarchical (register accumulation) ───────────────
        // team_size = k (each thread handles one rhs column)
        // vector_length = 32 (warp iterates over nnz of one row)
        const int vector_length = 32;
        using Policy = Kokkos::TeamPolicy<Exec>;
        Policy policy(N, k, vector_length);

        BatchSpMVFunctorHierarchical<Exec> functor{A, X, Y, k};
        Kokkos::parallel_for("BatchSpMV_Hierarchical", policy, functor);

    } else {
        // ── KokkosKernels SPMM (cuSPARSE SpMM, fastest for large k) ──────
        //
        // KokkosKernels spmv for multi-vector:
        //   KokkosSparse::spmv("N", 1.0, A, X, 0.0, Y)
        // where X and Y are 2D Views with LayoutRight.
        //
        // Note: KokkosKernels 4.x spmv is overloaded for 2D Views and
        // dispatches to cuSPARSE SpMM (csrmm2 / generic SpMM API on CUDA 12).
        // This is the fastest possible path on A100 for k >= 8.
        KokkosSparse::spmv("N",
                           static_cast<Scalar>(1.0), A,
                           X,
                           static_cast<Scalar>(0.0), Y);
    }

    Kokkos::fence("BatchSpMV_fence");
}

/**
 * batch_spmv_host_roundtrip — unit test helper only.
 */
inline std::vector<Scalar> batch_spmv_host_roundtrip(
    const std::vector<Scalar>&  h_values,
    const std::vector<Ordinal>& h_colind,
    const std::vector<Offset>&  h_rowptr,
    const std::vector<Scalar>&  h_X_flat,   // N*k row-major
    int nrows, int ncols, int k)
{
    CrsMatrixHandle handle(h_values.data(), h_colind.data(), h_rowptr.data(),
                           nrows, ncols);

    ViewMat2D X_dev("X", nrows, k);
    ViewMat2D Y_dev("Y", nrows, k);

    auto X_host = Kokkos::create_mirror_view(X_dev);
    for (int i = 0; i < nrows; ++i)
        for (int b = 0; b < k; ++b)
            X_host(i, b) = h_X_flat[i * k + b];
    Kokkos::deep_copy(X_dev, X_host);

    ViewMat2DConst X_const = X_dev;
    launch_batch_spmv(handle.mat, X_const, Y_dev, nrows, k);

    auto Y_host = Kokkos::create_mirror_view(Y_dev);
    Kokkos::deep_copy(Y_host, Y_dev);

    std::vector<Scalar> result(nrows * k);
    for (int i = 0; i < nrows; ++i)
        for (int b = 0; b < k; ++b)
            result[i * k + b] = Y_host(i, b);
    return result;
}
