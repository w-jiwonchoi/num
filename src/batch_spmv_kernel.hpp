#pragma once
#include "crs_matrix.hpp"

// ─── MDRangePolicy Batch SpMV: Y = A * X ─────────────────────────────────────
//
//  X has shape (N, k)  — k right-hand-side vectors stored column-major in rows.
//  Y has shape (N, k)  — output.
//
//  The 2D index space (row i, rhs b) is parallelised.  Because all k columns
//  of X are accessed for the same sparsity pattern, the non-zero column
//  indices stay hot in cache as k grows, giving super-linear scaling up to
//  the L2 capacity of the GPU.
//
template <class Exec = ExecSpace>
struct BatchSpMVFunctor {
    using CrsMat = KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>;
    using MatView = Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                                 typename Exec::memory_space>;
    using OutView = Kokkos::View<Scalar**,       Kokkos::LayoutRight,
                                 typename Exec::memory_space>;

    CrsMat  A;
    MatView X;  ///< (N, k) input
    OutView Y;  ///< (N, k) output

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int b) const {
        const int row_start = A.graph.row_map(i);
        const int row_end   = A.graph.row_map(i + 1);
        Scalar    acc       = 0.0;
        for (int j = row_start; j < row_end; ++j) {
            acc += A.values(j) * X(A.graph.entries(j), b);
        }
        Y(i, b) = acc;
    }
};

/**
 * launch_batch_spmv
 *
 * @param A    Device-side CRS matrix (N rows)
 * @param X    (N, k) input matrix, LayoutRight (row-major)
 * @param Y    (N, k) output matrix, LayoutRight
 * @param N    number of rows
 * @param k    number of RHS vectors (batch size)
 * @param tile Optional tiling hint {tile_row, tile_col} for MDRangePolicy.
 *             {0,0} → let Kokkos auto-tune.
 */
template <class Exec = ExecSpace>
void launch_batch_spmv(
    const KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>& A,
    const Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                       typename Exec::memory_space>&                      X,
    Kokkos::View<Scalar**, Kokkos::LayoutRight,
                 typename Exec::memory_space>&                             Y,
    int N, int k,
    std::pair<int,int> tile = {0, 0})
{
    using Policy = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>,
                                         Kokkos::Iterate::Right,
                                         Kokkos::Iterate::Right>;

    // Heuristic default tiles: 32 rows × 4 rhs columns (fits one warp nicely)
    int tr = (tile.first  == 0) ? std::min(32, N) : tile.first;
    int tc = (tile.second == 0) ? std::min(4,  k) : tile.second;

    Policy policy({0, 0}, {N, k}, {tr, tc});

    BatchSpMVFunctor<Exec> functor{A, X, Y};
    Kokkos::parallel_for("BatchSpMV_MDRange", policy, functor);
    Kokkos::fence("BatchSpMV_MDRange_fence");
}

/**
 * Convenience: run batch SpMV on host arrays, return flat (N*k) result.
 * For unit testing only — not for production use.
 */
inline std::vector<Scalar> batch_spmv_host_roundtrip(
    const std::vector<Scalar>&  h_values,
    const std::vector<Ordinal>& h_colind,
    const std::vector<Offset>&  h_rowptr,
    const std::vector<Scalar>&  h_X_flat,   // N*k, row-major
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
