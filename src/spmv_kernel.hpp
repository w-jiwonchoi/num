#pragma once
#include "crs_matrix.hpp"
#include <KokkosSparse_spmv.hpp>

// ─── Custom TeamPolicy SpMV ───────────────────────────────────────────────────
//
//  Each Kokkos "team" maps to one matrix row.
//  Threads within the team collaborate on the non-zero elements of that row
//  using a parallel reduction over TeamVectorRange, then one thread writes y[i].
//
template <class Exec = ExecSpace>
struct SpMVFunctor {
    using Policy     = Kokkos::TeamPolicy<Exec>;
    using Member     = typename Policy::member_type;
    using CrsMat     = KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>;
    using VecConst   = Kokkos::View<const Scalar*, typename Exec::memory_space>;
    using VecOut     = Kokkos::View<Scalar*,       typename Exec::memory_space>;

    CrsMat   A;
    VecConst x;
    VecOut   y;

    KOKKOS_INLINE_FUNCTION
    void operator()(const Member& team) const {
        const int i         = team.league_rank();
        const int row_start = A.graph.row_map(i);
        const int row_end   = A.graph.row_map(i + 1);

        Scalar partial = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(team, row_start, row_end),
            [&](const int j, Scalar& acc) {
                acc += A.values(j) * x(A.graph.entries(j));
            },
            partial
        );
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { y(i) = partial; });
    }
};

/**
 * launch_spmv_custom
 *
 * Launches the custom TeamPolicy kernel.
 * team_size = 0 → Kokkos auto-selects optimal warp/wavefront size.
 */
template <class Exec = ExecSpace>
void launch_spmv_custom(
    const KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>& A,
    const Kokkos::View<const Scalar*, typename Exec::memory_space>&     x,
    Kokkos::View<Scalar*, typename Exec::memory_space>&                  y,
    int team_size = 0)
{
    using Policy = Kokkos::TeamPolicy<Exec>;
    int nrows = A.numRows();

    Policy policy(nrows, Kokkos::AUTO);

    SpMVFunctor<Exec> functor{A, x, y};
    Kokkos::parallel_for("SpMV_TeamPolicy", policy, functor);
    Kokkos::fence("SpMV_TeamPolicy_fence");
}

/**
 * launch_spmv_kokkoskernels
 *
 * Delegates to KokkosKernels, which internally uses cuSPARSE on CUDA
 * and rocSPARSE on HIP — the platform-optimal baseline.
 *
 *   y = alpha * A * x + beta * y
 */
template <class Exec = ExecSpace>
void launch_spmv_kokkoskernels(
    const KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>& A,
    const Kokkos::View<const Scalar*, typename Exec::memory_space>&     x,
    Kokkos::View<Scalar*, typename Exec::memory_space>&                  y,
    Scalar alpha = 1.0,
    Scalar beta  = 0.0)
{
    KokkosSparse::spmv("N", alpha, A, x, beta, y);
    Kokkos::fence("SpMV_KokkosKernels_fence");
}

/**
 * Convenience wrapper: upload host data → run SpMV → copy result back.
 * Primarily used in unit tests.
 */
inline std::vector<Scalar> spmv_host_roundtrip(
    const std::vector<Scalar>&  h_values,
    const std::vector<Ordinal>& h_colind,
    const std::vector<Offset>&  h_rowptr,
    const std::vector<Scalar>&  h_x,
    int nrows, int ncols,
    bool use_kokkoskernels = false)
{
    CrsMatrixHandle handle(h_values.data(), h_colind.data(), h_rowptr.data(),
                           nrows, ncols);

    ViewVec1D      x_dev("x", h_x.size());
    ViewVec1D      y_dev("y", nrows);

    auto x_host = Kokkos::create_mirror_view(x_dev);
    for (size_t i = 0; i < h_x.size(); ++i) x_host(i) = h_x[i];
    Kokkos::deep_copy(x_dev, x_host);

    ViewVec1DConst x_const = x_dev;

    if (use_kokkoskernels)
        launch_spmv_kokkoskernels(handle.mat, x_const, y_dev);
    else
        launch_spmv_custom(handle.mat, x_const, y_dev);

    auto y_host = Kokkos::create_mirror_view(y_dev);
    Kokkos::deep_copy(y_host, y_dev);

    std::vector<Scalar> result(nrows);
    for (int i = 0; i < nrows; ++i) result[i] = y_host(i);
    return result;
}
