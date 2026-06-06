#pragma once
#include "crs_matrix.hpp"
#include "halo_exchange.hpp"
#include "spmv_kernel.hpp"
#include "vec_ops.hpp"
#include <stdexcept>

// ─── Distributed SpMV ─────────────────────────────────────────────────────────
//
//  Computes y_local = A_local * [x_local | x_ghost]
//  in three steps:
//    1. Halo exchange: fill x_ghost from remote ranks
//    2. Local SpMV with the extended x = [x_local, x_ghost]
//    3. Result already in y_local; no gather needed
//

/**
 * distribute_csr
 *
 * Partition a global CSR matrix (given as host arrays) across MPI ranks by
 * rows.  Each rank gets a contiguous block of rows.
 *
 * @param h_values   Global CSR values array
 * @param h_colind   Global CSR column indices
 * @param h_rowptr   Global CSR row pointer (length: global_nrows + 1)
 * @param global_nrows
 * @param global_ncols
 * @param comm       MPI communicator
 * @return           DistCrsHandle for this rank
 */
DistCrsHandle distribute_csr(
    const std::vector<Scalar>&  h_values,
    const std::vector<Ordinal>& h_colind,
    const std::vector<Offset>&  h_rowptr,
    int global_nrows, int global_ncols,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Even row distribution
    int base  = global_nrows / size;
    int extra = global_nrows % size;
    int local_row_start = rank * base + std::min(rank,   extra);
    int local_row_end   = local_row_start + base + (rank < extra ? 1 : 0);
    int local_nrows     = local_row_end - local_row_start;

    // Extract local submatrix arrays (host side)
    int local_nnz_start = h_rowptr[local_row_start];
    int local_nnz_end   = h_rowptr[local_row_end];
    int local_nnz       = local_nnz_end - local_nnz_start;

    std::vector<Scalar>  lv(h_values.begin() + local_nnz_start,
                            h_values.begin() + local_nnz_end);
    std::vector<Ordinal> lc(h_colind.begin() + local_nnz_start,
                            h_colind.begin() + local_nnz_end);
    std::vector<Offset>  lr(local_nrows + 1);
    for (int i = 0; i <= local_nrows; ++i)
        lr[i] = h_rowptr[local_row_start + i] - local_nnz_start;

    DistCrsHandle handle;
    handle.local          = CrsMatrixHandle(lv.data(), lc.data(), lr.data(),
                                            local_nrows, global_ncols);
    handle.ghost_map      = build_ghost_map(local_row_start, local_row_end,
                                            lr, lc, comm);
    handle.global_nrows   = global_nrows;
    handle.local_row_start= local_row_start;
    handle.local_row_end  = local_row_end;

    return handle;
}

/**
 * distributed_spmv
 *
 * y_local = A_dist * x_local   (in-place into y_local)
 *
 * Caller must ensure:
 *   x_local.extent(0) == local_nrows
 *   y_local.extent(0) == local_nrows
 */
void distributed_spmv(
    const DistCrsHandle& A_dist,
    const ViewVec1D&     x_local,
    ViewVec1D&           y_local,
    MPI_Comm             comm = MPI_COMM_WORLD)
{
    int local_nrows = A_dist.local_row_end - A_dist.local_row_start;
    int n_ghost     = A_dist.ghost_map.total_ghost;

    // Allocate ghost buffer (could be cached in DistCrsHandle for production)
    ViewVec1D x_ghost("x_ghost", n_ghost);

    // Step 1: fill ghost buffer
    halo_exchange(x_local, x_ghost, A_dist.ghost_map,
                  A_dist.local_row_start, comm);

    // Step 2: build extended x view = [x_local | x_ghost]
    // For a clean implementation we copy into a single contiguous buffer.
    // A production version would use a custom gather kernel instead.
    ViewVec1D x_ext("x_ext", local_nrows + n_ghost);
    Kokkos::parallel_for("fill_x_ext_local",
        Kokkos::RangePolicy<ExecSpace>(0, local_nrows),
        KOKKOS_LAMBDA(const int i) { x_ext(i) = x_local(i); });

    Kokkos::parallel_for("fill_x_ext_ghost",
        Kokkos::RangePolicy<ExecSpace>(0, n_ghost),
        KOKKOS_LAMBDA(const int i) { x_ext(local_nrows + i) = x_ghost(i); });

    Kokkos::fence("fill_x_ext_fence");

    // Step 3: local SpMV with extended x
    ViewVec1DConst x_ext_const = x_ext;
    launch_spmv_kokkoskernels(A_dist.local.mat, x_ext_const, y_local);
}
