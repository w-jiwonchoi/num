#pragma once
#include "crs_matrix.hpp"
#include "halo_exchange.hpp"
#include "spmv_kernel.hpp"
#include "vec_ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

// ─── Distributed SpMV ─────────────────────────────────────────────────────────
//
//  Computes y_local = A_local * [x_local | x_ghost]
//  in three steps:
//    1. Halo exchange: fill x_ghost from remote ranks
//    2. Local SpMV with the extended x = [x_local, x_ghost]
//    3. Result already in y_local; no gather needed
//
//  Bug fix: the original distribute_csr extracted local CSR rows but kept
//  GLOBAL column indices in h_colind.  When we then run local SpMV we pass an
//  extended vector x_ext = [x_local | x_ghost] where indices are:
//    0 .. local_nrows-1   → own rows  (was: global_start .. global_end-1)
//    local_nrows .. end   → ghost entries in ghost_map order
//  So all column indices must be remapped from global → local-extended.

/**
 * distribute_csr
 *
 * Partition a global CSR matrix (given as host arrays) across MPI ranks by
 * rows.  Each rank gets a contiguous block of rows.
 *
 * @param h_values   Global CSR values array
 * @param h_colind   Global CSR column indices (GLOBAL)
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

    // ── Even row distribution ─────────────────────────────────────────────
    int base  = global_nrows / size;
    int extra = global_nrows % size;
    int local_row_start = rank * base + std::min(rank,   extra);
    int local_row_end   = local_row_start + base + (rank < extra ? 1 : 0);
    int local_nrows     = local_row_end - local_row_start;

    // ── Extract local submatrix arrays (host side, GLOBAL col indices) ────
    int local_nnz_start = h_rowptr[local_row_start];
    int local_nnz_end   = h_rowptr[local_row_end];
    int local_nnz       = local_nnz_end - local_nnz_start;

    std::vector<Scalar>  lv(h_values.begin() + local_nnz_start,
                            h_values.begin() + local_nnz_end);
    std::vector<Ordinal> lc(h_colind.begin() + local_nnz_start,
                            h_colind.begin() + local_nnz_end);  // still GLOBAL
    std::vector<Offset>  lr(local_nrows + 1);
    for (int i = 0; i <= local_nrows; ++i)
        lr[i] = h_rowptr[local_row_start + i] - local_nnz_start;

    // ── Build ghost map (needs global col indices) ─────────────────────────
    GhostMap gm = build_ghost_map(local_row_start, local_row_end,
                                  lr, lc, comm);

    // ── Remap column indices: global → local-extended ─────────────────────
    //
    //  Bug fix: the original code kept global column indices and built
    //  x_ext = [x_local | x_ghost] of size (local_nrows + n_ghost).
    //  But then passed it to spmv with the local submatrix that still used
    //  GLOBAL column indices, so A(i,j) * x_ext(j) accessed wrong elements.
    //
    //  Correct mapping:
    //    global col c ∈ [local_row_start, local_row_end) → local idx = c - local_row_start
    //    global col c outside own range → local idx = local_nrows + ghost_offset(c)
    //
    // Build ghost global_id → ghost_index map from GhostMap
    std::unordered_map<int, int> ghost_idx_map;
    for (const auto& ri : gm.recv) {
        for (int i = 0; i < (int)ri.global_ids.size(); ++i) {
            ghost_idx_map[ri.global_ids[i]] = ri.offset + i;
        }
    }

    // Remap lc in-place
    for (int j = 0; j < local_nnz; ++j) {
        int gcol = lc[j];
        if (gcol >= local_row_start && gcol < local_row_end) {
            // Own row: simple offset
            lc[j] = gcol - local_row_start;
        } else {
            // Ghost entry
            auto it = ghost_idx_map.find(gcol);
            if (it == ghost_idx_map.end()) {
                throw std::runtime_error(
                    "distribute_csr: ghost column " + std::to_string(gcol) +
                    " not found in ghost map — logic error");
            }
            lc[j] = local_nrows + it->second;
        }
    }

    // ── Upload remapped local matrix to device ─────────────────────────────
    // ncols of the local matrix = local_nrows + n_ghost (extended x dimension)
    int local_ncols = local_nrows + gm.total_ghost;

    DistCrsHandle handle;
    handle.local          = CrsMatrixHandle(lv.data(), lc.data(), lr.data(),
                                            local_nrows, local_ncols);
    handle.ghost_map      = std::move(gm);
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

    // Allocate ghost buffer
    ViewVec1D x_ghost("x_ghost", n_ghost);

    // Step 1: fill ghost buffer via halo exchange
    halo_exchange(x_local, x_ghost, A_dist.ghost_map,
                  A_dist.local_row_start, comm);

    // Step 2: build extended x_ext = [x_local | x_ghost] contiguously
    // Column indices in A_dist.local have already been remapped to this layout.
    ViewVec1D x_ext("x_ext", local_nrows + n_ghost);

    Kokkos::parallel_for("fill_x_ext_local",
        Kokkos::RangePolicy<ExecSpace>(0, local_nrows),
        KOKKOS_LAMBDA(const int i) { x_ext(i) = x_local(i); });

    Kokkos::parallel_for("fill_x_ext_ghost",
        Kokkos::RangePolicy<ExecSpace>(0, n_ghost),
        KOKKOS_LAMBDA(const int i) { x_ext(local_nrows + i) = x_ghost(i); });

    Kokkos::fence("fill_x_ext_fence");

    // Step 3: local SpMV using the remapped matrix + extended x
    ViewVec1DConst x_ext_const = x_ext;
    launch_spmv_kokkoskernels(A_dist.local.mat, x_ext_const, y_local);
}
