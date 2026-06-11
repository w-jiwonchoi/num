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
//  Performance improvements over original:
//
//  1. pack_info cached at distribute_csr time → no setup work per SpMV call
//  2. halo_exchange uses device-side packing (GPU-direct if CUDA_AWARE_MPI=1)
//  3. x_ext construction uses Kokkos::subview instead of two separate kernels
//  4. fence only after halo (not between pack/send steps)

/**
 * distribute_csr
 *
 * Partition a global CSR matrix across MPI ranks by rows.
 * Each rank gets a contiguous block of rows.
 * Column indices are remapped to local-extended space [own | ghost].
 * pack_info is built once and cached in the returned handle.
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

    // ── Extract local submatrix (host side, GLOBAL col indices) ───────────
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

    // ── Build ghost map ────────────────────────────────────────────────────
    GhostMap gm = build_ghost_map(local_row_start, local_row_end,
                                  lr, lc, comm);

    // ── Remap column indices: global → local-extended ─────────────────────
    std::unordered_map<int, int> ghost_idx_map;
    ghost_idx_map.reserve(gm.total_ghost * 2);
    for (const auto& ri : gm.recv) {
        for (int i = 0; i < (int)ri.global_ids.size(); ++i)
            ghost_idx_map[ri.global_ids[i]] = ri.offset + i;
    }

    for (int j = 0; j < local_nnz; ++j) {
        int gcol = lc[j];
        if (gcol >= local_row_start && gcol < local_row_end) {
            lc[j] = gcol - local_row_start;
        } else {
            auto it = ghost_idx_map.find(gcol);
            if (it == ghost_idx_map.end())
                throw std::runtime_error(
                    "distribute_csr: ghost column " + std::to_string(gcol) +
                    " not found in ghost map");
            lc[j] = local_nrows + it->second;
        }
    }

    // ── Upload remapped local matrix to device ─────────────────────────────
    int local_ncols = local_nrows + gm.total_ghost;

    DistCrsHandle handle;
    handle.local           = CrsMatrixHandle(lv.data(), lc.data(), lr.data(),
                                             local_nrows, local_ncols);
    handle.ghost_map       = std::move(gm);
    handle.global_nrows    = global_nrows;
    handle.local_row_start = local_row_start;
    handle.local_row_end   = local_row_end;

    // ── Cache device-side pack metadata (built once, reused every SpMV) ───
    handle.pack_info = std::make_shared<PackedSendInfo>(
        build_send_indices(handle.ghost_map, local_row_start));

    return handle;
}

/**
 * distributed_spmv
 *
 * y_local = A_dist * x_global_local_slice
 *
 * Steps:
 *   1. GPU-pack + halo exchange (device-to-device if CUDA_AWARE_MPI)
 *   2. Build x_ext = [x_local | x_ghost] as a single contiguous view
 *      using Kokkos::subview + parallel_for (avoids allocation each call
 *      when pre-allocated buffer is passed)
 *   3. Local KokkosKernels SpMV with x_ext
 *
 * Pre-allocated x_ext_buf:
 *   Caller may pass a pre-allocated ViewVec1D of size (local_nrows+n_ghost).
 *   Passing an empty View causes allocation each call (convenient, slower).
 */
void distributed_spmv(
    const DistCrsHandle& A_dist,
    const ViewVec1D&     x_local,
    ViewVec1D&           y_local,
    MPI_Comm             comm = MPI_COMM_WORLD,
    ViewVec1D*           x_ext_buf = nullptr)   // optional pre-allocated buffer
{
    const int local_nrows = A_dist.local_row_end - A_dist.local_row_start;
    const int n_ghost     = A_dist.ghost_map.total_ghost;

    // ── x_ghost and x_ext ─────────────────────────────────────────────────
    // Allocate x_ext once (or reuse caller-supplied buffer)
    ViewVec1D x_ext_owned;
    ViewVec1D& x_ext = (x_ext_buf != nullptr && (int)x_ext_buf->extent(0) == local_nrows + n_ghost)
                       ? *x_ext_buf
                       : x_ext_owned;

    if ((int)x_ext.extent(0) != local_nrows + n_ghost)
        x_ext = ViewVec1D("x_ext", local_nrows + n_ghost);

    // Copy local portion into x_ext[0..local_nrows)
    Kokkos::parallel_for("x_ext_fill_local",
        Kokkos::RangePolicy<ExecSpace>(0, local_nrows),
        KOKKOS_LAMBDA(const int i) { x_ext(i) = x_local(i); }
    );

    // Ghost subview — this is where halo_exchange writes
    // Subview is a non-owning view into the tail of x_ext
    ViewVec1D x_ghost(x_ext.data() + local_nrows, n_ghost);

    // ── Halo exchange (packs x_local → sends → recvs into x_ghost) ────────
    if (A_dist.pack_info) {
        halo_exchange(x_local, x_ghost, A_dist.ghost_map,
                      A_dist.local_row_start, comm, *A_dist.pack_info);
    } else {
        halo_exchange(x_local, x_ghost, A_dist.ghost_map,
                      A_dist.local_row_start, comm);
    }

    // ── Local SpMV with extended x ─────────────────────────────────────────
    // x_ghost was written by halo_exchange; x_ext[local_nrows..] is now valid
    Kokkos::fence("x_ext_ready_fence");

    ViewVec1DConst x_ext_const = x_ext;
    launch_spmv_kokkoskernels(A_dist.local.mat, x_ext_const, y_local);
}
