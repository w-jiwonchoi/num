#pragma once
#include "crs_matrix.hpp"
#include "halo_exchange.hpp"
#include "spmv_kernel.hpp"
#include "vec_ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

// ─── Distributed SpMV (v0.2) ──────────────────────────────────────────────────
//
//  Performance fixes vs v0.1:
//   1. x_ext allocated ONCE per matrix (cached in handle) — was per-call
//   2. Interior/boundary row split → interior SpMV overlaps with halo comm
//   3. Host-staging fallback is now correct (no device ptr into MPI) and
//      buffer-persistent; CUDA-aware path used when compiled in
//   4. Single fence at the end instead of 4 fences per call

DistCrsHandle distribute_csr(
    const std::vector<Scalar>&  h_values,
    const std::vector<Ordinal>& h_colind,
    const std::vector<Offset>&  h_rowptr,
    int global_nrows, int /*global_ncols*/,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // ── Even row distribution ─────────────────────────────────────────────
    int base  = global_nrows / size;
    int extra = global_nrows % size;
    int local_row_start = rank * base + std::min(rank, extra);
    int local_row_end   = local_row_start + base + (rank < extra ? 1 : 0);
    int local_nrows     = local_row_end - local_row_start;

    // ── Extract local submatrix (GLOBAL col indices) ──────────────────────
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

    // ── Ghost map ──────────────────────────────────────────────────────────
    GhostMap gm = build_ghost_map(local_row_start, local_row_end, lr, lc, comm);

    // ── Remap columns: global → [own | ghost] ──────────────────────────────
    std::unordered_map<int, int> ghost_idx_map;
    ghost_idx_map.reserve(gm.total_ghost * 2);
    for (const auto& ri : gm.recv)
        for (int i = 0; i < (int)ri.global_ids.size(); ++i)
            ghost_idx_map[ri.global_ids[i]] = ri.offset + i;

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

    int local_ncols = local_nrows + gm.total_ghost;

    DistCrsHandle handle;
    handle.local           = CrsMatrixHandle(lv.data(), lc.data(), lr.data(),
                                             local_nrows, local_ncols);
    handle.global_nrows    = global_nrows;
    handle.local_row_start = local_row_start;
    handle.local_row_end   = local_row_end;

    // ── Interior / boundary split (for comm-compute overlap) ──────────────
    //  interior row : all column indices < local_nrows (no ghost reads)
    //  boundary row : at least one ghost column
    //  Both matrices keep the full (local_nrows × local_ncols) shape so
    //  y = 1*A_int*x + 0*y  followed by  y = 1*A_bnd*x + 1*y  composes
    //  exactly to the full local SpMV.
    {
        std::vector<Scalar>  iv, bv;
        std::vector<Ordinal> ic, bc;
        std::vector<Offset>  ir(local_nrows + 1, 0), br(local_nrows + 1, 0);
        iv.reserve(local_nnz);  ic.reserve(local_nnz);

        int n_bnd_rows = 0;
        for (int i = 0; i < local_nrows; ++i) {
            bool bnd = false;
            for (int j = lr[i]; j < lr[i + 1]; ++j)
                if (lc[j] >= local_nrows) { bnd = true; break; }
            if (bnd) {
                ++n_bnd_rows;
                for (int j = lr[i]; j < lr[i + 1]; ++j) {
                    bc.push_back(lc[j]); bv.push_back(lv[j]);
                }
            } else {
                for (int j = lr[i]; j < lr[i + 1]; ++j) {
                    ic.push_back(lc[j]); iv.push_back(lv[j]);
                }
            }
            ir[i + 1] = static_cast<Offset>(iv.size());
            br[i + 1] = static_cast<Offset>(bv.size());
        }

        // Split only useful if both parts are non-trivial
        if (size > 1 && n_bnd_rows > 0 && n_bnd_rows < local_nrows) {
            handle.interior = CrsMatrixHandle(
                iv.empty() ? nullptr : iv.data(),
                ic.empty() ? nullptr : ic.data(),
                ir.data(), local_nrows, local_ncols);
            handle.boundary = CrsMatrixHandle(
                bv.empty() ? nullptr : bv.data(),
                bc.empty() ? nullptr : bc.data(),
                br.data(), local_nrows, local_ncols);
            handle.has_split = true;
        }
    }

    handle.pack_info = std::make_shared<PackedSendInfo>(
        build_send_indices(gm, local_row_start));
    handle.ghost_map = std::move(gm);

    // Pre-allocate the extended vector buffer once
    handle.x_ext_cache = std::make_shared<ViewVec1D>(
        "x_ext_cache", local_ncols);

    return handle;
}

/**
 * distributed_spmv  (v0.2)
 *
 * @param use_kokkoskernels  local kernel: KokkosKernels/cuSPARSE (true)
 *                           or custom hierarchical TeamPolicy (false)
 * @param overlap            overlap interior SpMV with halo communication
 *                           (requires has_split; falls back otherwise)
 */
inline void distributed_spmv(
    const DistCrsHandle& A_dist,
    const ViewVec1D&     x_local,
    ViewVec1D&           y_local,
    MPI_Comm             comm              = MPI_COMM_WORLD,
    bool                 use_kokkoskernels = true,
    bool                 overlap           = true)
{
    const int local_nrows = A_dist.local_row_end - A_dist.local_row_start;
    const int n_ghost     = A_dist.ghost_map.total_ghost;

    // ── Cached x_ext (allocated once in distribute_csr) ───────────────────
    ViewVec1D x_ext;
    if (A_dist.x_ext_cache &&
        (int)A_dist.x_ext_cache->extent(0) == local_nrows + n_ghost) {
        x_ext = *A_dist.x_ext_cache;
    } else {
        x_ext = ViewVec1D("x_ext", local_nrows + n_ghost);
    }

    // Copy local portion into x_ext[0 .. local_nrows)
    Kokkos::parallel_for("x_ext_fill_local",
        Kokkos::RangePolicy<ExecSpace>(0, local_nrows),
        KOKKOS_LAMBDA(const int i) { x_ext(i) = x_local(i); });

    ViewVec1D x_ghost(x_ext.data() + local_nrows, n_ghost);
    ViewVec1DConst x_ext_const = x_ext;

    const bool do_overlap = overlap && A_dist.has_split && n_ghost > 0
                            && A_dist.pack_info;

    if (do_overlap) {
        // ── Overlapped path ────────────────────────────────────────────────
        std::vector<MPI_Request> reqs;
        reqs.reserve(A_dist.ghost_map.recv.size()
                   + A_dist.ghost_map.send.size());

        // 1. pack + post Isend/Irecv (pack fences internally)
        halo_exchange_begin(x_local, A_dist.ghost_map, comm,
                            *A_dist.pack_info, reqs);

        // 2. interior SpMV — reads only x_ext[0..local_nrows), runs on GPU
        //    while MPI moves halo data.  No fence here on purpose.
        KokkosSparse::spmv("N", static_cast<Scalar>(1.0),
                           A_dist.interior.mat, x_ext_const,
                           static_cast<Scalar>(0.0), y_local);

        // 3. complete halo, scatter ghosts into x_ext tail
        halo_exchange_end(x_ghost, A_dist.ghost_map,
                          *A_dist.pack_info, reqs);

        // 4. boundary SpMV accumulates on top (beta = 1)
        KokkosSparse::spmv("N", static_cast<Scalar>(1.0),
                           A_dist.boundary.mat, x_ext_const,
                           static_cast<Scalar>(1.0), y_local);
        Kokkos::fence("dist_spmv_overlap_done");

    } else {
        // ── Blocking path (also used when np == 1 or no split) ────────────
        if (n_ghost > 0 && A_dist.pack_info)
            halo_exchange(x_local, x_ghost, A_dist.ghost_map,
                          A_dist.local_row_start, comm, *A_dist.pack_info);

        if (use_kokkoskernels)
            launch_spmv_kokkoskernels(A_dist.local.mat, x_ext_const, y_local);
        else
            launch_spmv_custom(A_dist.local.mat, x_ext_const, y_local);
    }
}
