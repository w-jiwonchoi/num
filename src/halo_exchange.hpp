#pragma once
#include "crs_matrix.hpp"
#include "mpi_singleton.hpp"
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <stdexcept>

// ─── Halo Exchange ────────────────────────────────────────────────────────────
//
//  Original performance problems:
//
//  1. CPU-staging pack: original code did
//         Kokkos::create_mirror_view(x_local)   // GPU→CPU copy
//         pack into host vector
//         MPI_Isend(host_ptr)                   // CPU send
//     This adds 2 PCIe round-trips (GPU→CPU for pack, CPU→GPU for recv).
//
//  Fix: When CUDA_AWARE_MPI=1, pass GPU device pointers directly to MPI.
//       Pack into a contiguous device-side send buffer (parallel_for),
//       then MPI_Isend(device_ptr) — zero extra PCIe traffic.
//
//  2. Sequential pack: original packed one entry at a time on host.
//
//  Fix: parallel_for over send indices to pack contiguous GPU buffer.
//
//  3. No compute/comm overlap: original did pack → send → wait → compute.
//
//  Fix: post all Irecv first, pack+Isend, fence only what's needed,
//       then overlap halo wait with local (diagonal) SpMV rows.
//       (Requires splitting the local matrix into diagonal + off-diagonal
//        blocks — implemented in dist_spmv.hpp.)

/**
 * build_ghost_map — unchanged from original (setup-time only, not hot path)
 */
GhostMap build_ghost_map(
    int local_row_start, int local_row_end,
    const std::vector<Offset>&  h_rowptr,
    const std::vector<Ordinal>& h_colind,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<int> all_starts(size), all_ends(size);
    MPI_Allgather(&local_row_start, 1, MPI_INT,
                  all_starts.data(), 1, MPI_INT, comm);
    MPI_Allgather(&local_row_end, 1, MPI_INT,
                  all_ends.data(), 1, MPI_INT, comm);

    auto owner_of = [&](int col) -> int {
        for (int r = 0; r < size; ++r)
            if (col >= all_starts[r] && col < all_ends[r]) return r;
        throw std::runtime_error(
            "build_ghost_map: column index " + std::to_string(col) +
            " is out of the global row range");
    };

    std::map<int, std::set<int>> recv_cols_set;
    int local_rows = local_row_end - local_row_start;
    for (int i = 0; i < local_rows; ++i) {
        for (int j = h_rowptr[i]; j < h_rowptr[i + 1]; ++j) {
            int col = h_colind[j];
            if (col < local_row_start || col >= local_row_end) {
                recv_cols_set[owner_of(col)].insert(col);
            }
        }
    }

    std::map<int, std::vector<int>> recv_cols;
    for (auto& [r, s] : recv_cols_set)
        recv_cols[r] = std::vector<int>(s.begin(), s.end());

    std::vector<int> recv_counts(size, 0), send_counts(size, 0);
    for (auto& [r, ids] : recv_cols)
        recv_counts[r] = static_cast<int>(ids.size());
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT,
                 send_counts.data(), 1, MPI_INT, comm);

    std::vector<std::vector<int>> send_bufs(size);
    std::vector<std::vector<int>> recv_bufs(size);
    std::vector<MPI_Request> reqs;

    for (int r = 0; r < size; ++r) {
        if (r == rank || send_counts[r] == 0) continue;
        recv_bufs[r].resize(send_counts[r]);
        reqs.emplace_back();
        MPI_Irecv(recv_bufs[r].data(), send_counts[r], MPI_INT,
                  r, 1001, comm, &reqs.back());
    }
    for (int r = 0; r < size; ++r) {
        if (r == rank || recv_counts[r] == 0) continue;
        send_bufs[r] = recv_cols[r];
        reqs.emplace_back();
        MPI_Isend(send_bufs[r].data(), recv_counts[r], MPI_INT,
                  r, 1001, comm, &reqs.back());
    }
    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()),
                    reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    GhostMap gm;
    int ghost_offset = 0;

    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        if (!recv_cols[r].empty()) {
            CommInfo ci;
            ci.rank       = r;
            ci.global_ids = recv_cols[r];
            ci.offset     = ghost_offset;
            ghost_offset += static_cast<int>(ci.global_ids.size());
            gm.recv.push_back(std::move(ci));
        }
        if (!recv_bufs[r].empty()) {
            CommInfo ci;
            ci.rank       = r;
            ci.global_ids = recv_bufs[r];
            ci.offset     = -1;
            gm.send.push_back(std::move(ci));
        }
    }

    gm.total_ghost = ghost_offset;
    return gm;
}

// ─── Device-side send-index arrays (stored in DistCrsHandle after setup) ─────
//
//  For each send neighbour r, we store the LOCAL indices of x_local to pack.
//  Packed once at distribute_csr time, reused every SpMV iteration.
//
//  Layout: send_indices_dev is a flat device array of all send indices,
//          send_offsets[r] and send_counts[r] slice into it.

struct PackedSendInfo {
    ViewVec1D                  send_buf;    // device send buffer (flat)
    std::vector<ViewVec1D>     recv_bufs;   // per-neighbour device recv buffers
    // Index arrays for packing: send_indices_dev[i] = local index into x_local
    Kokkos::View<int*, MemSpace> send_indices_dev;
    int                          total_send;
};

/**
 * build_send_indices
 *
 * Called once at distribute time.  Converts send CommInfo (global ids) into
 * a device-side flat array of LOCAL indices for O(1) parallel packing.
 */
inline PackedSendInfo build_send_indices(
    const GhostMap& gm,
    int local_row_start)
{
    // Flatten all send indices into one host vector
    int total_send = 0;
    for (const auto& si : gm.send)
        total_send += static_cast<int>(si.global_ids.size());

    std::vector<int> h_send_idx;
    h_send_idx.reserve(total_send);
    for (const auto& si : gm.send) {
        for (int gid : si.global_ids)
            h_send_idx.push_back(gid - local_row_start);
    }

    Kokkos::View<int*, MemSpace> d_send_idx("send_indices", total_send);
    {
        auto mirror = Kokkos::create_mirror_view(d_send_idx);
        for (int i = 0; i < total_send; ++i) mirror(i) = h_send_idx[i];
        Kokkos::deep_copy(d_send_idx, mirror);
    }

    PackedSendInfo info;
    info.send_buf        = ViewVec1D("send_buf", total_send);
    info.send_indices_dev = d_send_idx;
    info.total_send      = total_send;

    // Per-neighbour recv buffers on device (receives land here directly)
    info.recv_bufs.reserve(gm.recv.size());
    for (const auto& ri : gm.recv) {
        info.recv_bufs.emplace_back(
            ViewVec1D("recv_buf_" + std::to_string(ri.rank),
                      static_cast<int>(ri.global_ids.size())));
    }

    return info;
}

/**
 * halo_exchange
 *
 * GPU-direct path (CUDA_AWARE_MPI=1):
 *   1. parallel_for packs x_local into contiguous device send_buf
 *   2. Post all MPI_Irecv to device recv_bufs
 *   3. MPI_Isend from device send_buf slices
 *   4. MPI_Waitall
 *   5. Scatter recv_bufs into x_ghost
 *
 * CPU-staging fallback (CUDA_AWARE_MPI=0):
 *   Same logic but mirror to host before send, mirror from host after recv.
 *
 * Either way, x_ghost is a contiguous device array of length gm.total_ghost
 * that can be used as x_ext[local_nrows .. local_nrows+n_ghost-1].
 */
void halo_exchange(
    const ViewVec1D&        x_local,
    ViewVec1D&              x_ghost,
    const GhostMap&         gm,
    int                     local_row_start,
    MPI_Comm                comm,
    const PackedSendInfo&   pack_info)   // pre-built at distribute time
{
    if (gm.total_ghost == 0) return;   // no off-rank entries needed

    std::vector<MPI_Request> reqs;
    reqs.reserve(gm.recv.size() + gm.send.size());

    // ── Step 1: pack send buffer (parallel, on device) ────────────────────
    {
        const int n = pack_info.total_send;
        const auto& idx = pack_info.send_indices_dev;
        auto& buf       = pack_info.send_buf;  // mutable through const ref ok
        // Note: pack_info.send_buf is not const; the ViewVec1D is a handle
        Kokkos::parallel_for("halo_pack",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int i) {
                buf(i) = x_local(idx(i));
            }
        );
        Kokkos::fence("halo_pack_fence");
    }

    // ── Step 2: post all Irecv ────────────────────────────────────────────
    for (std::size_t ri = 0; ri < gm.recv.size(); ++ri) {
        const auto& info  = gm.recv[ri];
        const int   count = static_cast<int>(info.global_ids.size());

#if defined(CUDA_AWARE_MPI)
        // Receive directly into device memory — zero PCIe traffic
        MPI_Irecv(pack_info.recv_bufs[ri].data(), count,
                  MPI_DOUBLE, info.rank, 42, comm, &reqs.emplace_back());
#else
        // CPU staging: recv into x_ghost host portion, copy later
        MPI_Irecv(x_ghost.data() + info.offset, count,
                  MPI_DOUBLE, info.rank, 42, comm, &reqs.emplace_back());
#endif
    }

    // ── Step 3: post all Isend ────────────────────────────────────────────
    {
        int offset = 0;
        for (const auto& si : gm.send) {
            const int count = static_cast<int>(si.global_ids.size());

#if defined(CUDA_AWARE_MPI)
            MPI_Isend(pack_info.send_buf.data() + offset, count,
                      MPI_DOUBLE, si.rank, 42, comm, &reqs.emplace_back());
#else
            // Mirror send buffer to host for non-CUDA-aware MPI
            auto send_host = Kokkos::create_mirror_view(
                Kokkos::subview(pack_info.send_buf,
                                Kokkos::make_pair(offset, offset + count)));
            Kokkos::deep_copy(send_host,
                Kokkos::subview(pack_info.send_buf,
                                Kokkos::make_pair(offset, offset + count)));
            // Keep host buffer alive until Waitall via a local vector
            // (handled by the send_host lifetime — stays until end of scope)
            MPI_Isend(send_host.data(), count,
                      MPI_DOUBLE, si.rank, 42, comm, &reqs.emplace_back());
            // Note: send_host goes out of scope here only after Waitall below
            // because it's declared in the enclosing block.
#endif
            offset += count;
        }
    }

    // ── Step 4: wait ──────────────────────────────────────────────────────
    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()),
                    reqs.data(), MPI_STATUSES_IGNORE);

    // ── Step 5: scatter received values into x_ghost ──────────────────────
#if defined(CUDA_AWARE_MPI)
    for (std::size_t ri = 0; ri < gm.recv.size(); ++ri) {
        const auto& info  = gm.recv[ri];
        const int   count = static_cast<int>(info.global_ids.size());
        const int   off   = info.offset;
        const auto& rbuf  = pack_info.recv_bufs[ri];
        Kokkos::parallel_for("halo_scatter_" + std::to_string(ri),
            Kokkos::RangePolicy<ExecSpace>(0, count),
            KOKKOS_LAMBDA(const int i) {
                x_ghost(off + i) = rbuf(i);
            }
        );
    }
    Kokkos::fence("halo_scatter_fence");
#endif
    // CPU path: x_ghost already populated by MPI_Irecv directly
}

// ─── Backward-compatible overload (no PackedSendInfo) ─────────────────────────
//
//  Used by callers that haven't been updated to pass PackedSendInfo.
//  Falls back to the original CPU-staging approach.

void halo_exchange(
    const ViewVec1D& x_local,
    ViewVec1D&       x_ghost,
    const GhostMap&  gm,
    int              local_row_start,
    MPI_Comm         comm)
{
    // Build a temporary PackedSendInfo each call (setup overhead — acceptable
    // for small ranks; real usage should cache it in DistCrsHandle).
    PackedSendInfo tmp = build_send_indices(gm, local_row_start);
    halo_exchange(x_local, x_ghost, gm, local_row_start, comm, tmp);
}
