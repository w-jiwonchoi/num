#pragma once
#include "crs_matrix.hpp"
#include "mpi_singleton.hpp"
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <stdexcept>

// ─── Halo Exchange (v0.2 — fixed) ────────────────────────────────────────────
//
//  Fixes vs v0.1:
//
//  [BUG-1] CPU-staging path posted MPI_Irecv directly into x_ghost.data(),
//          which is DEVICE memory. Without CUDA-aware MPI this is undefined
//          behaviour → the 4-GPU segfault. Fixed: persistent HOST recv
//          buffers, then deep_copy to device after Waitall.
//
//  [BUG-2] Host send mirrors were created inside the send loop and destroyed
//          before MPI_Waitall (use-after-free). Fixed: one persistent host
//          mirror of the packed send buffer, copied once per exchange.
//
//  [PERF]  All staging buffers are allocated ONCE in build_send_indices and
//          cached in PackedSendInfo — zero allocation on the hot path.
//
//  [PERF]  Exchange is split into begin() / end() so the caller can overlap
//          interior SpMV with communication (see dist_spmv.hpp).

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
            if (col < local_row_start || col >= local_row_end)
                recv_cols_set[owner_of(col)].insert(col);
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

    std::vector<std::vector<int>> send_bufs(size), recv_bufs(size);
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

// ─── PackedSendInfo: all buffers built once, reused every SpMV ───────────────

struct PackedSendInfo {
    ViewVec1D                    send_buf;          // device, flat
    std::vector<ViewVec1D>       recv_bufs;         // device, per-neighbour
    Kokkos::View<int*, MemSpace> send_indices_dev;
    int                          total_send = 0;

    // Persistent HOST staging (used only when CUDA-aware MPI is OFF).
    // These live as long as the handle → no use-after-free, no per-call alloc.
    typename ViewVec1D::HostMirror              send_buf_host;
    std::vector<typename ViewVec1D::HostMirror> recv_bufs_host;
};

inline PackedSendInfo build_send_indices(
    const GhostMap& gm,
    int local_row_start)
{
    int total_send = 0;
    for (const auto& si : gm.send)
        total_send += static_cast<int>(si.global_ids.size());

    std::vector<int> h_send_idx;
    h_send_idx.reserve(total_send);
    for (const auto& si : gm.send)
        for (int gid : si.global_ids)
            h_send_idx.push_back(gid - local_row_start);

    Kokkos::View<int*, MemSpace> d_send_idx("send_indices",
                                            std::max(total_send, 1));
    {
        auto mirror = Kokkos::create_mirror_view(d_send_idx);
        for (int i = 0; i < total_send; ++i) mirror(i) = h_send_idx[i];
        Kokkos::deep_copy(d_send_idx, mirror);
    }

    PackedSendInfo info;
    info.send_buf         = ViewVec1D("send_buf", std::max(total_send, 1));
    info.send_indices_dev = d_send_idx;
    info.total_send       = total_send;
    info.send_buf_host    = Kokkos::create_mirror_view(info.send_buf);

    info.recv_bufs.reserve(gm.recv.size());
    info.recv_bufs_host.reserve(gm.recv.size());
    for (const auto& ri : gm.recv) {
        info.recv_bufs.emplace_back(
            ViewVec1D("recv_buf_" + std::to_string(ri.rank),
                      static_cast<int>(ri.global_ids.size())));
        info.recv_bufs_host.emplace_back(
            Kokkos::create_mirror_view(info.recv_bufs.back()));
    }
    return info;
}

// ─── Split exchange: begin (pack + post sends/recvs) ─────────────────────────

inline void halo_exchange_begin(
    const ViewVec1D&          x_local,
    const GhostMap&           gm,
    MPI_Comm                  comm,
    PackedSendInfo&           pack,
    std::vector<MPI_Request>& reqs)
{
    if (gm.recv.empty() && gm.send.empty()) return;

    // Pack on device (parallel), fence so the buffer is complete
    if (pack.total_send > 0) {
        const int n  = pack.total_send;
        auto idx     = pack.send_indices_dev;
        auto buf     = pack.send_buf;
        Kokkos::parallel_for("halo_pack",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int i) { buf(i) = x_local(idx(i)); });
        Kokkos::fence("halo_pack_fence");
#if !defined(CUDA_AWARE_MPI)
        // One device→host copy of the whole packed buffer (FIX for BUG-2:
        // this mirror is persistent, alive until Waitall and beyond)
        Kokkos::deep_copy(pack.send_buf_host, pack.send_buf);
#endif
    }

    // Post all Irecv first
    for (std::size_t ri = 0; ri < gm.recv.size(); ++ri) {
        const auto& info  = gm.recv[ri];
        const int   count = static_cast<int>(info.global_ids.size());
        reqs.emplace_back();
#if defined(CUDA_AWARE_MPI)
        MPI_Irecv(pack.recv_bufs[ri].data(),      count, MPI_DOUBLE,
                  info.rank, 42, comm, &reqs.back());
#else
        // FIX for BUG-1: receive into HOST memory, never device memory
        MPI_Irecv(pack.recv_bufs_host[ri].data(), count, MPI_DOUBLE,
                  info.rank, 42, comm, &reqs.back());
#endif
    }

    // Post all Isend
    int offset = 0;
    for (const auto& si : gm.send) {
        const int count = static_cast<int>(si.global_ids.size());
        reqs.emplace_back();
#if defined(CUDA_AWARE_MPI)
        MPI_Isend(pack.send_buf.data()      + offset, count, MPI_DOUBLE,
                  si.rank, 42, comm, &reqs.back());
#else
        MPI_Isend(pack.send_buf_host.data() + offset, count, MPI_DOUBLE,
                  si.rank, 42, comm, &reqs.back());
#endif
        offset += count;
    }
}

// ─── Split exchange: end (wait + scatter into x_ghost on device) ─────────────

inline void halo_exchange_end(
    ViewVec1D&                x_ghost,
    const GhostMap&           gm,
    PackedSendInfo&           pack,
    std::vector<MPI_Request>& reqs)
{
    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()),
                    reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    for (std::size_t ri = 0; ri < gm.recv.size(); ++ri) {
        const auto& info  = gm.recv[ri];
        const int   count = static_cast<int>(info.global_ids.size());
        const int   off   = info.offset;
#if !defined(CUDA_AWARE_MPI)
        // Host → device upload of this neighbour's payload
        Kokkos::deep_copy(pack.recv_bufs[ri], pack.recv_bufs_host[ri]);
#endif
        auto rbuf = pack.recv_bufs[ri];
        Kokkos::parallel_for("halo_scatter",
            Kokkos::RangePolicy<ExecSpace>(0, count),
            KOKKOS_LAMBDA(const int i) { x_ghost(off + i) = rbuf(i); });
    }
    Kokkos::fence("halo_scatter_fence");
}

// ─── Convenience: blocking exchange ──────────────────────────────────────────

inline void halo_exchange(
    const ViewVec1D& x_local,
    ViewVec1D&       x_ghost,
    const GhostMap&  gm,
    int              /*local_row_start*/,
    MPI_Comm         comm,
    PackedSendInfo&  pack)
{
    if (gm.recv.empty() && gm.send.empty()) return;
    std::vector<MPI_Request> reqs;
    reqs.reserve(gm.recv.size() + gm.send.size());
    halo_exchange_begin(x_local, gm, comm, pack, reqs);
    halo_exchange_end(x_ghost, gm, pack, reqs);
}

// Backward-compatible overload (builds temp pack info — setup path only)
inline void halo_exchange(
    const ViewVec1D& x_local,
    ViewVec1D&       x_ghost,
    const GhostMap&  gm,
    int              local_row_start,
    MPI_Comm         comm)
{
    PackedSendInfo tmp = build_send_indices(gm, local_row_start);
    halo_exchange(x_local, x_ghost, gm, local_row_start, comm, tmp);
}
