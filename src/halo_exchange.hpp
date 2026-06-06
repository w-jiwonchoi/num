#pragma once
#include "crs_matrix.hpp"
#include "mpi_singleton.hpp"
#include <vector>
#include <stdexcept>

// ─── Halo Exchange ────────────────────────────────────────────────────────────
//
//  In a distributed SpMV each rank owns a contiguous row block.
//  Some non-zero entries in the local submatrix reference columns that belong
//  to neighbouring ranks ("ghost" entries).  Before computing y = A*x we must
//  fetch those remote x values into a local ghost buffer.
//
//  CUDA-aware MPI allows us to pass GPU pointers directly to MPI_Isend /
//  MPI_Irecv, skipping the GPU→CPU→GPU round-trip.
//

/**
 * build_ghost_map
 *
 * Scans the local submatrix for off-rank column references and constructs
 * a GhostMap describing the required communication.
 *
 * @param local_row_start  First global row index owned by this rank
 * @param local_row_end    One past the last global row index
 * @param h_rowptr         Host-side CSR rowptr (length: local_rows + 1)
 * @param h_colind         Host-side CSR column indices (length: nnz)
 * @param comm             MPI communicator
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

    // Gather row partition from all ranks  [start_0, start_1, ..., start_P, total]
    std::vector<int> row_starts(size + 1);
    MPI_Allgather(&local_row_start, 1, MPI_INT,
                  row_starts.data(), 1, MPI_INT, comm);
    row_starts[size] = local_row_end;   // overwrite sentinel with our end

    // Actually we need everyone's start and end
    std::vector<int> all_starts(size), all_ends(size);
    MPI_Allgather(&local_row_start, 1, MPI_INT, all_starts.data(), 1, MPI_INT, comm);
    MPI_Allgather(&local_row_end,   1, MPI_INT, all_ends.data(),   1, MPI_INT, comm);

    // Lambda: which rank owns global column col?
    auto owner_of = [&](int col) -> int {
        for (int r = 0; r < size; ++r)
            if (col >= all_starts[r] && col < all_ends[r]) return r;
        throw std::runtime_error("Column index out of global range");
    };

    // Collect unique off-rank columns, grouped by owner
    // recv[r] = sorted list of global col indices to receive from rank r
    std::map<int, std::vector<int>> recv_cols;
    for (int i = 0; i < local_row_end - local_row_start; ++i) {
        for (int j = h_rowptr[i]; j < h_rowptr[i+1]; ++j) {
            int col = h_colind[j];
            if (col < local_row_start || col >= local_row_end) {
                int owner = owner_of(col);
                recv_cols[owner].push_back(col);
            }
        }
    }
    // Deduplicate
    for (auto& [r, v] : recv_cols) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // Tell each neighbour which indices we expect
    // (exchange recv lists to compute send lists)
    std::vector<MPI_Request> reqs;

    // Send our recv request to each owner
    std::vector<std::vector<int>> send_to(size);
    std::vector<int> recv_counts(size, 0), send_counts(size, 0);

    for (auto& [r, ids] : recv_cols)
        recv_counts[r] = ids.size();

    // Inform others of how many indices we will request
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT,
                 send_counts.data(), 1, MPI_INT, comm);

    // Exchange actual index lists
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        if (recv_counts[r] > 0) {
            reqs.emplace_back();
            MPI_Isend(recv_cols[r].data(), recv_counts[r], MPI_INT,
                      r, 0, comm, &reqs.back());
        }
        if (send_counts[r] > 0) {
            send_to[r].resize(send_counts[r]);
            reqs.emplace_back();
            MPI_Irecv(send_to[r].data(), send_counts[r], MPI_INT,
                      r, 0, comm, &reqs.back());
        }
    }
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    // Construct GhostMap
    GhostMap gm;
    int ghost_offset = 0;
    for (int r = 0; r < size; ++r) {
        if (!recv_cols[r].empty()) {
            CommInfo ci;
            ci.rank       = r;
            ci.global_ids = recv_cols[r];
            ci.offset     = ghost_offset;
            ghost_offset += ci.global_ids.size();
            gm.recv.push_back(std::move(ci));
        }
        if (!send_to[r].empty()) {
            CommInfo ci;
            ci.rank       = r;
            ci.global_ids = send_to[r];
            ci.offset     = -1;   // offsets computed at exchange time
            gm.send.push_back(std::move(ci));
        }
    }
    gm.total_ghost = ghost_offset;
    return gm;
}

/**
 * halo_exchange
 *
 * Given the current local x vector and a ghost buffer, fill the ghost
 * buffer with remote values.
 *
 * @param x_local   Local portion of x (GPU memory, length = local_rows)
 * @param x_ghost   Ghost buffer       (GPU memory, length = gm.total_ghost)
 * @param gm        Ghost map (built once and reused every iteration)
 * @param local_row_start  Global index of first local row
 * @param comm      MPI communicator
 */
void halo_exchange(
    const ViewVec1D& x_local,
    ViewVec1D&       x_ghost,
    const GhostMap&  gm,
    int              local_row_start,
    MPI_Comm         comm)
{
    std::vector<MPI_Request> reqs;
    reqs.reserve(gm.recv.size() + gm.send.size());

    // Post receives first (always)
    for (const auto& ri : gm.recv) {
        MPI_Irecv(x_ghost.data() + ri.offset,
                  static_cast<int>(ri.global_ids.size()),
                  MPI_DOUBLE, ri.rank, 42, comm, &reqs.emplace_back());
    }

    // Pack and send — gather the required local x entries
    // We build one contiguous send buffer per neighbour on the GPU side.
    // For simplicity we pack on the host; for pure GPU-aware performance
    // this should be a gather kernel.  Mark as TODO for production.
    for (const auto& si : gm.send) {
        int n = si.global_ids.size();
        // Gather local entries to a host buffer then send
        // (CUDA-aware MPI would let us send device pointers directly if
        //  they are contiguous — here they are scattered, so we pack first)
        Kokkos::View<Scalar*, Kokkos::HostSpace> send_buf("send_buf", n);
        auto x_host = Kokkos::create_mirror_view(x_local);
        Kokkos::deep_copy(x_host, x_local);
        for (int i = 0; i < n; ++i)
            send_buf(i) = x_host(si.global_ids[i] - local_row_start);

        MPI_Isend(send_buf.data(), n, MPI_DOUBLE,
                  si.rank, 42, comm, &reqs.emplace_back());
    }

    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}
