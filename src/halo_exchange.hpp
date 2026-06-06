#pragma once
#include "crs_matrix.hpp"
#include "mpi_singleton.hpp"
#include <map>          // ← Bug fix: was missing, used std::map below
#include <set>
#include <vector>
#include <algorithm>
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

    // Gather row partition from all ranks
    // Bug fix: original code overwrote row_starts[size] which was the
    // wrong sentinel.  Use two separate allgather calls for starts and ends.
    std::vector<int> all_starts(size), all_ends(size);
    MPI_Allgather(&local_row_start, 1, MPI_INT,
                  all_starts.data(), 1, MPI_INT, comm);
    MPI_Allgather(&local_row_end, 1, MPI_INT,
                  all_ends.data(), 1, MPI_INT, comm);

    // Lambda: which rank owns global column col?
    // Uses the gathered partition table — O(size) but called at setup only.
    auto owner_of = [&](int col) -> int {
        for (int r = 0; r < size; ++r)
            if (col >= all_starts[r] && col < all_ends[r]) return r;
        throw std::runtime_error(
            "build_ghost_map: column index " + std::to_string(col) +
            " is out of the global row range");
    };

    // Collect unique off-rank columns, grouped by owner rank.
    // recv_cols[r] = sorted unique list of global col indices to receive from r.
    std::map<int, std::set<int>> recv_cols_set;
    int local_rows = local_row_end - local_row_start;
    for (int i = 0; i < local_rows; ++i) {
        for (int j = h_rowptr[i]; j < h_rowptr[i + 1]; ++j) {
            int col = h_colind[j];
            // Only off-rank columns need halo exchange
            if (col < local_row_start || col >= local_row_end) {
                int owner = owner_of(col);
                recv_cols_set[owner].insert(col);
            }
        }
    }

    // Convert sets to sorted vectors
    std::map<int, std::vector<int>> recv_cols;
    for (auto& [r, s] : recv_cols_set) {
        recv_cols[r] = std::vector<int>(s.begin(), s.end());
    }

    // Tell each neighbour which indices we expect from them.
    // Step 1: exchange recv counts via MPI_Alltoall
    std::vector<int> recv_counts(size, 0), send_counts(size, 0);
    for (auto& [r, ids] : recv_cols) {
        recv_counts[r] = static_cast<int>(ids.size());
    }
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT,
                 send_counts.data(), 1, MPI_INT, comm);

    // Step 2: exchange actual index lists
    // Bug fix: original code used MPI_Isend/Irecv and then pushed to reqs,
    //          but send_buf was a local scope variable that went out of scope
    //          before MPI_Waitall — undefined behaviour / use-after-free.
    //          Fix: store all send buffers in a vector that lives until Waitall.
    std::vector<std::vector<int>> send_bufs(size);   // ← lifetime extended
    std::vector<std::vector<int>> recv_bufs(size);
    std::vector<MPI_Request> reqs;

    // Post all receives first (good MPI practice)
    for (int r = 0; r < size; ++r) {
        if (r == rank || send_counts[r] == 0) continue;
        recv_bufs[r].resize(send_counts[r]);
        reqs.emplace_back();
        MPI_Irecv(recv_bufs[r].data(), send_counts[r], MPI_INT,
                  r, 1001, comm, &reqs.back());
    }

    // Then post sends
    for (int r = 0; r < size; ++r) {
        if (r == rank || recv_counts[r] == 0) continue;
        // send_bufs[r] lives until MPI_Waitall — no dangling pointer
        send_bufs[r] = recv_cols[r];
        reqs.emplace_back();
        MPI_Isend(send_bufs[r].data(), recv_counts[r], MPI_INT,
                  r, 1001, comm, &reqs.back());
    }

    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()),
                    reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    // Construct GhostMap
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
 * @param x_local         Local portion of x (GPU memory, length = local_rows)
 * @param x_ghost         Ghost buffer       (GPU memory, length = gm.total_ghost)
 * @param gm              Ghost map (built once and reused every iteration)
 * @param local_row_start Global index of first local row
 * @param comm            MPI communicator
 */
void halo_exchange(
    const ViewVec1D& x_local,
    ViewVec1D&       x_ghost,
    const GhostMap&  gm,
    int              local_row_start,
    MPI_Comm         comm)
{
    // Bug fix: original code allocated send_buf inside a loop as a local
    //          Kokkos::View, then called MPI_Isend and immediately pushed to
    //          reqs.  But the local send_buf went out of scope at the end of
    //          the loop body, destroying the allocation before MPI finished
    //          reading it — classic use-after-free.
    //
    //          Fix: collect all send buffers in a std::vector that lives until
    //          MPI_Waitall.

    // Step 0: mirror x_local to host once (needed for packing)
    auto x_host_mirror = Kokkos::create_mirror_view(x_local);
    Kokkos::deep_copy(x_host_mirror, x_local);

    std::vector<MPI_Request> reqs;
    reqs.reserve(gm.recv.size() + gm.send.size());

    // ── Post receives first ────────────────────────────────────────────────
    for (const auto& ri : gm.recv) {
        MPI_Irecv(x_ghost.data() + ri.offset,
                  static_cast<int>(ri.global_ids.size()),
                  MPI_DOUBLE, ri.rank, 42, comm, &reqs.emplace_back());
    }

    // ── Pack and post sends ────────────────────────────────────────────────
    // send_raw_ptrs: storage for packed doubles that outlives MPI_Waitall
    std::vector<std::vector<double>> send_storage(gm.send.size());

    for (std::size_t si_idx = 0; si_idx < gm.send.size(); ++si_idx) {
        const auto& si = gm.send[si_idx];
        int n = static_cast<int>(si.global_ids.size());

        // Pack required local x entries into a contiguous host buffer.
        // (For CUDA-aware MPI with contiguous data, we could send the device
        //  pointer directly; scattered data requires packing regardless.)
        send_storage[si_idx].resize(n);
        for (int i = 0; i < n; ++i) {
            int local_idx = si.global_ids[i] - local_row_start;
            send_storage[si_idx][i] = x_host_mirror(local_idx);
        }

        reqs.emplace_back();
        MPI_Isend(send_storage[si_idx].data(), n, MPI_DOUBLE,
                  si.rank, 42, comm, &reqs.back());
    }

    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()),
                    reqs.data(), MPI_STATUSES_IGNORE);
}
