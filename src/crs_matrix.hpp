#pragma once
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <vector>
#include <stdexcept>

// ─── Execution / memory space aliases ────────────────────────────────────────
#if defined(KOKKOS_ENABLE_CUDA)
  using ExecSpace  = Kokkos::Cuda;
  using MemSpace   = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
  using ExecSpace  = Kokkos::HIP;
  using MemSpace   = Kokkos::HIPSpace;
#else
  using ExecSpace  = Kokkos::OpenMP;
  using MemSpace   = Kokkos::HostSpace;
#endif

using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

// ─── Core type aliases ────────────────────────────────────────────────────────
using Scalar   = double;
using Ordinal  = int;
using Offset   = int;

using CrsMatrixDevice = KokkosSparse::CrsMatrix<Scalar, Ordinal, ExecSpace,
                                                void, Offset>;
using CrsMatrixHost   = KokkosSparse::CrsMatrix<Scalar, Ordinal,
                                                HostExecSpace, void, Offset>;

using ViewVec1D      = Kokkos::View<Scalar*,       MemSpace>;
using ViewVec1DConst = Kokkos::View<const Scalar*, MemSpace>;
using ViewMat2D      = Kokkos::View<Scalar**,  Kokkos::LayoutRight, MemSpace>;
using ViewMat2DConst = Kokkos::View<const Scalar**, Kokkos::LayoutRight, MemSpace>;

// ─── Handle wrapping a device-side CSR matrix ─────────────────────────────────
struct CrsMatrixHandle {
    CrsMatrixDevice mat;
    int nrows;
    int ncols;
    int nnz;

    CrsMatrixHandle() = default;

    /**
     * Build from raw host arrays (values, col-indices, row-pointers).
     * The arrays are copied to device memory.
     */
    CrsMatrixHandle(const Scalar*  h_values,
                    const Ordinal* h_colind,
                    const Offset*  h_rowptr,
                    int            nrows_,
                    int            ncols_)
        : nrows(nrows_), ncols(ncols_), nnz(h_rowptr[nrows_])
    {
        if (nrows_ <= 0 || ncols_ <= 0)
            throw std::invalid_argument("CrsMatrixHandle: non-positive dimensions");

        // Host views (unmanaged — just wrap pointers)
        using HViewVal = Kokkos::View<const Scalar*,  Kokkos::HostSpace,
                                      Kokkos::MemoryUnmanaged>;
        using HViewCol = Kokkos::View<const Ordinal*, Kokkos::HostSpace,
                                      Kokkos::MemoryUnmanaged>;
        using HViewRow = Kokkos::View<const Offset*,  Kokkos::HostSpace,
                                      Kokkos::MemoryUnmanaged>;

        HViewVal hv(h_values, nnz);
        HViewCol hc(h_colind, nnz);
        HViewRow hr(h_rowptr, nrows_ + 1);

        // Allocate device storage and deep copy
        auto dv = Kokkos::create_mirror_view_and_copy(MemSpace{}, hv);
        auto dc = Kokkos::create_mirror_view_and_copy(MemSpace{}, hc);
        auto dr = Kokkos::create_mirror_view_and_copy(MemSpace{}, hr);

        typename CrsMatrixDevice::staticcrsgraph_type graph(dc, dr);
        mat = CrsMatrixDevice("A", nrows_, ncols_, nnz, dv, graph);
    }
};

// ─── Ghost map for distributed SpMV ──────────────────────────────────────────
struct CommInfo {
    int              rank;      ///< MPI rank to communicate with
    std::vector<int> global_ids;///< Global column indices involved
    int              offset;    ///< Byte offset into ghost buffer
};

struct GhostMap {
    std::vector<CommInfo> send;   ///< What this rank sends out
    std::vector<CommInfo> recv;   ///< What this rank receives
    int total_ghost = 0;          ///< Total # of ghost entries needed
};

// ─── Distributed CSR matrix (one object per MPI rank) ────────────────────────
struct DistCrsHandle {
    CrsMatrixHandle local;       ///< Local submatrix (own rows)
    GhostMap        ghost_map;
    int             global_nrows;///< Total rows across all ranks
    int             local_row_start;
    int             local_row_end;

    DistCrsHandle() = default;
};
