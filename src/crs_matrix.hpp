#pragma once
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <vector>
#include <memory>
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
    int nrows = 0;
    int ncols = 0;
    int nnz   = 0;

    CrsMatrixHandle() = default;

    CrsMatrixHandle(const Scalar*  h_values,
                    const Ordinal* h_colind,
                    const Offset*  h_rowptr,
                    int            nrows_,
                    int            ncols_)
        : nrows(nrows_), ncols(ncols_), nnz(h_rowptr[nrows_])
    {
        if (nrows_ <= 0 || ncols_ <= 0)
            throw std::invalid_argument("CrsMatrixHandle: non-positive dimensions");

        using HViewVal = Kokkos::View<const Scalar*,  Kokkos::HostSpace,
                                      Kokkos::MemoryUnmanaged>;
        using HViewCol = Kokkos::View<const Ordinal*, Kokkos::HostSpace,
                                      Kokkos::MemoryUnmanaged>;
        using HViewRow = Kokkos::View<const Offset*,  Kokkos::HostSpace,
                                      Kokkos::MemoryUnmanaged>;

        HViewVal hv(h_values, nnz);
        HViewCol hc(h_colind, nnz);
        HViewRow hr(h_rowptr, nrows_ + 1);

        auto dv = Kokkos::create_mirror_view_and_copy(MemSpace{}, hv);
        auto dc = Kokkos::create_mirror_view_and_copy(MemSpace{}, hc);
        auto dr = Kokkos::create_mirror_view_and_copy(MemSpace{}, hr);

        typename CrsMatrixDevice::staticcrsgraph_type graph(dc, dr);
        mat = CrsMatrixDevice("A", ncols_, dv, graph);
    }
};

// ─── Ghost map for distributed SpMV ──────────────────────────────────────────
struct CommInfo {
    int              rank;
    std::vector<int> global_ids;
    int              offset;
};

struct GhostMap {
    std::vector<CommInfo> send;
    std::vector<CommInfo> recv;
    int total_ghost = 0;
};

struct PackedSendInfo;   // defined in halo_exchange.hpp

// ─── Distributed CSR matrix (one object per MPI rank) ────────────────────────
//
//  v0.2 additions:
//   - interior / boundary split matrices for communication-compute overlap.
//     interior: rows touching ONLY local columns  (can run during halo comm)
//     boundary: rows touching at least one ghost  (runs after halo completes)
//   - x_ext_cache: extended-vector buffer allocated once per matrix, reused
//     by every dist_spmv / CG iteration (was reallocated per call in v0.1).

struct DistCrsHandle {
    CrsMatrixHandle local;       // full local block (fallback / non-overlap)
    CrsMatrixHandle interior;    // valid iff has_split
    CrsMatrixHandle boundary;    // valid iff has_split
    bool            has_split = false;

    GhostMap        ghost_map;
    int             global_nrows    = 0;
    int             local_row_start = 0;
    int             local_row_end   = 0;

    std::shared_ptr<PackedSendInfo> pack_info;
    std::shared_ptr<ViewVec1D>      x_ext_cache;   // size local+ghost

    DistCrsHandle() = default;
};
