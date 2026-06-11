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

// ─── PackedSendInfo (forward declaration; defined in halo_exchange.hpp) ───────
//
//  We need the struct visible here so DistCrsHandle can hold it.
//  Forward-declare only; halo_exchange.hpp defines it fully and is always
//  included before any code that accesses the fields.
struct PackedSendInfo;

// ─── Distributed CSR matrix (one object per MPI rank) ────────────────────────
//
//  Added pack_info: pre-computed device-side send-index array.
//  Built once in distribute_csr, reused every halo_exchange call.
//  This eliminates the per-iteration setup overhead in the original code.

struct DistCrsHandle {
    CrsMatrixHandle              local;
    GhostMap                     ghost_map;
    int                          global_nrows;
    int                          local_row_start;
    int                          local_row_end;

    // ── Performance addition: cached device-side pack metadata ───────────
    //  Stored as shared_ptr so DistCrsHandle is copyable (needed by Python
    //  bindings which may copy the handle into Python objects).
    std::shared_ptr<PackedSendInfo> pack_info;  // null until distribute_csr sets it

    DistCrsHandle() = default;
};
