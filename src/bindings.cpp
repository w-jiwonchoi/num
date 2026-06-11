#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "crs_matrix.hpp"
#include "spmv_kernel.hpp"
#include "batch_spmv_kernel.hpp"
#include "dist_spmv.hpp"
#include "cg_solver.hpp"
#include "mpi_singleton.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

// ─── Shared_ptr handles exposed to Python ─────────────────────────────────────
using CrsHandle  = std::shared_ptr<CrsMatrixHandle>;
using DistHandle = std::shared_ptr<DistCrsHandle>;

// ─── Ndarray shorthand types ──────────────────────────────────────────────────
using f64_1d_cpu  = nb::ndarray<const double, nb::ndim<1>, nb::device::cpu>;
using f64_2d_cpu  = nb::ndarray<const double, nb::ndim<2>, nb::device::cpu>;
using i32_1d_cpu  = nb::ndarray<const int,    nb::ndim<1>, nb::device::cpu>;

#if defined(KOKKOS_ENABLE_CUDA)
using f64_1d_gpu  = nb::ndarray<double,       nb::ndim<1>, nb::device::cuda>;
using f64_2d_gpu  = nb::ndarray<double,       nb::ndim<2>, nb::device::cuda>;
using f64_1d_gpuc = nb::ndarray<const double, nb::ndim<1>, nb::device::cuda>;
using f64_2d_gpuc = nb::ndarray<const double, nb::ndim<2>, nb::device::cuda>;
#endif

NB_MODULE(kokkos_spmv, m) {

    m.doc() = R"(
kokkos_spmv — Multi-GPU Sparse Linear Algebra Library
=====================================================
Provides:
  - Single-GPU SpMV  : spmv(A, x, y)
  - Batch SpMV       : batch_spmv(A, X, Y)
  - Distributed SpMV : dist_spmv(A_dist, x_local, y_local)
  - CG solver        : cg_solve(A_dist, b, x, tol, max_iter)

All GPU arrays are accepted zero-copy from CuPy / PyTorch / JAX (cuda device).
)";

    // ── Initialization ───────────────────────────────────────────────────────
    m.def("init", [](int device_id) -> nb::tuple {
        if (!Kokkos::is_initialized()) {
            Kokkos::InitializationSettings settings;
            settings.set_device_id(device_id);
            Kokkos::initialize(settings);
            std::atexit([]() {
                if (Kokkos::is_initialized()) Kokkos::finalize();
            });
        }
        auto& mpi = MPIContext::instance();
        return nb::make_tuple(mpi.rank(), mpi.size());
    }, "device_id"_a = 0,
    "Initialise Kokkos and MPI.  Returns (rank, size).  "
    "Safe to call multiple times.  Pass device_id = rank % n_gpu for multi-GPU.");

    m.def("finalize", []() {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }, "Explicitly finalise Kokkos (optional — atexit handles it).");

    // ── MPI helpers ──────────────────────────────────────────────────────────
    m.def("rank",    []() { return MPIContext::instance().rank(); });
    m.def("size",    []() { return MPIContext::instance().size(); });
    m.def("barrier", []() { MPI_Barrier(MPIContext::instance().comm()); },
          "MPI_Barrier on COMM_WORLD.");

    // ── CrsMatrixHandle ───────────────────────────────────────────────────────
    nb::class_<CrsMatrixHandle>(m, "CrsMatrix",
        "Opaque handle to a device-side CSR matrix.")
        .def_ro("nrows", &CrsMatrixHandle::nrows)
        .def_ro("ncols", &CrsMatrixHandle::ncols)
        .def_ro("nnz",   &CrsMatrixHandle::nnz)
        .def("__repr__", [](const CrsMatrixHandle& h) {
            return "<CrsMatrix nrows=" + std::to_string(h.nrows)
                 + " ncols=" + std::to_string(h.ncols)
                 + " nnz="   + std::to_string(h.nnz) + ">";
        });

    nb::class_<DistCrsHandle>(m, "DistCrsMatrix",
        "Opaque handle to a distributed CSR matrix (one per MPI rank).")
        .def_ro("global_nrows",    &DistCrsHandle::global_nrows)
        .def_ro("local_row_start", &DistCrsHandle::local_row_start)
        .def_ro("local_row_end",   &DistCrsHandle::local_row_end);

    // ── upload_csr ───────────────────────────────────────────────────────────
    m.def("upload_csr",
        [](i32_1d_cpu rowptr_nb,
           i32_1d_cpu colind_nb,
           f64_1d_cpu values_nb,
           int nrows, int ncols) -> CrsMatrixHandle
        {
            return CrsMatrixHandle(
                values_nb.data(),
                colind_nb.data(),
                rowptr_nb.data(),
                nrows, ncols);
        },
        "rowptr"_a, "colind"_a, "values"_a, "nrows"_a, "ncols"_a,
        R"(
Upload a CSR matrix from host NumPy arrays to the GPU.

Parameters
----------
rowptr : ndarray[int32, ndim=1]   Row pointer array (length nrows+1)
colind : ndarray[int32, ndim=1]   Column index array (length nnz)
values : ndarray[float64, ndim=1] Non-zero value array (length nnz)
nrows, ncols : int                Matrix dimensions

Returns
-------
CrsMatrix  Opaque GPU handle
)");

    // ── distribute_csr ───────────────────────────────────────────────────────
    m.def("distribute_csr",
        [](i32_1d_cpu rowptr_nb,
           i32_1d_cpu colind_nb,
           f64_1d_cpu values_nb,
           int global_nrows, int global_ncols) -> DistCrsHandle
        {
            std::vector<Offset>  rowptr(rowptr_nb.data(),
                                        rowptr_nb.data() + global_nrows + 1);
            std::vector<Ordinal> colind(colind_nb.data(),
                                        colind_nb.data() + rowptr.back());
            std::vector<Scalar>  values(values_nb.data(),
                                        values_nb.data() + rowptr.back());

            return distribute_csr(values, colind, rowptr,
                                  global_nrows, global_ncols,
                                  MPIContext::instance().comm());
        },
        "rowptr"_a, "colind"_a, "values"_a, "global_nrows"_a, "global_ncols"_a,
        R"(
Partition a CSR matrix across MPI ranks.  Each rank receives its local
submatrix.  Device-side send-index arrays are pre-built and cached so that
subsequent dist_spmv calls do not repeat setup work.
)");

#if defined(KOKKOS_ENABLE_CUDA)
    // ── spmv (single GPU) ────────────────────────────────────────────────────
    m.def("spmv",
        [](const CrsMatrixHandle& A,
           f64_1d_gpuc            x_nb,
           f64_1d_gpu             y_nb,
           bool                   use_kokkoskernels)
        {
            if ((int)x_nb.shape(0) != A.ncols)
                throw std::invalid_argument("spmv: x length != A.ncols");
            if ((int)y_nb.shape(0) != A.nrows)
                throw std::invalid_argument("spmv: y length != A.nrows");

            ViewVec1DConst x(x_nb.data(), x_nb.shape(0));
            ViewVec1D      y(y_nb.data(), y_nb.shape(0));

            if (use_kokkoskernels)
                launch_spmv_kokkoskernels(A.mat, x, y);
            else
                launch_spmv_custom(A.mat, x, y);
        },
        "A"_a, "x"_a, "y"_a, "kokkoskernels"_a = true,
        R"(
Single-GPU SpMV: y = A @ x  (in-place into y)

Parameters
----------
A   : CrsMatrix   — GPU matrix handle
x   : cuda float64 ndarray of shape (ncols,)
y   : cuda float64 ndarray of shape (nrows,)  — output written here
kokkoskernels : bool — if True, use KokkosKernels/cuSPARSE backend
                       if False, use custom hierarchical TeamPolicy kernel
                       (the custom kernel is auto-tuned by nnz/row)
)");

    // ── batch_spmv (single GPU) ──────────────────────────────────────────────
    m.def("batch_spmv",
        [](const CrsMatrixHandle& A,
           f64_2d_gpuc            X_nb,
           f64_2d_gpu             Y_nb)
        {
            int N = X_nb.shape(0), k = X_nb.shape(1);
            if (N != A.ncols)
                throw std::invalid_argument("batch_spmv: X.shape[0] != A.ncols");
            if ((int)Y_nb.shape(0) != A.nrows || (int)Y_nb.shape(1) != k)
                throw std::invalid_argument("batch_spmv: Y must be (A.nrows, k)");

            ViewMat2DConst X(X_nb.data(), N, k);
            ViewMat2D      Y(Y_nb.data(), A.nrows, k);
            launch_batch_spmv(A.mat, X, Y, A.nrows, k);
        },
        "A"_a, "X"_a, "Y"_a,
        R"(
Batch SpMV: Y = A @ X  where X has shape (N, k).

Dispatch:
  k < 4   → MDRangePolicy with correct tile orientation (inner dim = rhs)
  4<=k<32 → TeamPolicy hierarchical (register accumulation per rhs column)
  k >= 32 → KokkosKernels SPMM (cuSPARSE SpMM, fastest for large k)

Parameters
----------
A : CrsMatrix             — GPU matrix handle
X : cuda float64 (N, k)  — input (C-contiguous / row-major)
Y : cuda float64 (N, k)  — output written here
)");

    // ── dist_spmv ────────────────────────────────────────────────────────────
    m.def("dist_spmv",
        [](const DistCrsHandle& A_dist,
           f64_1d_gpuc          x_nb,
           f64_1d_gpu           y_nb)
        {
            ViewVec1D x_local(const_cast<double*>(x_nb.data()), x_nb.shape(0));
            ViewVec1D y_local(y_nb.data(), y_nb.shape(0));
            distributed_spmv(A_dist, x_local, y_local,
                             MPIContext::instance().comm());
        },
        "A_dist"_a, "x_local"_a, "y_local"_a,
        R"(
Distributed SpMV: y_local = A_dist @ x_global (via GPU-packed halo exchange).

Improvements vs v0.1:
  - Send buffer packed on GPU (parallel_for over send indices)
  - GPU-direct MPI when compiled with -DENABLE_CUDA_AWARE_MPI=ON
  - x_ext allocation reused across calls when pack_info is cached
)");

    // ── cg_solve (distributed) ───────────────────────────────────────────────
    m.def("cg_solve",
        [](const DistCrsHandle& A_dist,
           f64_1d_gpuc          b_nb,
           f64_1d_gpu           x_nb,
           double               tol,
           int                  max_iter) -> nb::dict
        {
            ViewVec1D b_view(const_cast<double*>(b_nb.data()), b_nb.shape(0));
            ViewVec1D x_view(x_nb.data(), x_nb.shape(0));
            CGResult res = cg_solve_distributed(
                A_dist, b_view, x_view, tol, max_iter,
                MPIContext::instance().comm());

            nb::dict d;
            d["iters"]     = res.iters;
            d["residual"]  = res.final_res;
            d["converged"] = res.converged;
            return d;
        },
        "A_dist"_a, "b"_a, "x"_a,
        "tol"_a = 1e-10, "max_iter"_a = 1000,
        R"(
Distributed Conjugate Gradient solver for  A x = b.

A must be symmetric positive definite.  x is the initial guess and is
overwritten with the solution.

Returns dict with keys:
  'iters'     : int   — iterations performed
  'residual'  : float — ||r||/||b|| at termination
  'converged' : bool  — True if tol was reached
)");

    // ── cg_solve_local (single GPU) ──────────────────────────────────────────
    m.def("cg_solve_local",
        [](const CrsMatrixHandle& A,
           f64_1d_gpuc             b_nb,
           f64_1d_gpu              x_nb,
           double                  tol,
           int                     max_iter) -> nb::dict
        {
            ViewVec1D b_view(const_cast<double*>(b_nb.data()), b_nb.shape(0));
            ViewVec1D x_view(x_nb.data(), x_nb.shape(0));
            CGResult res = cg_solve_local(A, b_view, x_view, tol, max_iter);
            nb::dict d;
            d["iters"]     = res.iters;
            d["residual"]  = res.final_res;
            d["converged"] = res.converged;
            return d;
        },
        "A"_a, "b"_a, "x"_a, "tol"_a = 1e-10, "max_iter"_a = 1000,
        "Single-GPU CG solver (no MPI).  Useful for benchmarking and testing.");

#endif  // KOKKOS_ENABLE_CUDA

}  // NB_MODULE
