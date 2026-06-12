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

using CrsHandle  = std::shared_ptr<CrsMatrixHandle>;
using DistHandle = std::shared_ptr<DistCrsHandle>;

using f64_1d_cpu  = nb::ndarray<const double, nb::ndim<1>, nb::device::cpu>;
using f64_2d_cpu  = nb::ndarray<const double, nb::ndim<2>, nb::device::cpu>;
using i32_1d_cpu  = nb::ndarray<const int,    nb::ndim<1>, nb::device::cpu>;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
using f64_1d_gpu  = nb::ndarray<double,       nb::ndim<1>, nb::device::cuda>;
using f64_2d_gpu  = nb::ndarray<double,       nb::ndim<2>, nb::device::cuda>;
using f64_1d_gpuc = nb::ndarray<const double, nb::ndim<1>, nb::device::cuda>;
using f64_2d_gpuc = nb::ndarray<const double, nb::ndim<2>, nb::device::cuda>;
#define KSPMV_HAS_GPU 1
#endif

NB_MODULE(kokkos_spmv, m) {

    m.doc() = R"(
kokkos_spmv — Multi-GPU Sparse Linear Algebra Library (v0.3)
============================================================
  - Single-GPU SpMV     : spmv(A, x, y, kokkoskernels=True)
  - Batch SpMV (kk)     : batch_spmv(A, X, Y)
  - Batch SpMV (custom) : custom_batch_spmv(A, X, Y)   ← NEW: template-unrolled
  - Distributed SpMV    : dist_spmv(A_dist, x_local, y_local, ...)
  - CG solver           : cg_solve(A_dist, b, x, tol, max_iter)
GPU arrays are accepted zero-copy from CuPy / PyTorch / JAX.
)";

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
    }, "device_id"_a = 0);

    m.def("finalize", []() {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    });

    m.def("rank",    []() { return MPIContext::instance().rank(); });
    m.def("size",    []() { return MPIContext::instance().size(); });
    m.def("barrier", []() { MPI_Barrier(MPIContext::instance().comm()); });

    m.def("cuda_aware_mpi", []() {
#if defined(CUDA_AWARE_MPI)
        return true;
#else
        return false;
#endif
    });

    nb::class_<CrsMatrixHandle>(m, "CrsMatrix")
        .def_ro("nrows", &CrsMatrixHandle::nrows)
        .def_ro("ncols", &CrsMatrixHandle::ncols)
        .def_ro("nnz",   &CrsMatrixHandle::nnz)
        .def("__repr__", [](const CrsMatrixHandle& h) {
            return "<CrsMatrix nrows=" + std::to_string(h.nrows)
                 + " ncols=" + std::to_string(h.ncols)
                 + " nnz="   + std::to_string(h.nnz) + ">";
        });

    nb::class_<DistCrsHandle>(m, "DistCrsMatrix")
        .def_ro("global_nrows",    &DistCrsHandle::global_nrows)
        .def_ro("local_row_start", &DistCrsHandle::local_row_start)
        .def_ro("local_row_end",   &DistCrsHandle::local_row_end)
        .def_ro("has_split",       &DistCrsHandle::has_split);

    m.def("upload_csr",
        [](i32_1d_cpu rowptr_nb, i32_1d_cpu colind_nb, f64_1d_cpu values_nb,
           int nrows, int ncols) -> CrsMatrixHandle {
            return CrsMatrixHandle(values_nb.data(), colind_nb.data(),
                                   rowptr_nb.data(), nrows, ncols);
        }, "rowptr"_a, "colind"_a, "values"_a, "nrows"_a, "ncols"_a);

    m.def("distribute_csr",
        [](i32_1d_cpu rowptr_nb, i32_1d_cpu colind_nb, f64_1d_cpu values_nb,
           int global_nrows, int global_ncols) -> DistCrsHandle {
            std::vector<Offset>  rowptr(rowptr_nb.data(),
                                        rowptr_nb.data() + global_nrows + 1);
            std::vector<Ordinal> colind(colind_nb.data(),
                                        colind_nb.data() + rowptr.back());
            std::vector<Scalar>  values(values_nb.data(),
                                        values_nb.data() + rowptr.back());
            return distribute_csr(values, colind, rowptr, global_nrows,
                                  global_ncols, MPIContext::instance().comm());
        }, "rowptr"_a, "colind"_a, "values"_a, "global_nrows"_a, "global_ncols"_a);

#if defined(KSPMV_HAS_GPU)
    m.def("spmv",
        [](const CrsMatrixHandle& A, f64_1d_gpuc x_nb, f64_1d_gpu y_nb,
           bool use_kokkoskernels) {
            if ((int)x_nb.shape(0) != A.ncols)
                throw std::invalid_argument("spmv: x length != A.ncols");
            if ((int)y_nb.shape(0) != A.nrows)
                throw std::invalid_argument("spmv: y length != A.nrows");
            ViewVec1DConst x(x_nb.data(), x_nb.shape(0));
            ViewVec1D      y(y_nb.data(), y_nb.shape(0));
            if (use_kokkoskernels) launch_spmv_kokkoskernels(A.mat, x, y);
            else                   launch_spmv_custom(A.mat, x, y);
        }, "A"_a, "x"_a, "y"_a, "kokkoskernels"_a = true);

    // ── Batch SpMV (KokkosKernels / cuSPARSE path) ───────────────────────────
    m.def("batch_spmv",
        [](const CrsMatrixHandle& A, f64_2d_gpuc X_nb, f64_2d_gpu Y_nb) {
            int N = X_nb.shape(0), k = X_nb.shape(1);
            if (N != A.ncols)
                throw std::invalid_argument("batch_spmv: X.shape[0] != A.ncols");
            if ((int)Y_nb.shape(0) != A.nrows || (int)Y_nb.shape(1) != k)
                throw std::invalid_argument("batch_spmv: Y must be (A.nrows, k)");
            ViewMat2DConst X(X_nb.data(), N, k);
            ViewMat2D      Y(Y_nb.data(), A.nrows, k);
            launch_batch_spmv(A.mat, X, Y, A.nrows, k);
        }, "A"_a, "X"_a, "Y"_a);

    // ── Batch SpMV (custom template-unrolled path) — NEW ─────────────────────
    m.def("custom_batch_spmv",
        [](const CrsMatrixHandle& A, f64_2d_gpuc X_nb, f64_2d_gpu Y_nb) {
            int N = X_nb.shape(0), k = X_nb.shape(1);
            if (N != A.ncols)
                throw std::invalid_argument(
                    "custom_batch_spmv: X.shape[0] != A.ncols");
            if ((int)Y_nb.shape(0) != A.nrows || (int)Y_nb.shape(1) != k)
                throw std::invalid_argument(
                    "custom_batch_spmv: Y must be (A.nrows, k)");
            ViewMat2DConst X(X_nb.data(), N, k);
            ViewMat2D      Y(Y_nb.data(), A.nrows, k);
            launch_batch_spmv_custom(A.mat, X, Y, A.nrows, k);
        }, "A"_a, "X"_a, "Y"_a,
        R"(
Custom template-unrolled Batch SpMV.
k ∈ {8, 16, 32} → fully unrolled kernel (low launch overhead, ~JAX-level).
Other k          → hierarchical or cuSPARSE fallback.
Use instead of batch_spmv when k is small (< 64).
)");

    m.def("dist_spmv",
        [](const DistCrsHandle& A_dist, f64_1d_gpuc x_nb, f64_1d_gpu y_nb,
           bool kokkoskernels, bool overlap) {
            ViewVec1D x_local(const_cast<double*>(x_nb.data()), x_nb.shape(0));
            ViewVec1D y_local(y_nb.data(), y_nb.shape(0));
            distributed_spmv(A_dist, x_local, y_local,
                             MPIContext::instance().comm(),
                             kokkoskernels, overlap);
        }, "A_dist"_a, "x_local"_a, "y_local"_a,
           "kokkoskernels"_a = true, "overlap"_a = true);

    m.def("cg_solve",
        [](const DistCrsHandle& A_dist, f64_1d_gpuc b_nb, f64_1d_gpu x_nb,
           double tol, int max_iter) -> nb::dict {
            ViewVec1D b_view(const_cast<double*>(b_nb.data()), b_nb.shape(0));
            ViewVec1D x_view(x_nb.data(), x_nb.shape(0));
            CGResult res = cg_solve_distributed(A_dist, b_view, x_view, tol,
                                                max_iter, MPIContext::instance().comm());
            nb::dict d;
            d["iters"]     = res.iters;
            d["residual"]  = res.final_res;
            d["converged"] = res.converged;
            return d;
        }, "A_dist"_a, "b"_a, "x"_a, "tol"_a = 1e-10, "max_iter"_a = 1000);

    m.def("cg_solve_local",
        [](const CrsMatrixHandle& A, f64_1d_gpuc b_nb, f64_1d_gpu x_nb,
           double tol, int max_iter) -> nb::dict {
            ViewVec1D b_view(const_cast<double*>(b_nb.data()), b_nb.shape(0));
            ViewVec1D x_view(x_nb.data(), x_nb.shape(0));
            CGResult res = cg_solve_local(A, b_view, x_view, tol, max_iter);
            nb::dict d;
            d["iters"]     = res.iters;
            d["residual"]  = res.final_res;
            d["converged"] = res.converged;
            return d;
        }, "A"_a, "b"_a, "x"_a, "tol"_a = 1e-10, "max_iter"_a = 1000);
#endif
}
