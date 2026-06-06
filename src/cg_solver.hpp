#pragma once
#include "dist_spmv.hpp"
#include "vec_ops.hpp"
#include <cmath>
#include <stdexcept>

// ─── CG Solver ───────────────────────────────────────────────────────────────
//
//  Solves  A x = b  where A is symmetric positive definite.
//
//  Algorithm: standard Conjugate Gradient (Hestenes & Stiefel, 1952).
//
//  Communication cost per iteration:
//    - 1 distributed_spmv  : halo exchange  (O(√N) data, neighbour-only)
//    - 2-3 MPI_Allreduce   : global dot products (1 scalar, all-to-all)
//
//  The allreduce dominates at large GPU counts because it is latency-bound
//  (not bandwidth-bound).  This behaviour is intentionally measured and
//  exposed in the benchmark suite.
//

struct CGResult {
    int    iters;       ///< Number of iterations performed
    double final_res;   ///< ||r||_2 / ||b||_2 at termination
    bool   converged;   ///< Did we reach the tolerance?
};

/**
 * cg_solve_distributed
 *
 * @param A_dist   Distributed CSR matrix (already partitioned)
 * @param b_local  Local RHS portion (device memory, length = local_rows)
 * @param x_local  Initial guess / output (device memory, same size)
 * @param tol      Relative residual tolerance (default 1e-10)
 * @param max_iter Maximum iterations (default 1000)
 * @param comm     MPI communicator
 */
CGResult cg_solve_distributed(
    const DistCrsHandle& A_dist,
    const ViewVec1D&     b_local,
    ViewVec1D&           x_local,
    double               tol      = 1e-10,
    int                  max_iter = 1000,
    MPI_Comm             comm     = MPI_COMM_WORLD)
{
    const int n = b_local.extent(0);

    // Allocate working vectors
    ViewVec1D r ("r",  n);   // residual
    ViewVec1D p ("p",  n);   // search direction
    ViewVec1D Ap("Ap", n);   // A*p

    // r = b - A*x0
    distributed_spmv(A_dist, x_local, Ap, comm);          // Ap = A*x0
    // r = b - Ap
    {
        ViewVec1DConst b_const  = b_local;
        ViewVec1DConst Ap_const = Ap;
        vec_axpy<ExecSpace>(-1.0, Ap_const, b_const, r);  // r = -Ap + b
    }

    // p = r
    Kokkos::deep_copy(p, r);

    // Initial norms
    double b_norm2   = global_norm2<ExecSpace>(b_local, comm);
    if (b_norm2 == 0.0) b_norm2 = 1.0;  // avoid divide-by-zero for zero RHS

    ViewVec1DConst r_const = r;
    double r_dot = global_dot<ExecSpace>(r_const, r_const, comm);  // (r,r)

    // Early exit if already converged
    if (std::sqrt(r_dot) / b_norm2 < tol)
        return {0, std::sqrt(r_dot) / b_norm2, true};

    for (int iter = 0; iter < max_iter; ++iter) {
        // Ap = A * p  (halo exchange + local SpMV)
        distributed_spmv(A_dist, p, Ap, comm);

        // alpha = (r,r) / (p, Ap)
        ViewVec1DConst p_const  = p;
        ViewVec1DConst Ap_const = Ap;
        double pAp   = global_dot<ExecSpace>(p_const, Ap_const, comm);
        if (std::abs(pAp) < 1e-300)
            throw std::runtime_error("CG: p'Ap is near zero — A may not be SPD");
        double alpha = r_dot / pAp;

        // x = x + alpha*p
        {
            ViewVec1DConst p_c = p;
            ViewVec1DConst x_c = x_local;
            vec_axpy<ExecSpace>(alpha, p_c, x_c, x_local);
        }

        // r = r - alpha*Ap
        {
            ViewVec1DConst Ap_c = Ap;
            ViewVec1DConst r_c  = r;
            vec_axpy<ExecSpace>(-alpha, Ap_c, r_c, r);
        }

        // r_dot_new = (r_new, r_new)
        ViewVec1DConst r_c2 = r;
        double r_dot_new = global_dot<ExecSpace>(r_c2, r_c2, comm);

        double rel_res = std::sqrt(r_dot_new) / b_norm2;
        if (rel_res < tol)
            return {iter + 1, rel_res, true};

        // beta = (r_new, r_new) / (r_old, r_old)
        double beta = r_dot_new / r_dot;
        r_dot       = r_dot_new;

        // p = r + beta*p
        {
            ViewVec1DConst r_c = r;
            ViewVec1DConst p_c = p;
            vec_axpy2<ExecSpace>(1.0, r_c, beta, p_c, p);
        }
    }

    ViewVec1DConst r_final = r;
    double final_dot = global_dot<ExecSpace>(r_final, r_final, comm);
    return {max_iter, std::sqrt(final_dot) / b_norm2, false};
}

/**
 * cg_solve_local
 *
 * Single-GPU (single-rank) CG solve.  No MPI, no halo exchange.
 * Useful for benchmarking and unit tests.
 */
CGResult cg_solve_local(
    const CrsMatrixHandle& A,
    const ViewVec1D&       b,
    ViewVec1D&             x,
    double                 tol      = 1e-10,
    int                    max_iter = 1000)
{
    const int n = b.extent(0);
    ViewVec1D r("r", n), p("p", n), Ap("Ap", n);

    // r = b - A*x0
    {
        ViewVec1DConst x_c = x;
        launch_spmv_kokkoskernels(A.mat, x_c, Ap);        // Ap = A*x
        ViewVec1DConst b_c = b, Ap_c = Ap;
        vec_axpy<ExecSpace>(-1.0, Ap_c, b_c, r);          // r = b - Ap
    }
    Kokkos::deep_copy(p, r);

    double b_norm2 = vec_local_norm2<ExecSpace>(b);
    if (b_norm2 == 0.0) b_norm2 = 1.0;

    ViewVec1DConst r_c0 = r;
    double r_dot = vec_local_dot<ExecSpace>(r_c0, r_c0);

    if (std::sqrt(r_dot) / b_norm2 < tol)
        return {0, std::sqrt(r_dot) / b_norm2, true};

    for (int iter = 0; iter < max_iter; ++iter) {
        ViewVec1DConst p_c = p;
        launch_spmv_kokkoskernels(A.mat, p_c, Ap);        // Ap = A*p

        ViewVec1DConst Ap_c = Ap;
        double pAp  = vec_local_dot<ExecSpace>(p_c, Ap_c);
        double alpha = r_dot / pAp;

        {
            ViewVec1DConst p_c2 = p, x_c = x;
            vec_axpy<ExecSpace>(alpha, p_c2, x_c, x);      // x += alpha*p
        }
        {
            ViewVec1DConst Ap_c2 = Ap, r_c = r;
            vec_axpy<ExecSpace>(-alpha, Ap_c2, r_c, r);    // r -= alpha*Ap
        }

        ViewVec1DConst r_c = r;
        double r_dot_new = vec_local_dot<ExecSpace>(r_c, r_c);
        double rel_res   = std::sqrt(r_dot_new) / b_norm2;

        if (rel_res < tol)
            return {iter + 1, rel_res, true};

        double beta = r_dot_new / r_dot;
        r_dot       = r_dot_new;

        {
            ViewVec1DConst r_c2 = r, p_c2 = p;
            vec_axpy2<ExecSpace>(1.0, r_c2, beta, p_c2, p); // p = r + beta*p
        }
    }

    ViewVec1DConst r_f = r;
    return {max_iter, std::sqrt(vec_local_dot<ExecSpace>(r_f, r_f)) / b_norm2, false};
}
