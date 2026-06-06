#pragma once
#include "crs_matrix.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

// ─── Device-side vector operations ───────────────────────────────────────────
//
//  All kernels operate on Kokkos::View<Scalar*, MemSpace>.
//  They are thin wrappers to make the CG solver readable.
//

/**
 * axpy: z = alpha*x + y
 * If z aliases x or y, the result is still correct (single-pass write).
 */
template <class Exec = ExecSpace>
void vec_axpy(Scalar alpha,
              const Kokkos::View<const Scalar*, typename Exec::memory_space>& x,
              const Kokkos::View<const Scalar*, typename Exec::memory_space>& y,
              Kokkos::View<Scalar*, typename Exec::memory_space>&              z)
{
    const int n = x.extent(0);
    Kokkos::parallel_for("axpy", Kokkos::RangePolicy<Exec>(0, n),
        KOKKOS_LAMBDA(const int i) { z(i) = alpha * x(i) + y(i); });
    Kokkos::fence("axpy_fence");
}

/**
 * axpy2: z = alpha*x + beta*y   (general two-coefficient form used in CG p update)
 */
template <class Exec = ExecSpace>
void vec_axpy2(Scalar alpha,
               const Kokkos::View<const Scalar*, typename Exec::memory_space>& x,
               Scalar beta,
               const Kokkos::View<const Scalar*, typename Exec::memory_space>& y,
               Kokkos::View<Scalar*, typename Exec::memory_space>&              z)
{
    const int n = x.extent(0);
    Kokkos::parallel_for("axpy2", Kokkos::RangePolicy<Exec>(0, n),
        KOKKOS_LAMBDA(const int i) { z(i) = alpha * x(i) + beta * y(i); });
    Kokkos::fence("axpy2_fence");
}

/**
 * dot: returns sum_i x[i]*y[i]  (local, device-only)
 */
template <class Exec = ExecSpace>
Scalar vec_local_dot(
    const Kokkos::View<const Scalar*, typename Exec::memory_space>& x,
    const Kokkos::View<const Scalar*, typename Exec::memory_space>& y)
{
    const int n = x.extent(0);
    Scalar    result = 0.0;
    Kokkos::parallel_reduce("local_dot", Kokkos::RangePolicy<Exec>(0, n),
        KOKKOS_LAMBDA(const int i, Scalar& acc) { acc += x(i) * y(i); },
        result);
    Kokkos::fence("local_dot_fence");
    return result;
}

/**
 * norm2: ||x||_2  (local)
 */
template <class Exec = ExecSpace>
Scalar vec_local_norm2(
    const Kokkos::View<const Scalar*, typename Exec::memory_space>& x)
{
    return std::sqrt(vec_local_dot<Exec>(x, x));
}

/**
 * copy: dst = src
 */
template <class Exec = ExecSpace>
void vec_copy(const Kokkos::View<const Scalar*, typename Exec::memory_space>& src,
              Kokkos::View<Scalar*, typename Exec::memory_space>&              dst)
{
    Kokkos::deep_copy(dst, src);
}

/**
 * fill: x[i] = val  for all i
 */
template <class Exec = ExecSpace>
void vec_fill(Kokkos::View<Scalar*, typename Exec::memory_space>& x, Scalar val)
{
    Kokkos::deep_copy(x, val);
}

/**
 * scale: x *= alpha
 */
template <class Exec = ExecSpace>
void vec_scale(Kokkos::View<Scalar*, typename Exec::memory_space>& x, Scalar alpha)
{
    const int n = x.extent(0);
    Kokkos::parallel_for("scale", Kokkos::RangePolicy<Exec>(0, n),
        KOKKOS_LAMBDA(const int i) { x(i) *= alpha; });
    Kokkos::fence("scale_fence");
}

/**
 * global_dot — MPI_Allreduce of local dot product.
 * Returns the same value on every rank.
 */
template <class Exec = ExecSpace>
Scalar global_dot(
    const Kokkos::View<const Scalar*, typename Exec::memory_space>& x,
    const Kokkos::View<const Scalar*, typename Exec::memory_space>& y,
    MPI_Comm comm)
{
    Scalar local_val  = vec_local_dot<Exec>(x, y);
    Scalar global_val = 0.0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_val;
}

/**
 * global_norm2 — ||x||_2 across all MPI ranks.
 */
template <class Exec = ExecSpace>
Scalar global_norm2(
    const Kokkos::View<const Scalar*, typename Exec::memory_space>& x,
    MPI_Comm comm)
{
    return std::sqrt(global_dot<Exec>(x, x, comm));
}
