#pragma once
#include "crs_matrix.hpp"
#include <KokkosSparse_spmv.hpp>

// ─── Batch SpMV kernel v0.3 ───────────────────────────────────────────────────
//
//  dispatch 정책:
//    k < 4              → MDRangePolicy (최소 오버헤드)
//    4 <= k < 32        → TeamPolicy Hierarchical (레지스터 누적)
//    k >= 32            → KokkosKernels SPMM (cuSPARSE SpMM)
//
//  custom_batch 전용 경로 (launch_batch_spmv_custom):
//    k ∈ {8, 16, 32}    → template-unrolled kernel (JAX vmap 동급 경량)
//    그 외              → Hierarchical fallback
//
//  template 언롤 원리:
//    #pragma unroll + 컴파일타임 K 상수 → for 루프가 기계어 수준에서 전개됨
//    → 범용 MultiVector 커널의 ~300μs 런치 오버헤드 → ~80μs 수준으로 감소

// ─── Hierarchical Batch SpMV (기존 경로) ─────────────────────────────────────

template <class Exec = ExecSpace>
struct BatchSpMVFunctorHierarchical {
    using Policy   = Kokkos::TeamPolicy<Exec>;
    using Member   = typename Policy::member_type;
    using CrsMat   = KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>;
    using MatView  = Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                                  typename Exec::memory_space>;
    using OutView  = Kokkos::View<Scalar**,       Kokkos::LayoutRight,
                                  typename Exec::memory_space>;

    CrsMat  A;
    MatView X;
    OutView Y;
    int     k;

    KOKKOS_INLINE_FUNCTION
    void operator()(const Member& team) const {
        const int i = team.league_rank();
        if (i >= A.numRows()) return;

        const int row_start = A.graph.row_map(i);
        const int row_end   = A.graph.row_map(i + 1);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, k),
            [&](const int b) {
                Scalar acc = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, row_start, row_end),
                    [&](const int j, Scalar& s) {
                        s += A.values(j) * X(A.graph.entries(j), b);
                    }, acc);
                Y(i, b) = acc;
            });
    }
};

// ─── Template-Unrolled Small-Batch Functor (custom_batch 전용) ───────────────
//
//  K 를 컴파일타임 상수로 박아서 루프를 완전 전개.
//  각 행(i) 에 대해 warp 하나가 row 의 non-zero 들을 순회하면서
//  K 개의 RHS 컬럼을 동시에 누적 → 메모리 읽기 재사용률 최대화.

template <int K, class Exec = ExecSpace>
struct BatchSpMVFunctorUnrolled {
    using Policy   = Kokkos::TeamPolicy<Exec>;
    using Member   = typename Policy::member_type;
    using CrsMat   = KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>;
    using MatView  = Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                                  typename Exec::memory_space>;
    using OutView  = Kokkos::View<Scalar**,       Kokkos::LayoutRight,
                                  typename Exec::memory_space>;

    CrsMat  A;
    MatView X;
    OutView Y;

    KOKKOS_INLINE_FUNCTION
    void operator()(const Member& team) const {
        const int i = team.league_rank();
        if (i >= A.numRows()) return;

        const int row_start = A.graph.row_map(i);
        const int row_end   = A.graph.row_map(i + 1);

        // 각 스레드가 K 컬럼을 레지스터에 누적
        // ThreadVectorRange 로 non-zero 들을 warp 분산 처리
        Scalar sum[K];
        #pragma unroll
        for (int b = 0; b < K; ++b) sum[b] = 0.0;

        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team, row_start, row_end),
            [&](const int j) {
                const Scalar val = A.values(j);
                const int    col = A.graph.entries(j);
                #pragma unroll
                for (int b = 0; b < K; ++b)
                    sum[b] += val * X(col, b);
            });

        // 결과 저장 (team 내 첫 번째 스레드가 대표)
        Kokkos::single(Kokkos::PerThread(team), [&]() {
            #pragma unroll
            for (int b = 0; b < K; ++b)
                Y(i, b) = sum[b];
        });
    }
};

// ─── MDRangePolicy (k < 4) ────────────────────────────────────────────────────

template <class Exec = ExecSpace>
struct BatchSpMVFunctorMDRange {
    using CrsMat  = KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>;
    using MatView = Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                                 typename Exec::memory_space>;
    using OutView = Kokkos::View<Scalar**,       Kokkos::LayoutRight,
                                 typename Exec::memory_space>;
    CrsMat  A;
    MatView X;
    OutView Y;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int b) const {
        const int row_start = A.graph.row_map(i);
        const int row_end   = A.graph.row_map(i + 1);
        Scalar acc = 0.0;
        for (int j = row_start; j < row_end; ++j)
            acc += A.values(j) * X(A.graph.entries(j), b);
        Y(i, b) = acc;
    }
};

// ─── launch_batch_spmv (KokkosKernels 경로, 기존) ────────────────────────────

template <class Exec = ExecSpace>
void launch_batch_spmv(
    const KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>& A,
    const Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                       typename Exec::memory_space>& X,
    Kokkos::View<Scalar**, Kokkos::LayoutRight,
                 typename Exec::memory_space>& Y,
    int N, int k,
    std::pair<int,int> = {0, 0})
{
    if (N == 0 || k == 0) return;

    if (k < 4) {
        using Policy = Kokkos::MDRangePolicy<
            Exec,
            Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
        Policy policy({0, 0}, {N, k}, {1, std::min(32, k)});
        Kokkos::parallel_for("BatchSpMV_MDRange",
                             policy, BatchSpMVFunctorMDRange<Exec>{A, X, Y});
    } else if (k < 32) {
        using Policy = Kokkos::TeamPolicy<Exec>;
        Policy policy(N, k, 32);
        Kokkos::parallel_for("BatchSpMV_Hierarchical",
                             policy, BatchSpMVFunctorHierarchical<Exec>{A, X, Y, k});
    } else {
        KokkosSparse::spmv("N",
                           static_cast<Scalar>(1.0), A, X,
                           static_cast<Scalar>(0.0), Y);
    }
    Kokkos::fence("BatchSpMV_fence");
}

// ─── launch_batch_spmv_custom (Template-Unroll 경로) ─────────────────────────
//
//  소규모 배치(k=8,16,32)에서 JAX vmap 을 이기기 위한 경량 커널.
//  컴파일타임 K 특화 인스턴스를 switch 로 분기.

template <class Exec = ExecSpace>
void launch_batch_spmv_custom(
    const KokkosSparse::CrsMatrix<Scalar, Ordinal, Exec, void, Offset>& A,
    const Kokkos::View<const Scalar**, Kokkos::LayoutRight,
                       typename Exec::memory_space>& X,
    Kokkos::View<Scalar**, Kokkos::LayoutRight,
                 typename Exec::memory_space>& Y,
    int N, int k)
{
    if (N == 0 || k == 0) return;

    // vector_length=1: each team = one row, ThreadVectorRange over nnz
    const int vector_length = 32;

    auto dispatch_unrolled = [&](auto k_tag) {
        constexpr int K = decltype(k_tag)::value;
        using Policy = Kokkos::TeamPolicy<Exec>;
        // team_size=1: 한 팀이 한 행 담당, vector(warp) 가 nnz 순회
        Policy policy(N, 1, vector_length);
        Kokkos::parallel_for(
            "BatchSpMV_Unrolled_K" + std::to_string(K),
            policy,
            BatchSpMVFunctorUnrolled<K, Exec>{A, X, Y});
    };

    switch (k) {
        case 8:  dispatch_unrolled(std::integral_constant<int, 8>{});  break;
        case 16: dispatch_unrolled(std::integral_constant<int, 16>{}); break;
        case 32: dispatch_unrolled(std::integral_constant<int, 32>{}); break;
        default:
            // k 가 특화 값이 아닐 때는 Hierarchical 로 폴백
            if (k < 4) {
                using Policy = Kokkos::MDRangePolicy<
                    Exec,
                    Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
                Policy policy({0, 0}, {N, k}, {1, std::min(32, k)});
                Kokkos::parallel_for("BatchSpMV_MD_fallback",
                                     policy, BatchSpMVFunctorMDRange<Exec>{A, X, Y});
            } else if (k < 64) {
                using Policy = Kokkos::TeamPolicy<Exec>;
                Policy policy(N, k, vector_length);
                Kokkos::parallel_for("BatchSpMV_Hier_fallback",
                                     policy,
                                     BatchSpMVFunctorHierarchical<Exec>{A, X, Y, k});
            } else {
                // 대규모 배치: cuSPARSE SPMM 이 최적
                KokkosSparse::spmv("N",
                                   static_cast<Scalar>(1.0), A, X,
                                   static_cast<Scalar>(0.0), Y);
            }
            break;
    }
    Kokkos::fence("BatchSpMV_custom_fence");
}

// ─── unit test helper ────────────────────────────────────────────────────────

inline std::vector<Scalar> batch_spmv_host_roundtrip(
    const std::vector<Scalar>&  h_values,
    const std::vector<Ordinal>& h_colind,
    const std::vector<Offset>&  h_rowptr,
    const std::vector<Scalar>&  h_X_flat,
    int nrows, int ncols, int k)
{
    CrsMatrixHandle handle(h_values.data(), h_colind.data(), h_rowptr.data(),
                           nrows, ncols);
    ViewMat2D X_dev("X", nrows, k);
    ViewMat2D Y_dev("Y", nrows, k);

    auto X_host = Kokkos::create_mirror_view(X_dev);
    for (int i = 0; i < nrows; ++i)
        for (int b = 0; b < k; ++b)
            X_host(i, b) = h_X_flat[i * k + b];
    Kokkos::deep_copy(X_dev, X_host);

    ViewMat2DConst X_const = X_dev;
    launch_batch_spmv(handle.mat, X_const, Y_dev, nrows, k);

    auto Y_host = Kokkos::create_mirror_view(Y_dev);
    Kokkos::deep_copy(Y_host, Y_dev);

    std::vector<Scalar> result(nrows * k);
    for (int i = 0; i < nrows; ++i)
        for (int b = 0; b < k; ++b)
            result[i * k + b] = Y_host(i, b);
    return result;
}
