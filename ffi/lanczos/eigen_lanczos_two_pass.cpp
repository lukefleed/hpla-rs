// Eigen FFI wrapper for two-pass Lanczos computing exp(-A)b.
// Zero-copy via Eigen::Map over CSR or CSC arrays. All O(n) buffers are
// pre-allocated in setup; execute reuses them without any n-scaled
// allocation. Header-only: the SpMV and dense kernels are compiled with
// our flags (-O3 -march=native -ffast-math -flto).
//
// The context is templatized on StorageOrder (RowMajor = CSR,
// ColMajor = CSC) so the algorithm code is shared. Thin extern "C"
// wrappers provide the C ABI entry points for each format.

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr double breakdown_tol =
    std::numeric_limits<double>::epsilon() * 1000.0;

// Opaque backend context used by the Rust FFI layer.
//
// Ownership and lifetime:
// - `A` is a non-owning Eigen::Map over caller-owned compressed arrays.
// - `b` is an owned copy so repeated `execute` calls are idempotent.
// - Rolling vectors and tridiagonal scalars are allocated once and reused.
//
// Memory model:
// - O(n) persistent buffers for the two-pass recurrence.
// - O(k) temporaries in `execute` for tridiagonal eigensolver inputs.
template<int StorageOrder>
struct LanczosTwoPassContext {
    const Eigen::Map<const Eigen::SparseMatrix<double, StorageOrder, int32_t>> A;
    const Eigen::VectorXd b;
    const int32_t n;
    const int32_t krylov_dim;

    Eigen::VectorXd v_prev;
    Eigen::VectorXd v_curr;
    Eigen::VectorXd work;
    Eigen::VectorXd x;

    Eigen::VectorXd alphas;
    Eigen::VectorXd betas;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
    int32_t steps_taken;

    explicit LanczosTwoPassContext(
        int32_t nrows,
        int32_t ncols,
        int32_t nnz,
        const int32_t* outer_ptr,
        const int32_t* inner_idx,
        const double* values,
        const double* b_ptr,
        int32_t k
    )
        : A(nrows, ncols, nnz, outer_ptr, inner_idx, values)
        , b(Eigen::Map<const Eigen::VectorXd>(b_ptr, nrows))
        , n(nrows)
        , krylov_dim(k)
        , v_prev(Eigen::VectorXd::Zero(nrows))
        , v_curr(Eigen::VectorXd::Zero(nrows))
        , work(Eigen::VectorXd::Zero(nrows))
        , x(Eigen::VectorXd::Zero(nrows))
        , alphas(Eigen::VectorXd::Zero(std::max<int32_t>(k, 0)))
        , betas(Eigen::VectorXd::Zero(std::max<int32_t>(k - 1, 0)))
        , solver(std::max<int32_t>(k, 1))
        , steps_taken(0) {}

    ~LanczosTwoPassContext() = default;
};

// Executes the full two-pass Lanczos kernel for exp(-A)b.
//
// Algorithm:
// 1) Pass 1: builds the tridiagonal recurrence coefficients.
// 2) Solves exp(-T_m)e1 via eigendecomposition of the tridiagonal.
// 3) Pass 2: replays Lanczos vectors and reconstructs x = V_m * g.
//
// The routine is idempotent: all rolling state is reset at entry.
template<int StorageOrder>
void execute_impl(LanczosTwoPassContext<StorageOrder>* ctx) {
    if (ctx == nullptr || ctx->n <= 0 || ctx->krylov_dim <= 0) {
        if (ctx != nullptr) {
            ctx->x.setZero();
            ctx->steps_taken = 0;
        }
        return;
    }

    const double b_norm = ctx->b.norm();
    if (b_norm <= breakdown_tol) {
        ctx->x.setZero();
        ctx->steps_taken = 0;
        return;
    }
    const double inv_b_norm = 1.0 / b_norm;

    ctx->v_prev.setZero();
    ctx->v_curr = ctx->b * inv_b_norm;
    ctx->work.setZero();
    ctx->x.setZero();
    ctx->alphas.setZero();
    ctx->betas.setZero();
    ctx->steps_taken = 0;

    // ---- Pass 1: produce tridiagonal T_k ----
    double beta_prev = 0.0;
    for (int32_t i = 0; i < ctx->krylov_dim; ++i) {
        ctx->work.noalias() = ctx->A * ctx->v_curr;
        if (i > 0) {
            ctx->work -= beta_prev * ctx->v_prev;
        }

        const double alpha = ctx->v_curr.dot(ctx->work);
        ctx->alphas[i] = alpha;

        ctx->work -= alpha * ctx->v_curr;

        const double beta = ctx->work.norm();
        if (beta <= breakdown_tol) {
            ctx->steps_taken = i + 1;
            break;
        }

        if (i < ctx->krylov_dim - 1) {
            ctx->betas[i] = beta;
        }

        ctx->work *= 1.0 / beta;
        ctx->v_prev.swap(ctx->v_curr);
        ctx->v_curr.swap(ctx->work);
        beta_prev = beta;
        ctx->steps_taken = i + 1;
    }

    // ---- exp(-T_k) * e_1 via eigendecomposition ----
    const int32_t m = ctx->steps_taken;
    if (m <= 0) {
        ctx->x.setZero();
        return;
    }

    const Eigen::VectorXd neg_d = -ctx->alphas.head(m);
    const Eigen::VectorXd neg_e =
        (m > 1) ? Eigen::VectorXd(-ctx->betas.head(m - 1))
                 : Eigen::VectorXd();

    ctx->solver.computeFromTridiagonal(neg_d, neg_e);
    if (ctx->solver.info() != Eigen::Success) {
        ctx->x.setZero();
        return;
    }

    const auto& lambda = ctx->solver.eigenvalues();
    const auto& q = ctx->solver.eigenvectors();
    const Eigen::VectorXd weights =
        lambda.array().exp() * q.row(0).transpose().array();
    Eigen::VectorXd g = q * weights;
    g *= b_norm;

    // ---- Pass 2: reconstruct x = V_k * g without V_k ----
    ctx->v_prev.setZero();
    ctx->v_curr = ctx->b * inv_b_norm;
    ctx->x = g[0] * ctx->v_curr;

    for (int32_t j = 0; j < m - 1; ++j) {
        ctx->work.noalias() = ctx->A * ctx->v_curr;
        ctx->work -= ctx->alphas[j] * ctx->v_curr;
        if (j > 0) {
            ctx->work -= ctx->betas[j - 1] * ctx->v_prev;
        }

        const double beta_j = ctx->betas[j];
        if (beta_j <= breakdown_tol) {
            break;
        }
        ctx->work *= 1.0 / beta_j;

        ctx->x += g[j + 1] * ctx->work;

        ctx->v_prev.swap(ctx->v_curr);
        ctx->v_curr.swap(ctx->work);
    }
}

template<int StorageOrder>
void get_y_impl(LanczosTwoPassContext<StorageOrder>* ctx, double* out, int32_t len) {
    if (ctx == nullptr || out == nullptr || len <= 0) {
        return;
    }
    const int32_t count = std::min(len, ctx->n);
    for (int32_t i = 0; i < count; ++i) {
        out[i] = ctx->x[i];
    }
}

using CsrContext = LanczosTwoPassContext<Eigen::RowMajor>;
using CscContext = LanczosTwoPassContext<Eigen::ColMajor>;

}  // namespace

extern "C" {

// ---- CSR (RowMajor) ----

CsrContext* libeigen_lanczos_two_pass_setup(
    int32_t nrows, int32_t ncols, int32_t nnz,
    const int32_t* row_ptr, const int32_t* col_idx, const double* values,
    const double* b, int32_t krylov_dim
) {
    return new CsrContext(nrows, ncols, nnz, row_ptr, col_idx, values, b, krylov_dim);
}

void libeigen_lanczos_two_pass_execute(CsrContext* ctx) {
    execute_impl(ctx);
}

void libeigen_lanczos_two_pass_get_y(CsrContext* ctx, double* out, int32_t len) {
    get_y_impl(ctx, out, len);
}

void libeigen_lanczos_two_pass_teardown(CsrContext* ctx) {
    delete ctx;
}

// ---- CSC (ColMajor) ----

CscContext* libeigen_csc_lanczos_two_pass_setup(
    int32_t nrows, int32_t ncols, int32_t nnz,
    const int32_t* col_ptr, const int32_t* row_idx, const double* values,
    const double* b, int32_t krylov_dim
) {
    return new CscContext(nrows, ncols, nnz, col_ptr, row_idx, values, b, krylov_dim);
}

void libeigen_csc_lanczos_two_pass_execute(CscContext* ctx) {
    execute_impl(ctx);
}

void libeigen_csc_lanczos_two_pass_get_y(CscContext* ctx, double* out, int32_t len) {
    get_y_impl(ctx, out, len);
}

void libeigen_csc_lanczos_two_pass_teardown(CscContext* ctx) {
    delete ctx;
}

}  // extern "C"
