// Eigen FFI wrapper for one-pass Lanczos computing exp(-A)b.
// Stores the full Krylov basis V_k (n x k MatrixXd, O(nk) memory).
// Final reconstruction is a single dense GEMV: x = ||b|| * V_k * g.
//
// Templatized on StorageOrder (RowMajor = CSR, ColMajor = CSC).

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr double breakdown_tol =
    std::numeric_limits<double>::epsilon() * 1000.0;

template<int StorageOrder>
struct LanczosOnePassContext {
    const Eigen::Map<const Eigen::SparseMatrix<double, StorageOrder, int32_t>> A;
    const Eigen::VectorXd b;
    const int32_t n;
    const int32_t krylov_dim;

    Eigen::MatrixXd V_k;
    Eigen::VectorXd v_prev;
    Eigen::VectorXd v_curr;
    Eigen::VectorXd work;
    Eigen::VectorXd x;

    Eigen::VectorXd alphas;
    Eigen::VectorXd betas;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
    int32_t steps_taken;

    explicit LanczosOnePassContext(
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
        , V_k(Eigen::MatrixXd::Zero(nrows, std::max<int32_t>(k, 0)))
        , v_prev(Eigen::VectorXd::Zero(nrows))
        , v_curr(Eigen::VectorXd::Zero(nrows))
        , work(Eigen::VectorXd::Zero(nrows))
        , x(Eigen::VectorXd::Zero(nrows))
        , alphas(Eigen::VectorXd::Zero(std::max<int32_t>(k, 0)))
        , betas(Eigen::VectorXd::Zero(std::max<int32_t>(k - 1, 0)))
        , solver(std::max<int32_t>(k, 1))
        , steps_taken(0) {}

    ~LanczosOnePassContext() = default;
};

// One-pass Lanczos for exp(-A)b.
//
// 1) Builds tridiagonal T_m and stores each basis vector v_j into V_k.
//    No reorthogonalization (matches the faer bench configuration).
// 2) Solves g = exp(-T_m)*e_1 via eigendecomposition.
// 3) Reconstructs x = ||b|| * V_k(:,0:m) * g as a dense GEMV.
//
// Idempotent: all state is reset at entry.
template<int StorageOrder>
void execute_impl(LanczosOnePassContext<StorageOrder>* ctx) {
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

    // Store v_1 = b / ||b|| into basis column 0.
    ctx->V_k.col(0) = ctx->v_curr;

    // ---- Lanczos recurrence with basis storage ----
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
        ctx->steps_taken = i + 1;

        if (beta <= breakdown_tol) {
            break;
        }

        if (i < ctx->krylov_dim - 1) {
            ctx->betas[i] = beta;

            ctx->work *= 1.0 / beta;
            ctx->v_prev.swap(ctx->v_curr);
            ctx->v_curr.swap(ctx->work);
            beta_prev = beta;

            // Store the new basis vector after the swap.
            // v_curr now holds the normalized v_{i+2}.
            ctx->V_k.col(i + 1) = ctx->v_curr;
        }
    }

    // ---- exp(-T_m) * e_1 via eigendecomposition ----
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
    const Eigen::VectorXd g = q * weights;

    // ---- Reconstruct x = ||b|| * V_k(:,0:m) * g ----
    // noalias() is correct here: RHS is a Product (MatrixXd * VectorXd).
    ctx->x.noalias() = b_norm * (ctx->V_k.leftCols(m) * g);
}

template<int StorageOrder>
void get_y_impl(LanczosOnePassContext<StorageOrder>* ctx, double* out, int32_t len) {
    if (ctx == nullptr || out == nullptr || len <= 0) {
        return;
    }
    const int32_t count = std::min(len, ctx->n);
    for (int32_t i = 0; i < count; ++i) {
        out[i] = ctx->x[i];
    }
}

using CsrContext = LanczosOnePassContext<Eigen::RowMajor>;
using CscContext = LanczosOnePassContext<Eigen::ColMajor>;

}  // namespace

extern "C" {

// ---- CSR (RowMajor) ----

CsrContext* libeigen_lanczos_setup(
    int32_t nrows, int32_t ncols, int32_t nnz,
    const int32_t* row_ptr, const int32_t* col_idx, const double* values,
    const double* b, int32_t krylov_dim
) {
    return new CsrContext(nrows, ncols, nnz, row_ptr, col_idx, values, b, krylov_dim);
}

void libeigen_lanczos_execute(CsrContext* ctx) {
    execute_impl(ctx);
}

void libeigen_lanczos_get_y(CsrContext* ctx, double* out, int32_t len) {
    get_y_impl(ctx, out, len);
}

void libeigen_lanczos_teardown(CsrContext* ctx) {
    delete ctx;
}

// ---- CSC (ColMajor) ----

CscContext* libeigen_csc_lanczos_setup(
    int32_t nrows, int32_t ncols, int32_t nnz,
    const int32_t* col_ptr, const int32_t* row_idx, const double* values,
    const double* b, int32_t krylov_dim
) {
    return new CscContext(nrows, ncols, nnz, col_ptr, row_idx, values, b, krylov_dim);
}

void libeigen_csc_lanczos_execute(CscContext* ctx) {
    execute_impl(ctx);
}

void libeigen_csc_lanczos_get_y(CscContext* ctx, double* out, int32_t len) {
    get_y_impl(ctx, out, len);
}

void libeigen_csc_lanczos_teardown(CscContext* ctx) {
    delete ctx;
}

}  // extern "C"
