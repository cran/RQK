#include <RcppArmadillo.h>
#include <nloptrAPI.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kdtree.h"

// [[Rcpp::depends(RcppArmadillo, nloptr)]]

namespace {

const double kPenalty = 1e20;
const double kLocalULogRadius = 3.0;
const double kUMinBound = -3.0;
const double kUMaxBound = 3.0;
const int kLocalMaxeval = 100;
const std::size_t kLineSearchPoints1D = 10;
const std::size_t kLineSearchPoints2D = 5;
constexpr std::size_t kLoocvPairwiseCacheMaxBytes = 512ull * 1024ull * 1024ull;

void check_nlopt_result(nlopt_result res, const char* context) {
  if (res < 0) {
    std::ostringstream oss;
    oss << context << " (nlopt status: " << static_cast<int>(res) << ")";
    throw std::runtime_error(oss.str());
  }
}

void check_nlopt_optimize_result(nlopt_result res, const char* context) {
  if (res == NLOPT_ROUNDOFF_LIMITED) {
    return;
  }
  check_nlopt_result(res, context);
}

struct KNNResult {
  arma::umat idx;
  arma::mat dist;
};

struct ModelStats {
  arma::mat R;
  arma::vec d;
  arma::vec Rd;
  arma::vec dInv;
  arma::vec v;
  arma::vec b;
  double mu;
  double nu2;
  arma::vec loo_pred;
  arma::vec scores;
  double loocv_mse;
};

struct PredictCache {
  arma::vec Rd;
  arma::vec dInv;
  arma::vec v;
  arma::vec b;
  arma::mat Xw_train;
};

struct CholCacheVectors {
  arma::vec v;
  arma::vec b;
  arma::vec dInv;
};

std::size_t clamp_self_knn_count(std::size_t requested, std::size_t n) {
  if (requested == 0) {
    throw std::runtime_error("kNN requires at least one neighbor");
  }
  if (n <= 1) {
    throw std::runtime_error("kNN requires at least two rows");
  }
  return std::min(requested, n - 1);
}

arma::vec softmax(const arma::vec& u) {
  arma::vec z = u - u.max();
  arma::vec ez = arma::exp(z);
  const double s = arma::sum(ez);
  if (!(s > 0.0) || !std::isfinite(s)) {
    throw std::runtime_error("softmax normalization failed");
  }
  return ez / s;
}

Rcpp::NumericVector as_r_vector(const arma::vec& x) {
  Rcpp::NumericVector out(x.n_elem);
  std::copy(x.begin(), x.end(), out.begin());
  return out;
}

arma::vec softmax_grad(const arma::vec& grad_w, const arma::vec& w) {
  return w % (grad_w - arma::dot(grad_w, w));
}

arma::mat weight_matrix(const arma::mat& X, const arma::vec& w) {
  arma::mat Xw = X;
  for (arma::uword j = 0; j < X.n_cols; ++j) {
    Xw.col(j) *= std::sqrt(w[j]);
  }
  return Xw;
}

arma::mat pairwise_dist2(const arma::mat& X) {
  const arma::uword n = X.n_rows;
  const arma::vec s = arma::sum(arma::square(X), 1);
  arma::mat D2 = arma::repmat(s, 1, n) + arma::repmat(s.t(), n, 1) - 2.0 * (X * X.t());
  D2.transform([](double v) { return v < 0.0 ? 0.0 : v; });
  D2.diag().zeros();
  return D2;
}

arma::cube pairwise_dist2_by_feature(const arma::mat& X) {
  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  arma::cube D2_feat(n, n, p, arma::fill::zeros);

  for (arma::uword j = 0; j < p; ++j) {
    const arma::vec xj = X.col(j);
    const arma::vec sj = arma::square(xj);
    arma::mat D2j = arma::repmat(sj, 1, n) + arma::repmat(sj.t(), n, 1) - 2.0 * (xj * xj.t());
    D2j.transform([](double v) { return v < 0.0 ? 0.0 : v; });
    D2j.diag().zeros();
    D2_feat.slice(j) = std::move(D2j);
  }

  return D2_feat;
}

arma::mat weighted_pairwise_dist2_from_cache(const arma::cube& D2_feat, const arma::vec& w) {
  if (D2_feat.n_slices != w.n_elem) {
    throw std::runtime_error("weighted_pairwise_dist2_from_cache dimension mismatch");
  }

  arma::mat D2(D2_feat.n_rows, D2_feat.n_cols, arma::fill::zeros);
  for (arma::uword j = 0; j < w.n_elem; ++j) {
    const double wj = w[j];
    if (wj != 0.0) {
      D2 += wj * D2_feat.slice(j);
    }
  }
  D2.diag().zeros();
  return D2;
}

bool should_build_loocv_pairwise_cache(const arma::mat& X) {
  const long double n = static_cast<long double>(X.n_rows);
  const long double p = static_cast<long double>(X.n_cols);
  const long double bytes = n * n * p * static_cast<long double>(sizeof(double));
  return bytes <= static_cast<long double>(kLoocvPairwiseCacheMaxBytes);
}

arma::mat cross_dist2(const arma::mat& A, const arma::mat& B) {
  const arma::uword m = A.n_rows;
  const arma::uword n = B.n_rows;
  const arma::uword p = A.n_cols;

  if (B.n_cols != p) {
    throw std::runtime_error("cross_dist2 dimension mismatch");
  }

  const arma::vec sa = arma::sum(arma::square(A), 1);
  const arma::vec sb = arma::sum(arma::square(B), 1);
  arma::mat D2 = arma::repmat(sa, 1, n) + arma::repmat(sb.t(), m, 1) - 2.0 * (A * B.t());
  D2.transform([](double v) { return v < 0.0 ? 0.0 : v; });
  return D2;
}

arma::mat kernel_from_dist2(const arma::mat& D2, double theta, double q) {
  if (!(theta > 0.0) || !std::isfinite(theta)) {
    throw std::runtime_error("theta must be positive");
  }
  if (!(q > 0.0) || !std::isfinite(q)) {
    throw std::runtime_error("q must be positive");
  }
  arma::mat G = 1.0 + D2 / (theta * theta);
  G = arma::pow(G, -q);
  return G;
}

arma::vec compute_base_d(const arma::mat& R) {
  arma::vec denom = arma::sum(R, 1) - R.diag();
  arma::vec d_base = 1.0 / denom;
  return d_base;
}

bool is_finite_vec(const arma::vec& x) {
  for (double v : x) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

bool is_finite_mat(const arma::mat& X) {
  for (double v : X) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

arma::mat chol_solve(const arma::mat& U, const arma::mat& B) {
  arma::mat Y = arma::solve(arma::trimatl(U.t()), B, arma::solve_opts::fast);
  arma::mat X = arma::solve(arma::trimatu(U), Y, arma::solve_opts::fast);
  return X;
}

CholCacheVectors chol_cache_vectors(const arma::mat& U, const arma::mat& R,
                                    const arma::vec& y, const arma::vec& rhs_vec) {
  const arma::uword n = U.n_rows;
  if (U.n_cols != n || R.n_rows != n || R.n_cols != n || y.n_elem != n || rhs_vec.n_elem != n) {
    throw std::runtime_error("chol_cache_vectors dimension mismatch");
  }

  arma::mat C_y = R;
  C_y.each_row() %= y.t();

  arma::mat rhs(n, 2 * n + 1, arma::fill::zeros);
  rhs.col(0) = rhs_vec;
  rhs.cols(1, n) = C_y.t();
  rhs.cols(n + 1, 2 * n) = arma::eye<arma::mat>(n, n);

  arma::mat Y = arma::solve(arma::trimatl(U.t()), rhs, arma::solve_opts::fast);
  const arma::vec dInv = arma::sum(arma::square(Y.cols(n + 1, 2 * n)), 0).t();
  arma::mat X = arma::solve(arma::trimatu(U), Y.cols(0, n), arma::solve_opts::fast);

  return {X.col(0), X.cols(1, n).diag(), dInv};
}

KNNResult get_knn_self(const arma::mat& X, std::size_t k) {
  const std::size_t n = X.n_rows;
  const std::size_t p = X.n_cols;
  const std::size_t effective_k = clamp_self_knn_count(k, n);

  std::vector<double> data_row_major(n * p);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < p; ++j) {
      data_row_major[i * p + j] = X(i, j);
    }
  }

  rqk::KDTree tree(data_row_major, n, p, 16);

  arma::umat idx(n, effective_k, arma::fill::zeros);
  arma::mat dist(n, effective_k, arma::fill::zeros);

  for (std::size_t i = 0; i < n; ++i) {
    const double* query = &data_row_major[i * p];
    std::vector<std::size_t> nn_idx;
    std::vector<double> nn_d2;
    tree.knn_search(query, effective_k, nn_idx, nn_d2, true, i);

    if (nn_idx.size() != effective_k) {
      throw std::runtime_error("kNN search returned fewer neighbors than requested");
    }

    for (std::size_t j = 0; j < effective_k; ++j) {
      idx(i, j) = static_cast<arma::uword>(nn_idx[j]);
      dist(i, j) = std::sqrt(std::max(0.0, nn_d2[j]));
    }
  }

  return {idx, dist};
}

double max_range(const arma::mat& X) {
  double out = 0.0;
  for (arma::uword j = 0; j < X.n_cols; ++j) {
    const double r = X.col(j).max() - X.col(j).min();
    if (r > out) {
      out = r;
    }
  }
  return out;
}

double default_kernel_exponent(int p, bool noise) {
  return noise ? 0.5 * static_cast<double>(p) : 0.5 * static_cast<double>(p + 1);
}

double resolve_kernel_exponent(int p, bool noise, double q_input) {
  if (Rcpp::NumericVector::is_na(q_input)) {
    return default_kernel_exponent(p, noise);
  }
  if (!(q_input > 0.0) || !std::isfinite(q_input)) {
    throw std::runtime_error("q must be finite and positive");
  }
  return q_input;
}

ModelStats compute_model_stats(const arma::mat& X, const arma::vec& y, int p, double q,
                               double theta, double eta, const arma::vec& w,
                               const arma::cube* pairwise_d2_cache = nullptr) {
  const arma::uword n = X.n_rows;
  if (n <= 1) {
    throw std::runtime_error("n must be > 1");
  }

  arma::mat D2;
  if (pairwise_d2_cache != nullptr) {
    D2 = weighted_pairwise_dist2_from_cache(*pairwise_d2_cache, w);
  } else {
    const arma::mat Xw = weight_matrix(X, w);
    D2 = pairwise_dist2(Xw);
  }

  arma::mat R = kernel_from_dist2(D2, theta, q);
  R.diag() += eta;

  const arma::vec d_base = compute_base_d(R);
  if (!is_finite_vec(d_base)) {
    throw std::runtime_error("base d vector is not finite");
  }

  const double theta2q = std::pow(theta, 2.0 * q);
  const double theta4q = theta2q * theta2q;
  if (!(theta2q > 0.0) || !std::isfinite(theta4q)) {
    throw std::runtime_error("theta scaling is not finite");
  }

  const arma::vec d = theta2q * d_base;
  const arma::vec Rd = R * d;
  const double mu_num = arma::dot(d, Rd % y);
  const double mu_den = arma::dot(d, Rd);
  const double mu = mu_num / mu_den;

  const double nu2 = 1.0;

  arma::mat U;
  if (!arma::chol(U, R, "upper")) {
    throw std::runtime_error("Cholesky failed");
  }

  const arma::vec tvec = Rd % y;
  const CholCacheVectors cache_vecs = chol_cache_vectors(U, R, y, tvec);
  const arma::vec v = cache_vecs.v;
  const arma::vec dInv = cache_vecs.dInv;
  const arma::vec Rdiag = R.diag();

  const arma::vec Rd_loo = Rd - (Rdiag % d);
  const arma::vec b = cache_vecs.b;

  const arma::vec num = v - dInv % tvec - d % (b - dInv % (Rdiag % y));
  const arma::vec y_hat = - num / (dInv % Rd_loo);

  const double loocv_mse = arma::mean(arma::square(y - y_hat));
  const arma::vec s = arma::sqrt(arma::clamp(theta4q / (arma::square(Rd_loo) % dInv), 0.0,
                                             std::numeric_limits<double>::infinity()));
  const arma::vec scores = arma::abs(y - y_hat) / s;

  if (!is_finite_vec(Rd) || !is_finite_vec(y_hat) || !std::isfinite(mu)) {
    throw std::runtime_error("model statistics produced non-finite values");
  }

  return {R, d, Rd, dInv, v, b, mu, nu2, y_hat, scores, loocv_mse};
}

PredictCache build_predict_cache(const arma::mat& X_train, const arma::vec& y_train,
                                 const arma::vec& w, const arma::mat& R,
                                 const arma::vec& d) {
  arma::mat U;
  if (!arma::chol(U, R, "upper")) {
    throw std::runtime_error("Cholesky failed");
  }
  const arma::vec Rd = R * d;
  const arma::vec Dy = Rd % y_train;
  const CholCacheVectors cache_vecs = chol_cache_vectors(U, R, y_train, Dy);
  const arma::mat Xw_train = weight_matrix(X_train, w);
  return {Rd, cache_vecs.dInv, cache_vecs.v, cache_vecs.b, Xw_train};
}

struct InitialContext {
  const arma::mat* X;
  const arma::vec* y;
  int p;
  double q;
  bool noise;
};

void initial_eval_grad(const std::vector<double>& tmp, const InitialContext& ctx,
                       double& objective, std::vector<double>& grad, bool need_grad) {
  const arma::mat& X = *ctx.X;
  const arma::vec& y = *ctx.y;
  const int p = ctx.p;
  const double q = ctx.q;

  const arma::uword n = X.n_rows;

  if (tmp.size() != static_cast<std::size_t>(p + 1)) {
    throw std::runtime_error("initial parameter length mismatch");
  }

  const double logtheta = tmp[0];
  const double theta = std::exp(logtheta);

  arma::vec u(p);
  for (int j = 0; j < p; ++j) {
    u[j] = tmp[1 + j];
  }
  const arma::vec w = softmax(u);

  const arma::mat Xw = weight_matrix(X, w);
  const std::size_t n_neighbors = ctx.noise ? static_cast<std::size_t>(2 * p + 1)
                                            : static_cast<std::size_t>(p + 1);

  const KNNResult nn = get_knn_self(Xw, n_neighbors);
  const arma::umat& idx = nn.idx;
  const arma::mat d2 = arma::square(nn.dist);
  const std::size_t effective_neighbors = idx.n_cols;

  arma::mat y_nn(n, effective_neighbors, arma::fill::zeros);
  for (arma::uword i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < effective_neighbors; ++j) {
      y_nn(i, j) = y[idx(i, j)];
    }
  }

  const arma::mat G = theta * theta + d2;
  const arma::mat A = arma::pow(G, -q);

  const arma::vec num = arma::sum(y_nn % A, 1);
  const arma::vec den = arma::sum(A, 1);
  const arma::vec r = num / den;

  objective = arma::mean(arma::square(y - r));

  if (!need_grad) {
    return;
  }

  const arma::vec g_r = (-2.0 / static_cast<double>(y.n_elem)) * (y - r);

  const arma::mat dA_dtheta = (-2.0 * q * theta) * arma::pow(G, -(q + 1.0));

  const arma::vec dnum_dtheta = arma::sum(y_nn % dA_dtheta, 1);
  const arma::vec dden_dtheta = arma::sum(dA_dtheta, 1);
  const arma::vec dr_dtheta = (dnum_dtheta % den - num % dden_dtheta) / arma::square(den);

  const double grad_logtheta = arma::dot(g_r, dr_dtheta) * theta;

  arma::vec grad_w(p, arma::fill::zeros);
  const arma::mat dA_dd2 = (-q) * arma::pow(G, -(q + 1.0));

  for (int k = 0; k < p; ++k) {
    arma::mat dA_dwk(n, n_neighbors, arma::fill::zeros);

    for (arma::uword i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n_neighbors; ++j) {
        const arma::uword nn_idx = idx(i, j);
        const double dx = X(i, k) - X(nn_idx, k);
        const double dx2 = dx * dx;
        dA_dwk(i, j) = dA_dd2(i, j) * dx2;
      }
    }

    const arma::vec dnum_dwk = arma::sum(y_nn % dA_dwk, 1);
    const arma::vec dden_dwk = arma::sum(dA_dwk, 1);
    const arma::vec dr_dwk = (dnum_dwk % den - num % dden_dwk) / arma::square(den);
    grad_w[k] = arma::dot(g_r, dr_dwk);
  }

  const arma::vec grad_u = softmax_grad(grad_w, w);

  grad.resize(tmp.size());
  grad[0] = grad_logtheta;
  for (int j = 0; j < p; ++j) {
    grad[1 + j] = grad_u[j];
  }
}

double initial_objective_value(const std::vector<double>& x, std::vector<double>& grad,
                               void* data) {
  const auto* ctx = static_cast<InitialContext*>(data);
  try {
    double f = 0.0;
    initial_eval_grad(x, *ctx, f, grad, !grad.empty());
    return f;
  } catch (...) {
    if (!grad.empty()) {
      std::fill(grad.begin(), grad.end(), 0.0);
    }
    return kPenalty;
  }
}

double initial_objective_cb(unsigned n, const double* x, double* grad, void* data) {
  std::vector<double> xvec(x, x + n);
  std::vector<double> gvec;
  if (grad != nullptr) {
    gvec.assign(n, 0.0);
  }
  const double val = initial_objective_value(xvec, gvec, data);
  if (grad != nullptr) {
    std::copy(gvec.begin(), gvec.end(), grad);
  }
  return val;
}

std::vector<double> initial_opt(const arma::mat& X, const arma::vec& y,
                                double logtheta0,
                                double logtheta_lb, double logtheta_ub,
                                double q, bool noise) {
  const int p = static_cast<int>(X.n_cols);

  std::vector<double> x0(static_cast<std::size_t>(p + 1), 0.0);
  x0[0] = logtheta0;

  std::vector<double> lb(static_cast<std::size_t>(p + 1), -std::numeric_limits<double>::infinity());
  std::vector<double> ub(static_cast<std::size_t>(p + 1), std::numeric_limits<double>::infinity());
  lb[0] = logtheta_lb;
  ub[0] = logtheta_ub;
  for (int j = 0; j < p; ++j) {
    lb[1 + j] = kUMinBound;
    ub[1 + j] = kUMaxBound;
  }

  InitialContext ctx{&X, &y, p, q, noise};

  nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, static_cast<unsigned>(p + 1));
  if (opt == nullptr) {
    Rcpp::stop("initial optimization failed: nlopt_create returned null");
  }

  double minf = 0.0;
  try {
    check_nlopt_result(nlopt_set_min_objective(opt, initial_objective_cb, &ctx),
                       "initial optimization: set objective failed");
    check_nlopt_result(nlopt_set_lower_bounds(opt, lb.data()),
                       "initial optimization: set lower bounds failed");
    check_nlopt_result(nlopt_set_upper_bounds(opt, ub.data()),
                       "initial optimization: set upper bounds failed");
    check_nlopt_result(nlopt_set_xtol_rel(opt, 1e-4),
                       "initial optimization: set xtol_rel failed");
    check_nlopt_result(nlopt_set_maxeval(opt, 100),
                       "initial optimization: set maxeval failed");
    check_nlopt_optimize_result(nlopt_optimize(opt, x0.data(), &minf),
                                "initial optimization failed");
  } catch (...) {
    nlopt_destroy(opt);
    throw;
  }
  nlopt_destroy(opt);

  return x0;
}

struct LOOCVContext {
  const arma::mat* X;
  const arma::vec* y;
  int p;
  double q;
  bool noise;
  bool eta_variable;
  double eta_fixed;
  const arma::cube* pairwise_d2_cache;
};

double loocv_value(const std::vector<double>& x, const LOOCVContext& ctx) {
  try {
    const arma::mat& X = *ctx.X;
    const arma::vec& y = *ctx.y;
    const int p = ctx.p;
    const double q = ctx.q;

    std::size_t offset = 1;
    const double theta = std::exp(x[0]);
    double eta = ctx.eta_fixed;

    if (ctx.eta_variable) {
      offset = 2;
      eta = std::exp(x[1]);
    }

    arma::vec u(p);
    for (int j = 0; j < p; ++j) {
      u[j] = x[offset + static_cast<std::size_t>(j)];
    }

    const arma::vec w = softmax(u);
    const ModelStats st = compute_model_stats(X, y, p, q, theta, eta, w, ctx.pairwise_d2_cache);

    double obj = st.loocv_mse;

    if (!ctx.noise) {
      const arma::vec numerator_vec = st.R * st.v;
      const arma::vec m_train = numerator_vec / st.Rd;
      const double mse = arma::mean(arma::square(y - m_train));
      obj += 1e6 * arma::var(y) * mse;
    }

    return obj;
  } catch (...) {
    return kPenalty;
  }
}

double loocv_cb(unsigned n, const double* x, double* grad, void* data) {
  (void)grad;
  const auto* ctx = static_cast<LOOCVContext*>(data);
  std::vector<double> xvec(x, x + n);
  return loocv_value(xvec, *ctx);
}

struct InnerOptResult {
  double value;
  std::vector<double> solution;
};

struct InnerOptWorkspace {
  const LOOCVContext* ctx;
  bool eta_variable;
  std::size_t p;
  std::size_t dim;
  nlopt_opt opt;
  std::vector<double> x;
  std::vector<double> lb;
  std::vector<double> ub;
  std::vector<double> step;
  bool fixed_u_bounds_initialized;
  std::vector<double> fixed_u_center;

  InnerOptWorkspace(const LOOCVContext& ctx_)
      : ctx(&ctx_),
        eta_variable(ctx_.eta_variable),
        p(static_cast<std::size_t>(ctx_.p)),
        dim(ctx_.eta_variable ? p + 2 : p + 1),
        opt(nullptr),
        x(dim, 0.0),
        lb(dim, -std::numeric_limits<double>::infinity()),
        ub(dim, std::numeric_limits<double>::infinity()),
        step(dim, 0.0),
        fixed_u_bounds_initialized(false),
        fixed_u_center(p, 0.0) {
    
    opt = nlopt_create(NLOPT_LN_BOBYQA, static_cast<unsigned>(dim));
    if (opt == nullptr) {
      throw std::runtime_error("inner optimization: nlopt_create returned null");
    }

    try {
      check_nlopt_result(nlopt_set_min_objective(opt, loocv_cb, const_cast<LOOCVContext*>(ctx)),
                         "inner optimization: set objective failed");
      check_nlopt_result(nlopt_set_xtol_rel(opt, 1e-4),
                         "inner optimization: set xtol_rel failed");
      check_nlopt_result(nlopt_set_maxeval(opt, kLocalMaxeval),
                         "inner optimization: set maxeval failed");
    } catch (...) {
      nlopt_destroy(opt);
      opt = nullptr;
      throw;
    }
  }

  ~InnerOptWorkspace() {
    if (opt != nullptr) {
      nlopt_destroy(opt);
    }
  }

  InnerOptWorkspace(const InnerOptWorkspace&) = delete;
  InnerOptWorkspace& operator=(const InnerOptWorkspace&) = delete;

  InnerOptResult run(const arma::vec& u0, double logtheta_init,
                     bool theta_fixed = true, double logtheta_lb = 0.0,
                     double logtheta_ub = 0.0, double logeta_init = 0.0,
                     bool eta_fixed = true, double logeta_lb = 0.0,
                     double logeta_ub = 0.0) {
    if (u0.n_elem != p) {
      throw std::runtime_error("inner optimization: warm-start length mismatch");
    }

    x[0] = theta_fixed ? logtheta_init : std::clamp(logtheta_init, logtheta_lb, logtheta_ub);
    if (eta_variable) {
      x[1] = eta_fixed ? logeta_init : std::clamp(logeta_init, logeta_lb, logeta_ub);
    }
    const std::size_t offset = eta_variable ? 2 : 1;
    for (std::size_t j = 0; j < p; ++j) {
      x[offset + j] = std::clamp(static_cast<double>(u0[j]), kUMinBound, kUMaxBound);
    }

    if (!fixed_u_bounds_initialized) {
      for (std::size_t j = 0; j < p; ++j) {
        fixed_u_center[j] = x[offset + j];
      }
      fixed_u_bounds_initialized = true;
    }

    if (theta_fixed) {
      lb[0] = logtheta_init;
      ub[0] = logtheta_init;
    } else {
      lb[0] = logtheta_lb;
      ub[0] = logtheta_ub;
      x[0] = std::clamp(x[0], lb[0], ub[0]);
    }
    if (eta_variable) {
      if (eta_fixed) {
        lb[1] = logeta_init;
        ub[1] = logeta_init;
      } else {
        lb[1] = logeta_lb;
        ub[1] = logeta_ub;
        x[1] = std::clamp(x[1], lb[1], ub[1]);
      }
    }
    for (std::size_t j = 0; j < p; ++j) {
      lb[offset + j] = std::max(fixed_u_center[j] - kLocalULogRadius, kUMinBound);
      ub[offset + j] = std::min(fixed_u_center[j] + kLocalULogRadius, kUMaxBound);
      x[offset + j] = std::clamp(x[offset + j], lb[offset + j], ub[offset + j]);
    }

    double minf = 0.0;
    check_nlopt_result(nlopt_set_lower_bounds(opt, lb.data()),
                       "inner optimization: set lower bounds failed");
    check_nlopt_result(nlopt_set_upper_bounds(opt, ub.data()),
                       "inner optimization: set upper bounds failed");
    check_nlopt_optimize_result(nlopt_optimize(opt, x.data(), &minf),
                                "inner optimization failed");
    return {minf, x};
  }
};

InnerOptResult optimize_locally(const LOOCVContext& ctx, const arma::vec& u0,
                                double logtheta_init, bool theta_fixed,
                                double logtheta_lb, double logtheta_ub,
                                double logeta_init, bool eta_fixed,
                                double logeta_lb, double logeta_ub) {
  InnerOptWorkspace ws(ctx);
  return ws.run(u0, logtheta_init, theta_fixed, logtheta_lb, logtheta_ub,
                logeta_init, eta_fixed, logeta_lb, logeta_ub);
}

std::vector<double> make_log_grid(double lb, double ub, std::size_t n_points) {
  if (n_points == 0) {
    throw std::runtime_error("make_log_grid requires at least one point");
  }
  if (!(ub > lb)) {
    return std::vector<double>(n_points, lb);
  }
  std::vector<double> grid(n_points, 0.0);
  const double step = (ub - lb) / static_cast<double>(n_points);
  for (std::size_t i = 0; i < n_points; ++i) {
    grid[i] = lb + (static_cast<double>(i) + 0.5) * step;
  }
  return grid;
}

double loocv_with_fixed_u(const LOOCVContext& ctx, const arma::vec& u0,
                          double logtheta, double logeta) {
  const std::size_t p = static_cast<std::size_t>(ctx.p);
  std::vector<double> x(ctx.eta_variable ? p + 2 : p + 1, 0.0);
  x[0] = logtheta;
  const std::size_t offset = ctx.eta_variable ? 2 : 1;
  if (ctx.eta_variable) {
    x[1] = logeta;
  }
  for (std::size_t j = 0; j < p; ++j) {
    x[offset + j] = u0[j];
  }
  return loocv_value(x, ctx);
}

double kth_smallest(const arma::vec& x, std::size_t k_one_based) {
  if (k_one_based < 1 || k_one_based > x.n_elem) {
    throw std::runtime_error("k out of range in kth_smallest");
  }
  std::vector<double> vals = arma::conv_to<std::vector<double>>::from(x);
  const std::size_t idx = k_one_based - 1;
  std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(idx), vals.end());
  return vals[idx];
}

Rcpp::NumericMatrix interval_matrix(
    const std::vector<std::pair<double, double>>& intervals) {
  Rcpp::NumericMatrix out(intervals.size(), 2);
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    out(static_cast<R_xlen_t>(i), 0) = intervals[i].first;
    out(static_cast<R_xlen_t>(i), 1) = intervals[i].second;
  }
  out.attr("dimnames") =
      Rcpp::List::create(R_NilValue, Rcpp::CharacterVector::create("lower", "upper"));
  return out;
}

std::vector<std::pair<double, double>> merged_type2_intervals(
    const arma::vec& lower_sorted, const arma::vec& upper_sorted, std::size_t keep_count,
    double eps) {
  std::vector<std::pair<double, double>> candidates;
  candidates.reserve(keep_count);
  const std::size_t n = lower_sorted.n_elem;

  for (std::size_t t = 0; t < keep_count; ++t) {
    const double lower = lower_sorted[n - 1 - t];
    const double upper = upper_sorted[t];
    if (lower <= upper + eps) {
      candidates.emplace_back(lower, upper);
    }
  }

  if (candidates.empty()) {
    return candidates;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const std::pair<double, double>& lhs,
               const std::pair<double, double>& rhs) {
              if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
              }
              return lhs.first < rhs.first;
            });

  std::vector<std::pair<double, double>> merged;
  merged.reserve(candidates.size());
  merged.push_back(candidates.front());

  for (std::size_t i = 1; i < candidates.size(); ++i) {
    const auto& current = candidates[i];
    auto& last = merged.back();
    if (current.first <= last.second + eps) {
      last.second = std::max(last.second, current.second);
    } else {
      merged.push_back(current);
    }
  }

  return merged;
}

struct ThetaConfig {
  double logtheta0;
  double logtheta_lb;
  double logtheta_ub;
};

ThetaConfig make_theta_config(bool theta_fit, bool theta_provided, double theta,
                              double empirical_lb, double max_r_safe, int p, bool noise,
                              bool theta_bound_provided, double theta_lb, double theta_ub) {
  if (!theta_fit) {
    const double logtheta = std::log(theta);
    return ThetaConfig{logtheta, logtheta, logtheta};
  }

  double logtheta_lb = 0.0;
  double logtheta_ub = 0.0;
  if (theta_bound_provided) {
    logtheta_lb = std::log(theta_lb);
    logtheta_ub = std::log(theta_ub);
  } else {
    const double base_logtheta = theta_provided
                                     ? std::log(theta)
                                     : (noise ? empirical_lb
                                              : std::log(0.01 * static_cast<double>(p) * max_r_safe));
    logtheta_lb = std::min(base_logtheta, empirical_lb);
    logtheta_ub = std::max(base_logtheta, std::log(std::sqrt(static_cast<double>(p)) * max_r_safe));
  }

  const double logtheta0 = theta_provided
                               ? std::log(theta)
                               : (logtheta_ub + std::log1p(std::exp(logtheta_lb - logtheta_ub)) -
                                  std::log(2.0));

  if (theta_provided && (logtheta0 < logtheta_lb || logtheta0 > logtheta_ub)) {
    Rcpp::stop("theta must lie within theta_bound when theta_fit is TRUE");
  }

  return ThetaConfig{logtheta0, logtheta_lb, logtheta_ub};
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List initial_lengthscale_cpp(const arma::mat& X, const arma::vec& y, double theta,
                                   bool theta_fit, bool noise, double q) {
  if (X.n_rows != y.n_elem) {
    Rcpp::stop("nrow(X) must equal length(y)");
  }
  if (X.n_cols < 1) {
    Rcpp::stop("X must have at least one column");
  }
  if (X.n_rows < 3) {
    Rcpp::stop("RQK requires at least 3 rows");
  }
  if (!is_finite_mat(X) || !is_finite_vec(y)) {
    Rcpp::stop("X and y must be finite");
  }

  const bool theta_provided = !Rcpp::NumericVector::is_na(theta);
  if (!theta_fit && !theta_provided) {
    Rcpp::stop("If theta_fit is FALSE, theta must be provided");
  }
  if (theta_provided && (!(theta > 0.0) || !std::isfinite(theta))) {
    Rcpp::stop("theta must be finite and positive");
  }

  const int p = static_cast<int>(X.n_cols);
  const double q_effective = resolve_kernel_exponent(p, noise, q);

  const double max_r = max_range(X);
  const double max_r_safe = std::max(max_r, std::sqrt(std::numeric_limits<double>::epsilon()));
  if (!theta_provided && !(max_r > 0.0)) {
    Rcpp::stop("theta is required when all columns in X are constant");
  }

  const std::size_t k_emp = noise ? static_cast<std::size_t>(p + 1) : 1;
  const KNNResult knn_emp = get_knn_self(X, k_emp);
  const double empirical_lb =
      std::log(arma::mean(knn_emp.dist.col(knn_emp.dist.n_cols - 1)));

  const ThetaConfig theta_cfg =
      make_theta_config(theta_fit, theta_provided, theta, empirical_lb, max_r_safe, p, noise,
                        false, NA_REAL, NA_REAL);

  std::vector<double> init_sol =
      initial_opt(X, y, theta_cfg.logtheta0, theta_cfg.logtheta_lb, theta_cfg.logtheta_ub,
                  q_effective, noise);

  arma::vec u0(p, arma::fill::zeros);
  for (int j = 0; j < p; ++j) {
    u0[j] = init_sol[1 + j];
  }
  const arma::vec w0 = softmax(u0);

  return Rcpp::List::create(
      Rcpp::Named("theta") = std::exp(init_sol[0]),
      Rcpp::Named("logtheta") = init_sol[0],
      Rcpp::Named("q") = q_effective,
      Rcpp::Named("u") = as_r_vector(u0),
      Rcpp::Named("w") = as_r_vector(w0),
      Rcpp::Named("par") = Rcpp::wrap(init_sol),
      Rcpp::Named("logtheta_lb") = theta_cfg.logtheta_lb,
      Rcpp::Named("logtheta_ub") = theta_cfg.logtheta_ub);
}

// [[Rcpp::export]]
Rcpp::List rqk_fit_cpp(const arma::mat& X, const arma::vec& y, double theta,
                       bool theta_fit, bool noise, double nugget, double q,
                       double theta_lb, double theta_ub) {
  if (X.n_rows != y.n_elem) {
    Rcpp::stop("nrow(X) must equal length(y)");
  }
  if (X.n_cols < 1) {
    Rcpp::stop("X must have at least one column");
  }
  if (X.n_rows < 3) {
    Rcpp::stop("RQK requires at least 3 rows");
  }
  if (!is_finite_mat(X) || !is_finite_vec(y)) {
    Rcpp::stop("X and y must be finite");
  }

  const bool theta_provided = !Rcpp::NumericVector::is_na(theta);
  if (!theta_fit && !theta_provided) {
    Rcpp::stop("If theta_fit is FALSE, theta must be provided");
  }
  if (theta_provided && (!(theta > 0.0) || !std::isfinite(theta))) {
    Rcpp::stop("theta must be finite and positive");
  }
  const bool theta_lb_provided = !Rcpp::NumericVector::is_na(theta_lb);
  const bool theta_ub_provided = !Rcpp::NumericVector::is_na(theta_ub);
  if (theta_lb_provided != theta_ub_provided) {
    Rcpp::stop("theta_bound must provide both lower and upper bounds");
  }
  const bool theta_bound_provided = theta_lb_provided && theta_ub_provided;
  if (theta_bound_provided) {
    if (!std::isfinite(theta_lb) || !std::isfinite(theta_ub) ||
        !(theta_lb > 0.0) || !(theta_ub > 0.0)) {
      Rcpp::stop("theta_bound values must be finite and positive");
    }
    if (theta_lb > theta_ub) {
      Rcpp::stop("theta_bound lower bound must be <= upper bound");
    }
  }
  const bool nugget_provided = !Rcpp::NumericVector::is_na(nugget);
  if (nugget_provided) {
    if (!std::isfinite(nugget) || nugget < 0.0) {
      Rcpp::stop("nugget must be finite and nonnegative when provided");
    }
    if (noise && !(nugget > 0.0)) {
      Rcpp::stop("nugget must be positive when noise is TRUE");
    }
    if (noise && nugget > 1.0) {
      Rcpp::stop("nugget must be <= 1 when noise is TRUE");
    }
  }

  const int p = static_cast<int>(X.n_cols);
  const double q_effective = resolve_kernel_exponent(p, noise, q);

  const double max_r = max_range(X);
  const double max_r_safe = std::max(max_r, std::sqrt(std::numeric_limits<double>::epsilon()));
  if (!theta_provided && !(max_r > 0.0)) {
    Rcpp::stop("theta is required when all columns in X are constant");
  }

  const std::size_t k_emp = noise ? static_cast<std::size_t>(p + 1) : 1;
  const KNNResult knn_emp = get_knn_self(X, k_emp);
  const double empirical_lb =
      std::log(arma::mean(knn_emp.dist.col(knn_emp.dist.n_cols - 1)));

  ThetaConfig theta_cfg = make_theta_config(theta_fit, theta_provided, theta,
                                            empirical_lb, max_r_safe, p, noise,
                                            theta_bound_provided, theta_lb, theta_ub);
  double logtheta0 = theta_cfg.logtheta0;
  const double logtheta_lb = theta_cfg.logtheta_lb;
  const double logtheta_ub = theta_cfg.logtheta_ub;
  const bool eta_variable = noise && !nugget_provided;
  const double nugget_fixed = nugget_provided ? nugget : (noise ? 0.0 : 1e-8);
  const double logeta0 = eta_variable ? std::log(1e-2) : 0.0;
  const double logeta_lb = eta_variable ? std::log(1e-6) : 0.0;
  const double logeta_ub = eta_variable ? std::log(10.0) : 0.0;

  std::vector<double> init_sol =
      initial_opt(X, y, logtheta0, logtheta_lb, logtheta_ub, q_effective, noise);

  arma::vec u0(p, arma::fill::zeros);
  for (int j = 0; j < p; ++j) {
    u0[j] = init_sol[1 + j];
  }

  arma::cube loocv_pairwise_d2_cache;
  const arma::cube* loocv_pairwise_d2_cache_ptr = nullptr;
  if (should_build_loocv_pairwise_cache(X)) {
    loocv_pairwise_d2_cache = pairwise_dist2_by_feature(X);
    loocv_pairwise_d2_cache_ptr = &loocv_pairwise_d2_cache;
  }
  const LOOCVContext loocv_ctx{&X, &y, p, q_effective, noise, eta_variable, nugget_fixed,
                               loocv_pairwise_d2_cache_ptr};

  if (!theta_fit) {
    logtheta0 = std::log(theta);
  }

  std::vector<double> sol;
  double logtheta_start = logtheta0;
  double logeta_start = logeta0;

  try {
    if (theta_fit) {
      if (eta_variable) {
        const std::vector<double> theta_grid =
            make_log_grid(logtheta_lb, logtheta_ub, kLineSearchPoints2D);
        const std::vector<double> eta_grid =
            make_log_grid(logeta_lb, logeta_ub, kLineSearchPoints2D);
        double best_obj = std::numeric_limits<double>::infinity();
        for (double logtheta_candidate : theta_grid) {
          for (double logeta_candidate : eta_grid) {
            const double obj =
                loocv_with_fixed_u(loocv_ctx, u0, logtheta_candidate, logeta_candidate);
            if (obj < best_obj) {
              best_obj = obj;
              logtheta_start = logtheta_candidate;
              logeta_start = logeta_candidate;
            }
          }
        }
      } else {
        const std::vector<double> theta_grid =
            make_log_grid(logtheta_lb, logtheta_ub, kLineSearchPoints1D);
        double best_obj = std::numeric_limits<double>::infinity();
        for (double logtheta_candidate : theta_grid) {
          const double obj = loocv_with_fixed_u(loocv_ctx, u0, logtheta_candidate, 0.0);
          if (obj < best_obj) {
            best_obj = obj;
            logtheta_start = logtheta_candidate;
          }
        }
      }
    } else if (eta_variable) {
      const std::vector<double> eta_grid = make_log_grid(logeta_lb, logeta_ub, kLineSearchPoints1D);
      double best_obj = std::numeric_limits<double>::infinity();
      for (double logeta_candidate : eta_grid) {
        const double obj = loocv_with_fixed_u(loocv_ctx, u0, logtheta_start, logeta_candidate);
        if (obj < best_obj) {
          best_obj = obj;
          logeta_start = logeta_candidate;
        }
      }
    }

    const InnerOptResult final_inner =
        optimize_locally(loocv_ctx, u0, logtheta_start, !theta_fit, logtheta_lb, logtheta_ub,
                         logeta_start, !eta_variable, logeta_lb, logeta_ub);
    sol = final_inner.solution;
  } catch (const std::exception& e) {
    Rcpp::stop(std::string("Main optimization failed: ") + e.what());
  }

  double theta_hat = std::exp(sol[0]);
  double eta_hat = nugget_fixed;
  arma::vec u(p);

  if (eta_variable) {
    eta_hat = std::exp(sol[1]);
    for (int j = 0; j < p; ++j) {
      u[j] = sol[2 + j];
    }
  } else {
    for (int j = 0; j < p; ++j) {
      u[j] = sol[1 + j];
    }
  }

  arma::vec w = softmax(u);

  ModelStats st;
  try {
    st = compute_model_stats(X, y, p, q_effective, theta_hat, eta_hat, w);
  } catch (const std::exception& e) {
    Rcpp::stop(std::string("Failed to compute final RQK statistics: ") + e.what());
  }

  PredictCache cache;
  try {
    cache = build_predict_cache(X, y, w, st.R, st.d);
  } catch (const std::exception& e) {
    Rcpp::stop(std::string("Failed to build prediction cache: ") + e.what());
  }

  return Rcpp::List::create(
      Rcpp::Named("w") = as_r_vector(w),
      Rcpp::Named("theta") = theta_hat,
      Rcpp::Named("nugget") = eta_hat,
      Rcpp::Named("q") = q_effective,
      Rcpp::Named("noise") = noise,
      Rcpp::Named("LOO_pred") = st.loo_pred,
      Rcpp::Named("scores") = st.scores,
      Rcpp::Named("R") = st.R,
      Rcpp::Named("d") = st.d,
      Rcpp::Named("Rd") = cache.Rd,
      Rcpp::Named("dInv") = cache.dInv,
      Rcpp::Named("v") = cache.v,
      Rcpp::Named("b") = cache.b,
      Rcpp::Named("Xw_train") = cache.Xw_train,
      Rcpp::Named("mu") = st.mu,
      Rcpp::Named("X_train") = X,
      Rcpp::Named("y_train") = y,
      Rcpp::Named("p") = p);
}

// [[Rcpp::export]]
Rcpp::List rqk_predict_cpp(const Rcpp::List& fit, const arma::mat& X, double alpha,
                           int interval_type, double score_quantile) {
  if (!(alpha > 0.0 && alpha < 1.0)) {
    Rcpp::stop("alpha must be in (0, 1)");
  }
  if (interval_type < 0 || interval_type > 4) {
    Rcpp::stop("interval_type must be 0 or an integer in 1:4");
  }
  if (interval_type == 1 &&
      (!std::isfinite(score_quantile) || score_quantile < 0.0)) {
    Rcpp::stop("score_quantile must be finite and nonnegative for interval type 1");
  }

  const int p = Rcpp::as<int>(fit["p"]);
  const arma::mat X_train = Rcpp::as<arma::mat>(fit["X_train"]);
  const arma::vec y_train = Rcpp::as<arma::vec>(fit["y_train"]);
  const arma::vec w = Rcpp::as<arma::vec>(fit["w"]);
  const double theta = Rcpp::as<double>(fit["theta"]);
  const double q = fit.containsElementNamed("q")
                       ? Rcpp::as<double>(fit["q"])
                       : (fit.containsElementNamed("noise")
                              ? default_kernel_exponent(p, Rcpp::as<bool>(fit["noise"]))
                              : static_cast<double>(p));
  const double eta = fit.containsElementNamed("nugget")
                         ? Rcpp::as<double>(fit["nugget"])
                         : (fit.containsElementNamed("eta") ? Rcpp::as<double>(fit["eta"]) : 0.0);

  if (X.n_cols != X_train.n_cols) {
    Rcpp::stop("newdata must have the same number of columns as training data");
  }
  if (static_cast<int>(X.n_cols) != p) {
    Rcpp::stop("fit$p does not match training data columns");
  }

  if (!(q > 0.0) || !std::isfinite(q)) {
    Rcpp::stop("fit$q must be finite and positive");
  }

  const arma::uword n = X_train.n_rows;
  const arma::uword m_test = X.n_rows;
  const double pred_obs_diag = 1.0 + eta;
  const double theta2q = std::pow(theta, 2.0 * q);
  const double theta4q = theta2q * theta2q;
  const bool use_none = interval_type == 0;
  const bool use_type1 = interval_type == 1;
  if (!(theta2q > 0.0) || !std::isfinite(theta4q)) {
    Rcpp::stop("theta scaling is not finite");
  }

  arma::vec score_values;
  if (!use_none) {
    score_values = Rcpp::as<arma::vec>(fit["scores"]);
    if (score_values.n_elem != n) {
      Rcpp::stop("scores length mismatch in fit object");
    }
  }

  const arma::mat Xw = weight_matrix(X, w);

  arma::mat Xw_train;
  if (fit.containsElementNamed("Xw_train")) {
    Xw_train = Rcpp::as<arma::mat>(fit["Xw_train"]);
    if (Xw_train.n_rows != X_train.n_rows || Xw_train.n_cols != X_train.n_cols) {
      Xw_train = weight_matrix(X_train, w);
    }
  } else {
    Xw_train = weight_matrix(X_train, w);
  }

  arma::mat R;
  if (fit.containsElementNamed("R")) {
    R = Rcpp::as<arma::mat>(fit["R"]);
    if (R.n_rows != n || R.n_cols != n) {
      R = kernel_from_dist2(pairwise_dist2(Xw_train), theta, q);
      R.diag() += eta;
    }
  } else {
    R = kernel_from_dist2(pairwise_dist2(Xw_train), theta, q);
    R.diag() += eta;
  }

  arma::vec d_vec;
  if (fit.containsElementNamed("d")) {
    d_vec = Rcpp::as<arma::vec>(fit["d"]);
    if (d_vec.n_elem != n) {
      d_vec = theta2q * compute_base_d(R);
    }
  } else {
    d_vec = theta2q * compute_base_d(R);
  }

  arma::vec Rd;
  arma::vec dInv;
  arma::vec v;
  arma::vec b;

  const bool has_cache = fit.containsElementNamed("Rd") &&
                         fit.containsElementNamed("dInv") &&
                         fit.containsElementNamed("v") &&
                         fit.containsElementNamed("b");

  if (has_cache) {
    Rd = Rcpp::as<arma::vec>(fit["Rd"]);
    dInv = Rcpp::as<arma::vec>(fit["dInv"]);
    v = Rcpp::as<arma::vec>(fit["v"]);
    b = Rcpp::as<arma::vec>(fit["b"]);
  }

  const bool cache_ok = has_cache && Rd.n_elem == n && dInv.n_elem == n && v.n_elem == n &&
                        b.n_elem == n;

  if (!cache_ok) {
    const PredictCache cache = build_predict_cache(X_train, y_train, w, R, d_vec);
    Rd = cache.Rd;
    dInv = cache.dInv;
    v = cache.v;
    b = cache.b;
  }

  const arma::mat D2_cross = cross_dist2(Xw, Xw_train);
  const arma::mat g_mat = arma::pow((theta * theta) + D2_cross, -q);
  const arma::mat Gt = g_mat.t();
  const arma::vec denom_vec = g_mat * d_vec;
  const arma::vec numerator_vec = g_mat * v;

  arma::vec m(m_test, arma::fill::zeros);
  for (arma::uword i = 0; i < m_test; ++i) {
    m[i] = numerator_vec[i] / denom_vec[i];
  }

  if (use_none) {
    return Rcpp::List::create(Rcpp::Named("mean") = m);
  }

  arma::mat U;
  if (!arma::chol(U, R, "upper")) {
    Rcpp::stop("Cholesky failed");
  }
  const arma::mat Z = chol_solve(U, Gt);
  const arma::vec quad_g_vec = arma::sum(Gt % Z, 0).t();
  arma::mat Q;
  if (!use_type1) {
    Q = R * (Z.each_col() % y_train);
  }

  arma::vec lower(m_test);
  lower.fill(std::numeric_limits<double>::quiet_NaN());
  arma::vec upper(m_test);
  upper.fill(std::numeric_limits<double>::quiet_NaN());
  Rcpp::List intervals(m_test);

  const std::size_t kU = std::min<std::size_t>(
      static_cast<std::size_t>(n - 1),
      static_cast<std::size_t>(std::ceil((n + 1.0) * (1.0 - alpha))));
  const std::size_t kL = static_cast<std::size_t>(n) - kU + 1;

  const double eps = 1e-12;

  for (arma::uword i = 0; i < m_test; ++i) {
    const arma::vec g = Gt.col(i);
    const arma::vec z = Z.col(i);
    const double denom = denom_vec[i];
    const double numerator = numerator_vec[i];
    const double quad = theta4q * quad_g_vec[i];

    if (use_type1) {
      const double s_raw = std::sqrt(std::max(0.0, (pred_obs_diag - quad) / (denom * denom)));
      lower[i] = m[i] - score_quantile * s_raw;
      upper[i] = m[i] + score_quantile * s_raw;
      intervals[i] = interval_matrix(
          std::vector<std::pair<double, double>>{{lower[i], upper[i]}});
      continue;
    }

    arma::vec denom_loo = denom - g % d_vec;
    denom_loo.transform([&](double x) { return x > eps ? x : eps; });

    const arma::vec num_base = numerator - (z % v) / dInv;
    const arma::vec q_vec = Q.col(i);
    const arma::vec T = q_vec - (z % b) / dInv;
    const arma::vec num_loo = num_base - d_vec % T;

    const arma::vec m_loo = num_loo / denom_loo;

    const arma::vec s_loo = arma::sqrt(arma::clamp(
        arma::square(1.0 / denom_loo) %
            ((pred_obs_diag - quad) + theta4q * arma::square(z) / dInv),
        0.0, std::numeric_limits<double>::infinity()));

    const arma::vec lower_candidates = m_loo - s_loo % score_values;
    const arma::vec upper_candidates = m_loo + s_loo % score_values;

    if (interval_type == 4) {
      lower[i] = kth_smallest(lower_candidates, kL);
      upper[i] = kth_smallest(upper_candidates, kU);
      intervals[i] = interval_matrix(
          std::vector<std::pair<double, double>>{{lower[i], upper[i]}});
      continue;
    }

    const arma::vec lower_sorted = arma::sort(lower_candidates);
    const arma::vec upper_sorted = arma::sort(upper_candidates);
    const std::vector<std::pair<double, double>> merged =
        merged_type2_intervals(lower_sorted, upper_sorted, kU, eps);

    if (interval_type == 2) {
      intervals[i] = interval_matrix(merged);
      continue;
    }

    if (!merged.empty()) {
      lower[i] = merged.front().first;
      upper[i] = merged.back().second;
      intervals[i] = interval_matrix(
          std::vector<std::pair<double, double>>{{lower[i], upper[i]}});
    } else {
      intervals[i] = interval_matrix(std::vector<std::pair<double, double>>{});
    }
  }

  return Rcpp::List::create(Rcpp::Named("mean") = m, Rcpp::Named("lower") = lower,
                            Rcpp::Named("upper") = upper,
                            Rcpp::Named("intervals") = intervals);
}
