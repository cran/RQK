#ifndef RQK_KDTREE_H
#define RQK_KDTREE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

// Lightweight KD-tree for k-nearest-neighbor search, written for RQK.
namespace rqk {

class KDTree {
 public:
  KDTree(const std::vector<double>& data, std::size_t n_points, std::size_t dim,
         std::size_t leaf_size = 16)
      : data_(data), n_points_(n_points), dim_(dim), leaf_size_(leaf_size) {
    if (dim_ == 0) {
      throw std::invalid_argument("KDTree dimension must be > 0");
    }
    if (data_.size() != n_points_ * dim_) {
      throw std::invalid_argument("KDTree data size mismatch");
    }
    indices_.resize(n_points_);
    for (std::size_t i = 0; i < n_points_; ++i) {
      indices_[i] = i;
    }
    root_ = build(0, n_points_);
  }

  void knn_search(const double* query, std::size_t k, std::vector<std::size_t>& out_idx,
                  std::vector<double>& out_dist2, bool exclude_self = false,
                  std::size_t self_index = 0) const {
    if (k == 0) {
      out_idx.clear();
      out_dist2.clear();
      return;
    }

    std::priority_queue<Candidate> heap;
    search_node(root_, query, k, heap, exclude_self, self_index);

    const std::size_t m = heap.size();
    out_idx.resize(m);
    out_dist2.resize(m);

    for (std::size_t i = 0; i < m; ++i) {
      const Candidate c = heap.top();
      heap.pop();
      const std::size_t pos = m - 1 - i;
      out_idx[pos] = c.index;
      out_dist2[pos] = c.dist2;
    }
  }

 private:
  struct Node {
    bool leaf = false;
    std::size_t axis = 0;
    double split = 0.0;
    std::size_t pivot = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
    Node* left = nullptr;
    Node* right = nullptr;
  };

  struct Candidate {
    double dist2;
    std::size_t index;
    bool operator<(const Candidate& other) const { return dist2 < other.dist2; }
  };

  const std::vector<double>& data_;
  std::size_t n_points_;
  std::size_t dim_;
  std::size_t leaf_size_;
  std::vector<std::size_t> indices_;
  std::deque<Node> nodes_;
  Node* root_ = nullptr;

  inline double at(std::size_t row, std::size_t col) const {
    return data_[row * dim_ + col];
  }

  double dist2_point(const double* query, std::size_t idx) const {
    double acc = 0.0;
    for (std::size_t j = 0; j < dim_; ++j) {
      const double d = query[j] - at(idx, j);
      acc += d * d;
    }
    return acc;
  }

  Node* build(std::size_t begin, std::size_t end) {
    nodes_.push_back(Node{});
    Node* node = &nodes_.back();

    const std::size_t count = end - begin;
    if (count <= leaf_size_) {
      node->leaf = true;
      node->begin = begin;
      node->end = end;
      return node;
    }

    std::size_t axis = 0;
    double best_span = -1.0;
    for (std::size_t j = 0; j < dim_; ++j) {
      double min_v = std::numeric_limits<double>::infinity();
      double max_v = -std::numeric_limits<double>::infinity();
      for (std::size_t t = begin; t < end; ++t) {
        const double v = at(indices_[t], j);
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
      }
      const double span = max_v - min_v;
      if (span > best_span) {
        best_span = span;
        axis = j;
      }
    }

    const std::size_t mid = begin + count / 2;
    std::nth_element(indices_.begin() + static_cast<std::ptrdiff_t>(begin),
                     indices_.begin() + static_cast<std::ptrdiff_t>(mid),
                     indices_.begin() + static_cast<std::ptrdiff_t>(end),
                     [&](std::size_t a, std::size_t b) {
                       return at(a, axis) < at(b, axis);
                     });

    node->leaf = false;
    node->axis = axis;
    node->pivot = indices_[mid];
    node->split = at(node->pivot, axis);
    node->left = build(begin, mid);
    node->right = build(mid + 1, end);
    return node;
  }

  void maybe_push(std::priority_queue<Candidate>& heap, std::size_t k, double d2,
                  std::size_t idx) const {
    if (heap.size() < k) {
      heap.push({d2, idx});
      return;
    }
    if (d2 < heap.top().dist2) {
      heap.pop();
      heap.push({d2, idx});
    }
  }

  void search_node(const Node* node, const double* query, std::size_t k,
                   std::priority_queue<Candidate>& heap, bool exclude_self,
                   std::size_t self_index) const {
    if (node == nullptr) {
      return;
    }

    if (node->leaf) {
      for (std::size_t t = node->begin; t < node->end; ++t) {
        const std::size_t idx = indices_[t];
        if (exclude_self && idx == self_index) {
          continue;
        }
        const double d2 = dist2_point(query, idx);
        maybe_push(heap, k, d2, idx);
      }
      return;
    }

    if (!(exclude_self && node->pivot == self_index)) {
      const double d2 = dist2_point(query, node->pivot);
      maybe_push(heap, k, d2, node->pivot);
    }

    const double diff = query[node->axis] - node->split;
    const Node* near = (diff <= 0.0) ? node->left : node->right;
    const Node* far = (diff <= 0.0) ? node->right : node->left;

    search_node(near, query, k, heap, exclude_self, self_index);

    const double worst = heap.size() < k ? std::numeric_limits<double>::infinity()
                                         : heap.top().dist2;
    if (diff * diff < worst) {
      search_node(far, query, k, heap, exclude_self, self_index);
    }
  }
};

}  // namespace rqk

#endif  // RQK_KDTREE_H
