//
// Created by lukemartinlogan on 9/13/23.
//

#include <string>
#include <mpi.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/config_parse.h"
#include "hermes_shm/util/random.h"
#include <filesystem>
#include <algorithm>

#include "mega_mmap/vector_mmap_mpi.h"
#include "test_types.h"

namespace stdfs = std::filesystem;

struct Window {
  size_t off_;
  size_t size_;
};

template<typename T>
struct Node {
  int feature_;
  T joint_;
  float entropy_;
  size_t count_;
  int depth_;
  std::unique_ptr<Node> left_;
  std::unique_ptr<Node> right_;

  Node() {
    feature_ = -1;
    entropy_ = 0;
    count_ = 0;
    depth_ = 0;
    left_ = nullptr;
    right_ = nullptr;
  }

  Node(const Node &other) {
    feature_ = other.feature_;
    joint_ = other.joint_;
    entropy_ = other.entropy_;
    count_ = other.count_;
    depth_ = other.depth_;
  }

  Node &operator=(const Node &other) {
    feature_ = other.feature_;
    joint_ = other.joint_;
    entropy_ = other.entropy_;
    count_ = other.count_;
    depth_ = other.depth_;
    return *this;
  }

  template<typename Ar>
  void serialize(Ar &ar) {
    ar(feature_, joint_, entropy_, count_, depth_);
    ar(left_, right_);
  }

  T Predict(const T &row) {
    if (row.LessThan(joint_, feature_)) {
      if (left_) {
        return left_->Predict(row);
      }
    } else {
      if (right_) {
        return right_->Predict(row);
      }
    }
    return joint_;
  }
};

template<typename DataT,
         typename TreeT,
         typename GiniT,
         typename PredT,
         typename T>
class RandomForestClassifierMpi {
 public:
  std::string dir_;
  DataT data_;
  DataT test_data_;
  TreeT trees_;
  int rank_;
  int nprocs_;
  size_t window_size_;
  size_t num_windows_;
  size_t windows_per_proc_;
  int trees_per_proc_;
  int num_features_;
  int num_cols_;
  int max_features_;
  hshm::UniformDistribution shuffle_;
  hshm::UniformDistribution feature_dist_;
  std::vector<size_t> my_windows_;
  float tol_;
  int max_depth_;

 public:
  void Init(const std::string &train_path,
            const std::string &test_path,
            int num_features,
            int num_cols,
            size_t window_size,
            int rank, int nprocs,
            int trees_per_proc = 4,
            float tol = .0001,
            int max_depth = 5){
    dir_ = stdfs::path(train_path).parent_path();
    data_.Init(train_path, MM_READ_ONLY);
    test_data_.Init(test_path, MM_READ_ONLY);
    rank_ = rank;
    nprocs_ = nprocs;
    window_size_ = window_size / sizeof(T);
    num_features_ = num_features;
    max_features_ = sqrt(num_features);
    num_cols_ = num_cols;
    num_windows_ = data_.size() / window_size_;
    windows_per_proc_ = num_windows_ / nprocs_;
    tol_ = tol;
    max_depth_ = max_depth;
    trees_per_proc_ = trees_per_proc;
    trees_.Init(dir_ + "/trees",
                trees_per_proc_ * nprocs_,
                KILOBYTES(256),
                MM_WRITE_ONLY);

    shuffle_.Seed(2354235 * (rank_ + 1));
    shuffle_.Shape(0, num_windows_ - 1);
    feature_dist_.Seed(2354235);
    feature_dist_.Shape(0, num_features_ - 1);
  }

  void Run() {
    for (int i = 0; i < trees_per_proc_; ++i) {
      std::vector<size_t> sample = SubsampleDataset();
      std::unique_ptr<Node<T>> root = std::make_unique<Node<T>>();
      CreateDecisionTree(root, nullptr, sample);
      trees_[rank_ * trees_per_proc_ + i] = std::move(root);
    }
    trees_.Barrier(MM_READ_ONLY);
    float error = Predict(test_data_);
    HILOG(kInfo, "Prediction Accuracy: {}", 1 - error);
  }

  float Predict(DataT &data) {
    PredT preds;
    preds.Init(dir_ + "/preds", nprocs_, MM_WRITE_ONLY);
    size_t err_count = 0;
    // Get the offset and size of data to predict
    size_t size_pp = test_data_.size() / nprocs_;
    size_t off = rank_ * size_pp;
    if (off + size_pp > test_data_.size()) {
      size_pp = test_data_.size() - off;
    }
    // Predict and calculate classification error
    PredictLocal(data, off, size_pp, err_count);
    preds[rank_] = err_count;
    preds.Barrier(MM_READ_ONLY);
    float err = 0;
    for (int i = 0; i < nprocs_; ++i) {
      err += preds[i];
    }
    return err / data.size();
  }

  void PredictLocal(DataT &data, size_t off, size_t size,
                    size_t &err_count) {
    err_count = 0;
    size_t count = 0;
    for (size_t i = off; i < size; ++i) {
      T &row = data[i];
      std::unordered_map<float, int> rows;
      for (int i = 0; i < trees_.size(); ++i) {
        std::unique_ptr<Node<T>> &tree = trees_[i];
        T pred = tree->Predict(row);
        if (rows.find(pred.last()) == rows.end()) {
          rows[pred.last()] = 0;
        }
        rows[pred.last()] += 1;
      }
      auto it = std::max_element(rows.begin(), rows.end(),
                                 [](const std::pair<float, int> &a,
                                    const std::pair<float, int> &b) {
                                   return a.second < b.second;
                                 });
      float pred = it->first;
      err_count += (pred != row.last());
      count += 1;
    }
  }

  std::vector<int> SubsampleFeatures() {
    std::vector<int> features(max_features_);
    for (int i = 0; i < max_features_; ++i) {
      features[i] = feature_dist_.GetInt();
    }
    return features;
  }

  std::vector<size_t> SubsampleDataset() {
    size_t sample_size = 0;
    std::vector<Window> windows(windows_per_proc_);
    for (size_t i = 0; i < windows_per_proc_; ++i) {
      size_t idx = shuffle_.GetSize();
      windows[i].off_ = idx * window_size_;
      windows[i].size_ = window_size_;
      if (windows[i].off_ + windows[i].size_ > data_.size()) {
        windows[i].size_ = data_.size() - windows[i].off_;
      }
      sample_size += windows[i].size_;
    }
    // Specify sample offsets
    std::vector<size_t> sample(sample_size);
    size_t smpl_idx = 0;
    for (Window &window : windows) {
      for (size_t i = 0; i < window.size_; ++i) {
        sample[smpl_idx] = window.off_ + i;
        smpl_idx += 1;
      }
    }
    return sample;
  }

  std::vector<size_t> SubsubsampleDataset(const std::vector<size_t> &sample,
                                          size_t divisor) {
    if (divisor > sample.size()) {
      size_t i = 5;
      while (true) {
        divisor = sample.size() / i;
        if (divisor > 0) {
          break;
        }
        --i;
      }
    }
    size_t new_size = sample.size() / divisor;
    std::vector<size_t> new_sample(new_size);
    hshm::UniformDistribution dist;
    dist.Seed(2354235 * (rank_ + 1));
    dist.Shape(0, sample.size() - 1);
    for (size_t i = 0; i < new_sample.size(); ++i) {
      size_t idx = dist.GetSize();
      new_sample[i] = sample[idx];
    }
    return new_sample;
  }

  void CreateDecisionTree(std::unique_ptr<Node<T>> &node,
                          const std::unique_ptr<Node<T>> &parent,
                          const std::vector<size_t> &sample) {
    // Decide the feature to split on
    size_t num_bootstrap_samples = 30;
    std::vector<size_t> assign;
    std::vector<Node<T>> stats(max_features_ * num_bootstrap_samples);
    hshm::UniformDistribution sample_dist;
    sample_dist.Seed(2354235 * (rank_ + 1));
    size_t stat_idx = 0;
    for (int i = 0; i < num_bootstrap_samples; ++i) {
      std::vector<int> features = SubsampleFeatures();
      std::vector<size_t> subsample = SubsubsampleDataset(
          sample, num_bootstrap_samples);
      sample_dist.Shape(0, subsample.size() - 1);
      assign.resize(subsample.size());
      for (int feature : features) {
        stats[stat_idx] = *node;
        stats[stat_idx].feature_ = feature;
        size_t sample_idx = sample_dist.GetSize();
        stats[stat_idx].joint_ = data_[subsample[sample_idx]];
        stats[stat_idx].left_ = std::make_unique<Node<T>>();
        stats[stat_idx].right_ = std::make_unique<Node<T>>();
        Split(stats[stat_idx], subsample, feature,
              stats[stat_idx].joint_, assign);
        ++stat_idx;
      }
    }
    auto it = std::min_element(stats.begin(), stats.end(),
                               [](const Node<T> &a, const Node<T> &b) {
                                 return a.entropy_ < b.entropy_;
                               });
    Node<T> &min_stat = *it;
    (*node) = min_stat;
    node->left_ = std::move(min_stat.left_);
    node->right_ = std::move(min_stat.right_);
    node->entropy_ = min_stat.entropy_;
    node->left_->depth_ = node->depth_ + 1;
    node->right_->depth_ = node->depth_ + 1;
    bool low_entropy = node->entropy_ <= tol_;
    bool is_max_depth = node->depth_ >= max_depth_;
    if (low_entropy || is_max_depth) {
      node->left_ = nullptr;
      node->right_ = nullptr;
      return;
    }
    // Calculate decision tree for subsamples
    std::vector<size_t> left_sample, right_sample;
    DivideSample(*node, sample,
                 left_sample, right_sample);
    CreateDecisionTree(node->left_, node,
                       left_sample);
    CreateDecisionTree(node->right_, node,
                       right_sample);
  }

  void DivideSample(Node<T> &node,
                    const std::vector<size_t> &sample,
                    std::vector<size_t> &left,
                    std::vector<size_t> &right) {
    left.reserve(node.left_->count_);
    right.reserve(node.right_->count_);
    for (size_t i = 0; i < sample.size(); ++i) {
      size_t off = sample[i];
      if (data_[off].LessThan(node.joint_, node.feature_)) {
        left.emplace_back(off);
      } else {
        right.emplace_back(off);
      }
    }
  }

  void Split(Node<T> &node,
              const std::vector<size_t> &sample,
              int feature,
              T &joint,
              std::vector<size_t> &assign) {
    // Group by output
    GiniT gini;

    // Assign points to left or right using joint
    size_t count[2] = {0};
    for (size_t i = 0 ; i < sample.size(); ++i) {
      size_t off = sample[i];
      T &row = data_[off];
      if (row.LessThan(joint, feature)) {
        assign[i] = 0;
        count[0] += 1;
        gini.Induct(row, 0);
      } else {
        assign[i] = 1;
        count[1] += 1;
        gini.Induct(row, 1);
      }
    }
    node.left_->count_ = count[0];
    node.right_->count_ = count[1];
    node.entropy_ = gini.Get();
  }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  if (argc != 6) {
    HILOG(kFatal, "USAGE: ./mm_random_forest [algo] [train_path] [test_path] "
                  "[nfeature] [window_size]");
  }
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  std::string algo = argv[1];
  std::string train_path = argv[2];
  std::string test_path = argv[3];
  int nfeature = std::stoi(argv[4]);
  int ncol = nfeature + 1;
  size_t window_size = hshm::ConfigParse::ParseSize(argv[5]);
  HILOG(kInfo, "Running random forest on rank {}", rank);

  if (algo == "mmap") {
    RandomForestClassifierMpi<
        mm::VectorMmapMpi<ClassRow>,
        mm::VectorMmapMpi<std::unique_ptr<Node<ClassRow>>, true>,
        Gini<ClassRow>,
        mm::VectorMmapMpi<size_t>,
        ClassRow> rf;
    rf.Init(train_path, test_path,
            nfeature, ncol, window_size,
            rank, nprocs);
    rf.Run();
  } else if (algo == "mega") {
  } else {
    HILOG(kFatal, "Unknown algorithm: {}", algo);
  }
  MPI_Finalize();
}