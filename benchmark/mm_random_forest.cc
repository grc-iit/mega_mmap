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
#include "mega_mmap/vector_mega_mpi.h"

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

template<typename T>
class RandomForestClassifierMpi {
 public:
  using DataT = MM_VEC<T>;
  using TreeT = MM_VEC<std::unique_ptr<Node<T>>, true>;
  using GiniT = Gini<T>;
  using AssignT = MM_VEC<size_t>;

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
  hshm::UniformDistribution feature_dist_;
  std::vector<size_t> my_windows_;
  float tol_;
  int max_depth_;
  MPI_Comm world_;

 public:
  void Init(MPI_Comm world,
            const std::string &train_path,
            const std::string &test_path,
            int num_features,
            int num_cols,
            size_t window_size,
            int trees_per_proc = 4,
            float tol = .0001,
            int max_depth = 5){
    world_ = world;
    MPI_Comm_rank(world_, &rank_);
    MPI_Comm_size(world_, &nprocs_);
    dir_ = stdfs::path(train_path).parent_path();
    window_size_ = window_size;
    num_features_ = num_features;
    max_features_ = sqrt(num_features);
    num_cols_ = num_cols;
    tol_ = tol;
    max_depth_ = max_depth;
    // Load train data and partition
    data_.Init(train_path, MM_READ_ONLY);
    data_.BoundMemory(window_size_);
    data_.EvenPgas(rank_, nprocs_, data_.size());
    // Load test data and partition
    test_data_.Init(test_path, MM_READ_ONLY);
    test_data_.BoundMemory(window_size_);
    test_data_.EvenPgas(rank_, nprocs_, test_data_.size());
    // Memory bounding paramters
    num_windows_ = data_.size() / window_size_;
    windows_per_proc_ = num_windows_ / nprocs_;
    // Initialize final tree data structure
    trees_per_proc_ = trees_per_proc;
    trees_.Init(dir_ + "/trees",
                trees_per_proc_ * nprocs_,
                KILOBYTES(256),
                MM_WRITE_ONLY);
    trees_.EvenPgas(rank_, nprocs_, rank_ * trees_per_proc_ * nprocs_);
    // Initialize RNG
    feature_dist_.Seed(SEED);
    feature_dist_.Shape(0, num_features_ - 1);
  }

  void Run() {
    HILOG(kInfo, "Running random forest on rank {}", rank_);
    for (int i = 0; i < trees_per_proc_; ++i) {
      HILOG(kInfo, "Creating tree {} on rank {}", i, rank_);
      AssignT sample = SubsampleDataset();
      std::unique_ptr<Node<T>> root = std::make_unique<Node<T>>();
      CreateDecisionTree(root, nullptr, sample, 0);
      trees_[rank_ * trees_per_proc_ + i] = std::move(root);
    }
    trees_.Barrier(MM_READ_ONLY, world_);
    float error = Predict(test_data_);
    if (rank_ == 0) {
      HILOG(kInfo, "Prediction Accuracy: {}", 1 - error);
    }
  }

  float Predict(DataT &data) {
    AssignT preds;
    preds.Init(dir_ + "/preds", nprocs_, MM_WRITE_ONLY);
    preds.EvenPgas(rank_, nprocs_, data.size());
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
    preds.Barrier(MM_READ_ONLY, world_);
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

  AssignT SubsampleDataset() {
    Bounds bounds(rank_, nprocs_, data_.size());
    std::string sample_name =
        hshm::Formatter::format("{}/sample_{}_{}_{}",
                                dir_, 0, 0, rank_);
    AssignT sample;
    sample.Init(sample_name, bounds.size_, MM_WRITE_ONLY);
    sample.BoundMemory(window_size_);
    sample.EvenPgas(0, 1, data_.size());
    size_t off = 0;
    mm::UniformSampler sampler(MM_PAGE_SIZE,
                               data_.size(),
                               2354235 * (rank_ + 1));
    while (off < sample.size()) {
      sampler.SamplePage(sample, off);
    }
    sample.Hint(MM_READ_ONLY);
    return sample;
  }

  size_t SubsubsampleSize(AssignT &sample, size_t divisor) {
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
    return sample.size() / divisor;
  }

  AssignT SubsubsampleDataset(AssignT &sample,
                              size_t divisor,
                              uint64_t uuid,
                              int depth) {
    size_t new_size = SubsubsampleSize(sample, divisor);
    std::string new_sample_name =
        hshm::Formatter::format("{}/subsample_{}_{}_{}",
                                dir_, uuid, depth, rank_);
    AssignT new_sample;
    new_sample.Init(new_sample_name, new_size, MM_WRITE_ONLY);
    new_sample.BoundMemory(window_size_);
    new_sample.EvenPgas(0, 1, new_size);
    mm::UniformSampler sampler(MM_PAGE_SIZE,
                               sample.size(),
                               2354235 * (rank_ + 1));
    size_t off = 0;
    while (off < new_sample.size()) {
      sampler.SamplePage(sample, new_sample, off);
    }
    new_sample.Hint(MM_READ_ONLY);
    return new_sample;
  }

  void CreateDecisionTree(std::unique_ptr<Node<T>> &node,
                          const std::unique_ptr<Node<T>> &parent,
                          AssignT &sample,
                          uint64_t uuid) {
    HILOG(kInfo, "Creating decision tree on rank {} with {} samples",
          rank_, sample.size());
    // Decide the feature to split on
    size_t num_bootstrap_samples = 30;
    std::vector<Node<T>> stats(max_features_ * num_bootstrap_samples);
    hshm::UniformDistribution sample_dist;
    sample_dist.Seed(SEED * (rank_ + 1));
    size_t stat_idx = 0;
    for (int i = 0; i < num_bootstrap_samples; ++i) {
      std::vector<int> features = SubsampleFeatures();
      AssignT subsample = SubsubsampleDataset(
          sample, num_bootstrap_samples, uuid, node->depth_);
      sample_dist.Shape(0, subsample.size() - 1);
      for (int feature : features) {
        stats[stat_idx] = *node;
        stats[stat_idx].feature_ = feature;
        size_t sample_idx = sample_dist.GetSize();
        stats[stat_idx].joint_ = data_[subsample[sample_idx]];
        stats[stat_idx].left_ = std::make_unique<Node<T>>();
        stats[stat_idx].right_ = std::make_unique<Node<T>>();
        Split(stats[stat_idx], subsample, feature,
              stats[stat_idx].joint_);
        ++stat_idx;
      }
      subsample.Destroy();
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
      sample.Destroy();
      return;
    }
    // Calculate decision tree for subsamples
    AssignT left_sample, right_sample;
    size_t left_uuid = uuid;
    size_t right_uuid = uuid | (1 << node->depth_);
    std::string left_sample_name =
        hshm::Formatter::format("{}/sample_{}_{}_{}",
                                dir_, node->depth_ + 1,
                                left_uuid, rank_);
    std::string right_sample_name =
        hshm::Formatter::format("{}/sample_{}_{}_{}",
                                dir_, node->depth_ + 1,
                                right_uuid, rank_);
    left_sample.Init(left_sample_name,
                     node->left_->count_, MM_APPEND_ONLY);
    left_sample.BoundMemory(window_size_);
    right_sample.Init(right_sample_name,
                      node->right_->count_, MM_APPEND_ONLY);
    right_sample.BoundMemory(window_size_);
    DivideSample(*node, sample,
                 left_sample, right_sample);
    sample.Destroy();
    HILOG(kInfo, "Finished decision tree on rank {} with {} samples",
          rank_, sample.size());
    CreateDecisionTree(node->left_, node,
                       left_sample,
                       left_uuid);
    CreateDecisionTree(node->right_, node,
                       right_sample,
                       right_uuid);
  }

  void DivideSample(Node<T> &node,
                    AssignT &sample,
                    AssignT &left,
                    AssignT &right) {
    for (size_t i = 0; i < sample.size(); ++i) {
      size_t off = sample[i];
      if (data_[off].LessThan(node.joint_, node.feature_)) {
        left.emplace_back(off);
      } else {
        right.emplace_back(off);
      }
    }
    // left.flush_emplace(MPI_COMM_SELF, 0, 0);
    // right.flush_emplace(MPI_COMM_SELF, 0, 0);
    left.Hint(MM_READ_ONLY);
    right.Hint(MM_READ_ONLY);
  }

  void Split(Node<T> &node,
             AssignT &sample,
             int feature,
             T &joint) {
    // Group by output
    GiniT gini;

    // Assign points to left or right using joint
    size_t count[2] = {0};
    for (size_t i = 0 ; i < sample.size(); ++i) {
      size_t off = sample[i];
      T &row = data_[off];
      if (row.LessThan(joint, feature)) {
        count[0] += 1;
        gini.Induct(row, 0);
      } else {
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
  HILOG(kInfo, "Running random forest on rank {}", rank);
  std::string algo = argv[1];
  std::string train_path = argv[2];
  std::string test_path = argv[3];
  HILOG(kInfo, "HERE1");
  int nfeature = std::stoi(argv[4]);
  HILOG(kInfo, "HERE2");
  int ncol = nfeature + 1;
  HILOG(kInfo, "HERE3");
  size_t window_size = hshm::ConfigParse::ParseSize(argv[5]);
  HILOG(kInfo, "Parsed argument on {}", rank);

  if (algo == "mmap") {
  } else if (algo == "mega") {
    TRANSPARENT_HERMES();
    RandomForestClassifierMpi<ClassRow> rf;
    rf.Init(MPI_COMM_WORLD,
            train_path, test_path,
            nfeature, ncol, window_size);
    rf.Run();
  } else {
    HILOG(kFatal, "Unknown algorithm: {}", algo);
  }
  MPI_Finalize();
}
