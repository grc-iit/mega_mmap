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

/**
 * RandomForest algorihtm:
 * 1. For each tree
 * 2. Select a subset of features
 * 3. Select a subset of the dataset
 * 3. Determine the feature with the highest entropy in the subset
 * 4.
 * */

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
  int num_trees_;
  int num_features_;
  int num_cols_;
  int max_features_;
  int num_bootstrap_samples_ = 30;
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
            int num_trees = 4,
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
    data_.Init(train_path, MM_READ_ONLY | MM_STAGE);
    data_.BoundMemory(window_size_);
    data_.EvenPgas(rank_, nprocs_, data_.size());
    data_.Allocate();
    // Load test data and partition
    test_data_.Init(test_path, MM_READ_ONLY | MM_STAGE);
    test_data_.BoundMemory(window_size_);
    test_data_.EvenPgas(rank_, nprocs_, test_data_.size());
    test_data_.Allocate();
    // Memory bounding paramters
    num_windows_ = data_.size() / window_size_;
    windows_per_proc_ = num_windows_ / nprocs_;
    // Initialize final tree data structure
    num_trees_ = num_trees;
    trees_.Init(dir_ + "/trees",
                num_trees_ ,
                KILOBYTES(256),
                MM_WRITE_ONLY);
    trees_.Allocate();
    // Initialize RNG
    feature_dist_.Seed(SEED);
    feature_dist_.Shape(0, num_features_ - 1);
  }

  void Run() {
    HILOG(kInfo, "Running random forest on rank {}", rank_);
    if (rank_ == 0) {
      trees_.PgasTxBegin(0,
                         num_trees_, MM_WRITE_ONLY);
    }
    for (int i = 0; i < num_trees_; ++i) {
      HILOG(kInfo, "Creating tree {} on rank {}", i, rank_);
      std::unique_ptr<Node<T>> root = std::make_unique<Node<T>>();
      CreateDecisionTree(root, nullptr, data_, 0,
                         MPI_COMM_WORLD,
                         rank_, nprocs_);
      if (rank_ == 0) {
        trees_[i] = std::move(root);
      }
    }
    if (rank_ == 0) {
      trees_.TxEnd();
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

  inline size_t SubsampleSize(DataT &data) {
    size_t subsample_size = data.size() / nprocs_ / num_bootstrap_samples_;
    if (subsample_size == 0) {
      subsample_size = 1;
    }
    return subsample_size;
  }

  void CreateDecisionTree(std::unique_ptr<Node<T>> &node,
                          const std::unique_ptr<Node<T>> &parent,
                          DataT &sample,
                          uint64_t uuid,
                          MPI_Comm comm, int rank, int nprocs) {
    HILOG(kInfo, "Creating decision tree on rank {} with {} samples",
          rank_, sample.size());
    // Decide the feature to split on
    std::vector<Node<T>> stats(max_features_ * num_bootstrap_samples_);
    size_t stat_idx = 0;
    for (int i = 0; i < num_bootstrap_samples_; ++i) {
      std::vector<int> features = SubsampleFeatures();
      size_t subsample_size = SubsampleSize(sample);
      sample.RandTxBegin(SEED, 0,  sample.size(),
                         (subsample_size + 1) * features.size(),
                         MM_READ_ONLY);
      for (int feature : features) {
        stats[stat_idx] = *node;
        stats[stat_idx].feature_ = feature;
        stats[stat_idx].joint_ = sample.template TxGet<mm::RandIterTx>();
        stats[stat_idx].left_ = std::make_unique<Node<T>>();
        stats[stat_idx].right_ = std::make_unique<Node<T>>();
        Split(stats[stat_idx], sample,  subsample_size,
              feature, stats[stat_idx].joint_, uuid,
              comm, rank, nprocs);
        ++stat_idx;
      }
      sample.TxEnd();
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
    // Divide sample into left and right
    DataT left_sample, right_sample;
    size_t left_uuid = uuid;
    size_t right_uuid = uuid | (1 << node->depth_);
    std::string left_sample_name =
        hshm::Formatter::format("{}/sample_{}_{}",
                                dir_, node->depth_ + 1,
                                left_uuid);
    std::string right_sample_name =
        hshm::Formatter::format("{}/sample_{}_{}",
                                dir_, node->depth_ + 1,
                                right_uuid);
    left_sample.Init(left_sample_name,
                     node->left_->count_, MM_APPEND_ONLY);
    left_sample.BoundMemory(window_size_);
    left_sample.Allocate();
    right_sample.Init(right_sample_name,
                      node->right_->count_, MM_APPEND_ONLY);
    right_sample.BoundMemory(window_size_);
    right_sample.Allocate();
    DivideSample(*node, sample,
                 left_sample, right_sample,
                 comm, rank, nprocs);
    sample.Barrier(MM_READ_ONLY, comm);
    sample.Destroy();
    HILOG(kInfo, "Finished decision tree on rank {} with {} samples, {} and {}",
          rank_, sample.size(), left_sample.size(), right_sample.size());

    // Create next decision tree nodes
    int left_off = rank;
    int left_proc = nprocs / 2;
    int right_off = rank + left_proc;
    int right_proc = nprocs - left_proc;
    if (right_proc < 1) {
      right_off = rank;
      right_proc = 1;
    }
    if (left_proc < 1) {
      left_off = rank;
      left_proc = 1;
    }
//    MpiComm left_subcomm(comm, left_off, left_proc);
//    MpiComm right_subcomm(comm, right_off, right_proc);
    CreateDecisionTree(node->left_, node,
                       left_sample,
                       left_uuid,
                       comm, rank, nprocs);
    CreateDecisionTree(node->right_, node,
                       right_sample,
                       right_uuid,
                       comm, rank, nprocs);
  }

  void DivideSample(Node<T> &node,
                    DataT &sample,
                    DataT &left,
                    DataT &right,
                    MPI_Comm comm, int rank, int nprocs) {
    sample.EvenPgas(rank, nprocs, sample.size());
    sample.SeqTxBegin(sample.local_off(), sample.local_size(),
                      MM_READ_ONLY);
    HILOG(kInfo, "{}: Dividing sample of size {}", rank, sample.local_size());
    for (size_t i = sample.local_off(); i < sample.local_size(); ++i) {
      T &elmt = sample[i];
      if (elmt.LessThan(node.joint_, node.feature_)) {
        left.emplace_back(elmt);
      } else {
        right.emplace_back(elmt);
      }
    }
    sample.TxEnd();
    left.FlushEmplace(comm);
    right.FlushEmplace(comm);
    left.Hint(MM_READ_ONLY);
    right.Hint(MM_READ_ONLY);
  }

  void Split(Node<T> &node,
             DataT &sample,
             size_t subsample_size,
             int feature,
             T &joint,
             size_t uuid,
             MPI_Comm comm, int rank, int nprocs) {
    TreeT nodes;
    // Calculate the entropy per-node
    std::string nodes_name = hshm::Formatter::format(dir_ + "/nodes_{}_{}",
                                                     node.depth_, uuid);
    nodes.Init(nodes_name,
               nprocs, MM_WRITE_ONLY);
    HILOG(kInfo, "{}: Created {} for {} procs", rank_, nodes_name, nprocs);
    nodes.EvenPgas(rank, nprocs, nprocs);
    nodes.Allocate();
    nodes.PgasTxBegin(rank, 1, MM_WRITE_ONLY);
    nodes[rank] = std::make_unique<Node<T>>();
    nodes[rank]->left_ = std::make_unique<Node<T>>();
    nodes[rank]->right_ = std::make_unique<Node<T>>();
    LocalSplit(*nodes[rank], sample, subsample_size, feature, joint);
    nodes.TxEnd();
    nodes.Barrier(MM_READ_ONLY, comm);

    // Sum up node entropies
    nodes.SeqTxBegin(0, nprocs, MM_READ_ONLY);
    for (size_t i = 0; i < nprocs; ++i) {
      auto &onode = *nodes[i];
      node.entropy_ += onode.entropy_;
      node.left_->count_ += onode.left_->count_;
      node.right_->count_ += onode.right_->count_;
    }
    nodes.TxEnd();
    nodes.Barrier(MM_READ_ONLY, comm);
    // nodes.Destroy();
  }

  void LocalSplit(Node<T> &node,
                  DataT &sample,
                  size_t subsample_size,
                  int feature,
                  T &joint) {
    // Group by output
    GiniT gini;

    // Assign points to left or right using joint
    size_t count[2] = {0};
    for (size_t i = 0 ; i < subsample_size; ++i) {
      T &row = sample.template TxGet<mm::RandIterTx>();
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
  int nfeature = std::stoi(argv[4]);
  int ncol = nfeature + 1;
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
