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
#include "cereal/types/vector.hpp"

#include "mega_mmap/vector_mmap_mpi.h"
#include "mega_mmap/vector_mega_mpi.h"
#include "test_types.h"

namespace stdfs = std::filesystem;

template<typename T>
struct Node {
  int feature_;
  T joint_;
  float entropy_;
  int depth_;
  std::unique_ptr<Node> left_;
  std::unique_ptr<Node> right_;

  Node() {
    feature_ = -1;
    entropy_ = 0;
    depth_ = 0;
    left_ = nullptr;
    right_ = nullptr;
  }

  Node(const Node &other) {
    feature_ = other.feature_;
    joint_ = other.joint_;
    entropy_ = other.entropy_;
    depth_ = other.depth_;
    left_ = nullptr;
    right_ = nullptr;
  }

  Node &operator=(const Node &other) {
    feature_ = other.feature_;
    joint_ = other.joint_;
    entropy_ = other.entropy_;
    depth_ = other.depth_;
    left_ = nullptr;
    right_ = nullptr;
    return *this;
  }

  template<typename Ar>
  void serialize(Ar &ar) {
    ar(feature_, joint_, entropy_, depth_);
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
    typename NodeT,
    typename AssignT,
    typename OutT,
    typename BoolT,
    typename T>
class DbscanMpi {
 public:
  std::string dir_;
  std::string output_;
  DataT data_;
  TreeT trees_;
  int rank_;
  int nprocs_;
  size_t window_size_;
  size_t num_windows_;
  size_t windows_per_proc_;
  int num_features_;
  float dist_;
  int max_depth_;
  size_t min_pts_ = 32;
  std::string path_;
  std::unique_ptr<Node<T>> root_;
  int num_clusters_;
  std::unordered_map<T, int, T> agglo_;

 public:
  void Init(const std::string &path,
            size_t window_size,
            int rank, int nprocs,
            float dist){
    dir_ = stdfs::path(path).parent_path();
    output_ = dir_ + "/output.bin";
    data_.Init(path, MM_READ_ONLY);
    data_.BoundMemory(window_size);
    rank_ = rank;
    nprocs_ = nprocs;
    window_size_ = window_size / sizeof(T);
    num_features_ = data_[0].GetNumFeatures();
    num_windows_ = data_.size() / window_size_;
    windows_per_proc_ = num_windows_ / nprocs_;
    dist_ = dist;
    trees_.Init(dir_ + "/trees",
                nprocs_,
                KILOBYTES(512),
                MM_WRITE_ONLY);
    trees_.BoundMemory(window_size_);
    trees_.Pgas(rank_, 1);
    path_ = path;
    max_depth_ = 16;
  }

  AssignT RootSample() {
    AssignT sample;
    Bounds bounds(rank_, nprocs_, data_.size());
    HILOG(kInfo, "BOUNDS: {} {} {}", bounds.off_, bounds.size_, data_.size_)
    std::string sample_name =
        hshm::Formatter::format("{}/sample_{}_{}", dir_, 0, 0);
    sample.Init(sample_name, data_.size(), MM_WRITE_ONLY);
    sample.BoundMemory(window_size_);
    sample.Pgas(bounds.off_, bounds.size_);
    for (size_t i = 0; i < bounds.size_; ++i) {
      sample[bounds.off_ + i] = bounds.off_ + i;
    }
    sample.Barrier(MM_READ_ONLY);
    return sample;
  }

  void Run() {
    std::unique_ptr<Node<T>> root = std::make_unique<Node<T>>();
    AssignT sample = RootSample();
    CreateDecisionTree(root, sample, 0,
                       MPI_COMM_WORLD, 0, nprocs_);
    HILOG(kInfo, "Created decision tree")
    trees_[rank_] = std::move(root);
    trees_.Barrier(MM_READ_ONLY);
    std::unordered_set<T, T> joints = CombineDecisionTrees();
    Agglomerate(joints);
    Predict();
  }

  void CreateDecisionTree(std::unique_ptr<Node<T>> &node,
                          AssignT &sample,
                          uint64_t uuid,
                          MPI_Comm comm, int proc_off, int nprocs) {
    HILOG(kInfo, "{}: Beginning tree proc_off={} nprocs={} uuid={} depth={}",
          rank_, proc_off, nprocs, uuid, node->depth_);
    Bounds proc_bounds(rank_ - proc_off, nprocs,
                       sample.size());
    AssignT proc_sample = sample.Subset(proc_bounds.off_, proc_bounds.size_);
    proc_sample.Pgas(proc_bounds.off_, proc_bounds.size_);
    // Decide the feature to split on
    FindGlobalMedianAndFeature(*node, proc_sample,
                               node->depth_, uuid,
                               comm, proc_off, nprocs);
    node->left_ = std::make_unique<Node<T>>();
    node->right_ = std::make_unique<Node<T>>();
    node->left_->depth_ = node->depth_ + 1;
    node->right_->depth_ = node->depth_ + 1;
    // Determine whether to continue splitting
    if (!ShouldSplit(*node, comm, proc_off, nprocs, uuid)) {
      node->left_ = nullptr;
      node->right_ = nullptr;
      sample.Destroy();
      return;
    }
    // Divide subsamples to left and right
    AssignT left_sample, right_sample;
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
    left_sample.Init(left_sample_name, sample.size(), MM_APPEND_ONLY);
    left_sample.BoundMemory(window_size_);
    right_sample.Init(right_sample_name, sample.size(), MM_APPEND_ONLY);
    right_sample.BoundMemory(window_size_);
    DivideSample(*node, proc_sample,
                 left_sample, right_sample,
                 comm, proc_off, nprocs);
    sample.Barrier(0, comm);
    sample.Destroy();
    left_sample.Hint(MM_READ_ONLY);
    right_sample.Hint(MM_READ_ONLY);
    // Decide which nodes git which part of the sample
    int left_off = proc_off;
    int left_proc = nprocs / 2;
    int right_off = proc_off + left_proc;
    int right_proc = nprocs - left_proc;
    if (right_proc < 1) {
      right_off = proc_off;
      right_proc = 1;
    }
    if (left_proc < 1) {
      left_off = proc_off;
      left_proc = 1;
    }
    HILOG(kInfo, "{}: Finished tree proc_off={} nprocs={} uuid={} depth={}",
          rank_, proc_off, nprocs, uuid, node->depth_);
    if (left_sample.size() > window_size_ * nprocs) {
      if (left_off <= rank_ && rank_ < left_off + nprocs) {
        Bounds bounds(rank_ - proc_off, nprocs, left_sample.size());
        left_sample = left_sample.Subset(bounds.off_, bounds.size_);
        CreateDecisionTree(node->left_,
                           left_sample,
                           left_uuid,
                           comm, left_off, nprocs);
      }
    } else if (left_sample.size() > min_pts_) {
      MpiComm subcomm(comm, left_off, left_proc);
      if (left_off <= rank_ && rank_ < left_off + left_proc) {
        Bounds bounds(rank_ - left_off, left_proc, left_sample.size());
        left_sample = left_sample.Subset(bounds.off_, bounds.size_);
        CreateDecisionTree(node->left_,
                           left_sample,
                           left_uuid,
                           subcomm.comm_, left_off, left_proc);
      }
    } else {
      node->left_ = nullptr;
      left_sample.Destroy();
    }
    if (right_sample.size() > window_size_ * nprocs) {
      if (left_off <= rank_ && rank_ < left_off + nprocs) {
        Bounds bounds(rank_ - proc_off, nprocs, right_sample.size());
        right_sample = right_sample.Subset(bounds.off_, bounds.size_);
        CreateDecisionTree(node->right_,
                           right_sample,
                           right_uuid,
                           comm, left_off, nprocs);
      }
    } else if (right_sample.size() > min_pts_) {
      MpiComm subcomm(comm, right_off, right_proc);
      if (right_off <= rank_ && rank_ < right_off + right_proc) {
        Bounds bounds(rank_ - right_off, right_proc, right_sample.size());
        right_sample = right_sample.Subset(bounds.off_, bounds.size_);
        CreateDecisionTree(node->right_,
                           right_sample,
                           right_uuid,
                           subcomm.comm_, right_off, right_proc);
      }
    } else {
      node->right_ = nullptr;
      right_sample.Destroy();
    }
  }

  bool ShouldSplit(Node<T> &node,
                   MPI_Comm comm, int proc_off, int nprocs,
                   uint64_t uuid) {
    bool low_entropy = node.entropy_ <= dist_ / 2;
    bool is_max_depth = node.depth_ >= max_depth_;
    BoolT should_splits;
    std::string should_splits_name =
        hshm::Formatter::format("{}/should_split_{}_{}",
                                dir_, node.depth_, uuid);
    should_splits.Init(should_splits_name, nprocs, MM_WRITE_ONLY);
    should_splits.Pgas(rank_ - proc_off, 1);
    should_splits[rank_ - proc_off] = !(low_entropy || is_max_depth);
    should_splits.Barrier(MM_READ_ONLY, comm);
    bool ret = false;
    for (int i = 0; i < nprocs; ++i) {
      if (should_splits[i]) {
        ret = true;
        break;
      }
    }
    should_splits.Barrier(0, comm);
    should_splits.Destroy();
    return ret;
  }

  void FindGlobalMedianAndFeature(Node<T> &node, AssignT &sample,
                                  int depth, uint64_t uuid,
                                  MPI_Comm comm, int proc_off, int nprocs) {
    NodeT all_nodes;
    std::string all_nodes_name =
        hshm::Formatter::format("{}/nodes_{}_{}", dir_, depth, uuid);
    all_nodes.Init(all_nodes_name, nprocs, 256, MM_WRITE_ONLY);
    all_nodes.BoundMemory(window_size_);
    all_nodes.Pgas(rank_ - proc_off, 1);
    int subrank = rank_ - proc_off;
    all_nodes[subrank].resize(num_features_);
    FindLocalEntropy(all_nodes[subrank], sample);
    all_nodes.Barrier(MM_READ_ONLY, comm);
    // Determine the feature of interest by aggregating entropies
    std::vector<Node<T>> agg_fnodes;
    agg_fnodes.resize(num_features_);
    for (int feature = 0; feature < num_features_; ++feature) {
      agg_fnodes[feature].entropy_ = 0;
      for (int i = 0; i < nprocs; ++i) {
        std::vector<Node<T>> &fnodes = all_nodes[i];
        agg_fnodes[feature].entropy_ = std::max(
            fnodes[feature].entropy_,
            agg_fnodes[feature].entropy_);
        agg_fnodes[feature].feature_ = feature;
      }
    }
    auto it = std::max_element(agg_fnodes.begin(), agg_fnodes.end(),
                     [](const Node<T> &a, const Node<T> &b) {
                       return a.entropy_ < b.entropy_;
                     });
    node.entropy_ = it->entropy_;
    node.feature_ = it->feature_;
    // Calculate the median of the feature of interest
    std::vector<Node<T>> agg_mnodes;
    agg_mnodes.reserve(nprocs);
    for (int i = 0; i < nprocs; ++i) {
      Node<T> &fnode = all_nodes[i][node.feature_];
      agg_mnodes.emplace_back(fnode);
    }
    std::sort(agg_mnodes.begin(), agg_mnodes.end(),
              [](const Node<T> &a, const Node<T> &b) {
                return a.entropy_ < b.entropy_;
              });
    node.joint_ = agg_mnodes[agg_mnodes.size() / 2].joint_;
    sample.Barrier(0, comm);
    all_nodes.Destroy();
  }

  void FindLocalEntropy(std::vector<Node<T>> &nodes, AssignT &sample) {
    hshm::UniformDistribution dist;
    dist.Seed(2354235 * (rank_ + 1));
    dist.Shape(0, sample.size() - 1);
    // Randomly sample rows
    int subsample_size = 1024;
    if (subsample_size > sample.size()) {
      subsample_size = sample.size();
    }
    std::vector<T> subsample;
    subsample.reserve(subsample_size);
    for (size_t i = 0; i < subsample_size; ++i) {
      size_t idx = dist.GetSize();
      size_t off = sample[idx];
      subsample.emplace_back(data_[off]);
    }
    // Calculate the median of each feature
    for (int feature = 0; feature < num_features_; ++feature) {
      Node<T> &node = nodes[feature];
      std::sort(subsample.begin(), subsample.end(),
                [feature](const T &a, const T &b) {
                  return a.LessThan(b, feature);
                });
      node.feature_ = feature;
      node.joint_ = subsample[subsample.size() / 2];
      node.entropy_ = 0;
    }
    // Calculate entropy of each feature
    for (int feature = 0; feature < num_features_; ++feature) {
      Node<T> &node = nodes[feature];
      for (size_t i = 0; i < subsample.size(); ++i) {
        node.entropy_ = std::max<double>(subsample[i].Distance(node.joint_),
                                         node.entropy_);
      }
    }
  }

  void DivideSample(Node<T> &node,
                    AssignT &sample,
                    AssignT &left, AssignT &right,
                    MPI_Comm comm, int proc_off, int nprocs) {
    for (size_t i = 0; i < sample.size(); ++i) {
      size_t off = sample[i];
      if (data_[off].LessThan(node.joint_, node.feature_)) {
        left.emplace_back(off);
      } else {
        right.emplace_back(off);
      }
    }
    left.flush_emplace(comm, proc_off, nprocs);
    right.flush_emplace(comm, proc_off, nprocs);
  }

  std::unordered_set<T, T> CombineDecisionTrees() {
    std::unordered_set<T, T> joints;
    root_ = std::make_unique<Node<T>>(*trees_[0]);
    for (std::unique_ptr<Node<T>> &node : trees_) {
      CombineDecisionTree(root_, *node, joints);
    }
    return joints;
  }

  void CombineDecisionTree(std::unique_ptr<Node<T>> &root,
                           const Node<T> &node,
                           std::unordered_set<T, T> &joints) {
    if (node.left_ && node.left_->feature_ != -1) {
      if (!root->left_) {
        root->left_ = std::make_unique<Node<T>>(*node.left_);
      }
      CombineDecisionTree(root->left_, *node.left_, joints);
    }
    if (node.right_ && node.right_->feature_ != -1) {
      if (!root->right_) {
        root->right_ = std::make_unique<Node<T>>(*node.right_);
      }
      CombineDecisionTree(root->right_, *node.right_, joints);
    }
    if (node.left_ == nullptr &&
        node.right_ == nullptr) {
      if (node.entropy_ <= dist_ / 2) {
        joints.emplace(node.joint_);
      }
    }
  }

  void Agglomerate(std::unordered_set<T, T> &joints) {
    std::vector<std::vector<T>> agglo;
    for (const T &joint : joints) {
      int cluster_id;
      std::vector<T> &cluster = FindNearestAggloCluster(
          joint, agglo, cluster_id);
      cluster.emplace_back(joint);
      agglo_.emplace(joint, agglo.size());
    }
    num_clusters_ = agglo.size();
    if (rank_ == 0) {
      PrintClusters(agglo);
    }
  }

  void PrintClusters(std::vector<std::vector<T>> &agglo) {
    HILOG(kInfo, "Number of discovered clusters: {}", agglo.size())
    for (size_t i = 0; i < agglo.size(); ++i) {
      HILOG(kInfo, "Cluster {}: ", i);
      for (const T &joint : agglo[i]) {
        HILOG(kInfo, "  {}", joint.ToString());
      }
    }
  }

  std::vector<T>& FindNearestAggloCluster(
      const T &joint,
      std::vector<std::vector<T>> &agglo,
      int &cluster_id) {
    for (size_t i = 0; i < agglo.size(); ++i) {
      std::vector<T> &cluster = agglo[i];
      for (const T &cluster_joint : cluster) {
        if (joint.Distance(cluster_joint) < dist_) {
          cluster_id = i;
          return cluster;
        }
      }
    }
    agglo.emplace_back();
    cluster_id = agglo.size() - 1;
    return agglo[agglo.size() - 1];
  }

  void Predict() {
    OutT preds;
    preds.Init(output_, data_.size());
    preds.BoundMemory(window_size_);
    Bounds bounds(rank_, nprocs_, data_.size());
    preds.Pgas(bounds.off_, bounds.size_);
    size_t size_pp = data_.size() / nprocs_;
    size_t off = size_pp * rank_;
    size_t end = off + size_pp;
    if (end > data_.size()) {
      end = data_.size();
    }
    for (size_t i = off; i < end; ++i) {
      T joint = root_->Predict(data_[i]);
      int cluster = num_clusters_;
      if (agglo_.find(joint) != agglo_.end()) {
        cluster = agglo_[joint];
      }
      preds[i] = cluster;
    }
    preds.Barrier();
    preds.Close();
  }
};


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  if (argc != 5) {
    HILOG(kFatal, "USAGE: ./mm_dbscan [algo] [path] [window_size] [dist]");
  }
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  std::string algo = argv[1];
  std::string path = argv[2];
  size_t window_size = hshm::ConfigParse::ParseSize(argv[3]);
  float dist = std::stof(argv[4]);
  HILOG(kInfo, "Running {} on {} with window size {} with {} distance",
        algo, path, window_size, dist);

  if (algo == "mmap") {
    DbscanMpi<
        mm::VectorMmapMpi<RowND<3>>,
        mm::VectorMmapMpi<std::unique_ptr<Node<RowND<3>>>, true>,
        mm::VectorMmapMpi<std::vector<Node<RowND<3>>>, true>,
        mm::VectorMmapMpi<size_t>,
        mm::VectorMmapMpi<int>,
        mm::VectorMmapMpi<bool>,
        RowND<3>> dbscan;
    dbscan.Init(path, window_size, rank, nprocs, dist);
    dbscan.Run();
  } else if (algo == "mega") {
    DbscanMpi<
        mm::VectorMegaMpi<Row>,
        mm::VectorMegaMpi<std::unique_ptr<Node<Row>>, true>,
        mm::VectorMegaMpi<std::vector<Node<Row>>, true>,
        mm::VectorMegaMpi<size_t>,
        mm::VectorMegaMpi<int>,
        mm::VectorMegaMpi<int>,
        Row> dbscan;
    dbscan.Init(path, window_size, rank, nprocs, dist);
    dbscan.Run();
  } else {
    HILOG(kFatal, "Unknown algorithm: {}", algo);
  }
  MPI_Finalize();
}
