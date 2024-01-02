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

namespace stdfs = std::filesystem;

struct Row {
  float x_;
  float y_;

  Row() = default;

  Row(float x, float y) : x_(x), y_(y) {}

  Row(float x) : x_(x), y_(x) {}

  Row(const Row &other) {
    x_ = other.x_;
    y_ = other.y_;
  }

  Row &operator=(const Row &other) {
    x_ = other.x_;
    y_ = other.y_;
    return *this;
  }

  Row &operator+=(const Row &other) {
    x_ += other.x_;
    y_ += other.y_;
    return *this;
  }

  Row &operator/=(const size_t &other) {
    x_ /= other;
    y_ /= other;
    return *this;
  }

  double Distance(const Row &other) {
    return (x_ - other.x_) * (x_ - other.x_) +
        (y_ - other.y_) * (y_ - other.y_);
  }
};

struct LocalMax {
  size_t idx_ = 0;
  double dist_ = 0;
};

struct RowSum {
  Row row_;
  size_t count_;
  float inertia_;

  void Zero() {
    row_ = Row(0);
    count_ = 0;
    inertia_ = 0;
  }
};

template<typename DataT,
         typename MaxT,
         typename AssignT,
         typename SumT,
         typename T>
class KmeansMpi {
 public:
  std::string dir_;
  DataT data_;
  int rank_;
  int nprocs_;
  size_t window_size_;
  int k_;
  std::vector<Row> ks_;
  int max_iter_;
  size_t off_;
  size_t last_;
  size_t iter_;
  double inertia_;
  float inertia_diff_;
  float min_inertia_;

 public:
  void Init(const std::string &path,
            int rank, int nprocs,
            size_t window_size, int k, int max_iter = 300,
            float tol = .0001, float min_inertia = .1){
    dir_ = stdfs::path(path).parent_path();
    data_.Init(path);
    rank_ = rank;
    nprocs_ = nprocs;
    window_size_ = window_size;
    k_ = k;
    max_iter_ = max_iter;

    size_t data_per_proc = data_.size() / nprocs_;
    off_ = rank_ * data_per_proc;
    last_ = (rank_ + 1) * data_per_proc;
    if (last_ > data_.size()) {
      last_ = data_.size();
    }
    inertia_diff_ = tol;
    min_inertia_ = min_inertia;
  }

  void Run() {
    // Select initial centroids
    size_t first_k = SelectFirstK();
    std::vector<size_t> ks;
    ks.emplace_back(first_k);
    for (int i = 1; i < k_; ++i) {
      FindMax(ks);
    }
    for (int i = 0; i < k_; ++i) {
      ks_.emplace_back(data_[ks[i]]);
    }
    // Run kmeans
    KMeans();
    // Print results
    Print();
  }

  void Print() {
    HILOG(kInfo, "Intertia: {}", inertia_);
    HILOG(kInfo, "Iterations: {}", iter_);
    for (int i = 0; i < k_; ++i) {
      HILOG(kInfo, "Center {}: ({}, {})", i, ks_[i].x_, ks_[i].y_);
    }
  }

  void FindMax(std::vector<size_t> &ks) {
    MaxT local_maxes;
    local_maxes.Init(dir_ + "/" + "max", nprocs_);
    // Get maximum of all proc windows
    LocalMax local_max = FindLocalMax(ks);
    // Wait for all windows to finish
    local_maxes[rank_] = local_max;
    local_maxes.Barrier();
    // Find global max
    LocalMax global_max = FindGlobalMax(local_maxes);
    ks.emplace_back(global_max.idx_);
    local_maxes.Barrier();
  }

  LocalMax FindGlobalMax(MaxT &local_maxes) {
    LocalMax max;
    for (int i = 0; i < nprocs_; ++i) {
      LocalMax &cur_max = local_maxes[i];
      if (cur_max.dist_ > max.dist_) {
        max = cur_max;
      }
    }
    return max;
  }

  LocalMax FindLocalMax(std::vector<size_t> &ks) {
    size_t window_size = window_size_;
    LocalMax local_max;
    size_t off = off_, last = last_;
    while (off < last) {
      if (off + window_size > last) {
        window_size = last - off;
      }
      FindWindowMax(ks, off, off + window_size,
                    local_max.dist_, local_max.idx_);
      off += window_size;
    }
    return local_max;
  }

  void FindWindowMax(std::vector<size_t> &ks,
                     size_t off, size_t last,
                     double &max_dist,
                     size_t &max_idx) {
    for (size_t i = off; i < last; ++i) {
      Row &cur_pt = data_[i];
      double dist = 1;
      for (size_t j = 0; j < ks.size(); ++j) {
        Row &center = data_[ks[j]];
        dist *= center.Distance(cur_pt);
      }
      if (dist > max_dist) {
        if (std::find(ks.begin(), ks.end(), i) != ks.end()) {
          continue;
        }
        max_dist = dist;
        max_idx = i;
      }
    }
  }

  size_t SelectFirstK() {
    hshm::UniformDistribution dist;
    dist.Seed(23582);
    dist.Shape(0, data_.size_ - 1);
    size_t first_k = dist.GetSize();
    return first_k;
  }

  void KMeans() {
    inertia_ = 1;
    for (iter_ = 0; iter_ < max_iter_; ++iter_) {
      float cur_inertia = Assignment();
      if (cur_inertia <= min_inertia_) {
        break;
      }
      if (abs(cur_inertia - inertia_) / inertia_ < inertia_diff_) {
        break;
      }
      inertia_ = cur_inertia;
    }
  }

  float Assignment() {
    AssignT assign;
    SumT sum;
    assign.Init(dir_ + "/" + "assign", data_.size());
    sum.Init(dir_ + "/" + "sum", nprocs_ * k_);
    size_t off = off_, last = last_;
    for (size_t i = rank_ * k_; i < (rank_ + 1) * k_; ++i) {
      sum[i].Zero();
    }
    for (size_t i = off; i < last; ++i) {
      Row row = data_[i];
      assign[i] = FindClosestCenter(row);
      sum[rank_ * k_ + assign[i]].row_ += row;
      sum[rank_ * k_ + assign[i]].count_ += 1;
      sum[rank_ * k_ + assign[i]].inertia_ += row.Distance(ks_[assign[i]]);
    }
    sum.Barrier();
    float inertia = 0;
    for (int i = 0; i < ks_.size(); ++i) {
      Row avg(0);
      size_t count = 0;
      for (int j = 0; j < nprocs_; ++j) {
        avg += sum[j * k_ + i].row_;
        count += sum[j * k_ + i].count_;
        inertia += sum[j * k_ + i].inertia_;
      }
      avg /= count;
      ks_[i] = avg;
    }
    sum.Barrier();
    return inertia;
  }
  
  size_t FindClosestCenter(Row &row) {
    double min_dist = std::numeric_limits<double>::max();
    size_t min_idx = 0;
    for (size_t i = 0; i < ks_.size(); ++i) {
      Row &center = ks_[i];
      double dist = row.Distance(center);
      if (dist < min_dist) {
        min_dist = dist;
        min_idx = i;
      }
    }
    return min_idx;
  }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  if (argc < 5) {
    HILOG(kFatal, "USAGE: ./kmeans [algo] [path] [window_size] [k] [max_iter]");
  }
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  std::string algo = argv[1];
  std::string path = argv[2];
  size_t window_size = hshm::ConfigParse::ParseSize(argv[3]);
  int k = hshm::ConfigParse::ParseSize(argv[4]);
  int max_iter = std::stoi(argv[4]);
  HILOG(kInfo, "Running {} on {} with window size {} with {} centers", algo, path, window_size, k);

  if (algo == "mmap") {
    KmeansMpi<
        mm::VectorMmapMpi<Row>,
        mm::VectorMmapMpi<LocalMax>,
        mm::VectorMmapMpi<size_t>,
        mm::VectorMmapMpi<RowSum>,
        Row> kmeans;
    kmeans.Init(path, rank, nprocs, window_size, k, max_iter);
    kmeans.Run();
  } else if (algo == "mega") {
  } else {
    HILOG(kFatal, "Unknown algorithm: {}", algo);
  }
  MPI_Finalize();
}
