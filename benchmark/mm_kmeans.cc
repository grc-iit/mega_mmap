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

struct LocalMax {
  size_t idx_;
  double dist_;

  LocalMax() : idx_(0), dist_(0) {}

  LocalMax(size_t idx, double dist) : idx_(idx), dist_(dist) {}

  LocalMax(const LocalMax &other) : idx_(other.idx_), dist_(other.dist_) {}

  LocalMax &operator=(const LocalMax &other) {
    idx_ = other.idx_;
    dist_ = other.dist_;
    return *this;
  }

  LocalMax &operator+=(const LocalMax &other) {
    idx_ += other.idx_;
    dist_ += other.dist_;
    return *this;
  }

  LocalMax &operator/=(const size_t &other) {
    idx_ /= other;
    dist_ /= other;
    return *this;
  }
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

template<typename T>
struct Center {
  T center_;
  size_t count_;
  float inertia_;

  Center() : center_(0), count_(0), inertia_(0) {}

  Center(const T &center) : center_(center), count_(0), inertia_(0) {}

  void Zero() {
    center_ = T(0);
    count_ = 0;
    inertia_ = 0;
  }

  void Print() {
    HILOG(kInfo, "Center: ({}, {}, count={}, inertia={})",
          center_.x_, center_.y_,
          count_, inertia_);
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
  std::vector<Center<T>> ks_;
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
    HILOG(kInfo, "{}: Initialize mm kmeans", rank)
    dir_ = stdfs::path(path).parent_path();
    rank_ = rank;
    nprocs_ = nprocs;
    window_size_ = window_size;
    k_ = k;
    max_iter_ = max_iter;

    HILOG(kInfo, "{}: Beginning data definition", rank)
    data_.Init(path, MM_READ_ONLY);
    data_.BoundMemory(window_size);
    Bounds bounds(rank_, nprocs_, data_.size());
    data_.Pgas(bounds.off_, bounds.size_);
    HILOG(kInfo, "{}: Finished data definition", rank)

    off_ = bounds.off_;
    last_ = bounds.off_ + bounds.size_;
    inertia_diff_ = tol;
    min_inertia_ = min_inertia;
  }

  void Run() {
    // Select initial centroids
    HILOG(kInfo, "{}: Selecting first center", rank_)
    T first_k = SelectFirstCenter();
    HILOG(kInfo, "{}: Selected first center", rank_)
    ks_.emplace_back(first_k);
    for (int i = 1; i < k_; ++i) {
      HILOG(kInfo, "{}: Finding center {}", rank_, i)
      FindMax(ks_);
    }
    // Run kmeans
    KMeans();
    // Print results
    Print();
  }

  void Print() {
    if (rank_ == 0) {
      HILOG(kInfo, "Intertia: {}", inertia_);
      HILOG(kInfo, "Iterations: {}", iter_);
      for (int i = 0; i < k_; ++i) {
        ks_[i].Print();
      }
    }
  }

  /** 
   * Find the point that is furthest from all current centers.
   * */
  void FindMax(std::vector<Center<T>> &ks) {
    MaxT local_maxes;
    local_maxes.Init(dir_ + "/" + "max", nprocs_, MM_WRITE_ONLY);
    local_maxes.Pgas(rank_, 1);
    // Find point furthest away from all existing Ks
    LocalMax local_max = FindLocalMax(ks);
    // Wait for all processes to complete
    local_maxes[rank_] = local_max;
    local_maxes.Barrier(MM_READ_ONLY);
    // Find global max across processes
    LocalMax global_max = FindGlobalMax(local_maxes);
    ks.emplace_back(data_[global_max.idx_]);
    local_maxes.Barrier();
    local_maxes.Destroy();
  }

  /**
   * Find the point that is furthest from all current centers.
   * This aggregates the local maximums discovered by all
   * other procecsses.
   * */
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

  /**
   * The distance measure. Returns the smallest distance between the current
   * point and all center points. This way points that are near an existing
   * center are not selected.
   * */
  double MinOfCenterDists(T &cur_pt, std::vector<Center<T>> &ks) {
    double min_dist = std::numeric_limits<double>::max();
    for (size_t j = 0; j < ks.size(); ++j) {
      T &center = ks[j].center_;
      if (center == cur_pt) {
        return 0;
      }
      double dist = (center.Distance(cur_pt));
      if (dist < min_dist) {
        min_dist = dist;
      }
    }
    return min_dist;
  }

  /**
   * Find the point furthest away from the current set of centers in the local
   * branch of the dataset.
   * */
  LocalMax FindLocalMax(std::vector<Center<T>> &ks) {
    LocalMax local_max;
    for (size_t i = off_; i < last_; ++i) {
      T &cur_pt = data_[i];
      double dist = MinOfCenterDists(cur_pt, ks);
      if (dist > local_max.dist_) {
        local_max.dist_ = dist;
        local_max.idx_ = i;
      }
    }
    return local_max;
  }

  /**
   * Select the first center to initialize kmeans++.
   * */
  T SelectFirstCenter() {
    hshm::UniformDistribution dist;
    dist.Seed(23582);
    dist.Shape(0, data_.size_ - 1);
    size_t first_k = dist.GetSize();
    return data_[first_k];
  }

  /**
   * Runs the traditional KMeans algorithm. Assigns each point to a
   * center and then updates the center to be the average of all points
   * assigned to it.
   * */
  void KMeans() {
    inertia_ = 1;
    for (iter_ = 0; iter_ < max_iter_; ++iter_) {
      HILOG(kInfo, "{}: On iteration {}", rank_, iter_)
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

  /**
   * Assigns points to each current center and calculates the global
   * inertia and average of the assignment.
   * */
  float Assignment() {
    AssignT assign;
    SumT sum;
    assign.Init(dir_ + "/" + "assign", data_.size(), MM_WRITE_ONLY);
    assign.BoundMemory(window_size_);
    Bounds assign_bounds(rank_, nprocs_, data_.size());
    assign.Pgas(assign_bounds.off_, assign_bounds.size_);
    sum.Init(dir_ + "/" + "sum", nprocs_ * k_, MM_WRITE_ONLY);
    sum.Pgas(rank_ * k_, k_);
    // Calculate local assignment
    size_t off = off_, last = last_;
    for (size_t i = rank_ * k_; i < (rank_ + 1) * k_; ++i) {
      sum[i].Zero();
    }
    for (size_t i = off; i < last; ++i) {
      T row = data_[i];
      assign[i] = FindClosestCenter(row);
      sum[rank_ * k_ + assign[i]].row_ += row;
      sum[rank_ * k_ + assign[i]].count_ += 1;
      sum[rank_ * k_ + assign[i]].inertia_ +=
          pow(row.Distance(ks_[assign[i]].center_), 2);
    }
    sum.Barrier(MM_READ_ONLY);
    // Calculate global statistics from each local assignment
    float inertia = 0;
    for (int i = 0; i < ks_.size(); ++i) {
      T avg(0);
      size_t count = 0;
      float k_inertia = 0;
      for (int j = 0; j < nprocs_; ++j) {
        avg += sum[j * k_ + i].row_;
        count += sum[j * k_ + i].count_;
        k_inertia += sum[j * k_ + i].inertia_;
      }
      inertia += k_inertia;
      avg /= count;
      ks_[i].count_ = count;
      ks_[i].inertia_ = k_inertia;
      ks_[i].center_ = avg;
    }
    sum.Barrier();
    return inertia;
  }

  /**
   * Find the center closes to the row.
   * */
  size_t FindClosestCenter(T &row) {
    double min_dist = std::numeric_limits<double>::max();
    size_t min_idx = 0;
    for (size_t i = 0; i < ks_.size(); ++i) {
      double dist = row.Distance(ks_[i].center_);
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
  if (argc != 6) {
    HILOG(kFatal, "USAGE: ./mm_kmeans [algo] [path] [window_size] [k] [max_iter]");
  }
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  std::string algo = argv[1];
  std::string path = argv[2];
  size_t window_size = hshm::ConfigParse::ParseSize(argv[3]);
  int k = hshm::ConfigParse::ParseSize(argv[4]);
  int max_iter = std::stoi(argv[4]);
  HILOG(kInfo, "{}: Running {} on {} with window size {} with {} centers",
        rank, algo, path, window_size, k);

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
    TRANSPARENT_HERMES();
    KmeansMpi<
        mm::VectorMegaMpi<Row>,
        mm::VectorMegaMpi<LocalMax>,
        mm::VectorMegaMpi<size_t>,
        mm::VectorMegaMpi<RowSum>,
        Row> kmeans;
    kmeans.Init(path, rank, nprocs, window_size, k, max_iter);
    kmeans.Run();
  } else {
    HILOG(kFatal, "Unknown algorithm: {}", algo);
  }
  MPI_Finalize();
}
