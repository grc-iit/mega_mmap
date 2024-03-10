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
#include <cmath>

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

template<typename T>
struct RowSum {
  T row_;
  size_t count_;
  float inertia_;

  void Zero() {
    row_ = T(0);
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

template<typename T>
class KMeans {
 public:
  using DataT = MM_VEC<T>;
  using MaxT = MM_VEC<LocalMax>;
  using AssignT = MM_VEC<size_t>;
  using SumT = MM_VEC<RowSum<T>>;

 public:
  std::string dir_;
  DataT data_;
  int rank_;
  int nprocs_;
  size_t window_size_;
  int k_;
  std::vector<Center<T>> ks_;
  int max_iter_;
  size_t iter_;
  double inertia_;
  float min_inertia_;
  float tol_;
  MPI_Comm world_;

 public:
  /** Initialization with no known centers yet */
  void Init(MPI_Comm world,
            const std::string &path,
            size_t window_size,
            int k, int max_iter = 300,
            float tol = .0001, float min_inertia = .1) {
    world_ = world;
    MPI_Comm_rank(world_, &rank_);
    MPI_Comm_size(world_, &nprocs_);
    HILOG(kInfo, "{}: Initialize mm kmeans", rank_)
    dir_ = stdfs::path(path).parent_path();
    window_size_ = window_size;
    k_ = k;
    max_iter_ = max_iter;

    HILOG(kInfo, "{}: Beginning data definition", rank_)
    data_.Init(path, MM_READ_ONLY | MM_STAGE_READ_FROM_BACKEND);
    data_.BoundMemory(window_size);
    data_.EvenPgas(rank_, nprocs_, data_.size());
    data_.Allocate();
    HILOG(kInfo, "{}: Finished data definition", rank_)
    tol_ = tol;
    min_inertia_ = min_inertia;
  }

  /** Initialization with known centers */
  void Init(MPI_Comm world,
            DataT &data,
            size_t window_size,
            int k,
            int max_iter = 300,
            float tol = .0001,
            float min_inertia = .1) {
    world_ = world;
    MPI_Comm_rank(world_, &rank_);
    MPI_Comm_size(world_, &nprocs_);
    HILOG(kInfo, "{}: Initialize mm kmeans with known centers", rank_)
    data_ = data;
    dir_ = stdfs::path(data_.path_).parent_path();
    window_size_ = window_size;
    k_ = k;
    ks_.reserve(k);
    max_iter_ = max_iter;
    tol_ = tol;
    min_inertia_ = min_inertia;
    data_.EvenPgas(rank_, nprocs_, data_.size());
  }

  /**
   * Runs the traditional KMeans algorithm. Assigns each point to a
   * center and then updates the center to be the average of all points
   * assigned to it.
   * */
  void Fit() {
    inertia_ = 1;
    for (iter_ = 0; iter_ < max_iter_; ++iter_) {
      HILOG(kInfo, "{}: On iteration {}", rank_, iter_)
      float cur_inertia = Assignment();
      if (cur_inertia <= min_inertia_) {
        break;
      }
      if (abs(cur_inertia - inertia_) / inertia_ < tol_) {
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
    // Initialize assign vector
    AssignT assign;
    assign.Init(dir_ + "/" + "assign", data_.size(), MM_WRITE_ONLY);
    assign.BoundMemory(window_size_);
    assign.EvenPgas(rank_, nprocs_, data_.size());
    assign.Allocate();

    // Initialize sum vector
    SumT sum;
    sum.Init(dir_ + "/" + "sum", nprocs_ * k_, MM_WRITE_ONLY);
    sum.EvenPgas(rank_, nprocs_, nprocs_ * k_);
    sum.Allocate();

    // Zero out sums
    {
      auto tx = sum.SeqTxBegin(sum.local_off(), sum.local_size(), MM_WRITE_ONLY);
      for (size_t i = 0; i < sum.local_size(); ++i) {
        (*tx).Zero();
        ++tx;
      }
      sum.TxEnd(tx);
    }

    // Calculate local assignment
    {
      auto data_tx = data_.SeqTxBegin(data_.local_off(),
                                      data_.local_size(),
                                      MM_READ_ONLY);
      auto sum_tx = sum.SeqTxBegin(sum.local_off(),
                                   sum.local_size(),
                                   MM_WRITE_ONLY);
      auto assign_tx = assign.SeqTxBegin(assign.local_off(),
                                         assign.local_size(),
                                         MM_WRITE_ONLY);
      size_t off = data_.local_off();
      for (size_t i = off; i < data_.local_last(); ++i) {
        if ((i - off) % (16 * MM_PAGE_SIZE) == 0) {
          HILOG(kInfo, "{}: We are {}% done", rank_,
                (i - off) * 100.0 / data_.local_size())
        }
        T &row = *data_tx;
        size_t &assign_i = *assign_tx;
        assign_i = FindClosestCenter(row);
        (*sum_tx).row_ += row;
        (*sum_tx).count_ += 1;
        (*sum_tx).inertia_ +=
            pow(row.Distance(ks_[assign_i].center_), 2);

        ++data_tx;
        ++sum_tx;
        ++assign_tx;
      }
      data_.TxEnd(data_tx);
      sum.TxEnd(sum_tx);
      assign.TxEnd(assign_tx);
    }
    sum.Barrier(MM_READ_ONLY, world_);
    // Calculate global statistics from each local assignment
    float inertia = 0;
    {
      auto sum_tx = sum.SeqTxBegin(0, sum.size(), MM_READ_ONLY);
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
      sum.TxEnd(sum_tx);
    }
    sum.Barrier(0, world_);
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

  /** Print final centers and inertia */
  void Print() {
    if (rank_ == 0) {
      HILOG(kInfo, "Intertia: {}", inertia_);
      HILOG(kInfo, "Iterations: {}", iter_);
      for (int i = 0; i < k_; ++i) {
        ks_[i].Print();
      }
    }
  }
};

template<typename T>
class KMeansPpMpi : public KMeans<T> {
 public:
  using DataT = MM_VEC<T>;
  using MaxT = MM_VEC<LocalMax>;
  using AssignT = MM_VEC<size_t>;
  using SumT = MM_VEC<RowSum<T>>;
  using KMeans<T>::dir_;
  using KMeans<T>::rank_;
  using KMeans<T>::ks_;
  using KMeans<T>::k_;
  using KMeans<T>::data_;
  using KMeans<T>::nprocs_;
  using KMeans<T>::window_size_;
  using KMeans<T>::max_iter_;
  using KMeans<T>::tol_;
  using KMeans<T>::min_inertia_;
  using KMeans<T>::Print;
  using KMeans<T>::Fit;
  using KMeans<T>::world_;

 public:
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
    Fit();
  }

  /**
   * Find the point that is furthest from all current centers.
   * */
  void FindMax(std::vector<Center<T>> &ks) {
    // Initialize local max vector
    MaxT local_maxes;
    local_maxes.Init(dir_ + "/" + "max", nprocs_, MM_WRITE_ONLY);
    local_maxes.EvenPgas(rank_, nprocs_, nprocs_);
    local_maxes.Allocate();
    // Find point furthest away from all existing Ks
    LocalMax local_max = FindLocalMax(ks);
    // Wait for all processes to complete
    {
      auto tx = local_maxes.SeqTxBegin(rank_, 1, MM_WRITE_ONLY);
      local_maxes[rank_] = local_max;
      local_maxes.TxEnd(tx);
    }
    local_maxes.Barrier(MM_READ_ONLY, world_);
    // Find global max across processes
    LocalMax global_max = FindGlobalMax(local_maxes);
    ks.emplace_back(data_[global_max.idx_]);
    local_maxes.Barrier(0, world_);
    local_maxes.Destroy();
  }

  /**
   * Find the point that is furthest from all current centers.
   * This aggregates the local maximums discovered by all
   * other procecsses.
   * */
  LocalMax FindGlobalMax(MaxT &local_maxes) {
    LocalMax max;
    auto tx = local_maxes.SeqTxBegin(0, local_maxes.size(), MM_READ_ONLY);
    for (int i = 0; i < nprocs_; ++i) {
      LocalMax &cur_max = local_maxes[i];
      if (cur_max.dist_ > max.dist_) {
        max = cur_max;
      }
      ++tx;
    }
    local_maxes.TxEnd(tx);
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
    size_t off = data_.local_off();
    size_t last = data_.local_last();
    auto tx = data_.SeqTxBegin(off, last - off, MM_READ_ONLY);
    for (size_t i = off; i < last; ++i) {
      if ((i - off) % (256 * MM_PAGE_SIZE) == 0) {
        HILOG(kInfo, "{}: We are {}% done", rank_,
              (i - off) * 100.0 / (last - off))
      }
      T &cur_pt = *tx;
      double dist = MinOfCenterDists(cur_pt, ks);
      if (dist > local_max.dist_) {
        local_max.dist_ = dist;
        local_max.idx_ = i;
      }
      ++tx;
    }
    data_.TxEnd(tx);
    return local_max;
  }

  /**
   * Select the first center to initialize kmeans++.
   * */
  T SelectFirstCenter() {
    hshm::UniformDistribution dist;
    dist.Seed(SEED);
    dist.Shape(0, data_.size_ - 1);
    size_t first_k = dist.GetSize();
    auto tx = data_.SeqTxBegin(first_k, 1, MM_READ_ONLY);
    T& pt = data_[first_k];
    data_.TxEnd(tx);
    return pt;
  }
};

struct DataStat {
  double sum_;
  double min_;
  double max_;

  DataStat() {
    sum_ = 0;
    min_ = std::numeric_limits<double>::max();
    max_ = 0;
  }
};

template<typename T>
class KmeansLlMpi : public KMeans<T> {
 public:
  using CenterT = MM_VEC<T>;
  using DataT = MM_VEC<T>;
  using MaxT = MM_VEC<LocalMax>;
  using AssignT = MM_VEC<size_t>;
  using SumT = MM_VEC<RowSum<T>>;
  using KMeans<T>::dir_;
  using KMeans<T>::rank_;
  using KMeans<T>::ks_;
  using KMeans<T>::k_;
  using KMeans<T>::data_;
  using KMeans<T>::nprocs_;
  using KMeans<T>::window_size_;
  using KMeans<T>::max_iter_;
  using KMeans<T>::tol_;
  using KMeans<T>::min_inertia_;
  using KMeans<T>::Print;
  using KMeans<T>::Fit;
  using KMeans<T>::world_;

 public:
  void Run() {
    HILOG(kInfo, "Running KMeansLL")
    T first_k = SelectFirstCenter();
    ks_.emplace_back(first_k);
    DataStat stat = LocalStatPoints();
    size_t log_size = std::max<size_t>(log(data_.size()), 1);
    size_t log_sum = std::max<size_t>(log(stat.sum_), 1);
    size_t count = std::min<size_t>(log_size, log_sum);
    size_t l = std::max(k_ / nprocs_, 1);
    count = 5;
    DataT centers = SelectSubcluster(stat, count, l);
    AggregateCenters(centers);
    Fit();
  }

  /**
   * Aggregate the centers using kmeans++
   * */
  void AggregateCenters(DataT &centers) {
    centers.Hint(MM_WRITE_ONLY);
    if (rank_ == 0) {
      KMeansPpMpi<T> agg_kmeans;
      agg_kmeans.Init(MPI_COMM_SELF,
                      centers,
                      window_size_,
                      k_,
                      max_iter_,
                      tol_,
                      min_inertia_);
      agg_kmeans.Run();

      auto tx = centers.SeqTxBegin(0, centers.size(), MM_WRITE_ONLY);
      for (int i = 0; i < k_; ++i) {
        centers[i] = agg_kmeans.ks_[i].center_;
        ++tx;
      }
      centers.TxEnd(tx);
    }
    centers.Barrier(MM_READ_ONLY, world_);
    ks_.clear();
    {
      auto tx = centers.SeqTxBegin(0, centers.size(), MM_READ_ONLY);
      for (int i = 0; i < k_; ++i) {
        ks_.emplace_back(centers[i]);
        ++tx;
      }
      centers.TxEnd(tx);
    }
  }

#define NEAR_COUNT_PER_ITER 16

  /**
   * Locate a subset of data points that match the probability distribution
   * */
  DataT SelectSubcluster(DataStat &stat,
                         size_t count, size_t l) {
    HILOG(kInfo, "Selecting subclusters")
    // Initialize center vector
    DataT centers;
    centers.Init(dir_ + "/" + "centers",
                 nprocs_ * (l * count + 1), MM_WRITE_ONLY);
    centers.EvenPgas(rank_, nprocs_, nprocs_ * (l * count + 1), l * count + 1);
    centers.Allocate();

    // Educated random subsample
    {
      auto tx = data_.RandTxBegin(
          SEED, data_.local_off(), data_.local_size(),
          count * l * NEAR_COUNT_PER_ITER,
          MM_READ_ONLY);
      ks_.reserve(count * l);
      for (size_t i = 0; i < count; ++i) {
        HILOG(kInfo, "Selecting {} points for cluster {}, with k: {}",
              l, i, ks_.size());
        hshm::UniformDistribution rand_dist;
        rand_dist.Seed(SEED);
        rand_dist.Shape(0, sqrt(l * stat.max_ * stat.max_ / stat.sum_));
        for (size_t j = 0; j < l; ++j) {
          // Randomly choose a distance and page
          double dist = pow(rand_dist.GetDouble(), 2) * stat.sum_ / l;
          // Get the point with nearest distance to this distance
          T pt = FindNearestPoint(tx, dist);
          ks_.emplace_back(pt);
        }
        // Determine the range of distances
        DataStat new_stat = LocalStatPoints();
        stat = new_stat;
      }
      data_.TxEnd(tx);
    }

    // Broadcast centers
    {
      auto tx = centers.SeqTxBegin(centers.local_off(),
                                   centers.local_size(),
                                   MM_WRITE_ONLY);
      for (size_t i = 0; i < ks_.size(); ++i) {
        centers[i + centers.local_off()] = ks_[i].center_;
        ++tx;
      }
      centers.TxEnd(tx);
    }
    centers.Barrier(MM_READ_ONLY, world_);
    ks_.clear();
    ks_.reserve(centers.size());
    {
      auto tx = centers.SeqTxBegin(0, centers.size(), MM_READ_ONLY);
      for (size_t i = 0; i < centers.size(); ++i) {
        ks_.emplace_back(centers[i]);
        ++tx;
      }
      centers.TxEnd(tx);
    }
    return centers;
  }

  /**
   * Find point with nearest distance measurement
   * */
  T FindNearestPoint(mm::Tx<mm::RandomIterator, DataT, T> &tx, double dist) {
    T nearest_point;
    double nearest_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < NEAR_COUNT_PER_ITER; ++i) {
      T &cur_pt = *tx;
      double pt_dist = MinOfCenterDists(cur_pt, ks_);
      double cur_dist = abs(pt_dist - dist);
      if (cur_dist < nearest_dist) {
        nearest_dist = cur_dist;
        nearest_point = cur_pt;
      }
      ++tx;
    }
    return nearest_point;
  }

  /**
   * Collect statistics about the data points.
   * */
  DataStat LocalStatPoints() {
    DataStat stat;
    HILOG(kInfo, "Collecting local statistics of chunk")
    size_t off = data_.local_off();
    size_t subsample = data_.local_size() / 8;
    auto tx = data_.RandTxBegin(
        SEED, off, data_.local_size(), subsample,
        MM_READ_ONLY);
    for (size_t i = 0; i < subsample; ++i) {
      if (i % (data_.elmts_per_page_) == 0) {
        HILOG(kInfo, "{}: We are {}% done", rank_,
              i * 100.0 / subsample)
      }
      T &cur_pt = *tx;
      double dist = MinOfCenterDists(cur_pt, ks_);
      stat.sum_ += dist;
      if (dist < stat.min_) {
        stat.min_ = dist;
      }
      if (dist > stat.max_) {
        stat.max_ = dist;
      }
      ++tx;
    }
    data_.TxEnd(tx);
    HILOG(kInfo, "Finished collecting local statistics of chunk")
    return stat;
  }

  /**
   * Select the first center to initialize kmeans++.
   * */
  T SelectFirstCenter() {
    hshm::UniformDistribution dist;
    dist.Seed(SEED);
    dist.Shape(0, data_.size_ - 1);
    size_t first_k = dist.GetSize();
    auto tx = data_.SeqTxBegin(first_k, 1, MM_READ_ONLY);
    T first = data_[first_k];
    data_.TxEnd(tx);
    return first;
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

  HILOG(kInfo, "{}: Started Hermes transparent connection", rank);
  std::mutex mux;
  mux.lock();
  TRANSPARENT_HERMES();
  mux.unlock();
  HILOG(kInfo, "{}: Initialized Hermes transparent connection", rank);

  if (algo == "mmap") {
  } else if (algo == "mega") {
    KmeansLlMpi<Row> kmeans;
    // kmeans.Init(MPI_COMM_WORLD, path, window_size, k, max_iter);
    // kmeans.Run();
    // kmeans.Print();
  } else {
    HILOG(kFatal, "Unknown algorithm: {}", algo);
  }
  MPI_Finalize();
}
