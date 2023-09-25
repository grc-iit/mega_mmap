//
// Created by lukemartinlogan on 9/13/23.
//

#include <string>
#include <mpi.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/config_parse.h"
#include <filesystem>
#include <algorithm>
#include <random>

#define SEED 24645643

namespace stdfs = std::filesystem;

typedef float Distance;

/** Row-major array of data */
template<int dim, typename T = float>
struct RowData {
  T *data_;
  size_t count_;

  HSHM_ALWAYS_INLINE
  T& Get(size_t off, size_t comp) {
    return data_[off * dim + comp];
  }
};

/** Single-node KMeans */
template<int dim, int cdim = dim + 1>
class KMeans {
 public:
  // Select initial centers for the chunk
  void SelectCenters0(RowData<dim> &chunk,
                      RowData<cdim> &centers) {
    std::uniform_int_distribution<size_t> idx_gen(0, chunk.count_);
    std::mt19937 gen(SEED);
    for (size_t i = 0; i < centers.count_; ++i) {
      size_t idx = idx_gen(gen);
      centers.Get(i, 0) = chunk.Get(idx, 0);
      centers.Get(i, 1) = chunk.Get(idx, 1);
    }
  }

  // Select new centers for the chunk
  void SelectCenters(RowData<dim> &chunk,         // The chunk being kmean'd
                     RowData<1> &dist,          // The distance each chunk is from its nearest center
                     RowData<1, int> &cluster,  // The chosen cluster for each data point
                     RowData<cdim> &centers) {
    // Compute the new centers
    std::uniform_real_distribution<float> idx_gen(0, chunk.count_);
    std::mt19937 gen(SEED);
    for (size_t i = 0; i < centers.count_; ++i) {
      size_t idx = idx_gen(gen);
      centers.Get(i, 0) = chunk.Get(idx, 0);
      centers.Get(i, 1) = chunk.Get(idx, 1);
    }
  }

  // Assign each point to the nearest center
  float Assign(RowData<dim> &chunk,         // The chunk being kmean'd
               RowData<1> &dist,          // The distance each chunk is from its nearest center
               RowData<1, int> &cluster,  // The chosen cluster for each data point
               RowData<cdim> &centers) {
    float variance = 0;
    for (size_t i = 0; i < chunk.count_; ++i) {
      float x = chunk.Get(i, 0);
      float y = chunk.Get(i, 1);
      dist.Get(i, 0) = FLT_MAX;
      cluster.Get(i, 0) = 0;
      for (int k = 0; k < (int)centers.count_; ++k) {
        float dx = x - centers.Get(k, 0);
        float dy = y - centers.Get(k, 1);
        float tmp_dist = dx * dx + dy * dy;
        if (tmp_dist < dist.Get(i, 0)) {
          dist.Get(i, 0) = tmp_dist;
          cluster.Get(i, 0) = k;
        }
      }
      centers.Get(cluster.Get(i, 0), 0) += 1;
      variance += dist.Get(i, 0);
    }
    return variance;
  }

  // Run KMeans to completion
  void Run(RowData<dim> &chunk,         // The chunk being kmean'd
           RowData<1> &dist,          // The distance each chunk is from its nearest center
           RowData<1, int> &cluster,  // The chosen cluster for each data point
           RowData<cdim> &centers,
           int max_iter) {
    float orig_dist = 0;
    SelectCenters0(chunk, centers);
    float last_variance = 0;
    for (int i = 0; i < max_iter; ++i) {
      float variance = Assign(chunk, dist, cluster, centers);
      if (variance == 0) {
        break;
      }
      if ((last_variance > 0) && abs(variance - last_variance) / last_variance < .05) {
        break;
      }
      SelectCenters(chunk, dist, cluster, centers);
    }
  }
};

bool kmeans_windows(float *data, int rank, size_t off, size_t shift,
                    size_t proc_size, size_t window_size, size_t data_size,
                    int k) {
  bool is_sorted = true;
  size_t cur_size = 0;
  data += off;
  std::vector<std::pair<float, float>> centers(k);
  while (cur_size < proc_size && off < data_size) {
    printf("Completion of sorting windows on process %d: %.2f%%\n", rank, 100.0 * cur_size / proc_size);
    kmeans(data, window_size, centers);
    off += shift;
    data += shift;
    cur_size += shift;
    if (off + shift > data_size) {
      shift = data_size - off;
    }
    if (off + window_size > data_size) {
      shift = window_size - off;
    }
  }
  return is_sorted;
}

void kmeans_proc(float *data, int rank, int nprocs, size_t window_size, size_t data_size, int k) {
  size_t shift = window_size / 2;
  size_t proc_size = data_size / nprocs;
  size_t proc_off = rank * proc_size;
  while (true) {
    kmeans_windows(data, rank, proc_off, shift, proc_size, window_size, data_size);
    break;
  }
}

void kmeans_mmap(const std::string &path, int rank, int nprocs, size_t window_size, int k) {
  // Map the dataset
  size_t data_size = stdfs::file_size(path);
  int fd = open(path.c_str(), O_RDWR | O_CREAT, 0666);
  float *data = (float *) mmap(NULL, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  kmeans_proc(data, rank, nprocs, window_size, data_size);
}

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
  int k = hshm::ConfigParse::ParseSize(argv[3]);
  size_t window_size = hshm::ConfigParse::ParseSize(argv[4]);
  int max_iter = std::stoi(argv[4]);
  HILOG(kInfo, "Running {} on {} with window size {} with {} centers", algo, path, window_size, k);

  if (algo == "mmap") {
    kmeans_mmap(path, rank, nprocs, window_size, k);
  } else if (algo == "mega") {
  } else {
    HILOG(kFatal, "Unknown algorithm: {}", algo);
  }
  MPI_Finalize();
}
