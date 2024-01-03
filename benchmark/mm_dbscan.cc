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
         typename AssignT,
         typename SumT,
         typename T>
class DBScanMpi {
 public:
  std::string dir_;
  DataT data_;
  int rank_;
  int nprocs_;
  size_t window_size_;
  float dist_;
  size_t min_count_;
  hshm::UniformDistribution pts_;

 public:
  void Init(const std::string &path,
            int rank, int nprocs,
            size_t window_size, float dist, size_t min_count){
    dir_ = stdfs::path(path).parent_path();
    data_.Init(path);
    rank_ = rank;
    nprocs_ = nprocs;
    window_size_ = window_size;
    pts_.Seed(23582);
    pts_.Shape(0, data_.size_ - 1);
    dist_ = dist;
    min_count_ = min_count;
  }

  void Run() {
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
