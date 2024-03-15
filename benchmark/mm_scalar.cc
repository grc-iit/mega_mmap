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

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  if (argc != 4) {
    HILOG(kFatal, "USAGE: ./mm_dbscan [algo] [L] [window_size]");
  }
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  std::string algo = argv[1];
  size_t L = hshm::ConfigParse::ParseSize(argv[2]);
  size_t Lpp =  L / nprocs / sizeof(double);
  size_t window_size = std::stoul(argv[3]);
  HILOG(kInfo, "L: {}, Lpp: {}, window_size: {}", L, Lpp, window_size);

  double sum = 0;
  if (algo == "mega") {
    TRANSPARENT_HERMES();
    mm::VectorMegaMpi<double> vec;
    vec.Init("vec", L / sizeof(double), MM_WRITE_ONLY);
    vec.BoundMemory(hshm::ConfigParse::ParseSize(argv[2]));
    vec.EvenPgas(rank, nprocs, vec.size());
    vec.Allocate();

    HILOG(kInfo, "Beginning sequence: {} {}", vec.local_off(), vec.local_last())
    vec.SeqTxBegin(vec.local_off(), vec.local_size(),
                   MM_WRITE_ONLY);
    for (size_t i = 0; i < Lpp; ++i) {
      vec[i + vec.local_off()] = i;
    }
    vec.TxEnd();
    HILOG(kInfo, "Finished sequence")
  } else {
    std::vector<double> vec;
    vec.resize(Lpp);
    for (size_t i = 0; i < Lpp; ++i) {
      vec[i] = i;
    }
  }

  MPI_Finalize();
}
