//
// Created by llogan on 3/10/24.
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

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  TRANSPARENT_HERMES();
  MPI_Finalize();
}