/** Create a dataset of 2D particles. Particles are in 6 clusters. */

#include <string>
#include <mpi.h>
#include <vector>
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/config_parse.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  if (argc < 4) {
    HILOG(kFatal, "Usage: ./kmeans_df <data_path> <df_size> <window_size>");
  }
  int nprocs = 0, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string data_path = argv[1];
  size_t df_size = hshm::ConfigParse::ParseSize(argv[2]);
  size_t window_size = hshm::ConfigParse::ParseSize(argv[3]);
  size_t window_count = window_size / sizeof(float);
  size_t rep = df_size / window_size / nprocs;

  HILOG(kInfo, "Creating dataset {} of size {} with {} windows of size {} each", data_path, argv[2], rep, window_size);

  MPI_File file_handle;
  int ierr = MPI_File_open(MPI_COMM_WORLD, data_path.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file_handle);

  std::vector<std::pair<float, float>> ks = {
      {0, 0},
      {10, 10},
      {20, 20},
      {30, 30},
      {40, 40},
  };

  // Set randomness seed
  srand(rank * 250);

  // Each process creates a dataset which contains k clusters
  std::vector<float> window(window_count);
  for (size_t i = 0; i < rep; ++i) {
    HILOG(kInfo, "Creating window {}/{}", i, rep)
    for (size_t j = 0; j < window_count; j += 2) {
      int k_idx = rand() % ks.size();
      float kx = ks[k_idx].first;
      float ky = ks[k_idx].second;
      float x = 2.0 / (1 + (rand() % 8));
      float y = 2.0 / (1 + (rand() % 8));
      window[j + 0] = kx + x;
      window[j + 1] = ky + y;
    }
    MPI_File_write_shared(file_handle, window.data(), window_count, MPI_FLOAT, MPI_STATUS_IGNORE);
  }

  MPI_File_close(&file_handle);
  MPI_Finalize();
}
