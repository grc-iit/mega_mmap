/** Create a dataset of 2D particles. Particles are in 6 clusters. */

#include <string>
#include <mpi.h>
#include <vector>
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/config_parse.h"
#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/table.h>
#include <arrow/ipc/writer.h>
#include <parquet/arrow/reader.h>
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"
#include "parquet/arrow/writer.h"
#include "arrow/util/type_fwd.h"

class KmeansDf {
 public:
  int rank_;
  std::string data_path_;
  size_t window_count_;
  size_t rep_;
  std::vector<std::pair<float, float>> ks_;

 public:
  KmeansDf(int rank, const std::string &data_path, size_t window_count, size_t rep) :
  rank_(rank), data_path_(data_path), window_count_(window_count), rep_(rep) {
    ks_.reserve(30);
    for (int i = 0; i < 30; ++i) {
      ks_.emplace_back((float) i * 10, (float) i * 10);
    }
  }

  void to_shared_file() {
    MPI_File file_handle;
    MPI_File_open(MPI_COMM_WORLD,
                  data_path_.c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                  &file_handle);

    // Each process creates a dataset which contains k clusters
    std::vector<float> window(window_count_ * 2);
    for (size_t i = 0; i < rep_; ++i) {
      HILOG(kInfo, "Creating window {}/{}", i, rep_);
      for (size_t j = 0; j < window_count_; j += 2) {
        int k_idx = rand() % ks_.size();
        float kx = ks_[k_idx].first;
        float ky = ks_[k_idx].second;
        float x = 2.0 / (1 + (rand() % 8));
        float y = 2.0 / (1 + (rand() % 8));
        window[j + 0] = (kx + x);
        window[j + 1] = (ky + y);
      }
      MPI_File_write(file_handle, window.data(), window.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&file_handle);
  }

  void to_parquet() {
    // Each process creates a dataset which contains k clusters
    arrow::FloatBuilder window_x;
    arrow::FloatBuilder window_y;
    for (size_t i = 0; i < rep_; ++i) {
      HILOG(kInfo, "(rank {}) Creating window {}/{}", rank_, i + 1, rep_);
      for (size_t j = 0; j < window_count_; ++j) {
        int k_idx = rand() % ks_.size();
        float kx = ks_[k_idx].first;
        float ky = ks_[k_idx].second;
        float x = 2.0 / (1 + (rand() % 8));
        float y = 2.0 / (1 + (rand() % 8));
        window_x.Append(kx + x);
        window_y.Append(ky + y);
      }

      // Finish building the parquet table
      std::shared_ptr<arrow::Array> x_array;
      std::shared_ptr<arrow::Array> y_array;
      window_x.Finish(&x_array);
      window_y.Finish(&y_array);
      std::shared_ptr<arrow::Schema> schema = arrow::schema({arrow::field("x", arrow::float32()),
                                                             arrow::field("y", arrow::float32())});
      std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {x_array, y_array});

      // Create a schema with two columns: x and y
      std::string outfile_name = hshm::Formatter::format("{}_{}_{}_{}", data_path_, rank_, i, rep_);
      HILOG(kInfo, "(rank {}) Persisting window {}/{} to {}", rank_, i + 1, rep_, outfile_name);
      std::shared_ptr<arrow::io::FileOutputStream> outfile =
          arrow::io::FileOutputStream::Open(outfile_name).ValueOrDie();
      std::shared_ptr<parquet::arrow::FileWriter> writer =
          parquet::arrow::FileWriter::Open(
              *schema, arrow::default_memory_pool(), outfile).ValueOrDie();
      writer->WriteTable(*table);
      writer->Close();
      outfile->Close();
    }
  }

  void to_hdf5() {
    MPI_File file_handle;
    MPI_File_open(MPI_COMM_WORLD,
                  data_path_.c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                  &file_handle);

    // Each process creates a dataset which contains k clusters
    std::vector<float> window(window_count_ * 2);
    for (size_t i = 0; i < rep_; ++i) {
      HILOG(kInfo, "Creating window {}/{}", i, rep_);
      for (size_t j = 0; j < window_count_; j += 2) {
        int k_idx = rand() % ks_.size();
        float kx = ks_[k_idx].first;
        float ky = ks_[k_idx].second;
        float x = 2.0 / (1 + (rand() % 8));
        float y = 2.0 / (1 + (rand() % 8));
        window[j + 0] = (kx + x);
        window[j + 1] = (ky + y);
      }
      MPI_File_write(file_handle, window.data(), window.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&file_handle);
  }
};

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
  size_t window_count = window_size / (2 * sizeof(float));
  size_t rep = df_size / window_size / nprocs;
  if (rep == 0) {
    HILOG(kFatal, "Reduce number of processes");
  }
  HILOG(kInfo, "This process is rank {} with {} procs", rank, nprocs);
  HILOG(kInfo, "(rank {}) Will create {} bytes of data", rank, rep * window_size);
  KmeansDf kmeans_df(rank, data_path, window_count, rep);

  HILOG(kInfo, "Creating dataset {} of size {} with {} windows of size {} each", data_path, argv[2], rep, window_size);

  // Set randomness seed
  srand(rank * 250);

  // Each process creates a dataset which contains k clusters
  kmeans_df.to_parquet();

  MPI_Finalize();
}
