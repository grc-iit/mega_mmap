/** Create a dataset of 2D particles. Particles are in 6 clusters. */

#include <string>
#include <mpi.h>
#include <vector>
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/config_parse.h"
#include "hermes_shm/util/random.h"
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
#include "test_types.h"

class RandomForestDf {
 public:
  int rank_;
  std::string data_path_;
  size_t window_count_;
  size_t rep_;
  int k_;
  std::vector<ClassRow> centers_;
  hshm::UniformDistribution dist_;
  hshm::UniformDistribution dist2_;

 public:
  RandomForestDf(int rank, const std::string &data_path,
           int k, size_t window_count, size_t rep) :
      rank_(rank), data_path_(data_path), window_count_(window_count), rep_(rep) {
    centers_.reserve(k);
    for (int i = 0; i < k; ++i) {
      ClassRow ClassRow;
      ClassRow.x_ = (float) i * 10;
      ClassRow.y_ = (float) i * 10;
      ClassRow.class_ = i;
      centers_.emplace_back(ClassRow);
    }
    dist_.Seed(2354235);
    dist2_.Seed(2354235);
    dist_.Shape(0, (int)centers_.size() - 1);
    dist2_.Shape(0, 5);
  }

  void to_shared_file() {
    MPI_File file_handle;
    MPI_File_open(MPI_COMM_WORLD,
                  data_path_.c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                  &file_handle);

    // Each process creates a dataset which contains k clusters
    std::vector<ClassRow> window(window_count_);
    size_t off = rank_ * rep_ * window.size();
    for (size_t i = 0; i < rep_; ++i) {
      HILOG(kInfo, "Creating window {}/{} at offset {}",
            i, rep_, off);
      for (size_t j = 0; j < window.size(); ++j) {
        int k_idx = dist_.GetInt();
        float kx = centers_[k_idx].x_;
        float ky = centers_[k_idx].y_;
        window[j].x_ = (float)(kx + dist2_.GetDouble());
        window[j].y_ = (float)(ky + dist2_.GetDouble());
        window[j].class_ = k_idx;
      }
      MPI_File_write_at(file_handle,
                        off * sizeof(ClassRow),
                        window.data(),
                        window.size() * sizeof(ClassRow),
                        MPI_CHAR, MPI_STATUS_IGNORE);
      off += window.size();
    }

    MPI_File_close(&file_handle);
  }

  void to_parquet() {
    // Each process creates a dataset which contains k clusters
    arrow::FloatBuilder window_x;
    arrow::FloatBuilder window_y;
    arrow::FloatBuilder class_xy;
    arrow::Status status;
    for (size_t i = 0; i < rep_; ++i) {
      HILOG(kInfo, "(rank {}) Creating window {}/{}", rank_, i + 1, rep_);
      for (size_t j = 0; j < window_count_; ++j) {
        int k_idx = dist_.GetInt();
        float kx = centers_[k_idx].x_;
        float ky = centers_[k_idx].y_;
        status = window_x.Append((float)(kx + dist2_.GetDouble()));
        if (!status.ok()) {
          HILOG(kFatal, "Failed to append to window_x: {}",
                status.ToString());
        }
        status = window_y.Append((float)(ky + dist2_.GetDouble()));
        if (!status.ok()) {
          HILOG(kFatal, "Failed to append to window_y: {}",
                status.ToString());
        }
        status = class_xy.Append(k_idx);
        if (!status.ok()) {
          HILOG(kFatal, "Failed to append to class_xy: {}",
                status.ToString());
        }
      }

      // Finish building the parquet table
      std::shared_ptr<arrow::Array> x_array;
      std::shared_ptr<arrow::Array> y_array;
      std::shared_ptr<arrow::Array> class_array;
      status = window_x.Finish(&x_array);
      if (!status.ok()) {
        HILOG(kFatal, "Failed to finish window_x: {}",
              status.ToString());
      }
      status = window_y.Finish(&y_array);
      if (!status.ok()) {
        HILOG(kFatal, "Failed to finish window_y: {}",
              status.ToString());
      }
      status = class_xy.Finish(&class_array);
      if (!status.ok()) {
        HILOG(kFatal, "Failed to finish class_xy: {}",
              status.ToString());
      }
      std::shared_ptr<arrow::Schema> schema = arrow::schema({arrow::field("x", arrow::float32()),
                                                             arrow::field("y", arrow::float32()),
                                                             arrow::field("class", arrow::float32())});
      std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {x_array, y_array, class_array});

      // Create a schema with two columns: x and y
      std::string outfile_name = hshm::Formatter::format("{}_{}_{}_{}", data_path_, rank_, i, rep_);
      HILOG(kInfo, "(rank {}) Persisting window {}/{} to {}", rank_, i + 1, rep_, outfile_name);
      std::shared_ptr<arrow::io::FileOutputStream> outfile =
          arrow::io::FileOutputStream::Open(outfile_name).ValueOrDie();
      std::shared_ptr<parquet::arrow::FileWriter> writer =
          parquet::arrow::FileWriter::Open(
              *schema, arrow::default_memory_pool(), outfile).ValueOrDie();
      status = writer->WriteTable(*table);
      if (!status.ok()) {
        HILOG(kFatal, "Failed to write table: {}",
              status.ToString());
      }
      status = writer->Close();
      if (!status.ok()) {
        HILOG(kFatal, "Failed to close writer: {}",
              status.ToString());
      }
      status = outfile->Close();
      if (!status.ok()) {
        HILOG(kFatal, "Failed to close outfile: {}",
              status.ToString());
      }
    }
  }

  void to_hdf5() {
  }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  if (argc != 6) {
    HILOG(kFatal, "Usage: ./random_forest_df <k> <data_path> <df_size> <window_size> <type>");
  }
  int nprocs = 0, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int k = std::stoi(argv[1]);
  std::string data_path = argv[2];
  size_t df_size = hshm::ConfigParse::ParseSize(argv[3]);
  size_t window_size = hshm::ConfigParse::ParseSize(argv[4]);
  std::string type = argv[5];
  size_t window_count = window_size / (2 * sizeof(float));
  size_t rep = df_size / window_size / nprocs;
  if (rep == 0) {
    HILOG(kFatal, "Reduce number of processes");
  }
  HILOG(kInfo, "This process is rank {} with {} procs", rank, nprocs);
  HILOG(kInfo, "(rank {}) Will create {} bytes of data", rank, rep * window_size);
  RandomForestDf kmeans_df(rank, data_path, k, window_count, rep);

  HILOG(kInfo, "Creating dataset {} of size {} with {} windows of size {} each", data_path, argv[2], rep, window_size);

  // Each process creates a dataset which contains k clusters
  if (type == "parquet") {
    kmeans_df.to_parquet();
  } else if (type == "shared") {
    kmeans_df.to_shared_file();
  }

  MPI_Finalize();
}
