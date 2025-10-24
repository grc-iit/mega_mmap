/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2024-2025 Illinois Institute of Technology.
 * Gnosis Research Center.
 * All rights reserved.
 *
 * This file is part of MegaMmap.
 * Project website: https://github.com/grc-iit/megammap
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the conditions in the
 * LICENSE file are met. See the LICENSE file at the root of this
 * repository for details.
 *
 * Developed by: Gnosis Research Center
 *               Illinois Institute of Technology
 *               https://grc.iit.edu
 *
 * Contact: grc@iit.edu
 */

/**
 * @file mm_gadget2conv.cc
 * @brief Utility for converting Gadget2 simulation data formats.
 *
 * This file provides functionality to convert Gadget2 simulation data formats, utilizing HDF5 for data reading and processing.
 *
 * This project is part of the IoWarp project, a collaborative NSF-funded effort. It is also part of the production code in the github.com/iowarp codebase.
 *
 * @author Anthony Kougkas <akougkas@iit.edu>
 * @date 2025-10-24
 * @version 1.0
 */

#include <string>
#include <mpi.h>
#include <vector>
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/config_parse.h"
#include "hermes_shm/util/random.h"
#include <hdf5.h>

class Gadget2Conv {
 public:
  std::string in_path_;
  std::string out_path_;
  int rank_;
  int nprocs_;

 public:
  Gadget2Conv(int rank, int nprocs,
              const std::string &in_path, const std::string &out_path) {
    rank_ = rank;
    nprocs_ = nprocs;
    in_path_ = in_path;
    out_path_ = out_path;
  }

  std::vector<float> ReadHdf5(size_t &off_bytes) {
    // Open HDF5 file
    hid_t file_id = H5Fopen(in_path_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    // Open HDF5 group
    hid_t group_id = H5Gopen(file_id, "/PartType1/", H5P_DEFAULT);
    // Open existing HDF5 dataset
    hid_t dataset_id = H5Dopen(group_id, "Coordinates", H5P_DEFAULT);
    // Get dataspace
    hid_t dspace = H5Dget_space(dataset_id);
    // Get number of dimensions
    int ndims = H5Sget_simple_extent_ndims(dspace);
    // Get type of dataset
    hid_t dtype = H5Dget_type(dataset_id);
    // Get size of type
    size_t type_size = H5Tget_size(dtype);
    // Get the size of the dataset
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    // Define dataset subdimensions
    hsize_t start[ndims];
    hsize_t count[ndims];
    size_t count_agg = 1;
    // Find the largest dimension
    size_t sizepp = dims[0] / nprocs_;
    if (sizepp == 0) {
      HILOG(kFatal, "Number of processes must be less than or equal to "
                    "the largest dimension of the dataset")
    }
    start[0] = rank_ * sizepp;
    start[1] = 0;
    count[0] = sizepp;
    if (rank_ == nprocs_ - 1) {
      count[0] = dims[0] - sizepp * (nprocs_ - 1);
    }
    count[1] = 3;
    count_agg = count[0] * count[1];
    off_bytes = start[0] * count[1] * type_size;
    // Read subset of dataset
    hid_t memspace = H5Screate_simple(ndims, count, NULL);
    H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, NULL, count, NULL);
    std::vector<float> data(count_agg);
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace, dspace,
            H5P_DEFAULT, data.data());
    // Close HDF5 objects
    H5Sclose(memspace);
    H5Sclose(dspace);
    H5Dclose(dataset_id);
    H5Gclose(group_id);
    H5Fclose(file_id);
    return std::move(data);
  }

  void to_shared_file() {
    // Read HDF5 subset
    size_t off_bytes;
    std::vector<float> data = ReadHdf5(off_bytes);
    // Write to shared file
    MPI_File file_handle;
    MPI_File_open(MPI_COMM_WORLD,
                  out_path_.c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                  &file_handle);
    MPI_File_write_at(file_handle,
                      off_bytes,
                      data.data(),
                      data.size() * sizeof(float),
                      MPI_CHAR, MPI_STATUS_IGNORE);
  }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (argc != 3) {
    HILOG(kFatal, "Usage: ./mm_gadget2conv <in_path> <out_path>");
    return 1;
  }
  std::string in_path(argv[1]);
  std::string out_path(argv[2]);
  Gadget2Conv gadget2conv(rank, nprocs, in_path, out_path);
  gadget2conv.to_shared_file();
  MPI_Finalize();
  return 0;
}