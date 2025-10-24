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
 * @file mm_scalar.cc
 * @brief Benchmark for scalar operations using MegaMmap.
 *
 * This file contains benchmark code for scalar operations utilizing MegaMmap's vector classes. It uses MPI for distributed computing and leverages MegaMmap's vector classes for efficient data handling.
 *
 * This project is part of the IoWarp project, a collaborative NSF-funded effort. It is also part of the production code in the github.com/iowarp codebase.
 *
 * @author Anthony Kougkas <akougkas@iit.edu>
 * @date 2025-10-24
 * @version 1.0
 */

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
