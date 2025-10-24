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
 * @file hermes_test.cc
 * @brief Hermes integration tests and utilities
 *
 * Provides baseline functionality for testing Hermes buffering and
 * tiering capabilities with MegaMmap. Validates transparent Hermes
 * integration for memory-mapped I/O operations.
 *
 * @author Luke Martin Logan <llogan@iit.edu>
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