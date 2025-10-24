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
 * @file macros.h
 * @brief Core macros and data structures for MegaMmap memory-mapped vectors.
 *
 * Defines access mode flags and PGAS (Partitioned Global Address Space) metadata
 * for managing distributed memory-mapped vectors across multiple storage tiers.
 * Provides foundational types and bit flags used throughout the MegaMmap system.
 *
 * @author Luke Martin Logan <llogan@iit.edu>
 * @date 2025-10-24
 * @version 1.0
 */

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_MACROS_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_MACROS_H_

#include <hermes_shm/data_structures/data_structure.h>
#include <hrun/hrun_types.h>

#define MM_READ_ONLY BIT_OPT(u32, 0)
#define MM_WRITE_ONLY BIT_OPT(u32, 1)
#define MM_APPEND_ONLY BIT_OPT(u32, 2)
#define MM_READ_WRITE BIT_OPT(u32, 3)
#define MM_STAGE BIT_OPT(u32, 4)

namespace mm {

using hshm::bitfield32_t;

class Bounds {
 public:
  size_t off_, size_;
  int rank_, nprocs_;
 public:
  Bounds() = default;

  Bounds(const Bounds &other) {
    off_ = other.off_;
    size_ = other.size_;
    rank_ = other.rank_;
    nprocs_ = other.nprocs_;
  }

  Bounds &operator=(const Bounds &other) {
    off_ = other.off_;
    size_ = other.size_;
    rank_ = other.rank_;
    nprocs_ = other.nprocs_;
    return *this;
  }

  explicit Bounds(int rank, int nprocs,
                  size_t max_size) {
    EvenSplit(rank, nprocs, max_size);
  }

  void EvenSplit(int rank, int nprocs,
                 size_t max_size) {
    size_ = max_size / nprocs;
    if (rank == nprocs - 1) {
      size_ += max_size % nprocs;
    }
    off_ = rank * (max_size / nprocs);
    rank_ = rank;
    nprocs_ = nprocs;
  }
};

struct PGAS {
  size_t off_;
  size_t size_;
  size_t min_page_idx_;
  size_t min_page_off_;
  size_t max_page_idx_;
  size_t max_page_off_;
  size_t page_size_;

  void Init(size_t off, size_t size, size_t page_size) {
    off_ = off;
    size_ = size;
    min_page_idx_ = off / page_size;
    min_page_off_ = off % page_size;
    max_page_idx_ = (off + size) / page_size;
    max_page_off_ = (off + size) % page_size;
    page_size_ = page_size;
  }

  void GetPageBounds(size_t page_idx, size_t &page_off, size_t &page_size) {
    if (page_idx == min_page_idx_) {
      page_off = min_page_off_;
      page_size = page_size_ - min_page_off_;
    } else if (page_idx == max_page_idx_) {
      page_off = 0;
      page_size = max_page_off_;
    } else {
      page_off = 0;
      page_size = page_size_;
    }
  }
};

#define MM_PAGE_SIZE KILOBYTES(256)

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_MACROS_H_
