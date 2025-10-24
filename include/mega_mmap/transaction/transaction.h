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
 * @file transaction.h
 * @brief Abstract transaction base class for MegaMmap data access patterns.
 *
 * Defines the interface for prefetching and memory tiering coordination.
 * Transactions enable automatic data movement and optimization based on
 * access patterns across different memory tiers in the MegaMmap system.
 *
 * @author Luke Martin Logan <llogan@iit.edu>
 * @date 2025-10-24
 * @version 1.0
 */

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_TRANSACTION_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_TRANSACTION_H_

#include "hermes_shm/data_structures/data_structure.h"
#include "mega_mmap/macros.h"
#include "mega_mmap/vector.h"

namespace mm {

class Tx {
 public:
  size_t head_;  /**< Last access touched by ProcessLog */
  size_t tail_;  /**< Number of index operations */
  Vector *vec_;  /**< The vector where data is stored */

 public:
  explicit Tx(Vector *vec) {
    vec_ = vec;
    head_ = 0;
    tail_ = 0;
  }
  virtual ~Tx() = default;

  virtual void _ProcessLog(bool end) = 0;

  void ProcessLog(bool end) {
    _ProcessLog(end);
    head_ = tail_;
  }

  size_t NumPrefetchPages(size_t iter_size) {
    size_t iter_pages_left = (iter_size - tail_) / vec_->elmts_per_page_;
    size_t page_cap = (vec_->window_size_ - vec_->cur_memory_) /
        vec_->page_size_;
    if (iter_pages_left > page_cap) {
      return page_cap;
    }
    return iter_pages_left;
  }
};

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_TRANSACTION_H_
