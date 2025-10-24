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
 * @file rand_iter_tx.h
 * @brief Random iteration transaction for MegaMmap.
 *
 * Optimizes for random access patterns with locality-aware prefetching.
 * This transaction type uses probabilistic prediction to anticipate future
 * accesses and efficiently manage memory for irregular access patterns.
 *
 * @author Luke Martin Logan <llogan@iit.edu>
 * @date 2025-10-24
 * @version 1.0
 */

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_

#include "hermes_shm/util/random.h"

namespace mm {

class RandIterTx : public Tx {
 public:
  hshm::UniformDistribution gen_;
  hshm::UniformDistribution log_gen_;
  hshm::UniformDistribution prefetch_gen_;
  size_t size_;
  size_t base_ = 0;
  bitfield32_t flags_;
  size_t rand_left_;
  size_t rand_size_;
  size_t num_elmts_;
  size_t num_pages_;

 public:
  RandIterTx(Vector *vec, size_t seed, size_t rand_left, size_t rand_size,
             size_t size, uint32_t flags) : Tx(vec) {
    gen_.Seed(seed);
    gen_.Shape((double)rand_left,
               (double)rand_left + (double)rand_size);
    size_ = size;
    rand_left_ = rand_left;
    rand_size_ = rand_size;
    flags_.SetBits(flags);
    log_gen_ = gen_;
    prefetch_gen_ = gen_;
    num_elmts_ = vec_->elmts_per_page_;
    num_pages_ = 0;
  }

  virtual ~RandIterTx() = default;

  void _ProcessLog(bool end) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    HILOG(kInfo, "{}: Processing log: {} {}",
          rank, head_, tail_);
    // Get number of pages iterated over
    size_t num_pages = num_pages_;
    if (num_pages <= 1 && !end) {
      return;
    }
    if (!end) {
      num_pages -= 1;
    }

    // Evict processed pages
    HILOG(kInfo, "{}: Evicting {} pages",
          rank, num_pages)
    for (size_t i = 0; i < num_pages; ++i) {
      size_t page_idx = log_gen_.GetSize() / vec_->elmts_per_page_;
      vec_->Rescore(page_idx,
                    0,
                    vec_->elmts_per_page_,
                    0, flags_);
      num_pages_ -= 1;
    }

    // Prefetch future pages
    if (vec_->cur_memory_ >= vec_->window_size_ || end) {
      return;
    }
    prefetch_gen_ = log_gen_;
    size_t count = NumPrefetchPages(size_);
    HILOG(kInfo, "{}: Prefetching {} pages",
          rank, count)
    for (size_t i = 0; i < count; ++i) {
      size_t page_idx = prefetch_gen_.GetSize() / vec_->elmts_per_page_;
      vec_->Rescore(page_idx,
                    0,
                    vec_->elmts_per_page_,
                    1.0, flags_);
    }
  }

  size_t Get() {
    size_t off = tail_ % num_elmts_;
    if (off == 0) {
      size_t page_idx = (gen_.GetSize() / vec_->elmts_per_page_);
      num_elmts_ = vec_->elmts_per_page_;
      base_ = page_idx * vec_->elmts_per_page_;
      num_pages_ += 1;
    }
    return base_ + off;
  }
};

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_
