//
// Created by llogan on 3/11/24.
//

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_SEQ_TX_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_SEQ_TX_H_

#include "transaction.h"

namespace mm {

class SeqIterTx : public Tx {
 public:
  size_t off_;
  size_t size_;
  hshm::bitfield32_t flags_;

 public:
  SeqIterTx(Vector *vec, size_t off, size_t size, uint32_t flags) : Tx(vec) {
    off_ = off;
    size_ = size;
    flags_.SetBits(flags);
  }

  virtual ~SeqIterTx() = default;

  void ProcessLog() override {
    // Identify pages we have completed
    size_t max_pages = (tail_ - head_) / vec_->page_size_;
    std::vector<size_t> page_idxs;

    // Evict pages we no longer need

    // Prefetch pages
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_SEQ_TX_H_
