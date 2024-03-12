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
  /**
   * Sequentially iterate over a vector
   *
   * @param off Offset (in elements) into the vector
   * @param size Number of elements to iterate over
   * @param flags Access flags for this transaction
   * */
  SeqIterTx(Vector *vec, size_t off, size_t size, uint32_t flags) : Tx(vec) {
    off_ = off;
    size_ = size;
    flags_.SetBits(flags);
  }

  virtual ~SeqIterTx() = default;

  /** Process the accesses that have occurred */
  void ProcessLog(bool end) override {
    size_t first_page = (head_ + off_) / vec_->elmts_per_page_;
    size_t last_page = (tail_ + off_) / vec_->elmts_per_page_;
    if (first_page == last_page) {
      return;
    }

    // Evict pages we no longer need
    size_t first_mod = (head_ + off_) % vec_->elmts_per_page_;
    size_t first_rem = vec_->elmts_per_page_ - first_mod;
    vec_->Rescore(first_page, first_mod, first_rem,
                  0, flags_);
    for (size_t i = first_page + 1; i < last_page; ++i) {
      vec_->Rescore(i, 0, vec_->elmts_per_page_,
                    0, flags_);
    }

    // Prefetch future pages
    for (size_t i = last_page + 1; i < last_page + 1; ++i) {
      vec_->Rescore(i, 0, vec_->elmts_per_page_,
                    1.0, flags_);
    }
  }

  /** Get the current point in the transaction */
  size_t Get() {
    return tail_;
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_SEQ_TX_H_
