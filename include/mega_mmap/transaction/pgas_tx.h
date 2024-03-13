//
// Created by llogan on 3/11/24.
//

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_PGAS_TX_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_PGAS_TX_H_

#include "transaction.h"

namespace mm {

class PgasTx : public Tx {
 public:
  size_t off_;
  size_t size_;
  hshm::bitfield32_t flags_;

 public:
  /**
   * Read or modify vector partition at unpredictable points
   *
   * @param off Offset (in elements) into the vector
   * @param size Number of elements to iterate over
   * @param flags Access flags for this transaction
   * */
  PgasTx(Vector *vec, size_t off, size_t size, uint32_t flags) : Tx(vec) {
    off_ = off;
    size_ = size;
    flags_.SetBits(flags);
  }

  virtual ~PgasTx() = default;

  /** Process the accesses that have occurred */
  void _ProcessLog(bool end) override {
    if (end) {
      size_t first_page = off_ / vec_->elmts_per_page_;
      size_t last_page = (off_ + size_) / vec_->elmts_per_page_;
      for (size_t i = first_page; i <= last_page; ++i) {
        vec_->Rescore(i, 0, vec_->elmts_per_page_,
                      0, flags_);
      }
    }
  }

  /** Get the current point in the transaction */
  size_t Get() {
    return tail_;
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_PGAS_TX_H_
