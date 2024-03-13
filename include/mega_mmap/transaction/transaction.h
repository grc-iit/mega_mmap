//
// Created by llogan on 3/11/24.
//

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
};

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_TRANSACTION_H_
