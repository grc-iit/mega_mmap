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
  size_t head_;
  size_t tail_;
  Vector *vec_;

 public:
  explicit Tx(Vector *vec) {
    vec_ = vec;
  }
  virtual ~Tx() = default;

  virtual void ProcessLog() = 0;
};

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_TRANSACTION_H_
