//
// Created by llogan on 3/11/24.
//

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_

namespace mm {

class RandIterTx : public Tx {
 public:
  RandIterTx(Vector *vec, size_t seed, size_t rand_left, size_t rand_size,
             size_t size, uint32_t flags) : Tx(vec){
  }

  size_t Get() {
    return 0;
  }
};

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_
